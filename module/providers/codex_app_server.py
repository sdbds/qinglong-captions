"""Codex Python SDK app-server adapter for subscription-backed captioning."""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import importlib
import inspect
import json
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from module.providers.codex_schema import (
    CODEX_CAPTION_SCHEMA,
    CODEX_CAPTION_SCHEMA_VERSION,
    CodexCaptionOutputError,
    normalize_codex_caption_payload,
    parse_codex_caption_output,
)


DEFAULT_CODEX_BACKEND = "sdk_app_server"
DEFAULT_CODEX_AUTH_MODE = "chatgpt"
SUPPORTED_CODEX_BACKENDS = {DEFAULT_CODEX_BACKEND, "exec"}
SUPPORTED_CODEX_AUTH_MODES = {DEFAULT_CODEX_AUTH_MODE, "api_key", "existing"}
DEFAULT_CODEX_REASONING_EFFORT = "none"
SUPPORTED_CODEX_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
CODEX_CAPTION_DISABLE_MCP_SERVER_NAMES = ("node_repl",)
CODEX_CAPTION_DISABLE_MCP_CONFIG_OVERRIDES = tuple(
    f"mcp_servers.{name}.enabled=false" for name in CODEX_CAPTION_DISABLE_MCP_SERVER_NAMES
)
CODEX_CAPTION_DISABLE_PLUGIN_CONFIG_OVERRIDES = ("features.plugins=false",)
_MCP_SERVER_TABLE_RE = re.compile(r"^\s*\[\s*mcp_servers\.([A-Za-z0-9_-]+)(?:\.|\s*\])")

INSTALL_CODEX_SDK_HINT = "Install it with: uv sync --extra codex-subscription"
API_KEY_ENV_VARS = ("OPENAI_API_KEY", "CODEX_API_KEY")
CODEX_APP_SERVER_LOG_LEVEL_ENV = "CODEX_LOG_LEVEL"
DEFAULT_CODEX_APP_SERVER_LOG_LEVEL = "ERROR"
RUST_LOG_ENV = "RUST_LOG"
RETRYABLE_CLIENT_FAILURE_KINDS = frozenset({"timeout", "transport", "rate_limited", "closed"})
RESET_CLIENT_FAILURE_KINDS = frozenset({"timeout", "transport", "closed"})
REQUIRED_CODEX_SDK_SYMBOLS = ("Codex", "TextInput", "LocalImageInput", "TurnResult")
CODEX_SDK_CONFIG_SYMBOLS = ("CodexConfig", "AppServerConfig")
DEFAULT_CODEX_TIMEOUT_SECONDS = 60.0
_WINDOWS_PROCESS_TERMINATED_RE = re.compile(
    r"^SUCCESS: The process with PID \d+(?: \(child process of PID \d+\))? has been terminated\.\s*$"
)
CODEX_CAPTION_TOOLS: dict[str, Any] = {}


@dataclass(frozen=True)
class CodexAppServerConfig:
    model: str = "gpt-5.4"
    service_tier: str = ""
    reasoning_effort: str = DEFAULT_CODEX_REASONING_EFFORT
    timeout: float = DEFAULT_CODEX_TIMEOUT_SECONDS
    sandbox: str = "read-only"
    auth_mode: str = DEFAULT_CODEX_AUTH_MODE
    api_key: str = ""
    codex_home: str = ""
    runtime_path: str = ""
    isolated_cwd: str = ""
    config_overrides: tuple[str, ...] = ()


@dataclass(frozen=True)
class CodexAppServerResult:
    raw: str
    parsed: dict[str, Any]
    thread_id: str
    turn_id: str
    metadata: dict[str, Any]


class CodexAppServerError(RuntimeError):
    """Raised when the Codex SDK app-server path cannot complete a request."""

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        retryable: bool = False,
        detail: str = "",
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.detail = detail
        self.cause = cause


ProgressCallback = Callable[[str], None]


def classify_codex_app_server_failure(text: str) -> str:
    lower = text.lower()
    if "no module named" in lower and "openai_codex" in lower:
        return "sdk_missing"
    if "invalid json-rpc" in lower or "json-rpc line" in lower or "json-rpc payload" in lower:
        return "transport"
    if "not logged in" in lower or "sign in" in lower or "login" in lower and "codex" in lower:
        return "auth"
    if "authentication" in lower or "unauthorized" in lower or "oauth" in lower:
        return "auth"
    if "usage limit" in lower or "subscription usage limit" in lower or "try again at" in lower:
        return "usage_limit"
    if "429" in lower or "too many requests" in lower or "rate limit" in lower or "rate_limited" in lower:
        return "rate_limited"
    if "timed out" in lower or "timeout" in lower:
        return "timeout"
    if "connection" in lower or "transport" in lower or "server closed" in lower or "app-server is not running" in lower:
        return "transport"
    return "execution"


def _is_retryable_client_failure(kind: str) -> bool:
    return kind in RETRYABLE_CLIENT_FAILURE_KINDS


def _should_reset_client_after_failure(kind: str) -> bool:
    return kind in RESET_CLIENT_FAILURE_KINDS


def _is_codex_app_server_stdout_noise(line: str) -> bool:
    return bool(_WINDOWS_PROCESS_TERMINATED_RE.match(line.strip()))


def _patch_codex_sdk_stdout_noise_filter(_sdk: Any) -> None:
    try:
        client_module = importlib.import_module("openai_codex.client")
    except Exception:
        return

    app_server_error = getattr(client_module, "AppServerError", None) or getattr(client_module, "CodexError", None)
    transport_closed_error = getattr(client_module, "TransportClosedError", None)
    if app_server_error is None or transport_closed_error is None:
        return

    def _read_message_with_noise_filter(self: Any) -> dict[str, Any]:
        if self._proc is None or self._proc.stdout is None:
            raise transport_closed_error("app-server is not running")

        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise transport_closed_error(
                    f"app-server closed stdout. stderr_tail={self._stderr_tail()[:2000]}"
                )
            if _is_codex_app_server_stdout_noise(line):
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                raise app_server_error(f"Invalid JSON-RPC line: {line!r}") from exc
            if not isinstance(message, dict):
                raise app_server_error(f"Invalid JSON-RPC payload: {message!r}")
            return message

    for class_name in ("AppServerClient", "CodexClient"):
        client_cls = getattr(client_module, class_name, None)
        if client_cls is None or getattr(client_cls, "_qinglong_stdout_noise_filter", False):
            continue
        client_cls._read_message = _read_message_with_noise_filter
        client_cls._qinglong_stdout_noise_filter = True


def _coerce_sdk_exception(
    exc: Exception,
    config: CodexAppServerConfig,
    *,
    stage: str,
) -> CodexAppServerError:
    text = str(exc)
    kind = classify_codex_app_server_failure(text)
    retryable = _is_retryable_client_failure(kind)
    label = "request" if stage == "request" else stage
    message = f"Codex app-server {label} failed ({kind}): {text}"
    if kind == "auth" and config.auth_mode == DEFAULT_CODEX_AUTH_MODE:
        message = (
            "Codex ChatGPT subscription session is not available. "
            "Run Codex login first; this provider does not fall back to OPENAI_API_KEY."
        )
    return CodexAppServerError(message, kind=kind, retryable=retryable, detail=text, cause=exc)


def load_openai_codex_sdk() -> Any:
    try:
        import openai_codex as sdk  # type: ignore
    except ImportError as exc:
        raise CodexAppServerError(
            f"Codex Python SDK is not installed. {INSTALL_CODEX_SDK_HINT}",
            kind="sdk_missing",
            cause=exc,
        ) from exc
    missing = [name for name in REQUIRED_CODEX_SDK_SYMBOLS if getattr(sdk, name, None) is None]
    if not any(getattr(sdk, name, None) is not None for name in CODEX_SDK_CONFIG_SYMBOLS):
        missing.append("CodexConfig or AppServerConfig")
    if getattr(sdk, "is_retryable_error", None) is None and getattr(sdk, "retry_on_overload", None) is None:
        missing.append("is_retryable_error or retry_on_overload")
    if missing:
        missing_text = ", ".join(missing)
        raise CodexAppServerError(
            f"Codex Python SDK is missing required public SDK symbols: {missing_text}. {INSTALL_CODEX_SDK_HINT}",
            kind="sdk_missing",
        )
    _patch_codex_sdk_stdout_noise_filter(sdk)
    return sdk


def build_codex_caption_tools() -> dict[str, Any]:
    return dict(CODEX_CAPTION_TOOLS)


def build_codex_caption_thread_config() -> dict[str, Any]:
    return {
        "tools": build_codex_caption_tools(),
    }


def _configured_mcp_server_names(codex_home: str = "") -> tuple[str, ...]:
    if not codex_home:
        return ()
    config_path = Path(codex_home).expanduser() / "config.toml"
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    names: list[str] = []
    seen: set[str] = set()
    for line in lines:
        match = _MCP_SERVER_TABLE_RE.match(line)
        if match is None:
            continue
        name = match.group(1)
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return tuple(names)


def build_codex_disable_mcp_config_overrides(codex_home: str = "") -> tuple[str, ...]:
    """Return app-server config overrides that keep caption runs off user MCP/plugins."""
    names = (*CODEX_CAPTION_DISABLE_MCP_SERVER_NAMES, *_configured_mcp_server_names(codex_home))
    seen: set[str] = set()
    overrides: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        overrides.append(f"mcp_servers.{name}.enabled=false")
    overrides.extend(CODEX_CAPTION_DISABLE_PLUGIN_CONFIG_OVERRIDES)
    return tuple(overrides)


def _sdk_config_class(sdk: Any) -> Any | None:
    for name in CODEX_SDK_CONFIG_SYMBOLS:
        cls = getattr(sdk, name, None)
        if cls is not None:
            return cls
    return None


def build_thread_start_payload(config: CodexAppServerConfig, cwd: str | Path) -> dict[str, Any]:
    return {
        "cwd": str(Path(cwd).expanduser().resolve()),
        "model": config.model,
        "ephemeral": True,
        "config": build_codex_caption_thread_config(),
    }


def normalize_codex_reasoning_effort(value: Any, default: str = DEFAULT_CODEX_REASONING_EFFORT) -> str:
    effort = str(value or "").strip().lower()
    if not effort:
        effort = default
    if effort not in SUPPORTED_CODEX_REASONING_EFFORTS:
        raise CodexAppServerError(
            f"Unsupported Codex reasoning effort: {value}. "
            f"Expected one of: {', '.join(sorted(SUPPORTED_CODEX_REASONING_EFFORTS))}",
            kind="config",
        )
    return effort


def _run_maybe_awaitable(value: Any) -> Any:
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    raise CodexAppServerError(
        "Codex SDK returned an async result inside an already running event loop.",
        kind="protocol",
    )


def _coerce_timeout_seconds(value: Any) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, timeout)


def _timeout_deadline(timeout: Any) -> float | None:
    timeout_seconds = _coerce_timeout_seconds(timeout)
    if timeout_seconds <= 0:
        return None
    return time.monotonic() + timeout_seconds


def _remaining_timeout_for_stage(deadline: float | None, stage: str) -> float:
    if deadline is None:
        return 0.0
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise CodexAppServerError(
            f"Codex app-server {stage} timed out after exhausting the request timeout.",
            kind="timeout",
            retryable=True,
        )
    return remaining


def _emit_progress(callback: ProgressCallback | None, message: str) -> None:
    if callback is None:
        return
    try:
        callback(message)
    except Exception:
        pass


def _run_with_timeout(
    fn: Callable[[], Any],
    *,
    timeout: float,
    stage: str,
    progress_callback: ProgressCallback | None = None,
    heartbeat_seconds: float = 15.0,
    on_timeout: Callable[[], None] | None = None,
    timeout_cleanup_grace: float = 0.5,
) -> Any:
    """Run a blocking SDK call with a caller-visible timeout boundary."""
    timeout_seconds = _coerce_timeout_seconds(timeout)
    if timeout_seconds <= 0:
        return fn()

    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def _worker() -> None:
        try:
            result_queue.put(("ok", fn()))
        except BaseException as exc:  # noqa: BLE001 - forwarded to caller below.
            result_queue.put(("error", exc))

    worker = threading.Thread(target=_worker, name=f"codex-app-server-{stage}", daemon=True)
    worker.start()

    started_at = time.monotonic()
    heartbeat = max(0.0, float(heartbeat_seconds or 0.0))
    next_heartbeat = started_at + heartbeat if heartbeat > 0 else float("inf")
    while True:
        elapsed = time.monotonic() - started_at
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            if on_timeout is not None:
                try:
                    on_timeout()
                except Exception:
                    pass
                cleanup_grace = _coerce_timeout_seconds(timeout_cleanup_grace)
                if cleanup_grace > 0:
                    worker.join(timeout=cleanup_grace)
            raise CodexAppServerError(
                f"Codex app-server {stage} timed out after {timeout_seconds:.0f}s.",
                kind="timeout",
                retryable=True,
            )
        wait_seconds = min(remaining, max(0.05, next_heartbeat - time.monotonic()))
        try:
            status, value = result_queue.get(timeout=wait_seconds)
            break
        except queue.Empty:
            now = time.monotonic()
            if heartbeat > 0 and now >= next_heartbeat:
                _emit_progress(
                    progress_callback,
                    f"Codex app-server: {stage} still waiting ({now - started_at:.0f}s/{timeout_seconds:.0f}s)",
                )
                next_heartbeat = now + heartbeat

    if status == "error":
        raise value
    return value


def _get_attr_path(target: Any, path: tuple[str, ...]) -> Any | None:
    value = target
    for name in path:
        value = getattr(value, name, None)
        if value is None:
            return None
    return value if callable(value) else None


def _close_sdk_client(client: Any) -> None:
    close_attempts: tuple[tuple[tuple[str, ...], tuple[Any, ...]], ...] = (
        (("close",), ()),
        (("aclose",), ()),
        (("shutdown",), ()),
        (("stop",), ()),
        (("terminate",), ()),
        (("__exit__",), (None, None, None)),
        (("app_server", "close"), ()),
    )
    for path, args in close_attempts:
        method = _get_attr_path(client, path)
        if method is None:
            continue
        try:
            _run_maybe_awaitable(method(*args))
            return
        except Exception:
            continue


def _call_payload_method(target: Any, paths: tuple[tuple[str, ...], ...], payload: dict[str, Any]) -> Any:
    method = next((candidate for path in paths if (candidate := _get_attr_path(target, path)) is not None), None)
    if method is None:
        names = ", ".join(".".join(path) for path in paths)
        raise CodexAppServerError(f"Codex SDK client does not expose any expected method: {names}", kind="protocol")

    try:
        return _run_maybe_awaitable(method(**payload))
    except TypeError as kwargs_error:
        try:
            return _run_maybe_awaitable(method(payload))
        except TypeError:
            raise CodexAppServerError(
                f"Codex SDK method signature is not compatible: {method!r}",
                kind="protocol",
                cause=kwargs_error,
            ) from kwargs_error


def _call_optional_method(target: Any, paths: tuple[tuple[str, ...], ...], payload: dict[str, Any] | None = None) -> Any:
    method = next((candidate for path in paths if (candidate := _get_attr_path(target, path)) is not None), None)
    if method is None:
        return None
    payload = payload or {}
    try:
        return _run_maybe_awaitable(method(**payload))
    except TypeError:
        return _run_maybe_awaitable(method(payload)) if payload else _run_maybe_awaitable(method())


def _first_mapping_value(data: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(data, dict):
        for key in keys:
            if key in data and data[key] not in (None, ""):
                return data[key]
    for key in keys:
        value = getattr(data, key, None)
        if value not in (None, ""):
            return value
    return None


def _extract_thread_id(response: Any) -> str:
    value = _first_mapping_value(response, ("threadId", "thread_id", "id"))
    if value is None:
        raise CodexAppServerError("Codex SDK thread/start did not return a thread id.", kind="protocol")
    return str(value)


def _extract_turn_id(response: Any) -> str:
    value = _first_mapping_value(response, ("turnId", "turn_id", "id"))
    return "" if value is None else str(value)


def _extract_text_from_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_extract_text_from_content(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        if "text" in value:
            return str(value["text"])
        if "content" in value:
            return _extract_text_from_content(value["content"])
        if "message" in value:
            return _extract_text_from_content(value["message"])
    text = getattr(value, "text", None)
    if text is not None:
        return str(text)
    content = getattr(value, "content", None)
    if content is not None:
        return _extract_text_from_content(content)
    return ""


def _item_value(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _enum_text(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _extract_assistant_final_response_from_items(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    final_phases = {"", "final", "final_answer", "completed", "complete"}
    for item in reversed(items):
        root = _item_value(item, "root") or _item_value(item, "item") or item
        role = _enum_text(_item_value(root, "role")).lower()
        item_type = _enum_text(_item_value(root, "type")).lower()
        phase = _enum_text(_item_value(root, "phase")).lower()
        if phase and phase not in final_phases:
            continue
        is_assistant = role == "assistant" or "agentmessage" in item_type or item_type in {"message", "assistantmessage"}
        if not is_assistant:
            continue
        extracted = _extract_text_from_content(root)
        if extracted:
            return extracted
    return ""


def _extract_turn_output(response: Any) -> tuple[str, dict[str, Any] | None]:
    parsed = _first_mapping_value(response, ("parsed", "outputParsed", "output_parsed"))
    if isinstance(parsed, dict):
        return "", normalize_codex_caption_payload(parsed)

    raw = _first_mapping_value(response, ("final_response",))
    extracted = _extract_text_from_content(raw)
    if extracted:
        return extracted, None

    items = _first_mapping_value(response, ("items",))
    extracted = _extract_assistant_final_response_from_items(items)
    if extracted:
        return extracted, None

    if isinstance(response, str):
        return response, None
    return "", None


def _try_construct(type_obj: Any, *args: Any, **kwargs: Any) -> Any | None:
    try:
        return type_obj(*args, **kwargs)
    except TypeError:
        return None


def build_codex_app_server_env(
    config: CodexAppServerConfig,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    source_env = os.environ if base_env is None else base_env
    env = dict(source_env)
    for key in API_KEY_ENV_VARS:
        env.pop(key, None)
    env.pop(RUST_LOG_ENV, None)
    env[CODEX_APP_SERVER_LOG_LEVEL_ENV] = DEFAULT_CODEX_APP_SERVER_LOG_LEVEL

    if config.auth_mode == "api_key":
        api_key = config.api_key or source_env.get("OPENAI_API_KEY", "")
        if api_key:
            env["OPENAI_API_KEY"] = api_key

    if config.codex_home:
        env["CODEX_HOME"] = str(Path(config.codex_home).expanduser())
    return env


def _create_app_server_config(sdk: Any, config: CodexAppServerConfig) -> Any | None:
    cls = _sdk_config_class(sdk)
    if cls is None:
        raise CodexAppServerError(
            f"Codex Python SDK is missing CodexConfig/AppServerConfig. {INSTALL_CODEX_SDK_HINT}",
            kind="sdk_missing",
        )
    sanitized_env = build_codex_app_server_env(config)
    kwargs: dict[str, Any] = {"env": sanitized_env}
    if config.config_overrides:
        kwargs["config_overrides"] = tuple(config.config_overrides)
    if config.runtime_path:
        kwargs["codex_bin"] = str(Path(config.runtime_path).expanduser())
    if config.isolated_cwd:
        kwargs["cwd"] = str(Path(config.isolated_cwd).expanduser())
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise CodexAppServerError(
            "Codex SDK config signature is not compatible with the published SDK contract.",
            kind="protocol",
            cause=exc,
        ) from exc


def _create_sdk_client(config: CodexAppServerConfig) -> Any:
    sdk = load_openai_codex_sdk()
    codex_cls = getattr(sdk, "Codex", None)
    if codex_cls is None:
        raise CodexAppServerError("Codex Python SDK does not export Codex.", kind="protocol")

    app_server_config = _create_app_server_config(sdk, config)
    try:
        return codex_cls(app_server_config)
    except TypeError as exc:
        raise CodexAppServerError(
            "Codex SDK Codex(config) signature is not compatible with the published SDK contract.",
            kind="protocol",
            cause=exc,
        ) from exc


def _make_client(client_factory: Callable[..., Any] | None, config: CodexAppServerConfig) -> Any:
    if client_factory is None:
        return _create_sdk_client(config)
    try:
        return client_factory(config)
    except TypeError:
        return client_factory()


def _api_key_fingerprint(api_key: str) -> str:
    if not api_key:
        return ""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


def _cache_key(config: CodexAppServerConfig) -> tuple[Any, ...]:
    return (
        config.model,
        config.service_tier,
        config.reasoning_effort,
        config.timeout,
        config.sandbox,
        config.auth_mode,
        _api_key_fingerprint(config.api_key),
        str(Path(config.codex_home).expanduser()) if config.codex_home else "",
        str(Path(config.runtime_path).expanduser()) if config.runtime_path else "",
        str(Path(config.isolated_cwd).expanduser()) if config.isolated_cwd else "",
        tuple(config.config_overrides),
    )


def _positive_int(value: Any, default: int = 1) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


_SANDBOX_ALIASES = {
    "read-only": "read_only",
    "read_only": "read_only",
    "readonly": "read_only",
    "workspace-write": "workspace_write",
    "workspace_write": "workspace_write",
    "workspacewrite": "workspace_write",
    "danger-full-access": "full_access",
    "danger_full_access": "full_access",
    "full-access": "full_access",
    "full_access": "full_access",
}
_SANDBOX_MODE_ATTRS = {
    "read_only": "read_only",
    "workspace_write": "workspace_write",
    "full_access": "danger_full_access",
}
_SANDBOX_PUBLIC_ATTRS = {
    "read_only": "read_only",
    "workspace_write": "workspace_write",
    "full_access": "full_access",
}
_SANDBOX_POLICY_TYPES = {
    "read_only": ("ReadOnlySandboxPolicy", "readOnly"),
    "workspace_write": ("WorkspaceWriteSandboxPolicy", "workspaceWrite"),
    "full_access": ("DangerFullAccessSandboxPolicy", "dangerFullAccess"),
}


def normalize_codex_sandbox(value: Any) -> str:
    sandbox = str(value or "").strip().lower()
    if not sandbox:
        sandbox = "read-only"
    sandbox = sandbox.replace(" ", "-")
    normalized = _SANDBOX_ALIASES.get(sandbox)
    if normalized is None:
        expected = ", ".join(sorted(_SANDBOX_ALIASES))
        raise CodexAppServerError(
            f"Unsupported Codex sandbox: {value}. Expected one of: {expected}",
            kind="config",
        )
    return normalized


def _get_optional_module(sdk: Any, attr_name: str, module_name: str) -> Any | None:
    module = getattr(sdk, attr_name, None)
    if module is not None:
        return module
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _get_generated_v2_module(sdk: Any) -> Any | None:
    generated = getattr(sdk, "generated", None)
    v2_all = getattr(generated, "v2_all", None) if generated is not None else None
    if v2_all is not None:
        return v2_all
    try:
        return importlib.import_module("openai_codex.generated.v2_all")
    except Exception:
        return None


def _resolve_thread_sandbox_value(sdk: Any, sandbox: Any) -> Any | None:
    semantic = normalize_codex_sandbox(sandbox)
    public_sandbox = getattr(sdk, "Sandbox", None)
    if public_sandbox is not None:
        value = getattr(public_sandbox, _SANDBOX_PUBLIC_ATTRS[semantic], None)
        if value is not None:
            return value

    types_module = _get_optional_module(sdk, "types", "openai_codex.types")
    sandbox_mode = getattr(types_module, "SandboxMode", None) if types_module is not None else None
    if sandbox_mode is None:
        v2_all = _get_generated_v2_module(sdk)
        sandbox_mode = getattr(v2_all, "SandboxMode", None) if v2_all is not None else None
    if sandbox_mode is not None:
        value = getattr(sandbox_mode, _SANDBOX_MODE_ATTRS[semantic], None)
        if value is not None:
            return value
    return None


def _resolve_run_sandbox_policy_value(sdk: Any, sandbox: Any) -> Any | None:
    semantic = normalize_codex_sandbox(sandbox)
    v2_all = _get_generated_v2_module(sdk)
    if v2_all is None:
        return None
    wrapper_cls = getattr(v2_all, "SandboxPolicy", None)
    policy_cls_name, policy_type = _SANDBOX_POLICY_TYPES[semantic]
    policy_cls = getattr(v2_all, policy_cls_name, None)
    if wrapper_cls is None or policy_cls is None:
        return None
    try:
        return wrapper_cls(policy_cls(type=policy_type))
    except Exception:
        return None


def _call_signature_parameters(method: Any) -> set[str]:
    try:
        return set(inspect.signature(method).parameters)
    except (TypeError, ValueError):
        return set()


def _filter_kwargs_for_method(method: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    clean = {key: value for key, value in kwargs.items() if value is not None}
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return clean
    parameters = signature.parameters
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return clean
    return {key: value for key, value in clean.items() if key in parameters}


def _resolve_run_sandbox_kwargs(sdk: Any, thread: Any, sandbox: Any) -> dict[str, Any]:
    method = getattr(thread, "run", None)
    parameters = _call_signature_parameters(method)
    if "sandbox" in parameters:
        value = _resolve_thread_sandbox_value(sdk, sandbox)
        return {"sandbox": value} if value is not None else {}
    if "sandbox_policy" in parameters:
        value = _resolve_run_sandbox_policy_value(sdk, sandbox)
        return {"sandbox_policy": value} if value is not None else {}
    return {}


def _build_sdk_run_input(prompt: str, image_path: str | Path) -> list[Any]:
    image = str(Path(image_path).expanduser().resolve())
    sdk = load_openai_codex_sdk()
    return [sdk.TextInput(text=prompt), sdk.LocalImageInput(path=image)]


def _sdk_deny_all_approval_mode() -> Any | None:
    try:
        import openai_codex as sdk  # type: ignore

        approval_mode = getattr(sdk, "ApprovalMode", None)
        return getattr(approval_mode, "deny_all", None)
    except Exception:
        return None


def _stream_item_label(payload: Any) -> str:
    item = getattr(payload, "item", None)
    root = getattr(item, "root", item)
    item_type = str(getattr(root, "type", "") or type(root).__name__)
    phase = getattr(root, "phase", None)
    phase_value = str(getattr(phase, "value", phase) or "")
    return f"{item_type}, {phase_value}" if phase_value else item_type


def _short_delta_preview(value: Any, *, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _summarize_stream_event(event: Any) -> str:
    method = str(getattr(event, "method", "") or "notification")
    payload = getattr(event, "payload", None)
    if method in {"item/agentMessage/delta", "item/plan/delta", "item/reasoning/summaryTextDelta"}:
        preview = _short_delta_preview(getattr(payload, "delta", ""))
        suffix = f": {preview}" if preview else ""
        return f"Codex app-server: event {method}{suffix}"
    if method == "item/reasoning/textDelta":
        return "Codex app-server: event item/reasoning/textDelta"
    if "token" in method:
        usage = getattr(payload, "token_usage", None)
        total = getattr(usage, "total", None)
        total_tokens = getattr(total, "total_tokens", None)
        output_tokens = getattr(total, "output_tokens", None)
        reasoning_tokens = getattr(total, "reasoning_output_tokens", None)
        parts = [f"total={total_tokens}"] if total_tokens is not None else []
        if output_tokens is not None:
            parts.append(f"output={output_tokens}")
        if reasoning_tokens is not None:
            parts.append(f"reasoning={reasoning_tokens}")
        suffix = f" ({', '.join(parts)})" if parts else ""
        return f"Codex app-server: event {method}{suffix}"
    if method in {"item/started", "item/completed"}:
        return f"Codex app-server: event {method} ({_stream_item_label(payload)})"
    if method == "turn/completed":
        turn = getattr(payload, "turn", None)
        status = getattr(turn, "status", None)
        status_value = str(getattr(status, "value", status) or "")
        suffix = f" (status={status_value})" if status_value else ""
        return f"Codex app-server: event {method}{suffix}"
    return f"Codex app-server: event {method}"


def _call_thread_run(
    thread: Any,
    *,
    config: CodexAppServerConfig,
    prompt: str,
    image_path: str | Path,
    output_schema: dict[str, Any] | None,
    progress_callback: ProgressCallback | None = None,
) -> Any:
    sdk = load_openai_codex_sdk()
    input_items = _build_sdk_run_input(prompt, image_path)
    deny_all = _sdk_deny_all_approval_mode()
    method = getattr(thread, "run", None)
    if not callable(method):
        raise CodexAppServerError("Codex SDK thread object does not expose run().", kind="protocol")
    kwargs = {
        "approval_mode": deny_all,
        "model": config.model,
        "effort": normalize_codex_reasoning_effort(config.reasoning_effort),
        "service_tier": (config.service_tier or "").strip() or None,
        "output_schema": output_schema,
    }
    kwargs.update(_resolve_run_sandbox_kwargs(sdk, thread, config.sandbox))
    kwargs = _filter_kwargs_for_method(method, kwargs)
    try:
        _emit_progress(progress_callback, "Codex app-server: running image turn")
        return _run_maybe_awaitable(method(input_items, **kwargs))
    except TypeError as exc:
        raise CodexAppServerError("Codex SDK Thread.run signature is not compatible.", kind="protocol", cause=exc) from exc


class CodexAppServerCaptionClient:
    """Small stable wrapper around the experimental Codex Python SDK."""

    def __init__(self, config: CodexAppServerConfig, *, client_factory: Callable[..., Any] | None = None):
        auth_mode = (config.auth_mode or DEFAULT_CODEX_AUTH_MODE).strip()
        if auth_mode not in SUPPORTED_CODEX_AUTH_MODES:
            raise CodexAppServerError(f"Unsupported Codex auth mode: {auth_mode}", kind="config")
        self.config = CodexAppServerConfig(**{**config.__dict__, "auth_mode": auth_mode})
        self.client = _make_client(client_factory, self.config)
        self._auth_checked = False
        self._auth_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise CodexAppServerError(
                "Codex app-server client is closed.",
                kind="closed",
                retryable=_is_retryable_client_failure("closed"),
            )

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            client = self.client
        _close_sdk_client(client)

    def ensure_auth(self) -> None:
        self._ensure_open()
        if self._auth_checked:
            return
        with self._auth_lock:
            self._ensure_open()
            if self._auth_checked:
                return
            auth_mode = self.config.auth_mode
            if auth_mode == "api_key":
                api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY", "")
                if not api_key:
                    raise CodexAppServerError(
                        "Codex API key auth was requested, but no API key was provided.",
                        kind="auth",
                    )
                _call_optional_method(
                    self.client,
                    (
                        ("login_api_key",),
                        ("loginApiKey",),
                        ("auth", "login_api_key"),
                        ("account", "login_api_key"),
                    ),
                    {"api_key": api_key},
                )
            elif auth_mode == "chatgpt":
                # The SDK/app-server reuses the current ChatGPT/Codex auth state.
                # Avoid launching an interactive login flow in batch captioning.
                pass
            elif auth_mode == "existing":
                pass
            self._auth_checked = True

    def start_thread(self) -> tuple[str, Any | None]:
        sdk = load_openai_codex_sdk()
        cwd = self.config.isolated_cwd or Path.cwd()
        payload = build_thread_start_payload(self.config, cwd)
        method = getattr(self.client, "thread_start", None)
        if not callable(method):
            raise CodexAppServerError("Codex SDK client does not expose thread_start().", kind="protocol")
        sandbox = _resolve_thread_sandbox_value(sdk, self.config.sandbox)
        if sandbox is not None:
            payload["sandbox"] = sandbox
        deny_all = _sdk_deny_all_approval_mode()
        if deny_all is not None:
            payload["approval_mode"] = deny_all
        try:
            response = _run_maybe_awaitable(method(**_filter_kwargs_for_method(method, payload)))
        except TypeError as exc:
            raise CodexAppServerError("Codex SDK thread_start signature is not compatible.", kind="protocol", cause=exc) from exc
        thread = response if callable(getattr(response, "run", None)) else None
        if thread is None:
            raise CodexAppServerError("Codex SDK thread_start did not return a runnable thread.", kind="protocol")
        thread_id = _extract_thread_id(response)
        return thread_id, thread

    def caption_image(
        self,
        *,
        image_path: str | Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        timeout: float | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> CodexAppServerResult:
        self._ensure_open()
        try:
            return _run_with_timeout(
                lambda: self._caption_image_blocking(
                    image_path=image_path,
                    prompt=prompt,
                    output_schema=output_schema,
                    progress_callback=progress_callback,
                ),
                timeout=self.config.timeout if timeout is None else timeout,
                stage="caption_image",
                progress_callback=progress_callback,
                on_timeout=self.close,
            )
        except CodexAppServerError as exc:
            if _should_reset_client_after_failure(exc.kind):
                self.close()
            raise

    def _caption_image_blocking(
        self,
        *,
        image_path: str | Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> CodexAppServerResult:
        with self._request_lock:
            self._ensure_open()
            image = Path(image_path).expanduser()
            _emit_progress(progress_callback, f"Codex app-server: checking image {image.name}")
            if not image.exists():
                raise CodexAppServerError(f"Image file does not exist: {image}", kind="image_not_found")

            try:
                _emit_progress(progress_callback, "Codex app-server: checking auth state")
                self.ensure_auth()
                _emit_progress(progress_callback, "Codex app-server: starting ephemeral thread")
                thread_id, thread = self.start_thread()
                _emit_progress(progress_callback, f"Codex app-server: thread ready ({thread_id})")
                _emit_progress(progress_callback, "Codex app-server: sending image turn")
                response = _call_thread_run(
                    thread,
                    config=self.config,
                    prompt=prompt,
                    image_path=image,
                    output_schema=output_schema,
                    progress_callback=progress_callback,
                )
                _emit_progress(progress_callback, "Codex app-server: turn completed")
            except CodexAppServerError:
                raise
            except Exception as exc:
                raise _coerce_sdk_exception(exc, self.config, stage="request") from exc

            _emit_progress(progress_callback, "Codex app-server: parsing structured output")
            raw, parsed = _extract_turn_output(response)
            if parsed is None:
                try:
                    parsed = parse_codex_caption_output(raw)
                except CodexCaptionOutputError as exc:
                    raise CodexAppServerError(str(exc), kind="output", detail=exc.raw or raw, cause=exc) from exc
            if not raw:
                import json

                raw = json.dumps(parsed, ensure_ascii=False)

            return CodexAppServerResult(
                raw=raw,
                parsed=parsed,
                thread_id=thread_id,
                turn_id=_extract_turn_id(response),
                metadata={
                    "backend": DEFAULT_CODEX_BACKEND,
                    "auth_mode": self.config.auth_mode,
                    "model": self.config.model,
                    "service_tier": self.config.service_tier,
                    "reasoning_effort": normalize_codex_reasoning_effort(self.config.reasoning_effort),
                    "schema_version": CODEX_CAPTION_SCHEMA_VERSION,
                },
            )


class CodexAppServerClientPool:
    """Bounded pool of app-server clients for concurrent Codex caption jobs."""

    def __init__(self, config: CodexAppServerConfig, size: int):
        self.config = config
        self.size = _positive_int(size, 1)
        self._queue: queue.Queue[CodexAppServerCaptionClient] = queue.Queue(maxsize=self.size)
        self._close_lock = threading.Lock()
        self._closed = False
        for _ in range(self.size):
            self._queue.put(CodexAppServerCaptionClient(config))

    def _ensure_open(self) -> None:
        if self._closed:
            raise CodexAppServerError(
                "Codex app-server client pool is closed.",
                kind="closed",
                retryable=_is_retryable_client_failure("closed"),
            )

    def _return_client(self, client: CodexAppServerCaptionClient) -> None:
        with self._close_lock:
            if self._closed:
                should_close = True
            else:
                self._queue.put(client)
                should_close = False
        if should_close:
            client.close()

    def _replace_closed_client(self) -> None:
        try:
            replacement = CodexAppServerCaptionClient(self.config)
        except Exception:
            return
        self._return_client(replacement)

    def _acquire_client(self, deadline: float | None) -> CodexAppServerCaptionClient:
        while True:
            self._ensure_open()
            wait_seconds = 0.1
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CodexAppServerError(
                        "Codex app-server client pool timed out waiting for an available client.",
                        kind="timeout",
                        retryable=True,
                    )
                wait_seconds = min(wait_seconds, remaining)
            try:
                return self._queue.get(timeout=wait_seconds)
            except queue.Empty:
                continue

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        while True:
            try:
                client = self._queue.get_nowait()
            except queue.Empty:
                break
            client.close()

    def caption_image(
        self,
        *,
        image_path: str | Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        timeout: float | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> CodexAppServerResult:
        deadline = _timeout_deadline(timeout)
        client = self._acquire_client(deadline)
        client_timeout = _remaining_timeout_for_stage(deadline, "caption_image") if deadline is not None else timeout
        try:
            result = client.caption_image(
                image_path=image_path,
                prompt=prompt,
                output_schema=output_schema,
                timeout=client_timeout,
                progress_callback=progress_callback,
            )
        except CodexAppServerError as exc:
            if _should_reset_client_after_failure(exc.kind):
                client.close()
                self._replace_closed_client()
            else:
                self._return_client(client)
            raise
        except Exception:
            self._return_client(client)
            raise
        self._return_client(client)
        return result


_CLIENT_CACHE: dict[tuple[Any, ...], CodexAppServerCaptionClient] = {}
_CLIENT_POOLS: dict[tuple[Any, ...], CodexAppServerClientPool] = {}
_CLIENT_CACHE_LOCK = threading.Lock()
_CLIENT_CACHE_GENERATION = 0


def get_codex_app_server_client(
    config: CodexAppServerConfig,
    *,
    client_factory: Callable[..., Any] | None = None,
) -> CodexAppServerCaptionClient:
    if client_factory is not None:
        return CodexAppServerCaptionClient(config, client_factory=client_factory)
    key = _cache_key(config)
    with _CLIENT_CACHE_LOCK:
        client = _CLIENT_CACHE.get(key)
        generation = _CLIENT_CACHE_GENERATION
    if client is not None:
        return client

    new_client = CodexAppServerCaptionClient(config)
    with _CLIENT_CACHE_LOCK:
        existing = _CLIENT_CACHE.get(key)
        if existing is not None:
            replacement = existing
        elif generation == _CLIENT_CACHE_GENERATION:
            _CLIENT_CACHE[key] = new_client
            return new_client
        else:
            replacement = None

    new_client.close()
    if replacement is not None:
        return replacement
    raise CodexAppServerError("Codex app-server client startup was abandoned.", kind="timeout", retryable=True)


def get_codex_app_server_client_pool(
    config: CodexAppServerConfig,
    *,
    max_concurrency: int,
) -> CodexAppServerClientPool:
    pool_size = _positive_int(max_concurrency, 1)
    key = (*_cache_key(config), pool_size)
    with _CLIENT_CACHE_LOCK:
        pool = _CLIENT_POOLS.get(key)
        generation = _CLIENT_CACHE_GENERATION
    if pool is not None:
        return pool

    new_pool = CodexAppServerClientPool(config, pool_size)
    with _CLIENT_CACHE_LOCK:
        existing = _CLIENT_POOLS.get(key)
        if existing is not None:
            replacement = existing
        elif generation == _CLIENT_CACHE_GENERATION:
            _CLIENT_POOLS[key] = new_pool
            return new_pool
        else:
            replacement = None

    new_pool.close()
    if replacement is not None:
        return replacement
    raise CodexAppServerError("Codex app-server client pool startup was abandoned.", kind="timeout", retryable=True)


def reset_codex_app_server_client_cache() -> None:
    global _CLIENT_CACHE_GENERATION
    with _CLIENT_CACHE_LOCK:
        _CLIENT_CACHE_GENERATION += 1
        clients = list(_CLIENT_CACHE.values())
        pools = list(_CLIENT_POOLS.values())
        _CLIENT_CACHE.clear()
        _CLIENT_POOLS.clear()
    for pool in pools:
        pool.close()
    for client in clients:
        client.close()


def caption_image_with_app_server(
    config: CodexAppServerConfig,
    *,
    image_path: str | Path,
    prompt: str,
    output_schema: dict[str, Any] | None = None,
    client_factory: Callable[..., Any] | None = None,
    max_concurrency: int = 1,
    progress_callback: ProgressCallback | None = None,
) -> CodexAppServerResult:
    deadline = _timeout_deadline(config.timeout)
    pool_size = _positive_int(max_concurrency, 1)
    client: CodexAppServerCaptionClient | None = None
    try:
        if client_factory is None and pool_size > 1:
            _emit_progress(progress_callback, f"Codex app-server: acquiring pooled client (workers={pool_size})")
            pool = _run_with_timeout(
                lambda: get_codex_app_server_client_pool(config, max_concurrency=pool_size),
                timeout=_remaining_timeout_for_stage(deadline, "client_pool_startup"),
                stage="client_pool_startup",
                progress_callback=progress_callback,
            )
            return pool.caption_image(
                image_path=image_path,
                prompt=prompt,
                output_schema=output_schema,
                timeout=_remaining_timeout_for_stage(deadline, "caption_image"),
                progress_callback=progress_callback,
            )

        _emit_progress(progress_callback, "Codex app-server: acquiring client")
        client = _run_with_timeout(
            lambda: get_codex_app_server_client(config, client_factory=client_factory),
            timeout=_remaining_timeout_for_stage(deadline, "client_startup"),
            stage="client_startup",
            progress_callback=progress_callback,
        )
        return client.caption_image(
            image_path=image_path,
            prompt=prompt,
            output_schema=output_schema,
            timeout=_remaining_timeout_for_stage(deadline, "caption_image"),
            progress_callback=progress_callback,
        )
    except CodexAppServerError as exc:
        if _should_reset_client_after_failure(exc.kind):
            if client_factory is None:
                reset_codex_app_server_client_cache()
            elif client is not None:
                client.close()
        raise
    except Exception as exc:
        coerced = _coerce_sdk_exception(exc, config, stage="client_startup")
        if _should_reset_client_after_failure(coerced.kind):
            if client_factory is None:
                reset_codex_app_server_client_cache()
            elif client is not None:
                client.close()
        raise coerced from exc
    finally:
        if client_factory is not None and client is not None:
            client.close()


atexit.register(reset_codex_app_server_client_cache)
