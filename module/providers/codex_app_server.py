"""Codex Python SDK app-server adapter for subscription-backed captioning."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
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

INSTALL_CODEX_SDK_HINT = "Install it with: uv sync --extra codex-subscription"
API_KEY_ENV_VARS = ("OPENAI_API_KEY", "CODEX_API_KEY")


@dataclass(frozen=True)
class CodexAppServerConfig:
    model: str = "gpt-5.4-mini"
    timeout: float = 180.0
    sandbox: str = "read-only"
    auth_mode: str = DEFAULT_CODEX_AUTH_MODE
    api_key: str = ""
    codex_home: str = ""
    runtime_path: str = ""
    isolated_cwd: str = ""


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


def classify_codex_app_server_failure(text: str) -> str:
    lower = text.lower()
    if "no module named" in lower and "openai_codex" in lower:
        return "sdk_missing"
    if "not logged in" in lower or "sign in" in lower or "login" in lower and "codex" in lower:
        return "auth"
    if "authentication" in lower or "unauthorized" in lower or "oauth" in lower:
        return "auth"
    if "usage limit" in lower or "subscription usage limit" in lower or "try again at" in lower:
        return "usage_limit"
    if "timed out" in lower or "timeout" in lower:
        return "timeout"
    if "connection" in lower or "transport" in lower or "server closed" in lower:
        return "transport"
    return "execution"


def load_openai_codex_sdk() -> Any:
    try:
        import openai_codex as sdk  # type: ignore
    except ImportError as exc:
        raise CodexAppServerError(
            f"Codex Python SDK is not installed. {INSTALL_CODEX_SDK_HINT}",
            kind="sdk_missing",
            cause=exc,
        ) from exc
    return sdk


def build_thread_start_payload(config: CodexAppServerConfig, cwd: str | Path) -> dict[str, Any]:
    return {
        "cwd": str(Path(cwd).expanduser().resolve()),
        "model": config.model,
        "sandbox": config.sandbox,
        "approvalPolicy": "never",
        "ephemeral": True,
    }


def build_turn_start_payload(
    *,
    thread_id: str,
    prompt: str,
    image_path: str | Path,
    model: str,
    output_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "threadId": thread_id,
        "model": model,
        "input": [
            {"type": "text", "text": prompt},
            {"type": "localImage", "path": str(Path(image_path).expanduser().resolve())},
        ],
        "outputSchema": output_schema or dict(CODEX_CAPTION_SCHEMA),
    }


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


def _get_attr_path(target: Any, path: tuple[str, ...]) -> Any | None:
    value = target
    for name in path:
        value = getattr(value, name, None)
        if value is None:
            return None
    return value if callable(value) else None


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


def _extract_turn_output(response: Any) -> tuple[str, dict[str, Any] | None]:
    parsed = _first_mapping_value(response, ("parsed", "outputParsed", "output_parsed"))
    if isinstance(parsed, dict):
        return "", normalize_codex_caption_payload(parsed)

    raw = _first_mapping_value(
        response,
        (
            "finalResponse",
            "final_response",
            "finalMessage",
            "final_message",
            "outputText",
            "output_text",
            "message",
            "text",
            "raw",
            "content",
        ),
    )
    extracted = _extract_text_from_content(raw)
    if extracted:
        return extracted, None

    output = _first_mapping_value(response, ("output", "messages", "items"))
    extracted = _extract_text_from_content(output)
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

    if config.auth_mode == "api_key":
        api_key = config.api_key or source_env.get("OPENAI_API_KEY", "")
        if api_key:
            env["OPENAI_API_KEY"] = api_key

    if config.codex_home:
        env["CODEX_HOME"] = str(Path(config.codex_home).expanduser())
    return env


def _create_app_server_config(sdk: Any, config: CodexAppServerConfig) -> Any | None:
    cls = getattr(sdk, "AppServerConfig", None)
    if cls is None:
        return None

    attempts: list[dict[str, Any]] = []
    sanitized_env = build_codex_app_server_env(config)
    base: dict[str, Any] = {"env": sanitized_env}
    if config.runtime_path:
        base["codex_bin"] = str(Path(config.runtime_path).expanduser())
    if config.isolated_cwd:
        base["cwd"] = str(Path(config.isolated_cwd).expanduser())
    attempts.append(base)
    if config.runtime_path:
        attempts.append({"codexBin": str(Path(config.runtime_path).expanduser()), "env": sanitized_env})
    attempts.append({"env": sanitized_env})
    if not any(os.environ.get(key) for key in API_KEY_ENV_VARS):
        attempts.append({})

    for kwargs in attempts:
        built = _try_construct(cls, **kwargs)
        if built is not None:
            return built
    return None


def _create_sdk_client(config: CodexAppServerConfig) -> Any:
    sdk = load_openai_codex_sdk()
    codex_cls = getattr(sdk, "Codex", None)
    if codex_cls is None:
        raise CodexAppServerError("Codex Python SDK does not export Codex.", kind="protocol")

    app_server_config = _create_app_server_config(sdk, config)
    if (
        app_server_config is None
        and config.auth_mode != "api_key"
        and any(os.environ.get(key) for key in API_KEY_ENV_VARS)
    ):
        raise CodexAppServerError(
            "Codex SDK cannot launch app-server with a sanitized environment; refusing to inherit API key env vars.",
            kind="auth",
        )
    options_cls = getattr(sdk, "CodexOptions", None)
    options = None
    if options_cls is not None:
        option_attempts: list[dict[str, Any]] = [
            {"model": config.model, "app_server": app_server_config},
            {"model": config.model, "appServer": app_server_config},
            {"app_server": app_server_config},
            {"appServer": app_server_config},
            {"model": config.model},
            {},
        ]
        for kwargs in option_attempts:
            kwargs = {key: value for key, value in kwargs.items() if value is not None}
            options = _try_construct(options_cls, **kwargs)
            if options is not None:
                break

    client_attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    if options is not None:
        client_attempts.extend([((options,), {}), ((), {"options": options})])
    if app_server_config is not None:
        client_attempts.extend(
            [
                ((app_server_config,), {}),
                ((), {"config": app_server_config}),
                ((), {"app_server": app_server_config}),
                ((), {"appServer": app_server_config}),
            ]
        )
    client_attempts.append(((), {}))

    for args, kwargs in client_attempts:
        built = _try_construct(codex_cls, *args, **kwargs)
        if built is not None:
            return built
    raise CodexAppServerError("Codex Python SDK client could not be constructed.", kind="protocol")


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
        config.timeout,
        config.sandbox,
        config.auth_mode,
        _api_key_fingerprint(config.api_key),
        str(Path(config.codex_home).expanduser()) if config.codex_home else "",
        str(Path(config.runtime_path).expanduser()) if config.runtime_path else "",
        str(Path(config.isolated_cwd).expanduser()) if config.isolated_cwd else "",
    )


def _build_sdk_run_input(prompt: str, image_path: str | Path) -> list[Any]:
    image = str(Path(image_path).expanduser().resolve())
    try:
        import openai_codex as sdk  # type: ignore

        text_cls = getattr(sdk, "TextInput", None)
        image_cls = getattr(sdk, "LocalImageInput", None)
        if text_cls is not None and image_cls is not None:
            return [text_cls(text=prompt), image_cls(path=image)]
    except Exception:
        pass
    return [
        {"type": "text", "text": prompt},
        {"type": "localImage", "path": image},
    ]


def _sdk_deny_all_approval_mode() -> Any | None:
    try:
        import openai_codex as sdk  # type: ignore

        approval_mode = getattr(sdk, "ApprovalMode", None)
        return getattr(approval_mode, "deny_all", None)
    except Exception:
        return None


def _call_thread_run(
    thread: Any,
    *,
    config: CodexAppServerConfig,
    prompt: str,
    image_path: str | Path,
    output_schema: dict[str, Any] | None,
) -> Any:
    method = getattr(thread, "run", None)
    if not callable(method):
        raise CodexAppServerError("Codex SDK thread object does not expose run().", kind="protocol")
    input_items = _build_sdk_run_input(prompt, image_path)
    deny_all = _sdk_deny_all_approval_mode()
    attempts = [
        {"input": input_items, "model": config.model, "output_schema": output_schema, "approval_mode": deny_all},
        {"input": input_items, "model": config.model, "outputSchema": output_schema},
        {"input": input_items, "output_schema": output_schema, "approval_mode": deny_all},
        {"input": input_items, "outputSchema": output_schema},
        {"input": input_items},
    ]
    for kwargs in attempts:
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        try:
            return _run_maybe_awaitable(method(**kwargs))
        except TypeError:
            continue
    try:
        return _run_maybe_awaitable(method(input_items))
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
        self.thread_id = ""
        self.thread = None
        self._auth_checked = False

    def ensure_auth(self) -> None:
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

    def start_thread(self) -> str:
        self.thread_id = ""
        self.thread = None
        cwd = self.config.isolated_cwd or Path.cwd()
        payload = build_thread_start_payload(self.config, cwd)
        method = _get_attr_path(self.client, ("thread_start",))
        if method is not None:
            deny_all = _sdk_deny_all_approval_mode()
            sdk_payload = {
                "cwd": payload["cwd"],
                "model": payload["model"],
                "sandbox": payload["sandbox"],
                "ephemeral": payload["ephemeral"],
            }
            if deny_all is not None:
                sdk_payload["approval_mode"] = deny_all
            try:
                response = _run_maybe_awaitable(method(**sdk_payload))
            except TypeError:
                response = _call_payload_method(
                    self.client,
                    (
                        ("thread_start",),
                        ("start_thread",),
                        ("threadStart",),
                        ("threads", "start"),
                        ("thread", "start"),
                        ("app_server", "thread_start"),
                    ),
                    payload,
                )
        else:
            response = _call_payload_method(
                self.client,
                (
                    ("start_thread",),
                    ("threadStart",),
                    ("threads", "start"),
                    ("thread", "start"),
                    ("app_server", "thread_start"),
                ),
                payload,
            )
        if callable(getattr(response, "run", None)):
            self.thread = response
        self.thread_id = _extract_thread_id(response)
        return self.thread_id

    def caption_image(
        self,
        *,
        image_path: str | Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
    ) -> CodexAppServerResult:
        image = Path(image_path).expanduser()
        if not image.exists():
            raise CodexAppServerError(f"Image file does not exist: {image}", kind="image_not_found")

        try:
            self.ensure_auth()
            thread_id = self.start_thread()
            if self.thread is not None:
                response = _call_thread_run(
                    self.thread,
                    config=self.config,
                    prompt=prompt,
                    image_path=image,
                    output_schema=output_schema,
                )
            else:
                payload = build_turn_start_payload(
                    thread_id=thread_id,
                    prompt=prompt,
                    image_path=image,
                    model=self.config.model,
                    output_schema=output_schema,
                )
                response = _call_payload_method(
                    self.client,
                    (
                        ("turn_start",),
                        ("start_turn",),
                        ("turnStart",),
                        ("turns", "start"),
                        ("turn", "start"),
                        ("app_server", "turn_start"),
                    ),
                    payload,
                )
        except CodexAppServerError:
            raise
        except Exception as exc:
            text = str(exc)
            kind = classify_codex_app_server_failure(text)
            retryable = kind in {"timeout", "transport"}
            message = f"Codex app-server request failed ({kind}): {text}"
            if kind == "auth" and self.config.auth_mode == DEFAULT_CODEX_AUTH_MODE:
                message = (
                    "Codex ChatGPT subscription session is not available. "
                    "Run Codex login first; this provider does not fall back to OPENAI_API_KEY."
                )
            raise CodexAppServerError(message, kind=kind, retryable=retryable, detail=text, cause=exc) from exc

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
            thread_id=self.thread_id,
            turn_id=_extract_turn_id(response),
            metadata={
                "backend": DEFAULT_CODEX_BACKEND,
                "auth_mode": self.config.auth_mode,
                "model": self.config.model,
                "schema_version": CODEX_CAPTION_SCHEMA_VERSION,
            },
        )


_CLIENT_CACHE: dict[tuple[Any, ...], CodexAppServerCaptionClient] = {}


def get_codex_app_server_client(
    config: CodexAppServerConfig,
    *,
    client_factory: Callable[..., Any] | None = None,
) -> CodexAppServerCaptionClient:
    if client_factory is not None:
        return CodexAppServerCaptionClient(config, client_factory=client_factory)
    key = _cache_key(config)
    client = _CLIENT_CACHE.get(key)
    if client is None:
        client = CodexAppServerCaptionClient(config)
        _CLIENT_CACHE[key] = client
    return client


def reset_codex_app_server_client_cache() -> None:
    _CLIENT_CACHE.clear()


def caption_image_with_app_server(
    config: CodexAppServerConfig,
    *,
    image_path: str | Path,
    prompt: str,
    output_schema: dict[str, Any] | None = None,
    client_factory: Callable[..., Any] | None = None,
) -> CodexAppServerResult:
    client = get_codex_app_server_client(config, client_factory=client_factory)
    return client.caption_image(image_path=image_path, prompt=prompt, output_schema=output_schema)
