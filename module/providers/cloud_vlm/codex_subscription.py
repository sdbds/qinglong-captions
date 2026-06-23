"""Codex subscription-backed image caption provider."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from module.providers.base import CaptionResult, MediaContext, MediaModality, PromptContext, RetryConfig
from module.providers.cloud_vlm_base import CloudVLMProvider
from module.providers.capabilities import ProviderCapabilities
from module.providers.codex_app_server import (
    DEFAULT_CODEX_AUTH_MODE,
    DEFAULT_CODEX_BACKEND,
    DEFAULT_CODEX_REASONING_EFFORT,
    SUPPORTED_CODEX_BACKENDS,
    CodexAppServerConfig,
    CodexAppServerError,
    build_codex_disable_mcp_config_overrides,
    caption_image_with_app_server,
    normalize_codex_reasoning_effort,
)
from module.providers.codex_exec import (
    CodexExecConfig,
    CodexExecError,
    DEFAULT_CODEX_TIMEOUT_SECONDS,
    run_codex_exec_caption,
    write_default_caption_schema,
)
from module.providers.codex_schema import (
    CODEX_CAPTION_SCHEMA_VERSION,
    build_codex_caption_prompt,
    filter_caption_payload_by_mode,
    load_caption_schema,
)
from module.providers.image_template import (
    active_image_template,
    build_freeform_caption_prompt,
    image_template_output,
    parse_freeform_caption_output,
)
from module.providers.registry import register_provider
from module.providers.utils import encode_image_to_blob
from utils.console_util import print_exception


_CODEX_TRANSCODE_IMAGE_MIMES = {"image/avif", "image/heic", "image/heif"}
_CODEX_TRANSCODE_IMAGE_SUFFIXES = {".avif", ".heic", ".heif"}


def _codex_needs_jpeg_input(path: Path, mime: str = "") -> bool:
    return mime.lower() in _CODEX_TRANSCODE_IMAGE_MIMES or path.suffix.lower() in _CODEX_TRANSCODE_IMAGE_SUFFIXES


def _convert_to_temp_jpeg(path: Path, *, quality: int) -> Path:
    from PIL import Image

    temp_path: Path | None = None
    try:
        with Image.open(path) as image:
            image.load()
            if "xmp" in image.info:
                del image.info["xmp"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            with tempfile.NamedTemporaryFile(prefix=f"{path.stem}_codex_", suffix=".jpg", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            image.save(temp_path, format="JPEG", quality=quality)
            return temp_path
    except Exception as exc:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise CodexAppServerError(f"Failed to convert image to JPEG before Codex SDK input: {path}", kind="image") from exc


def _delete_temp_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _is_rate_limited_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        getattr(exc, "kind", "") == "rate_limited"
        or "429" in text
        or "too many requests" in text
        or "rate limit" in text
        or "rate_limited" in text
    )


def _codex_retry_wait(exc: Exception, args: Any) -> float | None:
    if _is_rate_limited_error(exc):
        try:
            return max(0.0, float(getattr(args, "wait_time", 10.0)))
        except (TypeError, ValueError):
            return 10.0
    if bool(getattr(exc, "retryable", False)):
        return 0.0
    return None


def _codex_retry_exhausted_result(args: Any, exc: Exception) -> CaptionResult:
    retryable = bool(getattr(exc, "retryable", False))
    if _is_rate_limited_error(exc):
        error_kind = "rate_limited"
    elif retryable:
        error_kind = str(getattr(exc, "kind", "") or "retryable")
    else:
        raise exc

    backend = getattr(args, "codex_backend", "") or DEFAULT_CODEX_BACKEND
    return CaptionResult(
        raw="",
        metadata={
            "provider": CodexSubscriptionProvider.name,
            "backend": backend,
            "model": getattr(args, "codex_model_name", "") or "gpt-5.4",
            "service_tier": _codex_service_tier(args),
            "reasoning_effort": _codex_reasoning_effort(args),
            "auth_mode": getattr(args, "codex_auth_mode", DEFAULT_CODEX_AUTH_MODE) or DEFAULT_CODEX_AUTH_MODE,
            "structured": False,
            "schema_version": CODEX_CAPTION_SCHEMA_VERSION,
            "skip_reason": error_kind,
            "error_kind": error_kind,
            "retry_exhausted": True,
            "error": str(exc),
        },
    )


def _codex_service_tier(args: Any) -> str:
    configured = str(getattr(args, "codex_service_tier", "") or "").strip()
    if configured:
        return configured
    return "fast" if bool(getattr(args, "codex_fast", False)) else ""


def _codex_reasoning_effort(args: Any) -> str:
    return normalize_codex_reasoning_effort(
        getattr(args, "codex_reasoning_effort", DEFAULT_CODEX_REASONING_EFFORT),
        default=DEFAULT_CODEX_REASONING_EFFORT,
    )


_CODEX_STREAM_EVENT_PREFIX = "Codex app-server: event "


def _compact_codex_stream_events(events: list[str]) -> str:
    compacted: list[str] = []
    previous = ""
    count = 0

    def _event_key(event: str) -> str:
        if event.startswith("item/agentMessage/delta"):
            return "item/agentMessage/delta"
        if event.startswith("thread/tokenUsage/updated"):
            return "thread/tokenUsage/updated"
        return event

    def _append_current() -> None:
        if not previous:
            return
        suffix = f" x{count}" if count > 1 else ""
        compacted.append(f"{previous}{suffix}")

    for event in events:
        key = _event_key(event)
        if key == previous:
            count += 1
            continue
        _append_current()
        previous = key
        count = 1
    _append_current()

    if len(compacted) > 12:
        compacted = compacted[:8] + [f"... {len(compacted) - 10} more events ..."] + compacted[-2:]
    return " | ".join(compacted)


class _CodexProgressCoalescer:
    def __init__(self, emit):
        self._emit = emit
        self._events: list[str] = []

    def __call__(self, message: str) -> None:
        text = str(message or "").strip()
        if not text:
            return
        if text.startswith(_CODEX_STREAM_EVENT_PREFIX):
            self._events.append(text[len(_CODEX_STREAM_EVENT_PREFIX) :])
            return
        self.flush()
        self._emit(text)

    def flush(self) -> None:
        if not self._events:
            return
        self._emit(f"Codex app-server stream: {_compact_codex_stream_events(self._events)}")
        self._events.clear()


@register_provider("codex_subscription")
class CodexSubscriptionProvider(CloudVLMProvider):
    """Use a logged-in Codex/ChatGPT subscription session to caption images."""

    name = "codex_subscription"
    capabilities = ProviderCapabilities(
        supports_structured_output=True,
        supports_images=True,
        supports_cloud_concurrency=True,
    )

    @classmethod
    def can_handle(cls, args: Any, mime: str) -> bool:
        if not bool(getattr(args, "codex_subscription", False)):
            return False
        if not mime.startswith("image"):
            return False
        if getattr(args, "ocr_model", "") or getattr(args, "vlm_image_model", ""):
            return False
        return True

    def prepare_media(self, uri: str, mime: str, args: Any) -> MediaContext:
        file_path = Path(uri)
        pixels = None
        if mime.startswith("image"):
            _blob, pixels = encode_image_to_blob(
                str(file_path),
                to_rgb=True,
                quality=self.get_image_quality(),
            )
        return MediaContext(
            uri=str(file_path),
            mime=mime,
            sha256hash="",
            modality=MediaModality.IMAGE,
            file_size=file_path.stat().st_size if file_path.exists() else 0,
            pixels=pixels,
        )

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        args = self.ctx.args
        backend = getattr(args, "codex_backend", "") or DEFAULT_CODEX_BACKEND
        if backend not in SUPPORTED_CODEX_BACKENDS:
            raise CodexAppServerError(f"Unsupported Codex backend: {backend}", kind="config")

        model = getattr(args, "codex_model_name", "") or "gpt-5.4"
        mode = getattr(args, "mode", "all")
        # Image prompt template: when a non-default template is active, yield the
        # forced rating schema and let the model follow the template freely.
        template_id = active_image_template(args)
        structured = not template_id
        if structured:
            prompt = build_codex_caption_prompt(system_prompt=prompts.system, user_prompt=prompts.user)
            output_contract = ""
        else:
            output_contract = image_template_output(self.ctx.config.get("prompts", {}), template_id)
            prompt = build_freeform_caption_prompt(
                system_prompt=prompts.system,
                user_prompt=prompts.user,
                output=output_contract,
            )
        service_tier = _codex_service_tier(args)
        reasoning_effort = _codex_reasoning_effort(args)
        image_name = Path(media.uri).name
        started_at = time.perf_counter()

        try:
            if backend == "exec":
                result = self._attempt_exec(media, prompt, structured=structured)
            else:
                result = self._attempt_app_server(media, prompt, structured=structured)
        except (CodexAppServerError, CodexExecError) as exc:
            if getattr(exc, "kind", "") != "timeout":
                raise
            elapsed = time.perf_counter() - started_at
            self.log(f"Codex caption timed out: {image_name} after {elapsed:.1f}s; returning empty caption", "yellow")
            return CaptionResult(
                raw="",
                metadata={
                    "provider": self.name,
                    "backend": backend,
                    "model": model,
                    "service_tier": service_tier,
                    "reasoning_effort": reasoning_effort,
                    "auth_mode": getattr(args, "codex_auth_mode", DEFAULT_CODEX_AUTH_MODE) or DEFAULT_CODEX_AUTH_MODE,
                    "structured": False,
                    "schema_version": CODEX_CAPTION_SCHEMA_VERSION,
                    "skip_reason": "timeout",
                    "error_kind": "timeout",
                    "duration_seconds": round(elapsed, 3),
                },
            )
        elapsed = time.perf_counter() - started_at

        if not structured:
            caption_raw, parsed = parse_freeform_caption_output(result.raw, output_contract)
            return CaptionResult(
                raw=caption_raw,
                parsed=parsed,
                metadata={
                    "provider": self.name,
                    "backend": backend,
                    "model": model,
                    "service_tier": service_tier,
                    "reasoning_effort": reasoning_effort,
                    "auth_mode": getattr(args, "codex_auth_mode", DEFAULT_CODEX_AUTH_MODE) or DEFAULT_CODEX_AUTH_MODE,
                    "structured": False,
                    "image_template": template_id,
                    "output_contract": output_contract,
                    "schema_version": CODEX_CAPTION_SCHEMA_VERSION,
                    "duration_seconds": round(elapsed, 3),
                    "duration_log_label": f"Codex caption completed: {image_name}",
                    "duration_log_style": "green",
                },
            )

        parsed = filter_caption_payload_by_mode(result.parsed, mode)
        return CaptionResult(
            raw=json.dumps(parsed, ensure_ascii=False),
            parsed=parsed,
            metadata={
                "provider": self.name,
                "backend": backend,
                "model": model,
                "service_tier": service_tier,
                "reasoning_effort": reasoning_effort,
                "auth_mode": getattr(args, "codex_auth_mode", DEFAULT_CODEX_AUTH_MODE) or DEFAULT_CODEX_AUTH_MODE,
                "structured": True,
                "schema_version": CODEX_CAPTION_SCHEMA_VERSION,
                "duration_seconds": round(elapsed, 3),
                "duration_log_label": f"Codex caption completed: {image_name}",
                "duration_log_style": "green",
            },
        )

    def _attempt_exec(self, media: MediaContext, prompt: str, *, structured: bool = True):
        args = self.ctx.args
        command = getattr(args, "codex_command", "") or "codex"
        model = getattr(args, "codex_model_name", "") or "gpt-5.4"
        timeout = float(getattr(args, "codex_timeout", DEFAULT_CODEX_TIMEOUT_SECONDS) or DEFAULT_CODEX_TIMEOUT_SECONDS)
        sandbox = getattr(args, "codex_sandbox", "") or "read-only"
        service_tier = _codex_service_tier(args)
        reasoning_effort = _codex_reasoning_effort(args)
        codex_home = getattr(args, "codex_home", "") or ""
        configured_isolated_cwd = getattr(args, "codex_isolated_cwd", "") or ""
        configured_schema = getattr(args, "codex_output_schema", "") or ""

        with tempfile.TemporaryDirectory(prefix="qinglong-codex-caption-") as tmp:
            tmp_path = Path(tmp)
            isolated_cwd = Path(configured_isolated_cwd).expanduser() if configured_isolated_cwd else tmp_path / "work"
            isolated_cwd.mkdir(parents=True, exist_ok=True)

            if not structured:
                # Freeform template path: no rating schema is forced.
                schema_path = None
            elif configured_schema:
                schema_path = Path(configured_schema).expanduser()
                if not schema_path.exists():
                    raise CodexExecError(
                        f"Codex output schema does not exist: {schema_path}",
                        kind="environment",
                    )
            else:
                schema_path = write_default_caption_schema(tmp_path / "caption_schema.json")

            output_path = tmp_path / "last_message.json"
            config = CodexExecConfig(
                command=command,
                model=model,
                service_tier=service_tier,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                sandbox=sandbox,
                codex_home=codex_home,
                isolated_cwd=str(isolated_cwd),
            )
            try:
                exec_result = run_codex_exec_caption(
                    config,
                    image_path=media.uri,
                    prompt=prompt,
                    schema_path=schema_path,
                    output_path=output_path,
                    structured=structured,
                )
            except CodexExecError:
                raise
            except Exception as exc:
                print_exception(self.ctx.console, exc, prefix="Codex subscription provider failed")
                raise

        return exec_result

    def _attempt_app_server(self, media: MediaContext, prompt: str, *, structured: bool = True):
        args = self.ctx.args
        configured_schema = getattr(args, "codex_output_schema", "") or ""
        model = getattr(args, "codex_model_name", "") or "gpt-5.4"
        timeout = float(getattr(args, "codex_timeout", DEFAULT_CODEX_TIMEOUT_SECONDS) or DEFAULT_CODEX_TIMEOUT_SECONDS)
        service_tier = _codex_service_tier(args)
        reasoning_effort = _codex_reasoning_effort(args)
        if structured:
            try:
                output_schema = load_caption_schema(configured_schema)
            except Exception as exc:
                raise CodexAppServerError(str(exc), kind="schema", cause=exc) from exc
        else:
            # Freeform template path: no rating schema is forced.
            output_schema = None

        configured_isolated_cwd = getattr(args, "codex_isolated_cwd", "") or ""
        if configured_isolated_cwd:
            isolated_cwd = Path(configured_isolated_cwd).expanduser()
        else:
            isolated_cwd = Path(tempfile.gettempdir()) / "qinglong-captions-codex-work"
        isolated_cwd.mkdir(parents=True, exist_ok=True)

        auth_mode = getattr(args, "codex_auth_mode", "") or DEFAULT_CODEX_AUTH_MODE
        codex_home = getattr(args, "codex_home", "") or ""
        disable_mcp = bool(getattr(args, "codex_disable_mcp", True))
        config = CodexAppServerConfig(
            model=model,
            service_tier=service_tier,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            sandbox=getattr(args, "codex_sandbox", "") or "read-only",
            auth_mode=auth_mode,
            api_key=getattr(args, "codex_api_key", "") or "",
            codex_home=codex_home,
            runtime_path=getattr(args, "codex_runtime_path", "") or "",
            isolated_cwd=str(isolated_cwd),
            config_overrides=build_codex_disable_mcp_config_overrides(codex_home) if disable_mcp else (),
        )
        app_server_max_concurrency = _positive_int(getattr(args, "codex_max_concurrency", 1), 1)
        source_image_path = Path(media.uri)
        send_image_path = source_image_path
        temp_image_path: Path | None = None
        if _codex_needs_jpeg_input(source_image_path, media.mime):
            temp_image_path = _convert_to_temp_jpeg(source_image_path, quality=self.get_image_quality())
            send_image_path = temp_image_path
        caption_kwargs = {
            "image_path": str(send_image_path),
            "prompt": prompt,
            "output_schema": output_schema,
            "structured": structured,
        }
        if app_server_max_concurrency > 1:
            caption_kwargs["max_concurrency"] = app_server_max_concurrency
        try:
            result = caption_image_with_app_server(
                config,
                **caption_kwargs,
            )
        except CodexAppServerError:
            raise
        except Exception as exc:
            print_exception(self.ctx.console, exc, prefix="Codex app-server provider failed")
            raise
        finally:
            _delete_temp_file(temp_image_path)
        return result

    def get_retry_config(self) -> RetryConfig:
        args = self.ctx.args
        return RetryConfig(
            max_retries=_positive_int(getattr(args, "max_retries", 2), 2),
            base_wait=0.0,
            classify_error=lambda exc: _codex_retry_wait(exc, args),
            on_exhausted=lambda exc: _codex_retry_exhausted_result(args, exc),
        )

    def display_name(self, mime: str) -> str:
        return "codex_subscription"


def _positive_int(value: Any, default: int = 1) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default
