"""Grok Build subscription-backed image caption provider."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from module.providers.base import CaptionResult, MediaContext, MediaModality, PromptContext, RetryConfig
from module.providers.capabilities import ProviderCapabilities
from module.providers.cloud_vlm_base import CloudVLMProvider
from module.providers.codex_schema import (
    CODEX_CAPTION_SCHEMA_VERSION,
    build_codex_caption_prompt,
    filter_caption_payload_by_mode,
)
from module.providers.grok_build_headless import (
    DEFAULT_GROK_BUILD_COMMAND,
    DEFAULT_GROK_BUILD_MODEL,
    DEFAULT_GROK_BUILD_PERMISSION_MODE,
    DEFAULT_GROK_BUILD_PROMPT_JSON_MAX_CHARS,
    DEFAULT_GROK_BUILD_SANDBOX,
    DEFAULT_GROK_BUILD_TIMEOUT_SECONDS,
    GrokBuildHeadlessConfig,
    GrokBuildHeadlessError,
    is_grok_build_source_image_mime,
    run_grok_build_headless_caption,
)
from module.providers.registry import register_provider
from module.providers.utils import encode_image_to_blob

DEFAULT_GROK_BUILD_BACKEND = "headless"
DEFAULT_GROK_BUILD_AUTH_MODE = "cached_token"
SUPPORTED_GROK_BUILD_BACKENDS = frozenset({DEFAULT_GROK_BUILD_BACKEND})
SUPPORTED_GROK_BUILD_AUTH_MODES = frozenset({DEFAULT_GROK_BUILD_AUTH_MODE, "existing"})


@register_provider("grok_build_subscription")
class GrokBuildSubscriptionProvider(CloudVLMProvider):
    """Use a logged-in Grok Build subscription session to caption images."""

    name = "grok_build_subscription"
    capabilities = ProviderCapabilities(
        supports_structured_output=True,
        supports_images=True,
        supports_cloud_concurrency=False,
        supported_mimes=None,
    )

    @classmethod
    def can_handle(cls, args: Any, mime: str) -> bool:
        if not bool(getattr(args, "grok_build_subscription", False)):
            return False
        if not is_grok_build_source_image_mime(mime):
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
        backend = getattr(args, "grok_build_backend", "") or DEFAULT_GROK_BUILD_BACKEND
        if backend not in SUPPORTED_GROK_BUILD_BACKENDS:
            raise GrokBuildHeadlessError(f"Unsupported Grok Build backend: {backend}", kind="config")

        auth_mode = getattr(args, "grok_build_auth_mode", "") or DEFAULT_GROK_BUILD_AUTH_MODE
        if auth_mode not in SUPPORTED_GROK_BUILD_AUTH_MODES:
            raise GrokBuildHeadlessError(f"Unsupported Grok Build auth mode for subscription provider: {auth_mode}", kind="config")

        model = getattr(args, "grok_build_model_name", "") or DEFAULT_GROK_BUILD_MODEL
        effort = getattr(args, "grok_build_effort", "") or ""
        reasoning_effort = getattr(args, "grok_build_reasoning_effort", "") or ""
        mode = getattr(args, "mode", "all")
        prompt = _build_grok_build_caption_prompt(prompts)
        image_name = Path(media.uri).name
        started_at = time.perf_counter()

        try:
            parsed, prompt_json_chars = self._attempt_headless(media, prompt)
        except GrokBuildHeadlessError as exc:
            if exc.kind != "timeout":
                raise
            elapsed = time.perf_counter() - started_at
            self.log(f"Grok Build caption timed out: {image_name} after {elapsed:.1f}s; returning empty caption", "yellow")
            return CaptionResult(
                raw="",
                metadata={
                    "provider": self.name,
                    "backend": backend,
                    "model": model,
                    "effort": effort,
                    "reasoning_effort": reasoning_effort,
                    "auth_mode": auth_mode,
                    "structured": False,
                    "schema_version": CODEX_CAPTION_SCHEMA_VERSION,
                    "skip_reason": "timeout",
                    "error_kind": "timeout",
                    "duration_seconds": round(elapsed, 3),
                },
            )

        elapsed = time.perf_counter() - started_at

        parsed = filter_caption_payload_by_mode(parsed, mode)
        return CaptionResult(
            raw=json.dumps(parsed, ensure_ascii=False),
            parsed=parsed,
            metadata={
                "provider": self.name,
                "backend": backend,
                "model": model,
                "effort": effort,
                "reasoning_effort": reasoning_effort,
                "auth_mode": auth_mode,
                "structured": True,
                "schema_version": CODEX_CAPTION_SCHEMA_VERSION,
                "prompt_json_chars": prompt_json_chars,
                "duration_seconds": round(elapsed, 3),
                "duration_log_label": f"Grok Build caption completed: {image_name}",
                "duration_log_style": "green",
            },
        )

    def _attempt_headless(self, media: MediaContext, prompt: str) -> tuple[dict, int]:
        args = self.ctx.args
        configured_isolated_cwd = getattr(args, "grok_build_isolated_cwd", "") or ""
        if configured_isolated_cwd:
            isolated_cwd = Path(configured_isolated_cwd).expanduser()
        else:
            isolated_cwd = Path(tempfile.gettempdir()) / "qinglong-captions-grok-build-work"
        isolated_cwd.mkdir(parents=True, exist_ok=True)

        config = GrokBuildHeadlessConfig(
            command=getattr(args, "grok_build_command", "") or DEFAULT_GROK_BUILD_COMMAND,
            model=getattr(args, "grok_build_model_name", "") or DEFAULT_GROK_BUILD_MODEL,
            effort=getattr(args, "grok_build_effort", "") or "",
            reasoning_effort=getattr(args, "grok_build_reasoning_effort", "") or "",
            timeout=float(
                getattr(args, "grok_build_timeout", DEFAULT_GROK_BUILD_TIMEOUT_SECONDS)
                or DEFAULT_GROK_BUILD_TIMEOUT_SECONDS
            ),
            isolated_cwd=str(isolated_cwd),
            permission_mode=getattr(args, "grok_build_permission_mode", "") or DEFAULT_GROK_BUILD_PERMISSION_MODE,
            sandbox=getattr(args, "grok_build_sandbox", "") or DEFAULT_GROK_BUILD_SANDBOX,
            prompt_json_max_chars=_positive_int(
                getattr(args, "grok_build_prompt_json_max_chars", DEFAULT_GROK_BUILD_PROMPT_JSON_MAX_CHARS),
                DEFAULT_GROK_BUILD_PROMPT_JSON_MAX_CHARS,
            ),
        )
        result = run_grok_build_headless_caption(
            config,
            image_path=media.uri,
            prompt=prompt,
            mime=media.mime,
        )
        return result.parsed, result.prompt_json_chars

    def get_retry_config(self) -> RetryConfig:
        return RetryConfig(max_retries=1, base_wait=0.0)

    def display_name(self, mime: str) -> str:
        return "grok_build_subscription"


def _build_grok_build_caption_prompt(prompts: PromptContext) -> str:
    prompt = build_codex_caption_prompt(system_prompt=prompts.system, user_prompt=prompts.user)
    return "\n".join(
        [
            prompt,
            "",
            "Grok Build runtime constraints:",
            "Do not read files, inspect the workspace, run tools, use web search, or include tool output.",
            "Use only the attached image and the project prompt context above.",
        ]
    )


def _positive_int(value: Any, default: int = 1) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default
