"""Codex subscription-backed image caption provider."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from module.providers.base import CaptionResult, MediaContext, MediaModality, PromptContext, RetryConfig
from module.providers.cloud_vlm_base import CloudVLMProvider
from module.providers.codex_app_server import (
    DEFAULT_CODEX_AUTH_MODE,
    DEFAULT_CODEX_BACKEND,
    SUPPORTED_CODEX_BACKENDS,
    CodexAppServerConfig,
    CodexAppServerError,
    caption_image_with_app_server,
)
from module.providers.codex_exec import (
    CodexExecConfig,
    CodexExecError,
    run_codex_exec_caption,
    write_default_caption_schema,
)
from module.providers.codex_schema import (
    CODEX_CAPTION_SCHEMA_VERSION,
    build_codex_caption_prompt,
    filter_caption_payload_by_mode,
    load_caption_schema,
)
from module.providers.registry import register_provider
from utils.console_util import print_exception


@register_provider("codex_subscription")
class CodexSubscriptionProvider(CloudVLMProvider):
    """Use a logged-in Codex/ChatGPT subscription session to caption images."""

    name = "codex_subscription"

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
        return MediaContext(
            uri=str(file_path),
            mime=mime,
            sha256hash="",
            modality=MediaModality.IMAGE,
            file_size=file_path.stat().st_size if file_path.exists() else 0,
        )

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        args = self.ctx.args
        backend = getattr(args, "codex_backend", "") or DEFAULT_CODEX_BACKEND
        if backend not in SUPPORTED_CODEX_BACKENDS:
            raise CodexAppServerError(f"Unsupported Codex backend: {backend}", kind="config")

        model = getattr(args, "codex_model_name", "") or "gpt-5.4-mini"
        mode = getattr(args, "mode", "all")
        prompt = build_codex_caption_prompt(system_prompt=prompts.system, user_prompt=prompts.user)

        if backend == "exec":
            parsed = self._attempt_exec(media, prompt)
        else:
            parsed = self._attempt_app_server(media, prompt)

        parsed = filter_caption_payload_by_mode(parsed, mode)
        return CaptionResult(
            raw=json.dumps(parsed, ensure_ascii=False),
            parsed=parsed,
            metadata={
                "provider": self.name,
                "backend": backend,
                "model": model,
                "auth_mode": getattr(args, "codex_auth_mode", DEFAULT_CODEX_AUTH_MODE) or DEFAULT_CODEX_AUTH_MODE,
                "structured": True,
                "schema_version": CODEX_CAPTION_SCHEMA_VERSION,
            },
        )

    def _attempt_exec(self, media: MediaContext, prompt: str) -> dict:
        args = self.ctx.args
        command = getattr(args, "codex_command", "") or "codex"
        model = getattr(args, "codex_model_name", "") or "gpt-5.4-mini"
        timeout = float(getattr(args, "codex_timeout", 180) or 180)
        sandbox = getattr(args, "codex_sandbox", "") or "read-only"
        codex_home = getattr(args, "codex_home", "") or ""
        configured_isolated_cwd = getattr(args, "codex_isolated_cwd", "") or ""
        configured_schema = getattr(args, "codex_output_schema", "") or ""

        with tempfile.TemporaryDirectory(prefix="qinglong-codex-caption-") as tmp:
            tmp_path = Path(tmp)
            isolated_cwd = Path(configured_isolated_cwd).expanduser() if configured_isolated_cwd else tmp_path / "work"
            isolated_cwd.mkdir(parents=True, exist_ok=True)

            if configured_schema:
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
                )
            except CodexExecError:
                raise
            except Exception as exc:
                print_exception(self.ctx.console, exc, prefix="Codex subscription provider failed")
                raise

        return exec_result.parsed

    def _attempt_app_server(self, media: MediaContext, prompt: str) -> dict:
        args = self.ctx.args
        configured_schema = getattr(args, "codex_output_schema", "") or ""
        try:
            output_schema = load_caption_schema(configured_schema)
        except Exception as exc:
            raise CodexAppServerError(str(exc), kind="schema", cause=exc) from exc

        configured_isolated_cwd = getattr(args, "codex_isolated_cwd", "") or ""
        if configured_isolated_cwd:
            isolated_cwd = Path(configured_isolated_cwd).expanduser()
        else:
            isolated_cwd = Path(tempfile.gettempdir()) / "qinglong-captions-codex-work"
        isolated_cwd.mkdir(parents=True, exist_ok=True)

        auth_mode = getattr(args, "codex_auth_mode", "") or DEFAULT_CODEX_AUTH_MODE
        config = CodexAppServerConfig(
            model=getattr(args, "codex_model_name", "") or "gpt-5.4-mini",
            timeout=float(getattr(args, "codex_timeout", 180) or 180),
            sandbox=getattr(args, "codex_sandbox", "") or "read-only",
            auth_mode=auth_mode,
            api_key=getattr(args, "codex_api_key", "") or "",
            codex_home=getattr(args, "codex_home", "") or "",
            runtime_path=getattr(args, "codex_runtime_path", "") or "",
            isolated_cwd=str(isolated_cwd),
        )
        try:
            result = caption_image_with_app_server(
                config,
                image_path=media.uri,
                prompt=prompt,
                output_schema=output_schema,
            )
        except CodexAppServerError:
            raise
        except Exception as exc:
            print_exception(self.ctx.console, exc, prefix="Codex app-server provider failed")
            raise
        return result.parsed

    def get_retry_config(self) -> RetryConfig:
        return RetryConfig(max_retries=1, base_wait=0.0)

    def display_name(self, mime: str) -> str:
        return "codex_subscription"
