"""Xiaomi MiMo OpenAI-compatible Provider.

MiMo exposes an OpenAI-compatible chat completions endpoint. The vision
captioning path mirrors the Kimi-Code/Kimi-VL provider shape while using
MiMo's documented base URL and token parameter.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from module.providers.base import CaptionResult, MediaContext, PromptContext
from module.providers.cloud_vlm.kimi_vl import attempt_kimi_vl, ensure_kimi_dual_caption_prompt
from module.providers.cloud_vlm_base import CloudVLMProvider
from module.providers.registry import register_provider
from module.providers.utils import build_vision_messages
from utils.console_util import print_exception

MIMO_DEFAULT_BASE_URL = "https://token-plan-sgp.xiaomimimo.com/v1"
MIMO_DEFAULT_MODEL = "mimo-v2.5"
MIMO_BASE64_VIDEO_LIMIT_BYTES = 50 * 1024 * 1024
MIMO_RECOMMENDED_TEMPERATURE = 1.0
MIMO_HIGH_RISK_REJECTION_TEXT = "The request was rejected because it was considered high risk"


@register_provider("mimo")
class MimoProvider(CloudVLMProvider):
    """Xiaomi MiMo provider using the OpenAI-compatible Chat Completions API."""

    @classmethod
    def can_handle(cls, args: Any, mime: str) -> bool:
        return getattr(args, "mimo_api_key", "") != "" and mime.startswith(("image", "video"))

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        from openai import OpenAI

        api_key = self.ctx.args.mimo_api_key
        base_url = getattr(self.ctx.args, "mimo_base_url", MIMO_DEFAULT_BASE_URL)
        model_path = getattr(self.ctx.args, "mimo_model_path", MIMO_DEFAULT_MODEL)

        client = OpenAI(api_key=api_key, base_url=base_url)

        messages = self._build_messages(media, prompts)
        if not messages:
            return CaptionResult(raw="")

        mimo_config = self.ctx.config.get("mimo", {}) if self.ctx.config else {}
        thinking = mimo_config.get("thinking", "disabled") if isinstance(mimo_config, dict) else "disabled"
        max_completion_tokens = self._resolve_max_completion_tokens(mimo_config)

        result = attempt_kimi_vl(
            client=client,
            model_path=model_path,
            messages=messages,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
            uri=media.uri,
            image_pixels=media.pixels,
            pair_pixels=media.pair_pixels,
            thinking=thinking,
            mode=getattr(self.ctx.args, "mode", "all"),
            max_tokens=max_completion_tokens,
            max_tokens_param="max_completion_tokens",
            temperature=MIMO_RECOMMENDED_TEMPERATURE,
        )

        try:
            parsed = json.loads(result) if isinstance(result, str) else result
            if isinstance(parsed, dict):
                mode = getattr(self.ctx.args, "mode", "all")
                if mode == "short":
                    parsed.pop("long", None)
                    parsed.pop("long_description", None)
                elif mode == "long":
                    parsed.pop("short", None)
                    parsed.pop("short_description", None)
                return CaptionResult(
                    raw=json.dumps(parsed, ensure_ascii=False),
                    parsed=parsed,
                    metadata={"provider": self.name, "model": model_path},
                )
        except Exception:
            pass

        return CaptionResult(
            raw=result if isinstance(result, str) else json.dumps(result, ensure_ascii=False),
            metadata={"provider": self.name, "model": model_path},
        )

    def _build_messages(self, media: MediaContext, prompts: PromptContext) -> list[dict[str, Any]]:
        if media.mime.startswith("video"):
            self._ensure_video_base64_within_limit(media)
            with open(media.uri, "rb") as f:
                video_base = base64.b64encode(f.read()).decode("utf-8")
            video_data_url = f"data:{media.mime};base64,{video_base}"
            return [
                {"role": "system", "content": prompts.system},
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": video_data_url}},
                        {"type": "text", "text": prompts.user},
                    ],
                },
            ]

        if media.mime.startswith("image"):
            if media.blob is None:
                return []

            pair_dir = getattr(self.ctx.args, "pair_dir", "")
            if pair_dir and not media.pair_blob:
                return []

            system_prompt = ensure_kimi_dual_caption_prompt(prompts.system)
            return build_vision_messages(
                system_prompt,
                prompts.user,
                media.blob,
                pair_blob=media.pair_blob if pair_dir else None,
                text_first=False,
            )

        return []

    @staticmethod
    def _resolve_max_completion_tokens(config: Any) -> int:
        if isinstance(config, dict):
            value = config.get("max_completion_tokens", 8192)
        else:
            value = 8192
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 8192

    def get_response_skip_reason(self, result: CaptionResult, media: MediaContext, args: Any) -> str:
        raw = str(result.raw or "")
        if MIMO_HIGH_RISK_REJECTION_TEXT.lower() in raw.lower():
            return "mimo_high_risk_rejection"
        return super().get_response_skip_reason(result, media, args)

    @staticmethod
    def _ensure_video_base64_within_limit(media: MediaContext) -> None:
        file_size = media.file_size
        if file_size <= 0:
            try:
                file_size = Path(media.uri).stat().st_size
            except OSError:
                return

        encoded_size = ((file_size + 2) // 3) * 4
        data_url_size = encoded_size + len(f"data:{media.mime};base64,")
        if data_url_size <= MIMO_BASE64_VIDEO_LIMIT_BYTES:
            return

        size_mb = data_url_size / (1024 * 1024)
        limit_mb = MIMO_BASE64_VIDEO_LIMIT_BYTES / (1024 * 1024)
        raise ValueError(
            f"MiMo video base64 limit exceeded: encoded input would be {size_mb:.1f} MB, "
            f"limit is {limit_mb:.0f} MB. Split or compress the video before using MiMo."
        )

    def get_retry_config(self):
        from module.providers.utils import classify_remote_api_error

        cfg = super().get_retry_config()
        cfg.classify_error = lambda e: classify_remote_api_error(
            e,
            base_wait=cfg.base_wait,
            retry_markers=("RETRY_EMPTY_CONTENT",),
            fast_fail_markers=("mimo video base64 limit exceeded",),
        )
        cfg.on_exhausted = lambda e: (
            print_exception(self.ctx.console, e, prefix="MiMo API retries exhausted", summary_style="yellow")
            or CaptionResult.failed(str(e), metadata={"provider": self.name, "retry_exhausted": True})
        )
        return cfg
