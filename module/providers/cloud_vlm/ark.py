"""Ark (Volcano Engine) Provider"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Optional

from rich.progress import Progress

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.cloud_vlm_base import CloudVLMProvider
from providers.registry import register_provider


# ---------------------------------------------------------------------------
# Ark streaming / attempt helpers (migrated from ark_provider.py)
# ---------------------------------------------------------------------------

def attempt_ark(*, client, model_path, messages, console, progress=None, task_id=None) -> str:
    """Single-attempt Ark request. Delegates to base class."""
    return CloudVLMProvider.attempt_openai_chat(
        client=client, model_path=model_path, messages=messages,
        console=console, progress=progress, task_id=task_id,
        model_path_replace=(".", "-"),
    )


# ---------------------------------------------------------------------------
# V2 Provider class
# ---------------------------------------------------------------------------

@register_provider("ark")
class ArkProvider(CloudVLMProvider):
    """Ark (Volcano Engine) API Provider"""

    @classmethod
    def can_handle(cls, args, mime: str) -> bool:
        return getattr(args, "ark_api_key", "") != "" and mime.startswith("video")

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        from volcenginesdkarkruntime import Ark

        client = Ark(api_key=self.ctx.args.ark_api_key)

        self.log(f"Ark model: {getattr(self.ctx.args, 'ark_model_path', '')}", "blue")

        # 读取配置
        ark_section = self.ctx.config.get("ark", {})
        cfg_fps = ark_section.get("fps")
        ark_fps = float(cfg_fps) if cfg_fps is not None else getattr(self.ctx.args, "ark_fps", 2)
        self.log(f"Ark fps: {ark_fps}", "blue")

        # 编码视频
        with open(media.uri, "rb") as video_file:
            video_base = base64.b64encode(video_file.read()).decode("utf-8")

        try:
            file_size = Path(media.uri).stat().st_size
        except Exception:
            file_size = -1
        self.log(f"Ark input size: {file_size} bytes; base64 length: {len(video_base)}", "blue")

        messages = [
            {"role": "system", "content": prompts.system},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:{media.mime};base64,{video_base}",
                            "fps": ark_fps,
                        },
                    },
                    {"type": "text", "text": prompts.user},
                ],
            },
        ]

        result = attempt_ark(
            client=client,
            model_path=self.ctx.args.ark_model_path,
            messages=messages,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
        )

        return CaptionResult(raw=result, metadata={"provider": self.name})

    def get_retry_config(self):
        cfg = super().get_retry_config()

        def classify(e):
            msg = str(e)
            if "429" in msg:
                return 59.0
            if "502" in msg or "RETRY_EMPTY_CONTENT" in msg:
                return cfg.base_wait
            return None

        cfg.classify_error = classify
        return cfg
