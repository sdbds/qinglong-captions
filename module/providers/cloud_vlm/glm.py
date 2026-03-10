"""GLM Provider"""
from __future__ import annotations

import base64
from typing import Any, Optional

from rich.progress import Progress

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.cloud_vlm_base import CloudVLMProvider
from providers.registry import register_provider


def attempt_glm(*, client, model_path, messages, console, progress=None, task_id=None) -> str:
    """Single-attempt GLM request. Delegates to base class."""
    return CloudVLMProvider.attempt_openai_chat(
        client=client, model_path=model_path, messages=messages,
        console=console, progress=progress, task_id=task_id,
    )


@register_provider("glm")
class GLMProvider(CloudVLMProvider):
    """GLM API Provider"""

    @classmethod
    def can_handle(cls, args, mime: str) -> bool:
        return getattr(args, 'glm_api_key', '') != "" and mime.startswith("video")

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        from zhipuai import ZhipuAI

        client = ZhipuAI(api_key=self.ctx.args.glm_api_key)

        # GLM 只支持视频
        with open(media.uri, "rb") as video_file:
            video_base = base64.b64encode(video_file.read()).decode("utf-8")

        messages = [
            {"role": "system", "content": prompts.system},
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_base}},
                    {"type": "text", "text": prompts.user},
                ],
            },
        ]

        result = attempt_glm(
            client=client,
            model_path=self.ctx.args.glm_model_path,
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
