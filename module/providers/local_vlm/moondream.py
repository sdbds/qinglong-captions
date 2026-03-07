"""Moondream Provider"""

from pathlib import Path
from typing import Any, List

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.local_vlm_base import LocalVLMProvider
from providers.registry import register_provider


@register_provider("moondream")
class MoondreamProvider(LocalVLMProvider):
    """Moondream Local VLM Provider"""

    default_model_id = "moondream/moondream3-preview"
    _attn_implementation = "eager"

    @classmethod
    def can_handle(cls, args: Any, mime: str) -> bool:
        vlm_model = getattr(args, "vlm_image_model", "")
        ocr_model = getattr(args, "ocr_model", "")
        # Moondream 可以作为 VLM 或 OCR 使用
        return (vlm_model == "moondream" or ocr_model == "moondream") and mime.startswith("image")

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        from module.providers.moondream_provider import attempt_moondream

        # 读取 captions
        captions: List[str] = []
        captions_path = Path(media.uri).with_suffix(".txt")
        if captions_path.exists():
            with open(captions_path, "r", encoding="utf-8") as f:
                captions = [line.strip() for line in f.readlines()]

        # 读取配置
        vlm_section = self.model_config
        reasoning = vlm_section.get("reasoning")
        ocr_mode = getattr(self.ctx.args, "ocr_model", "") == "moondream"

        result = attempt_moondream(
            model_id=self.model_id,
            mime=media.mime,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
            uri=media.uri,
            pixels=media.pixels,
            image=None,  # 实际图像在 attempt_moondream 内部加载
            captions=captions,
            tags_highlightrate=getattr(self.ctx.args, "tags_highlightrate", 0.0),
            prompt_text=prompts.user,
            reasoning=bool(reasoning) if reasoning is not None else False,
            ocr=ocr_mode,
            task=vlm_section.get("tasks", "caption"),
        )

        return CaptionResult(raw=result, metadata={"provider": self.name})

    def get_retry_config(self):
        cfg = super().get_retry_config()

        def classify(e):
            msg = str(e)
            if "RETRY_MOONDREAM_" in msg:
                return cfg.base_wait
            return None

        cfg.classify_error = classify
        return cfg
