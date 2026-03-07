"""
DeepSeek OCR Provider

基于 OCRProvider 基类的 DeepSeek OCR 实现
"""

from typing import Any

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.ocr_base import OCRProvider
from providers.registry import register_provider


@register_provider("deepseek_ocr")
class DeepSeekOCRProvider(OCRProvider):
    """DeepSeek OCR Provider"""

    default_model_id = "deepseek-ai/DeepSeek-OCR-2"
    default_prompt = "<image>\n<|grounding|>Convert the document to markdown."

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        """执行 DeepSeek OCR"""
        from module.providers.deepseek_ocr_provider import attempt_deepseek_ocr

        output_dir = media.extras.get("output_dir")

        result = attempt_deepseek_ocr(
            uri=media.uri,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
            model_id=self._get_model_config("model_id", self.default_model_id),
            prompt_text=prompts.user,
            pixels=media.pixels,
            output_dir=str(output_dir) if output_dir else None,
            base_size=getattr(self.ctx.args, "deepseek_base_size", 1024),
            image_size=getattr(self.ctx.args, "deepseek_image_size", 768),
            crop_mode=getattr(self.ctx.args, "deepseek_crop_mode", True),
        )

        # 包装为 CaptionResult
        return CaptionResult(
            raw=result if isinstance(result, str) else str(result), metadata={"provider": self.name, "output_dir": str(output_dir)}
        )
