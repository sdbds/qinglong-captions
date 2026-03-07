"""Hunyuan OCR Provider"""

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.ocr_base import OCRProvider
from providers.registry import register_provider


@register_provider("hunyuan_ocr")
class HunyuanOCRProvider(OCRProvider):
    """Hunyuan OCR Provider"""

    default_model_id = "tencent/HunyuanOCR"
    default_prompt = ""

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        from module.providers.hunyuan_ocr_provider import attempt_hunyuan_ocr

        output_dir = media.extras.get("output_dir")

        result = attempt_hunyuan_ocr(
            uri=media.uri,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
            model_id=self._get_model_config("model_id", self.default_model_id),
            prompt_text=prompts.user,
            pixels=media.pixels,
            output_dir=str(output_dir) if output_dir else None,
            max_new_tokens=self._get_model_config("max_new_tokens", 16384),
        )

        return CaptionResult(
            raw=result if isinstance(result, str) else str(result), metadata={"provider": self.name, "output_dir": str(output_dir)}
        )
