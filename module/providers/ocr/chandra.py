"""Chandra OCR Provider"""

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.ocr_base import OCRProvider
from providers.registry import register_provider


@register_provider("chandra_ocr")
class ChandraOCRProvider(OCRProvider):
    """Chandra OCR Provider"""

    default_model_id = "datalab-to/chandra"
    default_prompt = ""

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        from module.providers.chandra_ocr_provider import attempt_chandra_ocr

        output_dir = media.extras.get("output_dir")

        # 读取 prompt_type 配置
        prompt_type = self._get_model_config(
            "prompt_type", self.ctx.config.get("prompts", {}).get("chandra_ocr_prompt_type", "ocr_layout")
        )

        result = attempt_chandra_ocr(
            uri=media.uri,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
            model_id=self._get_model_config("model_id", self.default_model_id),
            prompt_type=prompt_type,
            pixels=media.pixels,
            output_dir=str(output_dir) if output_dir else None,
            max_new_tokens=self._get_model_config("max_new_tokens", 8192),
        )

        return CaptionResult(
            raw=result if isinstance(result, str) else str(result), metadata={"provider": self.name, "output_dir": str(output_dir)}
        )
