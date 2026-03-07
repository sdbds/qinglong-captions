"""OLMOCR Provider"""

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.ocr_base import OCRProvider
from providers.registry import register_provider


@register_provider("olmocr")
class OLMOCRProvider(OCRProvider):
    """OLMOCR Provider"""

    default_model_id = "allenai/olmOCR-2-7B-1025"
    default_prompt = ""

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        from module.providers.olmocr_provider import attempt_olmocr

        output_dir = media.extras.get("output_dir")

        result = attempt_olmocr(
            uri=media.uri,
            mime=media.mime,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
            model_id=self._get_model_config("model_id", self.default_model_id),
            prompt_text=prompts.user,
            pixels=media.pixels,
            base64_image=media.blob,
            output_dir=str(output_dir) if output_dir else None,
            temperature=self._get_model_config("temperature", 0.1),
            max_new_tokens=self._get_model_config("max_new_tokens", 512),
        )

        return CaptionResult(
            raw=result if isinstance(result, str) else str(result), metadata={"provider": self.name, "output_dir": str(output_dir)}
        )
