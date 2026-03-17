"""Dots OCR Provider shell."""

from __future__ import annotations

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.ocr_base import OCRProvider
from providers.registry import register_provider


@register_provider("dots_ocr")
class DotsOCRProvider(OCRProvider):
    """Concrete shell for dots OCR discovery and routing tests."""

    default_model_id = "davanstrien/dots.ocr-1.5"
    default_prompt = ""

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        raise NotImplementedError("dots_ocr implementation lands in later tasks")
