"""Dots OCR Provider shell."""

from __future__ import annotations

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.ocr_base import OCRProvider
from providers.registry import register_provider

DEFAULT_PROMPT_MODE = "prompt_layout_all_en"
DEFAULT_SVG_MODEL_ID = "davanstrien/dots.ocr-1.5-svg"


def _load_upstream_prompt_mapping() -> dict[str, str]:
    """Load the official dots OCR prompt-mode mapping from the upstream package."""
    try:
        from dots_ocr.utils import dict_promptmode_to_prompt
    except ImportError as exc:
        raise ImportError(
            "dots_ocr prompt mapping not available. Install the dots-ocr extra."
        ) from exc

    prompt_map = dict(dict_promptmode_to_prompt)
    if not prompt_map:
        raise ValueError("dots_ocr prompt mapping is empty")
    return {str(key): str(value) for key, value in prompt_map.items()}


@register_provider("dots_ocr")
class DotsOCRProvider(OCRProvider):
    """Concrete shell for dots OCR discovery and routing tests."""

    default_model_id = "davanstrien/dots.ocr-1.5"
    default_prompt = ""
    default_svg_model_id = DEFAULT_SVG_MODEL_ID

    def _resolve_prompt_mode_and_prompt(self) -> tuple[str, str]:
        prompt_map = _load_upstream_prompt_mapping()
        prompt_mode = str(
            self._get_model_config("prompt_mode", DEFAULT_PROMPT_MODE)
            or DEFAULT_PROMPT_MODE
        ).strip()
        if prompt_mode not in prompt_map:
            raise ValueError(f"Unsupported dots_ocr prompt_mode: {prompt_mode}")

        prompt_text = prompt_map[prompt_mode]
        provider_prompt = str(self._get_model_config("prompt", "") or "").strip()
        global_prompt = str(
            self.ctx.config.get("prompts", {}).get("dots_ocr_prompt", "") or ""
        ).strip()
        if provider_prompt:
            prompt_text = provider_prompt
        elif global_prompt:
            prompt_text = global_prompt
        return prompt_mode, prompt_text

    def _select_model_id(self, prompt_mode: str) -> str:
        if prompt_mode == "prompt_image_to_svg":
            return str(
                self._get_model_config("svg_model_id", self.default_svg_model_id)
                or self.default_svg_model_id
            )
        return str(
            self._get_model_config("model_id", self.default_model_id)
            or self.default_model_id
        )

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        raise NotImplementedError("dots_ocr implementation lands in later tasks")
