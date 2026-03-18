"""Qianfan OCR Provider."""

from __future__ import annotations

import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.ocr_base import OCRProvider
from providers.registry import register_provider
from utils.output_writer import write_markdown_output
from utils.parse_display import display_markdown
from utils.stream_util import pdf_to_images_high_quality
from utils.transformer_loader import resolve_device_dtype, transformerLoader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_TRANS_LOADER: Optional[transformerLoader] = None


def _build_transform(input_size: int):
    try:
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
    except ImportError as exc:
        raise ImportError("Qianfan OCR image preprocessing requires torchvision. Install the qianfan-ocr extra.") from exc

    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def _dynamic_preprocess(
    image: Image.Image,
    *,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda item: item[0] * item[1],
    )
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    tiles_per_row = max(1, target_width // image_size)

    processed_images: list[Image.Image] = []
    for idx in range(blocks):
        box = (
            (idx % tiles_per_row) * image_size,
            (idx // tiles_per_row) * image_size,
            ((idx % tiles_per_row) + 1) * image_size,
            ((idx // tiles_per_row) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def _load_image_tensor(image_file: str, *, input_size: int = 448, max_num: int = 12):
    with Image.open(image_file) as source:
        image = source.convert("RGB")

    transform = _build_transform(input_size=input_size)
    processed_images = _dynamic_preprocess(
        image,
        image_size=input_size,
        max_num=max_num,
        use_thumbnail=True,
    )
    pixel_values = [transform(frame) for frame in processed_images]
    return torch.stack(pixel_values)


@register_provider("qianfan_ocr")
class QianfanOCRProvider(OCRProvider):
    """Qianfan OCR Provider."""

    default_model_id = "baidu/Qianfan-OCR"
    default_prompt = "Parse this document to Markdown."
    _THINK_SUFFIX = "<think>"
    _THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

    def _provider_config(self) -> dict[str, Any]:
        cfg = self.ctx.config.get(self.name, {})
        return cfg if isinstance(cfg, dict) else {}

    def _compose_question(self) -> str:
        provider_cfg = self._provider_config()
        custom_prompt = str(provider_cfg.get("prompt", "") or "").strip()
        prompt_strategy = str(provider_cfg.get("prompt_strategy", "append") or "append").strip().lower()
        think_enabled = bool(provider_cfg.get("think_enabled", True))

        if prompt_strategy == "replace":
            question = custom_prompt or self.default_prompt
        elif prompt_strategy == "append":
            question = self.default_prompt
            if custom_prompt:
                question = f"{question}\n{custom_prompt}"
        else:
            raise ValueError(f"Unsupported qianfan_ocr prompt_strategy: {prompt_strategy}")

        question = question.strip()
        if think_enabled and not question.endswith(self._THINK_SUFFIX):
            question = f"{question}{self._THINK_SUFFIX}"
        return question

    def _clean_reasoning_output(self, text: str) -> str:
        cleaned = self._THINK_BLOCK_RE.sub("", text or "").strip()
        if not cleaned:
            raise ValueError("Qianfan OCR returned no markdown after stripping reasoning output")
        return cleaned

    def get_prompts(self, mime: str):
        return "", self._compose_question()

    def post_validate(self, result: CaptionResult, media: MediaContext, args: Any) -> CaptionResult:
        if "<think>" not in result.raw.lower():
            return result
        return replace(result, raw=self._clean_reasoning_output(result.raw))

    def _run_direct_generation(
        self,
        *,
        image_path: str,
        question: str,
        model_id: str,
        max_new_tokens: int,
        input_size: int = 448,
        max_num: int = 12,
    ) -> str:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Qianfan OCR requires transformers with trust_remote_code support. Install the qianfan-ocr extra.") from exc

        _, dtype, attn_impl = resolve_device_dtype()
        global _TRANS_LOADER
        if _TRANS_LOADER is None:
            _TRANS_LOADER = transformerLoader(attn_kw="_attn_implementation", device_map="auto")

        tokenizer = _TRANS_LOADER.get_or_load_processor(model_id, AutoTokenizer, console=self.ctx.console)
        model = _TRANS_LOADER.get_or_load_model(
            model_id,
            AutoModel,
            dtype=dtype,
            attn_impl=attn_impl,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_safetensors=True,
            console=self.ctx.console,
        )
        pixel_values = _load_image_tensor(image_path, input_size=input_size, max_num=max_num).to(dtype=dtype)
        with torch.no_grad():
            response = model.chat(
                tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config={"max_new_tokens": int(max_new_tokens)},
            )
        return response if isinstance(response, str) else str(response)

    def _write_clean_markdown(self, output_dir: Path, raw_text: str) -> str:
        cleaned = self._clean_reasoning_output(raw_text)
        write_markdown_output(output_dir, cleaned)
        return cleaned

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        runtime = self.get_runtime_backend()
        if runtime.is_openai:
            raise NotImplementedError("qianfan_ocr does not support the OpenAI-compatible runtime backend yet")

        start_time = time.time()
        output_dir = Path(media.extras.get("output_dir"))
        model_id = self._get_model_config("model_id", self.default_model_id)
        max_new_tokens = int(self._get_model_config("max_new_tokens", 16384))
        input_size = int(self._get_model_config("input_size", 448))
        max_num = int(self._get_model_config("max_num", 12))

        if media.mime.startswith("application/pdf"):
            page_contents: list[str] = []
            for idx, pil_img in enumerate(pdf_to_images_high_quality(media.uri), start=1):
                page_dir = output_dir / f"page_{idx:04d}"
                page_dir.mkdir(parents=True, exist_ok=True)
                page_img_path = page_dir / f"page_{idx:04d}.png"
                try:
                    pil_img.save(page_img_path)
                except Exception:
                    try:
                        pil_img.convert("RGB").save(page_img_path)
                    except Exception:
                        continue

                raw_page = self._run_direct_generation(
                    image_path=str(page_img_path),
                    question=prompts.user,
                    model_id=model_id,
                    max_new_tokens=max_new_tokens,
                    input_size=input_size,
                    max_num=max_num,
                )
                cleaned_page = self._write_clean_markdown(page_dir, raw_page)
                page_contents.append(cleaned_page.strip())

            content = "\n<--- Page Split --->\n".join(page_contents)
            write_markdown_output(output_dir, content)
        else:
            raw_output = self._run_direct_generation(
                image_path=media.uri,
                question=prompts.user,
                model_id=model_id,
                max_new_tokens=max_new_tokens,
                input_size=input_size,
                max_num=max_num,
            )
            content = self._write_clean_markdown(output_dir, raw_output)

        try:
            display_markdown(
                title=Path(media.uri).name,
                markdown_content=content,
                pixels=media.pixels,
                panel_height=32,
                console=self.ctx.console,
            )
        except Exception:
            pass

        elapsed = time.time() - start_time
        self.ctx.console.print(f"[blue]Caption generation took:[/blue] {elapsed:.2f} seconds")

        return CaptionResult(raw=content, metadata={"provider": self.name, "output_dir": str(output_dir)})
