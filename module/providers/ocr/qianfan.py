"""Qianfan OCR Provider."""

from __future__ import annotations

import re
import sys
import time
from contextlib import contextmanager
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
_MISSING = object()
_IMAGE_PLACEHOLDER_RE = re.compile(
    r"!\[(?P<label>[^\]]*)\]\(\s*(?:<box>)?\s*\[\[(?P<coords>[^\]]+)\]\]\s*(?:</box>)?\s*\)"
)
_COORD_TOKEN_RE = re.compile(r"<COORD_(\d+)>|(\d+)")
_LOCAL_RENDERED_IMAGE_RE = re.compile(r"!\[(?P<label>[^\]]*)\]\((?P<path>images/[^)]+)\)")


@contextmanager
def _suppress_broken_wandb_import():
    previous = sys.modules.get("wandb", _MISSING)
    sys.modules["wandb"] = None
    try:
        yield
    finally:
        if previous is _MISSING:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = previous


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


def _resolve_vision_device(model: Any):
    vision_model = getattr(model, "vision_model", None)
    if vision_model is not None:
        try:
            return next(vision_model.parameters()).device
        except Exception:
            pass

    try:
        return next(model.parameters()).device
    except Exception:
        pass

    return getattr(model, "device", None)


def _parse_normalized_coords(raw_coords: str) -> tuple[int, int, int, int] | None:
    values = [int(coord_a or coord_b) for coord_a, coord_b in _COORD_TOKEN_RE.findall(raw_coords or "")]
    if len(values) != 4:
        return None
    return values[0], values[1], values[2], values[3]


def _normalized_coords_to_pixels(
    coords: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = coords
    left = max(0, min(width, int(x1 / 1000 * width)))
    top = max(0, min(height, int(y1 / 1000 * height)))
    right = max(0, min(width, int(x2 / 1000 * width)))
    bottom = max(0, min(height, int(y2 / 1000 * height)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _render_placeholder_images(
    markdown: str,
    *,
    source_image_path: Path,
    asset_dir: Path,
    asset_prefix: str,
) -> str:
    if not markdown or not _IMAGE_PLACEHOLDER_RE.search(markdown):
        return markdown

    with Image.open(source_image_path) as source:
        page_image = source.convert("RGB")
    width, height = page_image.size
    asset_dir.mkdir(parents=True, exist_ok=True)
    crop_index = 0

    def replace_placeholder(match: re.Match[str]) -> str:
        nonlocal crop_index
        coords = _parse_normalized_coords(match.group("coords"))
        if coords is None:
            return match.group(0)
        pixel_box = _normalized_coords_to_pixels(coords, width=width, height=height)
        if pixel_box is None:
            return match.group(0)

        crop_index += 1
        asset_name = f"{asset_prefix}-{crop_index:03d}.png"
        asset_path = asset_dir / asset_name
        page_image.crop(pixel_box).save(asset_path)
        rendered_path = asset_path.relative_to(asset_dir.parent).as_posix()
        label = match.group("label") or "image"
        return f"![{label}]({rendered_path})"

    return _IMAGE_PLACEHOLDER_RE.sub(replace_placeholder, markdown)


def _prefix_rendered_image_paths(markdown: str, page_dir_name: str) -> str:
    if not markdown:
        return markdown

    def replace_image_path(match: re.Match[str]) -> str:
        label = match.group("label") or "image"
        path = match.group("path")
        return f"![{label}]({Path(page_dir_name, path).as_posix()})"

    return _LOCAL_RENDERED_IMAGE_RE.sub(replace_image_path, markdown)


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

        with _suppress_broken_wandb_import():
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
        pixel_device = _resolve_vision_device(model)
        pixel_values = _load_image_tensor(image_path, input_size=input_size, max_num=max_num).to(
            device=pixel_device,
            dtype=dtype,
        )
        with torch.no_grad():
            response = model.chat(
                tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config={"max_new_tokens": int(max_new_tokens)},
            )
        return response if isinstance(response, str) else str(response)

    def _write_clean_markdown(
        self,
        output_dir: Path,
        raw_text: str,
        *,
        source_image_path: Path,
        asset_prefix: str,
    ) -> str:
        cleaned = self._clean_reasoning_output(raw_text)
        rendered = _render_placeholder_images(
            cleaned,
            source_image_path=source_image_path,
            asset_dir=output_dir / "images",
            asset_prefix=asset_prefix,
        )
        write_markdown_output(output_dir, rendered)
        return rendered

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
                cleaned_page = self._write_clean_markdown(
                    page_dir,
                    raw_page,
                    source_image_path=page_img_path,
                    asset_prefix=f"{page_dir.name}-image",
                )
                page_contents.append(_prefix_rendered_image_paths(cleaned_page.strip(), page_dir.name))

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
            content = self._write_clean_markdown(
                output_dir,
                raw_output,
                source_image_path=Path(media.uri),
                asset_prefix="image",
            )

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
