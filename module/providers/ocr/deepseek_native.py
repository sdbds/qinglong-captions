"""DeepSeek-OCR-2 using the built-in Transformers v5 model, without repository code."""

from __future__ import annotations

import json
import re
import stat
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from rich.console import Console
from rich.progress import Progress
from rich_pixels import Pixels

from module.providers.backends import resolve_runtime_backend
from module.providers.base import CaptionResult, MediaContext, PromptContext
from module.providers.ocr_base import OCRProvider
from module.providers.registry import register_provider
from utils.ocr_artifacts import reserve_ocr_artifact_dir
from utils.ocr_model_config import DEEPSEEK_NATIVE_MODEL
from utils.parse_display import display_markdown
from utils.stream_util import iter_pdf_pages_high_quality
from utils.transformer_loader import load_pretrained_component, resolve_device_dtype, resolve_pretrained_device_map

from .document import ensure_complete_generation, generation_eos_ids, rebase_generated_images, write_layout_json

DEFAULT_MODEL_ID = DEEPSEEK_NATIVE_MODEL
MODEL_REVISION = "838dc37567aa97106fcd590e2be6a8f279881813"
_DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
_GROUNDING = re.compile(r"<\|ref\|>([^<>\n]+)<\|/ref\|>\s*<\|det\|>([^<>]*)<\|/det\|>")


@lru_cache(maxsize=1)
def _load_runtime(model_id: str, device: str, dtype: torch.dtype, attention: str):
    if model_id == DEFAULT_MODEL_ID:
        if Path(model_id).is_dir():
            raise ValueError(f"A local directory shadows the pinned Hub model {model_id}; use an absolute path for local weights")
        source = {"revision": MODEL_REVISION}
    else:
        path = Path(model_id).expanduser()
        if not (path / "config.json").is_file() or (
            json.loads((path / "config.json").read_text(encoding="utf-8")).get("model_type") != "deepseek_ocr2"
        ):
            raise ValueError(f"Native DeepSeek OCR requires {DEFAULT_MODEL_ID} or a local converted deepseek_ocr2 checkpoint")
        model_id = str(path.resolve())
        source = {"local_files_only": True}
    try:
        from transformers import AutoProcessor, DeepseekOcr2ForConditionalGeneration
    except ImportError as exc:
        raise ImportError("Install the deepseek-ocr-native extra for native DeepSeek-OCR-2 support") from exc

    processor = load_pretrained_component(AutoProcessor, model_id, trust_remote_code=False, **source)
    model = load_pretrained_component(
        DeepseekOcr2ForConditionalGeneration,
        model_id,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=dtype,
        device_map=resolve_pretrained_device_map(device, "auto") if device.startswith("cuda") else {"": device},
        attn_implementation=attention,
        **source,
    )
    return processor, model.eval()


def _generate_page(processor, model, image, prompt, max_new_tokens):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, model.dtype)
    prompt_length = inputs["input_ids"].shape[1]
    limit = model.config.text_config.max_position_embeddings - prompt_length
    if limit < 1:
        raise ValueError("DeepSeek OCR prompt and visual tokens exceed the model context")
    budget = min(max_new_tokens, limit)
    eos_ids = generation_eos_ids(model, processor)
    generated = model.generate(**inputs, do_sample=False, max_new_tokens=budget, eos_token_id=eos_ids)[0, prompt_length:]
    ensure_complete_generation(generated, max_new_tokens=budget, eos_token_ids=eos_ids)
    decode_kwargs = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    raw = processor.decode(generated, **decode_kwargs)
    # Raw includes the terminal EOS; remove only that token from the human-facing Markdown.
    prediction = processor.decode(generated[:-1], **decode_kwargs) if len(generated) and int(generated[-1]) in eos_ids else raw
    return raw, prediction


def _page_markdown(prediction: str, image: Image.Image, page_dir: Path, number: int) -> str:
    width, height = image.size
    regions = []
    matches = list(_GROUNDING.finditer(prediction))
    contents = [prediction[: matches[0].start()] if matches else prediction]
    image_count = 0
    for index, marker in enumerate(matches):
        label = marker[1].strip()
        boxes = json.loads(marker[2])
        if not isinstance(boxes, list) or not boxes:
            raise ValueError(f"Invalid DeepSeek grounding coordinates on page {number}: {boxes}")
        tail = prediction[marker.end() : matches[index + 1].start() if index + 1 < len(matches) else len(prediction)]
        links = []
        for box in boxes:
            if not isinstance(box, list) or len(box) != 4 or not all(type(v) is int for v in box):
                raise ValueError(f"Invalid DeepSeek grounding bbox on page {number}: {box}")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999):
                raise ValueError(f"Invalid DeepSeek grounding bbox on page {number}: {box}")
            # DeepSeek's reference cropper maps 999 to the full rendered width/height.
            xyxy = [x1 * width // 999, y1 * height // 999, x2 * width // 999, y2 * height // 999]
            region = {"label": label, "bbox": box, "bbox_pixels": xyxy, "text": tail.strip()}
            if label == "image":
                if xyxy[0] == xyxy[2] or xyxy[1] == xyxy[3]:
                    raise ValueError(f"Empty DeepSeek image crop on page {number}: {xyxy}")
                relative = f"images/{image_count}.png"
                path = page_dir / relative
                path.parent.mkdir(exist_ok=True)
                with image.crop(xyxy) as crop:
                    crop.save(path)
                region["crop"] = relative
                links.append(f"![image]({relative})")
                image_count += 1
            regions.append(region)
        if links:
            contents.append("\n\n" + "\n\n".join(links) + "\n\n")
        contents.append(tail)
    markdown = "".join(contents).strip()
    if re.search(r"<\|/?(?:ref|det)\|>", markdown):
        raise ValueError(f"Incomplete DeepSeek grounding markup on page {number}")
    write_layout_json(
        page_dir,
        {
            "schema_version": 1,
            "page_number": number,
            "width": width,
            "height": height,
            "coordinate_system": "normalized_0_999",
            "regions": regions,
        },
        filename="grounding.json",
    )
    return markdown


def _require_empty_output(path: Path) -> None:
    mode = path.lstat()
    if (
        not stat.S_ISDIR(mode.st_mode)
        or getattr(mode, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        or any(path.iterdir())
    ):
        raise FileExistsError(f"Native DeepSeek OCR requires a new or empty physical output directory: {path}")


@torch.inference_mode()
def attempt_deepseek_native(
    *,
    uri: str,
    console: Console,
    progress: Progress | None,
    task_id: Any | None,
    model_id: str = DEFAULT_MODEL_ID,
    prompt_text: str | None = None,
    pixels: Pixels | None = None,
    output_dir: str | None = None,
    supports_flex_attn: bool = False,
    base_size: int = 1024,
    image_size: int = 768,
    crop_mode: bool = True,
    max_new_tokens: int = 8192,
) -> str:
    """Publish page-local evidence and return Markdown relative to the artifact directory."""
    start = time.monotonic()
    if (base_size, image_size, crop_mode) != (1024, 768, True):
        raise ValueError("Native DeepSeek OCR uses the checkpoint geometry: base_size=1024, image_size=768, crop_mode=True")
    if max_new_tokens < 1:
        raise ValueError("DeepSeek OCR max_new_tokens must be positive")
    prompt = prompt_text if prompt_text and prompt_text.strip() else _DEFAULT_PROMPT
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt
    if prompt.count("<image>") != 1:
        raise ValueError("Pagewise DeepSeek OCR requires exactly one <image> placeholder")
    source = Path(uri)
    destination = Path(output_dir) if output_dir else reserve_ocr_artifact_dir(source)
    destination.mkdir(parents=True, exist_ok=True)
    _require_empty_output(destination)
    device, dtype, attention = resolve_device_dtype(supports_flex_attn=supports_flex_attn)
    # The native visual backbone does not implement FlashAttention 2.
    if attention == "flash_attention_2":
        attention = "sdpa"
    processor, model = _load_runtime(str(model_id), device, dtype, attention)
    is_pdf = source.suffix.lower() == ".pdf"
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        stage = Path(temp) / "document"
        stage.mkdir(mode=0o700)

        def process(image: Image.Image, number: int) -> str:
            page_dir = stage
            if is_pdf:
                page_dir = stage / f"page_{number:04d}"
                page_dir.mkdir()
                image.save(page_dir / f"{page_dir.name}.png")
            raw, prediction = _generate_page(processor, model, image, prompt, max_new_tokens)
            (page_dir / "raw.txt").write_text(raw, encoding="utf-8")
            markdown = _page_markdown(prediction, image, page_dir, number)
            (page_dir / "result.md").write_text(markdown, encoding="utf-8")
            return rebase_generated_images(markdown, page_dir, stage) if is_pdf else markdown

        if is_pdf:
            parts = []
            for page in iter_pdf_pages_high_quality(source):
                with page.image as image:
                    parts.append(process(image, page.page_number))
            if not parts:
                raise ValueError(f"PDF has no pages: {source}")
            content = "\n<--- Page Split --->\n".join(parts)
            (stage / "result.md").write_text(content, encoding="utf-8")
        else:
            with Image.open(source) as original, ImageOps.exif_transpose(original) as upright, upright.convert("RGB") as image:
                content = process(image, 1)

        # The outer OCR registry also reserves a fresh empty directory per attempt.
        # rmdir refuses any file added since preflight; never replace a prior run.
        _require_empty_output(destination)
        destination.rmdir()
        stage.rename(destination)

    if progress is not None and task_id is not None:
        progress.update(task_id, description="Generating captions")
    try:
        display_markdown(title=source.name, markdown_content=content, pixels=pixels, panel_height=32, console=console)
    except Exception as exc:
        console.print(f"Could not display OCR preview: {exc}", markup=False)
    console.print(f"[blue]Caption generation took:[/blue] {time.monotonic() - start:.2f} seconds")
    return content


@register_provider("deepseek_ocr_native")
class DeepSeekNativeOCRProvider(OCRProvider):
    """Opt-in local OCR using the native Transformers checkpoint format."""

    default_model_id = DEFAULT_MODEL_ID
    default_prompt = _DEFAULT_PROMPT

    def get_runtime_backend(self):
        return resolve_runtime_backend(
            self.ctx.args,
            self.ctx.config.get(self.name, {}),
            arg_prefix="local_runtime",
            shared_prefix="openai",
            default_model_id=self._get_model_config("model_id", self.default_model_id),
        )

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        if self.get_runtime_backend().is_openai:
            raise ValueError("deepseek_ocr_native requires the direct local runtime, not an OpenAI server")
        output_dir = self.artifact_output_dir(media)
        model_id = self._get_model_config("model_id", self.default_model_id)
        markdown = attempt_deepseek_native(
            uri=media.uri,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
            model_id=model_id,
            prompt_text=prompts.user,
            pixels=media.pixels,
            output_dir=str(output_dir),
            supports_flex_attn=bool(getattr(self, "_supports_flex_attn", False)),
            base_size=getattr(self.ctx.args, "deepseek_base_size", self._get_model_config("base_size", 1024)),
            image_size=getattr(self.ctx.args, "deepseek_image_size", self._get_model_config("image_size", 768)),
            crop_mode=getattr(self.ctx.args, "deepseek_crop_mode", self._get_model_config("crop_mode", True)),
            max_new_tokens=self._get_model_config("max_new_tokens", 8192),
        )
        return CaptionResult(
            raw=rebase_generated_images(markdown, output_dir, Path(media.uri).parent),
            metadata={"provider": self.name, "model_id": model_id, "output_dir": str(output_dir), "ocr_engine": "native"},
        )
