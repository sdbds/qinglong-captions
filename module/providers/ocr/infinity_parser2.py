"""Infinity Parser2 OCR provider."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from PIL import Image
from rich.console import Console
from rich.progress import Progress
from rich_pixels import Pixels

from module.providers.base import CaptionResult, MediaContext, MediaModality, PromptContext
from module.providers.ocr_base import OCRProvider
from module.providers.registry import register_provider
from utils.console_util import print_exception
from utils.output_writer import write_markdown_output
from utils.parse_display import display_markdown
from utils.stream_util import iter_pdf_pages_high_quality
from utils.transformer_loader import resolve_device_dtype, transformerLoader

DEFAULT_MODEL_ID = "infly/Infinity-Parser2-Flash"
DEFAULT_DOC2MD_PROMPT = "Please transform the document's contents into Markdown format."
DEFAULT_MAX_NEW_TOKENS = 32768
DEFAULT_MIN_PIXELS = 2048
DEFAULT_MAX_PIXELS = 16777216
DEFAULT_IMAGE_PATCH_SIZE = 16
SUPPORTED_TASK_TYPES = {"doc2md", "custom"}

_TRANS_LOADER: Optional[transformerLoader] = None
_FENCED_MARKDOWN_RE = re.compile(r"^\s*```(?:markdown|md)?\s*\n(?P<body>.*)\n```\s*$", re.DOTALL | re.IGNORECASE)


def _strip_fenced_markdown(text: str) -> str:
    cleaned = str(text or "").strip()
    match = _FENCED_MARKDOWN_RE.match(cleaned)
    if match:
        return match.group("body").strip()
    return cleaned


def _validate_task_type(task_type: object) -> str:
    normalized = str(task_type or "doc2md").strip().lower()
    if normalized not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"Unsupported infinity_parser2_ocr task_type: {normalized}")
    return normalized


def _resolve_prompt(*, provider_section: dict[str, Any], prompts: dict[str, Any], default_prompt: str) -> str:
    task_type = _validate_task_type(provider_section.get("task_type", "doc2md"))
    provider_prompt = str(provider_section.get("prompt", "") or "").strip()
    prompt_config = str(prompts.get("infinity_parser2_ocr_prompt", "") or "").strip()
    override_prompt = provider_prompt or prompt_config

    if task_type == "custom":
        if not override_prompt:
            raise ValueError("custom task requires a non-empty prompt")
        return override_prompt

    return override_prompt or default_prompt


def _build_messages(
    pil_image: Image.Image,
    prompt_text: str,
    *,
    min_pixels: int,
    max_pixels: int,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pil_image,
                    "min_pixels": int(min_pixels),
                    "max_pixels": int(max_pixels),
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def _model_device(model: Any) -> Any:
    try:
        return model.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except Exception:
        return "cpu"


def _move_inputs_to_device(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to"):
        try:
            return inputs.to(device)
        except TypeError:
            pass

    if not isinstance(inputs, dict):
        return inputs

    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _input_ids_from_inputs(inputs: Any) -> Any:
    if isinstance(inputs, dict):
        return inputs.get("input_ids")
    return getattr(inputs, "input_ids", None)


@torch.inference_mode()
def _infer_single_image(
    *,
    pil_image: Image.Image,
    prompt_text: str,
    processor: Any,
    model: Any,
    process_vision_info_fn: Callable[..., Any],
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    image_patch_size: int = DEFAULT_IMAGE_PATCH_SIZE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    messages = _build_messages(
        pil_image,
        prompt_text,
        min_pixels=int(min_pixels),
        max_pixels=int(max_pixels),
    )
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    image_inputs, _ = process_vision_info_fn(messages, image_patch_size=int(image_patch_size))
    inputs = processor(
        text=text,
        images=image_inputs,
        do_resize=False,
        padding=True,
        return_tensors="pt",
    )
    inputs = _move_inputs_to_device(inputs, _model_device(model))

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        do_sample=False,
    )
    input_ids = _input_ids_from_inputs(inputs)
    try:
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
    except Exception:
        generated_ids_trimmed = [generated_ids[0]] if hasattr(generated_ids, "__getitem__") else generated_ids

    decoded = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    output = decoded[0] if decoded else ""
    cleaned = _strip_fenced_markdown(output)
    if not cleaned:
        raise ValueError("Infinity Parser2 OCR returned empty output")
    return cleaned


def _load_infinity_parser2_runtime(
    *,
    model_id: str,
    console: Console,
) -> tuple[Any, Any, Callable[..., Any]]:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise ImportError("Infinity Parser2 OCR requires qwen-vl-utils. Install the infinity-parser2-ocr extra.") from exc

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise ImportError("Infinity Parser2 OCR requires transformers. Install the infinity-parser2-ocr extra.") from exc

    _, dtype, attn_impl = resolve_device_dtype()
    global _TRANS_LOADER
    if _TRANS_LOADER is None:
        _TRANS_LOADER = transformerLoader(
            attn_kw="attn_implementation",
            device_map="auto",
        )

    processor = _TRANS_LOADER.get_or_load_processor(model_id, AutoProcessor, console=console)
    model = _TRANS_LOADER.get_or_load_model(
        model_id,
        AutoModelForImageTextToText,
        dtype=dtype,
        attn_impl=attn_impl,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_safetensors=True,
        console=console,
    )
    return processor, model, process_vision_info


@torch.inference_mode()
def attempt_infinity_parser2_ocr(
    *,
    uri: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    model_id: str = DEFAULT_MODEL_ID,
    prompt_text: str = DEFAULT_DOC2MD_PROMPT,
    pixels: Optional[Pixels] = None,
    output_dir: Optional[str] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    image_patch_size: int = DEFAULT_IMAGE_PATCH_SIZE,
    temperature: float = 0.0,
    top_p: float = 1.0,
    processor: Optional[Any] = None,
    model: Optional[Any] = None,
    process_vision_info_fn: Optional[Callable[..., Any]] = None,
) -> str:
    start_time = time.time()
    source_path = Path(uri)
    if not output_dir:
        output_dir = str(source_path.with_suffix(""))

    if processor is None or model is None or process_vision_info_fn is None:
        processor, model, process_vision_info_fn = _load_infinity_parser2_runtime(
            model_id=model_id,
            console=console,
        )

    def infer_image(pil_img: Image.Image) -> str:
        return _infer_single_image(
            pil_image=pil_img.convert("RGB"),
            prompt_text=prompt_text,
            processor=processor,
            model=model,
            process_vision_info_fn=process_vision_info_fn,
            min_pixels=int(min_pixels),
            max_pixels=int(max_pixels),
            image_patch_size=int(image_patch_size),
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
        )

    if source_path.suffix.lower() == ".pdf":
        page_outputs: list[str] = []
        for rendered_page in iter_pdf_pages_high_quality(str(source_path)):
            page_index = rendered_page.page_number
            pil_img = rendered_page.image
            try:
                page_dir = Path(output_dir) / f"page_{page_index:04d}"
                page_img_path = page_dir / f"page_{page_index:04d}.png"
                page_dir.mkdir(parents=True, exist_ok=True)
                try:
                    pil_img.save(page_img_path)
                except Exception:
                    try:
                        pil_img.convert("RGB").save(page_img_path)
                    except Exception as exc:
                        print_exception(
                            console,
                            exc,
                            prefix=f"Infinity Parser2 OCR page {page_index} image save failed",
                            summary_style="yellow",
                        )
                        continue

                try:
                    page_content = infer_image(pil_img)
                except Exception as exc:
                    print_exception(console, exc, prefix=f"Infinity Parser2 OCR page {page_index} failed", summary_style="yellow")
                    continue

                if page_content.strip():
                    write_markdown_output(page_dir, page_content)
                    page_outputs.append(page_content.strip())
            finally:
                pil_img.close()

        content = "\n<--- Page Split --->\n".join(page_outputs).strip()
        if not content:
            raise RuntimeError("Infinity Parser2 OCR failed for all PDF pages")
        write_markdown_output(Path(output_dir), content)
    else:
        with Image.open(str(source_path)) as opened:
            content = infer_image(opened)
        write_markdown_output(Path(output_dir), content)

    try:
        display_markdown(
            title=source_path.name,
            markdown_content=content,
            pixels=pixels,
            panel_height=32,
            console=console,
        )
    except Exception:
        pass

    elapsed = time.time() - start_time
    if progress and task_id is not None:
        progress.update(task_id, description="Generating captions")
    console.print(f"[blue]Caption generation took:[/blue] {elapsed:.2f} seconds")
    return content


@register_provider("infinity_parser2_ocr")
class InfinityParser2OCRProvider(OCRProvider):
    """Local OCR provider for Infinity Parser2."""

    default_model_id = DEFAULT_MODEL_ID
    default_prompt = DEFAULT_DOC2MD_PROMPT

    def prepare_media(self, uri: str, mime: str, args: Any) -> MediaContext:
        return MediaContext(
            uri=uri,
            mime=mime,
            sha256hash="",
            modality=MediaModality.DOCUMENT if mime.startswith("application") else MediaModality.IMAGE,
            extras={"output_dir": Path(uri).with_suffix("")},
        )

    def get_prompts(self, mime: str):
        prompt = _resolve_prompt(
            provider_section=self.ctx.config.get(self.name, {}),
            prompts=self.ctx.config.get("prompts", {}),
            default_prompt=self.default_prompt,
        )
        return "", prompt

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        if self.get_runtime_backend().is_openai:
            raise NotImplementedError("infinity_parser2_ocr does not support the OpenAI-compatible runtime backend yet")

        output_dir = media.extras.get("output_dir")
        model_id = self._get_model_config("model_id", self.default_model_id)
        result = attempt_infinity_parser2_ocr(
            uri=media.uri,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
            model_id=model_id,
            prompt_text=prompts.user,
            pixels=media.pixels,
            output_dir=str(output_dir) if output_dir else None,
            max_new_tokens=self._get_model_config("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
            min_pixels=self._get_model_config("min_pixels", DEFAULT_MIN_PIXELS),
            max_pixels=self._get_model_config("max_pixels", DEFAULT_MAX_PIXELS),
            image_patch_size=self._get_model_config("image_patch_size", DEFAULT_IMAGE_PATCH_SIZE),
            temperature=self._get_model_config("temperature", 0.0),
            top_p=self._get_model_config("top_p", 1.0),
        )
        return CaptionResult(
            raw=result if isinstance(result, str) else str(result),
            metadata={
                "provider": self.name,
                "model_id": model_id,
                "output_dir": str(output_dir),
            },
        )
