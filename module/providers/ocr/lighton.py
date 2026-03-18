"""LightOn OCR Provider"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from rich.console import Console
from rich.progress import Progress
from rich_pixels import Pixels

from module.providers.base import CaptionResult, MediaContext, PromptContext
from module.providers.ocr_base import OCRProvider
from module.providers.registry import register_provider
from utils.parse_display import display_markdown
from utils.output_writer import write_markdown_output
from utils.stream_util import pdf_to_images_high_quality
from utils.transformer_loader import resolve_device_dtype, transformerLoader

_TRANS_LOADER: Optional[transformerLoader] = None


def _build_messages(image_path: str, prompt_text: Optional[str]) -> list[dict[str, Any]]:
    content: list[dict[str, str]] = [{"type": "image", "url": image_path}]
    if prompt_text and prompt_text.strip():
        content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


def _move_inputs_to_device(inputs: Any, *, device: torch.device, dtype: torch.dtype) -> Any:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue
        if torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


@torch.inference_mode()
def attempt_lighton_ocr(
    *,
    uri: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    model_id: str = "lightonai/LightOnOCR-2-1B",
    prompt_text: Optional[str] = None,
    pixels: Optional[Pixels] = None,
    output_dir: Optional[str] = None,
    max_new_tokens: int = 4096,
) -> str:
    """Run local LightOnOCR-2 inference on a single image or PDF."""
    start_time = time.time()

    try:
        from transformers import AutoProcessor, LightOnOcrForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "LightOnOcrForConditionalGeneration not available. "
            "Install the lighton-ocr extra or transformers[serving]>=5.0.0."
        ) from exc

    p = Path(uri)
    if not output_dir:
        output_dir = str(p.with_suffix(""))

    _, dtype, attn_impl = resolve_device_dtype()
    global _TRANS_LOADER
    if _TRANS_LOADER is None:
        _TRANS_LOADER = transformerLoader(attn_kw="attn_implementation", device_map="auto")

    processor = _TRANS_LOADER.get_or_load_processor(model_id, AutoProcessor, console=console)
    model = _TRANS_LOADER.get_or_load_model(
        model_id,
        LightOnOcrForConditionalGeneration,
        dtype=dtype,
        attn_impl=attn_impl,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_safetensors=True,
        console=console,
    )

    def _infer_single_image(image_path: str, pil_image: Image.Image) -> str:
        messages = _build_messages(image_path, prompt_text)
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        try:
            model_device = model.device
        except Exception:
            model_device = next(model.parameters()).device

        prepared_inputs = _move_inputs_to_device(inputs, device=model_device, dtype=dtype)
        output_ids = model.generate(
            **prepared_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        generated_ids = output_ids[:, prepared_inputs["input_ids"].shape[1]:]
        return processor.decode(generated_ids[0], skip_special_tokens=True)

    if p.suffix.lower() == ".pdf":
        images = pdf_to_images_high_quality(str(p))
        all_contents = []

        for idx, pil_img in enumerate(images):
            page_dir = Path(output_dir) / f"page_{idx + 1:04d}"
            page_dir.mkdir(parents=True, exist_ok=True)
            page_img_path = page_dir / f"page_{idx + 1:04d}.png"

            try:
                pil_img.save(page_img_path)
            except Exception:
                try:
                    pil_img.convert("RGB").save(page_img_path)
                except Exception:
                    continue

            page_content = _infer_single_image(str(page_img_path), pil_img)

            try:
                write_markdown_output(page_dir, page_content)
            except Exception:
                pass

            all_contents.append(page_content.strip())

        content = "\n<--- Page Split --->\n".join(all_contents)

        try:
            write_markdown_output(Path(output_dir), content)
        except Exception:
            pass

        try:
            display_markdown(
                title=p.name,
                markdown_content=content,
                pixels=pixels,
                panel_height=32,
                console=console,
            )
        except Exception:
            pass
    else:
        pil_img = Image.open(str(p)).convert("RGB")
        content = _infer_single_image(str(p), pil_img)

        try:
            write_markdown_output(Path(output_dir), content)
        except Exception:
            pass

        try:
            display_markdown(
                title=p.name,
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


@register_provider("lighton_ocr")
class LightOnOCRProvider(OCRProvider):
    """LightOn OCR Provider."""

    default_model_id = "lightonai/LightOnOCR-2-1B"
    default_prompt = ""

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        if self.get_runtime_backend().is_openai:
            return self.attempt_via_openai_backend(media, prompts)

        output_dir = media.extras.get("output_dir")

        result = attempt_lighton_ocr(
            uri=media.uri,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
            model_id=self._get_model_config("model_id", self.default_model_id),
            prompt_text=prompts.user,
            pixels=media.pixels,
            output_dir=str(output_dir) if output_dir else None,
            max_new_tokens=self._get_model_config("max_new_tokens", 4096),
        )

        return CaptionResult(
            raw=result if isinstance(result, str) else str(result),
            metadata={"provider": self.name, "output_dir": str(output_dir)},
        )
