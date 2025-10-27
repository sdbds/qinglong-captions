from __future__ import annotations

# All logs and comments are in English.
import time
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from rich.console import Console
from rich.progress import Progress
from rich_pixels import Pixels

from utils.parse_display import display_markdown
from utils.stream_util import pdf_to_images_high_quality


# Global lazy cache for model and processor
_OLM_MODEL: Optional[Any] = None
_OLM_PROCESSOR: Optional[Any] = None
_OLM_DEVICE: Optional[torch.device] = None


def _resolve_device_dtype() -> tuple[torch.device | str, torch.dtype, str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        return "cuda", torch.bfloat16, "flash_attention_2"
    # CPU fallback
    return "cpu", torch.float32, "eager"


def _get_model_and_processor(
    model_id: str,
    processor_id: str,
    console: Optional[Console] = None,
) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor, torch.device]:
    global _OLM_MODEL, _OLM_PROCESSOR, _OLM_DEVICE
    if _OLM_MODEL is not None and _OLM_PROCESSOR is not None and _OLM_DEVICE is not None:
        return _OLM_MODEL, _OLM_PROCESSOR, _OLM_DEVICE

    device, dtype, attn_impl = _resolve_device_dtype()
    if console:
        console.print(f"[green]Loading OLM OCR model:[/green] {model_id} on {device} ({dtype})")
        console.print(f"[green]Loading OLM OCR processor:[/green] {processor_id}")

    processor = AutoProcessor.from_pretrained(processor_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model = model.eval()

    _OLM_MODEL, _OLM_PROCESSOR, _OLM_DEVICE = model, processor, device
    return model, processor, device


def _generate_for_image(
    *,
    base64_image: str,
    prompt_text: str,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    device: torch.device,
    temperature: float,
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Convert base64 back to PIL image for processor
    import base64
    import io
    from PIL import Image
    
    image_data = base64.b64decode(base64_image)
    pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    inputs = processor(
        text=[text],
        images=[pil_image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for (k, v) in inputs.items()}

    output = model.generate(
        **inputs,
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
        num_return_sequences=1,
        do_sample=True,
    )

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_len:]
    text_output = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    # batch of size 1
    return text_output[0] if text_output else ""


@torch.inference_mode()
def attempt_olmocr(
    *,
    uri: str,
    mime: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    prompt_text: Optional[str] = None,
    pixels: Optional[Pixels] = None,
    output_dir: Optional[str] = None,
    base64_image: Optional[str] = None,
    model_id: str = "allenai/olmOCR-2-7B-1025",
    processor_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    temperature: float = 0.1,
    max_new_tokens: int = 512,
) -> str:
    """Run local OLM OCR on an image or PDF and return text content.

    Mirrors the DeepSeek-OCR provider flow: per-page processing for PDFs,
    sidecar result.md writes, and markdown preview in the console UI.
    """
    # Optional prompt builder from OLMOCR package
    try:
        from olmocr.prompts import build_no_anchoring_v4_yaml_prompt  # type: ignore
    except Exception:
        build_no_anchoring_v4_yaml_prompt = None  # type: ignore

    start_time = time.time()

    # Default prompt
    if not prompt_text:
        if build_no_anchoring_v4_yaml_prompt is not None:
            try:
                prompt_text = build_no_anchoring_v4_yaml_prompt()
            except Exception:
                prompt_text = "Extract structured text from the document image."
        else:
            prompt_text = "Extract structured text from the document image."

    if not output_dir:
        output_dir = uri.with_suffix("")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model, processor, device = _get_model_and_processor(model_id, processor_id, console)

    if mime.startswith("application/pdf"):
        images = pdf_to_images_high_quality(uri)
        all_contents: list[str] = []
        for idx, pil_img in enumerate(images):
            page_dir = Path(output_dir) / f"page_{idx+1:04d}"
            page_dir.mkdir(parents=True, exist_ok=True)
            page_img_path = page_dir / f"page_{idx+1:04d}.png"
            try:
                pil_img.save(page_img_path)
            except Exception:
                try:
                    pil_img.convert("RGB").save(page_img_path)
                except Exception:
                    continue

            try:
                # Use provided base64_image if available, otherwise convert from PIL image
                page_base64 = None
                if base64_image and base64_image.strip():
                    page_base64 = base64_image.strip()
                else:
                    import base64
                    import io
                    
                    buffer = io.BytesIO()
                    pil_img.convert("RGB").save(buffer, format="PNG")
                    page_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                page_content = _generate_for_image(
                    base64_image=page_base64,
                    prompt_text=str(prompt_text),
                    model=model,
                    processor=processor,
                    device=device,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as e:
                console.print(f"[yellow]OLM OCR page {idx+1} failed: {e}[/yellow]")
                page_content = ""

            try:
                (page_dir / "result.md").write_text(page_content, encoding="utf-8")
            except Exception:
                pass

            all_contents.append(page_content.strip())

        content = "\n<--- Page Split --->\n".join(all_contents)
        try:
            (Path(output_dir) / "result.md").write_text(content, encoding="utf-8")
        except Exception:
            pass

        try:
            display_markdown(
                title=uri.name,
                markdown_content=content,
                pixels=pixels,
                panel_height=32,
                console=console,
            )
        except Exception:
            pass

    else:
        # Use provided base64_image directly
        if not base64_image or not base64_image.strip():
            console.print(f"[red]OLM OCR requires base64 image data: {uri}[/red]")
            return ""

        final_base64 = base64_image.strip()

        content = _generate_for_image(
            base64_image=final_base64,
            prompt_text=str(prompt_text),
            model=model,
            processor=processor,
            device=device,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        try:
            (Path(output_dir) / "result.md").write_text(content, encoding="utf-8")
        except Exception:
            pass

        try:
            display_markdown(
                title=uri.name,
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
