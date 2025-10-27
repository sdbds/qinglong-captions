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
from utils.transformer_loader import transformerLoader, resolve_device_dtype


# Global lazy cache for model and processor
_TRANS_LOADER: Optional[transformerLoader] = None




def _generate_for_image(
    *,
    base64_image: str,
    prompt_text: str,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    device: str | torch.device,
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
    global _TRANS_LOADER
    if _TRANS_LOADER is None:
        _TRANS_LOADER = transformerLoader(attn_kw="attn_implementation", device_map="auto")
    inputs = _TRANS_LOADER.prepare_image_inputs(
        processor,
        messages,
        base64_image=base64_image,
        pil_image=None,
        device=device,
        tokenize=False,
        add_generation_prompt=True,
        padding=True,
        return_tensors="pt",
    )

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
        output_dir = str(Path(uri).with_suffix(""))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device, dtype, attn_impl = resolve_device_dtype()
    global _TRANS_LOADER
    if _TRANS_LOADER is None:
        _TRANS_LOADER = transformerLoader(attn_kw="attn_implementation", device_map="auto")

    processor = _TRANS_LOADER.get_or_load_processor(processor_id, AutoProcessor, console=console)
    model = _TRANS_LOADER.get_or_load_model(
        model_id,
        Qwen2_5_VLForConditionalGeneration,
        dtype=dtype,
        attn_impl=attn_impl,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        console=console,
    )

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
                title=Path(uri).name,
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
                title=Path(uri).name,
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
