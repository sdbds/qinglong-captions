from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from rich.console import Console
from rich.progress import Progress
from rich_pixels import Pixels
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from utils.parse_display import display_markdown
from utils.stream_util import pdf_to_images_high_quality
from utils.transformer_loader import resolve_device_dtype, transformerLoader

_TRANS_LOADER: Optional[transformerLoader] = None

# Default OCR prompt for document parsing (from official HuggingFace model card)
DEFAULT_OCR_PROMPT = (
    "Extract the text from the above document as if you were reading it naturally. "
    "Return the tables in html format. Return the equations in LaTeX representation. "
    "If there is an image in the document and image caption is not present, "
    "add a small description of the image inside the <img></img> tag; "
    "otherwise, add the image caption inside <img></img>. "
    "Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. "
    "Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. "
    "Prefer using ☐ and ☑ for check boxes."
)


def _build_messages(image_path: str, prompt_text: str) -> list[dict]:
    """Build chat messages for Nanonets OCR inference."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


@torch.inference_mode()
def attempt_nanonets_ocr(
    *,
    uri: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    model_id: str = "nanonets/Nanonets-OCR2-3B",
    prompt_text: Optional[str] = None,
    pixels: Optional[Pixels] = None,
    output_dir: Optional[str] = None,
    max_new_tokens: int = 15000,
) -> str:
    """Run local Nanonets-OCR2 on a single image and return markdown text.

    Args:
      uri: path to the image file or PDF
      console: Rich console for output
      progress: Optional progress bar
      task_id: Optional task ID for progress updates
      model_id: HuggingFace model ID (default: nanonets/Nanonets-OCR2-3B)
      prompt_text: OCR instruction; default extracts document to markdown
      pixels: Optional pixel art display
      output_dir: Output directory for results
      max_new_tokens: Maximum tokens to generate (default: 15000)

    Returns:
      Extracted text content in markdown format
    """
    start_time = time.time()

    if not prompt_text:
        prompt_text = DEFAULT_OCR_PROMPT

    p = Path(uri)
    if not output_dir:
        output_dir = str(p.with_suffix(""))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device, dtype, attn_impl = resolve_device_dtype()
    global _TRANS_LOADER
    if _TRANS_LOADER is None:
        _TRANS_LOADER = transformerLoader(attn_kw="attn_implementation", device_map="auto")

    # Load processor and model
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
    model.eval()

    def _infer_single_image(image_path: str, pil_image: Image.Image) -> str:
        """Perform OCR inference on a single image."""
        messages = _build_messages(image_path, prompt_text)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(
            text=[text],
            images=[pil_image],
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to model device
        device = next(model.parameters()).device
        inputs = inputs.to(device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # Trim input tokens from output
        generated_ids = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]

        output_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return output_texts[0] if output_texts else ""

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

            # Save individual page result
            try:
                result_md_path = page_dir / "result.md"
                result_md_path.write_text(page_content, encoding="utf-8")
            except Exception:
                pass

            all_contents.append(page_content.strip())

        content = "\n<--- Page Split --->\n".join(all_contents)

        # Save combined result
        try:
            (Path(output_dir) / "result.md").write_text(content, encoding="utf-8")
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
        # Single image processing
        pil_img = Image.open(str(p))
        content = _infer_single_image(str(p), pil_img)

        # Save result
        try:
            result_md_path = Path(output_dir) / "result.md"
            result_md_path.write_text(content, encoding="utf-8")
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
