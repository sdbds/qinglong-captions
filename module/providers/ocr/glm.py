"""GLM OCR Provider"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from rich.console import Console
from rich.progress import Progress
from rich_pixels import Pixels

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.ocr_base import OCRProvider
from providers.registry import register_provider
from utils.parse_display import display_markdown
from utils.output_writer import write_markdown_output
from utils.stream_util import pdf_to_images_high_quality
from utils.transformer_loader import resolve_device_dtype, transformerLoader

_TRANS_LOADER: Optional[transformerLoader] = None


@torch.inference_mode()
def attempt_glm_ocr(
    *,
    uri: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    model_id: str = "zai-org/GLM-OCR",
    prompt_text: Optional[str] = None,
    pixels: Optional[Pixels] = None,
    output_dir: Optional[str] = None,
    max_new_tokens: int = 8192,
) -> str:
    """Run local GLM-OCR on a single image and return markdown text.

    Args:
      uri: path to the image file
      prompt_text: OCR instruction; default is "Text Recognition:"
      max_new_tokens: maximum number of tokens to generate
    """
    start_time = time.time()

    # default prompt
    if not prompt_text:
        prompt_text = "Text Recognition:"

    p = Path(uri)
    if not output_dir:
        output_dir = str(p.with_suffix(""))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    device, dtype, attn_impl = resolve_device_dtype()
    global _TRANS_LOADER
    if _TRANS_LOADER is None:
        _TRANS_LOADER = transformerLoader(attn_kw="_attn_implementation", device_map="auto")

    processor = _TRANS_LOADER.get_or_load_processor(model_id, AutoProcessor, console=console)
    model = _TRANS_LOADER.get_or_load_model(
        model_id,
        AutoModelForImageTextToText,
        dtype=dtype,
        attn_impl=attn_impl,
        device_map="auto",
        console=console,
    )

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

            # Process with GLM-OCR
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": str(page_img_path)},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            inputs.pop("token_type_ids", None)

            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            output_text = processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            # Process line breaks for display - add two spaces before newlines for markdown line breaks
            output_text = output_text.replace("\n", "  \n")

            # Save result
            write_markdown_output(page_dir, output_text, filename=f"{p.stem}_{idx + 1:04d}.md")
            all_contents.append(output_text.strip())

        content = "\n<--- Page Split --->\n".join(all_contents)
        # Process line breaks for merged content - add two spaces before newlines for markdown line breaks
        content = content.replace("\n", "  \n")
        try:
            write_markdown_output(Path(output_dir), content, filename=f"{p.stem}_merged.md")
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
        img_path = str(p)

        # Load and process image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": img_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        inputs.pop("token_type_ids", None)

        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        content = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        # Process line breaks for display - add two spaces before newlines for markdown line breaks
        content = content.replace("\n", "  \n")

        # Save result
        try:
            write_markdown_output(Path(output_dir), content, filename=f"{p.stem}.md")
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


@register_provider("glm_ocr")
class GLMOCRProvider(OCRProvider):
    """GLM OCR Provider"""

    default_model_id = "zai-org/GLM-OCR"
    default_prompt = "Text Recognition:"

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        if self.get_runtime_backend().is_openai:
            return self.attempt_via_openai_backend(media, prompts)

        output_dir = media.extras.get("output_dir")

        result = attempt_glm_ocr(
            uri=media.uri,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
            model_id=self._get_model_config("model_id", self.default_model_id),
            prompt_text=prompts.user,
            pixels=media.pixels,
            output_dir=str(output_dir) if output_dir else None,
            max_new_tokens=self._get_model_config("max_new_tokens", 8192),
        )

        return CaptionResult(
            raw=result if isinstance(result, str) else str(result), metadata={"provider": self.name, "output_dir": str(output_dir)}
        )
