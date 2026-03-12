"""Chandra OCR Provider"""
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
from utils.stream_util import pdf_to_images_high_quality
from utils.transformer_loader import resolve_device_dtype, transformerLoader

_TRANS_LOADER: Optional[transformerLoader] = None


@torch.inference_mode()
def attempt_chandra_ocr(
    *,
    uri: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    model_id: str = "datalab-to/chandra",
    prompt_type: str = "ocr_layout",
    pixels: Optional[Pixels] = None,
    output_dir: Optional[str] = None,
    max_new_tokens: int = 8192,
) -> str:
    """Run local Chandra OCR on a single image or PDF and return markdown text.

    Args:
      uri: path to the image or PDF file
      prompt_type: Chandra prompt type; default is "ocr_layout"
      max_new_tokens: maximum number of tokens to generate
    """
    from chandra.model.hf import generate_hf, BatchInputItem
    from chandra.output import parse_markdown

    start_time = time.time()

    p = Path(uri)
    if not output_dir:
        output_dir = str(p.with_suffix(""))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel, AutoProcessor

    device, dtype, attn_impl = resolve_device_dtype()
    global _TRANS_LOADER
    if _TRANS_LOADER is None:
        _TRANS_LOADER = transformerLoader(attn_kw="_attn_implementation", device_map="auto")

    processor = _TRANS_LOADER.get_or_load_processor(model_id, AutoProcessor, console=console)
    model = _TRANS_LOADER.get_or_load_model(
        model_id,
        AutoModel,
        dtype=dtype,
        attn_impl=attn_impl,
        device_map="auto",
        console=console,
    )
    # Chandra requires processor attached to model
    model.processor = processor

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

            # Process with Chandra OCR
            inputs = [BatchInputItem(image=pil_img, prompt_type=prompt_type)]
            raw_output = generate_hf(model, inputs, max_new_tokens=max_new_tokens)
            output_text = parse_markdown(raw_output[0])

            # Process line breaks for display - add two spaces before newlines for markdown line breaks
            output_text = output_text.replace("\n", "  \n")

            # Save result
            result_md_path = page_dir / f"{p.stem}_{idx + 1:04d}.md"
            result_md_path.write_text(output_text, encoding="utf-8")
            all_contents.append(output_text.strip())

        content = "\n<--- Page Split --->\n".join(all_contents)
        # Process line breaks for merged content - add two spaces before newlines for markdown line breaks
        content = content.replace("\n", "  \n")
        try:
            (Path(output_dir) / f"{p.stem}_merged.md").write_text(content, encoding="utf-8")
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
        try:
            pil_img = pil_img.convert("RGB")
        except Exception:
            pass

        # Process with Chandra OCR
        inputs = [BatchInputItem(image=pil_img, prompt_type=prompt_type)]
        raw_output = generate_hf(model, inputs, max_new_tokens=max_new_tokens)
        content = parse_markdown(raw_output[0])

        # Process line breaks for display - add two spaces before newlines for markdown line breaks
        content = content.replace("\n", "  \n")

        # Save result
        try:
            result_md_path = Path(output_dir) / f"{p.stem}.md"
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


@register_provider("chandra_ocr")
class ChandraOCRProvider(OCRProvider):
    """Chandra OCR Provider"""

    default_model_id = "datalab-to/chandra"
    default_prompt = ""

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        if self.get_runtime_backend().is_openai:
            return self.attempt_via_openai_backend(media, prompts)

        output_dir = media.extras.get("output_dir")

        # 读取 prompt_type 配置
        prompt_type = self._get_model_config(
            "prompt_type", self.ctx.config.get("prompts", {}).get("chandra_ocr_prompt_type", "ocr_layout")
        )

        result = attempt_chandra_ocr(
            uri=media.uri,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
            model_id=self._get_model_config("model_id", self.default_model_id),
            prompt_type=prompt_type,
            pixels=media.pixels,
            output_dir=str(output_dir) if output_dir else None,
            max_new_tokens=self._get_model_config("max_new_tokens", 8192),
        )

        return CaptionResult(
            raw=result if isinstance(result, str) else str(result), metadata={"provider": self.name, "output_dir": str(output_dir)}
        )
