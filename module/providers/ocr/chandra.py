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

from module.providers.base import CaptionResult, MediaContext, PromptContext
from module.providers.ocr_base import OCRProvider
from module.providers.registry import register_provider
from utils.parse_display import display_markdown
from utils.output_writer import write_markdown_output
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
    model_id: str = "datalab-to/chandra-ocr-2",
    prompt_type: str = "ocr_layout",
    pixels: Optional[Pixels] = None,
    output_dir: Optional[str] = None,
    max_new_tokens: int = 12384,
) -> str:
    """Run local Chandra OCR on a single image or PDF and return markdown text.

    Args:
      uri: path to the image or PDF file
      prompt_type: Chandra prompt type; default is "ocr_layout"
      max_new_tokens: maximum number of tokens to generate
    """
    from chandra.model.hf import generate_hf
    from chandra.model.schema import BatchInputItem
    from chandra.output import parse_markdown

    start_time = time.time()

    def log_stage(message: str) -> None:
        console.print(f"[cyan][Chandra][/cyan] {message}")
        if progress and task_id is not None:
            progress.update(task_id, description=f"Chandra: {message}")

    def log_warning(message: str, exc: Optional[BaseException] = None) -> None:
        if exc is None:
            console.print(f"[yellow][Chandra][/yellow] {message}")
        else:
            console.print(f"[yellow][Chandra][/yellow] {message}: {type(exc).__name__}: {exc}")

    p = Path(uri)
    if not output_dir:
        output_dir = str(p.with_suffix(""))

    from transformers import AutoModelForImageTextToText, AutoProcessor

    device, dtype, attn_impl = resolve_device_dtype()
    global _TRANS_LOADER
    if _TRANS_LOADER is None:
        _TRANS_LOADER = transformerLoader(attn_kw="_attn_implementation", device_map="auto")

    processor = _TRANS_LOADER.get_or_load_processor(model_id, AutoProcessor, console=console)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    model = _TRANS_LOADER.get_or_load_model(
        model_id,
        AutoModelForImageTextToText,
        dtype=dtype,
        attn_impl=attn_impl,
        device_map="auto",
        console=console,
    )
    # Chandra requires processor attached to model
    model.processor = processor
    log_stage(f"Model ready: {model_id} on {device} (dtype={dtype}, prompt_type={prompt_type}, max_new_tokens={max_new_tokens})")

    if p.suffix.lower() == ".pdf":
        log_stage(f"Rendering PDF to images: {p.name}")
        images = pdf_to_images_high_quality(str(p))
        log_stage(f"Rendered {len(images)} page(s) from PDF")
        all_contents = []
        for idx, pil_img in enumerate(images):
            page_no = idx + 1
            log_stage(f"Processing PDF page {page_no}/{len(images)} ({pil_img.width}x{pil_img.height}, mode={pil_img.mode})")
            page_dir = Path(output_dir) / f"page_{idx + 1:04d}"
            page_dir.mkdir(parents=True, exist_ok=True)
            page_img_path = page_dir / f"page_{idx + 1:04d}.png"
            try:
                pil_img.save(page_img_path)
                log_stage(f"Saved page preview: {page_img_path}")
            except Exception as exc:
                log_warning(f"Direct page save failed, retrying with RGB conversion for {page_img_path}", exc)
                try:
                    pil_img.convert("RGB").save(page_img_path)
                    log_stage(f"Saved page preview after RGB conversion: {page_img_path}")
                except Exception as retry_exc:
                    log_warning(f"Skipping preview image save for page {page_no}", retry_exc)
                    continue

            # Process with Chandra OCR
            log_stage(f"Running OCR on PDF page {page_no}/{len(images)}")
            inputs = [BatchInputItem(image=pil_img, prompt_type=prompt_type)]
            generate_start = time.time()
            raw_output = generate_hf(inputs, model, max_output_tokens=max_new_tokens)
            generate_elapsed = time.time() - generate_start
            log_stage(
                f"OCR finished for PDF page {page_no}/{len(images)} in {generate_elapsed:.2f}s "
                f"(raw_chars={len(raw_output[0].raw)})"
            )
            output_text = parse_markdown(raw_output[0].raw)

            # Process line breaks for display - add two spaces before newlines for markdown line breaks
            output_text = output_text.replace("\n", "  \n")

            # Save result
            write_markdown_output(page_dir, output_text, filename=f"{p.stem}_{idx + 1:04d}.md")
            log_stage(f"Saved page markdown: {page_dir / f'{p.stem}_{idx + 1:04d}.md'}")
            all_contents.append(output_text.strip())

        content = "\n<--- Page Split --->\n".join(all_contents)
        # Process line breaks for merged content - add two spaces before newlines for markdown line breaks
        content = content.replace("\n", "  \n")
        try:
            write_markdown_output(Path(output_dir), content, filename=f"{p.stem}_merged.md")
            log_stage(f"Saved merged markdown: {Path(output_dir) / f'{p.stem}_merged.md'}")
        except Exception as exc:
            log_warning("Failed to save merged markdown", exc)

        try:
            display_markdown(
                title=p.name,
                markdown_content=content,
                pixels=pixels,
                panel_height=32,
                console=console,
            )
            log_stage("Rendered markdown preview in console")
        except Exception as exc:
            log_warning("Markdown preview skipped", exc)
    else:
        # Single image processing
        log_stage(f"Opening image: {p.name}")
        pil_img = Image.open(str(p))
        log_stage(f"Opened image ({pil_img.width}x{pil_img.height}, mode={pil_img.mode})")

        # Process with Chandra OCR
        log_stage("Running OCR on image")
        inputs = [BatchInputItem(image=pil_img, prompt_type=prompt_type)]
        generate_start = time.time()
        raw_output = generate_hf(inputs, model, max_output_tokens=max_new_tokens)
        generate_elapsed = time.time() - generate_start
        log_stage(f"OCR finished on image in {generate_elapsed:.2f}s (raw_chars={len(raw_output[0].raw)})")
        content = parse_markdown(raw_output[0].raw)

        # Process line breaks for display - add two spaces before newlines for markdown line breaks
        content = content.replace("\n", "  \n")

        # Save result
        try:
            write_markdown_output(Path(output_dir), content, filename=f"{p.stem}.md")
            log_stage(f"Saved markdown: {Path(output_dir) / f'{p.stem}.md'}")
        except Exception as exc:
            log_warning("Failed to save markdown", exc)

        try:
            display_markdown(
                title=p.name,
                markdown_content=content,
                pixels=pixels,
                panel_height=32,
                console=console,
            )
            log_stage("Rendered markdown preview in console")
        except Exception as exc:
            log_warning("Markdown preview skipped", exc)

    elapsed = time.time() - start_time
    if progress and task_id is not None:
        progress.update(task_id, description="Generating captions")
    console.print(f"[blue]Caption generation took:[/blue] {elapsed:.2f} seconds")

    return content


@register_provider("chandra_ocr")
class ChandraOCRProvider(OCRProvider):
    """Chandra OCR Provider"""

    default_model_id = "datalab-to/chandra-ocr-2"
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
            max_new_tokens=self._get_model_config("max_new_tokens", 12384),
        )

        return CaptionResult(
            raw=result if isinstance(result, str) else str(result), metadata={"provider": self.name, "output_dir": str(output_dir)}
        )
