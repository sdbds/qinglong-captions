"""
Unlimited-OCR Provider

Based on OCRProvider base class, using native Transformers path.
Model: https://huggingface.co/baidu/Unlimited-OCR
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Optional

import torch
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

_MULTI_PAGE_PROMPT = "<image>Multi page parsing."

_IMAGE_MODE_DEFAULTS = {
    "gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True},
    "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
}


def _resolve_image_mode_params(
    image_mode: str,
    config_base_size: Optional[int],
    config_image_size: Optional[int],
    config_crop_mode: Optional[bool],
) -> tuple[int, int, bool]:
    """Resolve (base_size, image_size, crop_mode) from image_mode and config overrides."""
    if image_mode not in _IMAGE_MODE_DEFAULTS:
        raise ValueError(f"Unsupported unlimited_ocr image_mode: {image_mode}")

    defaults = _IMAGE_MODE_DEFAULTS[image_mode]
    base_size = config_base_size if config_base_size is not None else defaults["base_size"]
    image_size = config_image_size if config_image_size is not None else defaults["image_size"]
    crop_mode = config_crop_mode if config_crop_mode is not None else defaults["crop_mode"]
    return base_size, image_size, crop_mode


@torch.inference_mode()
def attempt_unlimited_ocr(
    *,
    uri: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    model_id: str = "baidu/Unlimited-OCR",
    prompt_text: Optional[str] = None,
    pixels: Optional[Pixels] = None,
    output_dir: Optional[str] = None,
    image_mode: str = "gundam",
    base_size: Optional[int] = None,
    image_size: Optional[int] = None,
    crop_mode: Optional[bool] = None,
    max_length: int = 32768,
    no_repeat_ngram_size: int = 35,
) -> str:
    """Run local Unlimited-OCR on a single image or PDF and return markdown text.

    Args:
      uri: path to the image or PDF file
      prompt_text: OCR instruction for single image; PDF uses internal fixed prompt
      image_mode: "gundam" or "base"; PDF always uses base
    """
    start_time = time.time()

    if not prompt_text:
        prompt_text = "<image>document parsing."

    p = Path(uri)
    if not output_dir:
        output_dir = str(p.with_suffix(""))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel, AutoTokenizer

    _, dtype, attn_impl = resolve_device_dtype()
    global _TRANS_LOADER
    if _TRANS_LOADER is None:
        _TRANS_LOADER = transformerLoader(
            attn_kw="_attn_implementation",
            device_map="auto",
        )

    tokenizer = _TRANS_LOADER.get_or_load_processor(model_id, AutoTokenizer, console=console)
    model = _TRANS_LOADER.get_or_load_model(
        model_id,
        AutoModel,
        dtype=dtype,
        attn_impl=attn_impl,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_safetensors=True,
        console=console,
    )

    if p.suffix.lower() == ".pdf":
        content = _process_pdf(
            p=p,
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            console=console,
            pixels=pixels,
            max_length=max_length,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    else:
        content = _process_single_image(
            p=p,
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            output_dir=output_dir,
            image_mode=image_mode,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            max_length=max_length,
            no_repeat_ngram_size=no_repeat_ngram_size,
            console=console,
            pixels=pixels,
        )

    if not content or not content.strip():
        raise RuntimeError("Unlimited-OCR returned empty output")

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


def _process_single_image(
    *,
    p: Path,
    model,
    tokenizer,
    prompt_text: str,
    output_dir: str,
    image_mode: str,
    base_size: Optional[int],
    image_size: Optional[int],
    crop_mode: Optional[bool],
    max_length: int,
    no_repeat_ngram_size: int,
    console: Console,
    pixels: Optional[Pixels],
) -> str:
    """Process a single image using model.infer."""
    resolved_base, resolved_image, resolved_crop = _resolve_image_mode_params(
        image_mode, base_size, image_size, crop_mode
    )

    img_path = str(p)
    res = model.infer(
        tokenizer,
        prompt=prompt_text,
        image_file=img_path,
        output_path=output_dir,
        base_size=resolved_base,
        image_size=resolved_image,
        crop_mode=resolved_crop,
        save_results=True,
        max_length=max_length,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )

    try:
        mmd_path = Path(output_dir) / "result.mmd"
        if mmd_path.exists():
            result_md_path = Path(output_dir) / "result.md"
            shutil.move(mmd_path, result_md_path)
            content = result_md_path.read_text(encoding="utf-8")
        else:
            content = str(res) if not isinstance(res, str) else res
    except Exception:
        content = str(res) if not isinstance(res, str) else res

    return content


def _process_pdf(
    *,
    p: Path,
    model,
    tokenizer,
    output_dir: str,
    console: Console,
    pixels: Optional[Pixels],
    max_length: int,
    no_repeat_ngram_size: int,
) -> str:
    """Process a multi-page PDF using model.infer_multi (one-shot long-horizon)."""
    images = pdf_to_images_high_quality(str(p))
    if not images:
        raise RuntimeError("Unlimited-OCR multi-page parsing failed: no images extracted from PDF")

    image_files: list[str] = []
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
        image_files.append(str(page_img_path))

    if not image_files:
        raise RuntimeError("Unlimited-OCR multi-page parsing failed: no valid page images")

    try:
        res = model.infer_multi(
            tokenizer,
            prompt=_MULTI_PAGE_PROMPT,
            image_files=image_files,
            output_path=output_dir,
            image_size=1024,
            save_results=True,
            max_length=max_length,
            no_repeat_ngram_size=no_repeat_ngram_size,
            ngram_window=1024,
        )
    except Exception as exc:
        raise RuntimeError("Unlimited-OCR multi-page parsing failed") from exc

    # Try reading result.mmd first (same as deepseek), then fallback to return value
    try:
        mmd_path = Path(output_dir) / "result.mmd"
        if mmd_path.exists():
            result_md_path = Path(output_dir) / "result.md"
            shutil.move(mmd_path, result_md_path)
            content = result_md_path.read_text(encoding="utf-8")
        else:
            content = str(res) if not isinstance(res, str) else res
    except Exception:
        content = str(res) if not isinstance(res, str) else res

    return content


@register_provider("unlimited_ocr")
class UnlimitedOCRProvider(OCRProvider):
    """Unlimited-OCR Provider"""

    default_model_id = "baidu/Unlimited-OCR"
    default_prompt = "<image>document parsing."

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        """Execute Unlimited-OCR"""
        if self.get_runtime_backend().is_openai:
            raise NotImplementedError(
                "unlimited_ocr does not support the OpenAI-compatible runtime backend yet"
            )
        output_dir = media.extras.get("output_dir")

        result = attempt_unlimited_ocr(
            uri=media.uri,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
            model_id=self._get_model_config("model_id", self.default_model_id),
            prompt_text=prompts.user,
            pixels=media.pixels,
            output_dir=str(output_dir) if output_dir else None,
            image_mode=self._get_model_config("image_mode", "gundam"),
            base_size=self._get_model_config("base_size", None),
            image_size=self._get_model_config("image_size", None),
            crop_mode=self._get_model_config("crop_mode", None),
            max_length=self._get_model_config("max_length", 32768),
            no_repeat_ngram_size=self._get_model_config("no_repeat_ngram_size", 35),
        )

        return CaptionResult(
            raw=result if isinstance(result, str) else str(result),
            metadata={
                "provider": self.name,
                "model_id": self._get_model_config("model_id", self.default_model_id),
                "output_dir": str(output_dir),
            },
        )
