"""
Unlimited-OCR Provider

Based on OCRProvider base class, using native Transformers path.
Model: https://huggingface.co/baidu/Unlimited-OCR
"""
from __future__ import annotations

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
from utils.stream_util import iter_pdf_pages_high_quality
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
    page_budget: int = 30,
) -> str:
    """Run local Unlimited-OCR on a single image or PDF and return markdown text.

    Args:
      uri: path to the image or PDF file
      prompt_text: OCR instruction for single image; PDF uses internal fixed prompt
      image_mode: "gundam" or "base"; PDF always uses base
      page_budget: max pages per infer_multi call. PDFs beyond it are parsed in
        page_budget-sized chunks so the 32K context ceiling cannot silently
        truncate trailing pages. Set <=0 to always use a single call.
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
    try:
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
    except KeyError:
        # Unlimited-OCR custom code comments out mha_flash_attention_2 in
        # ATTENTION_CLASSES; fall back to eager if the resolved impl is unsupported.
        if console:
            console.print(
                f"[yellow]attn_impl={attn_impl} not supported by model, "
                f"falling back to eager[/yellow]"
            )
        attn_impl = "eager"
        model = _TRANS_LOADER.get_or_load_model(
            model_id,
            AutoModel,
            dtype=dtype,
            attn_impl="eager",
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
            page_budget=page_budget,
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

    # Add two spaces before newlines for markdown hard line breaks
    content = content.replace("\n", "  \n")

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

    # Unlimited-OCR writes result.md directly (not result.mmd like DeepSeek)
    result_md_path = Path(output_dir) / "result.md"
    if result_md_path.exists():
        content = result_md_path.read_text(encoding="utf-8")
    elif isinstance(res, str):
        content = res
    else:
        content = str(res)

    return content


def _run_infer_multi(
    *,
    model,
    tokenizer,
    image_files: list[str],
    out_dir: str,
    max_length: int,
    no_repeat_ngram_size: int,
) -> str:
    """One long-horizon infer_multi call over a batch of page images.

    Unlimited-OCR writes result.md directly into out_dir and returns a tuple
    (outputs, output_tokens); read the file, fall back to the return value.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    try:
        res = model.infer_multi(
            tokenizer,
            prompt=_MULTI_PAGE_PROMPT,
            image_files=image_files,
            output_path=out_dir,
            image_size=1024,
            save_results=True,
            max_length=max_length,
            no_repeat_ngram_size=no_repeat_ngram_size,
            ngram_window=1024,
        )
    except Exception as exc:
        raise RuntimeError("Unlimited-OCR multi-page parsing failed") from exc

    result_md_path = Path(out_dir) / "result.md"
    if result_md_path.exists():
        return result_md_path.read_text(encoding="utf-8")
    if isinstance(res, tuple) and res:
        return str(res[0])
    if isinstance(res, str):
        return res
    return str(res)


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
    page_budget: int,
) -> str:
    """Process a multi-page PDF using model.infer_multi (one-shot long-horizon).

    Within page_budget pages: a single infer_multi call (the model's native
    long-horizon path). Beyond it: parse in page_budget-sized chunks so the 32K
    context ceiling can't silently drop trailing pages — the paper itself caps
    one-shot parsing around 40-50 pages and ships no chunking. page_budget <=0
    forces a single call.
    """
    image_files: list[str] = []
    for rendered_page in iter_pdf_pages_high_quality(str(p)):
        page_number = rendered_page.page_number
        pil_img = rendered_page.image
        try:
            page_dir = Path(output_dir) / f"page_{page_number:04d}"
            page_dir.mkdir(parents=True, exist_ok=True)
            page_img_path = page_dir / f"page_{page_number:04d}.png"
            try:
                pil_img.save(page_img_path)
            except Exception:
                try:
                    pil_img.convert("RGB").save(page_img_path)
                except Exception:
                    continue
            image_files.append(str(page_img_path))
        finally:
            pil_img.close()

    if not image_files:
        raise RuntimeError("Unlimited-OCR multi-page parsing failed: no valid page images")

    # Within budget: one shot — identical to a plain single infer_multi call.
    if page_budget <= 0 or len(image_files) <= page_budget:
        return _run_infer_multi(
            model=model,
            tokenizer=tokenizer,
            image_files=image_files,
            out_dir=output_dir,
            max_length=max_length,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

    # Over budget: chunk so the 32K ceiling can't silently truncate. Each chunk
    # gets its own dir so per-chunk result.md files don't clobber each other;
    # a failed chunk propagates (fail loud > silently incomplete markdown).
    total = len(image_files)
    console.print(
        f"[yellow]Unlimited-OCR:[/yellow] {total} pages exceed page_budget={page_budget}; "
        f"parsing in {page_budget}-page chunks to avoid 32K truncation."
    )
    chunk_texts: list[str] = []
    for start in range(0, total, page_budget):
        chunk_idx = start // page_budget + 1
        batch = image_files[start : start + page_budget]
        chunk_dir = Path(output_dir) / f"chunk_{chunk_idx:04d}"
        chunk_text = _run_infer_multi(
            model=model,
            tokenizer=tokenizer,
            image_files=batch,
            out_dir=str(chunk_dir),
            max_length=max_length,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        if chunk_text and chunk_text.strip():
            chunk_texts.append(chunk_text.strip())

    content = "\n<--- Page Split --->\n".join(chunk_texts).strip()
    if not content:
        raise RuntimeError("Unlimited-OCR multi-page parsing failed")
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
            page_budget=self._get_model_config("page_budget", 30),
        )

        return CaptionResult(
            raw=result if isinstance(result, str) else str(result),
            metadata={
                "provider": self.name,
                "model_id": self._get_model_config("model_id", self.default_model_id),
                "output_dir": str(output_dir),
            },
        )
