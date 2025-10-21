# /// script
# dependencies = [
#   "torch==2.8.0",
#   "transformers==4.46.3",
#   "tokenizers==0.20.3",
#   "einops",
#   "addict",
#   "easydict",
#   "flash-attn==2.8.3; sys_platform == 'linux'",
#   "triton-windows ; sys_platform == 'win32'",
#   "flash-attn @ https://github.com/sdbds/flash-attention-for-windows/releases/download/2.8.3/flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSEfullbackward-cp311-cp311-win_amd64.whl; sys_platform == 'win32'",
#   "safetensors",
#   "rich>=13.5.0",
# ]
# ///
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModel, AutoTokenizer
from rich.console import Console
from rich.progress import Progress
from rich_pixels import Pixels

from utils.parse_display import display_markdown
import shutil


# Global lazy cache
_MODEL: Optional[Any] = None
_TOKENIZER: Optional[Any] = None


def _resolve_device_dtype() -> tuple[torch.device, torch.dtype, str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        return device, torch.bfloat16, "flash_attention_2"
    # CPU fallback
    return device, torch.float32, "eager"


def _get_model_and_tokenizer(model_name: str = "deepseek-ai/DeepSeek-OCR", console: Optional[Console] = None):
    global _MODEL, _TOKENIZER
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    dev, dt, attn_impl = _resolve_device_dtype()
    if console:
        console.print(f"[green]Loading DeepSeek-OCR: {model_name} (device={dev}, dtype={dt}, attn={attn_impl})[/green]")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True,
        _attn_implementation=attn_impl,
    )
    if dev.type == "cuda":
        model = model.cuda().to(dt)
    model = model.eval()

    _MODEL, _TOKENIZER = model, tokenizer
    return _MODEL, _TOKENIZER

@torch.inference_mode()
def attempt_deepseek_ocr(
    *,
    uri: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    prompt_text: Optional[str] = None,
    pixels: Optional[Pixels] = None,
    output_dir: Optional[str] = None,
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
    test_compress: bool = True,
) -> str:
    """Run local DeepSeek-OCR on a single image and return markdown text.

    Args:
      uri: path to the image file
      prompt_text: OCR instruction; default converts document to markdown
    """
    start_time = time.time()

    # default prompt from user spec
    if not prompt_text:
        prompt_text = "<image>\n<|grounding|>Convert the document to markdown. "

    # ensure path format
    img_path = str(Path(uri))

    # decide output dir if saving results
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model, tokenizer = _get_model_and_tokenizer(console=console)

    # run inference
    res = model.infer(
        tokenizer,
        prompt=prompt_text,
        image_file=img_path,
        output_path=output_dir,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        save_results=True,
        test_compress=test_compress,
    )

    # best-effort stringify
    try:
        mmd_path = Path(output_dir) / "result.mmd" if output_dir else None
        if mmd_path and mmd_path.exists():
            result_md_path = Path(output_dir) / "result.md"
            shutil.move(mmd_path, result_md_path)
            content = result_md_path.read_text(encoding="utf-8")
        else:
            content = str(res) if not isinstance(res, str) else res
    except Exception:
        content = str(res) if not isinstance(res, str) else res

    # display
    try:
        display_markdown(
            title=Path(uri).name,
            markdown_content=content,
            pixels=pixels,
            panel_height=32,
            console=console,
        )
    except Exception:
        # fallback: still return content
        pass

    elapsed = time.time() - start_time
    if progress and task_id is not None:
        progress.update(task_id, description="Generating captions")
    console.print(f"[blue]Caption generation took:[/blue] {elapsed:.2f} seconds")

    return content
