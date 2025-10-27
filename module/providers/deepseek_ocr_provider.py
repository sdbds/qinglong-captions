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
from utils.stream_util import pdf_to_images_high_quality
from utils.transformer_loader import transformerLoader, resolve_device_dtype


_TRANS_LOADER: Optional[transformerLoader] = None


@torch.inference_mode()
def attempt_deepseek_ocr(
    *,
    uri: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    model_id: str = "deepseek-ai/DeepSeek-OCR",
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

    p = Path(uri)
    if not output_dir:
        output_dir = str(p.with_suffix(""))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device, dtype, attn_impl = resolve_device_dtype()
    global _TRANS_LOADER
    if _TRANS_LOADER is None:
        _TRANS_LOADER = transformerLoader(
            attn_kw="_attn_implementation", device_map="auto"
        )

    tokenizer = _TRANS_LOADER.get_or_load_processor(
        model_id, AutoTokenizer, console=console
    )
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
        images = pdf_to_images_high_quality(str(p))
        all_contents = []
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

            res = model.infer(
                tokenizer,
                prompt=prompt_text,
                image_file=str(page_img_path),
                output_path=str(page_dir),
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=True,
                test_compress=test_compress,
            )

            try:
                mmd_path = page_dir / "result.mmd"
                if mmd_path.exists():
                    result_md_path = page_dir / "result.md"
                    shutil.move(mmd_path, result_md_path)
                    page_content = result_md_path.read_text(encoding="utf-8")
                else:
                    page_content = str(res) if not isinstance(res, str) else res
            except Exception:
                page_content = str(res) if not isinstance(res, str) else res
            all_contents.append(page_content.strip())

        content = "\n<--- Page Split --->\n".join(all_contents)
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
        img_path = str(p)
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
