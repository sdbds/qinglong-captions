"""
DeepSeek OCR Provider

基于 OCRProvider 基类的 DeepSeek OCR 实现
"""
from __future__ import annotations

import shutil
import time
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Any, Optional

import torch
from rich.console import Console
from rich.progress import Progress
from rich_pixels import Pixels

from module.providers.base import CaptionResult, MediaContext, PromptContext
from module.providers.ocr.document import rebase_generated_images
from module.providers.ocr_base import OCRProvider
from module.providers.registry import register_provider
from utils.ocr_model_config import DEEPSEEK_LEGACY_MODEL, DEEPSEEK_NATIVE_MODEL, deepseek_native_model_id
from utils.output_writer import write_markdown_output
from utils.parse_display import display_markdown
from utils.stream_util import iter_pdf_pages_high_quality
from utils.transformer_loader import resolve_device_dtype, snapshot_download_with_reporting, transformerLoader

_TRANS_LOADER: Optional[transformerLoader] = None


@lru_cache(maxsize=1)
def _official_legacy_snapshot() -> str:
    return snapshot_download_with_reporting(
        DEEPSEEK_LEGACY_MODEL, revision="aaa02f3811945a91062062994c5c4a3f4c0af2b0",
        allow_patterns=["*.py", "*.json", "*.safetensors"],
    )


def _legacy_source(model_id: str) -> str:
    if model_id == DEEPSEEK_LEGACY_MODEL and not Path(model_id).is_dir():
        return _official_legacy_snapshot()
    return model_id


@torch.inference_mode()
def attempt_deepseek_ocr(
    *,
    uri: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    model_id: str = DEEPSEEK_LEGACY_MODEL,
    prompt_text: Optional[str] = None,
    pixels: Optional[Pixels] = None,
    output_dir: Optional[str] = None,
    supports_flex_attn: bool = False,
    base_size: int = 1024,
    image_size: int = 768,
    crop_mode: bool = True,
) -> str:
    """Run local DeepSeek-OCR on a single image and return markdown text.

    Args:
      uri: path to the image file
      prompt_text: OCR instruction; default converts document to markdown
    """
    start_time = time.time()

    if version("transformers") != "4.46.3":
        raise RuntimeError("Legacy DeepSeek OCR requires its separate deepseek-ocr-legacy environment (Transformers 4.46.3).")
    model_id = _legacy_source(model_id)

    # default prompt from user spec
    if not prompt_text:
        prompt_text = "<image>\n<|grounding|>Convert the document to markdown. "

    p = Path(uri)
    if not output_dir:
        output_dir = str(p.with_suffix(""))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel, AutoTokenizer

    device, dtype, attn_impl = resolve_device_dtype()
    global _TRANS_LOADER
    if _TRANS_LOADER is None:
        _TRANS_LOADER = transformerLoader(
            attn_kw="_attn_implementation",
            device_map="auto",
            supports_flex_attn=supports_flex_attn,
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
        all_contents = []
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

                res = model.infer(
                    tokenizer,
                    prompt=prompt_text,
                    image_file=str(page_img_path),
                    output_path=str(page_dir),
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    save_results=True,
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
                all_contents.append(rebase_generated_images(page_content.strip(), page_dir, Path(output_dir)))
            finally:
                pil_img.close()

        content = "\n<--- Page Split --->\n".join(all_contents)
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


@register_provider("deepseek_ocr")
class DeepSeekOCRProvider(OCRProvider):
    """DeepSeek OCR Provider"""

    default_model_id = DEEPSEEK_LEGACY_MODEL
    default_prompt = "<image>\n<|grounding|>Convert the document to markdown."
    legacy_only = False

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        """执行 DeepSeek OCR"""
        if self.get_runtime_backend().is_openai:
            if self.legacy_only:
                raise ValueError("deepseek_ocr_legacy is a local rollback route; use deepseek_ocr for a configured remote service")
            return self.attempt_via_openai_backend(media, prompts)
        output_dir = self.artifact_output_dir(media)

        model_id = self._get_model_config("model_id", self.default_model_id)
        geometry = {
            key: getattr(self.ctx.args, f"deepseek_{key}", self._get_model_config(key, default))
            for key, default in (("base_size", 1024), ("image_size", 768), ("crop_mode", True))
        }
        native_id = None if self.legacy_only else deepseek_native_model_id(model_id, **geometry)
        options = dict(
            uri=media.uri,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
            prompt_text=prompts.user,
            pixels=media.pixels,
            output_dir=str(output_dir) if output_dir else None,
            supports_flex_attn=bool(getattr(self, "_supports_flex_attn", False)),
            **geometry,
        )
        if native_id is not None:
            from .deepseek_native import attempt_deepseek_native

            result = attempt_deepseek_native(
                model_id=native_id, max_new_tokens=self._get_model_config("max_new_tokens", 8192), **options,
            )
        else:
            if model_id == DEEPSEEK_NATIVE_MODEL and not Path(model_id).is_dir():
                model_id = DEEPSEEK_LEGACY_MODEL
            result = attempt_deepseek_ocr(model_id=model_id, **options)
        return CaptionResult(
            raw=rebase_generated_images(str(result), output_dir, Path(media.uri).parent),
            metadata={"provider": self.name, "output_dir": str(output_dir), "model_id": native_id or model_id,
                      "ocr_engine": "native" if native_id else "legacy"},
        )


@register_provider("deepseek_ocr_legacy")
class DeepSeekLegacyOCRProvider(DeepSeekOCRProvider):
    """Explicit rollback using the old model ABI and the existing DeepSeek preferences."""

    legacy_only = True

    def _get_model_config(self, key: str, default: Any = None) -> Any:
        return super()._get_model_config(key, self.ctx.config.get("deepseek_ocr", {}).get(key, default))

    def get_prompts(self, mime: str):
        prompts = self.ctx.config.get("prompts", {})
        return "", prompts.get("deepseek_ocr_legacy_prompt", prompts.get("deepseek_ocr_prompt", self.default_prompt))
