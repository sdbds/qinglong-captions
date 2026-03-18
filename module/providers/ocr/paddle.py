"""PaddleOCR Provider

最复杂的 OCR Provider，有很多配置选项
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from paddleocr import PaddleOCRVL
except ImportError as e:
    PaddleOCRVL = None
    _IMPORT_ERROR = e
from rich.console import Console
from rich.progress import Progress
from rich_pixels import Pixels

from module.providers.base import CaptionResult, MediaContext, PromptContext
from module.providers.ocr_base import OCRProvider
from module.providers.registry import register_provider
from utils.parse_display import display_markdown


def _filter_pipeline_kwargs(pipeline_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(PaddleOCRVL)
        valid = set(sig.parameters.keys())
        return {k: v for k, v in pipeline_kwargs.items() if k in valid}
    except Exception:
        return pipeline_kwargs


def _run_pipeline(
    pipeline,
    input_path: str,
    out_dir: Path,
    *,
    save: Dict[str, bool],
    pdf: Optional[Dict[str, Any]] = None,
) -> None:
    path_lower = str(input_path).lower()
    is_pdf = path_lower.endswith(".pdf")
    if is_pdf:
        output_iter = pipeline.predict(input=input_path)
        pages_res = list(output_iter)
        pdf = pdf if isinstance(pdf, dict) else {}

        restructure_kwargs: Dict[str, Any] = {}
        for k in ("merge_table", "relevel_titles", "merge_pages"):
            v = pdf.get(k)
            if v is None:
                continue
            restructure_kwargs[k] = bool(v)

        output = pipeline.restructure_pages(pages_res, **restructure_kwargs)
    else:
        output = pipeline.predict(input_path)

    for res in output:
        try:
            res.print()
        except Exception:
            pass

        save_methods = {
            "json": "save_to_json",
            "markdown": "save_to_markdown",
            "img": "save_to_img",
            "xlsx": "save_to_xlsx",
            "html": "save_to_html",
            "csv": "save_to_csv",
            "video": "save_to_video",
        }

        for key, method_name in save_methods.items():
            if not bool(save.get(key, False)):
                continue
            try:
                method = getattr(res, method_name, None)
                if callable(method):
                    method(save_path=str(out_dir))
            except Exception:
                pass


def attempt_paddle_ocr(
    *,
    uri: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    pixels: Optional[Pixels] = None,
    output_dir: Optional[str] = None,
    save: Optional[Dict[str, bool]] = None,
    pipeline_kwargs: Optional[Dict[str, Any]] = None,
    pdf_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    start = time.time()

    image_path = str(Path(uri))
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    effective_pipeline_kwargs = (
        _filter_pipeline_kwargs(pipeline_kwargs)
        if isinstance(pipeline_kwargs, dict)
        else {}
    )
    pipeline = PaddleOCRVL(**effective_pipeline_kwargs)
    out_dir = Path(output_dir) if output_dir else Path(uri).with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    save_flags = (
        save
        if isinstance(save, dict)
        else {
            "json": True,
            "markdown": True,
            "img": True,
            "xlsx": False,
            "html": False,
            "csv": False,
            "video": False,
        }
    )
    _run_pipeline(
        pipeline,
        image_path,
        out_dir,
        save=save_flags,
        pdf=pdf_kwargs,
    )

    md_files = sorted(out_dir.glob("*.md"))
    content_parts = []
    for md in md_files:
        try:
            content_parts.append(md.read_text(encoding="utf-8"))
        except Exception:
            continue
    content = "\n\n".join(content_parts).strip()

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

    elapsed = time.time() - start
    if progress and task_id is not None:
        progress.update(task_id, description="Generating captions")
    console.print(f"[blue]Caption generation took:[/blue] {elapsed:.2f} seconds")

    return content


@register_provider("paddle_ocr")
class PaddleOCRProvider(OCRProvider):
    """PaddleOCR Provider"""

    default_model_id = "PaddleOCR"  # Paddle 不使用 HuggingFace model_id
    default_prompt = ""

    def _get_paddle_config(self) -> Dict[str, Any]:
        """获取 PaddleOCR 特有的复杂配置"""
        section = self.ctx.config.get("paddle_ocr", {})

        # save flags
        save_flags = {
            "json": True,
            "markdown": True,
            "img": True,
            "xlsx": False,
            "html": False,
            "csv": False,
            "video": False,
        }

        save_section = section.get("save", {})
        for k, cfg_key in (
            ("json", "save_json"),
            ("markdown", "save_markdown"),
            ("img", "save_img"),
            ("xlsx", "save_xlsx"),
            ("html", "save_html"),
            ("csv", "save_csv"),
            ("video", "save_video"),
        ):
            v = save_section.get(cfg_key)
            if v is None:
                v = section.get(cfg_key)
            if v is not None:
                save_flags[k] = bool(v)

        # pipeline kwargs
        pipeline_kwargs = {}
        pipeline_section = section.get("pipeline", {})
        for key in (
            "use_doc_orientation_classify",
            "use_doc_unwarping",
            "use_layout_detection",
            "use_chart_recognition",
            "use_seal_recognition",
            "use_ocr_for_image_block",
            "enable_hpi",
            "use_tensorrt",
            "format_block_content",
            "merge_layout_blocks",
            "markdown_ignore_labels",
            "layout_threshold",
            "layout_nms",
            "layout_unclip_ratio",
            "layout_merge_bboxes_mode",
        ):
            v = pipeline_section.get(key)
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == "":
                continue
            # 处理布尔字符串
            if key in ("enable_hpi", "use_tensorrt") and isinstance(v, str):
                v_low = v.strip().lower()
                if v_low in ("true", "1", "yes", "on"):
                    v = True
                elif v_low in ("false", "0", "no", "off"):
                    v = False
                else:
                    continue
            # 处理数值
            if key in ("layout_threshold", "layout_nms", "layout_unclip_ratio") and isinstance(v, str):
                try:
                    v = float(v)
                except ValueError:
                    continue
            pipeline_kwargs[key] = v

        # PDF kwargs
        pdf_kwargs = {}
        pdf_section = section.get("pdf", {})
        for key in ("merge_table", "relevel_titles", "merge_pages"):
            v = pdf_section.get(key)
            if v is not None and not (isinstance(v, str) and v.strip() == ""):
                pdf_kwargs[key] = bool(v)

        return {
            "save_flags": save_flags,
            "pipeline_kwargs": pipeline_kwargs,
            "pdf_kwargs": pdf_kwargs,
        }

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        output_dir = media.extras.get("output_dir")
        config = self._get_paddle_config()

        result = attempt_paddle_ocr(
            uri=media.uri,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
            pixels=media.pixels,
            output_dir=str(output_dir) if output_dir else None,
            save=config["save_flags"],
            pipeline_kwargs=config["pipeline_kwargs"],
            pdf_kwargs=config["pdf_kwargs"],
        )

        return CaptionResult(
            raw=result if isinstance(result, str) else str(result),
            metadata={"provider": self.name, "output_dir": str(output_dir), "paddle_config": config},
        )
