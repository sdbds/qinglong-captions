"""PaddleOCR Provider

最复杂的 OCR Provider，有很多配置选项
"""

from pathlib import Path
from typing import Any, Dict

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.ocr_base import OCRProvider
from providers.registry import register_provider


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
        from module.providers.paddle_ocr_provider import attempt_paddle_ocr

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
