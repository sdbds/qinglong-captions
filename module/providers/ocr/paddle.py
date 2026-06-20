"""PaddleOCR Provider."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from rich.console import Console
from rich.progress import Progress
from rich_pixels import Pixels

from module.onnx_runtime import OnnxModelSpec, load_single_model_bundle, resolve_tool_runtime_config
from module.providers.base import CaptionResult, MediaContext, PromptContext
from module.providers.ocr_base import OCRProvider
from module.providers.registry import register_provider
from utils.parse_display import display_markdown

DEFAULT_PADDLE_BACKEND = "ppocrv6_onnx"
DEFAULT_PPOCRV6_TIER = "medium"
PADDLE_BACKENDS = {"ppocrv6_onnx", "ppocrv6_direct_onnx", "paddle_vl_native"}
PPOCRV6_TIERS = {"tiny", "small", "medium"}
PADDLEOCR_COMMON_INIT_KWARGS = {
    "cpu_threads",
    "device",
    "enable_cinn",
    "enable_hpi",
    "enable_mkldnn",
    "engine",
    "engine_config",
    "mkldnn_cache_capacity",
    "paddlex_config",
    "precision",
    "use_tensorrt",
}
PPOCRV6_OCR_INIT_KWARGS = {
    "doc_orientation_classify_model_dir",
    "doc_orientation_classify_model_name",
    "doc_unwarping_model_dir",
    "doc_unwarping_model_name",
    "lang",
    "ocr_version",
    "return_word_box",
    "text_det_box_thresh",
    "text_det_input_shape",
    "text_det_limit_side_len",
    "text_det_limit_type",
    "text_det_thresh",
    "text_det_unclip_ratio",
    "text_detection_model_dir",
    "text_detection_model_name",
    "text_rec_input_shape",
    "text_rec_score_thresh",
    "text_recognition_batch_size",
    "text_recognition_model_dir",
    "text_recognition_model_name",
    "textline_orientation_batch_size",
    "textline_orientation_model_dir",
    "textline_orientation_model_name",
    "use_doc_orientation_classify",
    "use_doc_unwarping",
    "use_textline_orientation",
}
PPOCRV6_ONNX_INIT_KWARGS = PADDLEOCR_COMMON_INIT_KWARGS | PPOCRV6_OCR_INIT_KWARGS
PADDLEOCR_VL_INIT_KWARGS = PADDLEOCR_COMMON_INIT_KWARGS | {
    "doc_orientation_classify_model_dir",
    "doc_orientation_classify_model_name",
    "doc_unwarping_model_dir",
    "doc_unwarping_model_name",
    "format_block_content",
    "layout_detection_model_dir",
    "layout_detection_model_name",
    "layout_merge_bboxes_mode",
    "layout_nms",
    "layout_threshold",
    "layout_unclip_ratio",
    "markdown_ignore_labels",
    "merge_layout_blocks",
    "pipeline_version",
    "use_chart_recognition",
    "use_doc_orientation_classify",
    "use_doc_unwarping",
    "use_layout_detection",
    "use_ocr_for_image_block",
    "use_queues",
    "use_seal_recognition",
    "vl_rec_api_key",
    "vl_rec_api_model_name",
    "vl_rec_backend",
    "vl_rec_max_concurrency",
    "vl_rec_model_dir",
    "vl_rec_model_name",
    "vl_rec_server_url",
}


def _import_paddle_ocr():
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:  # pragma: no cover - exercised through provider import failures
        raise ImportError("paddle_ocr backend 'ppocrv6_onnx' requires the paddleocr-onnx extra") from exc
    return PaddleOCR


def _import_paddle_ocr_vl():
    try:
        from paddleocr import PaddleOCRVL
    except ImportError as exc:  # pragma: no cover - exercised through provider import failures
        raise ImportError("paddle_ocr backend 'paddle_vl_native' requires the paddleocr-native extra") from exc
    return PaddleOCRVL


def _normalize_backend(value: Any) -> str:
    backend = str(value or DEFAULT_PADDLE_BACKEND).strip().lower()
    if backend not in PADDLE_BACKENDS:
        raise ValueError(f"Unsupported paddle_ocr backend: {backend!r}")
    return backend


def _normalize_model_tier(value: Any) -> str:
    tier = str(value or DEFAULT_PPOCRV6_TIER).strip().lower()
    if tier not in PPOCRV6_TIERS:
        raise ValueError(f"Unsupported PP-OCRv6 model_tier: {tier!r}")
    return tier


def _resolve_ppocrv6_model_names(model_tier: str) -> tuple[str, str]:
    tier = _normalize_model_tier(model_tier)
    return f"PP-OCRv6_{tier}_det", f"PP-OCRv6_{tier}_rec"


def _coerce_bool_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    return value


def _coerce_float_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return float(value)
    except ValueError:
        return value


def _coerce_int_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except ValueError:
        return value


def _filter_ppocrv6_onnx_kwargs(pipeline_kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in pipeline_kwargs.items() if key in PPOCRV6_ONNX_INIT_KWARGS}


def _filter_paddle_vl_kwargs(pipeline_kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in pipeline_kwargs.items() if key in PADDLEOCR_VL_INIT_KWARGS}


def _coerce_pipeline_kwargs(
    source: Mapping[str, Any],
    *,
    bool_keys: set[str],
    float_keys: set[str],
    int_keys: set[str],
    passthrough_keys: set[str],
) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for key in tuple(bool_keys | float_keys | int_keys | passthrough_keys):
        v = source.get(key)
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        if key in bool_keys:
            v = _coerce_bool_string(v)
            if isinstance(v, str):
                continue
        elif key in float_keys:
            v = _coerce_float_string(v)
            if isinstance(v, str):
                continue
        elif key in int_keys:
            v = _coerce_int_string(v)
            if isinstance(v, str):
                continue
        parsed[key] = v
    return parsed


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _to_builtin(value.tolist())
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def _result_payload(res: Any) -> dict[str, Any]:
    payload = getattr(res, "json", None)
    if callable(payload):
        payload = payload()
    if not isinstance(payload, Mapping):
        return {}
    if isinstance(payload.get("res"), Mapping):
        payload = payload["res"]
    return dict(payload)


def _extract_markdown_and_metadata(results: list[Any]) -> tuple[str, dict[str, Any]]:
    rec_texts: list[Any] = []
    rec_scores: list[Any] = []
    rec_boxes: list[Any] = []
    rec_polys: list[Any] = []

    for res in results:
        payload = _result_payload(res)
        texts = _to_builtin(payload.get("rec_texts", []))
        if isinstance(texts, list):
            rec_texts.extend(texts)
        scores = _to_builtin(payload.get("rec_scores", []))
        if isinstance(scores, list):
            rec_scores.extend(scores)
        boxes = _to_builtin(payload.get("rec_boxes", []))
        if isinstance(boxes, list):
            rec_boxes.extend(boxes)
        polys = _to_builtin(payload.get("rec_polys", []))
        if isinstance(polys, list):
            rec_polys.extend(polys)

    lines = [str(text).strip() for text in rec_texts if str(text or "").strip()]
    content = "\n".join(lines)
    return content, {
        "rec_texts": rec_texts,
        "rec_scores": rec_scores,
        "rec_boxes": rec_boxes,
        "rec_polys": rec_polys,
    }


def _save_result_outputs(res: Any, out_dir: Path, *, save: Mapping[str, bool]) -> None:
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


def _run_vl_pipeline(
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
        for k in ("merge_tables", "relevel_titles", "concatenate_pages"):
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
        _save_result_outputs(res, out_dir, save=save)


def _display_and_finish(
    *,
    uri: str,
    content: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    pixels: Optional[Pixels],
    start: float,
) -> None:
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


def attempt_ppocrv6_onnx(
    *,
    uri: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    pixels: Optional[Pixels] = None,
    output_dir: Optional[str] = None,
    save: Optional[Dict[str, bool]] = None,
    pipeline_kwargs: Optional[Dict[str, Any]] = None,
    model_tier: str = DEFAULT_PPOCRV6_TIER,
) -> tuple[str, dict[str, Any]]:
    start = time.time()
    image_path = str(Path(uri))
    out_dir = Path(output_dir) if output_dir else Path(uri).with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    PaddleOCR = _import_paddle_ocr()
    det_model_name, rec_model_name = _resolve_ppocrv6_model_names(model_tier)
    effective_pipeline_kwargs = _filter_ppocrv6_onnx_kwargs(pipeline_kwargs or {})
    effective_pipeline_kwargs.update(
        {
            "ocr_version": "PP-OCRv6",
            "engine": "onnxruntime",
            "text_detection_model_name": det_model_name,
            "text_recognition_model_name": rec_model_name,
        }
    )
    pipeline = PaddleOCR(**effective_pipeline_kwargs)

    results = list(pipeline.predict(input=image_path))
    save_flags = save if isinstance(save, dict) else _default_save_flags()
    for res in results:
        try:
            res.print()
        except Exception:
            pass
        _save_result_outputs(res, out_dir, save=save_flags)

    content, metadata = _extract_markdown_and_metadata(results)
    _display_and_finish(
        uri=uri,
        content=content,
        console=console,
        progress=progress,
        task_id=task_id,
        pixels=pixels,
        start=start,
    )
    metadata.update(
        {
            "backend": "ppocrv6_onnx",
            "engine": "onnxruntime",
            "model_tier": _normalize_model_tier(model_tier),
            "text_detection_model_name": det_model_name,
            "text_recognition_model_name": rec_model_name,
        }
    )
    return content, metadata


def attempt_paddle_vl_native(
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
) -> tuple[str, dict[str, Any]]:
    start = time.time()
    image_path = str(Path(uri))
    out_dir = Path(output_dir) if output_dir else Path(uri).with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    PaddleOCRVL = _import_paddle_ocr_vl()
    effective_pipeline_kwargs = _filter_paddle_vl_kwargs(pipeline_kwargs or {})
    pipeline = PaddleOCRVL(**effective_pipeline_kwargs)
    save_flags = save if isinstance(save, dict) else _default_save_flags()
    _run_vl_pipeline(
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

    _display_and_finish(
        uri=uri,
        content=content,
        console=console,
        progress=progress,
        task_id=task_id,
        pixels=pixels,
        start=start,
    )
    return content, {"backend": "paddle_vl_native", "deprecated": True}


def attempt_ppocrv6_direct_onnx(
    *,
    config: Mapping[str, Any],
    model_tier: str,
    console: Console,
) -> None:
    tier = _normalize_model_tier(model_tier)
    det_model_name, rec_model_name = _resolve_ppocrv6_model_names(tier)
    runtime_config = resolve_tool_runtime_config(config, tool_name="paddle_ocr")
    for component, model_name in (("det", det_model_name), ("rec", rec_model_name)):
        repo_id = f"PaddlePaddle/{model_name}_onnx"
        spec = OnnxModelSpec(
            repo_id=repo_id,
            onnx_filename="inference.onnx",
            local_dir=runtime_config.resolve_model_cache_dir(repo_id),
            bundle_key=f"paddle_ocr:{tier}:{component}",
            support_files={
                "inference_config": "inference.yml",
                "inference_program": "inference.json",
            },
        )
        load_single_model_bundle(spec=spec, runtime_config=runtime_config, logger=console.print)
    raise NotImplementedError(
        "ppocrv6_direct_onnx is experimental: ONNX model loading is wired, but PP-OCRv6 "
        "pre/post-processing parity is not implemented yet."
    )


def attempt_paddle_ocr(**kwargs) -> str:
    content, _metadata = attempt_paddle_vl_native(**kwargs)
    return content


def _default_save_flags() -> dict[str, bool]:
    return {
        "json": True,
        "markdown": True,
        "img": True,
        "xlsx": False,
        "html": False,
        "csv": False,
        "video": False,
    }


@register_provider("paddle_ocr")
class PaddleOCRProvider(OCRProvider):
    """PaddleOCR Provider."""

    default_model_id = "PaddleOCR"
    default_prompt = ""

    def _get_paddle_config(self) -> Dict[str, Any]:
        section = self.ctx.config.get("paddle_ocr", {})

        save_flags = _default_save_flags()
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

        common_bool_keys = {
            "enable_cinn",
            "enable_hpi",
            "enable_mkldnn",
            "use_doc_orientation_classify",
            "use_doc_unwarping",
            "use_tensorrt",
        }
        common_int_keys = {"cpu_threads", "mkldnn_cache_capacity"}
        common_passthrough_keys = {
            "device",
            "engine",
            "engine_config",
            "paddlex_config",
            "precision",
        }

        pipeline_section = section.get("pipeline", {})
        ocr_bool_keys = common_bool_keys | {
            "use_textline_orientation",
            "return_word_box",
        }
        ocr_float_keys = {
            "text_det_thresh",
            "text_det_box_thresh",
            "text_det_unclip_ratio",
            "text_rec_score_thresh",
        }
        ocr_int_keys = common_int_keys | {
            "text_det_limit_side_len",
            "text_recognition_batch_size",
            "textline_orientation_batch_size",
        }
        ocr_passthrough_keys = common_passthrough_keys | {
            "doc_orientation_classify_model_dir",
            "doc_orientation_classify_model_name",
            "doc_unwarping_model_dir",
            "doc_unwarping_model_name",
            "lang",
            "text_det_limit_type",
            "text_det_input_shape",
            "text_rec_input_shape",
            "text_detection_model_dir",
            "text_detection_model_name",
            "text_recognition_model_dir",
            "text_recognition_model_name",
            "textline_orientation_model_dir",
            "textline_orientation_model_name",
        }
        ocr_pipeline_kwargs = _coerce_pipeline_kwargs(
            pipeline_section,
            bool_keys=ocr_bool_keys,
            float_keys=ocr_float_keys,
            int_keys=ocr_int_keys,
            passthrough_keys=ocr_passthrough_keys,
        )

        vl_section = section.get("vl_pipeline", {})
        vl_bool_keys = common_bool_keys | {
            "format_block_content",
            "merge_layout_blocks",
            "use_chart_recognition",
            "use_layout_detection",
            "use_ocr_for_image_block",
            "use_queues",
            "use_seal_recognition",
        }
        vl_float_keys = {
            "layout_nms",
            "layout_threshold",
            "layout_unclip_ratio",
        }
        vl_int_keys = common_int_keys | {"vl_rec_max_concurrency"}
        vl_passthrough_keys = common_passthrough_keys | {
            "doc_orientation_classify_model_dir",
            "doc_orientation_classify_model_name",
            "doc_unwarping_model_dir",
            "doc_unwarping_model_name",
            "layout_detection_model_dir",
            "layout_detection_model_name",
            "layout_merge_bboxes_mode",
            "markdown_ignore_labels",
            "pipeline_version",
            "vl_rec_api_key",
            "vl_rec_api_model_name",
            "vl_rec_backend",
            "vl_rec_model_dir",
            "vl_rec_model_name",
            "vl_rec_server_url",
        }
        vl_pipeline_kwargs = {
            **_coerce_pipeline_kwargs(
                pipeline_section,
                bool_keys=vl_bool_keys,
                float_keys=vl_float_keys,
                int_keys=vl_int_keys,
                passthrough_keys=vl_passthrough_keys,
            ),
            **_coerce_pipeline_kwargs(
                vl_section,
                bool_keys=vl_bool_keys,
                float_keys=vl_float_keys,
                int_keys=vl_int_keys,
                passthrough_keys=vl_passthrough_keys,
            ),
        }

        pdf_kwargs = {}
        pdf_section = section.get("pdf", {})
        pdf_aliases = {
            "merge_tables": ("merge_tables", "merge_table"),
            "relevel_titles": ("relevel_titles",),
            "concatenate_pages": ("concatenate_pages", "merge_pages"),
        }
        for key, aliases in pdf_aliases.items():
            v = next((pdf_section.get(alias) for alias in aliases if pdf_section.get(alias) is not None), None)
            if v is not None and not (isinstance(v, str) and v.strip() == ""):
                v = _coerce_bool_string(v)
                if not isinstance(v, str):
                    pdf_kwargs[key] = bool(v)

        return {
            "backend": _normalize_backend(section.get("backend", DEFAULT_PADDLE_BACKEND)),
            "model_tier": _normalize_model_tier(section.get("model_tier", DEFAULT_PPOCRV6_TIER)),
            "save_flags": save_flags,
            "pipeline_kwargs": (
                vl_pipeline_kwargs
                if _normalize_backend(section.get("backend", DEFAULT_PADDLE_BACKEND)) == "paddle_vl_native"
                else ocr_pipeline_kwargs
            ),
            "pdf_kwargs": pdf_kwargs,
        }

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        del prompts
        output_dir = media.extras.get("output_dir")
        config = self._get_paddle_config()
        backend = config["backend"]

        if backend == "ppocrv6_onnx":
            result, metadata = attempt_ppocrv6_onnx(
                uri=media.uri,
                console=self.ctx.console,
                progress=self.ctx.progress,
                task_id=self.ctx.task_id,
                pixels=media.pixels,
                output_dir=str(output_dir) if output_dir else None,
                save=config["save_flags"],
                pipeline_kwargs=config["pipeline_kwargs"],
                model_tier=config["model_tier"],
            )
        elif backend == "ppocrv6_direct_onnx":
            attempt_ppocrv6_direct_onnx(
                config=self.ctx.config,
                model_tier=config["model_tier"],
                console=self.ctx.console,
            )
            result, metadata = "", {"backend": backend, "model_tier": config["model_tier"]}
        else:
            result, metadata = attempt_paddle_vl_native(
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
            metadata["model_tier"] = config["model_tier"]

        return CaptionResult(
            raw=result if isinstance(result, str) else str(result),
            metadata={
                "provider": self.name,
                "output_dir": str(output_dir),
                "paddle_config": config,
                **metadata,
            },
        )
