"""Dots OCR provider."""

from __future__ import annotations

import importlib.util
import io
import json
import math
import re
import sys
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Optional

from PIL import Image, ImageDraw

from providers.backends import OpenAIChatRuntime, find_model_config_section, resolve_runtime_backend
from providers.base import CaptionResult, MediaContext, MediaModality, PromptContext
from providers.ocr_base import OCRProvider
from providers.registry import register_provider
from providers.utils import build_vision_messages, encode_image_to_blob
from utils.output_writer import write_markdown_output
from utils.parse_display import display_markdown, extract_code_block_content
from utils.transformer_loader import resolve_device_dtype, transformerLoader

DEFAULT_PROMPT_MODE = "prompt_ocr"
DEFAULT_SVG_MODEL_ID = "davanstrien/dots.ocr-1.5-svg"
UPSTREAM_MIN_PIXELS = 3136
UPSTREAM_MAX_PIXELS = 11289600
UPSTREAM_DIRECT_MAX_NEW_TOKENS = 24000
UPSTREAM_RUNTIME_MAX_TOKENS = 16384
UPSTREAM_DPI = 200
UPSTREAM_FITZ_PREPROCESS = True
UPSTREAM_RUNTIME_TEMPERATURE = 0.1
UPSTREAM_RUNTIME_TOP_P = 1.0
LAYOUT_PROMPT_MODES = frozenset({
    "prompt_layout_all_en",
    "prompt_layout_only_en",
    "prompt_grounding_ocr",
})
_TRANS_LOADER: Optional[transformerLoader] = None

try:
    import importlib.metadata as importlib_metadata
except ImportError:  # pragma: no cover - Python < 3.8 fallback
    import importlib_metadata  # type: ignore[no-redef]


def _find_upstream_utils_file(relative_path: str) -> Path | None:
    """Locate a file inside the upstream dots_ocr package without importing its top-level package."""
    for dist_name in ("dots_ocr", "dots-ocr"):
        try:
            distribution = importlib_metadata.distribution(dist_name)
        except importlib_metadata.PackageNotFoundError:
            continue
        candidate = Path(distribution.locate_file(relative_path))
        if candidate.is_file():
            return candidate

    for entry in sys.path:
        candidate = Path(entry) / Path(relative_path)
        if candidate.is_file():
            return candidate
    return None


def _find_upstream_prompts_file() -> Path | None:
    """Locate the upstream prompts.py file without importing the broken package."""
    return _find_upstream_utils_file("dots_ocr/utils/prompts.py")


def _load_prompt_mapping_from_file(prompts_path: Path) -> dict[str, str]:
    """Load dict_promptmode_to_prompt from the upstream prompts.py source file."""
    spec = importlib.util.spec_from_file_location("_dots_ocr_prompts", str(prompts_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load prompts spec from {prompts_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompt_map = getattr(module, "dict_promptmode_to_prompt", None)
    if not isinstance(prompt_map, dict) or not prompt_map:
        raise ValueError(f"dict_promptmode_to_prompt missing from {prompts_path}")
    return {str(key): str(value) for key, value in prompt_map.items()}


def _load_upstream_prompt_mapping() -> dict[str, str]:
    """Load the official dots OCR prompt-mode mapping from the upstream package."""
    prompts_path = _find_upstream_prompts_file()
    if prompts_path is None:
        raise ImportError(
            "dots_ocr prompt mapping not available. Install the dots-ocr extra."
        )
    return _load_prompt_mapping_from_file(prompts_path)


def _load_upstream_pdf_images(pdf_path: str, dpi: int = 200):
    """Load PDF pages using the upstream dots_ocr PDF pipeline."""
    doc_utils_path = _find_upstream_utils_file("dots_ocr/utils/doc_utils.py")
    if doc_utils_path is None:
        raise ImportError(
            "dots_ocr PDF utilities are unavailable. Install the dots-ocr extra."
        )
    spec = importlib.util.spec_from_file_location("_dots_ocr_doc_utils", str(doc_utils_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load doc_utils spec from {doc_utils_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    load_images_from_pdf = getattr(module, "load_images_from_pdf", None)
    if load_images_from_pdf is None:
        raise ValueError(f"load_images_from_pdf missing from {doc_utils_path}")
    return load_images_from_pdf(pdf_path, dpi=dpi)


@lru_cache(maxsize=1)
def _load_output_cleaner_class():
    """Load the upstream OutputCleaner without importing the broken dots_ocr package."""
    cleaner_path = _find_upstream_utils_file("dots_ocr/utils/output_cleaner.py")
    if cleaner_path is None:
        raise ImportError(
            "dots_ocr output cleaner is unavailable. Install the dots-ocr extra."
        )
    spec = importlib.util.spec_from_file_location("_dots_ocr_output_cleaner", str(cleaner_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load output_cleaner spec from {cleaner_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cleaner_cls = getattr(module, "OutputCleaner", None)
    if cleaner_cls is None:
        raise ValueError(f"OutputCleaner missing from {cleaner_path}")
    return cleaner_cls


def _load_config_task_prompt_mapping(config) -> dict[str, str]:
    """Load optional dots_ocr prompt-mode overrides from prompts.task.dots_ocr."""
    prompts_section = config.get("prompts", {})
    task_section = prompts_section.get("task", {})
    dots_section = task_section.get("dots_ocr", {})
    if not isinstance(dots_section, Mapping):
        return {}
    return {str(key): str(value) for key, value in dots_section.items()}


def _download_model_snapshot(repo_id: str) -> str:
    """Resolve an HF repo id to a local snapshot path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "dots_ocr model download support is unavailable. Install the dots-ocr extra."
        ) from exc
    return str(snapshot_download(repo_id=repo_id))


@lru_cache(maxsize=8)
def _resolve_model_source(model_id: str) -> str:
    """Use a local snapshot path to avoid HF dynamic-module import issues on dotted repo names."""
    candidate = Path(model_id).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return str(Path(_download_model_snapshot(model_id)).resolve())


def _save_pdf_page_image(pil_image, page_path: Path) -> str:
    """Persist a rendered PDF page image and return its file path."""
    page_path = Path(page_path)
    page_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pil_image.save(page_path)
    except Exception:
        pil_image.convert("RGB").save(page_path)
    return str(page_path)


def _load_input_image(image_path: str):
    """Load an image like the upstream HF parser: materialize a PIL image before chat assembly."""
    with Image.open(image_path) as image:
        if image.mode == "RGBA":
            white_background = Image.new("RGB", image.size, (255, 255, 255))
            white_background.paste(image, mask=image.split()[3])
            return white_background
        return image.convert("RGB")


def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def _smart_resize_for_dots(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = UPSTREAM_MIN_PIXELS,
    max_pixels: int = UPSTREAM_MAX_PIXELS,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, _floor_by_factor(height / beta, factor))
        w_bar = max(factor, _floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((h_bar * w_bar) / max_pixels)
            h_bar = max(factor, _floor_by_factor(h_bar / beta, factor))
            w_bar = max(factor, _floor_by_factor(w_bar / beta, factor))
    return h_bar, w_bar


def _upstream_fetch_image_for_inference(
    image: Image.Image,
    *,
    min_pixels: int,
    max_pixels: int,
) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    resized_height, resized_width = _smart_resize_for_dots(
        height,
        width,
        factor=28,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if (resized_width, resized_height) != (width, height):
        image = image.resize((resized_width, resized_height))
    return image


def _fitz_preprocess_image(image: Image.Image, dpi: int = UPSTREAM_DPI):
    """Mirror upstream get_image_by_fitz_doc/fitz_doc_to_image for image inputs."""
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "dots_ocr fitz preprocess is unavailable. Install PyMuPDF."
        ) from exc

    data_bytes = io.BytesIO()
    image.save(data_bytes, format="PNG")
    pdf_bytes = fitz.open(stream=data_bytes.getvalue()).convert_to_pdf()
    doc = fitz.open("pdf", pdf_bytes)
    try:
        page = doc[0]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pixmap = page.get_pixmap(matrix=mat, alpha=False)
        if pixmap.width > 4500 or pixmap.height > 4500:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    finally:
        doc.close()


def _pil_image_to_base64(image: Image.Image, format_name: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format_name)
    encoded = buffer.getvalue()
    import base64
    return f"data:image/{format_name.lower()};base64,{base64.b64encode(encoded).decode('utf-8')}"


def _clean_latex_preamble(latex_text: str) -> str:
    patterns = [
        r"\\documentclass\{[^}]+\}",
        r"\\usepackage\{[^}]+\}",
        r"\\usepackage\[[^\]]*\]\{[^}]+\}",
        r"\\begin\{document\}",
        r"\\end\{document\}",
    ]
    cleaned = latex_text
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _has_latex_markdown(text: str) -> bool:
    if not isinstance(text, str):
        return False
    patterns = [
        r"\$\$.*?\$\$",
        r"\$[^$\n]+?\$",
        r"\\begin\{.*?\}.*?\\end\{.*?\}",
        r"\\[a-zA-Z]+\{.*?\}",
        r"\\[a-zA-Z]+",
        r"\\\[.*?\\\]",
        r"\\\(.*?\\\)",
    ]
    return any(re.search(pattern, text, re.DOTALL) for pattern in patterns)


def _get_formula_in_markdown(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    if text.startswith("$$") and text.endswith("$$"):
        inner = text[2:-2].strip()
        if "$" not in inner:
            return f"$$\n{inner}\n$$"
        return text
    if text.startswith("\\[") and text.endswith("\\]"):
        inner = text[2:-2].strip()
        return f"$$\n{inner}\n$$"
    if re.findall(r".*\\\[.*\\\].*", text):
        return text
    if re.findall(r"\$([^$]+)\$", text):
        return text
    if not _has_latex_markdown(text):
        return text
    if "usepackage" in text:
        text = _clean_latex_preamble(text)
    if text.startswith("`") and text.endswith("`"):
        text = text[1:-1]
    return f"$$\n{text}\n$$"


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).strip()
    if text[:2] == "`$" and text[-2:] == "$`":
        text = text[1:-1]
    return text


def _layoutjson_to_markdown(image: Image.Image, cells: list[dict], *, no_page_hf: bool = False) -> str:
    blocks: list[str] = []
    for cell in cells:
        bbox = cell.get("bbox")
        category = str(cell.get("category", ""))
        text = cell.get("text", "")
        if no_page_hf and category in {"Page-header", "Page-footer"}:
            continue
        if category == "Picture" and isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            image_crop = image.crop((x1, y1, x2, y2))
            blocks.append(f"![]({_pil_image_to_base64(image_crop)})")
        elif category == "Formula":
            blocks.append(_get_formula_in_markdown(text))
        else:
            blocks.append(_clean_text(text))
    return "\n\n".join(blocks)


def _post_process_cells(
    origin_image: Image.Image,
    cells: list[dict],
    *,
    input_width: int,
    input_height: int,
    min_pixels: int,
    max_pixels: int,
) -> list[dict]:
    original_width, original_height = origin_image.size
    resized_height, resized_width = _smart_resize_for_dots(
        input_height,
        input_width,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height
    processed: list[dict] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        updated = dict(cell)
        bbox = updated.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            updated["bbox"] = [
                int(float(bbox[0]) / scale_x),
                int(float(bbox[1]) / scale_y),
                int(float(bbox[2]) / scale_x),
                int(float(bbox[3]) / scale_y),
            ]
        processed.append(updated)
    return processed


def _draw_layout_on_image(image: Image.Image, cells: list[dict]) -> Image.Image:
    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    for index, cell in enumerate(cells):
        bbox = cell.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=2)
        draw.text((x1 + 4, y1 + 4), f"{index}", fill=(255, 0, 0))
    return rendered


@register_provider("dots_ocr")
class DotsOCRProvider(OCRProvider):
    """OCR provider backed by davanstrien/dots.ocr-1.5."""

    default_model_id = "davanstrien/dots.ocr-1.5"
    default_prompt = ""
    default_svg_model_id = DEFAULT_SVG_MODEL_ID

    def _log_stage_start(self, stage: str, details: str = "") -> float:
        suffix = f" ({details})" if details else ""
        self.ctx.console.print(f"dots_ocr stage start: {stage}{suffix}")
        return perf_counter()

    def _log_stage_done(self, stage: str, started_at: float) -> None:
        elapsed = perf_counter() - started_at
        self.ctx.console.print(f"dots_ocr stage done: {stage} ({elapsed:.2f}s)")

    def prepare_media(self, uri: str, mime: str, args) -> MediaContext:
        """Skip eager image encoding for local direct inference."""
        timer = self._log_stage_start("prepare_media", f"mime={mime}")
        blob = None
        pixels = None
        if mime.startswith("image") and self.get_runtime_backend().is_openai:
            blob, pixels = encode_image_to_blob(uri, to_rgb=True)

        output_dir = Path(uri).with_suffix("")

        media = MediaContext(
            uri=uri,
            mime=mime,
            sha256hash="",
            modality=MediaModality.DOCUMENT if mime.startswith("application") else MediaModality.IMAGE,
            blob=blob,
            pixels=pixels,
            extras={"output_dir": output_dir},
        )
        self._log_stage_done("prepare_media", timer)
        return media

    def resolve_prompts(self, uri: str, mime: str) -> PromptContext:
        _, prompt_text = self._resolve_prompt_mode_and_prompt()
        char_name, char_prompt = self._get_character_prompt(uri)
        return PromptContext(
            system="",
            user=f"{char_prompt}{prompt_text}" if char_prompt else prompt_text,
            character_name=char_name,
            character_prompt=char_prompt,
        )

    def _resolve_prompt_mode_and_prompt(self) -> tuple[str, str]:
        prompt_map = _load_upstream_prompt_mapping()
        task_prompt_map = _load_config_task_prompt_mapping(self.ctx.config)
        prompt_mode = str(
            self._get_model_config("prompt_mode", DEFAULT_PROMPT_MODE)
            or DEFAULT_PROMPT_MODE
        ).strip()
        if prompt_mode not in prompt_map:
            raise ValueError(f"Unsupported dots_ocr prompt_mode: {prompt_mode}")

        prompt_text = task_prompt_map.get(prompt_mode, prompt_map[prompt_mode])
        provider_prompt = str(self._get_model_config("prompt", "") or "").strip()
        global_prompt = str(
            self.ctx.config.get("prompts", {}).get("dots_ocr_prompt", "") or ""
        ).strip()
        if provider_prompt:
            prompt_text = provider_prompt
        elif global_prompt:
            prompt_text = global_prompt
        return prompt_mode, prompt_text

    def _select_model_id(self, prompt_mode: str) -> str:
        if prompt_mode == "prompt_image_to_svg":
            return str(
                self._get_model_config("svg_model_id", self.default_svg_model_id)
                or self.default_svg_model_id
            )
        return str(
            self._get_model_config("model_id", self.default_model_id)
            or self.default_model_id
        )

    def _resolve_image_pixel_limits(self) -> tuple[int, int]:
        min_pixels = int(
            self._get_model_config("min_pixels", UPSTREAM_MIN_PIXELS)
            or UPSTREAM_MIN_PIXELS
        )
        max_pixels = int(
            self._get_model_config("max_pixels", UPSTREAM_MAX_PIXELS)
            or UPSTREAM_MAX_PIXELS
        )
        return min_pixels, max_pixels

    def _resolve_image_preprocess_settings(self) -> tuple[bool, int]:
        fitz_preprocess = bool(
            self._get_model_config("fitz_preprocess", UPSTREAM_FITZ_PREPROCESS)
        )
        dpi = int(
            self._get_model_config("dpi", UPSTREAM_DPI)
            or UPSTREAM_DPI
        )
        return fitz_preprocess, dpi

    def _prepare_inference_image(
        self,
        origin_image: Image.Image,
        *,
        fitz_preprocess: bool,
        dpi: int,
        log: bool = True,
    ) -> Image.Image:
        input_image = origin_image
        if fitz_preprocess:
            stage_timer = self._log_stage_start("fitz_preprocess_image", f"dpi={dpi}")
            try:
                input_image = _fitz_preprocess_image(input_image, dpi=dpi)
            except ImportError as exc:
                self.ctx.console.print(f"dots_ocr fitz preprocess skipped: {exc}")
            self._log_stage_done("fitz_preprocess_image", stage_timer)
        min_pixels, max_pixels = self._resolve_image_pixel_limits()
        stage_timer = self._log_stage_start("upstream_fetch_image", f"min_pixels={min_pixels}, max_pixels={max_pixels}")
        input_image = _upstream_fetch_image_for_inference(
            input_image,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self._log_stage_done("upstream_fetch_image", stage_timer)
        if log:
            self.ctx.console.print(f"Original Size: {origin_image.width} x {origin_image.height}")
            self.ctx.console.print(f"Model Input Size: {input_image.width} x {input_image.height}")
        return input_image

    def _write_text_result(self, output_dir: Path, content: str) -> Path:
        return write_markdown_output(Path(output_dir), str(content), filename="result.md")

    def _write_svg_result(self, output_dir: Path, content: str) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "result.svg"
        result_path.write_text(str(content), encoding="utf-8")
        return result_path

    def _write_json_result(self, output_dir: Path, payload) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "result.json"
        if isinstance(payload, str):
            result_path.write_text(payload, encoding="utf-8")
        else:
            result_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return result_path

    def _write_layout_image_result(self, output_dir: Path, image: Image.Image) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "result.jpg"
        image.convert("RGB").save(result_path, format="JPEG", quality=95)
        return result_path

    def _build_svg_pdf_markdown(self, page_dirs: list[Path]) -> str:
        blocks: list[str] = []
        for page_number, page_dir in enumerate(page_dirs, start=1):
            blocks.append(f"## Page {page_number}\n![]({page_dir.name}/result.svg)")
        return "\n\n".join(blocks)

    def _post_process_layout_response(
        self,
        *,
        raw_content: str,
        prompt_mode: str,
        origin_image: Image.Image,
        input_image: Image.Image,
    ) -> dict:
        min_pixels, max_pixels = self._resolve_image_pixel_limits()
        try:
            cells = json.loads(raw_content)
            if not isinstance(cells, list):
                raise ValueError("layout output is not a list")
            cells = _post_process_cells(
                origin_image,
                cells,
                input_width=input_image.width,
                input_height=input_image.height,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            markdown = None
            markdown_nohf = None
            if prompt_mode != "prompt_layout_only_en":
                markdown = _layoutjson_to_markdown(origin_image, cells, no_page_hf=False)
                markdown_nohf = _layoutjson_to_markdown(origin_image, cells, no_page_hf=True)
            return {
                "filtered": False,
                "cells": cells,
                "markdown": markdown,
                "markdown_nohf": markdown_nohf,
                "layout_count": len(cells),
            }
        except Exception as exc:
            self.ctx.console.print(f"dots_ocr layout post-process fallback: {exc}")
            normalized_content = re.sub(r"\}\s*\{", "},{", raw_content)
            try:
                cleaned = json.loads(normalized_content)
            except Exception:
                try:
                    cleaner_cls = _load_output_cleaner_class()
                    cleaner = cleaner_cls()
                    cleaned = cleaner.clean_model_output(raw_content)
                except Exception:
                    cleaned = raw_content
            if isinstance(cleaned, list):
                markdown = "\n\n".join(
                    str(cell.get("text", "")).strip()
                    for cell in cleaned
                    if isinstance(cell, dict) and str(cell.get("text", "")).strip()
                )
                return {
                    "filtered": True,
                    "cells": cleaned,
                    "markdown": markdown,
                    "markdown_nohf": markdown,
                    "layout_count": len(cleaned),
                }
            markdown = str(cleaned)
            return {
                "filtered": True,
                "cells": None,
                "markdown": markdown,
                "markdown_nohf": markdown,
                "layout_count": 0,
            }

    def _finalize_text_result(
        self,
        *,
        output_dir: Path,
        prompt_mode: str,
        raw_content: str,
        origin_image: Image.Image | None = None,
        input_image: Image.Image | None = None,
    ) -> tuple[str, dict]:
        if prompt_mode not in LAYOUT_PROMPT_MODES or origin_image is None or input_image is None:
            self._write_text_result(output_dir, raw_content)
            return str(raw_content), {}

        processed = self._post_process_layout_response(
            raw_content=raw_content,
            prompt_mode=prompt_mode,
            origin_image=origin_image,
            input_image=input_image,
        )
        payload = processed["cells"] if processed["cells"] is not None else raw_content
        self._write_json_result(output_dir, payload)
        try:
            layout_image = _draw_layout_on_image(origin_image, processed["cells"] or [])
        except Exception:
            layout_image = origin_image
        self._write_layout_image_result(output_dir, layout_image)
        markdown = processed["markdown"]
        if markdown is not None:
            self._write_text_result(output_dir, markdown)
            markdown_nohf = processed.get("markdown_nohf")
            if markdown_nohf is not None:
                write_markdown_output(Path(output_dir), str(markdown_nohf), filename="result_nohf.md")
            return str(markdown), {
                "filtered": bool(processed["filtered"]),
                "layout_elements": int(processed["layout_count"]),
            }
        json_text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return str(json_text), {
            "filtered": bool(processed["filtered"]),
            "layout_elements": int(processed["layout_count"]),
        }

    def _run_direct_generation(
        self,
        *,
        image_path: str,
        prompt_mode: str,
        prompt_text: str,
        model_id: str,
        max_new_tokens: int,
        fitz_preprocess: bool = False,
        dpi: int = UPSTREAM_DPI,
    ) -> str:
        try:
            from qwen_vl_utils import process_vision_info
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "dots_ocr direct generation dependencies are unavailable. Install the dots-ocr extra."
            ) from exc

        _, dtype, attn_impl = resolve_device_dtype()
        global _TRANS_LOADER
        if _TRANS_LOADER is None:
            _TRANS_LOADER = transformerLoader(attn_kw="attn_implementation", device_map="auto")

        stage_timer = self._log_stage_start("resolve_model_source", f"model_id={model_id}")
        load_source = _resolve_model_source(model_id)
        self._log_stage_done("resolve_model_source", stage_timer)

        stage_timer = self._log_stage_start("load_processor")
        processor = _TRANS_LOADER.get_or_load_processor(
            load_source,
            AutoProcessor,
            console=self.ctx.console,
            trust_remote_code=True,
        )
        self._log_stage_done("load_processor", stage_timer)

        stage_timer = self._log_stage_start("load_model")
        model = _TRANS_LOADER.get_or_load_model(
            load_source,
            AutoModelForCausalLM,
            dtype=dtype,
            attn_impl=attn_impl,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_safetensors=True,
            console=self.ctx.console,
        )
        self._log_stage_done("load_model", stage_timer)

        stage_timer = self._log_stage_start("load_input_image", f"image={Path(image_path).name}")
        origin_image = _load_input_image(image_path)
        self._log_stage_done("load_input_image", stage_timer)
        input_image = self._prepare_inference_image(
            origin_image,
            fitz_preprocess=fitz_preprocess,
            dpi=dpi,
        )
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": input_image,
                },
                {"type": "text", "text": prompt_text},
            ],
        }]
        stage_timer = self._log_stage_start("apply_chat_template")
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        self._log_stage_done("apply_chat_template", stage_timer)

        stage_timer = self._log_stage_start("process_vision_info")
        image_inputs, video_inputs = process_vision_info(messages)
        self._log_stage_done("process_vision_info", stage_timer)

        stage_timer = self._log_stage_start("build_inputs")
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        self._log_stage_done("build_inputs", stage_timer)

        try:
            model_device = model.device
        except Exception:
            model_device = next(model.parameters()).device
        stage_timer = self._log_stage_start("move_inputs_to_device", f"device={model_device}")
        inputs = inputs.to(model_device)
        self._log_stage_done("move_inputs_to_device", stage_timer)

        stage_timer = self._log_stage_start("generate", f"max_new_tokens={int(max_new_tokens)}")
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
        )
        self._log_stage_done("generate", stage_timer)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        stage_timer = self._log_stage_start("decode")
        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        self._log_stage_done("decode", stage_timer)
        content = output_texts[0] if output_texts else ""
        if prompt_mode == "prompt_image_to_svg":
            extracted = extract_code_block_content(content, code_type="svg", console=self.ctx.console)
            return extracted or content
        return content

    def _get_runtime_backend_for_prompt_mode(self, prompt_mode: str):
        provider_section = self.ctx.config.get(self.name, {})
        selected_model_id = self._select_model_id(prompt_mode)
        model_section = find_model_config_section(
            self.ctx.config,
            selected_model_id,
            preferred_sections=(self.name,),
        )
        default_temperature = float(
            model_section.get(
                "temperature",
                provider_section.get("runtime_temperature", provider_section.get("temperature", UPSTREAM_RUNTIME_TEMPERATURE)),
            )
        )
        default_top_p = float(
            model_section.get(
                "top_p",
                provider_section.get("runtime_top_p", provider_section.get("top_p", UPSTREAM_RUNTIME_TOP_P)),
            )
        )
        default_max_tokens = int(
            model_section.get("runtime_max_tokens", provider_section.get("runtime_max_tokens", UPSTREAM_RUNTIME_MAX_TOKENS))
        )
        return resolve_runtime_backend(
            self.ctx.args,
            provider_section,
            arg_prefix="local_runtime",
            shared_prefix="openai",
            default_model_id=selected_model_id,
            default_temperature=default_temperature,
            default_top_p=default_top_p,
            default_max_tokens=default_max_tokens,
        )

    def _complete_via_openai_runtime(
        self,
        *,
        image_path: str,
        prompt_mode: str,
        prompt_text: str,
        fitz_preprocess: bool = False,
        dpi: int = UPSTREAM_DPI,
    ) -> str:
        runtime = self._get_runtime_backend_for_prompt_mode(prompt_mode)
        backend = OpenAIChatRuntime(runtime)
        stage_timer = self._log_stage_start("load_input_image", f"image={Path(image_path).name}")
        origin_image = _load_input_image(image_path)
        self._log_stage_done("load_input_image", stage_timer)
        input_image = self._prepare_inference_image(
            origin_image,
            fitz_preprocess=fitz_preprocess,
            dpi=dpi,
        )
        stage_timer = self._log_stage_start("encode_image_to_blob", f"image={Path(image_path).name}")
        with io.BytesIO() as buffer:
            input_image.save(buffer, format="JPEG", quality=95)
            blob = buffer.getvalue()
        import base64
        blob = base64.b64encode(blob).decode("utf-8")
        self._log_stage_done("encode_image_to_blob", stage_timer)
        if not blob:
            raise RuntimeError(f"DOTS_OCR_IMAGE_ENCODE_FAILED: {image_path}")

        messages = build_vision_messages(
            "",
            prompt_text,
            blob,
            text_first=False,
        )
        stage_timer = self._log_stage_start("openai_complete", f"model_id={runtime.model_id}")
        content = backend.complete(messages)
        self._log_stage_done("openai_complete", stage_timer)
        if prompt_mode == "prompt_image_to_svg":
            extracted = extract_code_block_content(content, code_type="svg", console=self.ctx.console)
            return extracted or content
        return content

    def _attempt_via_openai_backend(
        self,
        media: MediaContext,
        prompt_mode: str,
        prompt_text: str,
        *,
        fitz_preprocess: bool,
        dpi: int,
    ) -> CaptionResult:
        output_dir = Path(media.extras.get("output_dir") or Path(media.uri).with_suffix(""))
        output_dir.mkdir(parents=True, exist_ok=True)
        runtime = self._get_runtime_backend_for_prompt_mode(prompt_mode)

        if media.mime.startswith("application/pdf"):
            stage_timer = self._log_stage_start("upstream_pdf_to_images", f"file={Path(media.uri).name}")
            images = _load_upstream_pdf_images(media.uri)
            self._log_stage_done("upstream_pdf_to_images", stage_timer)
            if prompt_mode == "prompt_image_to_svg":
                page_dirs: list[Path] = []
                for index, pil_image in enumerate(images, start=1):
                    page_dir = output_dir / f"page_{index:04d}"
                    page_image_path = page_dir / f"page_{index:04d}.png"
                    try:
                        saved_image_path = _save_pdf_page_image(pil_image, page_image_path)
                    except Exception:
                        continue
                    svg_content = self._complete_via_openai_runtime(
                        image_path=str(saved_image_path),
                        prompt_mode=prompt_mode,
                        prompt_text=prompt_text,
                        fitz_preprocess=False,
                        dpi=dpi,
                    )
                    self._write_svg_result(page_dir, svg_content)
                    page_dirs.append(page_dir)

                root_markdown = self._build_svg_pdf_markdown(page_dirs)
                self._write_text_result(output_dir, root_markdown)
                return CaptionResult(
                    raw=root_markdown,
                    metadata={
                        "provider": self.name,
                        "output_dir": str(output_dir),
                        "runtime_backend": runtime.mode,
                        "runtime_model_id": runtime.model_id,
                        "prompt_mode": prompt_mode,
                    },
                )

            page_contents: list[str] = []
            total_layout_elements = 0
            any_filtered = False
            for index, pil_image in enumerate(images, start=1):
                page_dir = output_dir / f"page_{index:04d}"
                page_image_path = page_dir / f"page_{index:04d}.png"
                try:
                    saved_image_path = _save_pdf_page_image(pil_image, page_image_path)
                except Exception:
                    continue
                page_content = self._complete_via_openai_runtime(
                    image_path=str(saved_image_path),
                    prompt_mode=prompt_mode,
                    prompt_text=prompt_text,
                    fitz_preprocess=False,
                    dpi=dpi,
                )
                input_image = self._prepare_inference_image(
                    pil_image,
                    fitz_preprocess=False,
                    dpi=dpi,
                    log=False,
                )
                finalized_content, extra_metadata = self._finalize_text_result(
                    output_dir=page_dir,
                    prompt_mode=prompt_mode,
                    raw_content=page_content,
                    origin_image=pil_image,
                    input_image=input_image,
                )
                total_layout_elements += int(extra_metadata.get("layout_elements", 0))
                any_filtered = any_filtered or bool(extra_metadata.get("filtered"))
                page_contents.append(str(finalized_content).strip())

            content = "\n<--- Page Split --->\n".join(page_contents)
            self._write_text_result(output_dir, content)
            self._display_text_result(media, content)
            return CaptionResult(
                raw=content,
                metadata={
                    "provider": self.name,
                    "output_dir": str(output_dir),
                    "runtime_backend": runtime.mode,
                    "runtime_model_id": runtime.model_id,
                    "prompt_mode": prompt_mode,
                    "layout_elements": total_layout_elements,
                    "filtered": any_filtered,
                },
            )

        content = self._complete_via_openai_runtime(
            image_path=media.uri,
            prompt_mode=prompt_mode,
            prompt_text=prompt_text,
            fitz_preprocess=fitz_preprocess,
            dpi=dpi,
        )
        if prompt_mode == "prompt_image_to_svg":
            self._write_svg_result(output_dir, content)
            metadata = {}
        else:
            origin_image = _load_input_image(media.uri)
            input_image = self._prepare_inference_image(
                origin_image,
                fitz_preprocess=fitz_preprocess,
                dpi=dpi,
                log=False,
            )
            content, metadata = self._finalize_text_result(
                output_dir=output_dir,
                prompt_mode=prompt_mode,
                raw_content=content,
                origin_image=origin_image,
                input_image=input_image,
            )
            self._display_text_result(media, content)
        return CaptionResult(
            raw=str(content),
            metadata={
                "provider": self.name,
                "output_dir": str(output_dir),
                "runtime_backend": runtime.mode,
                "runtime_model_id": runtime.model_id,
                "prompt_mode": prompt_mode,
                **metadata,
            },
        )

    def _display_text_result(self, media: MediaContext, content: str) -> None:
        try:
            display_markdown(
                title=Path(media.uri).name,
                markdown_content=content,
                pixels=media.pixels,
                panel_height=32,
                console=self.ctx.console,
            )
        except Exception:
            pass

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        prompt_mode, resolved_prompt = self._resolve_prompt_mode_and_prompt()
        prompt_text = prompts.user or resolved_prompt
        model_id = self._select_model_id(prompt_mode)
        fitz_preprocess, dpi = self._resolve_image_preprocess_settings()
        output_dir = Path(media.extras.get("output_dir") or Path(media.uri).with_suffix(""))
        output_dir.mkdir(parents=True, exist_ok=True)
        self.ctx.console.print(f"dots_ocr prompt_mode: {prompt_mode}")
        self.ctx.console.print(f"dots_ocr prompt: {prompt_text}")

        if self.get_runtime_backend().is_openai:
            return self._attempt_via_openai_backend(media, prompt_mode, prompt_text, fitz_preprocess=fitz_preprocess, dpi=dpi)

        max_new_tokens = int(
            self._get_model_config("max_new_tokens", UPSTREAM_DIRECT_MAX_NEW_TOKENS)
            or UPSTREAM_DIRECT_MAX_NEW_TOKENS
        )
        if media.mime.startswith("application/pdf"):
            stage_timer = self._log_stage_start("upstream_pdf_to_images", f"file={Path(media.uri).name}")
            images = _load_upstream_pdf_images(media.uri)
            self._log_stage_done("upstream_pdf_to_images", stage_timer)
            if prompt_mode == "prompt_image_to_svg":
                page_dirs: list[Path] = []
                for index, pil_image in enumerate(images, start=1):
                    page_dir = output_dir / f"page_{index:04d}"
                    page_image_path = page_dir / f"page_{index:04d}.png"
                    try:
                        saved_image_path = _save_pdf_page_image(pil_image, page_image_path)
                    except Exception:
                        continue
                    svg_content = self._run_direct_generation(
                        image_path=str(saved_image_path),
                        prompt_mode=prompt_mode,
                        prompt_text=prompt_text,
                        model_id=model_id,
                        max_new_tokens=max_new_tokens,
                        fitz_preprocess=False,
                        dpi=dpi,
                    )
                    self._write_svg_result(page_dir, svg_content)
                    page_dirs.append(page_dir)

                root_markdown = self._build_svg_pdf_markdown(page_dirs)
                self._write_text_result(output_dir, root_markdown)
                return CaptionResult(
                    raw=root_markdown,
                    metadata={
                        "provider": self.name,
                        "output_dir": str(output_dir),
                        "prompt_mode": prompt_mode,
                        "model_id": model_id,
                    },
                )

            page_contents: list[str] = []
            total_layout_elements = 0
            any_filtered = False
            for index, pil_image in enumerate(images, start=1):
                page_dir = output_dir / f"page_{index:04d}"
                page_image_path = page_dir / f"page_{index:04d}.png"
                try:
                    saved_image_path = _save_pdf_page_image(pil_image, page_image_path)
                except Exception:
                    continue
                page_content = self._run_direct_generation(
                    image_path=str(saved_image_path),
                    prompt_mode=prompt_mode,
                    prompt_text=prompt_text,
                    model_id=model_id,
                    max_new_tokens=max_new_tokens,
                    fitz_preprocess=False,
                    dpi=dpi,
                )
                input_image = self._prepare_inference_image(
                    pil_image,
                    fitz_preprocess=False,
                    dpi=dpi,
                    log=False,
                )
                finalized_content, extra_metadata = self._finalize_text_result(
                    output_dir=page_dir,
                    prompt_mode=prompt_mode,
                    raw_content=page_content,
                    origin_image=pil_image,
                    input_image=input_image,
                )
                total_layout_elements += int(extra_metadata.get("layout_elements", 0))
                any_filtered = any_filtered or bool(extra_metadata.get("filtered"))
                page_contents.append(str(finalized_content).strip())

            content = "\n<--- Page Split --->\n".join(page_contents)
            self._write_text_result(output_dir, content)
            self._display_text_result(media, content)
            return CaptionResult(
                raw=content,
                metadata={
                    "provider": self.name,
                    "output_dir": str(output_dir),
                    "prompt_mode": prompt_mode,
                    "model_id": model_id,
                    "layout_elements": total_layout_elements,
                    "filtered": any_filtered,
                },
            )

        content = self._run_direct_generation(
            image_path=media.uri,
            prompt_mode=prompt_mode,
            prompt_text=prompt_text,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            fitz_preprocess=fitz_preprocess,
            dpi=dpi,
        )
        if prompt_mode == "prompt_image_to_svg":
            self._write_svg_result(output_dir, content)
            metadata = {}
        else:
            origin_image = _load_input_image(media.uri)
            input_image = self._prepare_inference_image(
                origin_image,
                fitz_preprocess=fitz_preprocess,
                dpi=dpi,
                log=False,
            )
            content, metadata = self._finalize_text_result(
                output_dir=output_dir,
                prompt_mode=prompt_mode,
                raw_content=content,
                origin_image=origin_image,
                input_image=input_image,
            )
            self._display_text_result(media, content)
        return CaptionResult(
            raw=str(content),
            metadata={
                "provider": self.name,
                "output_dir": str(output_dir),
                "prompt_mode": prompt_mode,
                "model_id": model_id,
                **metadata,
            },
        )
