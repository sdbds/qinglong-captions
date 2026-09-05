from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from utils.path_safety import safe_child_path, safe_sibling_path

if TYPE_CHECKING:
    from module.providers.base import CaptionResult


def _is_caption_result(value: object) -> bool:
    return (
        hasattr(value, "raw")
        and hasattr(value, "parsed")
        and hasattr(value, "metadata")
        and hasattr(value, "caption_extension")
    )


def _caption_extension_for_mime(mime: str) -> str:
    if mime.startswith("video") or mime.startswith("audio"):
        return ".srt"
    if mime.startswith("application"):
        return ".md"
    return ".txt"


def normalize_caption_extension(extension: object) -> Optional[str]:
    if extension is None:
        return None
    value = str(extension).strip()
    if not value:
        return None
    if not value.startswith("."):
        value = f".{value}"
    return value


def caption_extension_from_payload(payload: object) -> Optional[str]:
    if _is_caption_result(payload):
        return payload.caption_extension
    if not isinstance(payload, dict):
        return None
    return normalize_caption_extension(payload.get("caption_extension"))


def caption_output_path(source_path: Path, mime: str, output=None) -> Path:
    extension = caption_extension_from_payload(output) or _caption_extension_for_mime(mime)
    return safe_sibling_path(source_path, extension)


def caption_text(output, *, fallback: str = "") -> str:
    """Render semantic caption text consistently at generation and export boundaries."""
    if _is_caption_result(output):
        return caption_text(output.parsed, fallback=output.raw) if output.parsed is not None else output.raw
    if isinstance(output, dict):
        keys = ("long_description", "transcript", "translation_srt", "description", "short_description", "markdown", "text")
        task_kind = str(output.get("task_kind") or "").strip().lower()
        subtitle_format = str(output.get("subtitle_format") or "").strip().lower()
        if task_kind == "ast" or subtitle_format == "srt" or caption_extension_from_payload(output) == ".srt":
            keys = ("translation_srt", "transcript", "description", "long_description", "short_description", "markdown", "text")
        for key in keys:
            value = output.get(key)
            if str(value or "").strip():
                return str(value)
        return fallback
    if isinstance(output, list):
        return "\n".join(str(line) for line in output)
    return str(output or "")


def has_meaningful_text_content(content: object) -> bool:
    return bool(str(content).strip())


def write_markdown_output(output_dir: Path, content: str, filename: str = "result.md") -> Optional[Path]:
    if not has_meaningful_text_content(content):
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = safe_child_path(output_dir, filename, default_name="result.md")
    target.write_text(content, encoding="utf-8")
    return target


def write_caption_output(source_path: Path, output, mime: str) -> tuple[Path, Optional[Path]]:
    source_path = Path(source_path)
    text_path = caption_output_path(source_path, mime, output)
    json_path: Optional[Path] = None

    if _is_caption_result(output):
        if output.parsed is not None:
            json_path = safe_sibling_path(source_path, ".json")
            json_path.write_text(json.dumps(output.parsed, indent=2, ensure_ascii=False), encoding="utf-8")
            text_path.write_text(caption_text(output), encoding="utf-8")
        else:
            text_path.write_text(output.raw, encoding="utf-8")
        return text_path, json_path

    if isinstance(output, dict):
        json_path = safe_sibling_path(source_path, ".json")
        json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        text_path.write_text(caption_text(output), encoding="utf-8")
        return text_path, json_path

    if isinstance(output, list):
        text = "\n".join(str(line) for line in output)
    else:
        text = str(output)

    text_path.write_text(text, encoding="utf-8")
    return text_path, json_path
