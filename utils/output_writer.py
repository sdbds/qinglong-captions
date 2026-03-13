from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from utils.path_safety import safe_child_path, safe_sibling_path


def _caption_extension_for_mime(mime: str) -> str:
    if mime.startswith("video") or mime.startswith("audio"):
        return ".srt"
    if mime.startswith("application"):
        return ".md"
    return ".txt"


def caption_output_path(source_path: Path, mime: str) -> Path:
    return safe_sibling_path(source_path, _caption_extension_for_mime(mime))


def _structured_description(payload: dict) -> str:
    return (
        payload.get("long_description")
        or payload.get("description")
        or payload.get("short_description")
        or "No description available"
    )


def write_markdown_output(output_dir: Path, content: str, filename: str = "result.md") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = safe_child_path(output_dir, filename, default_name="result.md")
    target.write_text(content, encoding="utf-8")
    return target


def write_caption_output(source_path: Path, output, mime: str) -> tuple[Path, Optional[Path]]:
    source_path = Path(source_path)
    text_path = caption_output_path(source_path, mime)
    json_path: Optional[Path] = None

    if isinstance(output, dict):
        json_path = safe_sibling_path(source_path, ".json")
        json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        text_path.write_text(_structured_description(output), encoding="utf-8")
        return text_path, json_path

    if isinstance(output, list):
        text = "\n".join(str(line) for line in output)
    else:
        text = str(output)

    text_path.write_text(text, encoding="utf-8")
    return text_path, json_path
