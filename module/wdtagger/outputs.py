from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from module.wdtagger import constants
from utils.console_util import print_exception


def has_sidecar_caption(uri: str, caption_extension: str) -> bool:
    caption_path = Path(uri).with_suffix(caption_extension)
    if not caption_path.exists():
        return False

    try:
        return bool(caption_path.read_text(encoding="utf-8").strip())
    except OSError:
        return False


def read_sidecar_caption(uri: str, caption_extension: str) -> List[str]:
    caption_path = Path(uri).with_suffix(caption_extension)
    if not caption_path.exists():
        return []

    try:
        content = caption_path.read_text(encoding="utf-8")
    except OSError:
        return []

    return content.splitlines() if caption_extension.lower() == ".txt" else [content]


def write_sidecar_caption(uri: str, captions: List[str], *, caption_extension: str, caption_separator: str) -> None:
    output_path = Path(uri).with_suffix(caption_extension)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(caption_separator.join(captions))


def write_tags_json(train_data_dir: str, all_json_tags: Dict[str, Dict[str, List[str]]]) -> None:
    try:
        json_output_path = Path(train_data_dir) / "tags.json"
        with json_output_path.open("w", encoding="utf-8") as jf:
            json.dump(all_json_tags, jf, ensure_ascii=False, indent=2)
        constants.console.print(f"[bold green]JSON saved to:[/bold green] {json_output_path}")
    except Exception as e:
        print_exception(constants.console, e, prefix="Failed to save JSON")
