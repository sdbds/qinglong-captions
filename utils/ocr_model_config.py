"""Lightweight OCR protocol selection, shared by providers and launchers."""

from __future__ import annotations

import json
import re
from pathlib import Path


def ocr_model_version(family: str, model_id: str, model_version: str | int = "auto") -> int:
    supported = {"unlimited": (1, 2), "logics": (2, 3), "lighton": (1, 2, 3)}
    if family not in supported:
        raise ValueError(f"Unknown OCR family: {family}")
    override = str(model_version or "auto").strip().lower()
    if override != "auto":
        if override.removeprefix("v") not in {str(v) for v in supported[family]}:
            raise ValueError(f"Unsupported {family} model_version: {model_version}")
        return int(override.removeprefix("v"))

    name = str(model_id).replace("\\", "/").lower()
    patterns = {
        "unlimited": r"unlimited[-_]ocr[-_]?(2)(?:\b|_)",
        "logics": r"logics[-_]parsing[-_]v?(3)(?:\b|_)",
        "lighton": r"lighton[-_]?ocr[-_](1|2|3)(?:[-_/]|$)",
    }
    match = re.search(patterns[family], name)
    if match:
        return int(match[1])
    config_path = Path(model_id) / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if family == "logics" and config.get("model_type") == "qwen3_5":
            return 3
        if family == "lighton" and config.get("model_type") == "qwen3_5":
            return 3
        if family == "unlimited" and "_ring_window" in config:
            return 2
        original_name = str(config.get("_name_or_path", "")).lower()
        match = re.search(patterns[family], original_name)
        if match:
            return int(match[1])
    # Unnamed checkpoints keep the pre-upgrade behavior; use an explicit override
    # for server aliases and renamed 1B checkpoints whose architecture is shared.
    return 1 if family == "unlimited" else 2


def unlimited_ocr_extra(model_id: str, model_version: str | int = "auto") -> str:
    return "unlimited-ocr2" if ocr_model_version("unlimited", model_id, model_version) == 2 else "unlimited-ocr"


def unlimited_ocr_extra_from_config(config_path: str | Path) -> str:
    try:
        import tomllib
    except ImportError:  # The bootstrap launcher may still use Python 3.10.
        import toml as tomllib

    path = Path(config_path)
    config = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    section = config.get("unlimited_ocr", {})
    return unlimited_ocr_extra(section.get("model_id", "baidu/Unlimited-OCR-2"), section.get("model_version", "auto"))
