"""Lightweight OCR protocol selection, shared by providers and launchers."""

from __future__ import annotations

import json
import re
from pathlib import Path

DEEPSEEK_LEGACY_MODEL = "deepseek-ai/DeepSeek-OCR-2"
DEEPSEEK_NATIVE_MODEL = "deepseek-community/DeepSeek-OCR-2"
MOONDREAM_PHOTON_MODEL = "moondream3.1-9B-A2B"
MOONDREAM_PHOTON_REPO = f"moondream/{MOONDREAM_PHOTON_MODEL}"


def is_moondream_photon(model_id: str) -> bool:
    return model_id in (MOONDREAM_PHOTON_MODEL, MOONDREAM_PHOTON_REPO)


def moondream_needs_spatial_tasks(tasks: str, *, image_ocr: bool) -> bool:
    return not image_ocr and any(task.strip() in {"all", "point", "detect"} for task in tasks.split(","))


def moondream_extra_from_config(
    config_dir: str | Path = Path(__file__).resolve().parents[1] / "config",
    *, backend: str = "", has_base_url: bool = False, image_ocr: bool = False,
) -> str | None:
    """Return the local install profile, or None for tasks handled by the configured server."""
    from config.loader import load_config

    section = load_config(str(config_dir)).get("moondream", {})
    remote = has_base_url or section.get("runtime_base_url")
    mode = backend or ("openai" if remote else section.get("runtime_backend", "direct"))
    spatial = moondream_needs_spatial_tasks(str(section.get("tasks", "caption")), image_ocr=image_ocr)
    if str(mode).strip().lower() == "openai" and not spatial:
        return None
    return "moondream" if is_moondream_photon(section.get("model_id", MOONDREAM_PHOTON_REPO)) else "moondream-legacy"


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


def _config_section(config_path: str | Path, name: str) -> dict:
    try:
        import tomllib
    except ImportError:  # The bootstrap launcher may still use Python 3.10.
        import toml as tomllib

    path = Path(config_path)
    config = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return config.get(name, {})


def unlimited_ocr_extra_from_config(config_path: str | Path) -> str:
    section = _config_section(config_path, "unlimited_ocr")
    return unlimited_ocr_extra(section.get("model_id", "baidu/Unlimited-OCR-2"), section.get("model_version", "auto"))


def deepseek_native_model_id(
    model_id: str, *, base_size: int = 1024, image_size: int = 768, crop_mode: bool = True,
) -> str | None:
    """Translate only the verified stock checkpoint; never substitute a remote model for a local directory."""
    path = Path(model_id).expanduser()
    if path.is_dir():
        config = path / "config.json"
        if config.is_file() and json.loads(config.read_text(encoding="utf-8")).get("model_type") == "deepseek_ocr2":
            return str(path.resolve())
        return None
    if model_id in (DEEPSEEK_LEGACY_MODEL, DEEPSEEK_NATIVE_MODEL) and (base_size, image_size, crop_mode) == (1024, 768, True):
        return DEEPSEEK_NATIVE_MODEL
    return None


def deepseek_ocr_extra(model_id: str, *, base_size: int = 1024, image_size: int = 768, crop_mode: bool = True) -> str:
    native_id = deepseek_native_model_id(model_id, base_size=base_size, image_size=image_size, crop_mode=crop_mode)
    return "deepseek-ocr" if native_id is not None else "deepseek-ocr-legacy"


def deepseek_ocr_extra_from_config(config_path: str | Path) -> str:
    section = _config_section(config_path, "deepseek_ocr")
    return deepseek_ocr_extra(
        section.get("model_id", DEEPSEEK_LEGACY_MODEL), base_size=section.get("base_size", 1024),
        image_size=section.get("image_size", 768), crop_mode=section.get("crop_mode", True),
    )
