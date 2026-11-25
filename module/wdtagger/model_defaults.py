"""Dependency-free model IDs and defaults shared by the GUI and tagger runtimes."""

from __future__ import annotations

import re

CL_TAGGER_V2_OPTION = "cella110n/cl_tagger_v2"
CL_TAGGER_V2_BACKEND_REPO = CL_TAGGER_V2_OPTION
CL_TAGGER_V2_LEGACY_BACKEND_REPO = "celstk/cl-SigLIP2-lora-onnx"
CL_TAGGER_V2_DEFAULT_VERSION = "v2_01a"
CL_TAGGER_V2_FALLBACK_THRESHOLD = 0.5
CL_TAGGER_V2_THRESHOLD_OVERRIDES = {
    "v1_00": 0.6,
    "v1_01": 0.6,
    "v1_02": 0.9,
    "v2_00": 0.55,
    "v2_01a": 0.55,
}

PIXAI_REPO_ID = "bdsqlsz/pixai-tagger-v1.0-ONNX"
PIXAI_SOURCE_REPO_ID = "pixai-labs/pixai-tagger-v1.0"
PIXAI_THRESHOLDS = {"general": 0.17, "character": 0.27, "style": 0.15, "copyright": 0.24, "meta": 0.17, "rating": 0.41}

HARUKA_REPO_ID = "KaraKaraWitch/haruka-tagger-large"
HARUKA_THRESHOLDS = {"general": 0.35, "character": 0.35, "copyright": 0.35, "artist": 0.25, "year": 0.25, "rating": 0.50}

_VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)([a-z]+)?$")


class LogitsInputName(str):
    """An ONNX feed name for a legacy tagger that returns logits, not probabilities."""


def is_pixai_repo(repo_id: str) -> bool:
    return str(repo_id).strip() in {PIXAI_REPO_ID, PIXAI_SOURCE_REPO_ID}


def is_haruka_repo(repo_id: str) -> bool:
    return str(repo_id).strip() == HARUKA_REPO_ID


def model_category_defaults(repo_id: str) -> dict[str, float]:
    if is_haruka_repo(repo_id):
        return HARUKA_THRESHOLDS
    if is_pixai_repo(repo_id):
        return PIXAI_THRESHOLDS
    return {}


def is_cl_tagger_v2_repo(repo_id: str) -> bool:
    normalized = str(repo_id or "").strip()
    return normalized in {CL_TAGGER_V2_OPTION, CL_TAGGER_V2_BACKEND_REPO, CL_TAGGER_V2_LEGACY_BACKEND_REPO}


def normalize_cl_tagger_v2_version(version: str | None) -> str:
    value = str(version or CL_TAGGER_V2_DEFAULT_VERSION).strip()
    if not value:
        return CL_TAGGER_V2_DEFAULT_VERSION

    normalized = value.lower()
    if normalized.startswith("v"):
        normalized = normalized[1:]
    normalized = normalized.replace("_", ".")
    match = _VERSION_PATTERN.fullmatch(normalized)
    if match:
        major, minor, suffix = match.groups()
        minor_width = 3 if len(minor) > 2 else 2
        return f"v{int(major)}_{int(minor):0{minor_width}d}{suffix or ''}"
    return value


def default_cl_tagger_v2_threshold(version: str | None = None) -> float:
    normalized = normalize_cl_tagger_v2_version(version)
    return CL_TAGGER_V2_THRESHOLD_OVERRIDES.get(normalized, CL_TAGGER_V2_FALLBACK_THRESHOLD)
