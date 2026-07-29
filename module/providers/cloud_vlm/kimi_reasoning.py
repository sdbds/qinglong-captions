"""Shared Kimi K3 reasoning-effort configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

KIMI_OPEN_PLATFORM_K3_MODEL_IDS = frozenset({"kimi-k3"})
KIMI_CODE_K3_MODEL_IDS = frozenset({"k3", "k3-256k"})
K3_MODEL_IDS = KIMI_OPEN_PLATFORM_K3_MODEL_IDS | KIMI_CODE_K3_MODEL_IDS
DEFAULT_KIMI_MODEL_ID = "kimi-k3"
DEFAULT_KIMI_CODE_MODEL_ID = "k3-256k"
DEFAULT_K3_REASONING_EFFORT = "high"
SUPPORTED_K3_REASONING_EFFORTS = frozenset({"low", "high", "max"})
DEFAULT_KIMI_THINKING = "enabled"
SUPPORTED_KIMI_THINKING = frozenset({"enabled", "disabled"})


def is_k3_model(model_path: object) -> bool:
    return str(model_path or "").strip().lower() in K3_MODEL_IDS


def normalize_k3_reasoning_effort(
    value: object,
    default: str = DEFAULT_K3_REASONING_EFFORT,
) -> str:
    effort = str(default if value is None else value).strip().lower()
    if effort not in SUPPORTED_K3_REASONING_EFFORTS:
        expected = ", ".join(sorted(SUPPORTED_K3_REASONING_EFFORTS))
        raise ValueError(f"Unsupported K3 reasoning effort: {value}. Expected one of: {expected}")
    return effort


def configured_k3_reasoning_effort(config: Mapping[str, Any], section: str) -> str:
    section_config = config.get(section, {})
    if not isinstance(section_config, Mapping):
        raise ValueError(f"K3 configuration section [{section}] must be a table")
    return normalize_k3_reasoning_effort(section_config.get("reasoning_effort"))


def normalize_kimi_thinking(
    value: object,
    default: str = DEFAULT_KIMI_THINKING,
) -> str:
    thinking = str(default if value is None else value).strip().lower()
    if thinking not in SUPPORTED_KIMI_THINKING:
        expected = ", ".join(sorted(SUPPORTED_KIMI_THINKING))
        raise ValueError(f"Unsupported Kimi thinking mode: {value}. Expected one of: {expected}")
    return thinking


def configured_kimi_thinking(config: Mapping[str, Any], section: str) -> str:
    section_config = config.get(section, {})
    if not isinstance(section_config, Mapping):
        raise ValueError(f"Kimi configuration section [{section}] must be a table")
    return normalize_kimi_thinking(section_config.get("thinking"))
