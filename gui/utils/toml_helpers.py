"""Shared TOML utility functions for GUI components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import toml
from typing import TYPE_CHECKING

from module.providers.catalog import provider_config_sections

if TYPE_CHECKING:
    from module.gpu_profile import GPUProbeResult


_MODEL_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "model.toml"


@dataclass(frozen=True)
class CurrentModelFitAssessment:
    model_id: str
    status: str
    status_label: str
    source: str
    min_vram_gb: float | None = None


@dataclass(frozen=True)
class ModelListEntry:
    name: str
    model_id: str
    min_vram_gb: float | None = None


def unwrap(value):
    """Convert tomlkit wrapper types to plain Python types for JSON serialization."""
    if isinstance(value, dict):
        return {str(k): unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unwrap(v) for v in value]
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)
    return value


def guess_slider_range(key: str, value, is_float: bool):
    """Guess a reasonable slider min/max/step for a numeric config field."""
    if is_float:
        if value <= 0:
            return 0.0, 1.0, 0.01
        if value <= 1.0:
            return 0.0, 1.0, 0.01
        if value <= 2.0:
            return 0.0, 2.0, 0.05
        if value <= 10.0:
            return 0.0, 10.0, 0.1
        return 0.0, value * 3, 0.1
    else:
        if value <= 0:
            return 0, 100, 1
        if value <= 10:
            return 0, 20, 1
        if value <= 100:
            return 0, 200, 1
        if value <= 1000:
            return 0, 2000, 10
        if value <= 10000:
            return 0, 20000, 100
        return 0, value * 3, max(1, value // 100)


def load_model_toml(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path is not None else _MODEL_CONFIG_PATH
    if not path.exists():
        return {}
    return toml.load(path)


def find_model_config_section(route_name: str, *, config_path: str | Path | None = None) -> tuple[str, dict] | None:
    config = load_model_toml(config_path)
    for section_name in provider_config_sections(route_name):
        section = config.get(section_name)
        if isinstance(section, dict):
            return section_name, section
    return None


def load_model_list_entries(route_name: str, *, config_path: str | Path | None = None) -> tuple[ModelListEntry, ...]:
    result = find_model_config_section(route_name, config_path=config_path)
    if result is None:
        return ()

    _, section = result
    model_list = section.get("model_list")
    if not isinstance(model_list, dict):
        return ()

    entries: list[ModelListEntry] = []
    for name, raw_value in model_list.items():
        model_name = str(name)
        model_id = ""
        min_vram_gb: float | None = None

        if isinstance(raw_value, str):
            model_id = raw_value.strip()
        elif isinstance(raw_value, dict):
            model_id = str(raw_value.get("model_id", "") or "").strip()
            raw_meta = raw_value.get("meta")
            raw_min_vram_gb = None
            if isinstance(raw_meta, dict):
                raw_min_vram_gb = raw_meta.get("min_vram_gb")
            if raw_min_vram_gb in (None, ""):
                raw_min_vram_gb = raw_value.get("min_vram_gb")
            if raw_min_vram_gb not in (None, ""):
                try:
                    min_vram_gb = float(raw_min_vram_gb)
                except (TypeError, ValueError):
                    min_vram_gb = None

        if model_id:
            entries.append(ModelListEntry(name=model_name, model_id=model_id, min_vram_gb=min_vram_gb))

    return tuple(entries)


def load_model_id_options(
    route_name: str,
    *,
    current_model_id: str = "",
    config_path: str | Path | None = None,
) -> dict[str, str]:
    options = {entry.model_id: entry.model_id for entry in load_model_list_entries(route_name, config_path=config_path)}
    normalized_current = str(current_model_id or "").strip()
    if normalized_current and normalized_current not in options:
        options[normalized_current] = normalized_current
    return options


def load_current_route_model_ids(
    route_names: tuple[str, ...] | list[str],
    *,
    config_path: str | Path | None = None,
) -> dict[str, str]:
    route_model_ids: dict[str, str] = {}
    for route_name in route_names:
        result = find_model_config_section(route_name, config_path=config_path)
        if result is None:
            continue
        _, section = result
        model_id = str(section.get("model_id", "") or "").strip()
        if model_id:
            route_model_ids[route_name] = model_id
    return route_model_ids


def _available_vram_gb(probe: GPUProbeResult | None) -> float:
    if probe is None or not probe.cuda_available or probe.primary_device is None:
        return 0.0
    return float(probe.primary_device.total_vram_gb)


def assess_current_model_fit(
    route_name: str,
    *,
    current_model_id: str,
    probe: GPUProbeResult | None = None,
    config_path: str | Path | None = None,
) -> CurrentModelFitAssessment | None:
    normalized_model_id = str(current_model_id or "").strip()
    if not normalized_model_id:
        return None

    available_vram = _available_vram_gb(probe)
    for entry in load_model_list_entries(route_name, config_path=config_path):
        if entry.model_id != normalized_model_id:
            continue
        if entry.min_vram_gb is None:
            return CurrentModelFitAssessment(
                model_id=normalized_model_id,
                status="unknown",
                status_label="Current model_id is in model_list but missing min_vram_gb metadata",
                source="missing_meta",
            )
        if available_vram < entry.min_vram_gb:
            return CurrentModelFitAssessment(
                model_id=normalized_model_id,
                status="warning",
                status_label=f"Needs >= {entry.min_vram_gb:.0f} GB VRAM, available {available_vram:.1f} GB",
                source="model_list",
                min_vram_gb=entry.min_vram_gb,
            )
        return CurrentModelFitAssessment(
            model_id=normalized_model_id,
            status="ok",
            status_label=f"Needs >= {entry.min_vram_gb:.0f} GB VRAM, available {available_vram:.1f} GB",
            source="model_list",
            min_vram_gb=entry.min_vram_gb,
        )

    return CurrentModelFitAssessment(
        model_id=normalized_model_id,
        status="unknown",
        status_label="Current model_id is not in model_list and has no min_vram_gb metadata",
        source="unknown",
    )
