from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from config.config import (
    DEFAULT_CONSOLE_COLORS,
    DEFAULT_DATASET_SCHEMA,
    load_colors_from_toml,
    load_prompts_from_toml,
    load_schema_from_toml,
)
from config.loader import load_config as load_merged_config


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class RuntimeConfig(Mapping[str, Any]):
    _merged: Mapping[str, Any]
    schema: tuple[Any, ...]
    colors: Mapping[str, Any]
    prompts: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self._merged[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._merged)

    def __len__(self) -> int:
        return len(self._merged)

    def get(self, key: str, default: Any = None) -> Any:
        return self._merged.get(key, default)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._merged)


def coerce_runtime_config(config_like: Any) -> RuntimeConfig:
    if isinstance(config_like, RuntimeConfig):
        return config_like

    merged = dict(config_like or {})
    prompts = merged.get("prompts", {})
    colors = merged.get("colors", DEFAULT_CONSOLE_COLORS)
    schema = tuple(merged.get("schema", {}).get("fields", DEFAULT_DATASET_SCHEMA))
    frozen_merged = _freeze(merged)
    return RuntimeConfig(
        _merged=frozen_merged,
        schema=tuple(schema),
        colors=_freeze(colors),
        prompts=_freeze(prompts),
    )


def load_runtime_config(config_dir: str = "config") -> RuntimeConfig:
    merged = load_merged_config(config_dir)

    try:
        schema = tuple(load_schema_from_toml(config_dir))
    except Exception:
        schema = tuple(DEFAULT_DATASET_SCHEMA)

    try:
        colors = load_colors_from_toml(config_dir)
    except Exception:
        colors = DEFAULT_CONSOLE_COLORS.copy()

    try:
        prompts = load_prompts_from_toml(config_dir)
    except Exception:
        prompts = {}

    if "prompts" not in merged and prompts:
        merged["prompts"] = prompts

    return RuntimeConfig(
        _merged=_freeze(merged),
        schema=tuple(schema),
        colors=_freeze(colors),
        prompts=_freeze(prompts),
    )
