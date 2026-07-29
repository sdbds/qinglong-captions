"""Format-preserving GUI access to Kimi reasoning-effort settings."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import tomlkit

from module.providers.cloud_vlm.kimi_reasoning import (
    normalize_k3_reasoning_effort,
    normalize_kimi_thinking,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model.toml"
LEGACY_CONFIG_PATH = PROJECT_ROOT / "config" / "config.toml"
KIMI_CONFIG_SECTIONS = frozenset({"kimi_vl", "kimi_code"})


def _parse_config(path: Path) -> Any:
    return tomlkit.parse(path.read_text(encoding="utf-8"))


def _require_section(document: Any, section: str, path: Path) -> Any:
    if section not in KIMI_CONFIG_SECTIONS:
        raise ValueError(f"Unsupported Kimi configuration section: {section}")
    if section not in document:
        raise ValueError(f"Missing [{section}] section in {path}")
    return document[section]


def _stage_file(path: Path, content: bytes) -> Path:
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copymode(path, temp_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def _replace_file(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _load_kimi_setting(
    section: str,
    key: str,
    normalizer: Callable[[object], str],
    *,
    config_path: Path = MODEL_CONFIG_PATH,
) -> str:
    path = Path(config_path)
    document = _parse_config(path)
    section_config = _require_section(document, section, path)
    return normalizer(section_config.get(key))


def _save_kimi_setting(
    section: str,
    key: str,
    value: object,
    normalizer: Callable[[object], str],
    *,
    model_config_path: Path = MODEL_CONFIG_PATH,
    legacy_config_path: Path = LEGACY_CONFIG_PATH,
) -> str:
    normalized_value = normalizer(value)
    paths = (Path(model_config_path), Path(legacy_config_path))
    original_contents = [path.read_bytes() for path in paths]
    documents = [tomlkit.parse(content.decode("utf-8")) for content in original_contents]
    sections = [_require_section(document, section, path) for document, path in zip(documents, paths)]

    for section_config in sections:
        section_config[key] = normalized_value

    rendered = [tomlkit.dumps(document).encode("utf-8") for document in documents]
    staged_files: list[Path] = []
    rollback_files: list[Path] = []
    committed_indices: list[int] = []
    try:
        # Prepare both new values and rollback copies before replacing either live file.
        for path, content in zip(paths, rendered):
            staged_files.append(_stage_file(path, content))
        for path, content in zip(paths, original_contents):
            rollback_files.append(_stage_file(path, content))

        try:
            for index, path in enumerate(paths):
                _replace_file(staged_files[index], path)
                committed_indices.append(index)
        except Exception as write_error:
            rollback_errors: list[str] = []
            for index in reversed(committed_indices):
                try:
                    _replace_file(rollback_files[index], paths[index])
                except Exception as rollback_error:
                    rollback_errors.append(f"{paths[index]}: {rollback_error}")
            if rollback_errors:
                details = "; ".join(rollback_errors)
                raise RuntimeError(
                    f"Failed to keep Kimi configuration files synchronized; rollback failed: {details}"
                ) from write_error
            raise
    finally:
        for temp_path in (*staged_files, *rollback_files):
            temp_path.unlink(missing_ok=True)

    return normalized_value


def load_kimi_reasoning_effort(
    section: str,
    *,
    config_path: Path = MODEL_CONFIG_PATH,
) -> str:
    return _load_kimi_setting(
        section,
        "reasoning_effort",
        normalize_k3_reasoning_effort,
        config_path=config_path,
    )


def save_kimi_reasoning_effort(
    section: str,
    value: object,
    *,
    model_config_path: Path = MODEL_CONFIG_PATH,
    legacy_config_path: Path = LEGACY_CONFIG_PATH,
) -> str:
    return _save_kimi_setting(
        section,
        "reasoning_effort",
        value,
        normalize_k3_reasoning_effort,
        model_config_path=model_config_path,
        legacy_config_path=legacy_config_path,
    )


def load_kimi_thinking(
    section: str,
    *,
    config_path: Path = MODEL_CONFIG_PATH,
) -> str:
    return _load_kimi_setting(
        section,
        "thinking",
        normalize_kimi_thinking,
        config_path=config_path,
    )


def save_kimi_thinking(
    section: str,
    value: object,
    *,
    model_config_path: Path = MODEL_CONFIG_PATH,
    legacy_config_path: Path = LEGACY_CONFIG_PATH,
) -> str:
    return _save_kimi_setting(
        section,
        "thinking",
        value,
        normalize_kimi_thinking,
        model_config_path=model_config_path,
        legacy_config_path=legacy_config_path,
    )
