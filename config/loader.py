"""Unified config loader for split TOML configuration files.

Loads prompts.toml, model.toml, general.toml, and onnx.toml from the config
directory and merges them into a single dict, preserving the same structure as
the original monolithic config.toml.
"""

from pathlib import Path
from typing import Any, Dict

import toml

_CONFIG_FILES = ["prompts.toml", "model.toml", "general.toml", "onnx.toml"]


def load_config(config_dir: str = "config") -> Dict[str, Any]:
    """Load and merge split TOML config files into one dict.

    Falls back to legacy config.toml if the split files don't exist.

    Args:
        config_dir: Path to the directory containing the TOML config files.

    Returns:
        Merged configuration dictionary.
    """
    config_path = Path(config_dir)
    merged: Dict[str, Any] = {}

    # Try split files first
    split_found = False
    for name in _CONFIG_FILES:
        path = config_path / name
        if path.exists():
            split_found = True
            data = toml.load(path)
            merged.update(data)

    # Fall back to legacy config.toml if no split files found
    if not split_found:
        legacy = config_path / "config.toml"
        if legacy.exists():
            merged = toml.load(legacy)

    return merged
