from __future__ import annotations

from pathlib import Path


def safe_leaf_name(untrusted_name: str, default_name: str = "file") -> str:
    """Reduce an untrusted path-like string to a single safe filename."""
    raw_name = str(untrusted_name or "").strip().replace("\\", "/").rstrip("/")
    leaf_name = Path(raw_name).name
    if leaf_name in {"", ".", ".."}:
        return default_name
    return leaf_name


def safe_child_path(base_dir: Path, untrusted_name: str, default_name: str = "file") -> Path:
    """Build a child path constrained to base_dir."""
    base_dir = Path(base_dir).resolve()
    target = (base_dir / safe_leaf_name(untrusted_name, default_name=default_name)).resolve()
    try:
        target.relative_to(base_dir)
    except ValueError as exc:
        raise ValueError(f"Refusing to write outside base directory: {untrusted_name!r}") from exc
    return target
