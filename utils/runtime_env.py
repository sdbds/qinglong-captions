from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RUNTIME_CACHE_ENV_KEYS = (
    "VIRTUAL_ENV",
    "CUDA_VISIBLE_DEVICES",
    "PYTHONPATH",
)


def project_root_from(work_dir: str | Path | None = None) -> Path:
    if work_dir is None:
        return PROJECT_ROOT
    return Path(work_dir).resolve()


def _load_gui_env_overrides() -> dict[str, str]:
    try:
        from gui.utils.env_config import get_env_for_subprocess
    except Exception:
        return {}

    try:
        return get_env_for_subprocess()
    except Exception:
        return {}


def build_runtime_env(
    work_dir: str | Path | None = None,
    *,
    base_env: Mapping[str, str] | None = None,
    env_overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(base_env) if base_env is not None else os.environ.copy()
    env.update(_load_gui_env_overrides())
    if env_overrides:
        env.update({str(key): str(value) for key, value in env_overrides.items() if value is not None})

    project_root = str(project_root_from(work_dir))
    existing = env.get("PYTHONPATH", "")
    paths = [part for part in existing.split(os.pathsep) if part]
    if project_root not in paths:
        env["PYTHONPATH"] = project_root + (os.pathsep + existing if existing else "")

    return env


def resolve_runtime_python(work_dir: str | Path | None = None, env: Mapping[str, str] | None = None) -> str | None:
    work_path = project_root_from(work_dir)
    runtime_env = env or {}
    candidates: list[Path] = []

    for name in (".venv", "venv"):
        base = work_path / name
        if sys.platform == "win32":
            candidates.append(base / "Scripts" / "python.exe")
        else:
            candidates.append(base / "bin" / "python")

    venv_path = runtime_env.get("VIRTUAL_ENV")
    if venv_path:
        venv_base = Path(venv_path)
        if sys.platform == "win32":
            candidates.append(venv_base / "Scripts" / "python.exe")
        else:
            candidates.append(venv_base / "bin" / "python")

    candidates.append(Path(sys.executable))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def runtime_probe_cache_key(
    work_dir: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    *,
    python_path: str | None = None,
) -> tuple[str, ...]:
    work_path = str(project_root_from(work_dir))
    runtime_env = env or {}
    runtime_python = python_path or resolve_runtime_python(work_dir, runtime_env) or ""
    key = [work_path, runtime_python]
    for env_key in _RUNTIME_CACHE_ENV_KEYS:
        key.append(f"{env_key}={runtime_env.get(env_key, '')}")
    return tuple(key)
