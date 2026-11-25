"""Provision isolated caption runtimes without modifying the shared environment."""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
from pathlib import Path

_PROFILES = {
    "deepseek-ocr": "deepseek-ocr-native",
    "deepseek-ocr-native": "deepseek-ocr-native",
    "deepseek-ocr-legacy": "deepseek-ocr-legacy",
    "moondream-photon": "moondream-photon",
}


def runtime_profile(extras: list[str]) -> str | None:
    profiles = {_PROFILES[name] for name in extras if name in _PROFILES}
    if sys.platform == "win32" and "moondream" in extras:
        profiles.add("moondream-photon")
    if len(profiles) > 1:
        raise ValueError(f"Incompatible caption runtimes require separate runs: {', '.join(sorted(profiles))}")
    return next(iter(profiles), None)


def runtime_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def uses_project_env(root: Path, python: str | None, venv: str | None) -> bool:
    shared = {(root / name).resolve() for name in (".venv", "venv")}
    if venv and Path(venv).expanduser().resolve() not in shared:
        return False
    # Resolve the directory, not the executable: a POSIX venv python is often a symlink.
    return not python or Path(python).expanduser().parent.resolve() in {runtime_python(p).parent for p in shared}


def _physical_dir(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISDIR(info.st_mode) and not getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT


def _physical_file(path: Path) -> bool:
    info = path.lstat()
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_nlink == 1
        and not getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def _matches(path: Path, expected: bytes) -> bool:
    try:
        return _physical_file(path) and path.stat().st_size == len(expected) and path.read_bytes() == expected
    except FileNotFoundError:
        return False


def _validate_python(venv: Path) -> Path:
    python = runtime_python(venv)
    cfg = venv / "pyvenv.cfg"
    if not (_physical_file(cfg) and _physical_dir(python.parent) and python.is_file()):
        raise OSError(f"Invalid caption virtual environment: {venv}")
    settings = dict(
        (k.strip(), v.strip())
        for line in cfg.read_text(encoding="utf-8").splitlines()
        for k, sep, v in [line.partition("=")]
        if sep
    )
    version = settings.get("version_info") or settings.get("version") or ""
    if version.split(".")[:2] != ["3", "11"] or settings.get("include-system-site-packages", "").lower() != "false":
        raise OSError(f"Expected an isolated Python 3.11 caption environment at {venv}")
    return python


def ensure_runtime(project: Path, extra: str, *, uv: str = "uv") -> Path:
    from filelock import FileLock

    name = _PROFILES[extra]
    parent = project.resolve() / ".runtime_venvs"
    parent.mkdir(mode=0o700, exist_ok=True)
    if not _physical_dir(parent):
        raise OSError(f"Caption runtime root is not a physical directory: {parent}")
    target = parent / name
    lock = parent / f".{name}.create.lock"
    try:
        with lock.open("xb"):
            pass
    except FileExistsError:
        # A zero-length regular file needs no content read. Windows denies
        # reading the locked byte while another launcher is creating the venv.
        if not _physical_file(lock) or lock.lstat().st_size != 0:
            raise OSError(f"Unowned caption runtime lock: {lock}") from None
    with FileLock(str(lock), timeout=180, preserve_lock_file=True):
        owner = target / (".qinglong-photon-v1" if name == "moondream-photon" else ".qinglong-deepseek-v1")
        ready = target / ".ready-v1"
        identity = f"qinglong-captions:{name}\n".encode("ascii")
        try:
            target.mkdir(mode=0o700)
        except FileExistsError:
            if not _physical_dir(target) or not _matches(owner, identity):
                raise OSError(f"Caption runtime path is unowned; choose a different location for its contents: {target}") from None
            if _matches(ready, identity):
                return _validate_python(target)
            if any(entry != owner for entry in target.iterdir()):
                raise OSError(f"Incomplete caption environment. Move {target} aside before trying again.")
        else:
            with owner.open("xb") as stream:
                stream.write(identity)

        # --allow-existing preserves the exclusive owner marker; never clear or adopt a prior venv.
        env = dict(os.environ)
        env.pop("UV_VENV_CLEAR", None)
        env.pop("VIRTUAL_ENV", None)
        subprocess.run(
            [uv, "venv", "--no-project", "--python", "3.11", "--seed", "--allow-existing", str(target)],
            check=True,
            stdout=sys.stderr,
            env=env,
        )
        python = _validate_python(target)
        with ready.open("xb") as stream:
            stream.write(identity)
        return python


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--extra", required=True, choices=_PROFILES)
    parser.add_argument("--uv", default="uv")
    args = parser.parse_args()
    print(ensure_runtime(args.root, args.extra, uv=args.uv))
