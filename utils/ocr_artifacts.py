"""Filesystem ownership for OCR outputs which are not dataset inputs."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from pathlib import Path
from threading import Lock
from urllib.parse import quote


OCR_ARTIFACT_ROOT = ".qinglong-ocr-artifacts"
_OWNER_FILE = ".owner-v1"
_OWNER_CONTENT = b"qinglong-captions:ocr\n"
_root_lock = Lock()


def is_ocr_artifact_path(path: Path) -> bool:
    return any(part.casefold() == OCR_ARTIFACT_ROOT for part in Path(os.path.normpath(path)).parts)


def _is_physical(path_stat: os.stat_result) -> bool:
    return not (getattr(path_stat, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def require_ocr_artifact_root(root: Path) -> None:
    """Verify a physical, project-owned OCR root without changing it."""
    marker = root / _OWNER_FILE
    root_stat = root.lstat()
    if not stat.S_ISDIR(root_stat.st_mode) or not _is_physical(root_stat):
        raise OSError(f"OCR artifact path is not a physical directory: {root}")
    try:
        marker_stat = marker.lstat()
    except FileNotFoundError:
        marker_stat = None
    if (marker_stat is None or not stat.S_ISREG(marker_stat.st_mode)
            or not _is_physical(marker_stat) or marker_stat.st_nlink != 1
            or marker_stat.st_size != len(_OWNER_CONTENT) or marker.read_bytes() != _OWNER_CONTENT):
        raise OSError(f"OCR artifact directory is not owned by Qinglong Captions: {root}")


def ensure_ocr_artifact_root(root: Path) -> None:
    with _root_lock:
        try:
            root.mkdir(mode=0o700)
        except FileExistsError:
            require_ocr_artifact_root(root)
        else:
            with (root / _OWNER_FILE).open("xb") as marker_file:
                marker_file.write(_OWNER_CONTENT)


def ocr_artifact_run_prefix(source: Path) -> str:
    return hashlib.sha256(os.fsencode(source.name)).hexdigest()[:16] + "-"


def reserve_ocr_artifact_dir(source: str | Path) -> Path:
    source = Path(source)
    parent = source.parent.resolve()
    if is_ocr_artifact_path(source.absolute()) or is_ocr_artifact_path(parent):
        raise OSError(f"OCR artifact cannot be used as a source: {source}")
    root = parent / OCR_ARTIFACT_ROOT
    ensure_ocr_artifact_root(root)
    # Never reuse a run directory: an older sibling caption may still link to it.
    return Path(tempfile.mkdtemp(prefix=ocr_artifact_run_prefix(source), dir=root))


def relative_artifact_url(path: Path, caption_dir: Path) -> str:
    try:
        relative = Path(os.path.relpath(path, caption_dir.resolve())).as_posix()
    except ValueError:
        return path.resolve().as_uri()
    return quote(relative, safe="/")
