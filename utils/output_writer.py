from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from filelock import FileLock

from utils.caption_index import CAPTION_INDEX_NAME, CaptionFileTransaction, CaptionIndex, caption_file_identity, caption_index_key, load_caption_index, write_caption_index
from utils.path_safety import safe_child_path, safe_leaf_name, safe_sibling_path

if TYPE_CHECKING:
    from module.providers.base import CaptionResult


def _is_caption_result(value: object) -> bool:
    return (
        hasattr(value, "raw")
        and hasattr(value, "parsed")
        and hasattr(value, "metadata")
        and hasattr(value, "caption_extension")
    )


def _caption_extension_for_mime(mime: str) -> str:
    if mime.startswith("video") or mime.startswith("audio"):
        return ".srt"
    if mime.startswith("application"):
        return ".md"
    return ".txt"


def normalize_caption_extension(extension: object) -> Optional[str]:
    if extension is None:
        return None
    value = str(extension).strip()
    if not value:
        return None
    if not value.startswith("."):
        value = f".{value}"
    return value


def caption_extension_from_payload(payload: object) -> Optional[str]:
    if _is_caption_result(payload):
        return payload.caption_extension
    if not isinstance(payload, dict):
        return None
    return normalize_caption_extension(payload.get("caption_extension"))


def caption_output_path(source_path: Path, mime: str, output=None, *, caption_base=None, protected_paths=()) -> Path:
    extension = caption_extension_from_payload(output) or _caption_extension_for_mime(mime)
    return caption_path_avoiding_sources(
        safe_sibling_path(caption_base or source_path, extension), protected_paths, additional_paths=(source_path,),
    )


def disambiguate_caption_bases(bases: dict[str, Path]) -> dict[str, Path]:
    stems = Counter(caption_file_identity(base.with_suffix("").resolve()) for base in bases.values())
    planned = {}
    occupied = set()
    for uri, base in bases.items():
        if stems[caption_file_identity(base.with_suffix("").resolve())] > 1:
            source_name = Path(safe_leaf_name(uri))
            digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]
            base = safe_child_path(base.parent, f"{source_name.stem[:160]}__{digest}{source_name.suffix}")
        identity = caption_file_identity(base.with_suffix("").resolve())
        if identity in occupied:
            raise ValueError(f"Ambiguous caption base collision: {base}")
        occupied.add(identity)
        planned[uri] = base
    return planned


class ResolvedPaths(frozenset[Path]):
    """Normalize protected paths once, then reuse them throughout a batch."""

    def __new__(cls, paths=()):
        if isinstance(paths, cls):
            return paths
        return super().__new__(cls, (Path(path).resolve() for path in paths))


def caption_path_avoiding_sources(target: Path, protected_paths, *, additional_paths=()) -> Path:
    protected = ResolvedPaths(protected_paths)
    additional = ResolvedPaths(additional_paths)
    target = Path(target)
    while True:
        resolved = target.resolve()
        if resolved not in protected and resolved not in additional:
            return target
        target = target.with_name(f"{target.stem}.caption{target.suffix}")


def caption_text(output, *, fallback: str = "") -> str:
    """Render semantic caption text consistently at generation and export boundaries."""
    if _is_caption_result(output):
        return caption_text(output.parsed, fallback=output.raw) if output.parsed is not None else output.raw
    if isinstance(output, dict):
        keys = ("long_description", "transcript", "translation_srt", "description", "short_description", "markdown", "text")
        task_kind = str(output.get("task_kind") or "").strip().lower()
        subtitle_format = str(output.get("subtitle_format") or "").strip().lower()
        if task_kind == "ast" or subtitle_format == "srt" or caption_extension_from_payload(output) == ".srt":
            keys = ("translation_srt", "transcript", "description", "long_description", "short_description", "markdown", "text")
        for key in keys:
            value = output.get(key)
            if str(value or "").strip():
                return str(value)
        return fallback
    if isinstance(output, list):
        return "\n".join(str(line) for line in output)
    return str(output or "")


def has_meaningful_text_content(content: object) -> bool:
    return bool(str(content).strip())


def write_markdown_output(output_dir: Path, content: str, filename: str = "result.md") -> Optional[Path]:
    if not has_meaningful_text_content(content):
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = safe_child_path(output_dir, filename, default_name="result.md")
    target.write_text(content, encoding="utf-8")
    return target


def write_caption_output(source_path: Path, output, mime: str, *, caption_base=None, protected_paths=()) -> tuple[Path, Optional[Path]]:
    source_path = Path(source_path)
    protected = ResolvedPaths(protected_paths)
    text_path = caption_output_path(source_path, mime, output, caption_base=caption_base, protected_paths=protected)
    json_path: Optional[Path] = None
    payload = output.parsed if _is_caption_result(output) else output if isinstance(output, dict) else None
    text = caption_text(output) if _is_caption_result(output) or isinstance(output, (dict, list)) else str(output)
    if payload is not None:
        json_path = caption_path_avoiding_sources(text_path.with_suffix(".json"), protected, additional_paths=(source_path, text_path))
    directory = text_path.parent.resolve()
    index_path = directory / CAPTION_INDEX_NAME
    lock_path = directory / (CAPTION_INDEX_NAME + ".lock")
    reserved = {index_path, lock_path}
    if reserved.intersection(protected) or source_path.resolve() in reserved:
        raise ValueError("Caption index path collides with a primary source")
    if text_path.resolve() in reserved or (json_path is not None and json_path.resolve() in reserved):
        raise ValueError("Caption output path collides with its index")
    directory.mkdir(parents=True, exist_ok=True)
    if lock_path.is_file() and lock_path.stat().st_size:
        raise ValueError(f"Cannot overwrite unowned caption index lock: {lock_path}")
    with FileLock(str(lock_path), preserve_lock_file=True):
        if lock_path.stat().st_size:
            raise ValueError(f"Cannot overwrite unowned caption index lock: {lock_path}")
        index = load_caption_index(directory, strict=True) or CaptionIndex()
        key = caption_index_key(source_path, directory)
        index.validate_file_claim(key, text_path.name)
        if json_path is not None:
            index.validate_file_claim(key, json_path.name, require_owned=json_path.exists())
        with CaptionFileTransaction(directory) as publication:
            publication.watch(index_path, text_path)
            if json_path is not None:
                publication.watch(json_path)
                json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            text_path.write_text(text, encoding="utf-8")
            index.set_caption(key, text_path.name)
            if json_path is not None:
                index.claim_file(key, json_path.name)
            write_caption_index(directory, index)
    return text_path, json_path
