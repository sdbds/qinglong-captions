from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

from filelock import FileLock

from config.config import get_supported_extensions
from module.wdtagger import constants
from utils.caption_index import (
    CAPTION_INDEX_NAME,
    CaptionFileTransaction,
    CaptionIndex,
    caption_file_identity,
    caption_index_key,
    caption_index_path,
    load_caption_index,
    write_caption_index,
)
from utils.console_util import print_exception
from utils.output_writer import normalize_caption_extension
from utils.path_safety import safe_child_path, safe_sibling_path


_PRIMARY_EXTENSIONS = frozenset(
    extension
    for kind in ("image", "animation", "video", "audio", "application")
    for extension in get_supported_extensions(kind)
)


@contextmanager
def _locked_index(source: Path):
    directory = source.parent
    index_path = directory / CAPTION_INDEX_NAME
    lock_path = directory / (CAPTION_INDEX_NAME + ".lock")
    if source in (index_path, lock_path) or index_path.is_symlink() or lock_path.is_symlink():
        raise ValueError("Caption index path collides with a primary source or symlink")
    if lock_path.is_file() and lock_path.stat().st_size:
        raise ValueError(f"Cannot overwrite unowned caption index lock: {lock_path}")
    with FileLock(str(lock_path), preserve_lock_file=True):
        if lock_path.stat().st_size:
            raise ValueError(f"Cannot overwrite unowned caption index lock: {lock_path}")
        yield load_caption_index(directory, strict=True) or CaptionIndex()


def _caption_path(source: Path, extension: str, index: CaptionIndex) -> Path | None:
    key = caption_index_key(source, source.parent)
    active = index.captions.get(key)
    owned = ([active] if active is not None else []) + [name for name, owner in index.files.items() if owner == key]
    for name in owned:
        if name.casefold().endswith(extension.casefold()):
            if (source.parent / name).is_symlink():
                raise ValueError("Caption output must not be a symlink")
            return caption_index_path(source.parent, name)

    if source.with_suffix(extension).is_symlink():
        raise ValueError("Caption output must not be a symlink")
    legacy = safe_sibling_path(source, extension)
    if legacy == source:
        return None
    try:
        index.validate_file_claim(key, legacy.name)
    except ValueError:
        return None
    if legacy.exists():
        if not legacy.is_file() or legacy.suffix.lower() in _PRIMARY_EXTENSIONS or legacy.suffix.lower() == ".json":
            return None
        stem = caption_file_identity(source.stem)
        legacy_name = caption_file_identity(legacy.name)
        for sibling in source.parent.iterdir():
            if (not sibling.is_file() or sibling.resolve() == source or sibling.name in index.files
                    or sibling.suffix.lower() not in _PRIMARY_EXTENSIONS):
                continue
            sibling_stem = caption_file_identity(sibling.stem)
            if sibling_stem == stem:
                return None
            # Multi-part suffixes can overlap another source's legacy caption.
            if extension.count(".") > 1 and legacy_name.startswith(sibling_stem + "."):
                return None
    return legacy


def has_sidecar_caption(uri: str, caption_extension: str) -> bool:
    return any(line.strip() for line in read_sidecar_caption(uri, caption_extension))


def read_sidecar_caption(uri: str, caption_extension: str) -> List[str]:
    source = Path(uri).resolve()
    if not source.parent.is_dir():
        return []
    extension = normalize_caption_extension(caption_extension)
    if extension is None:
        raise ValueError("Caption extension must not be empty")
    safe_sibling_path(source, extension)
    with _locked_index(source) as index:
        caption_path = _caption_path(source, extension, index)
        if caption_path is None:
            return []
        try:
            content = caption_path.read_text(encoding="utf-8")
        except OSError:
            return []
    return content.splitlines() if extension.lower() == ".txt" else [content]


def write_sidecar_caption(
    uri: str, captions: List[str], *, caption_extension: str, caption_separator: str, append: bool = False,
) -> List[str]:
    source = Path(uri).resolve()
    source.parent.mkdir(parents=True, exist_ok=True)
    extension = normalize_caption_extension(caption_extension)
    if extension is None:
        raise ValueError("Caption extension must not be empty")
    safe_sibling_path(source, extension)
    key = caption_index_key(source, source.parent)
    with _locked_index(source) as index:
        output_path = _caption_path(source, extension, index)
        if output_path is None:
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
            output_path = safe_child_path(source.parent, f"{source.stem[:160]}__{digest}{extension}")
            index.validate_file_claim(key, output_path.name, require_owned=output_path.exists())
        reserved = {CAPTION_INDEX_NAME, CAPTION_INDEX_NAME + ".lock"}
        if output_path.name in reserved or (source.parent / output_path.name).is_symlink():
            raise ValueError("Caption output collides with its index or a symlink")
        index.validate_file_claim(key, output_path.name)
        published_tags = list(captions)
        if append and output_path.is_file():
            existing = output_path.read_text(encoding="utf-8").strip()
            if existing:
                published_tags = existing.split(caption_separator) + published_tags
        with CaptionFileTransaction(source.parent) as publication:
            publication.watch(source.parent / CAPTION_INDEX_NAME, output_path)
            output_path.write_text(caption_separator.join(published_tags), encoding="utf-8")
            index.set_caption(key, output_path.name)
            write_caption_index(source.parent, index)
    return published_tags


def write_tags_json(train_data_dir: str, all_json_tags: Dict[str, Dict[str, List[str]]]) -> None:
    try:
        json_output_path = Path(train_data_dir) / "tags.json"
        with json_output_path.open("w", encoding="utf-8") as jf:
            json.dump(all_json_tags, jf, ensure_ascii=False, indent=2)
        constants.console.print(f"[bold green]JSON saved to:[/bold green] {json_output_path}")
    except Exception as e:
        print_exception(constants.console, e, prefix="Failed to save JSON")
