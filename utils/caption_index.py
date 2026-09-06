from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

from module.music_export import atomic_output_path

CAPTION_INDEX_NAME = ".qinglong-captions.json"
CAPTION_TRANSACTION_PREFIX = ".qinglong-caption-"


class CaptionFileTransaction:
    """Restore watched outputs if publication fails while the index lock is held."""

    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()
        self._backup: Path | None = None
        self._before: dict[Path, Path | None] = {}
        self._rollback_failed = False

    def __enter__(self):
        return self

    def watch(self, *paths: Path) -> None:
        for path in paths:
            path = Path(path).resolve()
            if path.parent != self.directory:
                raise ValueError(f"Caption output escapes its directory: {path}")
            if path in self._before:
                continue
            backup = None
            if path.exists():
                if not path.is_file():
                    raise ValueError(f"Caption output is not a file: {path}")
                if self._backup is None:
                    self._backup = Path(tempfile.mkdtemp(prefix=CAPTION_TRANSACTION_PREFIX, dir=self.directory)).resolve()
                backup = self._backup / path.name
                shutil.copy2(path, backup)
            self._before[path] = backup

    def restore(self, *paths: Path) -> None:
        selected = [Path(path).resolve() for path in paths] if paths else list(self._before)
        try:
            for path in reversed(selected):
                if path not in self._before:
                    continue
                backup = self._before[path]
                if backup is None:
                    path.unlink(missing_ok=True)
                else:
                    os.replace(backup, path)
                del self._before[path]
        except OSError as exc:
            self._rollback_failed = True
            raise RuntimeError(f"Caption rollback failed; previous files are retained in {self._backup}") from exc

    def __exit__(self, exc_type, exc_value, traceback):
        if self._rollback_failed:
            return False
        if exc_type is not None:
            self.restore()
        if self._backup is not None and self._backup.parent == self.directory:
            shutil.rmtree(self._backup, ignore_errors=True)
        return False


@dataclass
class CaptionIndex:
    """Current annotations and retained ownership of earlier exported files."""

    captions: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    origins: dict[str, str] = field(default_factory=dict)
    _file_names: dict[str, str] = field(init=False, repr=False, compare=False)
    _file_owners: dict[str, str] = field(init=False, repr=False, compare=False)
    _active_owners: dict[str, str] = field(init=False, repr=False, compare=False)
    _source_names: set[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        canonical_files = {}
        self._source_names = {caption_file_identity(key) for key in self.captions.keys() | self.origins.keys()}
        self._file_names = {}
        self._file_owners = {}
        for name, owner in self.files.items():
            identity = caption_file_identity(name)
            if identity in self._source_names:
                raise ValueError("Caption file overlaps a primary source")
            previous_owner = self._file_owners.get(identity)
            if previous_owner is not None and previous_owner != owner:
                raise ValueError("Caption index ownership collision")
            previous_name = self._file_names.get(identity)
            active_name = self.captions.get(owner)
            if previous_name is not None:
                if active_name is None or caption_file_identity(active_name) != identity or name != active_name:
                    continue
                del canonical_files[previous_name]
            canonical_files[name] = owner
            self._file_names[identity] = name
            self._file_owners[identity] = owner
        self.files = canonical_files

        self._active_owners = {}
        for owner, name in self.captions.items():
            identity = caption_file_identity(name)
            if identity in self._source_names:
                raise ValueError("Caption file overlaps a primary source")
            previous_owner = self._active_owners.get(identity)
            if previous_owner is not None and previous_owner != owner:
                raise ValueError("Caption index ownership collision")
            self._active_owners[identity] = owner

    def _source_identity(self, source_key: str) -> str:
        if not isinstance(source_key, str) or not source_key:
            raise ValueError("Invalid caption index path")
        identity = caption_file_identity(source_key)
        if identity in self._file_owners or identity in self._active_owners:
            raise ValueError("Primary source overlaps an owned caption file")
        return identity

    def validate_file_claim(self, source_key: str, filename: str, *, require_owned: bool = False) -> str:
        source_identity = self._source_identity(source_key)
        if (not isinstance(filename, str) or filename in {"", ".", ".."}
                or Path(filename).name != filename or "/" in filename or "\\" in filename):
            raise ValueError("Invalid caption index filename")

        identity = caption_file_identity(filename)
        if identity == source_identity or identity in self._source_names:
            raise ValueError("Caption file overlaps a primary source")
        file_owner = self._file_owners.get(identity)
        active_owner = self._active_owners.get(identity)
        if ((file_owner is not None and file_owner != source_key)
                or (active_owner is not None and active_owner != source_key)):
            raise ValueError("Caption index ownership collision")
        if require_owned and file_owner is None:
            raise ValueError(f"Cannot overwrite unowned caption companion: {filename}")
        return identity

    def claim_file(self, source_key: str, filename: str) -> str:
        """Retain ownership without replacing the active annotation."""
        identity = self.validate_file_claim(source_key, filename)
        alias = self._file_names.get(identity)
        if alias is not None and alias != filename:
            del self.files[alias]
        self.files[filename] = source_key
        self._file_names[identity] = filename
        self._file_owners[identity] = source_key
        return identity

    def set_caption(self, source_key: str, filename: str) -> None:
        """Set the active caption while retaining genuinely distinct history."""
        identity = self.claim_file(source_key, filename)
        previous_active = self.captions.get(source_key)
        previous_identity = caption_file_identity(previous_active) if previous_active is not None else None
        if previous_identity != identity and self._active_owners.get(previous_identity) == source_key:
            del self._active_owners[previous_identity]
        self.captions[source_key] = filename
        self._active_owners[identity] = source_key
        if previous_active is None:
            self._source_names.add(caption_file_identity(source_key))

    def validate_origin(self, source_key: str, origin: Path, *, require_known: bool = False) -> str:
        self._source_identity(source_key)
        identity = caption_file_identity(Path(origin).resolve()).replace("\\", "/")
        previous = self.origins.get(source_key)
        if previous is not None and previous != identity:
            raise ValueError("Caption source origin ownership collision")
        if require_known and previous is None:
            raise ValueError(f"Cannot overwrite a source with unverified origin: {source_key}")
        return identity

    def set_origin(self, source_key: str, origin: Path) -> None:
        # Keep the local lookup key separate from the original dataset URI.
        self.origins[source_key] = self.validate_origin(source_key, origin)
        self._source_names.add(caption_file_identity(source_key))


def caption_file_identity(path: str | os.PathLike[str]) -> str:
    """Return the host filesystem's identity for a caption path."""
    return os.path.normcase(os.path.normpath(os.fspath(path)))


def caption_index_key(source: Path, directory: Path) -> str:
    source, directory = Path(source).resolve(), Path(directory).resolve()
    try:
        key = os.path.relpath(source, directory)
    except ValueError:
        key = str(source)
    return os.path.normcase(key).replace("\\", "/")


def caption_index_path(directory: Path, name: str) -> Path:
    if not isinstance(name, str) or name in {"", ".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("Invalid caption index filename")
    directory = Path(directory).resolve()
    target = (directory / name).resolve()
    if not target.is_relative_to(directory):
        raise ValueError("Caption index target escapes its directory")
    return target


def load_caption_index(directory: Path, *, strict: bool = False) -> CaptionIndex | None:
    path = Path(directory) / CAPTION_INDEX_NAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1 or payload.get("kind") != "caption_index":
            raise ValueError("Unknown caption index format")
        entries = payload.get("captions")
        if not isinstance(entries, dict):
            raise ValueError("Invalid caption index entries")
        sources = set()
        captions = {}
        targets = set()
        for key, name in entries.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Invalid caption index path")
            source = (Path(directory) / key).resolve()
            key = caption_index_key(source, directory)
            target = caption_index_path(directory, name)
            if key in captions or caption_file_identity(target) in targets:
                raise ValueError("Ambiguous caption index ownership")
            sources.add(source)
            captions[key] = name
            targets.add(caption_file_identity(target))
        raw_origins = payload.get("origins", {})
        if not isinstance(raw_origins, dict):
            raise ValueError("Invalid caption source origins")
        origins = {}
        for key, origin in raw_origins.items():
            if (not isinstance(key, str) or not key or not isinstance(origin, str)
                    or not (PurePosixPath(origin).is_absolute() or PureWindowsPath(origin).is_absolute())):
                raise ValueError("Invalid caption source origin")
            source = (Path(directory) / key).resolve()
            key = caption_index_key(source, directory)
            if key in origins or caption_file_identity(source) in targets:
                raise ValueError("Ambiguous caption source origin")
            # An exported directory may be imported on a different platform.
            origins[key] = origin
            sources.add(source)
        files = payload.get("files", {name: key for key, name in captions.items()})
        if not isinstance(files, dict):
            raise ValueError("Invalid caption index file ownership")
        owned = {}
        target_owners = {}
        for name, key in files.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Invalid caption index file path")
            target = caption_index_path(directory, name)
            key = caption_index_key(Path(directory) / key, directory)
            identity = caption_file_identity(target)
            if (target in sources or key not in captions
                    or caption_file_identity(name) == caption_file_identity(CAPTION_INDEX_NAME)):
                raise ValueError("Ambiguous caption index file ownership")
            previous = target_owners.get(identity)
            if previous is not None:
                _, previous_key = previous
                if previous_key != key:
                    raise ValueError("Ambiguous caption index file ownership")
                continue
            owned[name] = key
            target_owners[identity] = (name, key)
        for key, name in captions.items():
            identity = caption_file_identity(caption_index_path(directory, name))
            previous = target_owners.get(identity)
            if previous is None or previous[1] != key:
                raise ValueError("Missing caption index file ownership")
            previous_name, _ = previous
            if previous_name != name:
                del owned[previous_name]
                owned[name] = key
                target_owners[identity] = (name, key)
        return CaptionIndex(captions, owned, origins)
    except (OSError, ValueError) as exc:
        if strict:
            raise ValueError(f"Cannot overwrite invalid or unowned caption index: {path}") from exc
        return None


def write_caption_index(directory: Path, index: CaptionIndex) -> None:
    payload = {"kind": "caption_index", "schema_version": 1,
               "captions": index.captions, "files": index.files, "origins": index.origins}
    with atomic_output_path(Path(directory) / CAPTION_INDEX_NAME) as temporary:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
