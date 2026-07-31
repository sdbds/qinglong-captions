from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE_PATTERN = re.compile(r"^[A-Za-z]:[/\\]")


class ArtifactContractError(ValueError):
    """Raised when an artifact path or digest violates the public contract."""


def normalize_relative_path(path: str | Path) -> str:
    raw = str(path)
    if not raw or raw == ".":
        raise ArtifactContractError("artifact path must be a non-empty relative path")
    if Path(raw).is_absolute() or _WINDOWS_ABSOLUTE_PATTERN.match(raw):
        raise ArtifactContractError(f"artifact path must be relative: {raw}")

    normalized_input = raw.replace("\\", "/")
    raw_parts = normalized_input.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ArtifactContractError(f"artifact path contains an unsafe segment: {raw}")

    normalized = PurePosixPath(*raw_parts).as_posix()
    if normalized in {"", "."}:
        raise ArtifactContractError("artifact path must be a non-empty relative path")
    return normalized


def _validate_sha256(value: str) -> str:
    normalized = str(value)
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ArtifactContractError("digest must be a lowercase sha256:<hex> SHA-256 value")
    return normalized


@dataclass(frozen=True)
class FileDigest:
    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", normalize_relative_path(self.path))
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ArtifactContractError("artifact size must be a non-negative integer")
        object.__setattr__(self, "sha256", _validate_sha256(self.sha256))

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, payload: Any) -> "FileDigest":
        if not isinstance(payload, dict) or set(payload) != {"path", "size", "sha256"}:
            raise ArtifactContractError("file digest must contain exactly path, size, and sha256")
        return cls(path=payload["path"], size=payload["size"], sha256=payload["sha256"])


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def canonical_json_sha256(payload: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def sha256_file(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.is_file():
        raise ArtifactContractError(f"artifact is not a regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def describe_file(root: str | Path, relative_path: str | Path) -> FileDigest:
    normalized = normalize_relative_path(relative_path)
    resolved_root = Path(root).resolve()
    candidate = (resolved_root / Path(*PurePosixPath(normalized).parts)).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ArtifactContractError(f"artifact resolves outside item root: {normalized}") from exc
    if not candidate.is_file():
        raise ArtifactContractError(f"artifact is not a regular file: {normalized}")
    return FileDigest(path=normalized, size=candidate.stat().st_size, sha256=sha256_file(candidate))


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f"{target.name}.",
        suffix=".part",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, payload: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(payload) + b"\n")
