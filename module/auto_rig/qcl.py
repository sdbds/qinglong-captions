from __future__ import annotations

import hashlib
import numbers
import os
import stat
import struct
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .artifacts import FileDigest, atomic_write_bytes, describe_file

QCL_MAGIC = b"QCL1"
QCL_CODEC_VERSION = "canonical-label-map-qcl1"
_HEADER = struct.Struct("<4sII")
_UINT32_MAX = (1 << 32) - 1
_REPARSE_POINT_ATTRIBUTE = 0x400


class QclContractError(ValueError):
    """Raised when a canonical label map violates the QCL1 contract."""


def _require_dimension(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise QclContractError(f"{field} must be an integer")
    normalized = int(value)
    if not 0 < normalized <= _UINT32_MAX:
        raise QclContractError(f"{field} must fit a positive uint32")
    return normalized


def _label_array(
    labels: Iterable[int],
    *,
    width: int,
    height: int,
) -> array[int]:
    normalized = array("I")
    if normalized.itemsize != 4:  # pragma: no cover - unsupported Python ABI
        raise QclContractError("QCL1 requires a 32-bit unsigned-int array runtime")
    seen: set[int] = set()
    for value in labels:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise QclContractError("QCL1 labels must be uint32 integers")
        label = int(value)
        if not 0 <= label <= _UINT32_MAX:
            raise QclContractError("QCL1 labels must be uint32 integers")
        normalized.append(label)
        if label:
            seen.add(label)
    if len(normalized) != width * height:
        raise QclContractError("QCL1 label count differs from width*height")
    if not seen or seen != set(range(1, max(seen) + 1)):
        raise QclContractError("QCL1 positive labels must be contiguous from 1")
    return normalized


@dataclass(frozen=True, slots=True)
class CanonicalLabelMap:
    width: int
    height: int
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        width = _require_dimension(self.width, field="width")
        height = _require_dimension(self.height, field="height")
        normalized = _label_array(self.labels, width=width, height=height)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "labels", tuple(normalized))


def encode_qcl(labels: Iterable[int], *, width: int, height: int) -> bytes:
    """Encode canonical row-major labels into strict QCL1 bytes."""

    normalized_width = _require_dimension(width, field="width")
    normalized_height = _require_dimension(height, field="height")
    normalized = _label_array(
        labels,
        width=normalized_width,
        height=normalized_height,
    )
    if sys.byteorder != "little":  # pragma: no cover - supported CI is little-endian
        normalized.byteswap()
    return _HEADER.pack(QCL_MAGIC, normalized_width, normalized_height) + normalized.tobytes()


def decode_qcl(payload: bytes) -> CanonicalLabelMap:
    """Decode QCL1 bytes and reject every non-canonical representation."""

    if not isinstance(payload, bytes) or len(payload) < _HEADER.size:
        raise QclContractError("QCL1 payload is truncated")
    magic, width, height = _HEADER.unpack_from(payload)
    if magic != QCL_MAGIC:
        raise QclContractError("QCL1 magic mismatch")
    width = _require_dimension(width, field="width")
    height = _require_dimension(height, field="height")
    expected_size = _HEADER.size + width * height * 4
    if len(payload) != expected_size:
        raise QclContractError("QCL1 payload size or trailing bytes mismatch")
    labels = array("I")
    if labels.itemsize != 4:  # pragma: no cover - unsupported Python ABI
        raise QclContractError("QCL1 requires a 32-bit unsigned-int array runtime")
    labels.frombytes(payload[_HEADER.size :])
    if sys.byteorder != "little":  # pragma: no cover - supported CI is little-endian
        labels.byteswap()
    return CanonicalLabelMap(width=width, height=height, labels=tuple(labels))


def _is_reparse_point(path: Path) -> bool:
    status = os.lstat(path)
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE
    )


def materialize_qcl(item_root: str | Path, payload: bytes) -> FileDigest:
    """Materialize validated QCL1 bytes in the A-owned component cache."""

    decode_qcl(payload)
    root = Path(item_root).resolve(strict=True)
    if not root.is_dir():
        raise QclContractError("item root must be a directory")
    digest = hashlib.sha256(payload).hexdigest()
    relative_path = f"rig/cache/A/components/{digest}.qcl"
    target = root / Path(*relative_path.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if _is_reparse_point(target) or not target.is_file() or target.read_bytes() != payload:
            raise QclContractError("QCL1 same-name/different-byte invariant failed")
    else:
        atomic_write_bytes(target, payload)
    return describe_file(root, relative_path)


__all__ = [
    "QCL_CODEC_VERSION",
    "QCL_MAGIC",
    "CanonicalLabelMap",
    "QclContractError",
    "decode_qcl",
    "encode_qcl",
    "materialize_qcl",
]
