from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .artifacts import ArtifactContractError, FileDigest, describe_file
from .component_plan import (
    MASK_COMPONENT_PLAN_VERSION,
    MaskComponentPlan,
    NormalizedMaskPart,
)
from .jcs import jcs_sha256
from .qcl import QclContractError, decode_qcl

COMPONENT_GEOMETRY_LOADER_VERSION = "component-geometry-loader-v1"
_QCL_DIRECTORY = "rig/cache/A/components"
_REPARSE_POINT_ATTRIBUTE = 0x400


class ComponentGeometryError(ValueError):
    """Raised when A-owned component labels cannot be trusted as geometry."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class LoadedComponentGeometry:
    part: NormalizedMaskPart
    width: int
    height: int
    labels: tuple[int, ...]


def _error(code: str, message: str) -> ComponentGeometryError:
    return ComponentGeometryError(code, message)


def _is_reparse_point(path: Path) -> bool:
    status = os.lstat(path)
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE
    )


def _qcl_path(item_root: Path, digest: FileDigest) -> Path:
    pure = PurePosixPath(digest.path)
    if pure.parent.as_posix() != _QCL_DIRECTORY or pure.suffix != ".qcl":
        raise _error(
            "invalid_component_plan",
            f"component QCL path is outside its fixed namespace: {digest.path}",
        )
    current = item_root
    try:
        for part in pure.parts:
            current = current / part
            os.lstat(current)
            if _is_reparse_point(current):
                raise _error(
                    "input_contract_mismatch",
                    f"component QCL path contains a link: {digest.path}",
                )
        if not current.is_file():
            raise _error(
                "input_contract_mismatch",
                f"component QCL is not a regular file: {digest.path}",
            )
    except ComponentGeometryError:
        raise
    except OSError as exc:
        raise _error(
            "input_contract_mismatch",
            f"component QCL cannot be inspected: {digest.path}",
        ) from exc
    return current


def _read_authenticated_qcl(item_root: Path, expected: FileDigest):
    path = _qcl_path(item_root, expected)
    try:
        actual = describe_file(item_root, expected.path)
        payload = path.read_bytes()
    except (ArtifactContractError, OSError) as exc:
        raise _error(
            "input_contract_mismatch",
            f"component QCL cannot be read: {expected.path}",
        ) from exc
    payload_sha256 = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    if actual != expected or len(payload) != expected.size or payload_sha256 != expected.sha256:
        raise _error(
            "input_contract_mismatch",
            f"component QCL changed after planning: {expected.path}",
        )
    try:
        return decode_qcl(payload)
    except QclContractError as exc:
        raise _error(
            "input_contract_mismatch",
            f"component QCL is no longer canonical: {expected.path}",
        ) from exc


def _sha256_u8(values) -> str:
    return f"sha256:{hashlib.sha256(values.astype('uint8').tobytes(order='C')).hexdigest()}"


def _validate_part_labels(part: NormalizedMaskPart, label_map) -> None:
    import numpy as np

    width = part.xyxy[2] - part.xyxy[0]
    height = part.xyxy[3] - part.xyxy[1]
    if (label_map.width, label_map.height) != (width, height):
        raise _error(
            "invalid_component_plan",
            f"component QCL dimensions differ from Part bbox: {part.part_id}",
        )
    records = tuple(sorted(part.components, key=lambda item: item.label))
    if tuple(record.label for record in records) != tuple(range(1, len(records) + 1)):
        raise _error(
            "invalid_component_plan",
            f"component record labels are not canonical: {part.part_id}",
        )
    labels = np.asarray(label_map.labels, dtype=np.uint32).reshape(height, width)
    if set(int(value) for value in np.unique(labels)) - {0} != set(
        range(1, len(records) + 1)
    ):
        raise _error(
            "invalid_component_plan",
            f"component QCL labels differ from records: {part.part_id}",
        )
    if _sha256_u8(labels > 0) != part.cleaned_binary_mask_sha256:
        raise _error(
            "invalid_component_plan",
            f"whole-Part mask digest differs from QCL: {part.part_id}",
        )
    part_x, part_y = part.xyxy[:2]
    for record in records:
        if record.part_id != part.part_id or record.side != part.side:
            raise _error(
                "invalid_component_plan",
                f"component ownership differs from Part: {record.component_id}",
            )
        mask = labels == record.label
        ys, xs = np.nonzero(mask)
        pixel_count = int(mask.sum())
        if pixel_count != record.pixel_count:
            raise _error(
                "invalid_component_plan",
                f"component pixel count differs from QCL: {record.component_id}",
            )
        bbox = (
            part_x + int(xs.min()),
            part_y + int(ys.min()),
            part_x + int(xs.max()) + 1,
            part_y + int(ys.max()) + 1,
        )
        if bbox != record.bbox:
            raise _error(
                "invalid_component_plan",
                f"component bbox differs from QCL: {record.component_id}",
            )
        x1, y1, x2, y2 = bbox
        tight = mask[y1 - part_y : y2 - part_y, x1 - part_x : x2 - part_x]
        if _sha256_u8(tight) != record.cleaned_binary_mask_sha256:
            raise _error(
                "invalid_component_plan",
                f"component mask digest differs from QCL: {record.component_id}",
            )


def load_component_geometry(
    plan: MaskComponentPlan,
    *,
    item_root: str | Path,
) -> tuple[LoadedComponentGeometry, ...]:
    """Authenticate the sole A-owned QCL partition for downstream geometry."""

    if not isinstance(plan, MaskComponentPlan) or plan.schema_version != MASK_COMPONENT_PLAN_VERSION:
        raise _error("invalid_component_plan", "unsupported MaskComponentPlan")
    if plan.plan_sha256 != jcs_sha256(plan.semantic_payload()):
        raise _error("invalid_component_plan", "MaskComponentPlan digest mismatch")
    root = Path(item_root).expanduser().absolute()
    try:
        if not root.is_dir() or _is_reparse_point(root):
            raise _error("input_contract_mismatch", "item root must be a non-link directory")
        root = root.resolve(strict=True)
    except ComponentGeometryError:
        raise
    except OSError as exc:
        raise _error("input_contract_mismatch", "item root cannot be inspected") from exc

    loaded: list[LoadedComponentGeometry] = []
    for part in sorted(plan.parts, key=lambda item: item.part_id):
        label_map = _read_authenticated_qcl(root, part.qcl_file)
        _validate_part_labels(part, label_map)
        loaded.append(
            LoadedComponentGeometry(
                part=part,
                width=label_map.width,
                height=label_map.height,
                labels=label_map.labels,
            )
        )
    if tuple(item.part.part_id for item in loaded) != tuple(
        sorted({item.part.part_id for item in loaded})
    ):
        raise _error("invalid_component_plan", "normalized Part IDs are not unique")
    return tuple(loaded)


__all__ = [
    "COMPONENT_GEOMETRY_LOADER_VERSION",
    "ComponentGeometryError",
    "LoadedComponentGeometry",
    "load_component_geometry",
]
