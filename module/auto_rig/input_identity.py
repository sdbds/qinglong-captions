from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .artifacts import ArtifactContractError, FileDigest, describe_file
from .contracts import (
    AUTO_RIG_INPUT_CONTRACT_VERSION,
    AutoRigInputContract,
    AutoRigPartContract,
)
from .jcs import jcs_sha256
from .tag_registry import (
    CANONICAL_TAG_REGISTRY_VERSION,
    V3_BASE_TAGS,
    V3_RAW_TAGS,
    V3_SPLIT_FAMILIES,
)

TARGET_INPUT_IDENTITY_VERSION = "target-input-identity-v1"


class AutoRigInputIdentityError(ValueError):
    """Raised when a validated target snapshot can no longer be reproduced."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class TargetInputPartIdentity:
    source_tag: str
    xyxy: tuple[int, int, int, int]
    depth_median: float
    source_mode: str
    color_path: str
    depth_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source_tag": self.source_tag,
            "xyxy": list(self.xyxy),
            "depth_median": self.depth_median,
            "source_mode": self.source_mode,
            "color_path": self.color_path,
            "depth_path": self.depth_path,
        }


@dataclass(frozen=True, slots=True)
class TargetInputIdentity:
    schema_version: str
    input_contract_version: str
    canonical_tag_registry_version: str
    canonical_tag_registry_sha256: str
    tag_version: str
    canvas_width: int
    canvas_height: int
    canvas_resolution: int
    payload_mode: str
    save_to_psd: bool
    tblr_split: bool
    input_files: tuple[FileDigest, ...]
    parts: tuple[TargetInputPartIdentity, ...]
    target_input_fingerprint: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "input_contract_version": self.input_contract_version,
            "canonical_tag_registry_version": self.canonical_tag_registry_version,
            "canonical_tag_registry_sha256": self.canonical_tag_registry_sha256,
            "tag_version": self.tag_version,
            "canvas": {
                "width": self.canvas_width,
                "height": self.canvas_height,
                "resolution": self.canvas_resolution,
                "coordinate_space": "layerdiff_canvas",
                "origin": "top_left",
                "y_axis": "down",
            },
            "payload_mode": self.payload_mode,
            "save_to_psd": self.save_to_psd,
            "tblr_split": self.tblr_split,
            "input_files": [item.to_dict() for item in self.input_files],
            "parts": [part.to_dict() for part in self.parts],
        }


def _error(message: str) -> AutoRigInputIdentityError:
    return AutoRigInputIdentityError("input_contract_mismatch", message)


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise _error("validated target path escapes the item root") from exc


def _tag_registry_payload() -> dict[str, object]:
    return {
        "schema_version": CANONICAL_TAG_REGISTRY_VERSION,
        "raw_tags": list(V3_RAW_TAGS),
        "base_tags": list(V3_BASE_TAGS),
        "split_families": sorted(V3_SPLIT_FAMILIES),
        "side_codec": {"-r": "xmin", "-l": "xmax"},
        "semantic_slug_codec": "ascii-space-to-hyphen-v1",
    }


def _part_identity(root: Path, part: AutoRigPartContract) -> TargetInputPartIdentity:
    return TargetInputPartIdentity(
        source_tag=part.source_tag,
        xyxy=part.xyxy,
        depth_median=part.depth_median,
        source_mode=part.source.mode,
        color_path=_relative_path(root, part.source.color_path),
        depth_path=_relative_path(root, part.source.depth_path),
    )


def build_target_input_identity(contract: AutoRigInputContract) -> TargetInputIdentity:
    """Re-hash a validated see-through snapshot into the override-independent target ID."""

    if not isinstance(contract, AutoRigInputContract):
        raise _error("target identity requires AutoRigInputContract")
    if contract.schema_version != AUTO_RIG_INPUT_CONTRACT_VERSION:
        raise _error("input contract version is unsupported")
    try:
        root = contract.item_root.resolve(strict=True)
    except OSError as exc:
        raise _error("item root is unavailable") from exc

    expected_by_path = {
        _relative_path(root, contract.source_image_path): contract.source_image_sha256,
        _relative_path(root, contract.layerdiff_manifest_path): contract.layerdiff_manifest_sha256,
        _relative_path(root, contract.optimized_manifest_path): contract.optimized_manifest_sha256,
        _relative_path(root, contract.optimized_info_path): contract.optimized_info_sha256,
    }
    if any(value is None for value in expected_by_path.values()):
        raise _error("input contract lacks a manifest snapshot digest")
    for part in contract.parts:
        for path, expected in (
            (part.source.color_path, part.source.color_sha256),
            (part.source.depth_path, part.source.depth_sha256),
        ):
            relative = _relative_path(root, path)
            previous = expected_by_path.setdefault(relative, expected)
            if previous != expected:
                raise _error(f"target path has conflicting snapshot digests: {relative}")

    files = []
    for relative, expected in sorted(expected_by_path.items()):
        try:
            actual = describe_file(root, relative)
        except ArtifactContractError as exc:
            raise _error(f"target input is unavailable: {relative}") from exc
        if actual.sha256 != expected:
            raise _error(f"target input changed after validation: {relative}")
        files.append(actual)

    parts = tuple(
        sorted(
            (_part_identity(root, part) for part in contract.parts),
            key=lambda item: item.source_tag,
        )
    )
    if len({part.source_tag for part in parts}) != len(parts):
        raise _error("target part source tags are not unique")
    registry_sha256 = jcs_sha256(_tag_registry_payload())
    values = {
        "schema_version": TARGET_INPUT_IDENTITY_VERSION,
        "input_contract_version": contract.schema_version,
        "canonical_tag_registry_version": CANONICAL_TAG_REGISTRY_VERSION,
        "canonical_tag_registry_sha256": registry_sha256,
        "tag_version": contract.tag_version,
        "canvas_width": contract.canvas.width,
        "canvas_height": contract.canvas.height,
        "canvas_resolution": contract.canvas.resolution,
        "payload_mode": contract.payload_mode,
        "save_to_psd": contract.save_to_psd,
        "tblr_split": contract.tblr_split,
        "input_files": tuple(files),
        "parts": parts,
    }
    provisional = TargetInputIdentity(
        **values,
        target_input_fingerprint="",
    )
    return TargetInputIdentity(
        **values,
        target_input_fingerprint=jcs_sha256(provisional.semantic_payload()),
    )


__all__ = [
    "TARGET_INPUT_IDENTITY_VERSION",
    "AutoRigInputIdentityError",
    "TargetInputIdentity",
    "TargetInputPartIdentity",
    "build_target_input_identity",
]
