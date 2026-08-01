from __future__ import annotations

import hashlib
import numbers
from dataclasses import dataclass
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Iterable, Literal

from .artifacts import FileDigest
from .contracts import AutoRigInputContract
from .jcs import jcs_sha256
from .mask_sources import LoadedPartAlpha, load_validated_part_alphas
from .qcl import QCL_CODEC_VERSION, encode_qcl, materialize_qcl
from .tag_registry import V3_SPLIT_FAMILIES

MASK_COMPONENT_PLAN_VERSION = "mask-component-plan-v1"
MASK_COMPONENT_ID_SCHEMA = "mask-component-id-v1"
MASK_CLEANUP_SCHEMA = "mask-cleanup-v1"
MASK_SIDE_CLASSIFIER_VERSION = "mask-side-classifier-v1"
SIDE_MIN_COMPONENT_FRACTION_NUMERATOR = 1
SIDE_MIN_COMPONENT_FRACTION_DENOMINATOR = 20


class MaskComponentPlanError(ValueError):
    """Raised when a mask cannot become a deterministic component plan."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class MaskCleanupDescriptor:
    schema_version: str
    alpha_threshold_u8: int
    connectivity: int
    morphology: str
    min_component_area_px: int
    max_hole_area_px: int
    numpy_version: str
    scikit_image_version: str
    qcl_codec_version: str
    side_classifier_version: str
    side_min_component_fraction_numerator: int
    side_min_component_fraction_denominator: int
    component_id_schema: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "alpha_threshold_u8": self.alpha_threshold_u8,
            "connectivity": self.connectivity,
            "morphology": self.morphology,
            "min_component_area_px": self.min_component_area_px,
            "max_hole_area_px": self.max_hole_area_px,
            "numpy_version": self.numpy_version,
            "scikit_image_version": self.scikit_image_version,
            "qcl_codec_version": self.qcl_codec_version,
            "side_classifier_version": self.side_classifier_version,
            "side_min_component_fraction_numerator": self.side_min_component_fraction_numerator,
            "side_min_component_fraction_denominator": self.side_min_component_fraction_denominator,
            "component_id_schema": self.component_id_schema,
        }


@dataclass(frozen=True, slots=True)
class MaskComponentRecord:
    component_id: str
    label: int
    part_id: str
    side: Literal["xmin", "xmax"] | None
    bbox: tuple[int, int, int, int]
    cleaned_binary_mask_sha256: str
    pixel_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "label": self.label,
            "part_id": self.part_id,
            "side": self.side,
            "bbox": list(self.bbox),
            "cleaned_binary_mask_sha256": self.cleaned_binary_mask_sha256,
            "pixel_count": self.pixel_count,
        }


@dataclass(frozen=True, slots=True)
class NormalizedMaskPart:
    part_id: str
    source_part_id: str
    source_tag: str
    base_tag: str
    semantic_slug: str
    side: Literal["xmin", "xmax"] | None
    side_provenance: Literal["source_tag", "component_pair", "none"]
    source_xyxy: tuple[int, int, int, int]
    xyxy: tuple[int, int, int, int]
    depth_median: float
    cleaned_binary_mask_sha256: str
    qcl_file: FileDigest
    components: tuple[MaskComponentRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "part_id": self.part_id,
            "source_part_id": self.source_part_id,
            "source_tag": self.source_tag,
            "base_tag": self.base_tag,
            "semantic_slug": self.semantic_slug,
            "side": self.side,
            "side_provenance": self.side_provenance,
            "source_xyxy": list(self.source_xyxy),
            "xyxy": list(self.xyxy),
            "depth_median": self.depth_median,
            "cleaned_binary_mask_sha256": self.cleaned_binary_mask_sha256,
            "qcl_file": self.qcl_file.to_dict(),
            "components": [component.to_dict() for component in self.components],
        }


@dataclass(frozen=True, slots=True)
class MaskComponentPlan:
    schema_version: str
    canvas_edge: int
    cleanup: MaskCleanupDescriptor
    parts: tuple[NormalizedMaskPart, ...]
    projected_component_count: int
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "canvas_edge": self.canvas_edge,
            "cleanup": self.cleanup.to_dict(),
            "parts": [part.to_dict() for part in self.parts],
            "projected_component_count": self.projected_component_count,
        }


@dataclass(slots=True)
class _ComponentCandidate:
    bbox: tuple[int, int, int, int]
    mask_sha256: str
    pixel_count: int
    centroid_x_sum: int
    tight_mask: Any
    side: Literal["xmin", "xmax"] | None = None


def _error(code: str, message: str) -> MaskComponentPlanError:
    return MaskComponentPlanError(code, message)


def cleanup_area_threshold(canvas_edge: int) -> int:
    if isinstance(canvas_edge, bool) or not isinstance(canvas_edge, numbers.Integral):
        raise _error("invalid_canvas", "canvas edge must be a positive integer")
    edge = int(canvas_edge)
    if edge <= 0:
        raise _error("invalid_canvas", "canvas edge must be a positive integer")
    numerator = 4 * edge * edge
    denominator = 1024 * 1024
    rounded_half_up = (2 * numerator + denominator) // (2 * denominator)
    return max(4, rounded_half_up)


def _cleanup_descriptor(canvas_edge: int) -> MaskCleanupDescriptor:
    threshold = cleanup_area_threshold(canvas_edge)
    return MaskCleanupDescriptor(
        schema_version=MASK_CLEANUP_SCHEMA,
        alpha_threshold_u8=1,
        connectivity=8,
        morphology="identity",
        min_component_area_px=threshold,
        max_hole_area_px=threshold,
        numpy_version=distribution_version("numpy"),
        scikit_image_version=distribution_version("scikit-image"),
        qcl_codec_version=QCL_CODEC_VERSION,
        side_classifier_version=MASK_SIDE_CLASSIFIER_VERSION,
        side_min_component_fraction_numerator=SIDE_MIN_COMPONENT_FRACTION_NUMERATOR,
        side_min_component_fraction_denominator=SIDE_MIN_COMPONENT_FRACTION_DENOMINATOR,
        component_id_schema=MASK_COMPONENT_ID_SCHEMA,
    )


def _clean_binary_mask(loaded: LoadedPartAlpha, descriptor: MaskCleanupDescriptor):
    import numpy as np
    from skimage.morphology import remove_small_holes, remove_small_objects

    alpha = np.frombuffer(loaded.alpha_u8, dtype=np.uint8).reshape(
        loaded.height,
        loaded.width,
    )
    binary = alpha >= descriptor.alpha_threshold_u8
    without_specks = remove_small_objects(
        binary,
        min_size=descriptor.min_component_area_px,
        connectivity=2,
    )
    return remove_small_holes(
        without_specks,
        area_threshold=descriptor.max_hole_area_px,
        connectivity=2,
    )


def _label_components(mask):
    from skimage.measure import label

    return label(mask, connectivity=2, background=0)


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _extract_candidates(loaded: LoadedPartAlpha, labels) -> list[_ComponentCandidate]:
    import numpy as np

    source_x, source_y = loaded.part.xyxy[:2]
    candidates: list[_ComponentCandidate] = []
    for raw_label in (int(value) for value in np.unique(labels) if int(value) > 0):
        ys, xs = np.nonzero(labels == raw_label)
        local_x1 = int(xs.min())
        local_y1 = int(ys.min())
        local_x2 = int(xs.max()) + 1
        local_y2 = int(ys.max()) + 1
        tight_mask = labels[local_y1:local_y2, local_x1:local_x2] == raw_label
        bbox = (
            source_x + local_x1,
            source_y + local_y1,
            source_x + local_x2,
            source_y + local_y2,
        )
        candidates.append(
            _ComponentCandidate(
                bbox=bbox,
                mask_sha256=_sha256_bytes(tight_mask.astype(np.uint8).tobytes(order="C")),
                pixel_count=int(tight_mask.sum()),
                centroid_x_sum=int((xs + source_x).sum()),
                tight_mask=tight_mask,
            )
        )
    candidates.sort(key=lambda item: (item.bbox[1], item.bbox[0], item.mask_sha256))
    return candidates


def _assign_sides(
    loaded: LoadedPartAlpha,
    candidates: list[_ComponentCandidate],
    descriptor: MaskCleanupDescriptor,
) -> Literal["source_tag", "component_pair", "none"]:
    if loaded.part.side is not None:
        for candidate in candidates:
            candidate.side = loaded.part.side
        return "source_tag"
    if loaded.part.base_tag not in V3_SPLIT_FAMILIES or len(candidates) != 2:
        return "none"
    total_pixels = sum(candidate.pixel_count for candidate in candidates)
    if any(
        candidate.pixel_count * descriptor.side_min_component_fraction_denominator
        < total_pixels * descriptor.side_min_component_fraction_numerator
        for candidate in candidates
    ):
        return "none"
    first, second = candidates
    comparison = (
        first.centroid_x_sum * second.pixel_count
        - second.centroid_x_sum * first.pixel_count
    )
    if comparison == 0:
        return "none"
    xmin, xmax = (first, second) if comparison < 0 else (second, first)
    xmin.side = "xmin"
    xmax.side = "xmax"
    return "component_pair"


def _effective_part_id(loaded: LoadedPartAlpha, side: str | None) -> str:
    if side is None:
        return loaded.part.part_id
    return f"part/{loaded.part.semantic_slug}.{side}"


def _component_identity(
    *,
    part_id: str,
    bbox: tuple[int, int, int, int],
    mask_sha256: str,
) -> tuple[dict[str, object], str]:
    identity = {
        "schema": MASK_COMPONENT_ID_SCHEMA,
        "part_id": part_id,
        "component_bbox": list(bbox),
        "cleaned_binary_mask_sha256": mask_sha256,
    }
    digest = jcs_sha256(identity).removeprefix("sha256:")
    return identity, f"component/c_{digest}"


def _build_normalized_part(
    loaded: LoadedPartAlpha,
    *,
    side: Literal["xmin", "xmax"] | None,
    side_provenance: Literal["source_tag", "component_pair", "none"],
    candidates: list[_ComponentCandidate],
    item_root: Path,
) -> NormalizedMaskPart:
    import numpy as np

    part_id = _effective_part_id(loaded, side)
    candidates.sort(key=lambda item: (item.bbox[1], item.bbox[0], item.mask_sha256))
    x1 = min(item.bbox[0] for item in candidates)
    y1 = min(item.bbox[1] for item in candidates)
    x2 = max(item.bbox[2] for item in candidates)
    y2 = max(item.bbox[3] for item in candidates)
    labels = np.zeros((y2 - y1, x2 - x1), dtype=np.uint32)
    components: list[MaskComponentRecord] = []
    for label_value, candidate in enumerate(candidates, start=1):
        cx1, cy1, cx2, cy2 = candidate.bbox
        target = labels[cy1 - y1 : cy2 - y1, cx1 - x1 : cx2 - x1]
        target[candidate.tight_mask] = label_value
        _, component_id = _component_identity(
            part_id=part_id,
            bbox=candidate.bbox,
            mask_sha256=candidate.mask_sha256,
        )
        components.append(
            MaskComponentRecord(
                component_id=component_id,
                label=label_value,
                part_id=part_id,
                side=side,
                bbox=candidate.bbox,
                cleaned_binary_mask_sha256=candidate.mask_sha256,
                pixel_count=candidate.pixel_count,
            )
        )
    cleaned_mask_sha256 = _sha256_bytes(
        (labels > 0).astype(np.uint8).tobytes(order="C")
    )
    qcl_payload = encode_qcl(
        labels.ravel(order="C"),
        width=labels.shape[1],
        height=labels.shape[0],
    )
    qcl_file = materialize_qcl(item_root, qcl_payload)
    return NormalizedMaskPart(
        part_id=part_id,
        source_part_id=loaded.part.part_id,
        source_tag=loaded.part.source_tag,
        base_tag=loaded.part.base_tag,
        semantic_slug=loaded.part.semantic_slug,
        side=side,
        side_provenance=side_provenance,
        source_xyxy=loaded.part.xyxy,
        xyxy=(x1, y1, x2, y2),
        depth_median=loaded.part.depth_median,
        cleaned_binary_mask_sha256=cleaned_mask_sha256,
        qcl_file=qcl_file,
        components=tuple(components),
    )


def _validate_plan_parts(parts: tuple[NormalizedMaskPart, ...]) -> int:
    part_ids = [part.part_id for part in parts]
    if len(part_ids) != len(set(part_ids)):
        raise _error("component_identity_collision", "normalized Part IDs are not unique")
    component_ids: set[str] = set()
    projected_count = 0
    for part in parts:
        labels = tuple(component.label for component in part.components)
        if labels != tuple(range(1, len(part.components) + 1)):
            raise _error("invalid_component_plan", f"component labels are not canonical: {part.part_id}")
        px1, py1, px2, py2 = part.xyxy
        for component in part.components:
            cx1, cy1, cx2, cy2 = component.bbox
            if not (px1 <= cx1 < cx2 <= px2 and py1 <= cy1 < cy2 <= py2):
                raise _error("invalid_component_plan", f"component bbox escapes Part: {component.component_id}")
            if component.component_id in component_ids:
                raise _error(
                    "component_identity_collision",
                    f"component ID is not unique: {component.component_id}",
                )
            component_ids.add(component.component_id)
        projected_count += len(part.components)
    return projected_count


def build_mask_component_plan(
    loaded_parts: Iterable[LoadedPartAlpha],
    *,
    canvas_edge: int,
    item_root: str | Path,
) -> MaskComponentPlan:
    """Build and materialize the sole A-owned base component partition."""

    root = Path(item_root).resolve(strict=True)
    descriptor = _cleanup_descriptor(canvas_edge)
    loaded = tuple(sorted(loaded_parts, key=lambda item: item.part.part_id))
    if not loaded:
        raise _error("empty_component_input", "component planning requires at least one Part")
    source_ids = tuple(item.part.part_id for item in loaded)
    if len(source_ids) != len(set(source_ids)):
        raise _error("component_identity_collision", "source Part IDs are not unique")

    normalized_parts: list[NormalizedMaskPart] = []
    for item in loaded:
        cleaned = _clean_binary_mask(item, descriptor)
        labels = _label_components(cleaned)
        candidates = _extract_candidates(item, labels)
        if not candidates:
            raise _error(
                "empty_cleaned_mask",
                f"Part has no component after canonical cleanup: {item.part.part_id}",
            )
        side_provenance = _assign_sides(item, candidates, descriptor)
        grouped: dict[str | None, list[_ComponentCandidate]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate.side, []).append(candidate)
        for side, group in grouped.items():
            normalized_parts.append(
                _build_normalized_part(
                    item,
                    side=side,
                    side_provenance=side_provenance,
                    candidates=group,
                    item_root=root,
                )
            )

    parts = tuple(sorted(normalized_parts, key=lambda item: item.part_id))
    projected_count = _validate_plan_parts(parts)
    semantic_payload = {
        "schema_version": MASK_COMPONENT_PLAN_VERSION,
        "canvas_edge": int(canvas_edge),
        "cleanup": descriptor.to_dict(),
        "parts": [part.to_dict() for part in parts],
        "projected_component_count": projected_count,
    }
    return MaskComponentPlan(
        schema_version=MASK_COMPONENT_PLAN_VERSION,
        canvas_edge=int(canvas_edge),
        cleanup=descriptor,
        parts=parts,
        projected_component_count=projected_count,
        plan_sha256=jcs_sha256(semantic_payload),
    )


def build_base_mask_component_plan(contract: AutoRigInputContract) -> MaskComponentPlan:
    """Decode a validated base payload and build its A-owned component plan."""

    return build_mask_component_plan(
        load_validated_part_alphas(contract),
        canvas_edge=contract.canvas.resolution,
        item_root=contract.item_root,
    )


__all__ = [
    "MASK_CLEANUP_SCHEMA",
    "MASK_COMPONENT_ID_SCHEMA",
    "MASK_COMPONENT_PLAN_VERSION",
    "MASK_SIDE_CLASSIFIER_VERSION",
    "SIDE_MIN_COMPONENT_FRACTION_DENOMINATOR",
    "SIDE_MIN_COMPONENT_FRACTION_NUMERATOR",
    "MaskCleanupDescriptor",
    "MaskComponentPlan",
    "MaskComponentPlanError",
    "MaskComponentRecord",
    "NormalizedMaskPart",
    "build_base_mask_component_plan",
    "build_mask_component_plan",
    "cleanup_area_threshold",
]
