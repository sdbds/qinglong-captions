from __future__ import annotations

import hashlib
import numbers
from dataclasses import dataclass
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal

from .artifacts import FileDigest
from .contracts import AutoRigInputContract
from .jcs import jcs_sha256
from .mask_sources import LoadedPartAlpha, load_validated_part_alphas
from .qcl import QCL_CODEC_VERSION, encode_qcl, materialize_qcl
from .tag_registry import V3_SPLIT_FAMILIES

if TYPE_CHECKING:
    from .native_variants import NativeVariantCandidate, NativeVariantSet

MASK_COMPONENT_PLAN_VERSION = "mask-component-plan-v2"
MASK_COMPONENT_ID_SCHEMA = "mask-component-id-v1"
MASK_CLEANUP_SCHEMA = "mask-cleanup-v2"
MASK_SIDE_CLASSIFIER_VERSION = "mask-side-classifier-v3"
NATIVE_VARIANT_PARTITION_VERSION = "native-variant-partition-v1"
MASK_ALPHA_SUPPORT_THRESHOLD_U8 = 24
MASK_SIDE_CLASSIFIABLE_FAMILIES = V3_SPLIT_FAMILIES | frozenset({"legwear", "footwear"})
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
class NativeVariantPartitionRecord:
    schema_version: str
    variant_id: str
    part_id: str
    semantic_role: str
    status: Literal["ready", "empty_after_cleanup"]
    side_provenance: Literal["role_single", "role_component_pair", "none"]
    source_xyxy: tuple[int, int, int, int]
    xyxy: tuple[int, int, int, int] | None
    cleaned_binary_mask_sha256: str | None
    qcl_file: FileDigest | None
    components: tuple[MaskComponentRecord, ...]
    partition_sha256: str

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "variant_id": self.variant_id,
            "part_id": self.part_id,
            "semantic_role": self.semantic_role,
            "status": self.status,
            "side_provenance": self.side_provenance,
            "source_xyxy": list(self.source_xyxy),
            "xyxy": list(self.xyxy) if self.xyxy is not None else None,
            "cleaned_binary_mask_sha256": self.cleaned_binary_mask_sha256,
            "qcl_file": self.qcl_file.to_dict() if self.qcl_file is not None else None,
            "components": [component.to_dict() for component in self.components],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_payload(), "partition_sha256": self.partition_sha256}


@dataclass(frozen=True, slots=True)
class MaskComponentPlan:
    schema_version: str
    canvas_edge: int
    cleanup: MaskCleanupDescriptor
    parts: tuple[NormalizedMaskPart, ...]
    native_variant_set_sha256: str | None
    variant_partitions: tuple[NativeVariantPartitionRecord, ...]
    base_projected_component_count: int
    native_variant_projected_component_count: int
    projected_component_count: int
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "canvas_edge": self.canvas_edge,
            "cleanup": self.cleanup.to_dict(),
            "parts": [part.to_dict() for part in self.parts],
            "native_variant_set_sha256": self.native_variant_set_sha256,
            "variant_partitions": [partition.to_dict() for partition in self.variant_partitions],
            "base_projected_component_count": self.base_projected_component_count,
            "native_variant_projected_component_count": self.native_variant_projected_component_count,
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


@dataclass(frozen=True, slots=True)
class _VariantPartAdapter:
    part_id: str
    xyxy: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class _LoadedVariantAlpha:
    part: _VariantPartAdapter
    width: int
    height: int
    alpha_u8: bytes


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
        alpha_threshold_u8=MASK_ALPHA_SUPPORT_THRESHOLD_U8,
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
    if loaded.part.base_tag not in MASK_SIDE_CLASSIFIABLE_FAMILIES:
        return "none"
    total_pixels = sum(candidate.pixel_count for candidate in candidates)
    reliable = [
        candidate
        for candidate in candidates
        if candidate.pixel_count * descriptor.side_min_component_fraction_denominator
        >= total_pixels * descriptor.side_min_component_fraction_numerator
    ]
    if len(reliable) != 2:
        return "none"
    first, second = reliable
    comparison = first.centroid_x_sum * second.pixel_count - second.centroid_x_sum * first.pixel_count
    if comparison == 0:
        return "none"
    xmin, xmax = (first, second) if comparison < 0 else (second, first)
    midpoint_numerator = xmin.centroid_x_sum * xmax.pixel_count + xmax.centroid_x_sum * xmin.pixel_count
    midpoint_denominator = xmin.pixel_count * xmax.pixel_count
    for candidate in candidates:
        side_comparison = 2 * candidate.centroid_x_sum * midpoint_denominator - candidate.pixel_count * midpoint_numerator
        if side_comparison == 0:
            return "none"
        candidate.side = "xmin" if side_comparison < 0 else "xmax"
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
    cleaned_mask_sha256 = _sha256_bytes((labels > 0).astype(np.uint8).tobytes(order="C"))
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
        "native_variant_set_sha256": None,
        "variant_partitions": [],
        "base_projected_component_count": projected_count,
        "native_variant_projected_component_count": 0,
        "projected_component_count": projected_count,
    }
    return MaskComponentPlan(
        schema_version=MASK_COMPONENT_PLAN_VERSION,
        canvas_edge=int(canvas_edge),
        cleanup=descriptor,
        parts=parts,
        native_variant_set_sha256=None,
        variant_partitions=(),
        base_projected_component_count=projected_count,
        native_variant_projected_component_count=0,
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


def _variant_partition_record(
    candidate: NativeVariantCandidate,
    *,
    descriptor: MaskCleanupDescriptor,
    item_root: Path,
) -> NativeVariantPartitionRecord:
    import numpy as np

    x1, y1, x2, y2 = candidate.xyxy
    loaded = _LoadedVariantAlpha(
        part=_VariantPartAdapter(part_id=candidate.part_id, xyxy=candidate.xyxy),
        width=x2 - x1,
        height=y2 - y1,
        alpha_u8=candidate.alpha_u8,
    )
    labels = _label_components(_clean_binary_mask(loaded, descriptor))
    candidates = _extract_candidates(loaded, labels)
    if not candidates:
        content = {
            "schema_version": NATIVE_VARIANT_PARTITION_VERSION,
            "variant_id": candidate.variant_id,
            "part_id": candidate.part_id,
            "semantic_role": candidate.semantic_role,
            "status": "empty_after_cleanup",
            "side_provenance": "none",
            "source_xyxy": list(candidate.xyxy),
            "xyxy": None,
            "cleaned_binary_mask_sha256": None,
            "qcl_file": None,
            "components": [],
        }
        return NativeVariantPartitionRecord(
            schema_version=NATIVE_VARIANT_PARTITION_VERSION,
            variant_id=candidate.variant_id,
            part_id=candidate.part_id,
            semantic_role=candidate.semantic_role,
            status="empty_after_cleanup",
            side_provenance="none",
            source_xyxy=candidate.xyxy,
            xyxy=None,
            cleaned_binary_mask_sha256=None,
            qcl_file=None,
            components=(),
            partition_sha256=jcs_sha256(content),
        )

    side_provenance: Literal["role_single", "role_component_pair", "none"] = "none"
    if candidate.semantic_role in {"eye_closed.xmin", "eye_closed.xmax"}:
        side = candidate.semantic_role.rsplit(".", 1)[1]
        for component in candidates:
            component.side = side
        side_provenance = "role_single"
    elif candidate.semantic_role == "eye_closed.coupled" and len(candidates) == 2:
        total_pixels = sum(component.pixel_count for component in candidates)
        reliable = all(
            component.pixel_count * descriptor.side_min_component_fraction_denominator
            >= total_pixels * descriptor.side_min_component_fraction_numerator
            for component in candidates
        )
        first, second = candidates
        comparison = first.centroid_x_sum * second.pixel_count - second.centroid_x_sum * first.pixel_count
        if reliable and comparison != 0:
            xmin, xmax = (first, second) if comparison < 0 else (second, first)
            xmin.side = "xmin"
            xmax.side = "xmax"
            side_provenance = "role_component_pair"

    candidates.sort(key=lambda item: (item.bbox[1], item.bbox[0], item.mask_sha256))
    crop_x1 = min(item.bbox[0] for item in candidates)
    crop_y1 = min(item.bbox[1] for item in candidates)
    crop_x2 = max(item.bbox[2] for item in candidates)
    crop_y2 = max(item.bbox[3] for item in candidates)
    canonical_labels = np.zeros((crop_y2 - crop_y1, crop_x2 - crop_x1), dtype=np.uint32)
    components: list[MaskComponentRecord] = []
    for label_value, component in enumerate(candidates, start=1):
        cx1, cy1, cx2, cy2 = component.bbox
        target = canonical_labels[
            cy1 - crop_y1 : cy2 - crop_y1,
            cx1 - crop_x1 : cx2 - crop_x1,
        ]
        target[component.tight_mask] = label_value
        _, component_id = _component_identity(
            part_id=candidate.part_id,
            bbox=component.bbox,
            mask_sha256=component.mask_sha256,
        )
        components.append(
            MaskComponentRecord(
                component_id=component_id,
                label=label_value,
                part_id=candidate.part_id,
                side=component.side,
                bbox=component.bbox,
                cleaned_binary_mask_sha256=component.mask_sha256,
                pixel_count=component.pixel_count,
            )
        )
    cleaned_mask_sha256 = _sha256_bytes((canonical_labels > 0).astype(np.uint8).tobytes(order="C"))
    qcl_file = materialize_qcl(
        item_root,
        encode_qcl(
            canonical_labels.ravel(order="C"),
            width=canonical_labels.shape[1],
            height=canonical_labels.shape[0],
        ),
    )
    content = {
        "schema_version": NATIVE_VARIANT_PARTITION_VERSION,
        "variant_id": candidate.variant_id,
        "part_id": candidate.part_id,
        "semantic_role": candidate.semantic_role,
        "status": "ready",
        "side_provenance": side_provenance,
        "source_xyxy": list(candidate.xyxy),
        "xyxy": [crop_x1, crop_y1, crop_x2, crop_y2],
        "cleaned_binary_mask_sha256": cleaned_mask_sha256,
        "qcl_file": qcl_file.to_dict(),
        "components": [component.to_dict() for component in components],
    }
    return NativeVariantPartitionRecord(
        schema_version=NATIVE_VARIANT_PARTITION_VERSION,
        variant_id=candidate.variant_id,
        part_id=candidate.part_id,
        semantic_role=candidate.semantic_role,
        status="ready",
        side_provenance=side_provenance,
        source_xyxy=candidate.xyxy,
        xyxy=(crop_x1, crop_y1, crop_x2, crop_y2),
        cleaned_binary_mask_sha256=cleaned_mask_sha256,
        qcl_file=qcl_file,
        components=tuple(components),
        partition_sha256=jcs_sha256(content),
    )


def extend_mask_component_plan_with_variants(
    base_plan: MaskComponentPlan,
    variant_set: NativeVariantSet,
    item_root: str | Path,
) -> MaskComponentPlan:
    """Append candidate partitions without thresholding ordinary masks again."""

    if jcs_sha256(base_plan.semantic_payload()) != base_plan.plan_sha256:
        raise _error("invalid_component_plan", "base component-plan digest is invalid")
    if base_plan.native_variant_set_sha256 is not None or base_plan.variant_partitions:
        raise _error("invalid_component_plan", "component plan already contains NativeVariants")
    if jcs_sha256(variant_set.semantic_payload()) != variant_set.native_variant_set_sha256:
        raise _error("invalid_component_plan", "NativeVariant set digest is invalid")
    root = Path(item_root).resolve(strict=True)
    partitions = tuple(
        _variant_partition_record(entry, descriptor=base_plan.cleanup, item_root=root)
        for entry in sorted(variant_set.entries, key=lambda item: item.variant_id)
    )
    native_count = sum(len(partition.components) for partition in partitions)
    semantic_payload = {
        "schema_version": base_plan.schema_version,
        "canvas_edge": base_plan.canvas_edge,
        "cleanup": base_plan.cleanup.to_dict(),
        "parts": [part.to_dict() for part in base_plan.parts],
        "native_variant_set_sha256": variant_set.native_variant_set_sha256,
        "variant_partitions": [partition.to_dict() for partition in partitions],
        "base_projected_component_count": base_plan.base_projected_component_count,
        "native_variant_projected_component_count": native_count,
        "projected_component_count": base_plan.base_projected_component_count + native_count,
    }
    return MaskComponentPlan(
        schema_version=base_plan.schema_version,
        canvas_edge=base_plan.canvas_edge,
        cleanup=base_plan.cleanup,
        parts=base_plan.parts,
        native_variant_set_sha256=variant_set.native_variant_set_sha256,
        variant_partitions=partitions,
        base_projected_component_count=base_plan.base_projected_component_count,
        native_variant_projected_component_count=native_count,
        projected_component_count=base_plan.base_projected_component_count + native_count,
        plan_sha256=jcs_sha256(semantic_payload),
    )


__all__ = [
    "MASK_CLEANUP_SCHEMA",
    "MASK_COMPONENT_ID_SCHEMA",
    "MASK_COMPONENT_PLAN_VERSION",
    "MASK_SIDE_CLASSIFIER_VERSION",
    "NATIVE_VARIANT_PARTITION_VERSION",
    "SIDE_MIN_COMPONENT_FRACTION_DENOMINATOR",
    "SIDE_MIN_COMPONENT_FRACTION_NUMERATOR",
    "MaskCleanupDescriptor",
    "MaskComponentPlan",
    "MaskComponentPlanError",
    "MaskComponentRecord",
    "NormalizedMaskPart",
    "NativeVariantPartitionRecord",
    "build_base_mask_component_plan",
    "build_mask_component_plan",
    "cleanup_area_threshold",
    "extend_mask_component_plan_with_variants",
]
