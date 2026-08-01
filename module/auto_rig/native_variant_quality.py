from __future__ import annotations

import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from .artifacts import ArtifactContractError, sha256_file
from .component_plan import (
    MaskComponentPlan,
    NativeVariantPartitionRecord,
    NormalizedMaskPart,
)
from .draw_order import OrdinaryDrawOrderPlan
from .jcs import jcs_sha256
from .mask_sources import LoadedPartAlpha
from .native_variants import NativeVariantCandidate, NativeVariantSet
from .qcl import QclContractError, decode_qcl

NATIVE_VARIANT_ROLE_ENVELOPE_VERSION = "native-variant-role-envelope-v1"
BASE_FEATURE_SCALE_KIND = "sqrt_base_alpha_mass_v1"
NATIVE_VARIANT_COMPOSITE_PLAN_VERSION = "native-variant-composite-plan-v1"
NATIVE_VARIANT_QUALITY_PLAN_VERSION = "native-variant-quality-plan-v1"


class NativeVariantQualityError(ValueError):
    """Raised when NativeVariant quality inputs or registries are invalid."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class NativeVariantRoleEnvelope:
    semantic_role: str
    support_mode: Literal["support_preserving", "support_expanding"]
    k_numerator: int
    k_denominator: int
    c_numerator: int
    c_denominator: int
    radius_min_px: int
    radius_max_px: int

    def to_dict(self) -> dict[str, object]:
        return {
            "semantic_role": self.semantic_role,
            "support_mode": self.support_mode,
            "k_numerator": self.k_numerator,
            "k_denominator": self.k_denominator,
            "c_numerator": self.c_numerator,
            "c_denominator": self.c_denominator,
            "radius_min_px": self.radius_min_px,
            "radius_max_px": self.radius_max_px,
        }


@dataclass(frozen=True, slots=True)
class IntrusionPartMetric:
    part_id: str
    base_tag: str
    ratio: float

    def to_dict(self) -> dict[str, object]:
        return {
            "part_id": self.part_id,
            "base_tag": self.base_tag,
            "ratio": self.ratio,
        }


@dataclass(frozen=True, slots=True)
class NativeVariantBranchMetrics:
    branch_id: str
    side: Literal["xmin", "xmax"] | None
    composite_plan_id: str
    base_part_ids: tuple[str, ...]
    variant_component_ids: tuple[str, ...]
    coverage_leak: float
    base_alpha_mass_u8_sum: int
    variant_alpha_mass_u8_sum: int
    alpha_mass_ratio: float
    role_envelope_id: str
    base_feature_scale_kind: str
    base_support_bbox: tuple[int, int, int, int]
    base_support_aspect_ratio: float
    base_context_radius_px: int
    intrusion_face_ratio: float
    intrusion_other_parts_ratio: float
    intrusion_by_part: tuple[IntrusionPartMetric, ...]
    occlusion_intrusion: float
    spill_mass_diagnostic: float
    component_partition_digest: str
    eligible: bool
    rejection_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "branch_id": self.branch_id,
            "side": self.side,
            "composite_plan_id": self.composite_plan_id,
            "base_part_ids": list(self.base_part_ids),
            "variant_component_ids": list(self.variant_component_ids),
            "coverage_leak": self.coverage_leak,
            "base_alpha_mass_u8_sum": self.base_alpha_mass_u8_sum,
            "variant_alpha_mass_u8_sum": self.variant_alpha_mass_u8_sum,
            "alpha_mass_ratio": self.alpha_mass_ratio,
            "role_envelope_id": self.role_envelope_id,
            "base_feature_scale_kind": self.base_feature_scale_kind,
            "base_support_bbox": list(self.base_support_bbox),
            "base_support_aspect_ratio": self.base_support_aspect_ratio,
            "base_context_radius_px": self.base_context_radius_px,
            "intrusion_face_ratio": self.intrusion_face_ratio,
            "intrusion_other_parts_ratio": self.intrusion_other_parts_ratio,
            "intrusion_by_part": [entry.to_dict() for entry in self.intrusion_by_part],
            "occlusion_intrusion": self.occlusion_intrusion,
            "spill_mass_diagnostic": self.spill_mass_diagnostic,
            "component_partition_digest": self.component_partition_digest,
            "eligible": self.eligible,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True, slots=True)
class NativeVariantQualityResult:
    variant_id: str
    semantic_role: str
    eligible: bool
    rejection_reason: str | None
    branches: tuple[NativeVariantBranchMetrics, ...]
    result_sha256: str

    def content_payload(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "semantic_role": self.semantic_role,
            "eligible": self.eligible,
            "rejection_reason": self.rejection_reason,
            "branches": [branch.to_dict() for branch in self.branches],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_payload(), "result_sha256": self.result_sha256}


@dataclass(frozen=True, slots=True)
class NativeVariantBundleQualityResult:
    bundle_id: str
    semantic_roles: tuple[str, ...]
    variant_ids: tuple[str, ...]
    complete: bool
    eligible: bool
    rejection_reason: str | None
    result_sha256: str

    def content_payload(self) -> dict[str, object]:
        return {
            "bundle_id": self.bundle_id,
            "semantic_roles": list(self.semantic_roles),
            "variant_ids": list(self.variant_ids),
            "complete": self.complete,
            "eligible": self.eligible,
            "rejection_reason": self.rejection_reason,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_payload(), "result_sha256": self.result_sha256}


@dataclass(frozen=True, slots=True)
class NativeVariantQualityPlan:
    schema_version: str
    native_variant_set_sha256: str
    component_plan_sha256: str
    draw_order_plan_sha256: str
    role_registry_sha256: str
    composite_plan_version: str
    candidate_results: tuple[NativeVariantQualityResult, ...]
    bundle_results: tuple[NativeVariantBundleQualityResult, ...]
    quality_eligible_variant_ids: tuple[str, ...]
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "native_variant_set_sha256": self.native_variant_set_sha256,
            "component_plan_sha256": self.component_plan_sha256,
            "draw_order_plan_sha256": self.draw_order_plan_sha256,
            "role_registry_sha256": self.role_registry_sha256,
            "composite_plan_version": self.composite_plan_version,
            "candidate_results": [result.to_dict() for result in self.candidate_results],
            "bundle_results": [result.to_dict() for result in self.bundle_results],
            "quality_eligible_variant_ids": list(self.quality_eligible_variant_ids),
        }


_EYE_ROLES = ("eye_closed.xmin", "eye_closed.xmax", "eye_closed.coupled")
_MOUTH_FORM_ROLES = ("mouth_smile", "mouth_frown")

BUILTIN_NATIVE_VARIANT_ROLE_ENVELOPES = tuple(
    NativeVariantRoleEnvelope(
        semantic_role=role,
        support_mode="support_preserving",
        k_numerator=3,
        k_denominator=2,
        c_numerator=1,
        c_denominator=5,
        radius_min_px=2,
        radius_max_px=12,
    )
    for role in _EYE_ROLES
) + tuple(
    NativeVariantRoleEnvelope(
        semantic_role=role,
        support_mode="support_expanding",
        k_numerator=4,
        k_denominator=1,
        c_numerator=1,
        c_denominator=1,
        radius_min_px=4,
        radius_max_px=24,
    )
    for role in _MOUTH_FORM_ROLES
) + (
    NativeVariantRoleEnvelope(
        semantic_role="mouth_open",
        support_mode="support_expanding",
        k_numerator=12,
        k_denominator=1,
        c_numerator=2,
        c_denominator=1,
        radius_min_px=8,
        radius_max_px=48,
    ),
)

_BUILTIN_BY_ROLE = {
    row.semantic_role: row for row in BUILTIN_NATIVE_VARIANT_ROLE_ENVELOPES
}


def _registry_error(message: str) -> NativeVariantQualityError:
    return NativeVariantQualityError("invalid_primitive_registry", message)


def native_variant_role_registry_payload(
    rows: Iterable[NativeVariantRoleEnvelope] = BUILTIN_NATIVE_VARIANT_ROLE_ENVELOPES,
) -> dict[str, object]:
    normalized = tuple(rows)
    return {
        "schema_version": NATIVE_VARIANT_ROLE_ENVELOPE_VERSION,
        "base_feature_scale_kind": BASE_FEATURE_SCALE_KIND,
        "role_envelopes": [
            row.to_dict() for row in sorted(normalized, key=lambda row: row.semantic_role)
        ],
    }


def validate_native_variant_role_registry(
    rows: Iterable[NativeVariantRoleEnvelope] = BUILTIN_NATIVE_VARIANT_ROLE_ENVELOPES,
) -> str:
    normalized = tuple(rows)
    if any(not isinstance(row, NativeVariantRoleEnvelope) for row in normalized):
        raise _registry_error("role-envelope rows must use NativeVariantRoleEnvelope")
    roles = tuple(row.semantic_role for row in normalized)
    if len(roles) != len(set(roles)):
        raise _registry_error("role-envelope semantic roles must be unique")
    if set(roles) != set(_BUILTIN_BY_ROLE):
        raise _registry_error("role-envelope semantic role set differs from v1")
    for row in normalized:
        integer_values = (
            row.k_numerator,
            row.k_denominator,
            row.c_numerator,
            row.c_denominator,
            row.radius_min_px,
            row.radius_max_px,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) <= 0
            for value in integer_values
        ):
            raise _registry_error("role-envelope rationals and clamps must be positive integers")
        if row.radius_min_px > row.radius_max_px:
            raise _registry_error("role-envelope radius_min_px exceeds radius_max_px")
        if row != _BUILTIN_BY_ROLE[row.semantic_role]:
            raise _registry_error("role-envelope row differs from frozen v1 semantics")
    return jcs_sha256(native_variant_role_registry_payload(normalized))


def base_context_radius_px(
    semantic_role: str,
    *,
    base_alpha_mass_u8_sum: int,
) -> int:
    """Round c*sqrt(mass_u8/255) half-up using integer inequalities."""

    envelope = _BUILTIN_BY_ROLE.get(semantic_role)
    if envelope is None:
        raise NativeVariantQualityError("invalid_quality_input", f"unknown role: {semantic_role}")
    if (
        isinstance(base_alpha_mass_u8_sum, bool)
        or not isinstance(base_alpha_mass_u8_sum, numbers.Integral)
        or int(base_alpha_mass_u8_sum) <= 0
    ):
        raise NativeVariantQualityError(
            "invalid_quality_input",
            "base alpha mass must be a positive integer",
        )
    mass_u8 = int(base_alpha_mass_u8_sum)
    rounded = 0
    for candidate in range(1, envelope.radius_max_px + 1):
        left = 4 * envelope.c_numerator**2 * mass_u8
        right = (
            envelope.c_denominator**2
            * 255
            * (2 * candidate - 1) ** 2
        )
        if left < right:
            break
        rounded = candidate
    return min(
        envelope.radius_max_px,
        max(envelope.radius_min_px, rounded),
    )


def _quality_error(message: str) -> NativeVariantQualityError:
    return NativeVariantQualityError("invalid_quality_input", message)


def _load_label_array(
    item_root: Path,
    part: NormalizedMaskPart | NativeVariantPartitionRecord,
):
    import numpy as np

    if part.xyxy is None or part.qcl_file is None:
        raise _quality_error("component partition has no QCL support")
    path = item_root / Path(*part.qcl_file.path.split("/"))
    try:
        actual_sha = sha256_file(path)
        payload = path.read_bytes()
    except (ArtifactContractError, OSError) as exc:
        raise _quality_error("component QCL is unavailable") from exc
    if actual_sha != part.qcl_file.sha256 or len(payload) != part.qcl_file.size:
        raise _quality_error("component QCL digest differs from the plan")
    try:
        decoded = decode_qcl(payload)
    except QclContractError as exc:
        raise _quality_error("component QCL is invalid") from exc
    x1, y1, x2, y2 = part.xyxy
    if (decoded.width, decoded.height) != (x2 - x1, y2 - y1):
        raise _quality_error("component QCL dimensions differ from the plan")
    labels = np.asarray(decoded.labels, dtype=np.uint32).reshape(
        decoded.height,
        decoded.width,
    )
    expected_labels = {component.label for component in part.components}
    actual_labels = {int(value) for value in np.unique(labels) if int(value) > 0}
    if actual_labels != expected_labels:
        raise _quality_error("component QCL labels differ from the plan")
    return labels


def _ordinary_alpha_canvas(
    part: NormalizedMaskPart,
    loaded: LoadedPartAlpha,
    *,
    item_root: Path,
    canvas_edge: int,
):
    import numpy as np

    labels = _load_label_array(item_root, part)
    source = np.frombuffer(loaded.alpha_u8, dtype=np.uint8).reshape(
        loaded.height,
        loaded.width,
    )
    px1, py1, px2, py2 = part.xyxy
    sx1, sy1, sx2, sy2 = part.source_xyxy
    if loaded.part.xyxy != part.source_xyxy or source.shape != (sy2 - sy1, sx2 - sx1):
        raise _quality_error("loaded Part alpha differs from the component-plan source")
    crop = source[py1 - sy1 : py2 - sy1, px1 - sx1 : px2 - sx1]
    if crop.shape != labels.shape:
        raise _quality_error("loaded Part alpha crop differs from the QCL support")
    canvas = np.zeros((canvas_edge, canvas_edge), dtype=np.uint8)
    canvas[py1:py2, px1:px2] = np.where(labels > 0, crop, 0)
    return canvas


def _variant_alpha_canvas(
    candidate: NativeVariantCandidate,
    partition: NativeVariantPartitionRecord,
    *,
    item_root: Path,
    canvas_edge: int,
    component_labels: set[int],
):
    import numpy as np

    labels = _load_label_array(item_root, partition)
    source_x1, source_y1, source_x2, source_y2 = partition.source_xyxy
    source = np.frombuffer(candidate.alpha_u8, dtype=np.uint8).reshape(
        source_y2 - source_y1,
        source_x2 - source_x1,
    )
    if candidate.xyxy != partition.source_xyxy:
        raise _quality_error("variant partition belongs to different source geometry")
    px1, py1, px2, py2 = partition.xyxy or (0, 0, 0, 0)
    crop = source[
        py1 - source_y1 : py2 - source_y1,
        px1 - source_x1 : px2 - source_x1,
    ]
    selected = np.isin(labels, tuple(sorted(component_labels)))
    canvas = np.zeros((canvas_edge, canvas_edge), dtype=np.uint8)
    canvas[py1:py2, px1:px2] = np.where(selected, crop, 0)
    return canvas, selected


def _source_over_alpha(alphas: Iterable[object], *, canvas_edge: int):
    import numpy as np

    result = np.zeros((canvas_edge, canvas_edge), dtype=np.uint16)
    for alpha in alphas:
        foreground = np.asarray(alpha, dtype=np.uint32)
        background = result.astype(np.uint32)
        overlap = (foreground * background + 127) // 255
        result = (foreground + background - overlap).astype(np.uint16)
    return result


def _disk_dilate(mask, radius: int):
    import numpy as np
    from scipy.ndimage import binary_dilation

    coordinates = np.arange(-radius, radius + 1, dtype=np.int32)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    footprint = xx * xx + yy * yy <= radius * radius
    return binary_dilation(mask, structure=footprint)


def _support_bbox(support) -> tuple[int, int, int, int]:
    import numpy as np

    ys, xs = np.nonzero(support)
    if len(xs) == 0:
        raise _quality_error("base component support is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _visible_contributions(
    prefix: tuple[str, ...],
    ordinary_alphas: dict[str, object],
    *,
    canvas_edge: int,
) -> dict[str, object]:
    import numpy as np

    transmission = np.full((canvas_edge, canvas_edge), 255, dtype=np.uint16)
    contributions: dict[str, object] = {}
    for part_id in reversed(prefix):
        alpha = np.asarray(ordinary_alphas[part_id], dtype=np.uint32)
        transmitted = transmission.astype(np.uint32)
        contributions[part_id] = ((alpha * transmitted + 127) // 255).astype(np.uint16)
        transmission = ((transmitted * (255 - alpha) + 127) // 255).astype(np.uint16)
    return contributions


def _evaluate_branch(
    candidate: NativeVariantCandidate,
    partition: NativeVariantPartitionRecord,
    *,
    side: Literal["xmin", "xmax"] | None,
    component_labels: set[int],
    base_part_ids: tuple[str, ...],
    ordinary_parts: dict[str, NormalizedMaskPart],
    ordinary_alphas: dict[str, object],
    draw_order: OrdinaryDrawOrderPlan,
    item_root: Path,
    canvas_edge: int,
) -> NativeVariantBranchMetrics:
    import numpy as np

    if not component_labels or not base_part_ids:
        raise _quality_error("quality branch requires variant components and base Parts")
    base_support = np.zeros((canvas_edge, canvas_edge), dtype=bool)
    for part_id in base_part_ids:
        part = ordinary_parts[part_id]
        px1, py1, px2, py2 = part.xyxy
        base_support[py1:py2, px1:px2] |= _load_label_array(item_root, part) > 0
    base_alpha = _source_over_alpha(
        (ordinary_alphas[part_id] for part_id in sorted(base_part_ids)),
        canvas_edge=canvas_edge,
    )
    variant_alpha, _ = _variant_alpha_canvas(
        candidate,
        partition,
        item_root=item_root,
        canvas_edge=canvas_edge,
        component_labels=component_labels,
    )
    base_mass = int(base_alpha.sum(dtype=np.uint64))
    variant_mass = int(variant_alpha.sum(dtype=np.uint64))
    if base_mass <= 0 or variant_mass <= 0:
        raise _quality_error("quality branch alpha mass must be positive")
    coverage_numerator = int(
        (
            base_alpha.astype(np.uint64)
            * (255 - variant_alpha.astype(np.uint64))
        ).sum(dtype=np.uint64)
    )
    coverage_denominator = base_mass * 255
    coverage_leak = coverage_numerator / coverage_denominator
    alpha_mass_ratio = variant_mass / base_mass
    envelope = _BUILTIN_BY_ROLE[candidate.semantic_role]
    radius = base_context_radius_px(
        candidate.semantic_role,
        base_alpha_mass_u8_sum=base_mass,
    )
    authorized_face = _disk_dilate(base_support, radius)
    bbox = _support_bbox(base_support)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    aspect_ratio = max(width, height) / min(width, height)
    anchor_index = draw_order.ordinary_part_order.index(candidate.draw_anchor_part_id)
    prefix = tuple(draw_order.ordinary_part_order[: anchor_index + 1])
    contributions = _visible_contributions(
        prefix,
        ordinary_alphas,
        canvas_edge=canvas_edge,
    )
    intrusion_denominator = variant_mass * 255
    intrusion_metrics: list[IntrusionPartMetric] = []
    face_numerator = 0
    other_numerator = 0
    variant_u64 = variant_alpha.astype(np.uint64)
    for part_id in sorted(prefix):
        part = ordinary_parts[part_id]
        contribution = np.asarray(contributions[part_id], dtype=np.uint64)
        if part_id in base_part_ids:
            unauthorized = np.zeros_like(contribution)
        elif part.base_tag == "face":
            unauthorized = np.where(authorized_face, 0, contribution)
        else:
            unauthorized = contribution
        numerator = int((variant_u64 * unauthorized).sum(dtype=np.uint64))
        if part.base_tag == "face":
            face_numerator += numerator
        else:
            other_numerator += numerator
        intrusion_metrics.append(
            IntrusionPartMetric(
                part_id=part_id,
                base_tag=part.base_tag,
                ratio=numerator / intrusion_denominator,
            )
        )
    total_intrusion_numerator = face_numerator + other_numerator
    occlusion_intrusion = total_intrusion_numerator / intrusion_denominator
    spill_radius = max(4, (max(width, height) + 1) // 2)
    spill_support = _disk_dilate(base_support, spill_radius)
    spill_mass = int(variant_alpha[~spill_support].sum(dtype=np.uint64)) / variant_mass

    if coverage_numerator * 100 > coverage_denominator:
        reason = "coverage_leak"
    elif variant_mass * envelope.k_denominator > base_mass * envelope.k_numerator:
        reason = "alpha_mass_ratio_exceeded"
    elif total_intrusion_numerator * 100 > intrusion_denominator:
        reason = "occlusion_intrusion"
    else:
        reason = None
    component_ids = tuple(
        sorted(
            component.component_id
            for component in partition.components
            if component.label in component_labels
        )
    )
    identity = {
        "schema_version": NATIVE_VARIANT_COMPOSITE_PLAN_VERSION,
        "variant_id": candidate.variant_id,
        "side": side,
        "base_part_ids": list(base_part_ids),
        "variant_component_ids": list(component_ids),
        "component_partition_digest": partition.partition_sha256,
    }
    branch_token = side or "all"
    return NativeVariantBranchMetrics(
        branch_id=f"branch/{candidate.variant_id}.{branch_token}",
        side=side,
        composite_plan_id="composite/c_" + jcs_sha256(identity).removeprefix("sha256:"),
        base_part_ids=base_part_ids,
        variant_component_ids=component_ids,
        coverage_leak=coverage_leak,
        base_alpha_mass_u8_sum=base_mass,
        variant_alpha_mass_u8_sum=variant_mass,
        alpha_mass_ratio=alpha_mass_ratio,
        role_envelope_id=f"envelope/{candidate.semantic_role}",
        base_feature_scale_kind=BASE_FEATURE_SCALE_KIND,
        base_support_bbox=bbox,
        base_support_aspect_ratio=aspect_ratio,
        base_context_radius_px=radius,
        intrusion_face_ratio=face_numerator / intrusion_denominator,
        intrusion_other_parts_ratio=other_numerator / intrusion_denominator,
        intrusion_by_part=tuple(intrusion_metrics),
        occlusion_intrusion=occlusion_intrusion,
        spill_mass_diagnostic=spill_mass,
        component_partition_digest=partition.partition_sha256,
        eligible=reason is None,
        rejection_reason=reason,
    )


def _result(
    candidate: NativeVariantCandidate,
    branches: tuple[NativeVariantBranchMetrics, ...],
    *,
    rejection_reason: str | None,
) -> NativeVariantQualityResult:
    content = {
        "variant_id": candidate.variant_id,
        "semantic_role": candidate.semantic_role,
        "eligible": rejection_reason is None,
        "rejection_reason": rejection_reason,
        "branches": [branch.to_dict() for branch in branches],
    }
    return NativeVariantQualityResult(
        variant_id=candidate.variant_id,
        semantic_role=candidate.semantic_role,
        eligible=rejection_reason is None,
        rejection_reason=rejection_reason,
        branches=branches,
        result_sha256=jcs_sha256(content),
    )


def _candidate_branch_specs(
    candidate: NativeVariantCandidate,
    partition: NativeVariantPartitionRecord,
    ordinary_parts: dict[str, NormalizedMaskPart],
) -> tuple[
    tuple[
        Literal["xmin", "xmax"] | None,
        set[int],
        tuple[str, ...],
    ],
    ...,
] | None:
    if partition.status != "ready" or not partition.components:
        return None
    if candidate.semantic_role == "eye_closed.coupled":
        if partition.side_provenance != "role_component_pair" or len(partition.components) != 2:
            return None
        specs = []
        for side in ("xmin", "xmax"):
            components = tuple(
                component for component in partition.components if component.side == side
            )
            base_ids = tuple(
                sorted(
                    part_id
                    for part_id in candidate.base_part_ids
                    if ordinary_parts[part_id].side == side
                )
            )
            if len(components) != 1 or not base_ids:
                return None
            specs.append((side, {components[0].label}, base_ids))
        if any(ordinary_parts[part_id].side not in {"xmin", "xmax"} for part_id in candidate.base_part_ids):
            return None
        return tuple(specs)
    if candidate.semantic_role in {"eye_closed.xmin", "eye_closed.xmax"}:
        side = candidate.semantic_role.rsplit(".", 1)[1]
        if partition.side_provenance != "role_single" or any(
            component.side != side for component in partition.components
        ):
            return None
        if any(ordinary_parts[part_id].side != side for part_id in candidate.base_part_ids):
            return None
        return ((side, {component.label for component in partition.components}, candidate.base_part_ids),)
    if any(component.side is not None for component in partition.components):
        return None
    return ((None, {component.label for component in partition.components}, candidate.base_part_ids),)


def evaluate_native_variant_candidates(
    loaded_parts: Iterable[LoadedPartAlpha],
    component_plan: MaskComponentPlan,
    draw_order_plan: OrdinaryDrawOrderPlan,
    variant_set: NativeVariantSet,
    *,
    item_root: str | Path,
) -> tuple[NativeVariantQualityResult, ...]:
    """Evaluate manifest-valid candidates without making resource decisions."""

    validate_native_variant_role_registry()
    if jcs_sha256(component_plan.semantic_payload()) != component_plan.plan_sha256:
        raise _quality_error("component-plan digest is invalid")
    if jcs_sha256(draw_order_plan.semantic_payload()) != draw_order_plan.plan_sha256:
        raise _quality_error("draw-order digest is invalid")
    if jcs_sha256(variant_set.semantic_payload()) != variant_set.native_variant_set_sha256:
        raise _quality_error("NativeVariant set digest is invalid")
    if component_plan.native_variant_set_sha256 != variant_set.native_variant_set_sha256:
        raise _quality_error("component plan belongs to a different NativeVariant set")
    root = Path(item_root).resolve(strict=True)
    ordinary_parts = {part.part_id: part for part in component_plan.parts}
    if set(ordinary_parts) != set(draw_order_plan.ordinary_part_order):
        raise _quality_error("draw-order Parts differ from component-plan Parts")
    loaded_by_source = {loaded.part.part_id: loaded for loaded in loaded_parts}
    expected_source_ids = {part.source_part_id for part in component_plan.parts}
    if set(loaded_by_source) != expected_source_ids:
        raise _quality_error("loaded Part alpha set differs from the component plan")
    ordinary_alphas = {
        part.part_id: _ordinary_alpha_canvas(
            part,
            loaded_by_source[part.source_part_id],
            item_root=root,
            canvas_edge=component_plan.canvas_edge,
        )
        for part in component_plan.parts
    }
    partitions = {partition.variant_id: partition for partition in component_plan.variant_partitions}
    if set(partitions) != {candidate.variant_id for candidate in variant_set.entries}:
        raise _quality_error("variant partition set differs from the NativeVariant set")
    results: list[NativeVariantQualityResult] = []
    for candidate in sorted(variant_set.entries, key=lambda item: item.variant_id):
        partition = partitions[candidate.variant_id]
        if jcs_sha256(partition.content_payload()) != partition.partition_sha256:
            raise _quality_error("variant partition digest is invalid")
        branch_specs = _candidate_branch_specs(candidate, partition, ordinary_parts)
        if branch_specs is None:
            results.append(_result(candidate, (), rejection_reason="component_partition"))
            continue
        branches = tuple(
            _evaluate_branch(
                candidate,
                partition,
                side=side,
                component_labels=labels,
                base_part_ids=base_ids,
                ordinary_parts=ordinary_parts,
                ordinary_alphas=ordinary_alphas,
                draw_order=draw_order_plan,
                item_root=root,
                canvas_edge=component_plan.canvas_edge,
            )
            for side, labels, base_ids in branch_specs
        )
        rejection_priority = (
            "coverage_leak",
            "alpha_mass_ratio_exceeded",
            "occlusion_intrusion",
        )
        rejected = {branch.rejection_reason for branch in branches if not branch.eligible}
        rejection_reason = next(
            (reason for reason in rejection_priority if reason in rejected),
            None,
        )
        results.append(
            _result(
                candidate,
                branches,
                rejection_reason=rejection_reason,
            )
        )
    return tuple(results)


def _bundle_result(
    bundle_id: str,
    candidate_results: tuple[NativeVariantQualityResult, ...],
    *,
    complete: bool,
) -> NativeVariantBundleQualityResult:
    rejection_priority = (
        "component_partition",
        "coverage_leak",
        "alpha_mass_ratio_exceeded",
        "occlusion_intrusion",
    )
    if not complete:
        reason = "bundle_incomplete"
    else:
        rejected = {result.rejection_reason for result in candidate_results if not result.eligible}
        reason = next((item for item in rejection_priority if item in rejected), None)
    content = {
        "bundle_id": bundle_id,
        "semantic_roles": sorted(result.semantic_role for result in candidate_results),
        "variant_ids": sorted(result.variant_id for result in candidate_results),
        "complete": complete,
        "eligible": complete and reason is None,
        "rejection_reason": reason,
    }
    return NativeVariantBundleQualityResult(
        bundle_id=bundle_id,
        semantic_roles=tuple(content["semantic_roles"]),
        variant_ids=tuple(content["variant_ids"]),
        complete=complete,
        eligible=bool(content["eligible"]),
        rejection_reason=reason,
        result_sha256=jcs_sha256(content),
    )


def _build_bundle_results(
    candidate_results: tuple[NativeVariantQualityResult, ...],
) -> tuple[NativeVariantBundleQualityResult, ...]:
    by_role = {result.semantic_role: result for result in candidate_results}
    if len(by_role) != len(candidate_results):
        raise _quality_error("candidate quality results contain duplicate roles")
    bundles: list[NativeVariantBundleQualityResult] = []
    eye_roles = set(by_role) & set(_EYE_ROLES)
    if eye_roles:
        if "eye_closed.coupled" in eye_roles and len(eye_roles) != 1:
            raise _quality_error("coupled and single blink results overlap")
        complete = eye_roles in (
            {"eye_closed.coupled"},
            {"eye_closed.xmin", "eye_closed.xmax"},
        )
        bundles.append(
            _bundle_result(
                "blink.native",
                tuple(by_role[role] for role in sorted(eye_roles)),
                complete=complete,
            )
        )
    if "mouth_open" in by_role:
        bundles.append(
            _bundle_result(
                "mouth_open.native",
                (by_role["mouth_open"],),
                complete=True,
            )
        )
    mouth_form_roles = set(by_role) & set(_MOUTH_FORM_ROLES)
    if mouth_form_roles:
        bundles.append(
            _bundle_result(
                "mouth_form.native",
                tuple(by_role[role] for role in sorted(mouth_form_roles)),
                complete=mouth_form_roles == set(_MOUTH_FORM_ROLES),
            )
        )
    priority = {
        "blink.native": 0,
        "mouth_open.native": 1,
        "mouth_form.native": 2,
    }
    return tuple(sorted(bundles, key=lambda bundle: priority[bundle.bundle_id]))


def build_native_variant_quality_plan(
    loaded_parts: Iterable[LoadedPartAlpha],
    component_plan: MaskComponentPlan,
    draw_order_plan: OrdinaryDrawOrderPlan,
    variant_set: NativeVariantSet,
    *,
    item_root: str | Path,
) -> NativeVariantQualityPlan:
    """Freeze profile-independent candidate and atomic-bundle quality facts."""

    candidate_results = evaluate_native_variant_candidates(
        loaded_parts,
        component_plan,
        draw_order_plan,
        variant_set,
        item_root=item_root,
    )
    for result in candidate_results:
        if jcs_sha256(result.content_payload()) != result.result_sha256:
            raise _quality_error("candidate quality-result digest is invalid")
    bundle_results = _build_bundle_results(candidate_results)
    for result in bundle_results:
        if jcs_sha256(result.content_payload()) != result.result_sha256:
            raise _quality_error("bundle quality-result digest is invalid")
    eligible_ids = tuple(
        sorted(
            variant_id
            for bundle in bundle_results
            if bundle.eligible
            for variant_id in bundle.variant_ids
        )
    )
    role_registry_sha = validate_native_variant_role_registry()
    semantic_payload = {
        "schema_version": NATIVE_VARIANT_QUALITY_PLAN_VERSION,
        "native_variant_set_sha256": variant_set.native_variant_set_sha256,
        "component_plan_sha256": component_plan.plan_sha256,
        "draw_order_plan_sha256": draw_order_plan.plan_sha256,
        "role_registry_sha256": role_registry_sha,
        "composite_plan_version": NATIVE_VARIANT_COMPOSITE_PLAN_VERSION,
        "candidate_results": [result.to_dict() for result in candidate_results],
        "bundle_results": [result.to_dict() for result in bundle_results],
        "quality_eligible_variant_ids": list(eligible_ids),
    }
    return NativeVariantQualityPlan(
        schema_version=NATIVE_VARIANT_QUALITY_PLAN_VERSION,
        native_variant_set_sha256=variant_set.native_variant_set_sha256,
        component_plan_sha256=component_plan.plan_sha256,
        draw_order_plan_sha256=draw_order_plan.plan_sha256,
        role_registry_sha256=role_registry_sha,
        composite_plan_version=NATIVE_VARIANT_COMPOSITE_PLAN_VERSION,
        candidate_results=candidate_results,
        bundle_results=bundle_results,
        quality_eligible_variant_ids=eligible_ids,
        plan_sha256=jcs_sha256(semantic_payload),
    )


__all__ = [
    "BASE_FEATURE_SCALE_KIND",
    "BUILTIN_NATIVE_VARIANT_ROLE_ENVELOPES",
    "NATIVE_VARIANT_COMPOSITE_PLAN_VERSION",
    "NATIVE_VARIANT_QUALITY_PLAN_VERSION",
    "NATIVE_VARIANT_ROLE_ENVELOPE_VERSION",
    "IntrusionPartMetric",
    "NativeVariantBranchMetrics",
    "NativeVariantBundleQualityResult",
    "NativeVariantQualityError",
    "NativeVariantQualityResult",
    "NativeVariantQualityPlan",
    "NativeVariantRoleEnvelope",
    "base_context_radius_px",
    "build_native_variant_quality_plan",
    "evaluate_native_variant_candidates",
    "native_variant_role_registry_payload",
    "validate_native_variant_role_registry",
]
