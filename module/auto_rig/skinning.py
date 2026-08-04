from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Literal

from .bone_graph import BoneGraphPlan, validate_bone_graph
from .component_geometry import MESH_COMPONENT_SOURCE_VERSION, MeshComponentSource
from .component_plan import NormalizedMaskPart
from .jcs import jcs_sha256
from .joint_pipeline import StageAJointPlan, validate_stage_a_joint_plan
from .mesh_builder import MeshBuildPlan, MeshRecord, validate_mesh_plan
from .native_variants import NativeVariantSet

PART_BONE_BINDING_REGISTRY_VERSION = "part-bone-binding-registry-v2"
SKINNING_PLAN_VERSION = "skinning-plan-v3"
SKINNING_DESCRIPTOR_VERSION = "skinning-descriptor-v3"
SKINNING_ARC_PROJECTION_VERSION = "component-chain-assignment-v3"
SKINNING_TRANSITION_RADIUS_NUMERATOR = 1
SKINNING_TRANSITION_RADIUS_DENOMINATOR = 1
SKINNING_WEIGHT_DENOMINATOR = 1_000_000
SKINNING_MIN_INFLUENCE_UNITS = 1
SKINNING_MAX_INFLUENCES = 4

_HEAD_TAGS = (
    "front hair",
    "back hair",
    "headwear",
    "face",
    "irides",
    "eyebrow",
    "eyewhite",
    "eyelash",
    "eyewear",
    "ears",
    "earwear",
    "nose",
    "mouth",
)


class SkinningError(ValueError):
    """Raised when semantic binding or skin weights violate the shared Rig contract."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _error(code: str, message: str) -> SkinningError:
    return SkinningError(code, message)


@dataclass(frozen=True, slots=True)
class PartBoneBindingRule:
    rule_id: str
    base_tags: tuple[str, ...]
    mode: Literal["rigid", "limb", "dynamic"]
    target: str
    dynamic_candidate: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "base_tags": list(self.base_tags),
            "mode": self.mode,
            "target": self.target,
            "dynamic_candidate": self.dynamic_candidate,
        }


PART_BONE_BINDING_RULES = (
    PartBoneBindingRule("part-bone/head", _HEAD_TAGS, "rigid", "head", False),
    PartBoneBindingRule("part-bone/neck", ("neck", "neckwear"), "rigid", "neck", False),
    PartBoneBindingRule("part-bone/torso", ("topwear",), "rigid", "torso", False),
    PartBoneBindingRule(
        "part-bone/lower-torso",
        ("bottomwear",),
        "rigid",
        "lower_torso",
        False,
    ),
    PartBoneBindingRule("part-bone/arm", ("handwear",), "limb", "arm", False),
    PartBoneBindingRule(
        "part-bone/leg",
        ("legwear", "footwear"),
        "limb",
        "leg",
        False,
    ),
    PartBoneBindingRule(
        "part-bone/dynamic",
        ("tail", "wings", "objects"),
        "dynamic",
        "torso",
        True,
    ),
)


@dataclass(frozen=True, slots=True)
class PartBoneBinding:
    part_id: str
    source_kind: Literal["see_through", "native_variant"]
    variant_id: str | None
    anchor_part_id: str | None
    base_tag: str | None
    side: Literal["xmin", "xmax"] | None
    rule_id: str | None
    mode: Literal["rigid", "limb", "dynamic", "unknown"]
    semantic_candidate_bone_ids: tuple[str, ...]
    usable_bone_ids: tuple[str, ...]
    fallback_bone_ids: tuple[str, ...]
    selected_rigid_bone_id: str
    dynamic_candidate: bool
    binding_sha256: str

    def content_payload(self) -> dict[str, object]:
        return {
            "part_id": self.part_id,
            "source_kind": self.source_kind,
            "variant_id": self.variant_id,
            "anchor_part_id": self.anchor_part_id,
            "base_tag": self.base_tag,
            "side": self.side,
            "rule_id": self.rule_id,
            "mode": self.mode,
            "semantic_candidate_bone_ids": list(self.semantic_candidate_bone_ids),
            "usable_bone_ids": list(self.usable_bone_ids),
            "fallback_bone_ids": list(self.fallback_bone_ids),
            "selected_rigid_bone_id": self.selected_rigid_bone_id,
            "dynamic_candidate": self.dynamic_candidate,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_payload(), "binding_sha256": self.binding_sha256}


@dataclass(frozen=True, slots=True)
class SkinningDescriptor:
    schema_version: str
    part_bone_binding_registry_version: str
    arc_projection_version: str
    transition_radius_numerator: int
    transition_radius_denominator: int
    weight_denominator: int
    minimum_influence_units: int
    maximum_influences: int
    descriptor_sha256: str

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "part_bone_binding_registry_version": self.part_bone_binding_registry_version,
            "arc_projection_version": self.arc_projection_version,
            "transition_radius_numerator": self.transition_radius_numerator,
            "transition_radius_denominator": self.transition_radius_denominator,
            "weight_denominator": self.weight_denominator,
            "minimum_influence_units": self.minimum_influence_units,
            "maximum_influences": self.maximum_influences,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_payload(), "descriptor_sha256": self.descriptor_sha256}


@dataclass(frozen=True, slots=True)
class SkinInfluence:
    bone_id: str
    weight: float

    def to_dict(self) -> dict[str, object]:
        return {"bone_id": self.bone_id, "weight": self.weight}


@dataclass(frozen=True, slots=True)
class WeightedMeshVertex:
    position: tuple[float, float]
    uv: tuple[float, float]
    boundary: bool
    boundary_loop: int | None
    boundary_order: int | None
    influences: tuple[SkinInfluence, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "position": list(self.position),
            "uv": list(self.uv),
            "boundary": self.boundary,
            "boundary_loop": self.boundary_loop,
            "boundary_order": self.boundary_order,
            "influences": [influence.to_dict() for influence in self.influences],
        }


@dataclass(frozen=True, slots=True)
class WeightedMeshRecord:
    mesh_id: str
    mesh_sha256: str
    mesh_build_descriptor_sha256: str
    component_id: str
    part_id: str
    source_kind: Literal["see_through", "native_variant"]
    variant_id: str | None
    side: str | None
    component_bbox: tuple[int, int, int, int]
    component_mask_sha256: str
    binding_sha256: str
    allowed_bone_ids: tuple[str, ...]
    dynamic_candidate: bool
    vertices: tuple[WeightedMeshVertex, ...]
    boundary_loops: tuple[tuple[int, ...], ...]
    triangles: tuple[int, ...]
    weighted_mesh_sha256: str

    def content_payload(self) -> dict[str, object]:
        return {
            "mesh_id": self.mesh_id,
            "mesh_sha256": self.mesh_sha256,
            "mesh_build_descriptor_sha256": self.mesh_build_descriptor_sha256,
            "component_id": self.component_id,
            "part_id": self.part_id,
            "source_kind": self.source_kind,
            "variant_id": self.variant_id,
            "side": self.side,
            "component_bbox": list(self.component_bbox),
            "component_mask_sha256": self.component_mask_sha256,
            "binding_sha256": self.binding_sha256,
            "allowed_bone_ids": list(self.allowed_bone_ids),
            "dynamic_candidate": self.dynamic_candidate,
            "vertices": [vertex.to_dict() for vertex in self.vertices],
            "boundary_loops": [list(loop) for loop in self.boundary_loops],
            "triangles": list(self.triangles),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.content_payload(),
            "weighted_mesh_sha256": self.weighted_mesh_sha256,
        }


@dataclass(frozen=True, slots=True)
class SkinningDiagnostic:
    part_id: str
    component_ids: tuple[str, ...]
    code: Literal["rigid_fallback_applied", "unknown_part_semantics"]
    selected_bone_id: str
    requested_bone_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "part_id": self.part_id,
            "component_ids": list(self.component_ids),
            "code": self.code,
            "selected_bone_id": self.selected_bone_id,
            "requested_bone_ids": list(self.requested_bone_ids),
        }


@dataclass(frozen=True, slots=True)
class SkinningPlan:
    schema_version: str
    mesh_plan_sha256: str
    bone_graph_plan_sha256: str
    stage_a_joint_plan_sha256: str
    bone_ids: tuple[str, ...]
    descriptor: SkinningDescriptor
    part_bindings: tuple[PartBoneBinding, ...]
    weighted_meshes: tuple[WeightedMeshRecord, ...]
    diagnostics: tuple[SkinningDiagnostic, ...]
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mesh_plan_sha256": self.mesh_plan_sha256,
            "bone_graph_plan_sha256": self.bone_graph_plan_sha256,
            "stage_a_joint_plan_sha256": self.stage_a_joint_plan_sha256,
            "bone_ids": list(self.bone_ids),
            "descriptor": self.descriptor.to_dict(),
            "part_bindings": [binding.to_dict() for binding in self.part_bindings],
            "weighted_meshes": [mesh.to_dict() for mesh in self.weighted_meshes],
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
        }


def _rule_for_tag(base_tag: str) -> PartBoneBindingRule | None:
    matches = tuple(rule for rule in PART_BONE_BINDING_RULES if base_tag in rule.base_tags)
    if len(matches) > 1:
        raise _error("invalid_part_bone_registry", f"duplicate semantic rule for {base_tag}")
    return matches[0] if matches else None


def _semantic_candidates(
    rule: PartBoneBindingRule | None,
    *,
    side: str | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if rule is None:
        return (), ("bone/root",)
    if rule.target == "head":
        return ("bone/head",), ("bone/neck", "bone/torso", "bone/lower_torso", "bone/root")
    if rule.target == "neck":
        return ("bone/neck",), ("bone/torso", "bone/lower_torso", "bone/root")
    if rule.target == "torso":
        return ("bone/torso",), ("bone/lower_torso", "bone/root")
    if rule.target == "lower_torso":
        return ("bone/lower_torso",), ("bone/torso", "bone/root")
    if rule.target == "arm" and side in {"xmin", "xmax"}:
        other = "xmax" if side == "xmin" else "xmin"
        return (
            f"bone/upper_arm.{side}",
            f"bone/forearm.{side}",
            f"bone/hand.{side}",
            f"bone/upper_arm.{other}",
            f"bone/forearm.{other}",
            f"bone/hand.{other}",
        ), ("bone/torso", "bone/lower_torso", "bone/root")
    if rule.target == "arm" and side is None:
        return (
            "bone/upper_arm.xmin",
            "bone/forearm.xmin",
            "bone/hand.xmin",
            "bone/upper_arm.xmax",
            "bone/forearm.xmax",
            "bone/hand.xmax",
        ), ("bone/torso", "bone/lower_torso", "bone/root")
    if rule.target == "leg" and side in {"xmin", "xmax"}:
        other = "xmax" if side == "xmin" else "xmin"
        return (
            f"bone/thigh.{side}",
            f"bone/shin.{side}",
            f"bone/foot.{side}",
            f"bone/thigh.{other}",
            f"bone/shin.{other}",
            f"bone/foot.{other}",
        ), ("bone/lower_torso", "bone/torso", "bone/root")
    if rule.target == "leg" and side is None:
        return (
            "bone/thigh.xmin",
            "bone/shin.xmin",
            "bone/foot.xmin",
            "bone/thigh.xmax",
            "bone/shin.xmax",
            "bone/foot.xmax",
        ), ("bone/lower_torso", "bone/torso", "bone/root")
    if rule.mode == "dynamic":
        return ("bone/torso", "bone/root"), ("bone/lower_torso", "bone/root")
    return (), ("bone/root",)


def _make_binding(
    *,
    part_id: str,
    source_kind: Literal["see_through", "native_variant"],
    variant_id: str | None,
    anchor_part_id: str | None,
    base_tag: str | None,
    side: Literal["xmin", "xmax"] | None,
    rule: PartBoneBindingRule | None,
    emitted_bone_ids: set[str],
    inherited: PartBoneBinding | None = None,
) -> PartBoneBinding:
    if inherited is None:
        candidates, fallback = _semantic_candidates(rule, side=side)
        mode: Literal["rigid", "limb", "dynamic", "unknown"] = "unknown" if rule is None else rule.mode
        dynamic_candidate = False if rule is None else rule.dynamic_candidate
        rule_id = None if rule is None else rule.rule_id
    else:
        candidates = inherited.semantic_candidate_bone_ids
        fallback = inherited.fallback_bone_ids
        mode = inherited.mode
        dynamic_candidate = inherited.dynamic_candidate
        rule_id = inherited.rule_id
        base_tag = inherited.base_tag
        side = inherited.side
    usable = tuple(bone_id for bone_id in candidates if bone_id in emitted_bone_ids)
    if inherited is not None:
        selected = inherited.selected_rigid_bone_id
    else:
        prefer_fallback = rule is not None and rule.mode == "limb" and side is None
        selection_order = (*fallback, *usable) if prefer_fallback else (*usable, *fallback)
        selected = next(
            (bone_id for bone_id in selection_order if bone_id in emitted_bone_ids),
            "bone/root",
        )
    content = {
        "part_id": part_id,
        "source_kind": source_kind,
        "variant_id": variant_id,
        "anchor_part_id": anchor_part_id,
        "base_tag": base_tag,
        "side": side,
        "rule_id": rule_id,
        "mode": mode,
        "semantic_candidate_bone_ids": list(candidates),
        "usable_bone_ids": list(usable),
        "fallback_bone_ids": list(fallback),
        "selected_rigid_bone_id": selected,
        "dynamic_candidate": dynamic_candidate,
    }
    return PartBoneBinding(
        part_id=part_id,
        source_kind=source_kind,
        variant_id=variant_id,
        anchor_part_id=anchor_part_id,
        base_tag=base_tag,
        side=side,
        rule_id=rule_id,
        mode=mode,
        semantic_candidate_bone_ids=candidates,
        usable_bone_ids=usable,
        fallback_bone_ids=fallback,
        selected_rigid_bone_id=selected,
        dynamic_candidate=dynamic_candidate,
        binding_sha256=jcs_sha256(content),
    )


def build_part_bone_bindings(
    mesh_plan: MeshBuildPlan,
    bone_graph: BoneGraphPlan,
    *,
    normalized_parts: Iterable[NormalizedMaskPart],
    native_variant_set: NativeVariantSet | None,
) -> tuple[PartBoneBinding, ...]:
    """Resolve semantic candidates before any geometric weight computation."""

    validate_mesh_plan(mesh_plan)
    validate_bone_graph(bone_graph)
    emitted = {bone.bone_id for bone in bone_graph.bones}
    parts = tuple(normalized_parts)
    part_by_id = {part.part_id: part for part in parts}
    if len(part_by_id) != len(parts):
        raise _error("invalid_skinning_input", "normalized Part IDs are not unique")
    source_by_part = {source.part_id: source for source in mesh_plan.sources}
    ordinary_ids = {source.part_id for source in mesh_plan.sources if source.source_kind == "see_through"}
    if set(part_by_id) != ordinary_ids:
        raise _error("invalid_skinning_input", "normalized Parts differ from mesh sources")

    bindings: dict[str, PartBoneBinding] = {}
    for part_id in sorted(ordinary_ids):
        part = part_by_id[part_id]
        rule = _rule_for_tag(part.base_tag)
        bindings[part_id] = _make_binding(
            part_id=part_id,
            source_kind="see_through",
            variant_id=None,
            anchor_part_id=None,
            base_tag=part.base_tag,
            side=part.side,
            rule=rule,
            emitted_bone_ids=emitted,
        )

    render_ids = set(mesh_plan.render_variant_ids)
    if render_ids:
        if native_variant_set is None:
            raise _error("invalid_skinning_input", "admitted variants require their NativeVariant set")
        if jcs_sha256(native_variant_set.semantic_payload()) != native_variant_set.native_variant_set_sha256:
            raise _error("invalid_skinning_input", "NativeVariant set digest is invalid")
        candidate_by_id = {candidate.variant_id: candidate for candidate in native_variant_set.entries}
        if not render_ids <= set(candidate_by_id):
            raise _error("invalid_skinning_input", "mesh render variants differ from NativeVariant set")
        for variant_id in sorted(render_ids):
            candidate = candidate_by_id[variant_id]
            source = source_by_part.get(candidate.part_id)
            anchor = bindings.get(candidate.draw_anchor_part_id)
            if source is None or source.source_kind != "native_variant" or source.variant_id != variant_id or anchor is None:
                raise _error("invalid_skinning_input", "NativeVariant anchor/source is invalid")
            bindings[candidate.part_id] = _make_binding(
                part_id=candidate.part_id,
                source_kind="native_variant",
                variant_id=variant_id,
                anchor_part_id=candidate.draw_anchor_part_id,
                base_tag=anchor.base_tag,
                side=anchor.side,
                rule=None,
                emitted_bone_ids=emitted,
                inherited=anchor,
            )
    if set(bindings) != set(mesh_plan.final_part_ids):
        raise _error("invalid_skinning_input", "Part bindings differ from final mesh Parts")
    result = tuple(bindings[part_id] for part_id in sorted(bindings))
    for binding in result:
        if binding.binding_sha256 != jcs_sha256(binding.content_payload()):
            raise _error("invalid_skinning_plan", "Part binding digest mismatch")
        if binding.selected_rigid_bone_id not in emitted:
            raise _error("invalid_skinning_plan", "Part binding fallback bone is missing")
        if any(bone_id not in emitted for bone_id in binding.usable_bone_ids):
            raise _error("invalid_skinning_plan", "Part binding usable bone is missing")
    return result


def build_skinning_descriptor() -> SkinningDescriptor:
    content = {
        "schema_version": SKINNING_DESCRIPTOR_VERSION,
        "part_bone_binding_registry_version": PART_BONE_BINDING_REGISTRY_VERSION,
        "arc_projection_version": SKINNING_ARC_PROJECTION_VERSION,
        "transition_radius_numerator": SKINNING_TRANSITION_RADIUS_NUMERATOR,
        "transition_radius_denominator": SKINNING_TRANSITION_RADIUS_DENOMINATOR,
        "weight_denominator": SKINNING_WEIGHT_DENOMINATOR,
        "minimum_influence_units": SKINNING_MIN_INFLUENCE_UNITS,
        "maximum_influences": SKINNING_MAX_INFLUENCES,
    }
    return SkinningDescriptor(
        schema_version=SKINNING_DESCRIPTOR_VERSION,
        part_bone_binding_registry_version=PART_BONE_BINDING_REGISTRY_VERSION,
        arc_projection_version=SKINNING_ARC_PROJECTION_VERSION,
        transition_radius_numerator=SKINNING_TRANSITION_RADIUS_NUMERATOR,
        transition_radius_denominator=SKINNING_TRANSITION_RADIUS_DENOMINATOR,
        weight_denominator=SKINNING_WEIGHT_DENOMINATOR,
        minimum_influence_units=SKINNING_MIN_INFLUENCE_UNITS,
        maximum_influences=SKINNING_MAX_INFLUENCES,
        descriptor_sha256=jcs_sha256(content),
    )


def _authenticated_source_map(
    mesh_plan: MeshBuildPlan,
    component_sources: Iterable[MeshComponentSource],
) -> dict[str, MeshComponentSource]:
    sources = tuple(component_sources)
    source_by_id = {source.component_id: source for source in sources}
    if len(source_by_id) != len(sources) or set(source_by_id) != {source.component_id for source in mesh_plan.sources}:
        raise _error("invalid_skinning_input", "component sources differ from MeshBuildPlan")
    record_by_id = {source.component_id: source for source in mesh_plan.sources}
    for source in sources:
        record = record_by_id[source.component_id]
        mask_sha256 = "sha256:" + hashlib.sha256(source.binary_mask_u8).hexdigest()
        if (
            source.schema_version != MESH_COMPONENT_SOURCE_VERSION
            or len(source.binary_mask_u8) != source.width * source.height
            or set(source.binary_mask_u8) - {0, 1}
            or sum(source.binary_mask_u8) != source.pixel_count
            or mask_sha256 != source.component_mask_sha256
            or source.part_id != record.part_id
            or source.source_kind != record.source_kind
            or source.variant_id != record.variant_id
            or source.side != record.side
            or source.part_xyxy != record.part_xyxy
            or source.bbox != record.bbox
            or source.component_mask_sha256 != record.component_mask_sha256
            or source.pixel_count != record.pixel_count
        ):
            raise _error("invalid_skinning_input", "component source authentication failed")
    return source_by_id


def _rigid_influences(bone_id: str) -> tuple[SkinInfluence, ...]:
    return (SkinInfluence(bone_id=bone_id, weight=1.0),)


def _quantized_influences(
    raw: tuple[tuple[str, float], ...],
) -> tuple[SkinInfluence, ...]:
    combined: dict[str, float] = {}
    for bone_id, weight in raw:
        if not isinstance(bone_id, str) or not bone_id:
            raise _error("invalid_skinning_input", "arc weight bone ID is invalid")
        if not math.isfinite(weight) or weight < 0.0:
            raise _error("invalid_skinning_input", "arc weight is invalid")
        combined[bone_id] = combined.get(bone_id, 0.0) + weight
    total = sum(combined.values())
    if total <= 0.0:
        raise _error("invalid_skinning_input", "arc weights have zero mass")
    normalized = {bone_id: weight / total for bone_id, weight in combined.items()}
    units = {bone_id: int(math.floor(weight * SKINNING_WEIGHT_DENOMINATOR + 0.5)) for bone_id, weight in normalized.items()}
    units = {bone_id: value for bone_id, value in units.items() if value >= SKINNING_MIN_INFLUENCE_UNITS}
    if not units:
        winner = min(
            normalized,
            key=lambda bone_id: (-normalized[bone_id], bone_id),
        )
        units = {winner: SKINNING_WEIGHT_DENOMINATOR}
    recipient = min(
        units,
        key=lambda bone_id: (-normalized[bone_id], bone_id),
    )
    units[recipient] += SKINNING_WEIGHT_DENOMINATOR - sum(units.values())
    ordered = tuple(sorted(units))[:SKINNING_MAX_INFLUENCES]
    if len(ordered) < len(units):
        retained = {
            bone_id: units[bone_id]
            for bone_id in sorted(
                units,
                key=lambda bone_id: (-units[bone_id], bone_id),
            )[:SKINNING_MAX_INFLUENCES]
        }
        recipient = min(retained, key=lambda bone_id: (-retained[bone_id], bone_id))
        retained[recipient] += SKINNING_WEIGHT_DENOMINATOR - sum(retained.values())
        units = retained
        ordered = tuple(sorted(units))
    influences: list[SkinInfluence] = []
    accumulated = 0.0
    for index, bone_id in enumerate(ordered):
        if index + 1 == len(ordered):
            weight = 1.0 - accumulated
        else:
            weight = units[bone_id] / SKINNING_WEIGHT_DENOMINATOR
            accumulated += weight
        influences.append(SkinInfluence(bone_id=bone_id, weight=weight))
    return tuple(influences)


def compute_arc_length_influences(
    *,
    position: tuple[float, float],
    bone_ids: tuple[str, ...],
    joint_points: tuple[tuple[float, float], ...],
    internal_joint_radii: tuple[float, ...],
) -> tuple[SkinInfluence, ...]:
    """Project a vertex to a bent joint chain and blend only adjacent bones."""

    if (
        not bone_ids
        or len(set(bone_ids)) != len(bone_ids)
        or len(joint_points) != len(bone_ids) + 1
        or len(internal_joint_radii) != len(bone_ids) - 1
    ):
        raise _error("invalid_skinning_input", "arc chain cardinality is invalid")
    if any(not math.isfinite(value) for point in (*joint_points, position) for value in point) or any(
        not math.isfinite(radius) or radius <= 0.0 for radius in internal_joint_radii
    ):
        raise _error("invalid_skinning_input", "arc chain geometry is invalid")

    cumulative = [0.0]
    best: tuple[float, int, float, float] | None = None
    for index, (start, end) in enumerate(zip(joint_points[:-1], joint_points[1:], strict=True)):
        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        length_squared = delta_x * delta_x + delta_y * delta_y
        if length_squared <= 0.0:
            raise _error("invalid_skinning_input", "arc chain contains a zero-length segment")
        length = math.sqrt(length_squared)
        projection = ((position[0] - start[0]) * delta_x + (position[1] - start[1]) * delta_y) / length_squared
        projection = min(1.0, max(0.0, projection))
        projected_x = start[0] + projection * delta_x
        projected_y = start[1] + projection * delta_y
        distance_squared = (position[0] - projected_x) ** 2 + (position[1] - projected_y) ** 2
        arc_position = cumulative[-1] + projection * length
        candidate = (distance_squared, index, projection, arc_position)
        if best is None or candidate < best:
            best = candidate
        cumulative.append(cumulative[-1] + length)
    if best is None:
        raise _error("invalid_skinning_input", "arc chain has no segments")
    segment_index = best[1]
    arc_position = best[3]

    active_transition: tuple[float, int, float] | None = None
    for boundary_index, radius in enumerate(internal_joint_radii):
        half_width = radius * SKINNING_TRANSITION_RADIUS_NUMERATOR / SKINNING_TRANSITION_RADIUS_DENOMINATOR
        boundary = cumulative[boundary_index + 1]
        normalized_distance = abs(arc_position - boundary) / half_width
        if normalized_distance <= 1.0:
            candidate = (normalized_distance, boundary_index, half_width)
            if active_transition is None or candidate < active_transition:
                active_transition = candidate
    if active_transition is None:
        return _rigid_influences(bone_ids[segment_index])
    _, boundary_index, half_width = active_transition
    boundary = cumulative[boundary_index + 1]
    distal_weight = (arc_position - (boundary - half_width)) / (2.0 * half_width)
    distal_weight = min(1.0, max(0.0, distal_weight))
    return _quantized_influences(
        (
            (bone_ids[boundary_index], 1.0 - distal_weight),
            (bone_ids[boundary_index + 1], distal_weight),
        )
    )


def _component_radius_at(
    source: MeshComponentSource,
    point: tuple[float, float],
) -> float:
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    mask = np.frombuffer(source.binary_mask_u8, dtype=np.uint8).reshape(
        source.height,
        source.width,
    )
    local_x = point[0] - source.bbox[0]
    local_y = point[1] - source.bbox[1]
    column = min(source.width - 1, max(0, int(math.floor(local_x))))
    row = min(source.height - 1, max(0, int(math.floor(local_y))))
    if not bool(mask[row, column]):
        foreground = np.argwhere(mask)
        if foreground.size == 0:
            return 1.0
        row, column = min(
            ((int(item[0]), int(item[1])) for item in foreground),
            key=lambda item: (
                (item[1] + 0.5 - local_x) ** 2 + (item[0] + 0.5 - local_y) ** 2,
                item[0],
                item[1],
            ),
        )
    return max(1.0, float(distance_transform_edt(mask)[row, column]))


def _limb_chain_geometry(
    bone_ids: tuple[str, ...],
    *,
    bone_graph: BoneGraphPlan,
    joints: StageAJointPlan,
    source: MeshComponentSource,
) -> (
    tuple[
        tuple[str, ...],
        tuple[tuple[float, float], ...],
        tuple[float, ...],
    ]
    | None
):
    bone_by_id = {bone.bone_id: bone for bone in bone_graph.bones}
    chain = tuple(bone_by_id[bone_id] for bone_id in bone_ids)
    if len(chain) < 2 or any(bone.head is None or bone.tail is None for bone in chain):
        return None
    for first, second in zip(chain[:-1], chain[1:], strict=True):
        if first.tail != second.head or first.tail_joint_id != second.head_joint_id:
            return None
    points = (chain[0].head, *(bone.tail for bone in chain))
    if any(point is None for point in points):
        return None
    eligibility_by_id = {eligibility.joint_id: eligibility for eligibility in joints.joints.eligibilities}
    radii: list[float] = []
    for bone in chain[:-1]:
        if bone.tail_joint_id is None:
            return None
        eligibility = eligibility_by_id.get(bone.tail_joint_id)
        radius = (
            eligibility.local_limb_radius
            if eligibility is not None and eligibility.local_limb_radius is not None
            else _component_radius_at(source, bone.tail)
        )
        radii.append(radius)
    return (
        tuple(bone.bone_id for bone in chain),
        tuple(point for point in points if point is not None),
        tuple(radii),
    )


def _limb_chain_groups(binding: PartBoneBinding) -> tuple[tuple[str, ...], ...]:
    groups_by_side = {
        side: tuple(bone_id for bone_id in binding.usable_bone_ids if bone_id.endswith(f".{side}")) for side in ("xmin", "xmax")
    }
    if binding.side in {"xmin", "xmax"}:
        own = groups_by_side[binding.side]
        if len(own) < 2:
            return ()
        other_side = "xmax" if binding.side == "xmin" else "xmin"
        other = groups_by_side[other_side]
        return (own, other) if len(other) >= 2 else (own,)
    groups = (groups_by_side["xmin"], groups_by_side["xmax"])
    return groups if all(len(group) >= 2 for group in groups) else ()


def _chain_distance_squared(
    position: tuple[float, float],
    joint_points: tuple[tuple[float, float], ...],
) -> float:
    distances = []
    for start, end in zip(joint_points[:-1], joint_points[1:], strict=True):
        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        length_squared = delta_x * delta_x + delta_y * delta_y
        if length_squared <= 0.0:
            raise _error("invalid_skinning_input", "arc chain contains a zero-length segment")
        projection = ((position[0] - start[0]) * delta_x + (position[1] - start[1]) * delta_y) / length_squared
        projection = min(1.0, max(0.0, projection))
        projected_x = start[0] + projection * delta_x
        projected_y = start[1] + projection * delta_y
        distances.append((position[0] - projected_x) ** 2 + (position[1] - projected_y) ** 2)
    if not distances:
        raise _error("invalid_skinning_input", "arc chain has no segments")
    return min(distances)


def _median_chain_distance_squared(
    mesh: MeshRecord,
    joint_points: tuple[tuple[float, float], ...],
) -> float:
    distances = sorted(_chain_distance_squared(vertex.position, joint_points) for vertex in mesh.vertices)
    midpoint = len(distances) // 2
    if len(distances) % 2:
        return distances[midpoint]
    return (distances[midpoint - 1] + distances[midpoint]) / 2.0


def _component_chain_geometry(
    mesh: MeshRecord,
    binding: PartBoneBinding,
    geometries: tuple[
        tuple[
            tuple[str, ...],
            tuple[tuple[float, float], ...],
            tuple[float, ...],
        ],
        ...,
    ],
) -> (
    tuple[
        tuple[str, ...],
        tuple[tuple[float, float], ...],
        tuple[float, ...],
    ]
    | None
):
    if binding.side not in {"xmin", "xmax"}:
        return None
    if binding.base_tag == "footwear":
        center_x = sum(vertex.position[0] for vertex in mesh.vertices) / len(mesh.vertices)
        center_y = sum(vertex.position[1] for vertex in mesh.vertices) / len(mesh.vertices)

        def assignment_cost(item) -> tuple[float, int]:
            index, geometry = item
            terminal_x, terminal_y = geometry[1][-1]
            return (
                (center_x - terminal_x) ** 2 + (center_y - terminal_y) ** 2,
                index,
            )

    else:

        def assignment_cost(item) -> tuple[float, int]:
            index, geometry = item
            return (_median_chain_distance_squared(mesh, geometry[1]), index)

    return min(enumerate(geometries), key=assignment_cost)[1]


def _weighted_mesh(
    mesh: MeshRecord,
    binding: PartBoneBinding,
    influences: tuple[tuple[SkinInfluence, ...], ...],
) -> WeightedMeshRecord:
    if len(influences) != len(mesh.vertices):
        raise _error("invalid_skinning_plan", "vertex influence count differs from mesh")
    vertices = tuple(
        WeightedMeshVertex(
            position=vertex.position,
            uv=vertex.uv,
            boundary=vertex.boundary,
            boundary_loop=vertex.boundary_loop,
            boundary_order=vertex.boundary_order,
            influences=vertex_influences,
        )
        for vertex, vertex_influences in zip(mesh.vertices, influences, strict=True)
    )
    allowed = tuple(dict.fromkeys((*binding.usable_bone_ids, binding.selected_rigid_bone_id)))
    content = {
        "mesh_id": mesh.mesh_id,
        "mesh_sha256": mesh.mesh_sha256,
        "mesh_build_descriptor_sha256": mesh.build_descriptor_sha256,
        "component_id": mesh.component_id,
        "part_id": mesh.part_id,
        "source_kind": mesh.source_kind,
        "variant_id": mesh.variant_id,
        "side": mesh.side,
        "component_bbox": list(mesh.component_bbox),
        "component_mask_sha256": mesh.component_mask_sha256,
        "binding_sha256": binding.binding_sha256,
        "allowed_bone_ids": list(allowed),
        "dynamic_candidate": binding.dynamic_candidate,
        "vertices": [vertex.to_dict() for vertex in vertices],
        "boundary_loops": [list(loop) for loop in mesh.boundary_loops],
        "triangles": list(mesh.triangles),
    }
    return WeightedMeshRecord(
        mesh_id=mesh.mesh_id,
        mesh_sha256=mesh.mesh_sha256,
        mesh_build_descriptor_sha256=mesh.build_descriptor_sha256,
        component_id=mesh.component_id,
        part_id=mesh.part_id,
        source_kind=mesh.source_kind,
        variant_id=mesh.variant_id,
        side=mesh.side,
        component_bbox=mesh.component_bbox,
        component_mask_sha256=mesh.component_mask_sha256,
        binding_sha256=binding.binding_sha256,
        allowed_bone_ids=allowed,
        dynamic_candidate=binding.dynamic_candidate,
        vertices=vertices,
        boundary_loops=mesh.boundary_loops,
        triangles=mesh.triangles,
        weighted_mesh_sha256=jcs_sha256(content),
    )


def _mesh_geometry_payload(mesh: WeightedMeshRecord) -> dict[str, object]:
    return {
        "mesh_id": mesh.mesh_id,
        "component_id": mesh.component_id,
        "part_id": mesh.part_id,
        "source_kind": mesh.source_kind,
        "variant_id": mesh.variant_id,
        "side": mesh.side,
        "component_bbox": list(mesh.component_bbox),
        "component_mask_sha256": mesh.component_mask_sha256,
        "build_descriptor_sha256": mesh.mesh_build_descriptor_sha256,
        "vertices": [
            {
                "position": list(vertex.position),
                "uv": list(vertex.uv),
                "boundary": vertex.boundary,
                "boundary_loop": vertex.boundary_loop,
                "boundary_order": vertex.boundary_order,
            }
            for vertex in mesh.vertices
        ],
        "boundary_loops": [list(loop) for loop in mesh.boundary_loops],
        "triangles": list(mesh.triangles),
    }


def validate_skinning_plan(plan: SkinningPlan) -> SkinningPlan:
    """Validate deterministic weight, reference, and topology invariants."""

    if not isinstance(plan, SkinningPlan) or plan.schema_version != SKINNING_PLAN_VERSION:
        raise _error("invalid_skinning_plan", "unsupported SkinningPlan")
    if plan.plan_sha256 != jcs_sha256(plan.semantic_payload()):
        raise _error("invalid_skinning_plan", "SkinningPlan digest mismatch")
    if plan.descriptor != build_skinning_descriptor():
        raise _error("invalid_skinning_plan", "skinning descriptor differs from v1")
    if not plan.bone_ids or plan.bone_ids[0] != "bone/root" or len(plan.bone_ids) != len(set(plan.bone_ids)):
        raise _error("invalid_skinning_plan", "skinning bone inventory is invalid")
    bone_id_set = set(plan.bone_ids)
    bindings = {binding.part_id: binding for binding in plan.part_bindings}
    if len(bindings) != len(plan.part_bindings) or tuple(bindings) != tuple(sorted(bindings)):
        raise _error("invalid_skinning_plan", "Part bindings are not canonical")
    for binding in plan.part_bindings:
        if binding.binding_sha256 != jcs_sha256(binding.content_payload()):
            raise _error("invalid_skinning_plan", "Part binding digest mismatch")
        if binding.selected_rigid_bone_id not in bone_id_set or any(
            bone_id not in bone_id_set for bone_id in binding.usable_bone_ids
        ):
            raise _error("invalid_skinning_plan", "Part binding references an unknown bone")
    component_ids = tuple(mesh.component_id for mesh in plan.weighted_meshes)
    if component_ids != tuple(sorted(set(component_ids))):
        raise _error("invalid_skinning_plan", "weighted meshes are not canonical")
    for mesh in plan.weighted_meshes:
        binding = bindings.get(mesh.part_id)
        if (
            binding is None
            or mesh.binding_sha256 != binding.binding_sha256
            or mesh.dynamic_candidate != binding.dynamic_candidate
            or mesh.weighted_mesh_sha256 != jcs_sha256(mesh.content_payload())
            or mesh.mesh_sha256 != jcs_sha256(_mesh_geometry_payload(mesh))
            or not mesh.vertices
            or not mesh.triangles
            or len(mesh.triangles) % 3
        ):
            raise _error("invalid_skinning_plan", "weighted mesh provenance/topology is invalid")
        if len(mesh.allowed_bone_ids) != len(set(mesh.allowed_bone_ids)):
            raise _error("invalid_skinning_plan", "allowed bone IDs are not unique")
        if any(bone_id not in bone_id_set for bone_id in mesh.allowed_bone_ids):
            raise _error("invalid_skinning_plan", "allowed bone ID is absent from inventory")
        for vertex in mesh.vertices:
            influences = vertex.influences
            if not 1 <= len(influences) <= plan.descriptor.maximum_influences:
                raise _error("invalid_skinning_plan", "vertex influence count is invalid")
            bone_ids = tuple(influence.bone_id for influence in influences)
            if bone_ids != tuple(sorted(set(bone_ids))) or any(bone_id not in mesh.allowed_bone_ids for bone_id in bone_ids):
                raise _error("invalid_skinning_plan", "vertex influence bone is invalid")
            weights = tuple(influence.weight for influence in influences)
            if any(not math.isfinite(weight) or weight < 0.0 for weight in weights) or not math.isclose(
                sum(weights), 1.0, abs_tol=1e-12
            ):
                raise _error("invalid_skinning_plan", "vertex influence weights are invalid")
        if any(index < 0 or index >= len(mesh.vertices) for index in mesh.triangles):
            raise _error("invalid_skinning_plan", "weighted mesh triangle index is invalid")
    diagnostics = tuple((diagnostic.part_id, diagnostic.code, diagnostic.component_ids) for diagnostic in plan.diagnostics)
    if diagnostics != tuple(sorted(set(diagnostics))):
        raise _error("invalid_skinning_plan", "skinning diagnostics are not canonical")
    if any(diagnostic.part_id not in bindings for diagnostic in plan.diagnostics):
        raise _error("invalid_skinning_plan", "skinning diagnostic Part is unknown")
    if any(diagnostic.selected_bone_id not in bone_id_set for diagnostic in plan.diagnostics):
        raise _error("invalid_skinning_plan", "skinning diagnostic bone is unknown")
    return plan


def build_skinning_plan(
    mesh_plan: MeshBuildPlan,
    bone_graph: BoneGraphPlan,
    joints: StageAJointPlan,
    *,
    normalized_parts: Iterable[NormalizedMaskPart],
    component_sources: Iterable[MeshComponentSource],
    native_variant_set: NativeVariantSet | None,
) -> SkinningPlan:
    """Apply semantic candidates, then deterministic geometry weights per mesh."""

    validate_mesh_plan(mesh_plan)
    validate_bone_graph(bone_graph)
    validate_stage_a_joint_plan(joints)
    if (
        bone_graph.stage_a_joint_plan_sha256 != joints.plan_sha256
        or mesh_plan.canvas_edge != joints.joints.canvas_width
        or mesh_plan.canvas_edge != joints.joints.canvas_height
    ):
        raise _error("invalid_skinning_input", "mesh, bone, and joint plans describe different inputs")
    source_by_id = _authenticated_source_map(mesh_plan, component_sources)
    bindings = build_part_bone_bindings(
        mesh_plan,
        bone_graph,
        normalized_parts=normalized_parts,
        native_variant_set=native_variant_set,
    )
    binding_by_part = {binding.part_id: binding for binding in bindings}
    component_ids_by_part: dict[str, list[str]] = {}
    for source in mesh_plan.sources:
        component_ids_by_part.setdefault(source.part_id, []).append(source.component_id)

    weighted_mesh_list: list[WeightedMeshRecord] = []
    limb_geometry_fallback_parts: set[str] = set()
    for mesh in mesh_plan.meshes:
        binding = binding_by_part[mesh.part_id]
        vertex_influences: tuple[tuple[SkinInfluence, ...], ...]
        if binding.mode == "limb" and len(binding.usable_bone_ids) >= 2:
            chain_groups = _limb_chain_groups(binding)
            geometries = tuple(
                geometry
                for bone_ids in chain_groups
                if (
                    geometry := _limb_chain_geometry(
                        bone_ids,
                        bone_graph=bone_graph,
                        joints=joints,
                        source=source_by_id[mesh.component_id],
                    )
                )
                is not None
            )
            if len(geometries) != len(chain_groups) or not geometries:
                limb_geometry_fallback_parts.add(binding.part_id)
                vertex_influences = tuple(_rigid_influences(binding.selected_rigid_bone_id) for _vertex in mesh.vertices)
            else:
                component_geometry = _component_chain_geometry(
                    mesh,
                    binding,
                    geometries,
                )

                def influences_for_vertex(vertex) -> tuple[SkinInfluence, ...]:
                    if component_geometry is None:
                        _, selected_geometry = min(
                            enumerate(geometries),
                            key=lambda item: (
                                _chain_distance_squared(vertex.position, item[1][1]),
                                item[0],
                            ),
                        )
                    else:
                        selected_geometry = component_geometry
                    bone_ids, joint_points, radii = selected_geometry
                    return compute_arc_length_influences(
                        position=vertex.position,
                        bone_ids=bone_ids,
                        joint_points=joint_points,
                        internal_joint_radii=radii,
                    )

                vertex_influences = tuple(influences_for_vertex(vertex) for vertex in mesh.vertices)
        else:
            vertex_influences = tuple(_rigid_influences(binding.selected_rigid_bone_id) for _vertex in mesh.vertices)
        weighted_mesh_list.append(_weighted_mesh(mesh, binding, vertex_influences))
    weighted_meshes = tuple(weighted_mesh_list)

    diagnostics: list[SkinningDiagnostic] = []
    for binding in bindings:
        component_ids = tuple(sorted(component_ids_by_part[binding.part_id]))
        if binding.mode == "unknown":
            diagnostics.append(
                SkinningDiagnostic(
                    part_id=binding.part_id,
                    component_ids=component_ids,
                    code="unknown_part_semantics",
                    selected_bone_id=binding.selected_rigid_bone_id,
                    requested_bone_ids=binding.semantic_candidate_bone_ids,
                )
            )
        needs_fallback = (
            binding.mode == "unknown"
            or (binding.mode == "limb" and len(binding.usable_bone_ids) < 2)
            or binding.part_id in limb_geometry_fallback_parts
            or (
                binding.mode in {"rigid", "dynamic"}
                and (
                    not binding.semantic_candidate_bone_ids
                    or binding.selected_rigid_bone_id != binding.semantic_candidate_bone_ids[0]
                )
            )
        )
        if needs_fallback:
            diagnostics.append(
                SkinningDiagnostic(
                    part_id=binding.part_id,
                    component_ids=component_ids,
                    code="rigid_fallback_applied",
                    selected_bone_id=binding.selected_rigid_bone_id,
                    requested_bone_ids=binding.semantic_candidate_bone_ids,
                )
            )

    diagnostics.sort(key=lambda item: (item.part_id, item.code, item.component_ids))
    descriptor = build_skinning_descriptor()
    content = {
        "schema_version": SKINNING_PLAN_VERSION,
        "mesh_plan_sha256": mesh_plan.plan_sha256,
        "bone_graph_plan_sha256": bone_graph.plan_sha256,
        "stage_a_joint_plan_sha256": joints.plan_sha256,
        "bone_ids": [bone.bone_id for bone in bone_graph.bones],
        "descriptor": descriptor.to_dict(),
        "part_bindings": [binding.to_dict() for binding in bindings],
        "weighted_meshes": [mesh.to_dict() for mesh in weighted_meshes],
        "diagnostics": [diagnostic.to_dict() for diagnostic in diagnostics],
    }
    plan = SkinningPlan(
        schema_version=SKINNING_PLAN_VERSION,
        mesh_plan_sha256=mesh_plan.plan_sha256,
        bone_graph_plan_sha256=bone_graph.plan_sha256,
        stage_a_joint_plan_sha256=joints.plan_sha256,
        bone_ids=tuple(bone.bone_id for bone in bone_graph.bones),
        descriptor=descriptor,
        part_bindings=bindings,
        weighted_meshes=weighted_meshes,
        diagnostics=tuple(diagnostics),
        plan_sha256=jcs_sha256(content),
    )
    return validate_skinning_plan(plan)


__all__ = [
    "PART_BONE_BINDING_REGISTRY_VERSION",
    "PART_BONE_BINDING_RULES",
    "SKINNING_ARC_PROJECTION_VERSION",
    "SKINNING_DESCRIPTOR_VERSION",
    "SKINNING_MAX_INFLUENCES",
    "SKINNING_MIN_INFLUENCE_UNITS",
    "SKINNING_PLAN_VERSION",
    "SKINNING_TRANSITION_RADIUS_DENOMINATOR",
    "SKINNING_TRANSITION_RADIUS_NUMERATOR",
    "SKINNING_WEIGHT_DENOMINATOR",
    "PartBoneBinding",
    "PartBoneBindingRule",
    "SkinInfluence",
    "SkinningDescriptor",
    "SkinningDiagnostic",
    "SkinningError",
    "SkinningPlan",
    "WeightedMeshRecord",
    "WeightedMeshVertex",
    "build_part_bone_bindings",
    "build_skinning_descriptor",
    "build_skinning_plan",
    "compute_arc_length_influences",
    "validate_skinning_plan",
]
