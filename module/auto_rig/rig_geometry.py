from __future__ import annotations

import json
import re
import types
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from .artifacts import canonical_json_bytes
from .bone_graph import BoneGraphPlan, validate_bone_graph
from .component_plan import MaskComponentPlan
from .input_identity import TargetInputIdentity
from .jcs import jcs_sha256
from .joint_pipeline import StageAJointPlan, validate_stage_a_joint_plan
from .mesh_builder import MeshBuildError, MeshBuildPlan, validate_mesh_plan
from .native_variant_admission import (
    FINAL_DRAW_ORDER_PLAN_VERSION,
    FinalDrawOrderPlan,
)
from .native_variants import NativeVariantSet
from .skinning import SkinningPlan, validate_skinning_plan

COMPONENT_DRAW_ORDER_EXPANDER_VERSION = "component-draw-order-expander-v1"
RIG_GEOMETRY_CACHE_VERSION = "rig-geometry-cache-v1"
RIG_GEOMETRY_CACHE_PATH = "rig/cache/B/rig_geometry.json"
RIG_GEOMETRY_CACHE_GENERATOR = "qinglong-auto-rig-stage-b"
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CACHE_PAYLOAD_FIELDS = frozenset(
    {
        "cache_schema_version",
        "generator",
        "target_input_fingerprint",
        "stage_a_fingerprint",
        "stage_b_fingerprint",
        "component_plan_sha256",
        "native_variant_set_sha256",
        "native_variant_eligibility_sha256",
        "rig_overrides_sha256",
        "canvas",
        "parts",
        "final_draw_order",
        "joint_plan",
        "bone_graph",
        "mesh_plan",
        "skinning_plan",
        "component_draw_order",
        "diagnostics",
        "dependencies",
        "degradation_state",
        "degradation_codes",
        "cache_sha256",
    }
)


class ComponentDrawOrderError(ValueError):
    """Raised when final Part order cannot expand over the frozen A components."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _draw_error(message: str) -> ComponentDrawOrderError:
    return ComponentDrawOrderError("invalid_component_draw_order", message)


class RigGeometryCacheError(ValueError):
    """Raised when the B-owned geometry cache is incomplete or not reference-closed."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _cache_error(message: str) -> RigGeometryCacheError:
    return RigGeometryCacheError("invalid_rig_geometry_cache", message)


@dataclass(frozen=True, slots=True)
class ComponentDrawRankRecord:
    component_id: str
    mesh_id: str | None
    part_id: str
    part_draw_rank: int
    component_draw_rank: int

    def to_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "mesh_id": self.mesh_id,
            "part_id": self.part_id,
            "part_draw_rank": self.part_draw_rank,
            "component_draw_rank": self.component_draw_rank,
        }


@dataclass(frozen=True, slots=True)
class ComponentDrawOrderPlan:
    schema_version: str
    final_draw_order_plan_sha256: str
    mesh_build_plan_sha256: str
    records: tuple[ComponentDrawRankRecord, ...]
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "final_draw_order_plan_sha256": self.final_draw_order_plan_sha256,
            "mesh_build_plan_sha256": self.mesh_build_plan_sha256,
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True, slots=True)
class RigGeometryPartRecord:
    part_id: str
    source_kind: Literal["see_through", "native_variant"]
    source_part_id: str
    source_tag: str | None
    base_tag: str
    semantic_slug: str
    side: Literal["xmin", "xmax"] | None
    side_provenance: str
    source_xyxy: tuple[int, int, int, int]
    xyxy: tuple[int, int, int, int]
    depth_median: float
    cleaned_binary_mask_sha256: str
    component_ids: tuple[str, ...]
    variant_id: str | None
    semantic_role: str | None
    base_part_ids: tuple[str, ...]
    draw_anchor_part_id: str | None
    partition_sha256: str | None
    part_draw_rank: int
    depth_bucket: int
    draw_bundle_id: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "part_id": self.part_id,
            "source_kind": self.source_kind,
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
            "component_ids": list(self.component_ids),
            "variant_id": self.variant_id,
            "semantic_role": self.semantic_role,
            "base_part_ids": list(self.base_part_ids),
            "draw_anchor_part_id": self.draw_anchor_part_id,
            "partition_sha256": self.partition_sha256,
            "part_draw_rank": self.part_draw_rank,
            "depth_bucket": self.depth_bucket,
            "draw_bundle_id": self.draw_bundle_id,
        }


@dataclass(frozen=True, slots=True)
class RigGeometryDiagnosticRecord:
    source_stage: Literal["A", "B"]
    code: str
    subject_id: str
    related_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_stage": self.source_stage,
            "code": self.code,
            "subject_id": self.subject_id,
            "related_ids": list(self.related_ids),
        }


@dataclass(frozen=True, slots=True)
class RigGeometryDependencyRecord:
    name: str
    version: str
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "version": self.version, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class RigGeometryCache:
    cache_schema_version: str
    generator: str
    target: TargetInputIdentity
    stage_a_fingerprint: str
    stage_b_fingerprint: str
    component_plan_sha256: str
    native_variant_set_sha256: str
    native_variant_eligibility_sha256: str
    rig_overrides_sha256: str
    parts: tuple[RigGeometryPartRecord, ...]
    final_draw_order: FinalDrawOrderPlan
    joint_plan: StageAJointPlan
    bone_graph: BoneGraphPlan
    mesh_plan: MeshBuildPlan
    skinning_plan: SkinningPlan
    component_draw_order: ComponentDrawOrderPlan
    diagnostics: tuple[RigGeometryDiagnosticRecord, ...]
    dependencies: tuple[RigGeometryDependencyRecord, ...]
    degradation_state: Literal["clean", "degraded"]
    degradation_codes: tuple[str, ...]
    cache_sha256: str

    @property
    def target_input_fingerprint(self) -> str:
        return self.target.target_input_fingerprint

    def semantic_payload(self) -> dict[str, object]:
        return {
            "cache_schema_version": self.cache_schema_version,
            "generator": self.generator,
            "target_input_fingerprint": self.target.target_input_fingerprint,
            "stage_a_fingerprint": self.stage_a_fingerprint,
            "stage_b_fingerprint": self.stage_b_fingerprint,
            "component_plan_sha256": self.component_plan_sha256,
            "native_variant_set_sha256": self.native_variant_set_sha256,
            "native_variant_eligibility_sha256": self.native_variant_eligibility_sha256,
            "rig_overrides_sha256": self.rig_overrides_sha256,
            "canvas": {
                "width": self.target.canvas_width,
                "height": self.target.canvas_height,
                "resolution": self.target.canvas_resolution,
                "coordinate_space": "layerdiff_canvas",
                "origin": "top_left",
                "y_axis": "down",
                "tag_version": self.target.tag_version,
            },
            "parts": [part.to_dict() for part in self.parts],
            "final_draw_order": {
                **self.final_draw_order.semantic_payload(),
                "plan_sha256": self.final_draw_order.plan_sha256,
            },
            "joint_plan": {
                **self.joint_plan.semantic_payload(),
                "plan_sha256": self.joint_plan.plan_sha256,
            },
            "bone_graph": {
                **self.bone_graph.semantic_payload(),
                "plan_sha256": self.bone_graph.plan_sha256,
            },
            "mesh_plan": {
                **self.mesh_plan.semantic_payload(),
                "plan_sha256": self.mesh_plan.plan_sha256,
            },
            "skinning_plan": {
                **self.skinning_plan.semantic_payload(),
                "plan_sha256": self.skinning_plan.plan_sha256,
            },
            "component_draw_order": {
                **self.component_draw_order.semantic_payload(),
                "plan_sha256": self.component_draw_order.plan_sha256,
            },
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "dependencies": [item.to_dict() for item in self.dependencies],
            "degradation_state": self.degradation_state,
            "degradation_codes": list(self.degradation_codes),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "cache_sha256": self.cache_sha256}


def _validate_final_draw_order(plan: FinalDrawOrderPlan) -> None:
    if (
        not isinstance(plan, FinalDrawOrderPlan)
        or plan.schema_version != FINAL_DRAW_ORDER_PLAN_VERSION
        or plan.plan_sha256 != jcs_sha256(plan.semantic_payload())
    ):
        raise _draw_error("final draw-order plan is invalid")
    if tuple(record.part_id for record in plan.records) != plan.part_order:
        raise _draw_error("final draw records differ from Part order")
    if tuple(record.part_draw_rank for record in plan.records) != tuple(range(len(plan.records))):
        raise _draw_error("final Part draw ranks are not gapless")


def validate_component_draw_order(
    plan: ComponentDrawOrderPlan,
    *,
    final_draw: FinalDrawOrderPlan,
    mesh_plan: MeshBuildPlan,
) -> ComponentDrawOrderPlan:
    """Validate the stable expansion from Part ranks to component ranks."""

    _validate_final_draw_order(final_draw)
    try:
        validate_mesh_plan(mesh_plan)
    except MeshBuildError as exc:
        raise _draw_error("MeshBuildPlan is invalid") from exc
    if (
        not isinstance(plan, ComponentDrawOrderPlan)
        or plan.schema_version != COMPONENT_DRAW_ORDER_EXPANDER_VERSION
        or plan.final_draw_order_plan_sha256 != final_draw.plan_sha256
        or plan.mesh_build_plan_sha256 != mesh_plan.plan_sha256
        or plan.plan_sha256 != jcs_sha256(plan.semantic_payload())
    ):
        raise _draw_error("component draw-order plan provenance is invalid")
    source_by_id = {source.component_id: source for source in mesh_plan.sources}
    mesh_by_component = {mesh.component_id: mesh for mesh in mesh_plan.meshes}
    if len(mesh_by_component) != len(mesh_plan.meshes):
        raise _draw_error("mesh components are not unique")
    if any(component_id not in source_by_id for component_id in mesh_by_component):
        raise _draw_error("mesh lies outside the A component table")
    if tuple(record.component_draw_rank for record in plan.records) != tuple(range(len(plan.records))):
        raise _draw_error("component draw ranks are not gapless")
    if {record.component_id for record in plan.records} != set(source_by_id):
        raise _draw_error("component draw records differ from A component table")
    if len({record.component_id for record in plan.records}) != len(plan.records):
        raise _draw_error("component draw records are not unique")
    part_rank = {record.part_id: record.part_draw_rank for record in final_draw.records}
    expected_order = tuple(
        sorted(
            source_by_id,
            key=lambda component_id: (
                part_rank[source_by_id[component_id].part_id],
                component_id,
            ),
        )
    )
    if tuple(record.component_id for record in plan.records) != expected_order:
        raise _draw_error("component draw records are not canonical")
    for rank, record in enumerate(plan.records):
        source = source_by_id[record.component_id]
        mesh = mesh_by_component.get(record.component_id)
        if (
            record.part_id != source.part_id
            or record.part_draw_rank != part_rank.get(source.part_id)
            or record.component_draw_rank != rank
            or record.mesh_id != (mesh.mesh_id if mesh is not None else None)
        ):
            raise _draw_error("component draw record differs from source geometry")
    ranges: dict[str, list[int]] = {}
    for record in plan.records:
        ranges.setdefault(record.part_id, []).append(record.component_draw_rank)
    if set(ranges) != set(final_draw.part_order):
        raise _draw_error("component draw records differ from final Parts")
    for ranks in ranges.values():
        if ranks != list(range(min(ranks), max(ranks) + 1)):
            raise _draw_error("a Part's component ranks are not contiguous")
    return plan


def expand_component_draw_order(
    final_draw: FinalDrawOrderPlan,
    mesh_plan: MeshBuildPlan,
) -> ComponentDrawOrderPlan:
    """Expand final Part ranks by stable component ID without reading depth again."""

    _validate_final_draw_order(final_draw)
    try:
        validate_mesh_plan(mesh_plan)
    except MeshBuildError as exc:
        raise _draw_error("MeshBuildPlan is invalid") from exc
    source_by_id = {source.component_id: source for source in mesh_plan.sources}
    mesh_by_component = {mesh.component_id: mesh for mesh in mesh_plan.meshes}
    if any(component_id not in source_by_id for component_id in mesh_by_component):
        raise _draw_error("mesh lies outside the A component table")
    part_rank = {record.part_id: record.part_draw_rank for record in final_draw.records}
    if {source.part_id for source in mesh_plan.sources} != set(final_draw.part_order):
        raise _draw_error("mesh Parts differ from final draw order")
    component_ids = tuple(
        sorted(
            source_by_id,
            key=lambda component_id: (
                part_rank[source_by_id[component_id].part_id],
                component_id,
            ),
        )
    )
    records = tuple(
        ComponentDrawRankRecord(
            component_id=component_id,
            mesh_id=(mesh_by_component[component_id].mesh_id if component_id in mesh_by_component else None),
            part_id=source_by_id[component_id].part_id,
            part_draw_rank=part_rank[source_by_id[component_id].part_id],
            component_draw_rank=rank,
        )
        for rank, component_id in enumerate(component_ids)
    )
    content = {
        "schema_version": COMPONENT_DRAW_ORDER_EXPANDER_VERSION,
        "final_draw_order_plan_sha256": final_draw.plan_sha256,
        "mesh_build_plan_sha256": mesh_plan.plan_sha256,
        "records": [record.to_dict() for record in records],
    }
    plan = ComponentDrawOrderPlan(
        schema_version=COMPONENT_DRAW_ORDER_EXPANDER_VERSION,
        final_draw_order_plan_sha256=final_draw.plan_sha256,
        mesh_build_plan_sha256=mesh_plan.plan_sha256,
        records=records,
        plan_sha256=jcs_sha256(content),
    )
    return validate_component_draw_order(
        plan,
        final_draw=final_draw,
        mesh_plan=mesh_plan,
    )


def _cache_parts(
    component_plan: MaskComponentPlan,
    final_draw: FinalDrawOrderPlan,
    native_variant_set: NativeVariantSet | None,
) -> tuple[RigGeometryPartRecord, ...]:
    ordinary_by_id = {part.part_id: part for part in component_plan.parts}
    partition_by_variant = {partition.variant_id: partition for partition in component_plan.variant_partitions}
    candidate_by_part = (
        {candidate.part_id: candidate for candidate in native_variant_set.entries} if native_variant_set is not None else {}
    )
    draw_by_id = {record.part_id: record for record in final_draw.records}
    parts: list[RigGeometryPartRecord] = []
    for part_id in final_draw.part_order:
        draw = draw_by_id[part_id]
        ordinary = ordinary_by_id.get(part_id)
        if ordinary is not None:
            parts.append(
                RigGeometryPartRecord(
                    part_id=part_id,
                    source_kind="see_through",
                    source_part_id=ordinary.source_part_id,
                    source_tag=ordinary.source_tag,
                    base_tag=ordinary.base_tag,
                    semantic_slug=ordinary.semantic_slug,
                    side=ordinary.side,
                    side_provenance=ordinary.side_provenance,
                    source_xyxy=ordinary.source_xyxy,
                    xyxy=ordinary.xyxy,
                    depth_median=ordinary.depth_median,
                    cleaned_binary_mask_sha256=ordinary.cleaned_binary_mask_sha256,
                    component_ids=tuple(component.component_id for component in ordinary.components),
                    variant_id=None,
                    semantic_role=None,
                    base_part_ids=(),
                    draw_anchor_part_id=None,
                    partition_sha256=None,
                    part_draw_rank=draw.part_draw_rank,
                    depth_bucket=draw.depth_bucket,
                    draw_bundle_id=draw.draw_bundle_id,
                )
            )
            continue
        candidate = candidate_by_part.get(part_id)
        if candidate is None:
            raise _cache_error(f"final draw Part lacks ordinary/variant facts: {part_id}")
        partition = partition_by_variant.get(candidate.variant_id)
        anchor = ordinary_by_id.get(candidate.draw_anchor_part_id)
        if (
            partition is None
            or partition.status != "ready"
            or partition.xyxy is None
            or partition.cleaned_binary_mask_sha256 is None
            or not partition.components
            or anchor is None
        ):
            raise _cache_error(f"NativeVariant Part is not ready: {part_id}")
        parts.append(
            RigGeometryPartRecord(
                part_id=part_id,
                source_kind="native_variant",
                source_part_id=part_id,
                source_tag=None,
                base_tag=anchor.base_tag,
                semantic_slug=part_id.removeprefix("part/"),
                side=anchor.side,
                side_provenance="draw_anchor",
                source_xyxy=partition.source_xyxy,
                xyxy=partition.xyxy,
                depth_median=candidate.anchor_depth_median,
                cleaned_binary_mask_sha256=partition.cleaned_binary_mask_sha256,
                component_ids=tuple(component.component_id for component in partition.components),
                variant_id=candidate.variant_id,
                semantic_role=candidate.semantic_role,
                base_part_ids=candidate.base_part_ids,
                draw_anchor_part_id=candidate.draw_anchor_part_id,
                partition_sha256=partition.partition_sha256,
                part_draw_rank=draw.part_draw_rank,
                depth_bucket=draw.depth_bucket,
                draw_bundle_id=draw.draw_bundle_id,
            )
        )
    return tuple(parts)


def _cache_diagnostics(
    joints: StageAJointPlan,
    bone_graph: BoneGraphPlan,
    mesh_plan: MeshBuildPlan,
    skinning_plan: SkinningPlan,
) -> tuple[RigGeometryDiagnosticRecord, ...]:
    diagnostics: list[RigGeometryDiagnosticRecord] = []
    for item in (*joints.axial.diagnostics, *joints.limb.diagnostics):
        diagnostics.append(
            RigGeometryDiagnosticRecord(
                source_stage="A",
                code=item.reason,
                subject_id=item.joint_id,
                related_ids=item.evidence_ids,
            )
        )
    for item in bone_graph.diagnostics:
        diagnostics.append(
            RigGeometryDiagnosticRecord(
                source_stage="B",
                code=item.code,
                subject_id=item.bone_id,
                related_ids=item.joint_ids,
            )
        )
    for item in mesh_plan.diagnostics:
        diagnostics.append(
            RigGeometryDiagnosticRecord(
                source_stage="B",
                code=item.code,
                subject_id=item.component_id,
                related_ids=(item.part_id,),
            )
        )
    for item in skinning_plan.diagnostics:
        diagnostics.append(
            RigGeometryDiagnosticRecord(
                source_stage="B",
                code=item.code,
                subject_id=item.part_id,
                related_ids=(*item.component_ids, item.selected_bone_id),
            )
        )
    diagnostics.sort(
        key=lambda item: (
            item.source_stage,
            item.code,
            item.subject_id,
            item.related_ids,
        )
    )
    return tuple(diagnostics)


def _dependencies(
    component_plan: MaskComponentPlan,
    final_draw: FinalDrawOrderPlan,
    joints: StageAJointPlan,
    bone_graph: BoneGraphPlan,
    mesh_plan: MeshBuildPlan,
    skinning_plan: SkinningPlan,
    component_draw_order: ComponentDrawOrderPlan,
) -> tuple[RigGeometryDependencyRecord, ...]:
    records = (
        RigGeometryDependencyRecord(
            "mask-component-plan",
            component_plan.schema_version,
            component_plan.plan_sha256,
        ),
        RigGeometryDependencyRecord(
            "mask-cleanup",
            component_plan.cleanup.schema_version,
            jcs_sha256(component_plan.cleanup.to_dict()),
        ),
        RigGeometryDependencyRecord(
            "final-draw-order",
            final_draw.schema_version,
            final_draw.plan_sha256,
        ),
        RigGeometryDependencyRecord(
            "stage-a-joints",
            joints.schema_version,
            joints.plan_sha256,
        ),
        RigGeometryDependencyRecord(
            "bone-graph",
            bone_graph.schema_version,
            bone_graph.plan_sha256,
        ),
        RigGeometryDependencyRecord(
            "mesh-build-descriptor",
            mesh_plan.descriptor.schema_version,
            mesh_plan.descriptor.descriptor_sha256,
        ),
        RigGeometryDependencyRecord(
            "skinning-descriptor",
            skinning_plan.descriptor.schema_version,
            skinning_plan.descriptor.descriptor_sha256,
        ),
        RigGeometryDependencyRecord(
            "component-draw-order",
            component_draw_order.schema_version,
            component_draw_order.plan_sha256,
        ),
    )
    return tuple(sorted(records, key=lambda item: item.name))


def _nested_plan_digest(payload: object, *, field: str) -> None:
    if not isinstance(payload, dict) or not _SHA256_RE.fullmatch(str(payload.get("plan_sha256"))):
        raise _cache_error(f"{field} is not a serialized plan")
    content = {key: value for key, value in payload.items() if key != "plan_sha256"}
    if payload["plan_sha256"] != jcs_sha256(content):
        raise _cache_error(f"{field} digest mismatch")


def validate_rig_geometry_cache_payload(payload: object) -> dict[str, object]:
    """Validate the serialized B schema and reject every C-owned extension."""

    if not isinstance(payload, dict) or set(payload) != _CACHE_PAYLOAD_FIELDS:
        raise _cache_error("serialized cache fields differ from RigGeometryCache v1")
    if payload.get("cache_schema_version") != RIG_GEOMETRY_CACHE_VERSION:
        raise _cache_error("serialized cache version is unsupported")
    cache_sha256 = payload.get("cache_sha256")
    if not isinstance(cache_sha256, str) or not _SHA256_RE.fullmatch(cache_sha256):
        raise _cache_error("serialized cache digest is invalid")
    content = {key: value for key, value in payload.items() if key != "cache_sha256"}
    if cache_sha256 != jcs_sha256(content):
        raise _cache_error("serialized cache digest mismatch")
    for field in (
        "final_draw_order",
        "joint_plan",
        "bone_graph",
        "mesh_plan",
        "skinning_plan",
        "component_draw_order",
    ):
        _nested_plan_digest(payload[field], field=field)
    parts = payload.get("parts")
    if not isinstance(parts, list) or not parts:
        raise _cache_error("serialized cache has no Parts")
    part_ids = [part.get("part_id") for part in parts if isinstance(part, dict)]
    if len(part_ids) != len(parts) or len(part_ids) != len(set(part_ids)):
        raise _cache_error("serialized Part IDs are invalid")
    if [part.get("part_draw_rank") for part in parts] != list(range(len(parts))):
        raise _cache_error("serialized Part draw ranks are not gapless")
    component_ids = [component_id for part in parts for component_id in part.get("component_ids", [])]
    if len(component_ids) != len(set(component_ids)):
        raise _cache_error("serialized component IDs are not unique")
    component_draw = payload["component_draw_order"]
    records = component_draw.get("records") if isinstance(component_draw, dict) else None
    if not isinstance(records, list) or [record.get("component_draw_rank") for record in records] != list(range(len(records))):
        raise _cache_error("serialized component draw ranks are invalid")
    if {record.get("component_id") for record in records} != set(component_ids):
        raise _cache_error("serialized component draw records differ from Parts")
    bone_graph = payload["bone_graph"]
    bones = bone_graph.get("bones") if isinstance(bone_graph, dict) else None
    bone_ids = {bone.get("bone_id") for bone in bones or () if isinstance(bone, dict)}
    if "bone/root" not in bone_ids:
        raise _cache_error("serialized bone graph lacks root")
    skinning = payload["skinning_plan"]
    weighted_meshes = skinning.get("weighted_meshes") if isinstance(skinning, dict) else None
    if not isinstance(weighted_meshes, list):
        raise _cache_error("serialized skinning meshes are invalid")
    mesh_by_component = {mesh.get("component_id"): mesh for mesh in weighted_meshes if isinstance(mesh, dict)}
    if len(mesh_by_component) != len(weighted_meshes) or not set(mesh_by_component) <= set(component_ids):
        raise _cache_error("serialized weighted mesh references are invalid")
    for mesh in weighted_meshes:
        for vertex in mesh.get("vertices", []):
            influences = vertex.get("influences", []) if isinstance(vertex, dict) else []
            if not influences or any(
                influence.get("bone_id") not in bone_ids for influence in influences if isinstance(influence, dict)
            ):
                raise _cache_error("serialized influence references an unknown bone")
    state = payload.get("degradation_state")
    codes = payload.get("degradation_codes")
    if state not in {"clean", "degraded"} or not isinstance(codes, list):
        raise _cache_error("serialized degradation state is invalid")
    if (state == "clean") != (not codes):
        raise _cache_error("serialized degradation state differs from codes")
    return payload


def validate_rig_geometry_cache(cache: RigGeometryCache) -> RigGeometryCache:
    """Validate the B-owned cache and every nested A/B reference closure."""

    if (
        not isinstance(cache, RigGeometryCache)
        or cache.cache_schema_version != RIG_GEOMETRY_CACHE_VERSION
        or cache.generator != RIG_GEOMETRY_CACHE_GENERATOR
        or cache.cache_sha256 != jcs_sha256(cache.semantic_payload())
    ):
        raise _cache_error("cache identity or digest is invalid")
    if any(
        not _SHA256_RE.fullmatch(value)
        for value in (
            cache.target.target_input_fingerprint,
            cache.stage_a_fingerprint,
            cache.stage_b_fingerprint,
            cache.component_plan_sha256,
            cache.native_variant_set_sha256,
            cache.native_variant_eligibility_sha256,
            cache.rig_overrides_sha256,
        )
    ):
        raise _cache_error("cache fingerprint is not canonical")
    if cache.target.target_input_fingerprint != jcs_sha256(cache.target.semantic_payload()):
        raise _cache_error("target identity digest mismatch")
    try:
        _validate_final_draw_order(cache.final_draw_order)
        validate_stage_a_joint_plan(cache.joint_plan)
        validate_bone_graph(cache.bone_graph)
        validate_mesh_plan(cache.mesh_plan)
        validate_skinning_plan(cache.skinning_plan)
        validate_component_draw_order(
            cache.component_draw_order,
            final_draw=cache.final_draw_order,
            mesh_plan=cache.mesh_plan,
        )
    except RigGeometryCacheError:
        raise
    except ValueError as exc:
        raise _cache_error("nested geometry plan is invalid") from exc
    if (
        cache.target.target_input_fingerprint != cache.joint_plan.target_input_fingerprint
        or cache.target.canvas_width != cache.mesh_plan.canvas_edge
        or cache.target.canvas_height != cache.mesh_plan.canvas_edge
        or cache.component_plan_sha256 != cache.mesh_plan.component_plan_sha256
        or cache.rig_overrides_sha256 != cache.joint_plan.rig_overrides_sha256
        or cache.bone_graph.stage_a_joint_plan_sha256 != cache.joint_plan.plan_sha256
        or cache.skinning_plan.mesh_plan_sha256 != cache.mesh_plan.plan_sha256
        or cache.skinning_plan.bone_graph_plan_sha256 != cache.bone_graph.plan_sha256
        or cache.skinning_plan.stage_a_joint_plan_sha256 != cache.joint_plan.plan_sha256
    ):
        raise _cache_error("nested plans describe different geometry inputs")
    if tuple(part.part_id for part in cache.parts) != cache.final_draw_order.part_order:
        raise _cache_error("cache Parts differ from final draw order")
    if tuple(part.part_draw_rank for part in cache.parts) != tuple(range(len(cache.parts))):
        raise _cache_error("cache Part ranks are not gapless")
    component_ids = tuple(component_id for part in cache.parts for component_id in part.component_ids)
    if len(component_ids) != len(set(component_ids)) or set(component_ids) != {
        source.component_id for source in cache.mesh_plan.sources
    }:
        raise _cache_error("cache Part components differ from MeshBuildPlan")
    dependency_names = tuple(item.name for item in cache.dependencies)
    if dependency_names != tuple(sorted(set(dependency_names))) or any(
        not _SHA256_RE.fullmatch(item.sha256) for item in cache.dependencies
    ):
        raise _cache_error("cache dependencies are not canonical")
    diagnostic_keys = tuple((item.source_stage, item.code, item.subject_id, item.related_ids) for item in cache.diagnostics)
    if diagnostic_keys != tuple(sorted(set(diagnostic_keys))):
        raise _cache_error("cache diagnostics are not canonical")
    expected_codes = tuple(
        sorted(
            {
                item.code
                for item in cache.diagnostics
                if item.code in {"degenerate_mesh", "rigid_fallback_applied", "unknown_part_semantics"}
            }
        )
    )
    if cache.degradation_codes != expected_codes or cache.degradation_state != ("degraded" if expected_codes else "clean"):
        raise _cache_error("cache degradation state differs from diagnostics")
    validate_rig_geometry_cache_payload(cache.to_dict())
    return cache


def build_rig_geometry_cache(
    *,
    target: TargetInputIdentity,
    component_plan: MaskComponentPlan,
    final_draw: FinalDrawOrderPlan,
    joints: StageAJointPlan,
    bone_graph: BoneGraphPlan,
    mesh_plan: MeshBuildPlan,
    skinning_plan: SkinningPlan,
    native_variant_set: NativeVariantSet | None,
    stage_a_fingerprint: str,
    stage_b_fingerprint: str,
) -> RigGeometryCache:
    """Assemble the sole B-owned, C-readable geometry fact source."""

    if component_plan.plan_sha256 != jcs_sha256(component_plan.semantic_payload()):
        raise _cache_error("MaskComponentPlan digest mismatch")
    if native_variant_set is not None and (
        native_variant_set.native_variant_set_sha256 != jcs_sha256(native_variant_set.semantic_payload())
        or native_variant_set.native_variant_set_sha256 != final_draw.native_variant_set_sha256
    ):
        raise _cache_error("NativeVariant set differs from final draw order")
    if set(final_draw.part_order) != set(mesh_plan.final_part_ids):
        raise _cache_error("final draw Parts differ from MeshBuildPlan")
    component_draw = expand_component_draw_order(final_draw, mesh_plan)
    parts = _cache_parts(component_plan, final_draw, native_variant_set)
    diagnostics = _cache_diagnostics(joints, bone_graph, mesh_plan, skinning_plan)
    degradation_codes = tuple(
        sorted(
            {
                item.code
                for item in diagnostics
                if item.code in {"degenerate_mesh", "rigid_fallback_applied", "unknown_part_semantics"}
            }
        )
    )
    dependencies = _dependencies(
        component_plan,
        final_draw,
        joints,
        bone_graph,
        mesh_plan,
        skinning_plan,
        component_draw,
    )
    values = {
        "cache_schema_version": RIG_GEOMETRY_CACHE_VERSION,
        "generator": RIG_GEOMETRY_CACHE_GENERATOR,
        "target": target,
        "stage_a_fingerprint": stage_a_fingerprint,
        "stage_b_fingerprint": stage_b_fingerprint,
        "component_plan_sha256": component_plan.plan_sha256,
        "native_variant_set_sha256": final_draw.native_variant_set_sha256,
        "native_variant_eligibility_sha256": (final_draw.native_variant_eligibility_plan_sha256),
        "rig_overrides_sha256": joints.rig_overrides_sha256,
        "parts": parts,
        "final_draw_order": final_draw,
        "joint_plan": joints,
        "bone_graph": bone_graph,
        "mesh_plan": mesh_plan,
        "skinning_plan": skinning_plan,
        "component_draw_order": component_draw,
        "diagnostics": diagnostics,
        "dependencies": dependencies,
        "degradation_state": "degraded" if degradation_codes else "clean",
        "degradation_codes": degradation_codes,
    }
    provisional = RigGeometryCache(**values, cache_sha256="")
    cache = RigGeometryCache(
        **values,
        cache_sha256=jcs_sha256(provisional.semantic_payload()),
    )
    return validate_rig_geometry_cache(cache)


def rig_geometry_cache_bytes(cache: RigGeometryCache) -> bytes:
    """Serialize a validated cache as canonical ASCII JSON with one trailing newline."""

    validate_rig_geometry_cache(cache)
    return canonical_json_bytes(cache.to_dict()) + b"\n"


def _decode_cache_value(value: object, annotation: object) -> object:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if is_dataclass(annotation):
        if isinstance(value, annotation):
            return value
        if not isinstance(value, dict):
            raise _cache_error("serialized nested cache value is not an object")
        hints = get_type_hints(annotation)
        expected = {field.name for field in fields(annotation)}
        if set(value) != expected:
            raise _cache_error("serialized nested cache fields differ from the typed plan")
        return annotation(**{name: _decode_cache_value(value[name], hints[name]) for name in sorted(expected)})
    if origin is tuple:
        if not isinstance(value, list):
            raise _cache_error("serialized tuple cache value is not an array")
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(_decode_cache_value(item, arguments[0]) for item in value)
        if len(value) != len(arguments):
            raise _cache_error("serialized fixed tuple has the wrong length")
        return tuple(_decode_cache_value(item, item_type) for item, item_type in zip(value, arguments, strict=True))
    if origin is list:
        if not isinstance(value, list):
            raise _cache_error("serialized list cache value is not an array")
        item_type = arguments[0] if arguments else Any
        return [_decode_cache_value(item, item_type) for item in value]
    if origin in {Union, types.UnionType}:
        if value is None and type(None) in arguments:
            return None
        candidates = tuple(item for item in arguments if item is not type(None))
        if len(candidates) == 1:
            return _decode_cache_value(value, candidates[0])
    return value


def load_rig_geometry_cache(
    item_root: str | Path,
    *,
    target: TargetInputIdentity,
) -> RigGeometryCache:
    """Reload the authenticated B cache without recomputing meshes or weights."""

    root = Path(item_root).resolve(strict=True)
    path = root / Path(*RIG_GEOMETRY_CACHE_PATH.split("/"))
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _cache_error("serialized cache is not valid UTF-8 JSON") from exc
    validated = validate_rig_geometry_cache_payload(payload)
    if not isinstance(target, TargetInputIdentity):
        raise _cache_error("cache reload requires the current TargetInputIdentity")
    canvas = validated["canvas"]
    expected_canvas = {
        "width": target.canvas_width,
        "height": target.canvas_height,
        "resolution": target.canvas_resolution,
        "coordinate_space": "layerdiff_canvas",
        "origin": "top_left",
        "y_axis": "down",
        "tag_version": target.tag_version,
    }
    if validated["target_input_fingerprint"] != target.target_input_fingerprint or canvas != expected_canvas:
        raise _cache_error("serialized cache target identity differs from current input")
    typed_payload = {key: value for key, value in validated.items() if key not in {"target_input_fingerprint", "canvas"}}
    typed_payload["target"] = target
    try:
        cache = _decode_cache_value(typed_payload, RigGeometryCache)
    except RigGeometryCacheError:
        raise
    except (TypeError, ValueError) as exc:
        raise _cache_error("serialized cache cannot be restored to typed plans") from exc
    if not isinstance(cache, RigGeometryCache):
        raise _cache_error("serialized cache did not restore a RigGeometryCache")
    return validate_rig_geometry_cache(cache)


__all__ = [
    "COMPONENT_DRAW_ORDER_EXPANDER_VERSION",
    "RIG_GEOMETRY_CACHE_GENERATOR",
    "RIG_GEOMETRY_CACHE_PATH",
    "RIG_GEOMETRY_CACHE_VERSION",
    "ComponentDrawOrderError",
    "ComponentDrawOrderPlan",
    "ComponentDrawRankRecord",
    "RigGeometryCache",
    "RigGeometryCacheError",
    "RigGeometryDependencyRecord",
    "RigGeometryDiagnosticRecord",
    "RigGeometryPartRecord",
    "build_rig_geometry_cache",
    "expand_component_draw_order",
    "load_rig_geometry_cache",
    "rig_geometry_cache_bytes",
    "validate_component_draw_order",
    "validate_rig_geometry_cache",
    "validate_rig_geometry_cache_payload",
]
