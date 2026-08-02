from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from ...jcs import jcs_sha256
from .coordinates import (
    SpineCoordinatePlan,
    canvas_to_spine,
    validate_spine_coordinate_plan,
)

SPINE_BIND_PLAN_VERSION = "spine-bind-plan-v1"
SPINE_BIND_AFFINE_VERSION = "parent-world-inverse-trs-v1"

Affine = tuple[float, float, float, float, float, float]


class SpineBindPlanError(ValueError):
    """Raised when a Rig hierarchy cannot be represented as a Spine setup bind."""

    def __init__(self, message: str) -> None:
        super().__init__(f"invalid_spine_bind_plan: {message}")


def _error(message: str) -> SpineBindPlanError:
    return SpineBindPlanError(message)


def _clean(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if abs(rounded) < 1e-12 else rounded


def _point(value: object, *, field: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise _error(f"{field} must be an x/y pair")
    if any(
        not isinstance(item, (int, float))
        or isinstance(item, bool)
        or not math.isfinite(float(item))
        for item in value
    ):
        raise _error(f"{field} must contain finite numbers")
    return float(value[0]), float(value[1])


def _mul(left: Affine, right: Affine) -> Affine:
    la, lb, lc, ld, ltx, lty = left
    ra, rb, rc, rd, rtx, rty = right
    return (
        _clean(la * ra + lc * rb),
        _clean(lb * ra + ld * rb),
        _clean(la * rc + lc * rd),
        _clean(lb * rc + ld * rd),
        _clean(la * rtx + lc * rty + ltx),
        _clean(lb * rtx + ld * rty + lty),
    )


def _trs(x: float, y: float, angle: float) -> Affine:
    radians = math.radians(angle)
    cosine = _clean(math.cos(radians))
    sine = _clean(math.sin(radians))
    return cosine, sine, -sine, cosine, _clean(x), _clean(y)


def _inverse(matrix: Affine) -> Affine:
    a, b, c, d, tx, ty = matrix
    determinant = a * d - b * c
    if abs(determinant) < 1e-12:
        raise _error("world affine is singular")
    inverse_det = 1.0 / determinant
    ia = d * inverse_det
    ib = -b * inverse_det
    ic = -c * inverse_det
    id_value = a * inverse_det
    return (
        _clean(ia),
        _clean(ib),
        _clean(ic),
        _clean(id_value),
        _clean(-(ia * tx + ic * ty)),
        _clean(-(ib * tx + id_value * ty)),
    )


def _apply(matrix: Affine, point: tuple[float, float]) -> tuple[float, float]:
    a, b, c, d, tx, ty = matrix
    x, y = point
    return _clean(a * x + c * y + tx), _clean(b * x + d * y + ty)


def _normalize_angle(value: float) -> float:
    normalized = (value + 180.0) % 360.0 - 180.0
    return _clean(normalized)


@dataclass(frozen=True, slots=True)
class SpineBoneBind:
    bone_id: str
    parent_id: str | None
    index: int
    x: float
    y: float
    rotation: float
    world_rotation: float
    length: float
    world_matrix: Affine
    canvas_head: tuple[float, float] | None
    canvas_tail: tuple[float, float] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "bone_id": self.bone_id,
            "parent_id": self.parent_id,
            "index": self.index,
            "x": self.x,
            "y": self.y,
            "rotation": self.rotation,
            "world_rotation": self.world_rotation,
            "length": self.length,
            "world_matrix": list(self.world_matrix),
            "canvas_head": None if self.canvas_head is None else list(self.canvas_head),
            "canvas_tail": None if self.canvas_tail is None else list(self.canvas_tail),
        }


@dataclass(frozen=True, slots=True)
class SpineVertexInfluenceBind:
    bone_id: str
    bone_index: int
    x: float
    y: float
    weight: float

    def to_dict(self) -> dict[str, object]:
        return {
            "bone_id": self.bone_id,
            "bone_index": self.bone_index,
            "x": self.x,
            "y": self.y,
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class SpineVertexBind:
    vertex_index: int
    canvas_position: tuple[float, float]
    spine_position: tuple[float, float]
    influences: tuple[SpineVertexInfluenceBind, ...]
    maximum_reconstruction_error: float

    def to_dict(self) -> dict[str, object]:
        return {
            "vertex_index": self.vertex_index,
            "canvas_position": list(self.canvas_position),
            "spine_position": list(self.spine_position),
            "influences": [item.to_dict() for item in self.influences],
            "maximum_reconstruction_error": self.maximum_reconstruction_error,
        }


@dataclass(frozen=True, slots=True)
class SpineMeshBind:
    mesh_id: str
    part_id: str
    component_id: str
    slot_bone_id: str
    vertices: tuple[SpineVertexBind, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "mesh_id": self.mesh_id,
            "part_id": self.part_id,
            "component_id": self.component_id,
            "slot_bone_id": self.slot_bone_id,
            "vertices": [vertex.to_dict() for vertex in self.vertices],
        }


@dataclass(frozen=True, slots=True)
class SpineBindPlan:
    schema_version: str
    affine_version: str
    coordinate_plan_sha256: str
    bone_input_sha256: str
    mesh_input_sha256: str
    bones: tuple[SpineBoneBind, ...]
    meshes: tuple[SpineMeshBind, ...]
    maximum_reconstruction_error: float
    output_sha256: str
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "affine_version": self.affine_version,
            "coordinate_plan_sha256": self.coordinate_plan_sha256,
            "bone_input_sha256": self.bone_input_sha256,
            "mesh_input_sha256": self.mesh_input_sha256,
            "bones": [bone.to_dict() for bone in self.bones],
            "meshes": [mesh.to_dict() for mesh in self.meshes],
            "maximum_reconstruction_error": self.maximum_reconstruction_error,
            "output_sha256": self.output_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "plan_sha256": self.plan_sha256}


def _ancestors(bone_id: str, parents: Mapping[str, str | None]) -> tuple[str, ...]:
    result = []
    current: str | None = bone_id
    seen: set[str] = set()
    while current is not None:
        if current in seen:
            raise _error("bone hierarchy contains a cycle")
        seen.add(current)
        result.append(current)
        current = parents.get(current)
    return tuple(result)


def _nearest_common_ancestor(
    bone_ids: tuple[str, ...],
    parents: Mapping[str, str | None],
) -> str:
    if not bone_ids:
        raise _error("mesh vertex has no influences")
    chains = [_ancestors(bone_id, parents) for bone_id in bone_ids]
    other_sets = [set(chain) for chain in chains[1:]]
    for candidate in chains[0]:
        if all(candidate in values for values in other_sets):
            return candidate
    raise _error("influence bones do not share a common ancestor")


def _build_bones(
    source_bones: tuple[Mapping[str, object], ...],
    coordinate_plan: SpineCoordinatePlan,
) -> tuple[SpineBoneBind, ...]:
    result: list[SpineBoneBind] = []
    by_id: dict[str, SpineBoneBind] = {}
    seen: set[str] = set()
    for index, source in enumerate(source_bones):
        bone_id = source.get("bone_id")
        parent_id = source.get("parent_id")
        if not isinstance(bone_id, str) or not bone_id or bone_id in seen:
            raise _error("bone IDs must be non-empty and unique")
        if parent_id is not None and not isinstance(parent_id, str):
            raise _error(f"invalid parent for {bone_id}")
        if parent_id is not None and parent_id not in by_id:
            raise _error(f"bone parent is missing or not topological: {bone_id}")
        seen.add(bone_id)
        if bone_id == "bone/root":
            if index != 0 or parent_id is not None or source.get("head") is not None or source.get("tail") is not None:
                raise _error("synthetic root must be the first identity bone")
            bind = SpineBoneBind(
                bone_id=bone_id,
                parent_id=None,
                index=index,
                x=0.0,
                y=0.0,
                rotation=0.0,
                world_rotation=0.0,
                length=0.0,
                world_matrix=(1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
                canvas_head=None,
                canvas_tail=None,
            )
        else:
            if parent_id is None:
                raise _error(f"non-root bone lacks a parent: {bone_id}")
            canvas_head = _point(source.get("head"), field=f"{bone_id} head")
            canvas_tail = _point(source.get("tail"), field=f"{bone_id} tail")
            head = canvas_to_spine(coordinate_plan, *canvas_head)
            tail = canvas_to_spine(coordinate_plan, *canvas_tail)
            length = math.hypot(tail[0] - head[0], tail[1] - head[1])
            if length <= 1e-9:
                raise _error(f"non-root bone has zero length: {bone_id}")
            parent = by_id[parent_id]
            local_head = _apply(_inverse(parent.world_matrix), head)
            world_rotation = math.degrees(
                math.atan2(tail[1] - head[1], tail[0] - head[0])
            )
            local_rotation = _normalize_angle(world_rotation - parent.world_rotation)
            world_matrix = _mul(
                parent.world_matrix,
                _trs(local_head[0], local_head[1], local_rotation),
            )
            reconstructed_head = _apply(world_matrix, (0.0, 0.0))
            reconstructed_tail = _apply(world_matrix, (length, 0.0))
            error = max(
                math.dist(reconstructed_head, head),
                math.dist(reconstructed_tail, tail),
            )
            if error > 0.1:
                raise _error(f"bone setup reconstruction exceeded tolerance: {bone_id}")
            bind = SpineBoneBind(
                bone_id=bone_id,
                parent_id=parent_id,
                index=index,
                x=_clean(local_head[0]),
                y=_clean(local_head[1]),
                rotation=_clean(local_rotation),
                world_rotation=_clean(world_rotation),
                length=_clean(length),
                world_matrix=world_matrix,
                canvas_head=canvas_head,
                canvas_tail=canvas_tail,
            )
        result.append(bind)
        by_id[bone_id] = bind
    if not result or result[0].bone_id != "bone/root":
        raise _error("bone/root is required")
    return tuple(result)


def _build_meshes(
    source_meshes: tuple[Mapping[str, object], ...],
    bones: tuple[SpineBoneBind, ...],
    coordinate_plan: SpineCoordinatePlan,
) -> tuple[SpineMeshBind, ...]:
    bone_by_id = {bone.bone_id: bone for bone in bones}
    parents = {bone.bone_id: bone.parent_id for bone in bones}
    result = []
    seen_meshes: set[str] = set()
    for source in source_meshes:
        mesh_id = source.get("mesh_id")
        part_id = source.get("part_id")
        component_id = source.get("component_id")
        if not all(isinstance(value, str) and value for value in (mesh_id, part_id, component_id)):
            raise _error("mesh identity fields must be non-empty strings")
        assert isinstance(mesh_id, str)
        assert isinstance(part_id, str)
        assert isinstance(component_id, str)
        if mesh_id in seen_meshes:
            raise _error("mesh IDs must be unique")
        seen_meshes.add(mesh_id)
        raw_vertices = source.get("vertices")
        if not isinstance(raw_vertices, list) or not raw_vertices:
            raise _error(f"mesh has no vertices: {mesh_id}")
        vertices = []
        mesh_bone_ids: set[str] = set()
        for vertex_index, raw_vertex in enumerate(raw_vertices):
            if not isinstance(raw_vertex, Mapping):
                raise _error(f"invalid vertex in {mesh_id}")
            canvas_position = _point(
                raw_vertex.get("position"), field=f"{mesh_id} vertex position"
            )
            spine_position = canvas_to_spine(coordinate_plan, *canvas_position)
            raw_influences = raw_vertex.get("influences")
            if not isinstance(raw_influences, list) or not raw_influences:
                raise _error(f"mesh vertex has no influences: {mesh_id}")
            influences = []
            influence_ids: set[str] = set()
            total_weight = 0.0
            maximum_error = 0.0
            for raw_influence in raw_influences:
                if not isinstance(raw_influence, Mapping):
                    raise _error(f"invalid influence in {mesh_id}")
                bone_id = raw_influence.get("bone_id")
                if not isinstance(bone_id, str) or bone_id not in bone_by_id:
                    raise _error(f"influence references an unknown bone: {bone_id}")
                if bone_id in influence_ids:
                    raise _error(f"vertex repeats an influence bone: {mesh_id}")
                influence_ids.add(bone_id)
                mesh_bone_ids.add(bone_id)
                weight_value = raw_influence.get("weight")
                if not isinstance(weight_value, (int, float)) or isinstance(weight_value, bool):
                    raise _error(f"influence weight is not numeric: {mesh_id}")
                weight = float(weight_value)
                if not math.isfinite(weight) or weight <= 0.0 or weight > 1.0:
                    raise _error(f"influence weight is invalid: {mesh_id}")
                total_weight += weight
                bone = bone_by_id[bone_id]
                local = _apply(_inverse(bone.world_matrix), spine_position)
                reconstructed = _apply(bone.world_matrix, local)
                maximum_error = max(maximum_error, math.dist(reconstructed, spine_position))
                influences.append(
                    SpineVertexInfluenceBind(
                        bone_id=bone_id,
                        bone_index=bone.index,
                        x=_clean(local[0]),
                        y=_clean(local[1]),
                        weight=_clean(weight),
                    )
                )
            if abs(total_weight - 1.0) > 1e-6:
                raise _error(f"influence weights are not normalized: {mesh_id}")
            vertices.append(
                SpineVertexBind(
                    vertex_index=vertex_index,
                    canvas_position=canvas_position,
                    spine_position=spine_position,
                    influences=tuple(sorted(influences, key=lambda item: item.bone_id)),
                    maximum_reconstruction_error=_clean(maximum_error),
                )
            )
        slot_bone_id = _nearest_common_ancestor(tuple(sorted(mesh_bone_ids)), parents)
        result.append(
            SpineMeshBind(
                mesh_id=mesh_id,
                part_id=part_id,
                component_id=component_id,
                slot_bone_id=slot_bone_id,
                vertices=tuple(vertices),
            )
        )
    return tuple(result)


def build_spine_bind_plan(
    bones: Iterable[Mapping[str, object]],
    meshes: Iterable[Mapping[str, object]],
    coordinate_plan: SpineCoordinatePlan,
) -> SpineBindPlan:
    validate_spine_coordinate_plan(coordinate_plan)
    source_bones = tuple(bones)
    source_meshes = tuple(meshes)
    if any(not isinstance(value, Mapping) for value in source_bones + source_meshes):
        raise _error("bones and meshes must contain objects")
    materialized_bones = _build_bones(source_bones, coordinate_plan)
    materialized_meshes = _build_meshes(
        source_meshes, materialized_bones, coordinate_plan
    )
    maximum_error = max(
        (
            vertex.maximum_reconstruction_error
            for mesh in materialized_meshes
            for vertex in mesh.vertices
        ),
        default=0.0,
    )
    output_payload = {
        "bones": [bone.to_dict() for bone in materialized_bones],
        "meshes": [mesh.to_dict() for mesh in materialized_meshes],
        "maximum_reconstruction_error": maximum_error,
    }
    provisional = SpineBindPlan(
        schema_version=SPINE_BIND_PLAN_VERSION,
        affine_version=SPINE_BIND_AFFINE_VERSION,
        coordinate_plan_sha256=coordinate_plan.plan_sha256,
        bone_input_sha256=jcs_sha256(list(source_bones)),
        mesh_input_sha256=jcs_sha256(list(source_meshes)),
        bones=materialized_bones,
        meshes=materialized_meshes,
        maximum_reconstruction_error=maximum_error,
        output_sha256=jcs_sha256(output_payload),
        plan_sha256="",
    )
    plan = SpineBindPlan(
        schema_version=provisional.schema_version,
        affine_version=provisional.affine_version,
        coordinate_plan_sha256=provisional.coordinate_plan_sha256,
        bone_input_sha256=provisional.bone_input_sha256,
        mesh_input_sha256=provisional.mesh_input_sha256,
        bones=provisional.bones,
        meshes=provisional.meshes,
        maximum_reconstruction_error=provisional.maximum_reconstruction_error,
        output_sha256=provisional.output_sha256,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )
    if plan.maximum_reconstruction_error > 0.1:
        raise _error("weighted setup reconstruction exceeds 0.1 px")
    return plan


def reconstruct_spine_point(
    plan: SpineBindPlan,
    bone_id: str,
    local_point: tuple[float, float],
) -> tuple[float, float]:
    bone = next((item for item in plan.bones if item.bone_id == bone_id), None)
    if bone is None:
        raise _error(f"unknown bone: {bone_id}")
    return _apply(bone.world_matrix, _point(local_point, field="local point"))


def validate_spine_bind_plan(
    plan: SpineBindPlan,
    bones: Iterable[Mapping[str, object]],
    meshes: Iterable[Mapping[str, object]],
    coordinate_plan: SpineCoordinatePlan,
) -> SpineBindPlan:
    if not isinstance(plan, SpineBindPlan):
        raise _error("plan has the wrong type")
    if (
        plan.schema_version != SPINE_BIND_PLAN_VERSION
        or plan.affine_version != SPINE_BIND_AFFINE_VERSION
    ):
        raise _error("bind version is unsupported")
    if plan.plan_sha256 != jcs_sha256(plan.semantic_payload()):
        raise _error("plan digest mismatch")
    expected = build_spine_bind_plan(bones, meshes, coordinate_plan)
    if plan != expected:
        raise _error("plan differs from recomputed bind or reconstruction")
    return plan


__all__ = [
    "SPINE_BIND_AFFINE_VERSION",
    "SPINE_BIND_PLAN_VERSION",
    "SpineBindPlan",
    "SpineBindPlanError",
    "SpineBoneBind",
    "SpineMeshBind",
    "SpineVertexBind",
    "SpineVertexInfluenceBind",
    "build_spine_bind_plan",
    "reconstruct_spine_point",
    "validate_spine_bind_plan",
]
