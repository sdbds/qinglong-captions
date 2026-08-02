from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from ...jcs import jcs_sha256
from .bind_plan import SpineMeshBind, SpineVertexInfluenceBind
from .uv import canvas_to_spine_region_uv

SPINE_MESH_ENCODER_VERSION = "spine-mesh-encoder-v1"


class SpineMeshEncodingError(ValueError):
    """Raised when a Rig mesh cannot be encoded as Spine 4.2 flat arrays."""

    def __init__(self, message: str) -> None:
        super().__init__(f"invalid_spine_mesh: {message}")


def _error(message: str) -> SpineMeshEncodingError:
    return SpineMeshEncodingError(message)


def _finite(value: object, *, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise _error(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise _error(f"{field} must be finite")
    return 0.0 if abs(result) < 1e-12 else result


@dataclass(frozen=True, slots=True)
class SpineEncodedMesh:
    schema_version: str
    mesh_id: str
    weighted: bool
    vertex_count: int
    slot_bone_id: str
    attachment_key_name: str
    attachment_object_name: str
    atlas_region_name: str
    payload: dict[str, object]
    payload_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mesh_id": self.mesh_id,
            "weighted": self.weighted,
            "vertex_count": self.vertex_count,
            "slot_bone_id": self.slot_bone_id,
            "attachment_key_name": self.attachment_key_name,
            "attachment_object_name": self.attachment_object_name,
            "atlas_region_name": self.atlas_region_name,
            "payload": self.payload,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "payload_sha256": self.payload_sha256}


def _triangles(value: object, *, vertex_count: int) -> list[int]:
    if not isinstance(value, list) or not value or len(value) % 3:
        raise _error("triangles must be a non-empty flat triplet array")
    result = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise _error("triangles must contain flat integer indices")
        if not 0 <= item < vertex_count:
            raise _error("triangle index is outside the vertex array")
        result.append(item)
    return result


def _weighted_vertices(vertices) -> list[int | float]:
    result: list[int | float] = []
    for vertex in vertices:
        result.append(len(vertex.influences))
        for influence in vertex.influences:
            result.extend(
                (
                    influence.bone_index,
                    influence.x,
                    influence.y,
                    influence.weight,
                )
            )
    return result


def _rigid_vertices(
    vertices,
    *,
    slot_bone_id: str,
) -> list[float] | None:
    result = []
    for vertex in vertices:
        if (
            len(vertex.influences) != 1
            or vertex.influences[0].bone_id != slot_bone_id
            or abs(vertex.influences[0].weight - 1.0) > 1e-6
        ):
            return None
        result.extend((vertex.influences[0].x, vertex.influences[0].y))
    return result


def encode_spine_mesh(
    mesh: Mapping[str, object],
    bind: SpineMeshBind,
    *,
    source_xyxy: Sequence[float],
    attachment_key_name: str,
    attachment_object_name: str,
    atlas_region_name: str,
) -> SpineEncodedMesh:
    if not isinstance(mesh, Mapping) or not isinstance(bind, SpineMeshBind):
        raise _error("mesh and bind records have the wrong type")
    if mesh.get("mesh_id") != bind.mesh_id:
        raise _error("mesh and bind identities differ")
    raw_vertices = mesh.get("vertices")
    if not isinstance(raw_vertices, list) or len(raw_vertices) != len(bind.vertices):
        raise _error("mesh and bind vertex counts differ")
    for name in (attachment_key_name, attachment_object_name, atlas_region_name):
        if not isinstance(name, str) or not name or not name.isascii():
            raise _error("attachment and atlas names must be non-empty ASCII")

    uvs: list[float] = []
    for index, (raw_vertex, bound_vertex) in enumerate(
        zip(raw_vertices, bind.vertices, strict=True)
    ):
        if not isinstance(raw_vertex, Mapping):
            raise _error("mesh vertex is not an object")
        if bound_vertex.vertex_index != index:
            raise _error("bind vertex indices are not canonical")
        computed_uv = canvas_to_spine_region_uv(
            raw_vertex.get("position"), source_xyxy
        )
        raw_uv = raw_vertex.get("uv")
        if not isinstance(raw_uv, (list, tuple)) or len(raw_uv) != 2:
            raise _error("mesh vertex lacks a local UV pair")
        checked_uv = (
            _finite(raw_uv[0], field="mesh u"),
            _finite(raw_uv[1], field="mesh v"),
        )
        if math.dist(computed_uv, checked_uv) > 1e-9:
            raise _error("mesh UV differs from the Spine42UvAdapter result")
        uvs.extend(computed_uv)

    rigid_vertices = _rigid_vertices(
        bind.vertices, slot_bone_id=bind.slot_bone_id
    )
    weighted = rigid_vertices is None
    vertices: list[int | float] = (
        _weighted_vertices(bind.vertices) if weighted else rigid_vertices
    )
    payload: dict[str, object] = {
        "type": "mesh",
        "uvs": uvs,
        "triangles": _triangles(mesh.get("triangles"), vertex_count=len(raw_vertices)),
        "vertices": vertices,
    }
    if attachment_object_name != attachment_key_name:
        payload["name"] = attachment_object_name
    if atlas_region_name != attachment_object_name:
        payload["path"] = atlas_region_name
    provisional = SpineEncodedMesh(
        schema_version=SPINE_MESH_ENCODER_VERSION,
        mesh_id=bind.mesh_id,
        weighted=weighted,
        vertex_count=len(raw_vertices),
        slot_bone_id=bind.slot_bone_id,
        attachment_key_name=attachment_key_name,
        attachment_object_name=attachment_object_name,
        atlas_region_name=atlas_region_name,
        payload=payload,
        payload_sha256="",
    )
    return SpineEncodedMesh(
        schema_version=provisional.schema_version,
        mesh_id=provisional.mesh_id,
        weighted=provisional.weighted,
        vertex_count=provisional.vertex_count,
        slot_bone_id=provisional.slot_bone_id,
        attachment_key_name=provisional.attachment_key_name,
        attachment_object_name=provisional.attachment_object_name,
        atlas_region_name=provisional.atlas_region_name,
        payload=provisional.payload,
        payload_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def parse_weighted_vertices(
    values: object,
    *,
    vertex_count: int,
) -> tuple[tuple[tuple[int, float, float, float], ...], ...]:
    if not isinstance(values, list):
        raise _error("weighted vertices must be a flat list")
    cursor = 0
    result = []
    for _vertex_index in range(vertex_count):
        if cursor >= len(values):
            raise _error("weighted vertices ended before all vertices")
        count = values[cursor]
        cursor += 1
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 4:
            raise _error("weighted vertex bone count is invalid")
        influences = []
        for _influence_index in range(count):
            if cursor + 4 > len(values):
                raise _error("weighted influence is truncated")
            bone_index = values[cursor]
            if not isinstance(bone_index, int) or isinstance(bone_index, bool) or bone_index < 0:
                raise _error("weighted bone index is invalid")
            x = _finite(values[cursor + 1], field="weighted x")
            y = _finite(values[cursor + 2], field="weighted y")
            weight = _finite(values[cursor + 3], field="weighted weight")
            if not 0.0 < weight <= 1.0:
                raise _error("weighted influence weight is invalid")
            influences.append((bone_index, x, y, weight))
            cursor += 4
        if abs(sum(item[3] for item in influences) - 1.0) > 1e-6:
            raise _error("weighted vertex influences are not normalized")
        result.append(tuple(influences))
    if cursor != len(values):
        raise _error("weighted vertices contain trailing values")
    return tuple(result)


def validate_spine_encoded_mesh(
    encoded: SpineEncodedMesh,
    mesh: Mapping[str, object],
    bind: SpineMeshBind,
) -> SpineEncodedMesh:
    if not isinstance(encoded, SpineEncodedMesh):
        raise _error("encoded mesh has the wrong type")
    if encoded.schema_version != SPINE_MESH_ENCODER_VERSION:
        raise _error("mesh encoder version is unsupported")
    if encoded.payload_sha256 != jcs_sha256(encoded.semantic_payload()):
        raise _error("encoded mesh digest mismatch")
    if encoded.weighted:
        parsed = parse_weighted_vertices(
            encoded.payload.get("vertices"), vertex_count=encoded.vertex_count
        )
        for source_vertex, parsed_vertex in zip(bind.vertices, parsed, strict=True):
            expected = tuple(
                (
                    influence.bone_index,
                    influence.x,
                    influence.y,
                    influence.weight,
                )
                for influence in source_vertex.influences
            )
            if parsed_vertex != expected:
                raise _error("weighted vertices differ from the BindPlan")
    else:
        expected_rigid = _rigid_vertices(
            bind.vertices, slot_bone_id=bind.slot_bone_id
        )
        if expected_rigid is None or encoded.payload.get("vertices") != expected_rigid:
            raise _error("rigid vertices differ from the slot-bone BindPlan")
    _triangles(mesh.get("triangles"), vertex_count=encoded.vertex_count)
    return encoded


__all__ = [
    "SPINE_MESH_ENCODER_VERSION",
    "SpineEncodedMesh",
    "SpineMeshEncodingError",
    "encode_spine_mesh",
    "parse_weighted_vertices",
    "validate_spine_encoded_mesh",
]
