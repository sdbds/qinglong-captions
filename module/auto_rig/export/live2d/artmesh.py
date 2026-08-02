from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from ...jcs import jcs_sha256
from ...rig_document import RigDocument, validate_rig_document
from .binding_plan import Live2DBindingPlan
from .coordinates import Live2DCoordinatePlan, canvas_to_artmesh_local
from .symbols import (
    Live2DSymbolView,
    require_live2d_symbol,
    validate_live2d_symbol_view,
)
from .uv_kernel import canonical_top_left_to_moc_uv

LIVE2D_ARTMESH_PLAN_VERSION = "live2d-artmesh-plan-v1"


class Live2DArtMeshError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(f"invalid_live2d_artmesh_plan: {message}")


def _error(message: str) -> Live2DArtMeshError:
    return Live2DArtMeshError(message)


@dataclass(frozen=True, slots=True)
class Live2DPartPlan:
    part_id: str
    export_name: str
    draw_rank: int
    setup_opacity: float
    symbol_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "part_id": self.part_id,
            "export_name": self.export_name,
            "draw_rank": self.draw_rank,
            "setup_opacity": self.setup_opacity,
            "symbol_sha256": self.symbol_sha256,
        }


@dataclass(frozen=True, slots=True)
class Live2DArtMeshPlanRecord:
    mesh_id: str
    part_id: str
    component_id: str
    primitive_target_id: str
    export_name: str
    symbol_sha256: str
    parent_instance_id: str | None
    texture_index: int
    draw_order: int
    setup_opacity: float
    positions: tuple[tuple[float, float], ...]
    uvs: tuple[tuple[float, float], ...]
    triangle_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "mesh_id": self.mesh_id,
            "part_id": self.part_id,
            "component_id": self.component_id,
            "primitive_target_id": self.primitive_target_id,
            "export_name": self.export_name,
            "symbol_sha256": self.symbol_sha256,
            "parent_instance_id": self.parent_instance_id,
            "texture_index": self.texture_index,
            "draw_order": self.draw_order,
            "setup_opacity": self.setup_opacity,
            "positions": [list(point) for point in self.positions],
            "uvs": [list(point) for point in self.uvs],
            "triangle_indices": list(self.triangle_indices),
        }


@dataclass(frozen=True, slots=True)
class Live2DArtMeshPlan:
    schema_version: str
    rig_document_sha256: str
    symbol_view_sha256: str
    binding_plan_sha256: str
    coordinate_plan_sha256: str
    texture_page_count: int
    parts: tuple[Live2DPartPlan, ...]
    artmeshes: tuple[Live2DArtMeshPlanRecord, ...]
    total_vertex_count: int
    total_index_count: int
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "rig_document_sha256": self.rig_document_sha256,
            "symbol_view_sha256": self.symbol_view_sha256,
            "binding_plan_sha256": self.binding_plan_sha256,
            "coordinate_plan_sha256": self.coordinate_plan_sha256,
            "texture_page_count": self.texture_page_count,
            "parts": [part.to_dict() for part in self.parts],
            "artmeshes": [artmesh.to_dict() for artmesh in self.artmeshes],
            "total_vertex_count": self.total_vertex_count,
            "total_index_count": self.total_index_count,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "plan_sha256": self.plan_sha256}


def _finite_pair(value: object, *, field: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise _error(f"{field} must be an x/y pair")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise _error(f"{field} must be finite")
    return result  # type: ignore[return-value]


def _assemble(
    rig: RigDocument,
    symbols: Live2DSymbolView,
    bindings: Live2DBindingPlan,
    coordinates: Live2DCoordinatePlan,
) -> Live2DArtMeshPlan:
    payload = rig.to_dict()
    raw_parts = payload["parts"]
    parts = []
    part_by_id: dict[str, Mapping[str, object]] = {}
    for raw in sorted(raw_parts, key=lambda value: value["part_draw_rank"]):
        if not isinstance(raw, Mapping):
            raise _error("Rig Part is not an object")
        part_id = raw.get("part_id")
        rank = raw.get("part_draw_rank")
        visibility = raw.get("setup_visibility")
        if (
            not isinstance(part_id, str)
            or not isinstance(rank, int)
            or isinstance(rank, bool)
            or visibility not in (0, 1)
            or part_id in part_by_id
        ):
            raise _error("Rig Part identity/draw/setup contract is invalid")
        symbol = require_live2d_symbol(
            symbols, kind="live2d_part", source_internal_ids=(part_id,)
        )
        parts.append(
            Live2DPartPlan(
                part_id=part_id,
                export_name=symbol.export_name,
                draw_rank=rank,
                setup_opacity=float(visibility),
                symbol_sha256=symbol.symbol_sha256,
            )
        )
        part_by_id[part_id] = raw
    if [part.draw_rank for part in parts] != list(range(len(parts))):
        raise _error("Part draw ranks are not gapless")

    pages = payload["texture_pages"]
    if not isinstance(pages, list) or not 1 <= len(pages) <= 4:
        raise _error("Live2D requires one to four texture pages")
    placement_by_part: dict[str, Mapping[str, object]] = {}
    for page in pages:
        if (
            not isinstance(page, Mapping)
            or page.get("width") != 2048
            or page.get("height") != 2048
        ):
            raise _error("Live2D texture pages must be 2048x2048")
        for placement in page.get("placements", []):
            if not isinstance(placement, Mapping):
                raise _error("texture placement is not an object")
            part_id = placement.get("part_id")
            if not isinstance(part_id, str) or part_id in placement_by_part:
                raise _error("texture placement Part is invalid or duplicated")
            placement_by_part[part_id] = placement
    if set(placement_by_part) != set(part_by_id):
        raise _error("texture placements do not cover the Part set")

    attachment_by_mesh = {
        attachment.mesh_id: attachment for attachment in bindings.artmesh_attachments
    }
    artmeshes = []
    for mesh in sorted(payload["meshes"], key=lambda value: value["component_draw_rank"]):
        if not isinstance(mesh, Mapping):
            raise _error("Rig mesh is not an object")
        mesh_id = mesh.get("mesh_id")
        part_id = mesh.get("part_id")
        component_id = mesh.get("component_id")
        draw_order = mesh.get("component_draw_rank")
        if (
            not isinstance(mesh_id, str)
            or not isinstance(part_id, str)
            or not isinstance(component_id, str)
            or not isinstance(draw_order, int)
        ):
            raise _error("Rig mesh identity is invalid")
        attachment = attachment_by_mesh.get(mesh_id)
        if attachment is None:
            raise _error("Rig mesh lacks a Live2D attachment")
        symbol = require_live2d_symbol(
            symbols,
            kind="live2d_artmesh",
            source_internal_ids=(part_id, component_id),
        )
        if (
            symbol.export_name != attachment.export_name
            or attachment.primitive_target_id == ""
        ):
            raise _error("ArtMesh attachment differs from the global symbol")
        placement = placement_by_part[part_id]
        page_index = placement.get("page_index")
        if not isinstance(page_index, int) or not 0 <= page_index < len(pages):
            raise _error("ArtMesh texture index is out of range")
        vertices = mesh.get("vertices")
        indices = mesh.get("triangles")
        if not isinstance(vertices, list) or not vertices:
            raise _error("ArtMesh has no vertices")
        if (
            not isinstance(indices, list)
            or not indices
            or len(indices) % 3
            or any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= len(vertices)
                or index > 32767
                for index in indices
            )
        ):
            raise _error("ArtMesh triangle indices violate signed-int16 topology")
        positions = []
        uvs = []
        for vertex in vertices:
            if not isinstance(vertex, Mapping):
                raise _error("ArtMesh vertex is not an object")
            positions.append(
                canvas_to_artmesh_local(
                    coordinates,
                    attachment.parent_instance_id,
                    _finite_pair(vertex.get("position"), field="vertex position"),
                )
            )
            local_uv = _finite_pair(vertex.get("uv"), field="vertex UV")
            canonical_uv = (
                float(placement["u0"])
                + local_uv[0]
                * (float(placement["u1"]) - float(placement["u0"])),
                float(placement["v_top0"])
                + local_uv[1]
                * (float(placement["v_top1"]) - float(placement["v_top0"])),
            )
            if not all(0.0 <= value <= 1.0 for value in canonical_uv):
                raise _error("ArtMesh UV lies outside its canonical page")
            uvs.append(canonical_top_left_to_moc_uv(canonical_uv))
        artmeshes.append(
            Live2DArtMeshPlanRecord(
                mesh_id=mesh_id,
                part_id=part_id,
                component_id=component_id,
                primitive_target_id=attachment.primitive_target_id,
                export_name=symbol.export_name,
                symbol_sha256=symbol.symbol_sha256,
                parent_instance_id=attachment.parent_instance_id,
                texture_index=page_index,
                draw_order=draw_order,
                setup_opacity=float(part_by_id[part_id]["setup_visibility"]),
                positions=tuple(positions),
                uvs=tuple(uvs),
                triangle_indices=tuple(indices),
            )
        )
    if [mesh.draw_order for mesh in artmeshes] != list(range(len(artmeshes))):
        raise _error("ArtMesh draw order is not gapless")
    values = {
        "schema_version": LIVE2D_ARTMESH_PLAN_VERSION,
        "rig_document_sha256": rig.document_sha256,
        "symbol_view_sha256": symbols.view_sha256,
        "binding_plan_sha256": bindings.plan_sha256,
        "coordinate_plan_sha256": coordinates.plan_sha256,
        "texture_page_count": len(pages),
        "parts": tuple(parts),
        "artmeshes": tuple(artmeshes),
        "total_vertex_count": sum(len(mesh.positions) for mesh in artmeshes),
        "total_index_count": sum(len(mesh.triangle_indices) for mesh in artmeshes),
    }
    provisional = Live2DArtMeshPlan(**values, plan_sha256="")
    return Live2DArtMeshPlan(
        **values, plan_sha256=jcs_sha256(provisional.semantic_payload())
    )


def build_live2d_artmesh_plan(
    rig: RigDocument,
    symbols: Live2DSymbolView,
    bindings: Live2DBindingPlan,
    coordinates: Live2DCoordinatePlan,
) -> Live2DArtMeshPlan:
    validate_rig_document(rig)
    validate_live2d_symbol_view(symbols)
    return _assemble(rig, symbols, bindings, coordinates)


def validate_live2d_artmesh_plan(
    plan: Live2DArtMeshPlan,
    rig: RigDocument,
    symbols: Live2DSymbolView,
    bindings: Live2DBindingPlan,
    coordinates: Live2DCoordinatePlan,
) -> Live2DArtMeshPlan:
    if not isinstance(plan, Live2DArtMeshPlan):
        raise _error("ArtMesh plan has the wrong type")
    validate_rig_document(rig)
    validate_live2d_symbol_view(symbols)
    if plan != _assemble(rig, symbols, bindings, coordinates):
        raise _error("ArtMesh plan differs from the canonical Rig projection")
    if plan.plan_sha256 != jcs_sha256(plan.semantic_payload()):
        raise _error("ArtMesh plan digest mismatch")
    return plan


__all__ = [
    "LIVE2D_ARTMESH_PLAN_VERSION",
    "Live2DArtMeshError",
    "Live2DArtMeshPlan",
    "Live2DArtMeshPlanRecord",
    "Live2DPartPlan",
    "build_live2d_artmesh_plan",
    "validate_live2d_artmesh_plan",
]
