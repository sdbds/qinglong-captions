from __future__ import annotations

from pathlib import Path

import pytest

from module.auto_rig.export.live2d.artmesh import (
    build_live2d_artmesh_plan,
    validate_live2d_artmesh_plan,
)
from module.auto_rig.export.live2d.uv_kernel import moc_to_canonical_top_left_uv
from tests.test_auto_rig_live2d_coordinates import _coordinate_plans


def test_artmesh_plan_preserves_draw_order_topology_texture_and_uv(
    tmp_path: Path,
) -> None:
    rig, symbols, _registry, bindings, coordinates = _coordinate_plans(tmp_path)
    plan = build_live2d_artmesh_plan(rig, symbols, bindings, coordinates)
    payload = rig.to_dict()

    assert validate_live2d_artmesh_plan(
        plan, rig, symbols, bindings, coordinates
    ) is plan
    assert len(plan.parts) == len(payload["parts"])
    assert len(plan.artmeshes) == len(payload["meshes"])
    assert [mesh.draw_order for mesh in plan.artmeshes] == list(
        range(len(plan.artmeshes))
    )
    placement_by_part = {
        placement["part_id"]: placement
        for page in payload["texture_pages"]
        for placement in page["placements"]
    }
    source_mesh = {mesh["mesh_id"]: mesh for mesh in payload["meshes"]}
    for artmesh in plan.artmeshes:
        source = source_mesh[artmesh.mesh_id]
        placement = placement_by_part[artmesh.part_id]
        assert artmesh.texture_index == placement["page_index"]
        assert artmesh.triangle_indices == tuple(source["triangles"])
        assert max(artmesh.triangle_indices) < len(artmesh.positions)
        first_source_uv = source["vertices"][0]["uv"]
        expected = (
            placement["u0"]
            + first_source_uv[0] * (placement["u1"] - placement["u0"]),
            placement["v_top0"]
            + first_source_uv[1]
            * (placement["v_top1"] - placement["v_top0"]),
        )
        assert moc_to_canonical_top_left_uv(artmesh.uvs[0]) == pytest.approx(
            expected
        )


def test_artmesh_plan_uses_global_symbols_for_part_and_component_ids(
    tmp_path: Path,
) -> None:
    rig, symbols, _registry, bindings, coordinates = _coordinate_plans(tmp_path)
    plan = build_live2d_artmesh_plan(rig, symbols, bindings, coordinates)

    assert all(part.export_name.isascii() for part in plan.parts)
    assert all(mesh.export_name.isascii() for mesh in plan.artmeshes)
    assert len({part.export_name for part in plan.parts}) == len(plan.parts)
    assert len({mesh.export_name for mesh in plan.artmeshes}) == len(plan.artmeshes)

