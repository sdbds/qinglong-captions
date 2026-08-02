from __future__ import annotations

from pathlib import Path

import pytest

from module.auto_rig.export.live2d.binding_plan import build_live2d_binding_plan
from module.auto_rig.export.live2d.coordinates import (
    build_live2d_coordinate_plan,
    canvas_to_artmesh_local,
    live2d_local_to_canvas,
    validate_live2d_coordinate_plan,
)
from module.auto_rig.export.live2d.rigid_drivers import build_rigid_driver_registry
from module.auto_rig.export.live2d.symbols import build_live2d_symbol_view
from tests.test_auto_rig_rig_document import _build


def _coordinate_plans(tmp_path: Path):
    *_, rig = _build(tmp_path)
    payload = rig.to_dict()
    symbols = build_live2d_symbol_view(payload["export_symbols"])
    registry = build_rigid_driver_registry(payload["control_specs"])
    bindings = build_live2d_binding_plan(rig, symbols, registry)
    coordinates = build_live2d_coordinate_plan(rig, bindings)
    return rig, symbols, registry, bindings, coordinates


def test_coordinate_plan_reconstructs_every_setup_vertex_through_live_stack(
    tmp_path: Path,
) -> None:
    rig, _symbols, _registry, bindings, plan = _coordinate_plans(tmp_path)

    assert validate_live2d_coordinate_plan(plan, rig, bindings) is plan
    assert plan.ppu == 768.0
    assert plan.maximum_round_trip_error <= 0.1
    frame_by_id = {frame.instance_id: frame for frame in plan.rotation_frames}
    instances = {record.control_id: record for record in bindings.rotation_instances}
    assert frame_by_id[instances["control/body_sway"].instance_id].scale == pytest.approx(
        1.0 / plan.ppu
    )
    assert frame_by_id[instances["control/idle"].instance_id].origin == (0.0, 0.0)
    assert frame_by_id[instances["control/head_nod"].instance_id].origin == (
        0.0,
        0.0,
    )

    attachment_by_mesh = {
        attachment.mesh_id: attachment for attachment in bindings.artmesh_attachments
    }
    for mesh in rig.to_dict()["meshes"]:
        attachment = attachment_by_mesh[mesh["mesh_id"]]
        for vertex in mesh["vertices"]:
            point = tuple(vertex["position"])
            local = canvas_to_artmesh_local(plan, attachment.parent_instance_id, point)
            recovered = live2d_local_to_canvas(
                plan, attachment.parent_instance_id, local
            )
            assert recovered == pytest.approx(point, abs=0.1)


def test_coordinate_helpers_reject_unknown_parent_instances(tmp_path: Path) -> None:
    _rig, _symbols, _registry, _bindings, plan = _coordinate_plans(tmp_path)

    with pytest.raises(ValueError, match="unknown"):
        canvas_to_artmesh_local(plan, "primitive/not-present", (100.0, 100.0))

