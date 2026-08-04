from __future__ import annotations

from pathlib import Path

import pytest

from module.auto_rig.export.live2d.artmesh import build_live2d_artmesh_plan
from module.auto_rig.export.live2d.keyforms import (
    build_live2d_keyform_plan,
    evaluate_artmesh_keyform,
    validate_live2d_keyform_plan,
)
from tests.test_auto_rig_live2d_coordinates import _coordinate_plans


def _flatten_points(points: tuple[tuple[float, float], ...]) -> tuple[float, ...]:
    return tuple(coordinate for point in points for coordinate in point)


def test_keyform_plan_merges_deform_opacity_and_preserves_default_rest(
    tmp_path: Path,
) -> None:
    rig, symbols, _registry, bindings, coordinates = _coordinate_plans(tmp_path)
    artmeshes = build_live2d_artmesh_plan(rig, symbols, bindings, coordinates)
    plan = build_live2d_keyform_plan(rig, bindings, coordinates, artmeshes)

    assert validate_live2d_keyform_plan(plan, rig, bindings, coordinates, artmeshes) is plan
    assert plan.total_baked_vertex_positions > 0
    assert plan.total_baked_vertex_positions <= 1_000_000
    by_parameter = {}
    for record in plan.artmesh_keyforms:
        if record.parameter_id is not None:
            by_parameter.setdefault(record.parameter_id, []).append(record)
        setup = next(mesh for mesh in artmeshes.artmeshes if mesh.mesh_id == record.mesh_id)
        positions, opacity = evaluate_artmesh_keyform(record, record.parameter_default)
        assert _flatten_points(positions) == pytest.approx(_flatten_points(setup.positions), abs=1e-9)
        assert opacity == pytest.approx(setup.setup_opacity, abs=1 / 255)

    assert {record.parameter_values for record in by_parameter["parameter/brow_y.xmin"]} == {(-1.0, 0.0, 1.0)}
    mouth = by_parameter["parameter/mouth_open_y"]
    assert len(mouth) == 1
    assert mouth[0].parameter_values == (0.0, 1.0)
    blink = by_parameter["parameter/eye_open.xmax"]
    assert any(record.has_opacity for record in blink)
    assert all(len(record.parameter_values) <= 17 for record in plan.artmesh_keyforms)


def test_keyform_interpolation_is_exact_at_all_stored_stops(tmp_path: Path) -> None:
    rig, symbols, _registry, bindings, coordinates = _coordinate_plans(tmp_path)
    artmeshes = build_live2d_artmesh_plan(rig, symbols, bindings, coordinates)
    plan = build_live2d_keyform_plan(rig, bindings, coordinates, artmeshes)

    for record in plan.artmesh_keyforms:
        for index, value in enumerate(record.parameter_values):
            positions, opacity = evaluate_artmesh_keyform(record, value)
            assert _flatten_points(positions) == pytest.approx(_flatten_points(record.positions[index]), abs=1e-9)
            assert opacity == pytest.approx(record.opacities[index], abs=1e-9)


def test_head_depth_parallax_projects_to_live2d_hair_keyforms(
    tmp_path: Path,
) -> None:
    rig, symbols, _registry, bindings, coordinates = _coordinate_plans(
        tmp_path,
        hair=True,
    )
    artmeshes = build_live2d_artmesh_plan(rig, symbols, bindings, coordinates)
    plan = build_live2d_keyform_plan(rig, bindings, coordinates, artmeshes)
    part_by_id = {part["part_id"]: part for part in rig.to_dict()["parts"]}
    mesh_by_id = {mesh.mesh_id: mesh for mesh in artmeshes.artmeshes}
    shifts: dict[str, list[float]] = {}

    for record in plan.artmesh_keyforms:
        if record.parameter_id != "parameter/angle_x":
            continue
        mesh = mesh_by_id[record.mesh_id]
        tag = part_by_id[mesh.part_id]["base_tag"]
        if tag not in {"back hair", "front hair"}:
            continue
        default_positions, _opacity = evaluate_artmesh_keyform(record, record.parameter_default)
        maximum_positions, _opacity = evaluate_artmesh_keyform(record, max(record.parameter_values))
        shifts.setdefault(tag, []).append(
            sum(after[0] - before[0] for before, after in zip(default_positions, maximum_positions, strict=True))
            / len(default_positions)
        )

    assert set(shifts) == {"back hair", "front hair"}
    assert min(shifts["front hair"]) > 0.0
    assert max(shifts["back hair"]) < 0.0
