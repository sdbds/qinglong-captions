from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from module.auto_rig.export.live2d.rigid_drivers import (
    RigidDriverRegistryError,
    build_rigid_driver_registry,
    validate_rigid_driver_registry,
)
from module.auto_rig.export.live2d.symbols import (
    Live2DSymbolError,
    build_live2d_symbol_view,
    require_live2d_symbol,
    validate_live2d_symbol_view,
)
from module.auto_rig.jcs import jcs_sha256
from tests.test_auto_rig_rig_document import _build


def test_live2d_symbol_view_resolves_all_typed_names_without_sanitize(
    tmp_path: Path,
) -> None:
    *_, rig = _build(tmp_path)
    payload = rig.to_dict()
    view = build_live2d_symbol_view(payload["export_symbols"])
    first_mesh = payload["meshes"][0]

    part = require_live2d_symbol(
        view,
        kind="live2d_part",
        source_internal_ids=(first_mesh["part_id"],),
    )
    artmesh = require_live2d_symbol(
        view,
        kind="live2d_artmesh",
        source_internal_ids=(first_mesh["part_id"], first_mesh["component_id"]),
    )
    parameter = require_live2d_symbol(
        view,
        kind="live2d_parameter",
        parameter_id="parameter/angle_x",
    )
    deformer = require_live2d_symbol(
        view,
        kind="live2d_rotation_deformer",
        source_internal_ids=("bone/head",),
        parameter_id="parameter/angle_x",
    )
    motion = require_live2d_symbol(
        view,
        kind="live2d_motion",
        preset_id="clip/head_shake",
    )

    assert part.export_name.isascii()
    assert artmesh.component_id == first_mesh["component_id"]
    assert parameter.export_name == "ParamAngleX"
    assert deformer.export_name == "head__rot_angle_x"
    assert motion.export_name == "head_shake"
    assert validate_live2d_symbol_view(view) is view
    assert all(record.format_id == "live2d_moc3_v4_00" for record in view.symbols)


def test_live2d_symbols_and_rigid_registry_reject_renames_and_rank_collisions(
    tmp_path: Path,
) -> None:
    *_, rig = _build(tmp_path)
    payload = rig.to_dict()
    table = payload["export_symbols"]
    symbol = next(
        row
        for row in table["symbols"]
        if row["typed_primitive_key"]["format_id"] == "live2d_moc3_v4_00"
    )
    symbol["export_name"] = "writer_sanitized"
    with pytest.raises(Live2DSymbolError, match="digest"):
        build_live2d_symbol_view(table)

    registry = build_rigid_driver_registry(payload["control_specs"])
    rows = list(registry.rows)
    rows[1] = replace(rows[1], stack_rank=rows[0].stack_rank, row_sha256="")
    rows[1] = replace(rows[1], row_sha256=jcs_sha256(rows[1].semantic_payload()))
    provisional = replace(registry, rows=tuple(rows), registry_sha256="")
    collided = replace(
        provisional,
        registry_sha256=jcs_sha256(provisional.semantic_payload()),
    )
    with pytest.raises(RigidDriverRegistryError, match="rank"):
        validate_rigid_driver_registry(collided, payload["control_specs"])


def test_rigid_driver_registry_freezes_globally_unique_production_order(
    tmp_path: Path,
) -> None:
    *_, rig = _build(tmp_path)
    registry = build_rigid_driver_registry(rig.to_dict()["control_specs"])

    assert [(row.control_id, row.parameter_id, row.stack_rank) for row in registry.rows] == [
        ("control/body_sway", "parameter/body_angle_x", 100),
        ("control/idle", "parameter/auto_idle", 200),
        ("control/head_shake", "parameter/angle_x", 300),
        ("control/head_nod", "parameter/angle_y", 400),
    ]
    assert validate_rigid_driver_registry(
        registry, rig.to_dict()["control_specs"]
    ) is registry
