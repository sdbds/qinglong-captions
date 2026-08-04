from __future__ import annotations

from pathlib import Path

from module.auto_rig.export.live2d.binding_plan import (
    build_live2d_binding_plan,
    validate_live2d_binding_plan,
)
from module.auto_rig.export.live2d.rigid_drivers import build_rigid_driver_registry
from module.auto_rig.export.live2d.symbols import build_live2d_symbol_view
from tests.test_auto_rig_rig_document import _build


def _plan(tmp_path: Path):
    *_, rig = _build(tmp_path)
    payload = rig.to_dict()
    symbols = build_live2d_symbol_view(payload["export_symbols"])
    registry = build_rigid_driver_registry(payload["control_specs"])
    plan = build_live2d_binding_plan(rig, symbols, registry)
    return rig, symbols, registry, plan


def test_binding_plan_uses_exact_supported_artifacts_and_parameter_targets(
    tmp_path: Path,
) -> None:
    rig, symbols, registry, plan = _plan(tmp_path)

    assert validate_live2d_binding_plan(plan, rig, symbols, registry) is plan
    assert plan.supported_preset_ids == (
        "arm_sway",
        "blink",
        "body_sway",
        "breath",
        "head_nod",
        "head_shake",
        "idle",
        "leg_sway",
        "surprised",
        "talk",
        "wink_screen_left",
        "wink_screen_right",
    )
    assert "wave.xmin" not in plan.supported_preset_ids
    assert "wave.xmax" not in plan.supported_preset_ids
    assert {record.primitive_kind for record in plan.bindings} == {
        "live2d_artmesh",
        "live2d_rotation_deformer",
    }
    assert not any("warp" in record.primitive_kind for record in plan.bindings)
    assert {parameter.export_name for parameter in plan.parameters} >= {
        "ParamAngleX",
        "ParamAngleY",
        "ParamArmSway",
        "ParamAutoIdle",
        "ParamBodyAngleX",
        "ParamBreath",
        "ParamLegSway",
        "ParamMouthOpenY",
    }

    blink_groups: dict[tuple[str, str], set[str]] = {}
    for record in plan.bindings:
        if "blink" not in record.preset_ids:
            continue
        blink_groups.setdefault((record.parameter_id, record.primitive_target_id), set()).update(record.properties)
    assert any(properties == {"deform", "opacity"} for properties in blink_groups.values())


def test_rotation_liveness_freezes_same_bone_stack_and_prunes_limb_nodes(
    tmp_path: Path,
) -> None:
    rig, _symbols, _registry, plan = _plan(tmp_path)
    instances = {record.control_id: record for record in plan.rotation_instances}

    assert instances["control/body_sway"].stack_rank == 100
    assert instances["control/body_sway"].parent_instance_id is None
    assert instances["control/idle"].stack_rank == 200
    assert instances["control/idle"].parent_instance_id == instances["control/body_sway"].instance_id
    assert instances["control/head_shake"].stack_rank == 300
    assert instances["control/head_shake"].parent_instance_id == instances["control/idle"].instance_id
    assert instances["control/head_nod"].stack_rank == 400
    assert instances["control/head_nod"].parent_instance_id == instances["control/head_shake"].instance_id

    assert {record.bone_id for record in plan.rotation_instances} == {
        "bone/torso",
        "bone/head",
    }
    assert any(bone_id.startswith("bone/forearm") for bone_id in plan.pruned_bone_ids)
    assert any(bone_id.startswith("bone/hand") for bone_id in plan.pruned_bone_ids)
    assert all(
        attachment.parent_instance_id is None
        or attachment.parent_instance_id in {instance.instance_id for instance in plan.rotation_instances}
        for attachment in plan.artmesh_attachments
    )
    assert {attachment.mesh_id for attachment in plan.artmesh_attachments} == {mesh["mesh_id"] for mesh in rig.to_dict()["meshes"]}
