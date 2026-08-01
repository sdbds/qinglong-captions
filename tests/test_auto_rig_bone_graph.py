from __future__ import annotations

from dataclasses import replace

import pytest

from module.auto_rig.bone_graph import (
    BONE_GRAPH_PLAN_VERSION,
    BONE_IDS,
    BoneGraphError,
    BoneGraphPlan,
    build_bone_graph,
    validate_bone_graph,
)
from module.auto_rig.jcs import jcs_sha256
from module.auto_rig.joint_pipeline import build_stage_a_joint_plan
from module.auto_rig.overrides import (
    load_rig_override_source,
    validate_rig_override_source,
)
from tests.test_auto_rig_anatomy import _part, _rectangle
from tests.test_auto_rig_limb_geometry import _anatomy
from tests.test_auto_rig_overrides import _payload, _target, _write_override

_JOINT_COORDINATES = {
    "joint/pelvis": (100, 200),
    "joint/spine": (100, 160),
    "joint/neck": (100, 120),
    "joint/head_base": (100, 100),
    "joint/head_top": (100, 60),
    "joint/shoulder.xmin": (70, 125),
    "joint/elbow.xmin": (50, 160),
    "joint/wrist.xmin": (40, 195),
    "joint/hand_tip.xmin": (35, 220),
    "joint/shoulder.xmax": (130, 125),
    "joint/elbow.xmax": (150, 160),
    "joint/wrist.xmax": (160, 195),
    "joint/hand_tip.xmax": (165, 220),
    "joint/hip.xmin": (80, 200),
    "joint/knee.xmin": (75, 260),
    "joint/ankle.xmin": (70, 320),
    "joint/toe.xmin": (55, 335),
    "joint/hip.xmax": (120, 200),
    "joint/knee.xmax": (125, 260),
    "joint/ankle.xmax": (130, 320),
    "joint/toe.xmax": (145, 335),
}


def _joint_stage(
    tmp_path,
    *,
    omitted: tuple[str, ...] = (),
    coordinates: dict[str, tuple[float, float]] | None = None,
):
    joint_coordinates = _JOINT_COORDINATES if coordinates is None else coordinates
    target = _target(tmp_path)
    payload = _payload(target.target_input_fingerprint)
    payload["joints"] = {
        joint_id: {"x": point[0], "y": point[1], "allow_outside": True}
        for joint_id, point in joint_coordinates.items()
        if joint_id not in omitted
    }
    payload["tag_aliases"] = {}
    _write_override(tmp_path, payload)
    overrides = validate_rig_override_source(load_rig_override_source(tmp_path), target)
    anatomy = _anatomy(
        tmp_path,
        _part(
            "face",
            xyxy=(90, 50, 110, 90),
            points=_rectangle(20, 40),
        ),
    )
    return build_stage_a_joint_plan(
        anatomy,
        target=target,
        overrides=overrides,
    )


def test_bone_graph_emits_the_complete_declarative_topology(tmp_path) -> None:
    plan = build_bone_graph(_joint_stage(tmp_path))

    assert isinstance(plan, BoneGraphPlan)
    assert plan.schema_version == BONE_GRAPH_PLAN_VERSION
    assert tuple(bone.bone_id for bone in plan.bones) == BONE_IDS
    assert plan.bones[0].bone_id == "bone/root"
    assert plan.bones[0].length == 0
    assert all(bone.length > 0 for bone in plan.bones[1:])
    assert next(
        bone for bone in plan.bones if bone.bone_id == "bone/forearm.xmin"
    ).parent_id == "bone/upper_arm.xmin"


def test_bone_graph_omits_bones_whose_own_joints_are_unresolved(tmp_path) -> None:
    plan = build_bone_graph(_joint_stage(tmp_path, omitted=("joint/wrist.xmin",)))
    bone_ids = {bone.bone_id for bone in plan.bones}

    assert "bone/upper_arm.xmin" in bone_ids
    assert "bone/forearm.xmin" not in bone_ids
    assert "bone/hand.xmin" not in bone_ids
    assert {
        item.bone_id
        for item in plan.diagnostics
        if item.code == "bone_joint_unresolved"
    } >= {"bone/forearm.xmin", "bone/hand.xmin"}


def test_bone_graph_promotes_a_child_to_the_closest_emitted_ancestor(
    tmp_path,
) -> None:
    plan = build_bone_graph(_joint_stage(tmp_path, omitted=("joint/neck",)))
    head = next(bone for bone in plan.bones if bone.bone_id == "bone/head")
    bone_ids = {bone.bone_id for bone in plan.bones}

    assert "bone/torso" not in bone_ids
    assert "bone/neck" not in bone_ids
    assert head.parent_id == "bone/lower_torso"
    assert head.parent_promotion is True


@pytest.mark.parametrize(
    "coordinates",
    (
        {**_JOINT_COORDINATES, "joint/spine": _JOINT_COORDINATES["joint/pelvis"]},
        {**_JOINT_COORDINATES, "joint/spine": (5000, 5000)},
    ),
)
def test_bone_graph_rejects_invalid_lengths_from_authoritative_overrides(
    tmp_path,
    coordinates: dict[str, tuple[float, float]],
) -> None:
    with pytest.raises(BoneGraphError, match="invalid_bone_length"):
        build_bone_graph(_joint_stage(tmp_path, coordinates=coordinates))


def test_bone_graph_validator_rejects_recomputed_digest_with_changed_limits(
    tmp_path,
) -> None:
    plan = build_bone_graph(_joint_stage(tmp_path))
    tampered = replace(plan, maximum_canvas_diagonals=100.0, plan_sha256="")
    tampered = replace(tampered, plan_sha256=jcs_sha256(tampered.semantic_payload()))

    with pytest.raises(BoneGraphError, match="length limits"):
        validate_bone_graph(tampered)
