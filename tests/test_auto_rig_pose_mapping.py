from __future__ import annotations

from module.auto_rig.joint_pipeline import build_stage_a_joint_plan
from module.auto_rig.overrides import load_rig_override_source, validate_rig_override_source
from module.auto_rig.pose.contracts import RawPoseResult, ScoredKeypoint
from module.auto_rig.pose.mapping import map_raw_pose_result
from module.auto_rig.pose.selection import decide_pose_trigger
from tests.test_auto_rig_anatomy import _part
from tests.test_auto_rig_limb_geometry import _anatomy, _bent_arm_points, _torso
from tests.test_auto_rig_overrides import _target


def _points(count: int) -> list[ScoredKeypoint]:
    return [ScoredKeypoint(x=300 + index, y=200 + index, score=0.9) for index in range(count)]


def test_coco17_mapping_uses_character_right_as_canvas_xmin_and_derives_centers() -> None:
    points = _points(17)
    points[5] = ScoredKeypoint(500, 200, 0.8)  # left shoulder -> xmax
    points[6] = ScoredKeypoint(200, 210, 0.9)  # right shoulder -> xmin
    points[7] = ScoredKeypoint(520, 300, 0.8)
    points[8] = ScoredKeypoint(180, 310, 0.9)
    points[9] = ScoredKeypoint(540, 400, 0.8)
    points[10] = ScoredKeypoint(160, 410, 0.9)
    points[11] = ScoredKeypoint(470, 450, 0.85)
    points[12] = ScoredKeypoint(230, 455, 0.95)

    observations = map_raw_pose_result(
        RawPoseResult(
            provider_id="sdpose-body17",
            provider_version="sdpose-body17-bundle-v1",
            layout="coco17",
            canvas_width=768,
            canvas_height=768,
            keypoints=tuple(points),
        )
    )
    by_joint = {item.joint_id: item for item in observations}

    assert by_joint["joint/shoulder.xmin"].x == 200
    assert by_joint["joint/shoulder.xmax"].x == 500
    assert by_joint["joint/elbow.xmin"].x == 180
    assert by_joint["joint/wrist.xmax"].x == 540
    assert (by_joint["joint/neck"].x, by_joint["joint/neck"].y) == (350, 205)
    assert (by_joint["joint/pelvis"].x, by_joint["joint/pelvis"].y) == (350, 452.5)
    assert by_joint["joint/neck"].pose_score == 0.8


def test_crowdpose14_mapping_uses_its_native_order_and_does_not_treat_top_head_as_nose() -> None:
    points = _points(14)
    points[0] = ScoredKeypoint(500, 200, 0.9)  # left shoulder
    points[1] = ScoredKeypoint(200, 200, 0.9)  # right shoulder
    points[12] = ScoredKeypoint(360, 80, 0.99)  # top head, intentionally ignored
    points[13] = ScoredKeypoint(350, 170, 0.95)  # neck

    observations = map_raw_pose_result(
        RawPoseResult(
            provider_id="detrpose-x-crowdpose",
            provider_version="detrpose-inference-only-v1",
            layout="crowdpose14",
            canvas_width=768,
            canvas_height=768,
            keypoints=tuple(points),
        )
    )
    by_joint = {item.joint_id: item for item in observations}

    assert by_joint["joint/shoulder.xmin"].x == 200
    assert by_joint["joint/shoulder.xmax"].x == 500
    assert (by_joint["joint/neck"].x, by_joint["joint/neck"].y) == (350, 170)
    assert "joint/head_top" not in by_joint


def test_low_score_points_and_out_of_canvas_points_are_omitted_not_clamped() -> None:
    points = _points(14)
    points[0] = ScoredKeypoint(500, 200, 0.1)
    points[1] = ScoredKeypoint(-1, 200, 0.99)
    result = RawPoseResult(
        provider_id="detrpose-x-crowdpose",
        provider_version="detrpose-inference-only-v1",
        layout="crowdpose14",
        canvas_width=768,
        canvas_height=768,
        keypoints=tuple(points),
    )

    by_joint = {item.joint_id: item for item in map_raw_pose_result(result, minimum_score=0.3)}

    assert "joint/shoulder.xmin" not in by_joint
    assert "joint/shoulder.xmax" not in by_joint


def test_auto_pose_trigger_requires_merged_limb_and_unresolved_hinge(tmp_path) -> None:
    target = _target(tmp_path)
    overrides = validate_rig_override_source(load_rig_override_source(tmp_path), target)
    merged = _anatomy(
        tmp_path,
        _torso(),
        _part(
            "handwear",
            xyxy=(50, 100, 102, 180),
            points=_bent_arm_points(),
        ),
    )
    geometry = build_stage_a_joint_plan(merged, target=target, overrides=overrides)

    decision = decide_pose_trigger("auto", merged, geometry)

    assert decision.trigger is True
    assert decision.reason == "merged_limb_unresolved_hinge"
    assert "joint/elbow.xmin" in decision.unresolved_joint_ids
    assert decide_pose_trigger("disabled", merged, geometry).trigger is False
    assert decide_pose_trigger("sdpose", merged, geometry).trigger is True


def test_auto_pose_trigger_does_not_run_for_sided_limb_geometry(tmp_path) -> None:
    target = _target(tmp_path)
    overrides = validate_rig_override_source(load_rig_override_source(tmp_path), target)
    sided = _anatomy(
        tmp_path,
        _torso(),
        _part(
            "handwear",
            side="xmin",
            xyxy=(50, 100, 102, 180),
            points=_bent_arm_points(),
        ),
    )
    geometry = build_stage_a_joint_plan(sided, target=target, overrides=overrides)

    assert decide_pose_trigger("auto", sided, geometry).trigger is False
