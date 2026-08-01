from __future__ import annotations

from dataclasses import replace

import pytest

from module.auto_rig.jcs import jcs_sha256
from module.auto_rig.joint_observations import make_joint_observation
from module.auto_rig.joint_pipeline import (
    POSE_OBSERVATION_BATCH_VERSION,
    STAGE_A_JOINT_PLAN_VERSION,
    JointPipelineError,
    StageAJointPlan,
    build_pose_observation_batch,
    build_stage_a_joint_plan,
    validate_stage_a_joint_plan,
)
from module.auto_rig.joint_registry import JOINT_IDS
from module.auto_rig.overrides import (
    load_rig_override_source,
    validate_rig_override_source,
)
from tests.test_auto_rig_anatomy import _part
from tests.test_auto_rig_limb_geometry import _anatomy, _bent_arm_points, _torso
from tests.test_auto_rig_overrides import _payload, _target, _write_override


def test_stage_a_joint_plan_builds_all_joint_results_with_pose_disabled(
    tmp_path,
) -> None:
    target = _target(tmp_path)
    overrides = validate_rig_override_source(load_rig_override_source(tmp_path), target)
    anatomy = _anatomy(
        tmp_path,
        _torso(),
        _part(
            "handwear",
            side="xmin",
            xyxy=(50, 100, 102, 180),
            points=_bent_arm_points(),
        ),
    )

    stage_plan = build_stage_a_joint_plan(
        anatomy,
        target=target,
        overrides=overrides,
    )

    assert isinstance(stage_plan, StageAJointPlan)
    assert stage_plan.schema_version == STAGE_A_JOINT_PLAN_VERSION
    assert stage_plan.target_input_fingerprint == target.target_input_fingerprint
    assert stage_plan.pose.enabled is False
    assert stage_plan.pose.observations == ()
    assert tuple(item.joint_id for item in stage_plan.joints.resolutions) == JOINT_IDS
    assert not any(item.source == "pose" for item in stage_plan.joints.observations)


def test_stage_a_joint_plan_accepts_an_explicit_pose_provider_batch(
    tmp_path,
) -> None:
    target = _target(tmp_path)
    overrides = validate_rig_override_source(load_rig_override_source(tmp_path), target)
    anatomy = _anatomy(
        tmp_path,
        _torso(),
        _part(
            "handwear",
            xyxy=(50, 100, 102, 180),
            points=_bent_arm_points(),
        ),
    )
    wrist = make_joint_observation(
        joint_id="joint/wrist.xmin",
        source="pose",
        x=75,
        y=165,
        confidence_class="model",
        canvas_width=768,
        canvas_height=768,
        evidence_ids=("pose/keypoint/9",),
        pose_score=0.95,
        algorithm_version="synthetic-pose-v1",
    )
    pose = build_pose_observation_batch(
        anatomy,
        (wrist,),
        provider_id="synthetic",
        provider_version="synthetic-v1",
        provider_fingerprint="sha256:" + "1" * 64,
        preprocessing_fingerprint="sha256:" + "2" * 64,
    )

    stage_plan = build_stage_a_joint_plan(
        anatomy,
        target=target,
        overrides=overrides,
        pose=pose,
    )
    resolution = next(
        item for item in stage_plan.joints.resolutions if item.joint_id == "joint/wrist.xmin"
    )

    assert stage_plan.pose.schema_version == POSE_OBSERVATION_BATCH_VERSION
    assert stage_plan.pose.enabled is True
    assert resolution.status == "resolved"
    assert resolution.source == "pose"
    assert resolution.reason == "pose_validated"


def test_stage_a_joint_plan_keeps_authorized_outside_override_warning(
    tmp_path,
) -> None:
    target = _target(tmp_path)
    payload = _payload(target.target_input_fingerprint)
    payload["joints"] = {
        "joint/elbow.xmin": {"x": -5, "y": 120, "allow_outside": True},
    }
    _write_override(tmp_path, payload)
    overrides = validate_rig_override_source(load_rig_override_source(tmp_path), target)
    anatomy = _anatomy(tmp_path, _torso())

    stage_plan = build_stage_a_joint_plan(
        anatomy,
        target=target,
        overrides=overrides,
    )
    resolution = next(
        item for item in stage_plan.joints.resolutions if item.joint_id == "joint/elbow.xmin"
    )

    assert stage_plan.override_outside_joint_ids == ("joint/elbow.xmin",)
    assert resolution.status == "resolved"
    assert resolution.x == -5

    missing_warning = replace(
        stage_plan,
        override_outside_joint_ids=(),
        plan_sha256="",
    )
    missing_warning = replace(
        missing_warning,
        plan_sha256=jcs_sha256(missing_warning.semantic_payload()),
    )
    with pytest.raises(JointPipelineError, match="outside override warning set"):
        validate_stage_a_joint_plan(missing_warning)


def test_stage_a_joint_plan_rejects_override_outside_missing_anatomy_without_opt_in(
    tmp_path,
) -> None:
    target = _target(tmp_path)
    payload = _payload(target.target_input_fingerprint)
    payload["joints"] = {"joint/elbow.xmin": {"x": 300, "y": 310}}
    _write_override(tmp_path, payload)
    overrides = validate_rig_override_source(load_rig_override_source(tmp_path), target)
    anatomy = _anatomy(tmp_path, _torso())

    with pytest.raises(JointPipelineError, match="outside its anatomy evidence"):
        build_stage_a_joint_plan(
            anatomy,
            target=target,
            overrides=overrides,
        )


def test_stage_a_joint_plan_rejects_pose_observation_outside_anatomy_evidence(
    tmp_path,
) -> None:
    target = _target(tmp_path)
    overrides = validate_rig_override_source(load_rig_override_source(tmp_path), target)
    anatomy = _anatomy(
        tmp_path,
        _torso(),
        _part(
            "handwear",
            xyxy=(50, 100, 102, 180),
            points=_bent_arm_points(),
        ),
    )
    pose = build_pose_observation_batch(
        anatomy,
        (
            make_joint_observation(
                joint_id="joint/wrist.xmin",
                source="pose",
                x=300,
                y=310,
                confidence_class="model",
                canvas_width=768,
                canvas_height=768,
                evidence_ids=("pose/keypoint/9",),
                pose_score=0.95,
                algorithm_version="synthetic-pose-v1",
            ),
        ),
        provider_id="synthetic",
        provider_version="synthetic-v1",
        provider_fingerprint="sha256:" + "1" * 64,
        preprocessing_fingerprint="sha256:" + "2" * 64,
    )

    with pytest.raises(JointPipelineError, match="pose observation is outside"):
        build_stage_a_joint_plan(
            anatomy,
            target=target,
            overrides=overrides,
            pose=pose,
        )


def test_pose_batch_rejects_non_string_fingerprints_as_contract_errors(
    tmp_path,
) -> None:
    anatomy = _anatomy(tmp_path, _torso())

    with pytest.raises(JointPipelineError, match="provider_fingerprint"):
        build_pose_observation_batch(
            anatomy,
            (),
            provider_id="synthetic",
            provider_version="synthetic-v1",
            provider_fingerprint=None,
            preprocessing_fingerprint="sha256:" + "2" * 64,
        )


def test_stage_a_joint_plan_validator_rejects_a_tampered_nested_batch(
    tmp_path,
) -> None:
    target = _target(tmp_path)
    overrides = validate_rig_override_source(load_rig_override_source(tmp_path), target)
    anatomy = _anatomy(tmp_path, _torso())
    stage_plan = build_stage_a_joint_plan(
        anatomy,
        target=target,
        overrides=overrides,
    )
    tampered = replace(
        stage_plan,
        pose=replace(stage_plan.pose, anatomy_plan_sha256="sha256:" + "0" * 64),
    )

    assert validate_stage_a_joint_plan(stage_plan) is stage_plan
    with pytest.raises(JointPipelineError, match="pose observation batch digest"):
        validate_stage_a_joint_plan(tampered)
