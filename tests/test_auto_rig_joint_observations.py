from __future__ import annotations

from pathlib import Path

import pytest

from module.auto_rig.joint_observations import (
    JOINT_OBSERVATION_PLAN_VERSION,
    GeometryConfidenceFactors,
    JointEligibility,
    JointObservationContractError,
    build_override_joint_observations,
    make_joint_observation,
    resolve_joint_observations,
)
from module.auto_rig.joint_registry import JOINT_IDS, JOINT_REGISTRY_VERSION
from module.auto_rig.overrides import (
    RIG_OVERRIDES_PATH,
    load_rig_override_source,
    validate_rig_override_source,
)
from tests.test_auto_rig_overrides import _payload, _target, _write_override


def _eligibilities(
    *eligible: str,
    ambiguous: tuple[str, ...] = (),
    radius: float = 10.0,
) -> tuple[JointEligibility, ...]:
    records = []
    for joint_id in JOINT_IDS:
        if joint_id in eligible:
            records.append(
                JointEligibility(
                    joint_id=joint_id,
                    status="eligible",
                    reason="anatomy_available",
                    evidence_ids=("mask/limb/test.xmin",),
                    local_limb_radius=radius,
                )
            )
        elif joint_id in ambiguous:
            records.append(
                JointEligibility(
                    joint_id=joint_id,
                    status="ambiguous",
                    reason="merged_limb",
                    evidence_ids=("mask/limb/test.merged",),
                    local_limb_radius=None,
                )
            )
        else:
            records.append(
                JointEligibility(
                    joint_id=joint_id,
                    status="missing",
                    reason="part_missing",
                    evidence_ids=(),
                    local_limb_radius=None,
                )
            )
    return tuple(records)


def _factors() -> GeometryConfidenceFactors:
    return GeometryConfidenceFactors(
        connectivity=1.0,
        main_path_length=0.9,
        branch_ratio=0.8,
        endpoint_contact=0.7,
        curvature_peak=0.6,
        mask_interior_distance=0.9,
        bilateral_consistency=0.8,
    )


def _observation(
    joint_id: str,
    source: str,
    x: float,
    y: float,
    *,
    confidence_class: str,
):
    return make_joint_observation(
        joint_id=joint_id,
        source=source,
        x=x,
        y=y,
        confidence_class=confidence_class,
        canvas_width=768,
        canvas_height=768,
        evidence_ids=("component/c_test", "mask/limb/test.xmin"),
        geometry_factors=_factors() if source == "geometry" else None,
        pose_score=0.95 if source == "pose" else None,
        algorithm_version=f"{source}-test-v1",
    )


def test_joint_registry_is_complete_and_reused_by_override_parser(tmp_path: Path) -> None:
    assert JOINT_REGISTRY_VERSION == "joint-registry-v1"
    assert JOINT_IDS[0:5] == (
        "joint/pelvis",
        "joint/spine",
        "joint/neck",
        "joint/head_base",
        "joint/head_top",
    )
    target = _target(tmp_path)
    payload = _payload(target.target_input_fingerprint)
    payload["joints"] = {JOINT_IDS[-1]: {"x": 1, "y": 2}}
    _write_override(tmp_path, payload)

    source = load_rig_override_source(tmp_path)

    assert source.joints[0].joint_id == JOINT_IDS[-1]


def test_joint_observation_identity_is_content_addressed_and_order_independent() -> None:
    forward = make_joint_observation(
        joint_id="joint/elbow.xmin",
        source="geometry",
        x=120,
        y=240,
        confidence_class="high",
        canvas_width=768,
        canvas_height=768,
        evidence_ids=("mask/z", "component/a"),
        geometry_factors=_factors(),
        pose_score=None,
        algorithm_version="limb-geometry-v1",
    )
    reversed_evidence = make_joint_observation(
        joint_id="joint/elbow.xmin",
        source="geometry",
        x=120,
        y=240,
        confidence_class="high",
        canvas_width=768,
        canvas_height=768,
        evidence_ids=("component/a", "mask/z"),
        geometry_factors=_factors(),
        pose_score=None,
        algorithm_version="limb-geometry-v1",
    )

    assert forward == reversed_evidence
    assert forward.observation_id.startswith("observation/o_")
    assert len(forward.observation_id.removeprefix("observation/o_")) == 64


@pytest.mark.parametrize(
    "kwargs",
    (
        {
            "source": "pose",
            "confidence_class": "model",
            "pose_score": None,
            "geometry_factors": None,
        },
        {
            "source": "geometry",
            "confidence_class": "high",
            "pose_score": None,
            "geometry_factors": None,
        },
        {
            "source": "override",
            "confidence_class": "authoritative",
            "pose_score": None,
            "geometry_factors": None,
            "x": 800,
        },
    ),
)
def test_joint_observation_rejects_cross_scale_or_unapproved_outside_values(
    kwargs: dict[str, object],
) -> None:
    values = {
        "joint_id": "joint/elbow.xmin",
        "x": 10,
        "y": 20,
        "canvas_width": 768,
        "canvas_height": 768,
        "evidence_ids": ("mask/test",),
        "algorithm_version": "test-v1",
        **kwargs,
    }

    with pytest.raises(JointObservationContractError):
        make_joint_observation(**values)


def test_override_observation_is_authoritative_over_geometry_and_pose(tmp_path: Path) -> None:
    target = _target(tmp_path)
    payload = _payload(target.target_input_fingerprint)
    payload["joints"] = {"joint/elbow.xmin": {"x": 300, "y": 310}}
    _write_override(tmp_path, payload)
    overrides = validate_rig_override_source(load_rig_override_source(tmp_path), target)
    override = build_override_joint_observations(
        overrides,
        canvas_width=768,
        canvas_height=768,
    )[0]
    geometry = _observation(
        "joint/elbow.xmin", "geometry", 100, 100, confidence_class="high"
    )
    pose = _observation("joint/elbow.xmin", "pose", 110, 110, confidence_class="model")

    plan = resolve_joint_observations(
        _eligibilities("joint/elbow.xmin"),
        (pose, override, geometry),
        canvas_width=768,
        canvas_height=768,
    )
    resolved = next(item for item in plan.resolutions if item.joint_id == "joint/elbow.xmin")

    assert resolved.status == "resolved"
    assert resolved.source == "override"
    assert (resolved.x, resolved.y) == (300.0, 310.0)


def test_override_observation_precedes_missing_geometry_eligibility(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    payload = _payload(target.target_input_fingerprint)
    payload["joints"] = {"joint/elbow.xmin": {"x": 300, "y": 310}}
    _write_override(tmp_path, payload)
    overrides = validate_rig_override_source(load_rig_override_source(tmp_path), target)
    override = build_override_joint_observations(
        overrides,
        canvas_width=768,
        canvas_height=768,
    )[0]

    plan = resolve_joint_observations(
        _eligibilities(),
        (override,),
        canvas_width=768,
        canvas_height=768,
    )
    resolved = next(item for item in plan.resolutions if item.joint_id == "joint/elbow.xmin")

    assert resolved.status == "resolved"
    assert resolved.source == "override"
    assert resolved.reason == "override_applied"


def test_high_geometry_wins_without_mixing_pose_score_into_geometry_factors() -> None:
    geometry = _observation(
        "joint/knee.xmin", "geometry", 100, 100, confidence_class="high"
    )
    far_pose = _observation("joint/knee.xmin", "pose", 300, 300, confidence_class="model")

    plan = resolve_joint_observations(
        _eligibilities("joint/knee.xmin"),
        (far_pose, geometry),
        canvas_width=768,
        canvas_height=768,
    )
    resolved = next(item for item in plan.resolutions if item.joint_id == "joint/knee.xmin")

    assert resolved.source == "geometry"
    assert resolved.quality == "high"
    assert geometry.pose_score is None
    assert far_pose.geometry_factors is None


@pytest.mark.parametrize(
    ("pose_xy", "expected_status", "expected_source", "expected_reason"),
    (
        ((112, 112), "resolved", "pose", "pose_refines_low_geometry"),
        ((140, 140), "unresolved", None, "pose_disagreement"),
    ),
)
def test_low_geometry_uses_pose_only_within_local_limb_width(
    pose_xy: tuple[int, int],
    expected_status: str,
    expected_source: str | None,
    expected_reason: str,
) -> None:
    geometry = _observation(
        "joint/elbow.xmax", "geometry", 100, 100, confidence_class="low"
    )
    pose = _observation(
        "joint/elbow.xmax", "pose", *pose_xy, confidence_class="model"
    )

    plan = resolve_joint_observations(
        _eligibilities("joint/elbow.xmax", radius=10),
        (geometry, pose),
        canvas_width=768,
        canvas_height=768,
    )
    resolved = next(item for item in plan.resolutions if item.joint_id == "joint/elbow.xmax")

    assert resolved.status == expected_status
    assert resolved.source == expected_source
    assert resolved.reason == expected_reason


def test_resolver_keeps_weak_prior_missing_and_ambiguous_results_explicit() -> None:
    prior = _observation(
        "joint/wrist.xmin",
        "length_prior",
        200,
        220,
        confidence_class="weak",
    )
    plan = resolve_joint_observations(
        _eligibilities(
            "joint/wrist.xmin",
            ambiguous=("joint/wrist.xmax",),
        ),
        (prior,),
        canvas_width=768,
        canvas_height=768,
    )
    by_id = {item.joint_id: item for item in plan.resolutions}

    assert plan.schema_version == JOINT_OBSERVATION_PLAN_VERSION
    assert by_id["joint/wrist.xmin"].status == "resolved"
    assert by_id["joint/wrist.xmin"].quality == "weak"
    assert by_id["joint/wrist.xmax"].status == "unresolved"
    assert by_id["joint/wrist.xmax"].reason == "merged_limb"
    assert by_id["joint/wrist.xmax"].x is None
    assert by_id["joint/ankle.xmin"].status == "missing"
    assert by_id["joint/ankle.xmin"].x is None


def test_joint_resolution_plan_is_independent_of_observation_input_order() -> None:
    geometry = _observation(
        "joint/elbow.xmin", "geometry", 100, 100, confidence_class="low"
    )
    pose = _observation("joint/elbow.xmin", "pose", 110, 110, confidence_class="model")
    eligibility = _eligibilities("joint/elbow.xmin", radius=10)

    forward = resolve_joint_observations(
        eligibility,
        (geometry, pose),
        canvas_width=768,
        canvas_height=768,
    )
    reversed_input = resolve_joint_observations(
        tuple(reversed(eligibility)),
        (pose, geometry),
        canvas_width=768,
        canvas_height=768,
    )

    assert forward == reversed_input
