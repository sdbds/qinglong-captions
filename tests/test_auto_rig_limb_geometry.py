from __future__ import annotations

from pathlib import Path

from module.auto_rig.anatomy import build_anatomy_mask_geometry
from module.auto_rig.axial_geometry import build_axial_joint_evidence
from module.auto_rig.component_geometry import load_component_geometry
from module.auto_rig.component_plan import build_mask_component_plan
from module.auto_rig.limb_geometry import (
    LIMB_JOINT_GEOMETRY_VERSION,
    LIMB_JOINT_IDS,
    LimbGeometryEvidenceBatch,
    build_limb_joint_evidence,
)
from tests.test_auto_rig_anatomy import _part, _rectangle


def _anatomy(tmp_path: Path, *parts):
    plan = build_mask_component_plan(parts, canvas_edge=768, item_root=tmp_path)
    return build_anatomy_mask_geometry(
        load_component_geometry(plan, item_root=tmp_path),
        canvas_edge=768,
    )


def _torso():
    return _part(
        "topwear",
        xyxy=(100, 80, 150, 160),
        points=_rectangle(50, 80),
    )


def _bent_arm_points() -> set[tuple[int, int]]:
    horizontal = {(x, y) for y in range(10) for x in range(20, 52)}
    vertical = {(x, y) for y in range(80) for x in range(20, 30)}
    return horizontal | vertical


def _wrist_arm_points() -> set[tuple[int, int]]:
    forearm = {(x, y) for y in range(68) for x in range(20, 30)}
    wrist = {(x, y) for y in range(65, 78) for x in range(23, 27)}
    palm = {(x, y) for y in range(75, 100) for x in range(15, 35)}
    return forearm | wrist | palm


def _branched_arm_points() -> set[tuple[int, int]]:
    stem = {(x, y) for y in range(62) for x in range(22, 30)}
    branches: set[tuple[int, int]] = set()
    for offset in range(34):
        y = 60 + offset
        left_x = 25 - offset // 2
        right_x = 26 + offset // 2
        branches.update(
            (x, y)
            for center in (left_x, right_x)
            for x in range(center - 3, center + 4)
        )
    return stem | branches


def _legwear_points() -> set[tuple[int, int]]:
    return (
        {(x, y) for y in range(82) for x in range(10)}
        | {(x, y) for y in range(82) for x in range(25, 35)}
    )


def _footwear_points() -> set[tuple[int, int]]:
    left = {(x, y) for y in range(12) for x in range(25)}
    right = {(x, y) for y in range(12) for x in range(40, 65)}
    return left | right


def test_limb_geometry_finds_bent_elbow_and_tip_but_not_a_fake_wrist(
    tmp_path: Path,
) -> None:
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
    axial = build_axial_joint_evidence(anatomy)

    batch = build_limb_joint_evidence(anatomy, axial)
    observations = {item.joint_id: item for item in batch.observations}
    eligibility = {item.joint_id: item for item in batch.eligibilities}

    assert isinstance(batch, LimbGeometryEvidenceBatch)
    assert batch.provider_version == LIMB_JOINT_GEOMETRY_VERSION
    assert tuple(item.joint_id for item in batch.eligibilities) == LIMB_JOINT_IDS
    assert observations["joint/elbow.xmin"].source == "geometry"
    assert observations["joint/elbow.xmin"].confidence_class == "high"
    assert observations["joint/elbow.xmin"].geometry_factors.branch_ratio < 0.1
    assert observations["joint/hand_tip.xmin"].source == "geometry"
    assert eligibility["joint/wrist.xmin"].status == "ambiguous"
    assert eligibility["joint/wrist.xmin"].reason == "wrist_no_stable_bottleneck"
    assert "joint/wrist.xmin" not in observations


def test_limb_geometry_accepts_wrist_only_with_distal_palm_widening(
    tmp_path: Path,
) -> None:
    anatomy = _anatomy(
        tmp_path,
        _torso(),
        _part(
            "handwear",
            side="xmin",
            xyxy=(70, 100, 105, 200),
            points=_wrist_arm_points(),
        ),
    )

    batch = build_limb_joint_evidence(anatomy, build_axial_joint_evidence(anatomy))
    wrist = next(item for item in batch.observations if item.joint_id == "joint/wrist.xmin")

    assert wrist.source == "geometry"
    assert wrist.confidence_class == "high"
    assert 160 <= wrist.y <= 180


def test_limb_geometry_rejects_an_ambiguous_branched_hand_tip(
    tmp_path: Path,
) -> None:
    anatomy = _anatomy(
        tmp_path,
        _torso(),
        _part(
            "handwear",
            side="xmin",
            xyxy=(70, 100, 120, 194),
            points=_branched_arm_points(),
        ),
    )

    batch = build_limb_joint_evidence(anatomy, build_axial_joint_evidence(anatomy))
    eligibility = {item.joint_id: item for item in batch.eligibilities}
    observations = {item.joint_id for item in batch.observations}

    assert eligibility["joint/hand_tip.xmin"].status == "ambiguous"
    assert eligibility["joint/hand_tip.xmin"].reason == "hand_tip_branch_ambiguous"
    assert "joint/hand_tip.xmin" not in observations


def test_limb_geometry_uses_length_prior_for_straight_knee_and_contact_for_foot(
    tmp_path: Path,
) -> None:
    anatomy = _anatomy(
        tmp_path,
        _torso(),
        _part(
            "legwear",
            xyxy=(100, 158, 135, 240),
            points=_legwear_points(),
        ),
        _part(
            "footwear",
            xyxy=(85, 238, 150, 250),
            points=_footwear_points(),
        ),
    )

    batch = build_limb_joint_evidence(anatomy, build_axial_joint_evidence(anatomy))
    observations = {item.joint_id: item for item in batch.observations}

    assert observations["joint/knee.xmin"].source == "length_prior"
    assert observations["joint/knee.xmin"].confidence_class == "weak"
    assert observations["joint/ankle.xmin"].source == "geometry"
    assert observations["joint/toe.xmin"].source == "geometry"
    assert observations["joint/toe.xmin"].x < observations["joint/ankle.xmin"].x
    assert observations["joint/toe.xmax"].x > observations["joint/ankle.xmax"].x


def test_limb_geometry_keeps_ankle_and_toe_unresolved_without_footwear(
    tmp_path: Path,
) -> None:
    anatomy = _anatomy(
        tmp_path,
        _torso(),
        _part(
            "legwear",
            xyxy=(100, 158, 135, 240),
            points=_legwear_points(),
        ),
    )

    batch = build_limb_joint_evidence(anatomy, build_axial_joint_evidence(anatomy))
    eligibility = {item.joint_id: item for item in batch.eligibilities}
    observations = {item.joint_id for item in batch.observations}

    for joint_id in ("joint/ankle.xmin", "joint/toe.xmin"):
        assert eligibility[joint_id].status == "ambiguous"
        assert eligibility[joint_id].reason == "footwear_missing"
        assert joint_id not in observations


def test_limb_geometry_does_not_split_a_merged_ambiguous_arm(tmp_path: Path) -> None:
    anatomy = _anatomy(
        tmp_path,
        _torso(),
        _part(
            "handwear",
            xyxy=(50, 100, 102, 180),
            points=_bent_arm_points(),
        ),
    )

    batch = build_limb_joint_evidence(anatomy, build_axial_joint_evidence(anatomy))
    eligibility = {item.joint_id: item for item in batch.eligibilities}

    for side in ("xmin", "xmax"):
        for joint in ("elbow", "wrist", "hand_tip"):
            record = eligibility[f"joint/{joint}.{side}"]
            assert record.status == "ambiguous"
            assert record.reason == "merged_limb"
            assert "mask/limb/handwear.merged" in record.evidence_ids
