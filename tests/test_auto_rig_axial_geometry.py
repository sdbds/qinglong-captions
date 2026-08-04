from __future__ import annotations

import math
from pathlib import Path

from module.auto_rig.anatomy import build_anatomy_mask_geometry
from module.auto_rig.axial_geometry import (
    AXIAL_JOINT_GEOMETRY_VERSION,
    GEOMETRY_EVIDENCE_BATCH_VERSION,
    build_axial_joint_evidence,
)
from module.auto_rig.component_geometry import load_component_geometry
from module.auto_rig.component_plan import build_mask_component_plan
from tests.test_auto_rig_anatomy import _part, _rectangle


def _anatomy(tmp_path: Path, *parts):
    plan = build_mask_component_plan(parts, canvas_edge=768, item_root=tmp_path)
    return build_anatomy_mask_geometry(
        load_component_geometry(plan, item_root=tmp_path),
        canvas_edge=768,
    )


def _two_leg_points() -> set[tuple[int, int]]:
    return {(x, y) for y in range(52) for x in range(12)} | {(x, y) for y in range(52) for x in range(25, 37)}


def _rotated_rectangle_points(
    width: int,
    height: int,
    *,
    center: tuple[float, float],
    half_width: float,
    half_height: float,
    angle_degrees: float,
) -> set[tuple[int, int]]:
    angle = math.radians(angle_degrees)
    transverse = (math.cos(angle), math.sin(angle))
    downward = (-math.sin(angle), math.cos(angle))
    result = set()
    for y in range(height):
        for x in range(width):
            dx = x + 0.5 - center[0]
            dy = y + 0.5 - center[1]
            across = dx * transverse[0] + dy * transverse[1]
            along = dx * downward[0] + dy * downward[1]
            if abs(across) <= half_width and abs(along) <= half_height:
                result.add((x, y))
    return result


def test_axial_geometry_solves_torso_head_and_contact_joints(tmp_path: Path) -> None:
    anatomy = _anatomy(
        tmp_path,
        _part(
            "front hair",
            xyxy=(0, 0, 20, 20),
            points=_rectangle(20, 20),
        ),
        _part(
            "face",
            xyxy=(100, 40, 140, 92),
            points=_rectangle(40, 52),
        ),
        _part(
            "neck",
            xyxy=(115, 88, 125, 104),
            points=_rectangle(10, 16),
        ),
        _part(
            "topwear",
            xyxy=(90, 100, 150, 170),
            points=_rectangle(60, 70),
        ),
        _part(
            "handwear",
            side="xmin",
            xyxy=(60, 108, 92, 142),
            points=_rectangle(32, 34),
        ),
        _part(
            "handwear",
            side="xmax",
            xyxy=(148, 108, 180, 142),
            points=_rectangle(32, 34),
        ),
        _part(
            "legwear",
            xyxy=(102, 168, 139, 220),
            points=_two_leg_points(),
        ),
    )

    batch = build_axial_joint_evidence(anatomy)
    observations = {item.joint_id: item for item in batch.observations}
    eligibilities = {item.joint_id: item for item in batch.eligibilities}

    assert batch.schema_version == GEOMETRY_EVIDENCE_BATCH_VERSION
    assert batch.provider_version == AXIAL_JOINT_GEOMETRY_VERSION
    assert set(observations) == {
        "joint/pelvis",
        "joint/spine",
        "joint/neck",
        "joint/head_base",
        "joint/head_top",
        "joint/shoulder.xmin",
        "joint/shoulder.xmax",
        "joint/hip.xmin",
        "joint/hip.xmax",
    }
    assert observations["joint/head_top"].y < observations["joint/head_base"].y
    assert observations["joint/head_top"].y >= 40
    assert observations["joint/pelvis"].y > observations["joint/spine"].y
    assert observations["joint/spine"].y > observations["joint/neck"].y
    assert observations["joint/shoulder.xmin"].x < observations["joint/shoulder.xmax"].x
    assert observations["joint/hip.xmin"].x < observations["joint/hip.xmax"].x
    assert all(record.status == "eligible" for record in eligibilities.values())


def test_axial_geometry_uses_torso_principal_axis_for_rotated_silhouette(
    tmp_path: Path,
) -> None:
    points = _rotated_rectangle_points(
        180,
        180,
        center=(90, 90),
        half_width=22,
        half_height=70,
        angle_degrees=25,
    )
    anatomy = _anatomy(
        tmp_path,
        _part(
            "topwear",
            xyxy=(100, 100, 280, 280),
            points=points,
        ),
    )

    batch = build_axial_joint_evidence(anatomy)
    observations = {item.joint_id: item for item in batch.observations}
    pelvis = observations["joint/pelvis"]
    spine = observations["joint/spine"]
    direction = (spine.x - pelvis.x, spine.y - pelvis.y)
    magnitude = math.hypot(*direction)
    expected_headward = (math.sin(math.radians(25)), -math.cos(math.radians(25)))
    alignment = (direction[0] * expected_headward[0] + direction[1] * expected_headward[1]) / magnitude

    assert alignment > 0.95


def test_axial_geometry_keeps_dominant_torso_axis_high_confidence_with_tiny_island(
    tmp_path: Path,
) -> None:
    torso = _rectangle(60, 70)
    surviving_island = {(x, y) for y in range(72, 74) for x in range(2)}
    anatomy = _anatomy(
        tmp_path,
        _part(
            "topwear",
            xyxy=(90, 100, 150, 174),
            points=torso | surviving_island,
        ),
    )

    batch = build_axial_joint_evidence(anatomy)
    observations = {item.joint_id: item for item in batch.observations}

    assert observations["joint/pelvis"].confidence_class == "high"
    assert observations["joint/spine"].confidence_class == "high"
    assert observations["joint/pelvis"].geometry_factors.connectivity == 1.0
    assert observations["joint/spine"].geometry_factors.connectivity == 1.0


def test_axial_geometry_uses_head_anchor_for_a_broad_torso_axis(
    tmp_path: Path,
) -> None:
    anatomy = _anatomy(
        tmp_path,
        _part(
            "face",
            xyxy=(112, 30, 148, 76),
            points=_rectangle(36, 46),
        ),
        _part(
            "neck",
            xyxy=(124, 72, 136, 92),
            points=_rectangle(12, 20),
        ),
        _part(
            "topwear",
            xyxy=(85, 88, 175, 168),
            points=_rectangle(90, 80),
        ),
    )

    batch = build_axial_joint_evidence(anatomy)
    observations = {item.joint_id: item for item in batch.observations}

    assert observations["joint/pelvis"].confidence_class == "high"
    assert observations["joint/spine"].confidence_class == "high"
    assert observations["joint/pelvis"].y > observations["joint/spine"].y


def test_axial_geometry_rejects_head_neck_gap_over_relative_limit(tmp_path: Path) -> None:
    anatomy = _anatomy(
        tmp_path,
        _part(
            "face",
            xyxy=(100, 40, 140, 90),
            points=_rectangle(40, 50),
        ),
        _part(
            "neck",
            xyxy=(118, 100, 122, 112),
            points=_rectangle(4, 12),
        ),
    )

    batch = build_axial_joint_evidence(anatomy)
    eligibilities = {item.joint_id: item for item in batch.eligibilities}

    assert eligibilities["joint/head_base"].status == "ambiguous"
    assert eligibilities["joint/head_base"].reason == "head_neck_contact_gap"
    assert eligibilities["joint/head_top"].status == "ambiguous"
    assert {item.joint_id for item in batch.observations}.isdisjoint({"joint/head_base", "joint/head_top"})


def test_axial_geometry_marks_fragmented_head_axis_unresolved(tmp_path: Path) -> None:
    eyes = _rectangle(8, 8) | {(x, y) for y in range(8) for x in range(24, 32)}
    anatomy = _anatomy(
        tmp_path,
        _part(
            "eyewhite",
            xyxy=(100, 40, 132, 48),
            points=eyes,
        ),
        _part(
            "neck",
            xyxy=(112, 48, 120, 60),
            points=_rectangle(8, 12),
        ),
    )

    batch = build_axial_joint_evidence(anatomy)
    eligibility = {item.joint_id: item for item in batch.eligibilities}["joint/head_top"]

    assert eligibility.status == "ambiguous"
    assert eligibility.reason == "head_core_fragmented"


def test_axial_geometry_keeps_missing_torso_joints_without_fake_coordinates(
    tmp_path: Path,
) -> None:
    anatomy = _anatomy(
        tmp_path,
        _part(
            "face",
            xyxy=(100, 40, 140, 90),
            points=_rectangle(40, 50),
        ),
    )

    batch = build_axial_joint_evidence(anatomy)
    eligibility = {item.joint_id: item for item in batch.eligibilities}
    observations = {item.joint_id for item in batch.observations}

    for joint_id in ("joint/pelvis", "joint/spine", "joint/neck"):
        assert eligibility[joint_id].status == "missing"
        assert joint_id not in observations


def test_merged_limb_shoulder_eligibility_references_the_real_merged_mask(
    tmp_path: Path,
) -> None:
    anatomy = _anatomy(
        tmp_path,
        _part(
            "topwear",
            xyxy=(90, 100, 150, 170),
            points=_rectangle(60, 70),
        ),
        _part(
            "handwear",
            xyxy=(55, 105, 185, 145),
            points=_rectangle(130, 40),
        ),
    )

    batch = build_axial_joint_evidence(anatomy)
    eligibility = {item.joint_id: item for item in batch.eligibilities}

    for side in ("xmin", "xmax"):
        record = eligibility[f"joint/shoulder.{side}"]
        assert record.status == "ambiguous"
        assert record.reason == "merged_limb"
        assert "mask/limb/handwear.merged" in record.evidence_ids
        assert f"mask/limb/handwear.{side}" not in record.evidence_ids
