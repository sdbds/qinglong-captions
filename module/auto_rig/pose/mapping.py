from __future__ import annotations

from collections.abc import Iterable

from ..joint_observations import JointObservation, make_joint_observation
from .contracts import RawPoseResult, ScoredKeypoint

POSE_KEYPOINT_MAPPING_VERSION = "pose-keypoint-mapping-v1"

_COCO17_DIRECT = {
    5: "joint/shoulder.xmax",
    6: "joint/shoulder.xmin",
    7: "joint/elbow.xmax",
    8: "joint/elbow.xmin",
    9: "joint/wrist.xmax",
    10: "joint/wrist.xmin",
    11: "joint/hip.xmax",
    12: "joint/hip.xmin",
    13: "joint/knee.xmax",
    14: "joint/knee.xmin",
    15: "joint/ankle.xmax",
    16: "joint/ankle.xmin",
}

_CROWDPOSE14_DIRECT = {
    0: "joint/shoulder.xmax",
    1: "joint/shoulder.xmin",
    2: "joint/elbow.xmax",
    3: "joint/elbow.xmin",
    4: "joint/wrist.xmax",
    5: "joint/wrist.xmin",
    6: "joint/hip.xmax",
    7: "joint/hip.xmin",
    8: "joint/knee.xmax",
    9: "joint/knee.xmin",
    10: "joint/ankle.xmax",
    11: "joint/ankle.xmin",
    13: "joint/neck",
}


def _usable(
    point: ScoredKeypoint,
    result: RawPoseResult,
    minimum_score: float,
) -> bool:
    return bool(point.score >= minimum_score and 0.0 <= point.x < result.canvas_width and 0.0 <= point.y < result.canvas_height)


def _observation(
    result: RawPoseResult,
    *,
    joint_id: str,
    point: ScoredKeypoint,
    evidence_indices: Iterable[int],
) -> JointObservation:
    return make_joint_observation(
        joint_id=joint_id,
        source="pose",
        x=point.x,
        y=point.y,
        confidence_class="model",
        canvas_width=result.canvas_width,
        canvas_height=result.canvas_height,
        evidence_ids=tuple(f"pose/{result.provider_id}/{result.layout}/{index}" for index in evidence_indices),
        pose_score=point.score,
        algorithm_version=POSE_KEYPOINT_MAPPING_VERSION,
    )


def _midpoint(first: ScoredKeypoint, second: ScoredKeypoint) -> ScoredKeypoint:
    return ScoredKeypoint(
        x=(first.x + second.x) / 2.0,
        y=(first.y + second.y) / 2.0,
        score=min(first.score, second.score),
    )


def map_raw_pose_result(
    result: RawPoseResult,
    *,
    minimum_score: float = 0.3,
) -> tuple[JointObservation, ...]:
    if not 0.0 <= minimum_score <= 1.0:
        raise ValueError("minimum pose score must be in [0,1]")
    mapping = _COCO17_DIRECT if result.layout == "coco17" else _CROWDPOSE14_DIRECT
    observations: list[JointObservation] = []
    for index, joint_id in mapping.items():
        point = result.keypoints[index]
        if _usable(point, result, minimum_score):
            observations.append(
                _observation(
                    result,
                    joint_id=joint_id,
                    point=point,
                    evidence_indices=(index,),
                )
            )

    if result.layout == "coco17":
        for joint_id, indices in (
            ("joint/neck", (5, 6)),
            ("joint/pelvis", (11, 12)),
        ):
            first, second = (result.keypoints[index] for index in indices)
            if _usable(first, result, minimum_score) and _usable(second, result, minimum_score):
                observations.append(
                    _observation(
                        result,
                        joint_id=joint_id,
                        point=_midpoint(first, second),
                        evidence_indices=indices,
                    )
                )
    elif all(_usable(result.keypoints[index], result, minimum_score) for index in (6, 7)):
        observations.append(
            _observation(
                result,
                joint_id="joint/pelvis",
                point=_midpoint(result.keypoints[6], result.keypoints[7]),
                evidence_indices=(6, 7),
            )
        )

    return tuple(sorted(observations, key=lambda item: item.observation_id))


__all__ = ["POSE_KEYPOINT_MAPPING_VERSION", "map_raw_pose_result"]
