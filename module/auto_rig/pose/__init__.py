"""Optional pose observation providers for auto-rig Stage A."""

from .artifacts import DETRPOSE_X_CROWDPOSE_MODEL, SDPOSE_BODY_SOURCE
from .contracts import PoseLayout, PoseMode, RawPoseResult, ScoredKeypoint
from .mapping import map_raw_pose_result
from .selection import PoseTriggerDecision, decide_pose_trigger

__all__ = [
    "DETRPOSE_X_CROWDPOSE_MODEL",
    "SDPOSE_BODY_SOURCE",
    "PoseLayout",
    "PoseMode",
    "PoseTriggerDecision",
    "RawPoseResult",
    "ScoredKeypoint",
    "decide_pose_trigger",
    "map_raw_pose_result",
]
