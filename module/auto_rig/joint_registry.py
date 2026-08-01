from __future__ import annotations

JOINT_REGISTRY_VERSION = "joint-registry-v1"

JOINT_IDS = (
    "joint/pelvis",
    "joint/spine",
    "joint/neck",
    "joint/head_base",
    "joint/head_top",
    "joint/shoulder.xmin",
    "joint/shoulder.xmax",
    "joint/elbow.xmin",
    "joint/elbow.xmax",
    "joint/wrist.xmin",
    "joint/wrist.xmax",
    "joint/hand_tip.xmin",
    "joint/hand_tip.xmax",
    "joint/hip.xmin",
    "joint/hip.xmax",
    "joint/knee.xmin",
    "joint/knee.xmax",
    "joint/ankle.xmin",
    "joint/ankle.xmax",
    "joint/toe.xmin",
    "joint/toe.xmax",
)
JOINT_ID_SET = frozenset(JOINT_IDS)

__all__ = ["JOINT_IDS", "JOINT_ID_SET", "JOINT_REGISTRY_VERSION"]
