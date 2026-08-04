from __future__ import annotations

from dataclasses import dataclass

from ..anatomy import AnatomyMaskGeometry, validate_anatomy_mask_geometry
from ..joint_pipeline import StageAJointPlan, validate_stage_a_joint_plan
from .contracts import PoseMode

_ARM_HINGES = frozenset(
    {
        "joint/elbow.xmin",
        "joint/elbow.xmax",
        "joint/wrist.xmin",
        "joint/wrist.xmax",
    }
)
_LEG_HINGES = frozenset(
    {
        "joint/knee.xmin",
        "joint/knee.xmax",
        "joint/ankle.xmin",
        "joint/ankle.xmax",
    }
)


@dataclass(frozen=True, slots=True)
class PoseTriggerDecision:
    mode: PoseMode
    trigger: bool
    reason: str
    merged_families: tuple[str, ...]
    unresolved_joint_ids: tuple[str, ...]


def decide_pose_trigger(
    mode: PoseMode,
    anatomy: AnatomyMaskGeometry,
    geometry_plan: StageAJointPlan,
) -> PoseTriggerDecision:
    if mode not in {"disabled", "auto", "sdpose", "detrpose", "compare"}:
        raise ValueError(f"unknown pose mode: {mode}")
    validate_anatomy_mask_geometry(anatomy)
    validate_stage_a_joint_plan(geometry_plan)
    unresolved = frozenset(item.joint_id for item in geometry_plan.joints.resolutions if item.status == "unresolved")
    merged = tuple(
        sorted(record.family for record in anatomy.plan.limb_states if record.state in {"merged-separable", "merged-ambiguous"})
    )
    relevant: set[str] = set()
    if "handwear" in merged:
        relevant.update(_ARM_HINGES)
    if "legwear" in merged or "footwear" in merged:
        relevant.update(_LEG_HINGES)
    unresolved_relevant = tuple(sorted(unresolved & relevant))

    if mode == "disabled":
        return PoseTriggerDecision(mode, False, "pose_disabled", merged, unresolved_relevant)
    if mode in {"sdpose", "detrpose", "compare"}:
        return PoseTriggerDecision(mode, True, "pose_explicit", merged, unresolved_relevant)
    trigger = bool(merged and unresolved_relevant)
    return PoseTriggerDecision(
        mode,
        trigger,
        "merged_limb_unresolved_hinge" if trigger else "geometry_sufficient",
        merged,
        unresolved_relevant,
    )


__all__ = ["PoseTriggerDecision", "decide_pose_trigger"]
