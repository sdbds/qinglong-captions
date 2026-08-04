from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from PIL import Image

from ..anatomy import AnatomyMaskGeometry
from ..artifacts import canonical_json_sha256
from ..input_identity import TargetInputIdentity
from ..joint_pipeline import (
    StageAJointPlan,
    build_pose_observation_batch,
    build_stage_a_joint_plan,
    partition_pose_observations_by_anatomy,
)
from ..overrides import ValidatedRigOverrides
from .contracts import PoseMode, RawPoseResult
from .mapping import map_raw_pose_result
from .selection import PoseTriggerDecision, decide_pose_trigger

POSE_EXECUTION_VERSION = "pose-execution-v1"
POSE_PREPROCESSING_CONTRACT_VERSION = "pose-preprocessing-v1"


@dataclass(frozen=True, slots=True)
class PoseExecutionResult:
    joint_plan: StageAJointPlan
    selected_provider_id: str | None
    degraded: bool
    report: dict[str, object]


def _runtime_payload(provider: object) -> dict[str, object]:
    report = getattr(provider, "runtime_report", None)
    if report is None:
        raise RuntimeError("pose provider omitted its runtime report")
    if is_dataclass(report):
        payload = asdict(report)
    elif hasattr(report, "__dict__"):
        payload = dict(vars(report))
    else:
        slots = getattr(report, "__slots__", ())
        payload = {name: getattr(report, name) for name in slots}
    if not payload:
        raise RuntimeError("pose provider runtime report is empty")
    canonical_json_sha256(payload)
    return payload


def _person_bbox(anatomy: AnatomyMaskGeometry) -> tuple[int, int, int, int]:
    metric = next(
        (item for item in anatomy.plan.metrics if item.metric_id == "mask/pose_body"),
        None,
    )
    if metric is None or metric.status != "available" or metric.bbox is None:
        raise RuntimeError("pose body mask union is unavailable")
    return metric.bbox


def _provider_keys(mode: PoseMode) -> tuple[str, ...]:
    if mode in {"auto", "sdpose"}:
        return ("sdpose",)
    if mode == "detrpose":
        return ("detrpose",)
    if mode == "compare":
        return ("sdpose", "detrpose")
    return ()


def _provider(
    key: str,
    providers: Mapping[str, object],
    provider_resolver: Callable[[str], object] | None,
) -> object:
    if key in providers:
        return providers[key]
    if provider_resolver is None:
        raise RuntimeError(f"pose provider is unavailable: {key}")
    return provider_resolver(key)


def _resolution_gain(
    geometry_plan: StageAJointPlan,
    candidate: StageAJointPlan,
) -> int:
    before = {item.joint_id for item in geometry_plan.joints.resolutions if item.status != "resolved"}
    return sum(
        item.joint_id in before and item.status == "resolved" and item.source == "pose" for item in candidate.joints.resolutions
    )


def _trigger_payload(decision: PoseTriggerDecision) -> dict[str, object]:
    return {
        "mode": decision.mode,
        "trigger": decision.trigger,
        "reason": decision.reason,
        "merged_families": list(decision.merged_families),
        "unresolved_joint_ids": list(decision.unresolved_joint_ids),
    }


def execute_pose_observation(
    *,
    mode: PoseMode,
    anatomy: AnatomyMaskGeometry,
    geometry_plan: StageAJointPlan,
    target: TargetInputIdentity,
    overrides: ValidatedRigOverrides,
    image: Image.Image,
    providers: Mapping[str, object] | None = None,
    provider_resolver: Callable[[str], object] | None = None,
    minimum_score: float = 0.3,
) -> PoseExecutionResult:
    """Run optional providers as observations; the existing resolver remains authoritative."""

    decision = decide_pose_trigger(mode, anatomy, geometry_plan)
    report: dict[str, object] = {
        "schema_version": POSE_EXECUTION_VERSION,
        "trigger": _trigger_payload(decision),
        "person_bbox": None,
        "provider_runs": [],
        "selected_provider_id": None,
    }
    if not decision.trigger:
        return PoseExecutionResult(
            joint_plan=geometry_plan,
            selected_provider_id=None,
            degraded=False,
            report=report,
        )
    if image.size != (anatomy.plan.canvas_edge, anatomy.plan.canvas_edge):
        raise ValueError("pose source image size differs from the anatomy canvas")

    bbox = _person_bbox(anatomy)
    report["person_bbox"] = list(bbox)
    available = dict(providers or {})
    successes: list[tuple[tuple[int, int, int, int], StageAJointPlan, dict[str, object]]] = []
    runs: list[dict[str, object]] = []
    preference = {"detrpose": 0, "sdpose": 1}
    for key in _provider_keys(mode):
        try:
            provider = _provider(key, available, provider_resolver)
            raw = provider.infer(image, person_bbox=bbox)
            if not isinstance(raw, RawPoseResult):
                raise RuntimeError("pose provider returned an untyped result")
            expected_provider = {
                "sdpose": "sdpose-body17",
                "detrpose": "detrpose-x-crowdpose",
            }[key]
            if raw.provider_id != expected_provider:
                raise RuntimeError(f"pose provider identity mismatch: expected {expected_provider}")
            mapped = map_raw_pose_result(raw, minimum_score=minimum_score)
            accepted, rejected = partition_pose_observations_by_anatomy(
                anatomy,
                geometry_plan.joints.eligibilities,
                mapped,
            )
            runtime = _runtime_payload(provider)
            provider_fingerprint = canonical_json_sha256(
                {
                    "provider_id": raw.provider_id,
                    "provider_version": raw.provider_version,
                    "runtime": runtime,
                }
            )
            preprocessing_fingerprint = canonical_json_sha256(
                {
                    "schema_version": POSE_PREPROCESSING_CONTRACT_VERSION,
                    "provider_id": raw.provider_id,
                    "canvas_size": list(image.size),
                    "person_bbox": list(bbox),
                    "input_size": runtime.get("input_size"),
                    "bbox_padding": runtime.get("bbox_padding"),
                    "alpha_background_rgb": [255, 255, 255],
                }
            )
            batch = build_pose_observation_batch(
                anatomy,
                accepted,
                provider_id=raw.provider_id,
                provider_version=raw.provider_version,
                provider_fingerprint=provider_fingerprint,
                preprocessing_fingerprint=preprocessing_fingerprint,
            )
            candidate = build_stage_a_joint_plan(
                anatomy,
                target=target,
                overrides=overrides,
                pose=batch,
            )
            gain = _resolution_gain(geometry_plan, candidate)
            run: dict[str, object] = {
                "provider_key": key,
                "provider_id": raw.provider_id,
                "provider_version": raw.provider_version,
                "status": "succeeded",
                "runtime": runtime,
                "provider_fingerprint": provider_fingerprint,
                "preprocessing_fingerprint": preprocessing_fingerprint,
                "raw_keypoints": [{"x": item.x, "y": item.y, "score": item.score} for item in raw.keypoints],
                "mapped_observation_ids": [item.observation_id for item in mapped],
                "accepted_observation_ids": [item.observation_id for item in accepted],
                "rejected_observations": [item.to_dict() for item in rejected],
                "pose_resolved_gain": gain,
                "joint_plan_sha256": candidate.plan_sha256,
                "selected": False,
            }
            runs.append(run)
            score = (gain, len(accepted), -len(rejected), preference[key])
            successes.append((score, candidate, run))
        except Exception as exc:
            runs.append(
                {
                    "provider_key": key,
                    "status": "failed",
                    "failure_code": "pose_provider_failed",
                    "error_type": type(exc).__name__,
                    "selected": False,
                }
            )
            if mode in {"sdpose", "detrpose"}:
                raise

    report["provider_runs"] = runs
    if not successes:
        return PoseExecutionResult(
            joint_plan=geometry_plan,
            selected_provider_id=None,
            degraded=True,
            report=report,
        )
    _, selected_plan, selected_run = max(successes, key=lambda item: item[0])
    selected_run["selected"] = True
    selected_provider_id = str(selected_run["provider_id"])
    report["selected_provider_id"] = selected_provider_id
    still_unresolved = {item.joint_id for item in selected_plan.joints.resolutions if item.status != "resolved"}
    degraded = bool(still_unresolved.intersection(decision.unresolved_joint_ids))
    return PoseExecutionResult(
        joint_plan=selected_plan,
        selected_provider_id=selected_provider_id,
        degraded=degraded,
        report=report,
    )


__all__ = [
    "POSE_EXECUTION_VERSION",
    "POSE_PREPROCESSING_CONTRACT_VERSION",
    "PoseExecutionResult",
    "execute_pose_observation",
]
