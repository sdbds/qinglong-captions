from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .anatomy import AnatomyMaskGeometry, validate_anatomy_mask_geometry
from .axial_geometry import (
    AXIAL_JOINT_GEOMETRY_VERSION,
    GEOMETRY_EVIDENCE_BATCH_VERSION,
    GeometryEvidenceBatch,
    build_axial_joint_evidence,
)
from .input_identity import TargetInputIdentity
from .jcs import jcs_sha256
from .joint_observations import (
    JOINT_OBSERVATION_PLAN_VERSION,
    JointEligibility,
    JointObservation,
    JointObservationPlan,
    build_override_joint_observations,
    resolve_joint_observations,
)
from .joint_registry import JOINT_IDS
from .limb_geometry import (
    LIMB_GEOMETRY_EVIDENCE_VERSION,
    LIMB_JOINT_GEOMETRY_VERSION,
    LimbGeometryEvidenceBatch,
    build_limb_joint_evidence,
)
from .overrides import ValidatedRigOverrides

POSE_OBSERVATION_BATCH_VERSION = "pose-observation-batch-v1"
OBSERVATION_ANATOMY_VALIDATOR_VERSION = "observation-anatomy-validator-v1"
STAGE_A_JOINT_PLAN_VERSION = "stage-a-joint-plan-v1"

EVIDENCE_MASK_MIN_TOLERANCE_PX = 2.0
EVIDENCE_MASK_RADIUS_NUMERATOR = 1
EVIDENCE_MASK_RADIUS_DENOMINATOR = 1
EVIDENCE_MASK_MAX_CANVAS_PERCENT = 5

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class JointPipelineError(ValueError):
    """Raised when Stage A joint evidence is not one coherent input snapshot."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _error(message: str) -> JointPipelineError:
    return JointPipelineError("invalid_stage_a_joint_plan", message)


@dataclass(frozen=True, slots=True)
class PoseObservationBatch:
    schema_version: str
    enabled: bool
    provider_id: str | None
    provider_version: str | None
    provider_fingerprint: str | None
    preprocessing_fingerprint: str | None
    anatomy_plan_sha256: str
    canvas_width: int
    canvas_height: int
    observations: tuple[JointObservation, ...]
    batch_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "enabled": self.enabled,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "provider_fingerprint": self.provider_fingerprint,
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
            "anatomy_plan_sha256": self.anatomy_plan_sha256,
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "observations": [item.to_dict() for item in self.observations],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "batch_sha256": self.batch_sha256}


@dataclass(frozen=True, slots=True)
class StageAJointPlan:
    schema_version: str
    target_input_fingerprint: str
    anatomy_plan_sha256: str
    rig_overrides_sha256: str
    observation_anatomy_validator_version: str
    axial: GeometryEvidenceBatch
    limb: LimbGeometryEvidenceBatch
    pose: PoseObservationBatch
    override_observation_ids: tuple[str, ...]
    override_outside_joint_ids: tuple[str, ...]
    joints: JointObservationPlan
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target_input_fingerprint": self.target_input_fingerprint,
            "anatomy_plan_sha256": self.anatomy_plan_sha256,
            "rig_overrides_sha256": self.rig_overrides_sha256,
            "observation_anatomy_validator_version": (
                self.observation_anatomy_validator_version
            ),
            "axial": {
                **self.axial.semantic_payload(),
                "batch_sha256": self.axial.batch_sha256,
            },
            "limb": {
                **self.limb.semantic_payload(),
                "batch_sha256": self.limb.batch_sha256,
            },
            "pose": self.pose.to_dict(),
            "override_observation_ids": list(self.override_observation_ids),
            "override_outside_joint_ids": list(self.override_outside_joint_ids),
            "joints": {
                **self.joints.semantic_payload(),
                "plan_sha256": self.joints.plan_sha256,
            },
        }


def _disabled_pose_batch(anatomy: AnatomyMaskGeometry) -> PoseObservationBatch:
    edge = anatomy.plan.canvas_edge
    values = {
        "schema_version": POSE_OBSERVATION_BATCH_VERSION,
        "enabled": False,
        "provider_id": None,
        "provider_version": None,
        "provider_fingerprint": None,
        "preprocessing_fingerprint": None,
        "anatomy_plan_sha256": anatomy.plan.plan_sha256,
        "canvas_width": edge,
        "canvas_height": edge,
        "observations": (),
    }
    provisional = PoseObservationBatch(**values, batch_sha256="")
    return PoseObservationBatch(
        **values,
        batch_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def build_pose_observation_batch(
    anatomy: AnatomyMaskGeometry,
    observations: Iterable[JointObservation],
    *,
    provider_id: str,
    provider_version: str,
    provider_fingerprint: str,
    preprocessing_fingerprint: str,
) -> PoseObservationBatch:
    """Bind provider observations to the anatomy and preprocessing snapshot."""

    validate_anatomy_mask_geometry(anatomy)
    if not isinstance(provider_id, str) or not provider_id:
        raise _error("pose provider_id must be non-empty")
    if not isinstance(provider_version, str) or not provider_version:
        raise _error("pose provider_version must be non-empty")
    if not isinstance(provider_fingerprint, str) or not _SHA256_PATTERN.fullmatch(
        provider_fingerprint
    ):
        raise _error("pose provider_fingerprint must be a canonical SHA-256")
    if not isinstance(
        preprocessing_fingerprint, str
    ) or not _SHA256_PATTERN.fullmatch(preprocessing_fingerprint):
        raise _error("pose preprocessing_fingerprint must be a canonical SHA-256")
    raw_observations = tuple(observations)
    if any(not isinstance(item, JointObservation) for item in raw_observations):
        raise _error("pose batch requires typed joint observations")
    ordered_observations = tuple(
        sorted(raw_observations, key=lambda item: item.observation_id)
    )
    edge = anatomy.plan.canvas_edge
    values = {
        "schema_version": POSE_OBSERVATION_BATCH_VERSION,
        "enabled": True,
        "provider_id": provider_id,
        "provider_version": provider_version,
        "provider_fingerprint": provider_fingerprint,
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "anatomy_plan_sha256": anatomy.plan.plan_sha256,
        "canvas_width": edge,
        "canvas_height": edge,
        "observations": ordered_observations,
    }
    provisional = PoseObservationBatch(**values, batch_sha256="")
    batch = PoseObservationBatch(
        **values,
        batch_sha256=jcs_sha256(provisional.semantic_payload()),
    )
    _validate_pose_batch(batch, anatomy)
    return batch


def _validate_target(
    target: TargetInputIdentity,
    anatomy: AnatomyMaskGeometry,
) -> None:
    if not isinstance(target, TargetInputIdentity):
        raise _error("joint plan requires a TargetInputIdentity")
    if target.target_input_fingerprint != jcs_sha256(target.semantic_payload()):
        raise _error("target input fingerprint mismatch")
    edge = anatomy.plan.canvas_edge
    if target.canvas_width != edge or target.canvas_height != edge:
        raise _error("target canvas differs from anatomy canvas")


def _validate_overrides(
    overrides: ValidatedRigOverrides,
    target: TargetInputIdentity,
) -> None:
    if not isinstance(overrides, ValidatedRigOverrides):
        raise _error("joint plan requires ValidatedRigOverrides")
    identity = overrides.identity
    if identity.rig_overrides_sha256 != jcs_sha256(identity.semantic_payload()):
        raise _error("override input identity mismatch")
    if identity.present:
        if overrides.target_input_fingerprint != target.target_input_fingerprint:
            raise _error("validated overrides belong to a different target")
    elif overrides.target_input_fingerprint is not None:
        raise _error("absent override input cannot name a target")


def _mask_distance(
    anatomy: AnatomyMaskGeometry,
    metric_id: str,
    point: tuple[float, float],
) -> float:
    import numpy as np

    try:
        pixels = anatomy.mask(metric_id)
    except KeyError:
        return float("inf")
    if pixels.bbox is None:
        return float("inf")
    mask = np.frombuffer(pixels.binary_mask_u8, dtype=np.uint8).reshape(
        pixels.height,
        pixels.width,
    )
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return float("inf")
    x1, y1, _, _ = pixels.bbox
    dx = xs.astype(np.float64) + x1 + 0.5 - point[0]
    dy = ys.astype(np.float64) + y1 + 0.5 - point[1]
    return float(np.sqrt(np.min(dx * dx + dy * dy)))


def _validate_override_geometry(
    anatomy: AnatomyMaskGeometry,
    overrides: ValidatedRigOverrides,
    eligibilities: tuple[JointEligibility, ...],
) -> tuple[str, ...]:
    eligibility_by_id = {item.joint_id: item for item in eligibilities}
    edge = anatomy.plan.canvas_edge
    warning_ids: list[str] = []
    for override in overrides.joints:
        eligibility = eligibility_by_id[override.joint_id]
        tolerance = _evidence_tolerance(
            eligibility,
            canvas_edge=edge,
        )
        distance = _distance_to_eligibility_evidence(
            anatomy,
            eligibility,
            (override.x, override.y),
        )
        if distance > tolerance and not override.allow_outside:
            raise _error(
                f"joint override is outside its anatomy evidence: {override.joint_id}"
            )
        if override.allow_outside:
            warning_ids.append(override.joint_id)
    return tuple(sorted(warning_ids))


def _evidence_tolerance(
    eligibility: JointEligibility,
    *,
    canvas_edge: int,
) -> float:
    radius = eligibility.local_limb_radius
    tolerance = max(
        EVIDENCE_MASK_MIN_TOLERANCE_PX,
        0.0
        if radius is None
        else radius
        * EVIDENCE_MASK_RADIUS_NUMERATOR
        / EVIDENCE_MASK_RADIUS_DENOMINATOR,
    )
    return min(
        tolerance,
        canvas_edge * EVIDENCE_MASK_MAX_CANVAS_PERCENT / 100.0,
    )


def _distance_to_eligibility_evidence(
    anatomy: AnatomyMaskGeometry,
    eligibility: JointEligibility,
    point: tuple[float, float],
) -> float:
    metric_ids = tuple(
        evidence_id
        for evidence_id in eligibility.evidence_ids
        if evidence_id.startswith("mask/")
    )
    return min(
        (
            _mask_distance(anatomy, metric_id, point)
            for metric_id in metric_ids
        ),
        default=float("inf"),
    )


def _validate_pose_geometry(
    anatomy: AnatomyMaskGeometry,
    pose: PoseObservationBatch,
    eligibilities: tuple[JointEligibility, ...],
) -> None:
    eligibility_by_id = {item.joint_id: item for item in eligibilities}
    edge = anatomy.plan.canvas_edge
    for observation in pose.observations:
        eligibility = eligibility_by_id[observation.joint_id]
        distance = _distance_to_eligibility_evidence(
            anatomy,
            eligibility,
            (observation.x, observation.y),
        )
        tolerance = _evidence_tolerance(eligibility, canvas_edge=edge)
        if distance > tolerance:
            raise _error(
                f"pose observation is outside anatomy evidence: {observation.joint_id}"
            )


def _validate_pose_batch(
    pose: PoseObservationBatch,
    anatomy: AnatomyMaskGeometry,
) -> None:
    _validate_pose_batch_values(
        pose,
        anatomy_plan_sha256=anatomy.plan.plan_sha256,
        canvas_width=anatomy.plan.canvas_edge,
        canvas_height=anatomy.plan.canvas_edge,
    )


def _validate_pose_batch_values(
    pose: PoseObservationBatch,
    *,
    anatomy_plan_sha256: str,
    canvas_width: int,
    canvas_height: int,
) -> None:
    if not isinstance(pose, PoseObservationBatch):
        raise _error("pose input must use PoseObservationBatch")
    if pose.schema_version != POSE_OBSERVATION_BATCH_VERSION:
        raise _error("unsupported pose observation batch version")
    if pose.batch_sha256 != jcs_sha256(pose.semantic_payload()):
        raise _error("pose observation batch digest mismatch")
    if pose.anatomy_plan_sha256 != anatomy_plan_sha256:
        raise _error("pose observations describe a different anatomy plan")
    if pose.canvas_width != canvas_width or pose.canvas_height != canvas_height:
        raise _error("pose observation canvas differs from anatomy canvas")
    if pose.enabled:
        if not all(
            isinstance(value, str) and value
            for value in (pose.provider_id, pose.provider_version)
        ):
            raise _error("enabled pose input requires provider identity")
        if not all(
            isinstance(value, str) and _SHA256_PATTERN.fullmatch(value)
            for value in (pose.provider_fingerprint, pose.preprocessing_fingerprint)
        ):
            raise _error("enabled pose input requires canonical fingerprints")
    elif any(
        value is not None
        for value in (
            pose.provider_id,
            pose.provider_version,
            pose.provider_fingerprint,
            pose.preprocessing_fingerprint,
        )
    ) or pose.observations:
        raise _error("disabled pose input must be an empty identity")
    if any(item.source != "pose" for item in pose.observations):
        raise _error("pose batch contains a non-pose observation")
    if any(
        item.canvas_width != canvas_width or item.canvas_height != canvas_height
        for item in pose.observations
    ):
        raise _error("pose observation canvas differs from anatomy canvas")
    keys = tuple(item.joint_id for item in pose.observations)
    if len(keys) != len(set(keys)):
        raise _error("pose batch contains duplicate joint observations")


def validate_stage_a_joint_plan(plan: StageAJointPlan) -> StageAJointPlan:
    """Validate a persisted Stage A joint plan without consulting mask pixels."""

    if not isinstance(plan, StageAJointPlan):
        raise _error("joint plan must use StageAJointPlan")
    if plan.schema_version != STAGE_A_JOINT_PLAN_VERSION:
        raise _error("unsupported Stage A joint plan version")
    if not _SHA256_PATTERN.fullmatch(plan.target_input_fingerprint):
        raise _error("target input fingerprint is not canonical")
    if not _SHA256_PATTERN.fullmatch(plan.anatomy_plan_sha256):
        raise _error("anatomy plan digest is not canonical")
    if not _SHA256_PATTERN.fullmatch(plan.rig_overrides_sha256):
        raise _error("override input digest is not canonical")
    if (
        plan.observation_anatomy_validator_version
        != OBSERVATION_ANATOMY_VALIDATOR_VERSION
    ):
        raise _error("observation anatomy validator version mismatch")

    axial = plan.axial
    if not isinstance(axial, GeometryEvidenceBatch):
        raise _error("axial evidence must use GeometryEvidenceBatch")
    if (
        axial.schema_version != GEOMETRY_EVIDENCE_BATCH_VERSION
        or axial.provider != "axial"
        or axial.provider_version != AXIAL_JOINT_GEOMETRY_VERSION
    ):
        raise _error("axial evidence version mismatch")
    if axial.batch_sha256 != jcs_sha256(axial.semantic_payload()):
        raise _error("axial evidence batch digest mismatch")
    if axial.anatomy_plan_sha256 != plan.anatomy_plan_sha256:
        raise _error("axial evidence describes a different anatomy plan")

    limb = plan.limb
    if not isinstance(limb, LimbGeometryEvidenceBatch):
        raise _error("limb evidence must use LimbGeometryEvidenceBatch")
    if (
        limb.schema_version != LIMB_GEOMETRY_EVIDENCE_VERSION
        or limb.provider_version != LIMB_JOINT_GEOMETRY_VERSION
    ):
        raise _error("limb evidence version mismatch")
    if limb.batch_sha256 != jcs_sha256(limb.semantic_payload()):
        raise _error("limb evidence batch digest mismatch")
    if limb.anatomy_plan_sha256 != plan.anatomy_plan_sha256:
        raise _error("limb evidence describes a different anatomy plan")
    if limb.axial_batch_sha256 != axial.batch_sha256:
        raise _error("limb evidence references a different axial batch")
    if (limb.canvas_width, limb.canvas_height) != (
        axial.canvas_width,
        axial.canvas_height,
    ):
        raise _error("geometry evidence canvases differ")

    _validate_pose_batch_values(
        plan.pose,
        anatomy_plan_sha256=plan.anatomy_plan_sha256,
        canvas_width=axial.canvas_width,
        canvas_height=axial.canvas_height,
    )
    if plan.joints.schema_version != JOINT_OBSERVATION_PLAN_VERSION:
        raise _error("joint observation plan version mismatch")
    if plan.joints.plan_sha256 != jcs_sha256(plan.joints.semantic_payload()):
        raise _error("joint observation plan digest mismatch")

    eligibility_by_id = {
        item.joint_id: item for item in axial.eligibilities + limb.eligibilities
    }
    if set(eligibility_by_id) != set(JOINT_IDS):
        raise _error("geometry evidence does not cover the joint registry")
    expected_eligibilities = tuple(eligibility_by_id[joint_id] for joint_id in JOINT_IDS)
    if plan.joints.eligibilities != expected_eligibilities:
        raise _error("joint eligibility projection differs from geometry evidence")

    override_ids = tuple(sorted(plan.override_observation_ids))
    if plan.override_observation_ids != override_ids:
        raise _error("override observation identities are not canonical")
    observation_by_id = {
        item.observation_id: item for item in plan.joints.observations
    }
    if any(observation_id not in observation_by_id for observation_id in override_ids):
        raise _error("override observation identity is absent from the joint plan")
    if any(observation_by_id[item].source != "override" for item in override_ids):
        raise _error("override observation identity names another source")
    actual_override_ids = tuple(
        item.observation_id
        for item in plan.joints.observations
        if item.source == "override"
    )
    if actual_override_ids != override_ids:
        raise _error("joint plan contains undeclared override observations")
    expected_observations = tuple(
        sorted(
            axial.observations
            + limb.observations
            + plan.pose.observations
            + tuple(observation_by_id[item] for item in override_ids),
            key=lambda item: item.observation_id,
        )
    )
    if plan.joints.observations != expected_observations:
        raise _error("joint observation projection differs from source batches")

    outside_ids = tuple(sorted(plan.override_outside_joint_ids))
    if plan.override_outside_joint_ids != outside_ids:
        raise _error("outside override joint identities are not canonical")
    override_by_joint = {
        observation_by_id[item].joint_id: observation_by_id[item]
        for item in override_ids
    }
    if any(
        joint_id not in override_by_joint or not override_by_joint[joint_id].allow_outside
        for joint_id in outside_ids
    ):
        raise _error("outside override warning lacks an authorized observation")
    expected_outside_ids = tuple(
        sorted(item.joint_id for item in override_by_joint.values() if item.allow_outside)
    )
    if outside_ids != expected_outside_ids:
        raise _error("outside override warning set differs from override coordinates")

    expected_joints = resolve_joint_observations(
        expected_eligibilities,
        expected_observations,
        canvas_width=axial.canvas_width,
        canvas_height=axial.canvas_height,
    )
    if plan.joints != expected_joints:
        raise _error("joint resolutions differ from the frozen resolver")
    if plan.plan_sha256 != jcs_sha256(plan.semantic_payload()):
        raise _error("Stage A joint plan digest mismatch")
    return plan


def build_stage_a_joint_plan(
    anatomy: AnatomyMaskGeometry,
    *,
    target: TargetInputIdentity,
    overrides: ValidatedRigOverrides,
    pose: PoseObservationBatch | None = None,
) -> StageAJointPlan:
    """Fuse Stage A geometry, optional pose evidence, and authoritative overrides."""

    validate_anatomy_mask_geometry(anatomy)
    _validate_target(target, anatomy)
    _validate_overrides(overrides, target)
    pose_batch = _disabled_pose_batch(anatomy) if pose is None else pose
    _validate_pose_batch(pose_batch, anatomy)

    axial = build_axial_joint_evidence(anatomy)
    limb = build_limb_joint_evidence(anatomy, axial)
    eligibilities = axial.eligibilities + limb.eligibilities
    _validate_pose_geometry(anatomy, pose_batch, eligibilities)
    override_outside_joint_ids = _validate_override_geometry(
        anatomy,
        overrides,
        eligibilities,
    )
    override_observations = build_override_joint_observations(
        overrides,
        canvas_width=target.canvas_width,
        canvas_height=target.canvas_height,
    )
    observations = (
        axial.observations
        + limb.observations
        + pose_batch.observations
        + override_observations
    )
    joints = resolve_joint_observations(
        eligibilities,
        observations,
        canvas_width=target.canvas_width,
        canvas_height=target.canvas_height,
    )
    values = {
        "schema_version": STAGE_A_JOINT_PLAN_VERSION,
        "target_input_fingerprint": target.target_input_fingerprint,
        "anatomy_plan_sha256": anatomy.plan.plan_sha256,
        "rig_overrides_sha256": overrides.identity.rig_overrides_sha256,
        "observation_anatomy_validator_version": (
            OBSERVATION_ANATOMY_VALIDATOR_VERSION
        ),
        "axial": axial,
        "limb": limb,
        "pose": pose_batch,
        "override_observation_ids": tuple(
            sorted(item.observation_id for item in override_observations)
        ),
        "override_outside_joint_ids": override_outside_joint_ids,
        "joints": joints,
    }
    provisional = StageAJointPlan(**values, plan_sha256="")
    plan = StageAJointPlan(
        **values,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )
    return validate_stage_a_joint_plan(plan)


__all__ = [
    "OBSERVATION_ANATOMY_VALIDATOR_VERSION",
    "POSE_OBSERVATION_BATCH_VERSION",
    "STAGE_A_JOINT_PLAN_VERSION",
    "JointPipelineError",
    "PoseObservationBatch",
    "StageAJointPlan",
    "build_pose_observation_batch",
    "build_stage_a_joint_plan",
    "validate_stage_a_joint_plan",
]
