from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal

from .jcs import jcs_sha256
from .joint_registry import JOINT_ID_SET, JOINT_IDS, JOINT_REGISTRY_VERSION
from .overrides import ValidatedRigOverrides

JOINT_OBSERVATION_SCHEMA_VERSION = "joint-observation-v1"
JOINT_OBSERVATION_PLAN_VERSION = "joint-observation-plan-v1"
JOINT_RESOLVER_VERSION = "joint-resolver-v1"
POSE_DISAGREEMENT_RADIUS_NUMERATOR = 2
POSE_DISAGREEMENT_RADIUS_DENOMINATOR = 1

ObservationSource = Literal["geometry", "pose", "override", "length_prior"]
ConfidenceClass = Literal["high", "low", "model", "authoritative", "weak"]
EligibilityStatus = Literal["eligible", "ambiguous", "missing"]


class JointObservationContractError(ValueError):
    """Raised when joint evidence mixes scales or violates the resolver contract."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _error(message: str) -> JointObservationContractError:
    return JointObservationContractError("invalid_joint_observation", message)


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise _error(f"{field} must be finite")
    return 0.0 if normalized == 0.0 else normalized


def _canvas_edge(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise _error(f"{field} must be a positive integer")
    return value


def _stable_ids(values: Iterable[str], *, field: str) -> tuple[str, ...]:
    normalized = tuple(values)
    if any(not isinstance(value, str) or not value for value in normalized):
        raise _error(f"{field} must contain non-empty strings")
    if len(normalized) != len(set(normalized)):
        raise _error(f"{field} must not contain duplicates")
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class GeometryConfidenceFactors:
    connectivity: float | None = None
    main_path_length: float | None = None
    branch_ratio: float | None = None
    endpoint_contact: float | None = None
    curvature_peak: float | None = None
    mask_interior_distance: float | None = None
    bilateral_consistency: float | None = None

    def __post_init__(self) -> None:
        for field in (
            "connectivity",
            "main_path_length",
            "branch_ratio",
            "endpoint_contact",
            "curvature_peak",
            "mask_interior_distance",
            "bilateral_consistency",
        ):
            value = getattr(self, field)
            if value is None:
                continue
            normalized = _finite_number(value, field=f"geometry_factors.{field}")
            if not 0.0 <= normalized <= 1.0:
                raise _error(f"geometry_factors.{field} must be in [0,1]")
            object.__setattr__(self, field, normalized)

    def to_dict(self) -> dict[str, float | None]:
        return {
            "connectivity": self.connectivity,
            "main_path_length": self.main_path_length,
            "branch_ratio": self.branch_ratio,
            "endpoint_contact": self.endpoint_contact,
            "curvature_peak": self.curvature_peak,
            "mask_interior_distance": self.mask_interior_distance,
            "bilateral_consistency": self.bilateral_consistency,
        }


@dataclass(frozen=True, slots=True)
class JointEligibility:
    joint_id: str
    status: EligibilityStatus
    reason: str
    evidence_ids: tuple[str, ...]
    local_limb_radius: float | None

    def __post_init__(self) -> None:
        if self.joint_id not in JOINT_ID_SET:
            raise _error(f"unknown joint eligibility ID: {self.joint_id}")
        if self.status not in {"eligible", "ambiguous", "missing"}:
            raise _error(f"unknown joint eligibility status: {self.status}")
        if not isinstance(self.reason, str) or not self.reason:
            raise _error("joint eligibility reason must be non-empty")
        object.__setattr__(
            self,
            "evidence_ids",
            _stable_ids(self.evidence_ids, field="eligibility.evidence_ids"),
        )
        if self.local_limb_radius is not None:
            radius = _finite_number(
                self.local_limb_radius,
                field="eligibility.local_limb_radius",
            )
            if radius <= 0.0:
                raise _error("eligibility.local_limb_radius must be positive")
            object.__setattr__(self, "local_limb_radius", radius)
        if self.status == "missing" and self.local_limb_radius is not None:
            raise _error("missing joint eligibility cannot have a local limb radius")

    def to_dict(self) -> dict[str, object]:
        return {
            "joint_id": self.joint_id,
            "status": self.status,
            "reason": self.reason,
            "evidence_ids": list(self.evidence_ids),
            "local_limb_radius": self.local_limb_radius,
        }


@dataclass(frozen=True, slots=True)
class JointObservation:
    schema_version: str
    observation_id: str
    joint_id: str
    source: ObservationSource
    x: float
    y: float
    canvas_width: int
    canvas_height: int
    confidence_class: ConfidenceClass
    evidence_ids: tuple[str, ...]
    geometry_factors: GeometryConfidenceFactors | None
    pose_score: float | None
    allow_outside: bool
    algorithm_version: str

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "joint_id": self.joint_id,
            "source": self.source,
            "x": self.x,
            "y": self.y,
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "confidence_class": self.confidence_class,
            "evidence_ids": list(self.evidence_ids),
            "geometry_factors": (
                self.geometry_factors.to_dict()
                if self.geometry_factors is not None
                else None
            ),
            "pose_score": self.pose_score,
            "allow_outside": self.allow_outside,
            "algorithm_version": self.algorithm_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_payload(), "observation_id": self.observation_id}


@dataclass(frozen=True, slots=True)
class JointResolverDescriptor:
    schema_version: str
    joint_registry_version: str
    precedence: tuple[str, ...]
    pose_disagreement_radius_numerator: int
    pose_disagreement_radius_denominator: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "joint_registry_version": self.joint_registry_version,
            "precedence": list(self.precedence),
            "pose_disagreement_radius_numerator": self.pose_disagreement_radius_numerator,
            "pose_disagreement_radius_denominator": self.pose_disagreement_radius_denominator,
        }


@dataclass(frozen=True, slots=True)
class JointResolution:
    joint_id: str
    status: Literal["resolved", "unresolved", "missing"]
    quality: Literal["authoritative", "high", "model", "weak", "none"]
    source: ObservationSource | None
    x: float | None
    y: float | None
    selected_observation_id: str | None
    candidate_observation_ids: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "joint_id": self.joint_id,
            "status": self.status,
            "quality": self.quality,
            "source": self.source,
            "x": self.x,
            "y": self.y,
            "selected_observation_id": self.selected_observation_id,
            "candidate_observation_ids": list(self.candidate_observation_ids),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class JointObservationPlan:
    schema_version: str
    canvas_width: int
    canvas_height: int
    resolver: JointResolverDescriptor
    eligibilities: tuple[JointEligibility, ...]
    observations: tuple[JointObservation, ...]
    resolutions: tuple[JointResolution, ...]
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "resolver": self.resolver.to_dict(),
            "eligibilities": [item.to_dict() for item in self.eligibilities],
            "observations": [item.to_dict() for item in self.observations],
            "resolutions": [item.to_dict() for item in self.resolutions],
        }


def make_joint_observation(
    *,
    joint_id: str,
    source: ObservationSource,
    x: float,
    y: float,
    confidence_class: ConfidenceClass,
    canvas_width: int,
    canvas_height: int,
    evidence_ids: Iterable[str],
    geometry_factors: GeometryConfidenceFactors | None = None,
    pose_score: float | None = None,
    allow_outside: bool = False,
    algorithm_version: str,
) -> JointObservation:
    if joint_id not in JOINT_ID_SET:
        raise _error(f"unknown joint observation ID: {joint_id}")
    width = _canvas_edge(canvas_width, field="canvas_width")
    height = _canvas_edge(canvas_height, field="canvas_height")
    normalized_x = _finite_number(x, field="joint.x")
    normalized_y = _finite_number(y, field="joint.y")
    if type(allow_outside) is not bool:
        raise _error("allow_outside must be a boolean")
    if source not in {"geometry", "pose", "override", "length_prior"}:
        raise _error(f"unknown joint observation source: {source}")
    expected_class = {
        "pose": {"model"},
        "override": {"authoritative"},
        "length_prior": {"weak"},
        "geometry": {"high", "low"},
    }[source]
    if confidence_class not in expected_class:
        raise _error(f"confidence class does not match source: {source}")
    if (geometry_factors is not None) != (source == "geometry"):
        raise _error("geometry factors must exist only for geometry observations")
    normalized_pose_score = None
    if source == "pose":
        if pose_score is None:
            raise _error("pose observation requires its raw score")
        normalized_pose_score = _finite_number(pose_score, field="pose_score")
        if not 0.0 <= normalized_pose_score <= 1.0:
            raise _error("pose_score must be in [0,1]")
    elif pose_score is not None:
        raise _error("pose_score must exist only for pose observations")
    outside = not (0.0 <= normalized_x < width and 0.0 <= normalized_y < height)
    if outside and not (source == "override" and allow_outside):
        raise _error("joint observation is outside the Rig canvas")
    if allow_outside and source != "override":
        raise _error("allow_outside is reserved for override observations")
    if not isinstance(algorithm_version, str) or not algorithm_version:
        raise _error("algorithm_version must be a non-empty string")
    values = {
        "schema_version": JOINT_OBSERVATION_SCHEMA_VERSION,
        "joint_id": joint_id,
        "source": source,
        "x": normalized_x,
        "y": normalized_y,
        "canvas_width": width,
        "canvas_height": height,
        "confidence_class": confidence_class,
        "evidence_ids": _stable_ids(evidence_ids, field="observation.evidence_ids"),
        "geometry_factors": geometry_factors,
        "pose_score": normalized_pose_score,
        "allow_outside": allow_outside,
        "algorithm_version": algorithm_version,
    }
    provisional = JointObservation(**values, observation_id="")
    digest = jcs_sha256(provisional.content_payload()).removeprefix("sha256:")
    return JointObservation(**values, observation_id=f"observation/o_{digest}")


def build_override_joint_observations(
    overrides: ValidatedRigOverrides,
    *,
    canvas_width: int,
    canvas_height: int,
) -> tuple[JointObservation, ...]:
    if not isinstance(overrides, ValidatedRigOverrides):
        raise _error("override observations require ValidatedRigOverrides")
    evidence_id = f"override/{overrides.identity.rig_overrides_sha256}"
    return tuple(
        make_joint_observation(
            joint_id=joint.joint_id,
            source="override",
            x=joint.x,
            y=joint.y,
            confidence_class="authoritative",
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            evidence_ids=(evidence_id,),
            allow_outside=joint.allow_outside,
            algorithm_version="rig-overrides-schema-v1",
        )
        for joint in overrides.joints
    )


def _selected(
    observation: JointObservation,
    *,
    candidates: tuple[JointObservation, ...],
    quality: Literal["authoritative", "high", "model", "weak"],
    reason: str,
) -> JointResolution:
    return JointResolution(
        joint_id=observation.joint_id,
        status="resolved",
        quality=quality,
        source=observation.source,
        x=observation.x,
        y=observation.y,
        selected_observation_id=observation.observation_id,
        candidate_observation_ids=tuple(item.observation_id for item in candidates),
        reason=reason,
    )


def _unresolved(
    eligibility: JointEligibility,
    candidates: tuple[JointObservation, ...],
    *,
    status: Literal["unresolved", "missing"],
    reason: str,
) -> JointResolution:
    return JointResolution(
        joint_id=eligibility.joint_id,
        status=status,
        quality="none",
        source=None,
        x=None,
        y=None,
        selected_observation_id=None,
        candidate_observation_ids=tuple(item.observation_id for item in candidates),
        reason=reason,
    )


def _resolve_one(
    eligibility: JointEligibility,
    candidates: tuple[JointObservation, ...],
) -> JointResolution:
    by_source = {item.source: item for item in candidates}
    override = by_source.get("override")
    if override is not None:
        return _selected(
            override,
            candidates=candidates,
            quality="authoritative",
            reason="override_applied",
        )
    if eligibility.status == "missing":
        return _unresolved(
            eligibility,
            candidates,
            status="missing",
            reason=eligibility.reason,
        )
    geometry = by_source.get("geometry")
    pose = by_source.get("pose")
    prior = by_source.get("length_prior")
    if geometry is not None and geometry.confidence_class == "high":
        return _selected(
            geometry,
            candidates=candidates,
            quality="high",
            reason="geometry_high_confidence",
        )
    if geometry is not None and pose is not None:
        radius = eligibility.local_limb_radius
        if radius is None:
            return _unresolved(
                eligibility,
                candidates,
                status="unresolved",
                reason="missing_disagreement_scale",
            )
        threshold = (
            radius
            * POSE_DISAGREEMENT_RADIUS_NUMERATOR
            / POSE_DISAGREEMENT_RADIUS_DENOMINATOR
        )
        distance_squared = (geometry.x - pose.x) ** 2 + (geometry.y - pose.y) ** 2
        if distance_squared > threshold * threshold:
            return _unresolved(
                eligibility,
                candidates,
                status="unresolved",
                reason="pose_disagreement",
            )
        return _selected(
            pose,
            candidates=candidates,
            quality="model",
            reason="pose_refines_low_geometry",
        )
    if pose is not None:
        return _selected(
            pose,
            candidates=candidates,
            quality="model",
            reason="pose_validated",
        )
    if prior is not None:
        return _selected(
            prior,
            candidates=candidates,
            quality="weak",
            reason="weak_length_prior",
        )
    if geometry is not None:
        return _unresolved(
            eligibility,
            candidates,
            status="unresolved",
            reason="low_confidence_geometry",
        )
    return _unresolved(
        eligibility,
        candidates,
        status="unresolved",
        reason=eligibility.reason if eligibility.status == "ambiguous" else "no_observation",
    )


def resolve_joint_observations(
    eligibilities: Iterable[JointEligibility],
    observations: Iterable[JointObservation],
    *,
    canvas_width: int,
    canvas_height: int,
) -> JointObservationPlan:
    width = _canvas_edge(canvas_width, field="canvas_width")
    height = _canvas_edge(canvas_height, field="canvas_height")
    raw_eligibilities = tuple(eligibilities)
    if any(not isinstance(item, JointEligibility) for item in raw_eligibilities):
        raise _error("resolver requires typed joint eligibility records")
    by_joint = {item.joint_id: item for item in raw_eligibilities}
    if len(by_joint) != len(raw_eligibilities) or set(by_joint) != JOINT_ID_SET:
        raise _error("resolver requires exactly one eligibility for every registered joint")
    ordered_eligibilities = tuple(by_joint[joint_id] for joint_id in JOINT_IDS)

    raw_observations = tuple(observations)
    if any(not isinstance(item, JointObservation) for item in raw_observations):
        raise _error("resolver requires typed joint observations")
    ordered_observations = tuple(sorted(raw_observations, key=lambda item: item.observation_id))
    if len({item.observation_id for item in ordered_observations}) != len(ordered_observations):
        raise _error("joint observations contain duplicate identities")
    source_keys = [(item.joint_id, item.source) for item in ordered_observations]
    if len(source_keys) != len(set(source_keys)):
        raise _error("resolver accepts at most one observation per joint and source")
    if any(
        item.canvas_width != width or item.canvas_height != height
        for item in ordered_observations
    ):
        raise _error("observation canvas differs from resolver canvas")

    resolutions = []
    for eligibility in ordered_eligibilities:
        candidates = tuple(
            item for item in ordered_observations if item.joint_id == eligibility.joint_id
        )
        resolutions.append(_resolve_one(eligibility, candidates))
    resolver = JointResolverDescriptor(
        schema_version=JOINT_RESOLVER_VERSION,
        joint_registry_version=JOINT_REGISTRY_VERSION,
        precedence=("override", "geometry_high", "pose", "length_prior"),
        pose_disagreement_radius_numerator=POSE_DISAGREEMENT_RADIUS_NUMERATOR,
        pose_disagreement_radius_denominator=POSE_DISAGREEMENT_RADIUS_DENOMINATOR,
    )
    values = {
        "schema_version": JOINT_OBSERVATION_PLAN_VERSION,
        "canvas_width": width,
        "canvas_height": height,
        "resolver": resolver,
        "eligibilities": ordered_eligibilities,
        "observations": ordered_observations,
        "resolutions": tuple(resolutions),
    }
    provisional = JointObservationPlan(**values, plan_sha256="")
    return JointObservationPlan(
        **values,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )


__all__ = [
    "JOINT_OBSERVATION_PLAN_VERSION",
    "JOINT_OBSERVATION_SCHEMA_VERSION",
    "JOINT_RESOLVER_VERSION",
    "POSE_DISAGREEMENT_RADIUS_DENOMINATOR",
    "POSE_DISAGREEMENT_RADIUS_NUMERATOR",
    "GeometryConfidenceFactors",
    "JointEligibility",
    "JointObservation",
    "JointObservationContractError",
    "JointObservationPlan",
    "JointResolution",
    "JointResolverDescriptor",
    "build_override_joint_observations",
    "make_joint_observation",
    "resolve_joint_observations",
]
