from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from importlib.metadata import version as distribution_version
from typing import Literal

from .anatomy import AnatomyMaskGeometry, validate_anatomy_mask_geometry
from .axial_geometry import (
    GeometryEvidenceBatch,
    GeometryEvidenceDiagnostic,
    _component_count,
    _contact,
    _full_mask,
    _local_radius,
)
from .jcs import jcs_sha256
from .joint_observations import (
    GeometryConfidenceFactors,
    JointEligibility,
    JointObservation,
    make_joint_observation,
)
from .joint_registry import JOINT_IDS

LIMB_GEOMETRY_EVIDENCE_VERSION = "limb-geometry-evidence-v1"
LIMB_JOINT_GEOMETRY_VERSION = "limb-joint-geometry-v1"
LIMB_JOINT_IDS = tuple(
    joint_id
    for joint_id in JOINT_IDS
    if joint_id.split("/", 1)[1].split(".", 1)[0]
    in {"elbow", "wrist", "hand_tip", "knee", "ankle", "toe"}
)


class LimbGeometryError(ValueError):
    """Raised when limb geometry cannot be derived from its frozen evidence."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class LimbGeometryDescriptor:
    schema_version: str
    numpy_version: str
    scipy_version: str
    scikit_image_version: str
    bend_search_start_percent: int
    bend_search_end_percent: int
    bend_window_radius_multiplier_numerator: int
    bend_window_radius_multiplier_denominator: int
    bend_min_degrees: int
    bend_high_degrees: int
    elbow_prior_percent: int
    knee_prior_percent: int
    prior_min_length_diameters: int
    prior_max_length_diameters: int
    wrist_search_start_percent: int
    wrist_search_end_percent: int
    wrist_proximal_widening_percent: int
    wrist_distal_widening_percent: int
    endpoint_branch_ratio_percent: int
    ankle_contact_radius_multiplier: int
    toe_min_length_radius_percent: int

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class LimbGeometryEvidenceBatch:
    schema_version: str
    provider_version: str
    canvas_width: int
    canvas_height: int
    anatomy_plan_sha256: str
    axial_batch_sha256: str
    descriptor: LimbGeometryDescriptor
    eligibilities: tuple[JointEligibility, ...]
    observations: tuple[JointObservation, ...]
    diagnostics: tuple[GeometryEvidenceDiagnostic, ...]
    batch_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "provider_version": self.provider_version,
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "anatomy_plan_sha256": self.anatomy_plan_sha256,
            "axial_batch_sha256": self.axial_batch_sha256,
            "descriptor": self.descriptor.to_dict(),
            "eligibilities": [item.to_dict() for item in self.eligibilities],
            "observations": [item.to_dict() for item in self.observations],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


@dataclass(frozen=True, slots=True)
class _SkeletonPath:
    points: tuple[tuple[float, float], ...]
    radii: tuple[float, ...]
    cumulative_lengths: tuple[float, ...]
    length: float
    branch_ratio: float


def _error(message: str) -> LimbGeometryError:
    return LimbGeometryError("invalid_limb_geometry", message)


def _descriptor() -> LimbGeometryDescriptor:
    return LimbGeometryDescriptor(
        schema_version=LIMB_JOINT_GEOMETRY_VERSION,
        numpy_version=distribution_version("numpy"),
        scipy_version=distribution_version("scipy"),
        scikit_image_version=distribution_version("scikit-image"),
        bend_search_start_percent=18,
        bend_search_end_percent=82,
        bend_window_radius_multiplier_numerator=2,
        bend_window_radius_multiplier_denominator=1,
        bend_min_degrees=30,
        bend_high_degrees=45,
        elbow_prior_percent=50,
        knee_prior_percent=52,
        prior_min_length_diameters=4,
        prior_max_length_diameters=20,
        wrist_search_start_percent=55,
        wrist_search_end_percent=82,
        wrist_proximal_widening_percent=125,
        wrist_distal_widening_percent=140,
        endpoint_branch_ratio_percent=90,
        ankle_contact_radius_multiplier=2,
        toe_min_length_radius_percent=150,
    )


def _trace_main_path(mask, proximal: tuple[float, float]) -> _SkeletonPath | None:
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    if _component_count(mask) != 1:
        return None
    skeleton = skeletonize(mask)
    coordinates = [tuple(int(value) for value in row) for row in np.argwhere(skeleton)]
    if len(coordinates) < 2:
        return None
    coordinate_set = set(coordinates)
    start = min(
        coordinates,
        key=lambda item: (
            (item[1] + 0.5 - proximal[0]) ** 2
            + (item[0] + 0.5 - proximal[1]) ** 2,
            item[0],
            item[1],
        ),
    )
    offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    def neighbors(node: tuple[int, int]):
        y, x = node
        return tuple(
            (y + dy, x + dx)
            for dy, dx in offsets
            if (y + dy, x + dx) in coordinate_set
        )

    distances = {start: 0.0}
    parents: dict[tuple[int, int], tuple[int, int]] = {}
    queue = [(0.0, start[0], start[1])]
    while queue:
        distance, y, x = heapq.heappop(queue)
        node = (y, x)
        if distance != distances[node]:
            continue
        for neighbor in neighbors(node):
            dy = neighbor[0] - y
            dx = neighbor[1] - x
            candidate = distance + (math.sqrt(2.0) if dx and dy else 1.0)
            previous = distances.get(neighbor)
            if previous is None or candidate < previous - 1e-12:
                distances[neighbor] = candidate
                parents[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor[0], neighbor[1]))
    endpoints = [
        node for node in coordinates if node != start and len(neighbors(node)) <= 1
    ]
    if not endpoints:
        endpoints = [node for node in coordinates if node != start]
    endpoints.sort(
        key=lambda node: (
            -distances.get(node, -1.0),
            -(
                (node[1] + 0.5 - proximal[0]) ** 2
                + (node[0] + 0.5 - proximal[1]) ** 2
            ),
            node[0],
            node[1],
        )
    )
    end = endpoints[0]
    length = distances[end]
    if length <= 0.0:
        return None
    endpoint_distances = sorted(
        (distances[node] for node in endpoints if node in distances),
        reverse=True,
    )
    branch_ratio = (
        endpoint_distances[1] / endpoint_distances[0]
        if len(endpoint_distances) > 1 and endpoint_distances[0] > 0.0
        else 0.0
    )
    path = [end]
    while path[-1] != start:
        parent = parents.get(path[-1])
        if parent is None:
            return None
        path.append(parent)
    path.reverse()
    distance_field = distance_transform_edt(mask)
    points = tuple((node[1] + 0.5, node[0] + 0.5) for node in path)
    radii = tuple(max(1.0, float(distance_field[node])) for node in path)
    cumulative = [0.0]
    for first, second in zip(path, path[1:]):
        cumulative.append(
            cumulative[-1]
            + (
                math.sqrt(2.0)
                if first[0] != second[0] and first[1] != second[1]
                else 1.0
            )
        )
    return _SkeletonPath(
        points=points,
        radii=radii,
        cumulative_lengths=tuple(cumulative),
        length=length,
        branch_ratio=branch_ratio,
    )


def _path_index_at_fraction(path: _SkeletonPath, fraction: float) -> int:
    target = path.length * fraction
    return min(
        range(len(path.points)),
        key=lambda index: (abs(path.cumulative_lengths[index] - target), index),
    )


def _bend(path: _SkeletonPath, descriptor: LimbGeometryDescriptor):
    import numpy as np

    best = None
    for index in range(1, len(path.points) - 1):
        fraction = path.cumulative_lengths[index] / path.length
        if not (
            descriptor.bend_search_start_percent / 100.0
            <= fraction
            <= descriptor.bend_search_end_percent / 100.0
        ):
            continue
        window = max(
            3.0,
            path.radii[index]
            * descriptor.bend_window_radius_multiplier_numerator
            / descriptor.bend_window_radius_multiplier_denominator,
        )
        before_candidates = [
            candidate
            for candidate in range(index)
            if path.cumulative_lengths[candidate]
            <= path.cumulative_lengths[index] - window
        ]
        after_candidates = [
            candidate
            for candidate in range(index + 1, len(path.points))
            if path.cumulative_lengths[candidate]
            >= path.cumulative_lengths[index] + window
        ]
        if not before_candidates or not after_candidates:
            continue
        before = before_candidates[-1]
        after = after_candidates[0]
        incoming = np.asarray(path.points[index]) - np.asarray(path.points[before])
        outgoing = np.asarray(path.points[after]) - np.asarray(path.points[index])
        denominator = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
        if denominator <= 0.0:
            continue
        cosine = float(np.clip((incoming @ outgoing) / denominator, -1.0, 1.0))
        degrees = math.degrees(math.acos(cosine))
        candidate = (degrees, -path.cumulative_lengths[index], index)
        if best is None or candidate > best:
            best = candidate
    if best is None or best[0] < descriptor.bend_min_degrees:
        return None
    return best[2], best[0]


def _wrist(path: _SkeletonPath, descriptor: LimbGeometryDescriptor) -> int | None:
    candidates = [
        index
        for index in range(1, len(path.points) - 1)
        if descriptor.wrist_search_start_percent / 100.0
        <= path.cumulative_lengths[index] / path.length
        <= descriptor.wrist_search_end_percent / 100.0
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda index: (path.radii[index], index))
    for index in candidates:
        radius = path.radii[index]
        before = [
            path.radii[candidate]
            for candidate in range(index)
            if path.cumulative_lengths[candidate] >= path.length * 0.35
        ]
        after = [
            path.radii[candidate]
            for candidate in range(index + 1, len(path.points))
            if path.cumulative_lengths[candidate] <= path.length * 0.97
        ]
        if not before or not after:
            continue
        if (
            max(before) * 100
            >= radius * descriptor.wrist_proximal_widening_percent
            and max(after) * 100
            >= radius * descriptor.wrist_distal_widening_percent
        ):
            return index
    return None


def _geometry_factors(
    path: _SkeletonPath,
    index: int,
    *,
    curvature_degrees: float | None = None,
    endpoint_contact: float | None = None,
) -> GeometryConfidenceFactors:
    radius = path.radii[index]
    return GeometryConfidenceFactors(
        connectivity=1.0,
        main_path_length=min(1.0, path.length / max(radius * 8.0, 1.0)),
        branch_ratio=max(0.0, min(1.0, path.branch_ratio)),
        endpoint_contact=endpoint_contact,
        curvature_peak=(
            None
            if curvature_degrees is None
            else min(1.0, curvature_degrees / 90.0)
        ),
        mask_interior_distance=1.0,
        bilateral_consistency=None,
    )


def _geometry_observation(
    joint_id: str,
    path: _SkeletonPath,
    index: int,
    *,
    evidence_ids: tuple[str, ...],
    edge: int,
    curvature_degrees: float | None = None,
    endpoint_contact: float | None = None,
) -> JointObservation:
    return make_joint_observation(
        joint_id=joint_id,
        source="geometry",
        x=path.points[index][0],
        y=path.points[index][1],
        confidence_class="high",
        canvas_width=edge,
        canvas_height=edge,
        evidence_ids=evidence_ids,
        geometry_factors=_geometry_factors(
            path,
            index,
            curvature_degrees=curvature_degrees,
            endpoint_contact=endpoint_contact,
        ),
        algorithm_version=LIMB_JOINT_GEOMETRY_VERSION,
    )


def _prior_observation(
    joint_id: str,
    path: _SkeletonPath,
    index: int,
    *,
    evidence_ids: tuple[str, ...],
    edge: int,
) -> JointObservation:
    return make_joint_observation(
        joint_id=joint_id,
        source="length_prior",
        x=path.points[index][0],
        y=path.points[index][1],
        confidence_class="weak",
        canvas_width=edge,
        canvas_height=edge,
        evidence_ids=evidence_ids,
        algorithm_version=LIMB_JOINT_GEOMETRY_VERSION,
    )


def build_limb_joint_evidence(
    anatomy: AnatomyMaskGeometry,
    axial: GeometryEvidenceBatch,
) -> LimbGeometryEvidenceBatch:
    """Generate limb joints from medial-axis paths and inter-part contacts."""

    validate_anatomy_mask_geometry(anatomy)
    if not isinstance(axial, GeometryEvidenceBatch):
        raise _error("limb geometry requires an axial evidence batch")
    if axial.batch_sha256 != jcs_sha256(axial.semantic_payload()):
        raise _error("axial evidence batch digest mismatch")
    if axial.provider != "axial" or axial.anatomy_plan_sha256 != anatomy.plan.plan_sha256:
        raise _error("axial evidence does not describe this anatomy plan")
    descriptor = _descriptor()
    edge = anatomy.plan.canvas_edge
    metric_by_id = {metric.metric_id: metric for metric in anatomy.plan.metrics}
    state_by_family = {state.family: state for state in anatomy.plan.limb_states}
    axial_observations = {item.joint_id: item for item in axial.observations}
    observations: list[JointObservation] = []
    eligibility_by_id: dict[str, JointEligibility] = {}
    diagnostics: list[GeometryEvidenceDiagnostic] = []

    def unavailable(
        joint_id: str,
        *,
        status: Literal["ambiguous", "missing"],
        reason: str,
        evidence_ids: tuple[str, ...],
    ) -> None:
        eligibility_by_id[joint_id] = JointEligibility(
            joint_id=joint_id,
            status=status,
            reason=reason,
            evidence_ids=evidence_ids,
            local_limb_radius=None,
        )
        diagnostics.append(
            GeometryEvidenceDiagnostic(
                joint_id=joint_id,
                reason=reason,
                evidence_ids=tuple(sorted(evidence_ids)),
            )
        )

    for side in ("xmin", "xmax"):
        arm_ids = tuple(f"joint/{name}.{side}" for name in ("elbow", "wrist", "hand_tip"))
        arm_metric_id = f"mask/limb/handwear.{side}"
        arm_evidence = (arm_metric_id, f"joint/shoulder.{side}")
        arm_state = state_by_family["handwear"]
        shoulder = axial_observations.get(f"joint/shoulder.{side}")
        if arm_state.state == "merged-ambiguous":
            for joint_id in arm_ids:
                unavailable(
                    joint_id,
                    status="ambiguous",
                    reason="merged_limb",
                    evidence_ids=arm_evidence,
                )
        elif metric_by_id[arm_metric_id].status == "missing":
            for joint_id in arm_ids:
                unavailable(
                    joint_id,
                    status="missing",
                    reason="arm_missing",
                    evidence_ids=arm_evidence,
                )
        elif shoulder is None:
            for joint_id in arm_ids:
                unavailable(
                    joint_id,
                    status="ambiguous",
                    reason="shoulder_unresolved",
                    evidence_ids=arm_evidence,
                )
        else:
            arm_path = _trace_main_path(
                _full_mask(anatomy, arm_metric_id),
                (shoulder.x, shoulder.y),
            )
            if arm_path is None:
                for joint_id in arm_ids:
                    unavailable(
                        joint_id,
                        status="ambiguous",
                        reason="limb_medial_axis_unstable",
                        evidence_ids=arm_evidence,
                    )
            else:
                bend = _bend(arm_path, descriptor)
                if bend is None:
                    prior_index = _path_index_at_fraction(
                        arm_path,
                        descriptor.elbow_prior_percent / 100.0,
                    )
                    diameter_ratio = arm_path.length / max(
                        arm_path.radii[prior_index] * 2.0,
                        1.0,
                    )
                    if (
                        descriptor.prior_min_length_diameters
                        <= diameter_ratio
                        <= descriptor.prior_max_length_diameters
                    ):
                        eligibility_by_id[arm_ids[0]] = JointEligibility(
                            joint_id=arm_ids[0],
                            status="eligible",
                            reason="straight_limb_length_prior",
                            evidence_ids=arm_evidence,
                            local_limb_radius=arm_path.radii[prior_index],
                        )
                        observations.append(
                            _prior_observation(
                                arm_ids[0],
                                arm_path,
                                prior_index,
                                evidence_ids=arm_evidence,
                                edge=edge,
                            )
                        )
                    else:
                        unavailable(
                            arm_ids[0],
                            status="ambiguous",
                            reason="elbow_no_curvature_peak",
                            evidence_ids=arm_evidence,
                        )
                else:
                    bend_index, degrees = bend
                    eligibility_by_id[arm_ids[0]] = JointEligibility(
                        joint_id=arm_ids[0],
                        status="eligible",
                        reason="limb_curvature_peak",
                        evidence_ids=arm_evidence,
                        local_limb_radius=arm_path.radii[bend_index],
                    )
                    observations.append(
                        _geometry_observation(
                            arm_ids[0],
                            arm_path,
                            bend_index,
                            evidence_ids=arm_evidence,
                            edge=edge,
                            curvature_degrees=degrees,
                        )
                    )

                wrist_index = _wrist(arm_path, descriptor)
                if wrist_index is None:
                    unavailable(
                        arm_ids[1],
                        status="ambiguous",
                        reason="wrist_no_stable_bottleneck",
                        evidence_ids=arm_evidence,
                    )
                else:
                    eligibility_by_id[arm_ids[1]] = JointEligibility(
                        joint_id=arm_ids[1],
                        status="eligible",
                        reason="wrist_bottleneck_with_palm_widening",
                        evidence_ids=arm_evidence,
                        local_limb_radius=arm_path.radii[wrist_index],
                    )
                    observations.append(
                        _geometry_observation(
                            arm_ids[1],
                            arm_path,
                            wrist_index,
                            evidence_ids=arm_evidence,
                            edge=edge,
                        )
                    )

                endpoint_index = len(arm_path.points) - 1
                if (
                    arm_path.branch_ratio * 100
                    > descriptor.endpoint_branch_ratio_percent
                ):
                    unavailable(
                        arm_ids[2],
                        status="ambiguous",
                        reason="hand_tip_branch_ambiguous",
                        evidence_ids=arm_evidence,
                    )
                else:
                    eligibility_by_id[arm_ids[2]] = JointEligibility(
                        joint_id=arm_ids[2],
                        status="eligible",
                        reason="limb_main_path_endpoint",
                        evidence_ids=arm_evidence,
                        local_limb_radius=arm_path.radii[endpoint_index],
                    )
                    observations.append(
                        _geometry_observation(
                            arm_ids[2],
                            arm_path,
                            endpoint_index,
                            evidence_ids=arm_evidence,
                            edge=edge,
                        )
                    )

        leg_ids = tuple(f"joint/{name}.{side}" for name in ("knee", "ankle", "toe"))
        leg_metric_id = f"mask/limb/legwear.{side}"
        foot_metric_id = f"mask/limb/footwear.{side}"
        leg_evidence = (leg_metric_id, f"joint/hip.{side}")
        leg_state = state_by_family["legwear"]
        hip = axial_observations.get(f"joint/hip.{side}")
        if leg_state.state == "merged-ambiguous":
            for joint_id in leg_ids:
                unavailable(
                    joint_id,
                    status="ambiguous",
                    reason="merged_limb",
                    evidence_ids=leg_evidence,
                )
            continue
        if metric_by_id[leg_metric_id].status == "missing":
            for joint_id in leg_ids:
                unavailable(
                    joint_id,
                    status="missing",
                    reason="leg_missing",
                    evidence_ids=leg_evidence,
                )
            continue
        if hip is None:
            for joint_id in leg_ids:
                unavailable(
                    joint_id,
                    status="ambiguous",
                    reason="hip_unresolved",
                    evidence_ids=leg_evidence,
                )
            continue
        leg_mask = _full_mask(anatomy, leg_metric_id)
        leg_path = _trace_main_path(leg_mask, (hip.x, hip.y))
        if leg_path is None:
            for joint_id in leg_ids:
                unavailable(
                    joint_id,
                    status="ambiguous",
                    reason="limb_medial_axis_unstable",
                    evidence_ids=leg_evidence,
                )
            continue

        bend = _bend(leg_path, descriptor)
        if bend is None:
            prior_index = _path_index_at_fraction(
                leg_path,
                descriptor.knee_prior_percent / 100.0,
            )
            diameter_ratio = leg_path.length / max(
                leg_path.radii[prior_index] * 2.0,
                1.0,
            )
            if (
                descriptor.prior_min_length_diameters
                <= diameter_ratio
                <= descriptor.prior_max_length_diameters
            ):
                eligibility_by_id[leg_ids[0]] = JointEligibility(
                    joint_id=leg_ids[0],
                    status="eligible",
                    reason="straight_limb_length_prior",
                    evidence_ids=leg_evidence,
                    local_limb_radius=leg_path.radii[prior_index],
                )
                observations.append(
                    _prior_observation(
                        leg_ids[0],
                        leg_path,
                        prior_index,
                        evidence_ids=leg_evidence,
                        edge=edge,
                    )
                )
            else:
                unavailable(
                    leg_ids[0],
                    status="ambiguous",
                    reason="knee_no_curvature_peak",
                    evidence_ids=leg_evidence,
                )
        else:
            bend_index, degrees = bend
            eligibility_by_id[leg_ids[0]] = JointEligibility(
                joint_id=leg_ids[0],
                status="eligible",
                reason="limb_curvature_peak",
                evidence_ids=leg_evidence,
                local_limb_radius=leg_path.radii[bend_index],
            )
            observations.append(
                _geometry_observation(
                    leg_ids[0],
                    leg_path,
                    bend_index,
                    evidence_ids=leg_evidence,
                    edge=edge,
                    curvature_degrees=degrees,
                )
            )

        if metric_by_id[foot_metric_id].status == "missing":
            for joint_id in leg_ids[1:]:
                unavailable(
                    joint_id,
                    status="ambiguous",
                    reason="footwear_missing",
                    evidence_ids=(leg_metric_id, foot_metric_id),
                )
            continue
        foot_mask = _full_mask(anatomy, foot_metric_id)
        endpoint_radius = leg_path.radii[-1]
        contact = _contact(
            leg_mask,
            foot_mask,
            max_gap=max(2.0, endpoint_radius * descriptor.ankle_contact_radius_multiplier),
        )
        if contact is None:
            for joint_id in leg_ids[1:]:
                unavailable(
                    joint_id,
                    status="ambiguous",
                    reason="leg_foot_contact_gap",
                    evidence_ids=(leg_metric_id, foot_metric_id),
                )
            continue
        ankle_point, _, contact_score = contact
        ankle_radius = min(
            _local_radius(leg_mask, ankle_point),
            _local_radius(foot_mask, ankle_point),
        )
        ankle_evidence = (leg_metric_id, foot_metric_id)
        eligibility_by_id[leg_ids[1]] = JointEligibility(
            joint_id=leg_ids[1],
            status="eligible",
            reason="leg_foot_contact",
            evidence_ids=ankle_evidence,
            local_limb_radius=ankle_radius,
        )
        ankle_path_index = len(leg_path.points) - 1
        observations.append(
            make_joint_observation(
                joint_id=leg_ids[1],
                source="geometry",
                x=ankle_point[0],
                y=ankle_point[1],
                confidence_class="high",
                canvas_width=edge,
                canvas_height=edge,
                evidence_ids=ankle_evidence,
                geometry_factors=_geometry_factors(
                    leg_path,
                    ankle_path_index,
                    endpoint_contact=contact_score,
                ),
                algorithm_version=LIMB_JOINT_GEOMETRY_VERSION,
            )
        )
        foot_path = _trace_main_path(foot_mask, ankle_point)
        if foot_path is None:
            unavailable(
                leg_ids[2],
                status="ambiguous",
                reason="foot_axis_unstable",
                evidence_ids=ankle_evidence,
            )
            continue
        toe_index = len(foot_path.points) - 1
        if (
            foot_path.branch_ratio * 100 > descriptor.endpoint_branch_ratio_percent
            or foot_path.length * 100
            < foot_path.radii[toe_index] * descriptor.toe_min_length_radius_percent
        ):
            unavailable(
                leg_ids[2],
                status="ambiguous",
                reason="foot_axis_unstable",
                evidence_ids=ankle_evidence,
            )
        else:
            eligibility_by_id[leg_ids[2]] = JointEligibility(
                joint_id=leg_ids[2],
                status="eligible",
                reason="foot_main_path_endpoint",
                evidence_ids=ankle_evidence,
                local_limb_radius=foot_path.radii[toe_index],
            )
            observations.append(
                _geometry_observation(
                    leg_ids[2],
                    foot_path,
                    toe_index,
                    evidence_ids=ankle_evidence,
                    edge=edge,
                )
            )

    ordered_eligibilities = tuple(eligibility_by_id[joint_id] for joint_id in LIMB_JOINT_IDS)
    ordered_observations = tuple(sorted(observations, key=lambda item: item.observation_id))
    ordered_diagnostics = tuple(
        sorted(diagnostics, key=lambda item: (item.joint_id, item.reason))
    )
    values = {
        "schema_version": LIMB_GEOMETRY_EVIDENCE_VERSION,
        "provider_version": LIMB_JOINT_GEOMETRY_VERSION,
        "canvas_width": edge,
        "canvas_height": edge,
        "anatomy_plan_sha256": anatomy.plan.plan_sha256,
        "axial_batch_sha256": axial.batch_sha256,
        "descriptor": descriptor,
        "eligibilities": ordered_eligibilities,
        "observations": ordered_observations,
        "diagnostics": ordered_diagnostics,
    }
    provisional = LimbGeometryEvidenceBatch(**values, batch_sha256="")
    return LimbGeometryEvidenceBatch(
        **values,
        batch_sha256=jcs_sha256(provisional.semantic_payload()),
    )


__all__ = [
    "LIMB_GEOMETRY_EVIDENCE_VERSION",
    "LIMB_JOINT_GEOMETRY_VERSION",
    "LIMB_JOINT_IDS",
    "LimbGeometryDescriptor",
    "LimbGeometryError",
    "LimbGeometryEvidenceBatch",
    "build_limb_joint_evidence",
]
