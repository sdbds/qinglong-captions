from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from importlib.metadata import version as distribution_version
from typing import Literal

from .anatomy import AnatomyMaskGeometry, validate_anatomy_mask_geometry
from .jcs import jcs_sha256
from .joint_observations import (
    GeometryConfidenceFactors,
    JointEligibility,
    JointObservation,
    make_joint_observation,
)

GEOMETRY_EVIDENCE_BATCH_VERSION = "geometry-evidence-batch-v1"
AXIAL_JOINT_GEOMETRY_VERSION = "axial-joint-geometry-v2"
AXIAL_DOMINANT_COMPONENT_PERCENT = 95

AXIAL_JOINT_IDS = (
    "joint/pelvis",
    "joint/spine",
    "joint/neck",
    "joint/head_base",
    "joint/head_top",
    "joint/shoulder.xmin",
    "joint/shoulder.xmax",
    "joint/hip.xmin",
    "joint/hip.xmax",
)


class AxialGeometryError(ValueError):
    """Raised when authenticated anatomy cannot be evaluated deterministically."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class AxialGeometryDescriptor:
    schema_version: str
    numpy_version: str
    scipy_version: str
    scikit_image_version: str
    pelvis_axis_percentile: int
    spine_axis_percentile: int
    section_band_percent: int
    head_contact_width_percent: int
    limb_contact_width_percent: int
    head_axis_percentile: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "numpy_version": self.numpy_version,
            "scipy_version": self.scipy_version,
            "scikit_image_version": self.scikit_image_version,
            "pelvis_axis_percentile": self.pelvis_axis_percentile,
            "spine_axis_percentile": self.spine_axis_percentile,
            "section_band_percent": self.section_band_percent,
            "head_contact_width_percent": self.head_contact_width_percent,
            "limb_contact_width_percent": self.limb_contact_width_percent,
            "head_axis_percentile": self.head_axis_percentile,
        }


@dataclass(frozen=True, slots=True)
class GeometryEvidenceDiagnostic:
    joint_id: str
    reason: str
    evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "joint_id": self.joint_id,
            "reason": self.reason,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True, slots=True)
class GeometryEvidenceBatch:
    schema_version: str
    provider: Literal["axial"]
    provider_version: str
    canvas_width: int
    canvas_height: int
    anatomy_plan_sha256: str
    descriptor: AxialGeometryDescriptor
    eligibilities: tuple[JointEligibility, ...]
    observations: tuple[JointObservation, ...]
    diagnostics: tuple[GeometryEvidenceDiagnostic, ...]
    batch_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "provider_version": self.provider_version,
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "anatomy_plan_sha256": self.anatomy_plan_sha256,
            "descriptor": self.descriptor.to_dict(),
            "eligibilities": [item.to_dict() for item in self.eligibilities],
            "observations": [item.to_dict() for item in self.observations],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


@dataclass(frozen=True, slots=True)
class _AxisFrame:
    centroid_x: float
    centroid_y: float
    headward_x: float
    headward_y: float
    transverse_x: float
    transverse_y: float
    axial_min: float
    axial_max: float
    transverse_min: float
    transverse_max: float
    eigen_ratio: float
    connected: bool
    target_anchored: bool

    @property
    def axial_span(self) -> float:
        return self.axial_max - self.axial_min

    @property
    def transverse_span(self) -> float:
        return self.transverse_max - self.transverse_min


def _error(message: str) -> AxialGeometryError:
    return AxialGeometryError("invalid_axial_geometry", message)


def _descriptor() -> AxialGeometryDescriptor:
    return AxialGeometryDescriptor(
        schema_version=AXIAL_JOINT_GEOMETRY_VERSION,
        numpy_version=distribution_version("numpy"),
        scipy_version=distribution_version("scipy"),
        scikit_image_version=distribution_version("scikit-image"),
        pelvis_axis_percentile=20,
        spine_axis_percentile=55,
        section_band_percent=3,
        head_contact_width_percent=15,
        limb_contact_width_percent=10,
        head_axis_percentile=95,
    )


def _full_mask(anatomy: AnatomyMaskGeometry, metric_id: str):
    import numpy as np

    pixels = anatomy.mask(metric_id)
    edge = anatomy.plan.canvas_edge
    canvas = np.zeros((edge, edge), dtype=np.bool_)
    if pixels.bbox is None:
        return canvas
    x1, y1, x2, y2 = pixels.bbox
    tight = np.frombuffer(pixels.binary_mask_u8, dtype=np.uint8).reshape(
        pixels.height,
        pixels.width,
    )
    canvas[y1:y2, x1:x2] = tight > 0
    return canvas


def _centroid(mask) -> tuple[float, float] | None:
    import numpy as np

    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return float(xs.mean() + 0.5), float(ys.mean() + 0.5)


def _component_count(mask) -> int:
    import numpy as np
    from scipy.ndimage import label

    _, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    return int(count)


def _principal_axis(mask, target_mask) -> tuple[_AxisFrame, object, object]:
    import numpy as np
    from scipy.ndimage import label

    labels, component_count = label(
        mask,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    connected = component_count == 1
    axis_mask = mask
    if component_count > 1:
        counts = np.bincount(labels.ravel())[1:]
        dominant_label = int(np.argmax(counts)) + 1
        dominant_count = int(counts[dominant_label - 1])
        total_count = int(counts.sum())
        connected = dominant_count * 100 >= total_count * AXIAL_DOMINANT_COMPONENT_PERCENT
        axis_mask = labels == dominant_label
    ys, xs = np.nonzero(axis_mask)
    if len(xs) < 4:
        raise _error("torso mask has too few pixels for an axis")
    points = np.column_stack((xs.astype(np.float64) + 0.5, ys.astype(np.float64) + 0.5))
    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / float(len(points))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal = eigenvectors[:, int(np.argmax(eigenvalues))]
    target = _centroid(target_mask)
    target_anchored = False
    if target is not None:
        toward_target = np.asarray(target, dtype=np.float64) - centroid
        target_distance = float(np.linalg.norm(toward_target))
        if target_distance > 1e-12:
            headward = toward_target / target_distance
            target_anchored = True
        else:
            headward = principal
    else:
        headward = principal
    if not target_anchored and float(headward[1]) > 0.0:
        headward = -headward
    transverse = np.asarray((-headward[1], headward[0]), dtype=np.float64)
    if float(transverse[0]) < 0.0:
        transverse = -transverse
    axial_projection = centered @ headward
    transverse_projection = centered @ transverse
    largest = float(max(eigenvalues))
    smallest = float(min(eigenvalues))
    ratio = largest / max(smallest, 1e-12)
    return (
        _AxisFrame(
            centroid_x=float(centroid[0]),
            centroid_y=float(centroid[1]),
            headward_x=float(headward[0]),
            headward_y=float(headward[1]),
            transverse_x=float(transverse[0]),
            transverse_y=float(transverse[1]),
            axial_min=float(axial_projection.min()),
            axial_max=float(axial_projection.max()),
            transverse_min=float(transverse_projection.min()),
            transverse_max=float(transverse_projection.max()),
            eigen_ratio=ratio,
            connected=connected,
            target_anchored=target_anchored,
        ),
        points,
        axial_projection,
    )


def _axis_section(
    frame: _AxisFrame,
    points,
    projections,
    *,
    percentile: int,
    band_percent: int,
) -> tuple[float, float, float]:
    import numpy as np

    target = frame.axial_min + frame.axial_span * percentile / 100.0
    half_band = max(1.0, frame.axial_span * band_percent / 100.0)
    selected = np.abs(projections - target) <= half_band
    if not np.any(selected):
        selected[int(np.argmin(np.abs(projections - target)))] = True
    section = points[selected]
    relative = section - np.asarray((frame.centroid_x, frame.centroid_y))
    transverse = np.asarray((frame.transverse_x, frame.transverse_y))
    cross = relative @ transverse
    center = section.mean(axis=0)
    radius = max(1.0, float(cross.max() - cross.min()) / 2.0)
    return float(center[0]), float(center[1]), radius


def _boundary(mask):
    from scipy.ndimage import binary_erosion

    return mask & ~binary_erosion(mask, structure=None, border_value=0)


def _contact(mask_a, mask_b, *, max_gap: float):
    import numpy as np
    from scipy.spatial import cKDTree

    overlap = mask_a & mask_b
    if np.any(overlap):
        ys, xs = np.nonzero(overlap)
        return (float(xs.mean() + 0.5), float(ys.mean() + 0.5)), 0.0, 1.0
    points_a = np.argwhere(_boundary(mask_a))
    points_b = np.argwhere(_boundary(mask_b))
    if not len(points_a) or not len(points_b):
        return None
    tree = cKDTree(points_b.astype(float))
    distances, indices = tree.query(points_a.astype(float), k=1)
    minimum = float(distances.min())
    gap = max(0.0, minimum - 1.0)
    if gap > max_gap:
        return None
    selected = np.flatnonzero(np.isclose(distances, minimum, rtol=0.0, atol=1e-12))
    a = points_a[selected].astype(float)
    b = points_b[indices[selected]].astype(float)
    midpoint_yx = ((a + b) / 2.0).mean(axis=0) + 0.5
    score = 1.0 - gap / max(max_gap, 1e-12)
    return (float(midpoint_yx[1]), float(midpoint_yx[0])), gap, float(score)


def _local_radius(mask, point: tuple[float, float]) -> float:
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    distance = distance_transform_edt(mask)
    y = min(max(int(math.floor(point[1])), 0), mask.shape[0] - 1)
    x = min(max(int(math.floor(point[0])), 0), mask.shape[1] - 1)
    if mask[y, x]:
        return max(1.0, float(distance[y, x]))
    ys, xs = np.nonzero(mask)
    index = int(np.argmin((xs + 0.5 - point[0]) ** 2 + (ys + 0.5 - point[1]) ** 2))
    return max(1.0, float(distance[ys[index], xs[index]]))


def _factors(
    frame: _AxisFrame | None,
    *,
    endpoint_contact: float | None,
    local_radius: float,
    reference_width: float,
) -> GeometryConfidenceFactors:
    return GeometryConfidenceFactors(
        connectivity=1.0 if frame is None or frame.connected else 0.0,
        main_path_length=(None if frame is None else min(1.0, frame.axial_span / max(reference_width, 1.0))),
        branch_ratio=None,
        endpoint_contact=endpoint_contact,
        curvature_peak=None,
        mask_interior_distance=min(
            1.0,
            local_radius / max(reference_width / 2.0, 1.0),
        ),
        bilateral_consistency=None,
    )


def _observation(
    joint_id: str,
    point: tuple[float, float],
    *,
    confidence_class: Literal["high", "low"],
    evidence_ids: tuple[str, ...],
    factors: GeometryConfidenceFactors,
    canvas_edge: int,
) -> JointObservation:
    return make_joint_observation(
        joint_id=joint_id,
        source="geometry",
        x=point[0],
        y=point[1],
        confidence_class=confidence_class,
        canvas_width=canvas_edge,
        canvas_height=canvas_edge,
        evidence_ids=evidence_ids,
        geometry_factors=factors,
        algorithm_version=AXIAL_JOINT_GEOMETRY_VERSION,
    )


def _head_top(mask, base: tuple[float, float], neck_mask):
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    if _component_count(mask) != 1:
        return None, "head_core_fragmented"
    skeleton = skeletonize(mask)
    coordinates = [tuple(int(value) for value in row) for row in np.argwhere(skeleton)]
    if len(coordinates) < 2:
        return None, "head_axis_unstable"
    coordinate_set = set(coordinates)
    start = min(
        coordinates,
        key=lambda item: (
            (item[1] + 0.5 - base[0]) ** 2 + (item[0] + 0.5 - base[1]) ** 2,
            item[0],
            item[1],
        ),
    )
    distances = {start: 0.0}
    parents: dict[tuple[int, int], tuple[int, int]] = {}
    queue = [(0.0, start[0], start[1])]
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
    while queue:
        distance, y, x = heapq.heappop(queue)
        node = (y, x)
        if distance != distances[node]:
            continue
        for dy, dx in offsets:
            neighbor = (y + dy, x + dx)
            if neighbor not in coordinate_set:
                continue
            candidate = distance + (math.sqrt(2.0) if dx and dy else 1.0)
            previous = distances.get(neighbor)
            if previous is None or candidate < previous - 1e-12:
                distances[neighbor] = candidate
                parents[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor[0], neighbor[1]))
    maximum = max(distances.values())
    _, mask_xs = np.nonzero(mask)
    head_width = int(mask_xs.max()) - int(mask_xs.min()) + 1
    if maximum < max(4.0, head_width * 0.25):
        return None, "head_axis_too_short"
    neck_center = _centroid(neck_mask)
    head_center = _centroid(mask)
    origin = neck_center if neck_center is not None else base
    away = (head_center[0] - origin[0], head_center[1] - origin[1]) if head_center is not None else (0.0, -1.0)
    farthest = [node for node, distance in distances.items() if math.isclose(distance, maximum)]
    end = min(
        farthest,
        key=lambda node: (
            -((node[1] + 0.5 - base[0]) * away[0] + (node[0] + 0.5 - base[1]) * away[1]),
            node[0],
            node[1],
        ),
    )
    path = [end]
    while path[-1] != start:
        path.append(parents[path[-1]])
    path.reverse()
    target_distance = maximum * 0.95
    selected = min(
        path,
        key=lambda node: (abs(distances[node] - target_distance), node[0], node[1]),
    )
    radius = float(distance_transform_edt(mask)[selected])
    return (
        (selected[1] + 0.5, selected[0] + 0.5, maximum, max(1.0, radius)),
        None,
    )


def build_axial_joint_evidence(anatomy: AnatomyMaskGeometry) -> GeometryEvidenceBatch:
    """Generate torso, head, shoulder, and hip evidence from frozen anatomy masks."""

    validate_anatomy_mask_geometry(anatomy)
    descriptor = _descriptor()
    edge = anatomy.plan.canvas_edge
    metric_by_id = {metric.metric_id: metric for metric in anatomy.plan.metrics}
    torso = _full_mask(anatomy, "mask/torso_core")
    head = _full_mask(anatomy, "mask/head_core")
    neck = _full_mask(anatomy, "mask/neck")
    target = head | neck
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

    torso_frame = None
    torso_points = None
    torso_projections = None
    torso_width = 0.0
    if metric_by_id["mask/torso_core"].status == "available":
        torso_frame, torso_points, torso_projections = _principal_axis(torso, target)
        torso_width = max(1.0, torso_frame.transverse_span)
        torso_quality: Literal["high", "low"] = (
            "high" if torso_frame.connected and (torso_frame.target_anchored or torso_frame.eigen_ratio >= 1.25) else "low"
        )
        for joint_id, percentile in (
            ("joint/pelvis", descriptor.pelvis_axis_percentile),
            ("joint/spine", descriptor.spine_axis_percentile),
        ):
            x, y, radius = _axis_section(
                torso_frame,
                torso_points,
                torso_projections,
                percentile=percentile,
                band_percent=descriptor.section_band_percent,
            )
            eligibility_by_id[joint_id] = JointEligibility(
                joint_id=joint_id,
                status="eligible",
                reason="torso_axis_available",
                evidence_ids=("mask/torso_core",),
                local_limb_radius=radius,
            )
            observations.append(
                _observation(
                    joint_id,
                    (x, y),
                    confidence_class=torso_quality,
                    evidence_ids=("mask/torso_core",),
                    factors=_factors(
                        torso_frame,
                        endpoint_contact=None,
                        local_radius=radius,
                        reference_width=torso_width,
                    ),
                    canvas_edge=edge,
                )
            )
    else:
        for joint_id in ("joint/pelvis", "joint/spine"):
            unavailable(
                joint_id,
                status="missing",
                reason="torso_missing",
                evidence_ids=("mask/torso_core",),
            )

    if torso_frame is None or metric_by_id["mask/neck"].status == "missing":
        unavailable(
            "joint/neck",
            status="missing",
            reason="torso_or_neck_missing",
            evidence_ids=("mask/torso_core", "mask/neck"),
        )
    else:
        contact = _contact(torso, neck, max_gap=max(2.0, torso_width * 0.10))
        if contact is None:
            unavailable(
                "joint/neck",
                status="ambiguous",
                reason="torso_neck_contact_gap",
                evidence_ids=("mask/torso_core", "mask/neck"),
            )
        else:
            point, _, contact_score = contact
            radius = _local_radius(neck, point)
            eligibility_by_id["joint/neck"] = JointEligibility(
                joint_id="joint/neck",
                status="eligible",
                reason="torso_neck_contact",
                evidence_ids=("mask/neck", "mask/torso_core"),
                local_limb_radius=radius,
            )
            observations.append(
                _observation(
                    "joint/neck",
                    point,
                    confidence_class="high",
                    evidence_ids=("mask/neck", "mask/torso_core"),
                    factors=_factors(
                        torso_frame,
                        endpoint_contact=contact_score,
                        local_radius=radius,
                        reference_width=torso_width,
                    ),
                    canvas_edge=edge,
                )
            )

    head_base_point = None
    head_width = float(metric_by_id["mask/head_core"].width)
    if metric_by_id["mask/head_core"].status == "missing":
        unavailable(
            "joint/head_base",
            status="missing",
            reason="head_core_missing",
            evidence_ids=("mask/head_core",),
        )
    elif metric_by_id["mask/neck"].status == "missing":
        unavailable(
            "joint/head_base",
            status="ambiguous",
            reason="neck_missing",
            evidence_ids=("mask/head_core", "mask/neck"),
        )
    else:
        contact = _contact(
            head,
            neck,
            max_gap=max(0.0, head_width * descriptor.head_contact_width_percent / 100.0),
        )
        if contact is None:
            unavailable(
                "joint/head_base",
                status="ambiguous",
                reason="head_neck_contact_gap",
                evidence_ids=("mask/head_core", "mask/neck"),
            )
        else:
            head_base_point, _, contact_score = contact
            radius = _local_radius(head, head_base_point)
            eligibility_by_id["joint/head_base"] = JointEligibility(
                joint_id="joint/head_base",
                status="eligible",
                reason="head_neck_contact",
                evidence_ids=("mask/head_core", "mask/neck"),
                local_limb_radius=radius,
            )
            observations.append(
                _observation(
                    "joint/head_base",
                    head_base_point,
                    confidence_class="high",
                    evidence_ids=("mask/head_core", "mask/neck"),
                    factors=_factors(
                        None,
                        endpoint_contact=contact_score,
                        local_radius=radius,
                        reference_width=max(head_width, 1.0),
                    ),
                    canvas_edge=edge,
                )
            )

    if metric_by_id["mask/head_core"].status == "missing":
        unavailable(
            "joint/head_top",
            status="missing",
            reason="head_core_missing",
            evidence_ids=("mask/head_core",),
        )
    elif _component_count(head) != 1:
        unavailable(
            "joint/head_top",
            status="ambiguous",
            reason="head_core_fragmented",
            evidence_ids=("mask/head_core",),
        )
    elif head_base_point is None:
        unavailable(
            "joint/head_top",
            status="ambiguous",
            reason="head_base_unresolved",
            evidence_ids=("mask/head_core", "mask/neck"),
        )
    else:
        result, reason = _head_top(head, head_base_point, neck)
        if result is None:
            unavailable(
                "joint/head_top",
                status="ambiguous",
                reason=reason or "head_axis_unstable",
                evidence_ids=("mask/head_core", "mask/neck"),
            )
        else:
            x, y, path_length, radius = result
            eligibility_by_id["joint/head_top"] = JointEligibility(
                joint_id="joint/head_top",
                status="eligible",
                reason="head_geodesic_axis",
                evidence_ids=("mask/head_core", "mask/neck"),
                local_limb_radius=radius,
            )
            observations.append(
                _observation(
                    "joint/head_top",
                    (x, y),
                    confidence_class="high",
                    evidence_ids=("mask/head_core", "mask/neck"),
                    factors=GeometryConfidenceFactors(
                        connectivity=1.0,
                        main_path_length=min(1.0, path_length / max(head_width, 1.0)),
                        branch_ratio=None,
                        endpoint_contact=1.0,
                        curvature_peak=None,
                        mask_interior_distance=min(
                            1.0,
                            radius / max(head_width / 2.0, 1.0),
                        ),
                        bilateral_consistency=None,
                    ),
                    canvas_edge=edge,
                )
            )

    limb_state_by_family = {state.family: state for state in anatomy.plan.limb_states}
    for joint_prefix, family in (("shoulder", "handwear"), ("hip", "legwear")):
        state = limb_state_by_family[family]
        for side in ("xmin", "xmax"):
            joint_id = f"joint/{joint_prefix}.{side}"
            limb_metric_id = f"mask/limb/{family}.{side}"
            evidence_ids = ("mask/torso_core", limb_metric_id)
            if state.state == "merged-ambiguous":
                unavailable(
                    joint_id,
                    status="ambiguous",
                    reason="merged_limb",
                    evidence_ids=(
                        "mask/torso_core",
                        f"mask/limb/{family}.merged",
                    ),
                )
                continue
            metric = metric_by_id[limb_metric_id]
            if torso_frame is None or state.state == "missing" or metric.status == "missing":
                unavailable(
                    joint_id,
                    status="missing",
                    reason="limb_or_torso_missing",
                    evidence_ids=evidence_ids,
                )
                continue
            limb_mask = _full_mask(anatomy, limb_metric_id)
            contact = _contact(
                torso,
                limb_mask,
                max_gap=max(
                    2.0,
                    torso_width * descriptor.limb_contact_width_percent / 100.0,
                ),
            )
            if contact is None:
                unavailable(
                    joint_id,
                    status="ambiguous",
                    reason="limb_torso_contact_gap",
                    evidence_ids=evidence_ids,
                )
                continue
            point, _, contact_score = contact
            radius = _local_radius(limb_mask, point)
            eligibility_by_id[joint_id] = JointEligibility(
                joint_id=joint_id,
                status="eligible",
                reason="limb_torso_contact",
                evidence_ids=evidence_ids,
                local_limb_radius=radius,
            )
            observations.append(
                _observation(
                    joint_id,
                    point,
                    confidence_class="high",
                    evidence_ids=evidence_ids,
                    factors=_factors(
                        torso_frame,
                        endpoint_contact=contact_score,
                        local_radius=radius,
                        reference_width=torso_width,
                    ),
                    canvas_edge=edge,
                )
            )

    ordered_eligibilities = tuple(eligibility_by_id[joint_id] for joint_id in AXIAL_JOINT_IDS)
    ordered_observations = tuple(sorted(observations, key=lambda item: item.observation_id))
    ordered_diagnostics = tuple(sorted(diagnostics, key=lambda item: (item.joint_id, item.reason)))
    values = {
        "schema_version": GEOMETRY_EVIDENCE_BATCH_VERSION,
        "provider": "axial",
        "provider_version": AXIAL_JOINT_GEOMETRY_VERSION,
        "canvas_width": edge,
        "canvas_height": edge,
        "anatomy_plan_sha256": anatomy.plan.plan_sha256,
        "descriptor": descriptor,
        "eligibilities": ordered_eligibilities,
        "observations": ordered_observations,
        "diagnostics": ordered_diagnostics,
    }
    provisional = GeometryEvidenceBatch(**values, batch_sha256="")
    return GeometryEvidenceBatch(
        **values,
        batch_sha256=jcs_sha256(provisional.semantic_payload()),
    )


__all__ = [
    "AXIAL_JOINT_GEOMETRY_VERSION",
    "AXIAL_JOINT_IDS",
    "GEOMETRY_EVIDENCE_BATCH_VERSION",
    "AxialGeometryDescriptor",
    "AxialGeometryError",
    "GeometryEvidenceBatch",
    "GeometryEvidenceDiagnostic",
    "build_axial_joint_evidence",
]
