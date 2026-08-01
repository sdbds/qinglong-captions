from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from .component_geometry import (
    MESH_COMPONENT_SOURCE_VERSION,
    MeshComponentSource,
    load_mesh_component_sources,
)
from .component_plan import MaskComponentPlan
from .jcs import jcs_sha256

MESH_BUILD_PLAN_VERSION = "mesh-build-plan-v1"
MESH_COMPONENT_ID_SCHEMA = "mesh-component-id-v1"
MESH_DESCRIPTOR_VERSION = "mesh-build-descriptor-v1"
MESH_CONTOUR_POLICY_VERSION = "canonical-marching-squares-v1"
MESH_SAMPLER_VERSION = "scale-relative-mesh-sampler-v1"
MESH_TRIANGLE_FILTER_VERSION = "alpha-quarter-sample-filter-v1"
MESH_QUANTIZATION_DENOMINATOR = 256
MESH_SYMBOLIC_PERTURBATION_DENOMINATOR = 4096
MESH_BOUNDARY_SPACING_CANVAS_DENOMINATOR = 512
MESH_INTERIOR_SPACING_CANVAS_DENOMINATOR = 128
MESH_MIN_BOUNDARY_SPACING_PX = 1
MESH_MIN_INTERIOR_SPACING_PX = 2
MESH_QHULL_OPTIONS = "Qbb Qc Qz Q12"
MESH_TRIANGLE_EDGE_SAMPLE_NUMERATORS = (1, 2, 3)
MESH_TRIANGLE_EDGE_SAMPLE_DENOMINATOR = 4
MESH_MAX_CIRCUMRADIUS_LOCAL_RADIUS_RATIO = 8.0

_COMPONENT_ID_RE = re.compile(r"component/c_[0-9a-f]{64}\Z")
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class MeshBuildError(ValueError):
    """Raised when authenticated component geometry cannot form a valid mesh plan."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _error(code: str, message: str) -> MeshBuildError:
    return MeshBuildError(code, message)


@dataclass(frozen=True, slots=True)
class MeshBuildDescriptor:
    schema_version: str
    contour_policy_version: str
    sampler_version: str
    triangle_filter_version: str
    numpy_version: str
    scipy_version: str
    scipy_qhull_provider: str
    scikit_image_version: str
    qhull_options: str
    quantization_denominator: int
    symbolic_perturbation_denominator: int
    boundary_spacing_canvas_denominator: int
    interior_spacing_canvas_denominator: int
    minimum_boundary_spacing_px: int
    minimum_interior_spacing_px: int
    edge_sample_numerators: tuple[int, ...]
    edge_sample_denominator: int
    maximum_circumradius_local_radius_ratio: float
    descriptor_sha256: str

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contour_policy_version": self.contour_policy_version,
            "sampler_version": self.sampler_version,
            "triangle_filter_version": self.triangle_filter_version,
            "numpy_version": self.numpy_version,
            "scipy_version": self.scipy_version,
            "scipy_qhull_provider": self.scipy_qhull_provider,
            "scikit_image_version": self.scikit_image_version,
            "qhull_options": self.qhull_options,
            "quantization_denominator": self.quantization_denominator,
            "symbolic_perturbation_denominator": self.symbolic_perturbation_denominator,
            "boundary_spacing_canvas_denominator": (
                self.boundary_spacing_canvas_denominator
            ),
            "interior_spacing_canvas_denominator": (
                self.interior_spacing_canvas_denominator
            ),
            "minimum_boundary_spacing_px": self.minimum_boundary_spacing_px,
            "minimum_interior_spacing_px": self.minimum_interior_spacing_px,
            "edge_sample_numerators": list(self.edge_sample_numerators),
            "edge_sample_denominator": self.edge_sample_denominator,
            "maximum_circumradius_local_radius_ratio": (
                self.maximum_circumradius_local_radius_ratio
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_payload(), "descriptor_sha256": self.descriptor_sha256}


@dataclass(frozen=True, slots=True)
class MeshSourceRecord:
    component_id: str
    part_id: str
    source_kind: Literal["see_through", "native_variant"]
    variant_id: str | None
    semantic_role: str | None
    side: str | None
    part_xyxy: tuple[int, int, int, int]
    bbox: tuple[int, int, int, int]
    component_mask_sha256: str
    pixel_count: int
    source_sha256: str

    def content_payload(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "part_id": self.part_id,
            "source_kind": self.source_kind,
            "variant_id": self.variant_id,
            "semantic_role": self.semantic_role,
            "side": self.side,
            "part_xyxy": list(self.part_xyxy),
            "bbox": list(self.bbox),
            "component_mask_sha256": self.component_mask_sha256,
            "pixel_count": self.pixel_count,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_payload(), "source_sha256": self.source_sha256}


@dataclass(frozen=True, slots=True)
class MeshVertex:
    position: tuple[float, float]
    uv: tuple[float, float]
    boundary: bool
    boundary_loop: int | None
    boundary_order: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "position": list(self.position),
            "uv": list(self.uv),
            "boundary": self.boundary,
            "boundary_loop": self.boundary_loop,
            "boundary_order": self.boundary_order,
        }


@dataclass(frozen=True, slots=True)
class MeshRecord:
    mesh_id: str
    component_id: str
    part_id: str
    source_kind: Literal["see_through", "native_variant"]
    variant_id: str | None
    side: str | None
    component_bbox: tuple[int, int, int, int]
    component_mask_sha256: str
    build_descriptor_sha256: str
    vertices: tuple[MeshVertex, ...]
    boundary_loops: tuple[tuple[int, ...], ...]
    triangles: tuple[int, ...]
    mesh_sha256: str

    def content_payload(self) -> dict[str, object]:
        return {
            "mesh_id": self.mesh_id,
            "component_id": self.component_id,
            "part_id": self.part_id,
            "source_kind": self.source_kind,
            "variant_id": self.variant_id,
            "side": self.side,
            "component_bbox": list(self.component_bbox),
            "component_mask_sha256": self.component_mask_sha256,
            "build_descriptor_sha256": self.build_descriptor_sha256,
            "vertices": [vertex.to_dict() for vertex in self.vertices],
            "boundary_loops": [list(loop) for loop in self.boundary_loops],
            "triangles": list(self.triangles),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.content_payload(), "mesh_sha256": self.mesh_sha256}


@dataclass(frozen=True, slots=True)
class MeshBuildDiagnostic:
    component_id: str
    part_id: str
    code: Literal["degenerate_mesh"]

    def to_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "part_id": self.part_id,
            "code": self.code,
        }


@dataclass(frozen=True, slots=True)
class MeshBuildPlan:
    schema_version: str
    component_plan_sha256: str
    canvas_edge: int
    render_variant_ids: tuple[str, ...]
    final_part_ids: tuple[str, ...]
    descriptor: MeshBuildDescriptor
    sources: tuple[MeshSourceRecord, ...]
    meshes: tuple[MeshRecord, ...]
    diagnostics: tuple[MeshBuildDiagnostic, ...]
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "component_plan_sha256": self.component_plan_sha256,
            "canvas_edge": self.canvas_edge,
            "render_variant_ids": list(self.render_variant_ids),
            "final_part_ids": list(self.final_part_ids),
            "descriptor": self.descriptor.to_dict(),
            "sources": [source.to_dict() for source in self.sources],
            "meshes": [mesh.to_dict() for mesh in self.meshes],
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
        }


def build_mesh_descriptor() -> MeshBuildDescriptor:
    """Capture every dependency and option that can alter v1 mesh topology."""

    import numpy
    import scipy
    import skimage

    content = {
        "schema_version": MESH_DESCRIPTOR_VERSION,
        "contour_policy_version": MESH_CONTOUR_POLICY_VERSION,
        "sampler_version": MESH_SAMPLER_VERSION,
        "triangle_filter_version": MESH_TRIANGLE_FILTER_VERSION,
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "scipy_qhull_provider": f"scipy.spatial.Delaunay@{scipy.__version__}",
        "scikit_image_version": skimage.__version__,
        "qhull_options": MESH_QHULL_OPTIONS,
        "quantization_denominator": MESH_QUANTIZATION_DENOMINATOR,
        "symbolic_perturbation_denominator": MESH_SYMBOLIC_PERTURBATION_DENOMINATOR,
        "boundary_spacing_canvas_denominator": (
            MESH_BOUNDARY_SPACING_CANVAS_DENOMINATOR
        ),
        "interior_spacing_canvas_denominator": (
            MESH_INTERIOR_SPACING_CANVAS_DENOMINATOR
        ),
        "minimum_boundary_spacing_px": MESH_MIN_BOUNDARY_SPACING_PX,
        "minimum_interior_spacing_px": MESH_MIN_INTERIOR_SPACING_PX,
        "edge_sample_numerators": list(MESH_TRIANGLE_EDGE_SAMPLE_NUMERATORS),
        "edge_sample_denominator": MESH_TRIANGLE_EDGE_SAMPLE_DENOMINATOR,
        "maximum_circumradius_local_radius_ratio": (
            MESH_MAX_CIRCUMRADIUS_LOCAL_RADIUS_RATIO
        ),
    }
    return MeshBuildDescriptor(
        schema_version=MESH_DESCRIPTOR_VERSION,
        contour_policy_version=MESH_CONTOUR_POLICY_VERSION,
        sampler_version=MESH_SAMPLER_VERSION,
        triangle_filter_version=MESH_TRIANGLE_FILTER_VERSION,
        numpy_version=numpy.__version__,
        scipy_version=scipy.__version__,
        scipy_qhull_provider=f"scipy.spatial.Delaunay@{scipy.__version__}",
        scikit_image_version=skimage.__version__,
        qhull_options=MESH_QHULL_OPTIONS,
        quantization_denominator=MESH_QUANTIZATION_DENOMINATOR,
        symbolic_perturbation_denominator=MESH_SYMBOLIC_PERTURBATION_DENOMINATOR,
        boundary_spacing_canvas_denominator=MESH_BOUNDARY_SPACING_CANVAS_DENOMINATOR,
        interior_spacing_canvas_denominator=MESH_INTERIOR_SPACING_CANVAS_DENOMINATOR,
        minimum_boundary_spacing_px=MESH_MIN_BOUNDARY_SPACING_PX,
        minimum_interior_spacing_px=MESH_MIN_INTERIOR_SPACING_PX,
        edge_sample_numerators=MESH_TRIANGLE_EDGE_SAMPLE_NUMERATORS,
        edge_sample_denominator=MESH_TRIANGLE_EDGE_SAMPLE_DENOMINATOR,
        maximum_circumradius_local_radius_ratio=(
            MESH_MAX_CIRCUMRADIUS_LOCAL_RADIUS_RATIO
        ),
        descriptor_sha256=jcs_sha256(content),
    )


def derive_mesh_id(
    *,
    component_id: str,
    component_mask_sha256: str,
    mesh_plan_version: str = MESH_BUILD_PLAN_VERSION,
) -> str:
    """Derive a non-truncated mesh ID from the complete typed identity record."""

    if not _COMPONENT_ID_RE.fullmatch(component_id):
        raise _error("invalid_mesh_identity", "component ID is not canonical")
    if not _SHA256_RE.fullmatch(component_mask_sha256):
        raise _error("invalid_mesh_identity", "component mask digest is not canonical")
    if not isinstance(mesh_plan_version, str) or not mesh_plan_version:
        raise _error("invalid_mesh_identity", "mesh plan version is empty")
    identity = {
        "schema": MESH_COMPONENT_ID_SCHEMA,
        "component_id": component_id,
        "component_mask_sha256": component_mask_sha256,
        "mesh_plan_version": mesh_plan_version,
    }
    return "mesh/m_" + jcs_sha256(identity).removeprefix("sha256:")


def _source_record(source: MeshComponentSource) -> MeshSourceRecord:
    content = {
        "component_id": source.component_id,
        "part_id": source.part_id,
        "source_kind": source.source_kind,
        "variant_id": source.variant_id,
        "semantic_role": source.semantic_role,
        "side": source.side,
        "part_xyxy": list(source.part_xyxy),
        "bbox": list(source.bbox),
        "component_mask_sha256": source.component_mask_sha256,
        "pixel_count": source.pixel_count,
    }
    return MeshSourceRecord(
        component_id=source.component_id,
        part_id=source.part_id,
        source_kind=source.source_kind,
        variant_id=source.variant_id,
        semantic_role=source.semantic_role,
        side=source.side,
        part_xyxy=source.part_xyxy,
        bbox=source.bbox,
        component_mask_sha256=source.component_mask_sha256,
        pixel_count=source.pixel_count,
        source_sha256=jcs_sha256(content),
    )


def _quantize(value: float) -> int:
    scaled = value * MESH_QUANTIZATION_DENOMINATOR
    return int(math.floor(scaled + 0.5))


def _signed_area_q(
    first: tuple[int, int],
    second: tuple[int, int],
    third: tuple[int, int],
) -> int:
    return (second[0] - first[0]) * (third[1] - first[1]) - (
        second[1] - first[1]
    ) * (third[0] - first[0])


def _polygon_area_q(points: tuple[tuple[int, int], ...]) -> int:
    return sum(
        first[0] * second[1] - first[1] * second[0]
        for first, second in zip(points, (*points[1:], points[0]), strict=True)
    )


def _resample_closed_loop(
    points: tuple[tuple[float, float], ...],
    spacing: float,
) -> tuple[tuple[int, int], ...]:
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        return ()
    raw_area = sum(
        first[0] * second[1] - first[1] * second[0]
        for first, second in zip(points, (*points[1:], points[0]), strict=True)
    )
    if raw_area == 0.0:
        return ()
    if raw_area < 0.0:
        points = tuple(reversed(points))
    raw_start = min(
        range(len(points)),
        key=lambda index: (points[index][1], points[index][0], index),
    )
    points = (*points[raw_start:], *points[:raw_start])
    segments = tuple(
        math.hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(points, (*points[1:], points[0]), strict=True)
    )
    perimeter = sum(segments)
    if not math.isfinite(perimeter) or perimeter <= 0.0:
        return ()
    count = max(4, int(math.ceil(perimeter / spacing)))
    sampled: list[tuple[int, int]] = []
    segment_index = 0
    traversed = 0.0
    for sample_index in range(count):
        target = sample_index * perimeter / count
        while (
            segment_index + 1 < len(segments)
            and traversed + segments[segment_index] < target
        ):
            traversed += segments[segment_index]
            segment_index += 1
        first = points[segment_index]
        second = points[(segment_index + 1) % len(points)]
        length = segments[segment_index]
        fraction = 0.0 if length == 0.0 else (target - traversed) / length
        coordinate = (
            first[0] + fraction * (second[0] - first[0]),
            first[1] + fraction * (second[1] - first[1]),
        )
        quantized = (_quantize(coordinate[0]), _quantize(coordinate[1]))
        if not sampled or sampled[-1] != quantized:
            sampled.append(quantized)
    if len(sampled) > 1 and sampled[0] == sampled[-1]:
        sampled.pop()
    if len(set(sampled)) < 3:
        return ()
    loop = tuple(sampled)
    area = _polygon_area_q(loop)
    if area == 0:
        return ()
    if area < 0:
        loop = tuple(reversed(loop))
    start = min(range(len(loop)), key=lambda index: (loop[index][1], loop[index][0], index))
    return (*loop[start:], *loop[:start])


def _boundary_loops(
    source: MeshComponentSource,
    *,
    canvas_edge: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    import numpy as np
    from skimage.measure import find_contours

    mask = np.frombuffer(source.binary_mask_u8, dtype=np.uint8).reshape(
        source.height,
        source.width,
    )
    padded = np.pad(mask, 1, mode="constant")
    contours = find_contours(
        padded,
        level=0.5,
        fully_connected="high",
        positive_orientation="high",
    )
    spacing = max(
        float(MESH_MIN_BOUNDARY_SPACING_PX),
        canvas_edge / MESH_BOUNDARY_SPACING_CANVAS_DENOMINATOR,
    )
    x1, y1 = source.bbox[:2]
    loops: list[tuple[tuple[int, int], ...]] = []
    for contour in contours:
        mapped = tuple(
            (x1 + float(column) - 0.5, y1 + float(row) - 0.5)
            for row, column in contour
        )
        loop = _resample_closed_loop(mapped, spacing)
        if loop:
            loops.append(loop)
    loops.sort(
        key=lambda loop: (
            min(point[1] for point in loop),
            min(point[0] for point in loop),
            len(loop),
            loop,
        )
    )
    return tuple(loops)


def _axis_candidates(value: float, size: int) -> tuple[int, ...]:
    if value < -1e-9 or value > size + 1e-9:
        return ()
    floor_value = math.floor(value)
    candidates = {floor_value}
    if math.isclose(value, round(value), abs_tol=1e-9):
        candidates.add(int(round(value)) - 1)
    return tuple(sorted(candidate for candidate in candidates if 0 <= candidate < size))


def _mask_contains_local(mask, x: float, y: float) -> bool:
    return any(
        bool(mask[row, column])
        for row in _axis_candidates(y, mask.shape[0])
        for column in _axis_candidates(x, mask.shape[1])
    )


def _interior_points(
    source: MeshComponentSource,
    *,
    canvas_edge: int,
    boundary_points: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    import numpy as np

    mask = np.frombuffer(source.binary_mask_u8, dtype=np.uint8).reshape(
        source.height,
        source.width,
    )
    spacing = max(
        float(MESH_MIN_INTERIOR_SPACING_PX),
        canvas_edge / MESH_INTERIOR_SPACING_CANVAS_DENOMINATOR,
    )
    x1, y1 = source.bbox[:2]
    points: set[tuple[int, int]] = set()
    local_y = spacing / 2.0
    while local_y < source.height:
        local_x = spacing / 2.0
        while local_x < source.width:
            if _mask_contains_local(mask, local_x, local_y):
                point = (_quantize(x1 + local_x), _quantize(y1 + local_y))
                if point not in boundary_points:
                    points.add(point)
            local_x += spacing
        local_y += spacing
    return tuple(sorted(points, key=lambda point: (point[1], point[0])))


def _symbolic_perturbation(component_id: str, rank: int) -> tuple[float, float]:
    digest = hashlib.sha256(f"{component_id}:{rank}".encode("ascii")).digest()
    scale = 1.0 / (MESH_SYMBOLIC_PERTURBATION_DENOMINATOR * 4.0)
    x_unit = int.from_bytes(digest[:4], "big") / 2**32
    y_unit = int.from_bytes(digest[4:8], "big") / 2**32
    return ((2.0 * x_unit - 1.0) * scale, (2.0 * y_unit - 1.0) * scale)


def _run_delaunay(points):
    import numpy as np
    from scipy.spatial import Delaunay

    return np.asarray(
        Delaunay(points, qhull_options=MESH_QHULL_OPTIONS).simplices,
        dtype=np.int64,
    )


def _triangle_samples(
    points: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    first, second, third = points
    samples = [
        (
            (first[0] + second[0] + third[0]) / 3.0,
            (first[1] + second[1] + third[1]) / 3.0,
        )
    ]
    for start, end in ((first, second), (second, third), (third, first)):
        for numerator in MESH_TRIANGLE_EDGE_SAMPLE_NUMERATORS:
            fraction = numerator / MESH_TRIANGLE_EDGE_SAMPLE_DENOMINATOR
            samples.append(
                (
                    start[0] + fraction * (end[0] - start[0]),
                    start[1] + fraction * (end[1] - start[1]),
                )
            )
    return tuple(samples)


def _circumradius(points) -> float:
    first, second, third = points
    lengths = (
        math.dist(first, second),
        math.dist(second, third),
        math.dist(third, first),
    )
    twice_area = abs(
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )
    if twice_area == 0.0:
        return math.inf
    return lengths[0] * lengths[1] * lengths[2] / (2.0 * twice_area)


def _canonical_triangle(indices: tuple[int, int, int]) -> tuple[int, int, int]:
    minimum = min(range(3), key=indices.__getitem__)
    return (*indices[minimum:], *indices[:minimum])


def _build_component_mesh(
    source: MeshComponentSource,
    *,
    canvas_edge: int,
    descriptor: MeshBuildDescriptor,
) -> MeshRecord | None:
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import QhullError

    loops_q = _boundary_loops(source, canvas_edge=canvas_edge)
    boundary_q: list[tuple[int, int]] = []
    loop_coordinates: list[tuple[tuple[int, int], ...]] = []
    seen_boundary: set[tuple[int, int]] = set()
    for loop in loops_q:
        unique_loop: list[tuple[int, int]] = []
        for point in loop:
            if point not in seen_boundary:
                seen_boundary.add(point)
                boundary_q.append(point)
                unique_loop.append(point)
        if len(unique_loop) >= 3:
            loop_coordinates.append(tuple(unique_loop))
    interior_q = tuple(
        sorted(
            set(
                _interior_points(
                    source,
                    canvas_edge=canvas_edge,
                    boundary_points=seen_boundary,
                )
            ),
            key=lambda point: (point[1], point[0]),
        )
    )
    points_q = tuple(boundary_q) + interior_q
    if len(points_q) < 3:
        return None
    points = tuple(
        (x / MESH_QUANTIZATION_DENOMINATOR, y / MESH_QUANTIZATION_DENOMINATOR)
        for x, y in points_q
    )
    topology_points = np.asarray(
        [
            (
                point[0] + _symbolic_perturbation(source.component_id, rank)[0],
                point[1] + _symbolic_perturbation(source.component_id, rank)[1],
            )
            for rank, point in enumerate(points)
        ],
        dtype=np.float64,
    )
    try:
        if len(points) == 3:
            simplexes = np.asarray(((0, 1, 2),), dtype=np.int64)
        else:
            simplexes = _run_delaunay(topology_points)
    except QhullError:
        return None

    mask = np.frombuffer(source.binary_mask_u8, dtype=np.uint8).reshape(
        source.height,
        source.width,
    )
    radii = distance_transform_edt(mask)
    bbox_x, bbox_y = source.bbox[:2]
    minimum_radius = max(
        0.5,
        canvas_edge / MESH_BOUNDARY_SPACING_CANVAS_DENOMINATOR / 2.0,
    )
    accepted: set[tuple[int, int, int]] = set()
    for raw in simplexes:
        indices = tuple(int(index) for index in raw)
        if len(set(indices)) != 3 or any(index < 0 or index >= len(points) for index in indices):
            continue
        area_q = _signed_area_q(*(points_q[index] for index in indices))
        if area_q == 0:
            continue
        if area_q < 0:
            indices = (indices[0], indices[2], indices[1])
        triangle_points = tuple(points[index] for index in indices)
        samples = _triangle_samples(triangle_points)
        local_samples = tuple((x - bbox_x, y - bbox_y) for x, y in samples)
        if not all(_mask_contains_local(mask, x, y) for x, y in local_samples):
            continue
        centroid_x, centroid_y = local_samples[0]
        columns = _axis_candidates(centroid_x, source.width)
        rows = _axis_candidates(centroid_y, source.height)
        local_radius = max(
            (float(radii[row, column]) for row in rows for column in columns),
            default=0.0,
        )
        if (
            _circumradius(triangle_points)
            > descriptor.maximum_circumradius_local_radius_ratio
            * max(minimum_radius, local_radius)
        ):
            continue
        accepted.add(_canonical_triangle(indices))
    if not accepted:
        return None

    used_old = tuple(sorted({index for triangle in accepted for index in triangle}))
    remap = {old: new for new, old in enumerate(used_old)}
    compact_points = tuple(points[index] for index in used_old)
    compact_triangles = tuple(
        sorted(
            _canonical_triangle(tuple(remap[index] for index in triangle))
            for triangle in accepted
        )
    )
    compact_loops: list[tuple[int, ...]] = []
    boundary_metadata: dict[int, tuple[int, int]] = {}
    coordinate_to_old = {coordinate: index for index, coordinate in enumerate(points_q)}
    for loop in loop_coordinates:
        compact = tuple(
            remap[coordinate_to_old[coordinate]]
            for coordinate in loop
            if coordinate_to_old[coordinate] in remap
        )
        if len(compact) < 3:
            continue
        loop_index = len(compact_loops)
        compact_loops.append(compact)
        for order, index in enumerate(compact):
            boundary_metadata.setdefault(index, (loop_index, order))

    part_x1, part_y1, part_x2, part_y2 = source.part_xyxy
    part_width = part_x2 - part_x1
    part_height = part_y2 - part_y1
    vertices = tuple(
        MeshVertex(
            position=position,
            uv=(
                (position[0] - part_x1) / part_width,
                (position[1] - part_y1) / part_height,
            ),
            boundary=index in boundary_metadata,
            boundary_loop=(boundary_metadata[index][0] if index in boundary_metadata else None),
            boundary_order=(boundary_metadata[index][1] if index in boundary_metadata else None),
        )
        for index, position in enumerate(compact_points)
    )
    flat_triangles = tuple(index for triangle in compact_triangles for index in triangle)
    mesh_id = derive_mesh_id(
        component_id=source.component_id,
        component_mask_sha256=source.component_mask_sha256,
    )
    content = {
        "mesh_id": mesh_id,
        "component_id": source.component_id,
        "part_id": source.part_id,
        "source_kind": source.source_kind,
        "variant_id": source.variant_id,
        "side": source.side,
        "component_bbox": list(source.bbox),
        "component_mask_sha256": source.component_mask_sha256,
        "build_descriptor_sha256": descriptor.descriptor_sha256,
        "vertices": [vertex.to_dict() for vertex in vertices],
        "boundary_loops": [list(loop) for loop in compact_loops],
        "triangles": list(flat_triangles),
    }
    return MeshRecord(
        mesh_id=mesh_id,
        component_id=source.component_id,
        part_id=source.part_id,
        source_kind=source.source_kind,
        variant_id=source.variant_id,
        side=source.side,
        component_bbox=source.bbox,
        component_mask_sha256=source.component_mask_sha256,
        build_descriptor_sha256=descriptor.descriptor_sha256,
        vertices=vertices,
        boundary_loops=tuple(compact_loops),
        triangles=flat_triangles,
        mesh_sha256=jcs_sha256(content),
    )


def _validate_descriptor(descriptor: MeshBuildDescriptor) -> None:
    expected = build_mesh_descriptor()
    if descriptor != expected:
        raise _error("invalid_mesh_plan", "mesh build descriptor differs from v1")


def validate_mesh_plan(plan: MeshBuildPlan) -> MeshBuildPlan:
    """Validate the reference closure and canonical topology of a mesh plan."""

    if not isinstance(plan, MeshBuildPlan) or plan.schema_version != MESH_BUILD_PLAN_VERSION:
        raise _error("invalid_mesh_plan", "unsupported MeshBuildPlan")
    if plan.plan_sha256 != jcs_sha256(plan.semantic_payload()):
        raise _error("invalid_mesh_plan", "MeshBuildPlan digest mismatch")
    _validate_descriptor(plan.descriptor)
    if type(plan.canvas_edge) is not int or plan.canvas_edge <= 0:
        raise _error("invalid_mesh_plan", "mesh canvas is invalid")
    if plan.render_variant_ids != tuple(sorted(set(plan.render_variant_ids))):
        raise _error("invalid_mesh_plan", "render variant IDs are not canonical")
    if plan.final_part_ids != tuple(sorted(set(plan.final_part_ids))):
        raise _error("invalid_mesh_plan", "final Part IDs are not canonical")
    source_ids = tuple(source.component_id for source in plan.sources)
    if source_ids != tuple(sorted(set(source_ids))):
        raise _error("invalid_mesh_plan", "mesh source IDs are not canonical")
    source_by_id = {source.component_id: source for source in plan.sources}
    if {source.part_id for source in plan.sources} != set(plan.final_part_ids):
        raise _error("invalid_mesh_plan", "mesh sources differ from the final Part set")
    for source in plan.sources:
        if source.source_sha256 != jcs_sha256(source.content_payload()):
            raise _error("invalid_mesh_plan", "mesh source digest mismatch")
        if source.source_kind not in {"see_through", "native_variant"}:
            raise _error("invalid_mesh_plan", "mesh source kind is invalid")
        if source.source_kind == "native_variant" and source.variant_id not in plan.render_variant_ids:
            raise _error("invalid_mesh_plan", "mesh source uses an unadmitted variant")
    mesh_ids = tuple(mesh.component_id for mesh in plan.meshes)
    if mesh_ids != tuple(sorted(set(mesh_ids))):
        raise _error("invalid_mesh_plan", "mesh records are not canonical")
    diagnostic_ids = tuple(diagnostic.component_id for diagnostic in plan.diagnostics)
    if diagnostic_ids != tuple(sorted(set(diagnostic_ids))):
        raise _error("invalid_mesh_plan", "mesh diagnostics are not canonical")
    if set(mesh_ids) & set(diagnostic_ids) or set(mesh_ids) | set(diagnostic_ids) != set(source_ids):
        raise _error("invalid_mesh_plan", "mesh results do not cover each source exactly once")
    for diagnostic in plan.diagnostics:
        source = source_by_id.get(diagnostic.component_id)
        if source is None or diagnostic.part_id != source.part_id or diagnostic.code != "degenerate_mesh":
            raise _error("invalid_mesh_plan", "mesh diagnostic is invalid")
    for mesh in plan.meshes:
        source = source_by_id.get(mesh.component_id)
        if source is None:
            raise _error("invalid_mesh_plan", "mesh references an unknown component")
        if (
            mesh.part_id != source.part_id
            or mesh.source_kind != source.source_kind
            or mesh.variant_id != source.variant_id
            or mesh.side != source.side
            or mesh.component_bbox != source.bbox
            or mesh.component_mask_sha256 != source.component_mask_sha256
            or mesh.build_descriptor_sha256 != plan.descriptor.descriptor_sha256
            or mesh.mesh_id
            != derive_mesh_id(
                component_id=mesh.component_id,
                component_mask_sha256=mesh.component_mask_sha256,
            )
            or mesh.mesh_sha256 != jcs_sha256(mesh.content_payload())
        ):
            raise _error("invalid_mesh_plan", "mesh provenance differs from its source")
        if len(mesh.vertices) < 3 or not mesh.triangles or len(mesh.triangles) % 3:
            raise _error("invalid_mesh_plan", "mesh topology is empty")
        boundary_sequence = tuple(vertex.boundary for vertex in mesh.vertices)
        if boundary_sequence != tuple(sorted(boundary_sequence, reverse=True)):
            raise _error("invalid_mesh_plan", "mesh vertices are not boundary-first")
        for vertex in mesh.vertices:
            if (
                not all(math.isfinite(value) for value in (*vertex.position, *vertex.uv))
                or any(
                    not math.isclose(
                        value * MESH_QUANTIZATION_DENOMINATOR,
                        round(value * MESH_QUANTIZATION_DENOMINATOR),
                        abs_tol=1e-9,
                    )
                    for value in vertex.position
                )
                or any(value < 0.0 or value > 1.0 for value in vertex.uv)
            ):
                raise _error("invalid_mesh_plan", "mesh vertex is invalid")
            if vertex.boundary != (vertex.boundary_loop is not None):
                raise _error("invalid_mesh_plan", "mesh boundary metadata is inconsistent")
            if vertex.boundary != (vertex.boundary_order is not None):
                raise _error("invalid_mesh_plan", "mesh boundary order is inconsistent")
        loop_keys: list[tuple[object, ...]] = []
        loop_vertex_indices: set[int] = set()
        for loop_index, loop in enumerate(mesh.boundary_loops):
            if len(loop) < 3 or len(set(loop)) != len(loop):
                raise _error("invalid_mesh_plan", "mesh boundary loop is invalid")
            if any(index < 0 or index >= len(mesh.vertices) for index in loop):
                raise _error("invalid_mesh_plan", "mesh boundary index is out of range")
            if any(
                mesh.vertices[index].boundary_loop != loop_index
                or mesh.vertices[index].boundary_order != order
                for order, index in enumerate(loop)
            ):
                raise _error("invalid_mesh_plan", "mesh boundary loop metadata differs")
            coordinates_q = tuple(
                tuple(_quantize(value) for value in mesh.vertices[index].position)
                for index in loop
            )
            start_key = (coordinates_q[0][1], coordinates_q[0][0], loop[0])
            if start_key != min(
                (coordinate[1], coordinate[0], index)
                for coordinate, index in zip(coordinates_q, loop, strict=True)
            ):
                raise _error("invalid_mesh_plan", "mesh boundary start is not canonical")
            if _polygon_area_q(coordinates_q) <= 0:
                raise _error("invalid_mesh_plan", "mesh boundary winding is not canonical")
            loop_keys.append(
                (
                    min(coordinate[1] for coordinate in coordinates_q),
                    min(coordinate[0] for coordinate in coordinates_q),
                    len(loop),
                    coordinates_q,
                )
            )
            loop_vertex_indices.update(loop)
        if loop_keys != sorted(loop_keys):
            raise _error("invalid_mesh_plan", "mesh boundary loops are not canonical")
        if loop_vertex_indices != {
            index for index, vertex in enumerate(mesh.vertices) if vertex.boundary
        }:
            raise _error("invalid_mesh_plan", "mesh boundary flags differ from loops")
        triangles = tuple(
            tuple(mesh.triangles[offset : offset + 3])
            for offset in range(0, len(mesh.triangles), 3)
        )
        if triangles != tuple(sorted(set(triangles))):
            raise _error("invalid_mesh_plan", "mesh triangles are not canonical")
        used: set[int] = set()
        for triangle in triangles:
            if len(set(triangle)) != 3 or any(
                index < 0 or index >= len(mesh.vertices) for index in triangle
            ):
                raise _error("invalid_mesh_plan", "mesh triangle index is invalid")
            points_q = tuple(
                tuple(_quantize(value) for value in mesh.vertices[index].position)
                for index in triangle
            )
            if _signed_area_q(*points_q) <= 0 or triangle != _canonical_triangle(triangle):
                raise _error("invalid_mesh_plan", "mesh triangle winding/order is invalid")
            used.update(triangle)
        if used != set(range(len(mesh.vertices))):
            raise _error("invalid_mesh_plan", "mesh contains unused vertices")
    return plan


def build_mesh_plan(
    component_plan: MaskComponentPlan,
    *,
    item_root: str | Path,
    render_variant_ids: Iterable[str],
    final_part_ids: Iterable[str],
) -> MeshBuildPlan:
    """Build deterministic component-local meshes from authenticated A labels."""

    render_ids = tuple(render_variant_ids)
    final_ids_input = tuple(final_part_ids)
    if len(render_ids) != len(set(render_ids)):
        raise _error("invalid_mesh_input", "render variant IDs are not unique")
    if len(final_ids_input) != len(set(final_ids_input)):
        raise _error("invalid_mesh_input", "final Part IDs are not unique")
    render_ids = tuple(sorted(render_ids))
    final_ids = tuple(sorted(final_ids_input))
    sources = tuple(
        sorted(
            load_mesh_component_sources(
                component_plan,
                item_root=item_root,
                render_variant_ids=render_ids,
            ),
            key=lambda source: source.component_id,
        )
    )
    if {source.part_id for source in sources} != set(final_ids):
        raise _error("invalid_mesh_input", "final Part set differs from admitted component sources")
    if any(source.schema_version != MESH_COMPONENT_SOURCE_VERSION for source in sources):
        raise _error("invalid_mesh_input", "component source version is unsupported")
    descriptor = build_mesh_descriptor()
    source_records = tuple(_source_record(source) for source in sources)
    meshes: list[MeshRecord] = []
    diagnostics: list[MeshBuildDiagnostic] = []
    for source in sources:
        mesh = _build_component_mesh(
            source,
            canvas_edge=component_plan.canvas_edge,
            descriptor=descriptor,
        )
        if mesh is None:
            diagnostics.append(
                MeshBuildDiagnostic(
                    component_id=source.component_id,
                    part_id=source.part_id,
                    code="degenerate_mesh",
                )
            )
        else:
            meshes.append(mesh)
    meshes.sort(key=lambda mesh: mesh.component_id)
    diagnostics.sort(key=lambda diagnostic: diagnostic.component_id)
    content = {
        "schema_version": MESH_BUILD_PLAN_VERSION,
        "component_plan_sha256": component_plan.plan_sha256,
        "canvas_edge": component_plan.canvas_edge,
        "render_variant_ids": list(render_ids),
        "final_part_ids": list(final_ids),
        "descriptor": descriptor.to_dict(),
        "sources": [source.to_dict() for source in source_records],
        "meshes": [mesh.to_dict() for mesh in meshes],
        "diagnostics": [diagnostic.to_dict() for diagnostic in diagnostics],
    }
    plan = MeshBuildPlan(
        schema_version=MESH_BUILD_PLAN_VERSION,
        component_plan_sha256=component_plan.plan_sha256,
        canvas_edge=component_plan.canvas_edge,
        render_variant_ids=render_ids,
        final_part_ids=final_ids,
        descriptor=descriptor,
        sources=source_records,
        meshes=tuple(meshes),
        diagnostics=tuple(diagnostics),
        plan_sha256=jcs_sha256(content),
    )
    return validate_mesh_plan(plan)


__all__ = [
    "MESH_BOUNDARY_SPACING_CANVAS_DENOMINATOR",
    "MESH_BUILD_PLAN_VERSION",
    "MESH_COMPONENT_ID_SCHEMA",
    "MESH_CONTOUR_POLICY_VERSION",
    "MESH_DESCRIPTOR_VERSION",
    "MESH_INTERIOR_SPACING_CANVAS_DENOMINATOR",
    "MESH_MAX_CIRCUMRADIUS_LOCAL_RADIUS_RATIO",
    "MESH_MIN_BOUNDARY_SPACING_PX",
    "MESH_MIN_INTERIOR_SPACING_PX",
    "MESH_QHULL_OPTIONS",
    "MESH_QUANTIZATION_DENOMINATOR",
    "MESH_SAMPLER_VERSION",
    "MESH_SYMBOLIC_PERTURBATION_DENOMINATOR",
    "MESH_TRIANGLE_EDGE_SAMPLE_DENOMINATOR",
    "MESH_TRIANGLE_EDGE_SAMPLE_NUMERATORS",
    "MESH_TRIANGLE_FILTER_VERSION",
    "MeshBuildDescriptor",
    "MeshBuildDiagnostic",
    "MeshBuildError",
    "MeshBuildPlan",
    "MeshRecord",
    "MeshSourceRecord",
    "MeshVertex",
    "build_mesh_plan",
    "build_mesh_descriptor",
    "derive_mesh_id",
    "validate_mesh_plan",
]
