from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

import module.auto_rig.component_plan as component_plan_module
import module.auto_rig.mesh_builder as mesh_builder_module
from module.auto_rig.component_geometry import load_mesh_component_sources
from module.auto_rig.component_plan import build_mask_component_plan
from module.auto_rig.jcs import jcs_sha256
from module.auto_rig.mesh_builder import (
    MESH_BOUNDARY_SPACING_CANVAS_DENOMINATOR,
    MESH_BUILD_PLAN_VERSION,
    MESH_COMPONENT_ID_SCHEMA,
    MESH_INTERIOR_SPACING_CANVAS_DENOMINATOR,
    MESH_QHULL_OPTIONS,
    MESH_QUANTIZATION_DENOMINATOR,
    MESH_SYMBOLIC_PERTURBATION_DENOMINATOR,
    build_mesh_descriptor,
    build_mesh_plan,
    derive_mesh_id,
    validate_mesh_plan,
)
from tests.test_auto_rig_component_plan import _loaded_part


def test_mesh_descriptor_freezes_topology_affecting_dependencies_and_options() -> None:
    descriptor = build_mesh_descriptor()

    assert MESH_BUILD_PLAN_VERSION == "mesh-build-plan-v1"
    assert MESH_COMPONENT_ID_SCHEMA == "mesh-component-id-v1"
    assert MESH_QUANTIZATION_DENOMINATOR == 256
    assert MESH_SYMBOLIC_PERTURBATION_DENOMINATOR == 4096
    assert MESH_BOUNDARY_SPACING_CANVAS_DENOMINATOR == 512
    assert MESH_INTERIOR_SPACING_CANVAS_DENOMINATOR == 128
    assert "QJ" not in MESH_QHULL_OPTIONS.split()
    assert descriptor.qhull_options == MESH_QHULL_OPTIONS
    assert descriptor.quantization_denominator == MESH_QUANTIZATION_DENOMINATOR
    assert descriptor.symbolic_perturbation_denominator == MESH_SYMBOLIC_PERTURBATION_DENOMINATOR
    assert descriptor.scipy_version == "1.15.3"
    assert descriptor.scikit_image_version == "0.25.2"
    assert descriptor.descriptor_sha256 == jcs_sha256(descriptor.content_payload())


def test_mesh_id_uses_the_full_typed_component_identity() -> None:
    component_id = "component/c_" + "1" * 64
    mask_sha256 = "sha256:" + "2" * 64

    mesh_id = derive_mesh_id(
        component_id=component_id,
        component_mask_sha256=mask_sha256,
    )

    assert re.fullmatch(r"mesh/m_[0-9a-f]{64}", mesh_id)
    assert mesh_id != derive_mesh_id(
        component_id="component/c_" + "3" * 64,
        component_mask_sha256=mask_sha256,
    )
    assert mesh_id != derive_mesh_id(
        component_id=component_id,
        component_mask_sha256="sha256:" + "4" * 64,
    )
    assert mesh_id != derive_mesh_id(
        component_id=component_id,
        component_mask_sha256=mask_sha256,
        mesh_plan_version="mesh-build-plan-v2",
    )


def _component_plan(
    tmp_path: Path,
    *,
    points: set[tuple[int, int]],
    xyxy: tuple[int, int, int, int] = (10, 20, 30, 40),
    canvas_edge: int = 64,
):
    loaded = _loaded_part(
        source_tag="objects",
        base_tag="objects",
        semantic_slug="objects",
        part_id="part/objects",
        side=None,
        xyxy=xyxy,
        points=points,
    )
    return build_mask_component_plan((loaded,), canvas_edge=canvas_edge, item_root=tmp_path)


def _filled_rectangle(x1: int, y1: int, x2: int, y2: int) -> set[tuple[int, int]]:
    return {(x, y) for y in range(y1, y2) for x in range(x1, x2)}


def _signed_area(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def test_mesh_plan_builds_a_canonical_quantized_rectangle(tmp_path: Path) -> None:
    component_plan = _component_plan(
        tmp_path,
        points=_filled_rectangle(2, 3, 15, 12),
    )

    plan = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=("part/objects",),
    )

    assert validate_mesh_plan(plan) is plan
    assert plan.component_plan_sha256 == component_plan.plan_sha256
    assert len(plan.meshes) == 1
    assert plan.diagnostics == ()
    mesh = plan.meshes[0]
    assert mesh.component_id == component_plan.parts[0].components[0].component_id
    assert mesh.mesh_id == derive_mesh_id(
        component_id=mesh.component_id,
        component_mask_sha256=mesh.component_mask_sha256,
    )
    assert mesh.vertices
    assert len({vertex.position for vertex in mesh.vertices}) == len(mesh.vertices)
    assert len(mesh.triangles) >= 3
    assert len(mesh.triangles) % 3 == 0
    assert tuple(mesh.triangles) == tuple(
        index
        for triangle in sorted(tuple(mesh.triangles[offset : offset + 3]) for offset in range(0, len(mesh.triangles), 3))
        for index in triangle
    )
    assert all(
        float(coordinate * MESH_QUANTIZATION_DENOMINATOR).is_integer() for vertex in mesh.vertices for coordinate in vertex.position
    )
    assert all(0.0 <= value <= 1.0 for vertex in mesh.vertices for value in vertex.uv)
    assert tuple(vertex.boundary for vertex in mesh.vertices) == tuple(
        sorted((vertex.boundary for vertex in mesh.vertices), reverse=True)
    )
    for offset in range(0, len(mesh.triangles), 3):
        a, b, c = (mesh.vertices[index].position for index in mesh.triangles[offset : offset + 3])
        assert _signed_area(a, b, c) > 0.0


def test_mesh_plan_handles_near_collinear_and_cocircular_samples_without_duplicates(
    tmp_path: Path,
) -> None:
    thin_l = _filled_rectangle(1, 1, 18, 3) | _filled_rectangle(16, 3, 18, 10)
    component_plan = _component_plan(tmp_path, points=thin_l)

    first = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=("part/objects",),
    )
    second = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=("part/objects",),
    )

    assert first == second
    assert len(first.meshes) == 1
    positions = tuple(vertex.position for vertex in first.meshes[0].vertices)
    assert len(positions) == len(set(positions))
    assert all(
        abs(delta) < 1 / MESH_SYMBOLIC_PERTURBATION_DENOMINATOR
        for rank in range(len(positions))
        for delta in mesh_builder_module._symbolic_perturbation(
            first.meshes[0].component_id,
            rank,
        )
    )


def _mesh_uncovered_source_pixels(source, mesh):
    triangles = tuple(
        tuple(mesh.vertices[index].position for index in mesh.triangles[offset : offset + 3])
        for offset in range(0, len(mesh.triangles), 3)
    )
    uncovered = []
    for row in range(source.height):
        for column in range(source.width):
            if source.binary_mask_u8[row * source.width + column] != 1:
                continue
            point = (
                source.bbox[0] + column + 0.5,
                source.bbox[1] + row + 0.5,
            )
            if not any(
                all(
                    _signed_area(first, second, point) >= -1e-9
                    for first, second in zip(
                        triangle,
                        (*triangle[1:], triangle[0]),
                        strict=True,
                    )
                )
                for triangle in triangles
            ):
                uncovered.append((column, row))
    return tuple(uncovered)


@pytest.mark.parametrize(
    "points",
    (
        (_filled_rectangle(1, 1, 12, 12) - _filled_rectangle(5, 4, 12, 9)),
        (_filled_rectangle(1, 1, 14, 14) - _filled_rectangle(5, 5, 10, 10)),
    ),
    ids=("concave-c", "hole"),
)
def test_mesh_plan_covers_every_source_alpha_pixel(
    tmp_path: Path,
    points: set[tuple[int, int]],
) -> None:
    component_plan = _component_plan(tmp_path, points=points)
    sources = load_mesh_component_sources(component_plan, item_root=tmp_path)

    plan = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=("part/objects",),
    )

    assert _mesh_uncovered_source_pixels(sources[0], plan.meshes[0]) == ()


def test_mesh_plan_reindexes_demoted_hole_vertices_boundary_first(
    tmp_path: Path,
) -> None:
    points = _filled_rectangle(1, 1, 63, 63)
    points -= _filled_rectangle(23, 26, 26, 28)
    points -= _filled_rectangle(27, 28, 36, 33)
    component_plan = _component_plan(
        tmp_path,
        points=points,
        xyxy=(0, 0, 64, 64),
        canvas_edge=1280,
    )

    plan = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=("part/objects",),
    )

    mesh = plan.meshes[0]
    assert tuple(vertex.boundary for vertex in mesh.vertices) == tuple(
        sorted((vertex.boundary for vertex in mesh.vertices), reverse=True)
    )


def test_mesh_plan_recanonicalizes_a_filtered_boundary_loop_start(
    tmp_path: Path,
) -> None:
    rows = (
        "......................##########..",
        "......................#########...",
        ".....................##########...",
        ".....................##########...",
        "....................##########....",
        "....................##########....",
        "........#..........###########....",
        "...#....#.........###########.....",
        "...#.###..##......###########.....",
        "...#.####.#.......##########......",
        "...#..####.###...###########......",
        "...#...####.#...###########.......",
        "...###..####...############.......",
        "...####.#####..###########........",
        ".....###..###############.......##",
        "....#####..#######################",
        "...#######..######################",
        "...#####.#########################",
        ".######....######################.",
        ".###.....#########...####.........",
        ".#.......########.................",
        ".......########...................",
        ".....#########....................",
        "....#########.....................",
    )
    points = {(x, y) for y, row in enumerate(rows) for x, value in enumerate(row) if value == "#"}
    component_plan = _component_plan(
        tmp_path,
        points=points,
        xyxy=(0, 0, len(rows[0]), len(rows)),
        canvas_edge=1280,
    )

    plan = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=("part/objects",),
    )

    assert validate_mesh_plan(plan) is plan


def test_mesh_plan_resorts_filtered_boundary_loops(
    tmp_path: Path,
) -> None:
    rows = tuple(
        row[18:]
        for row in (
            ".......................#########################.",
            ".......................##########################",
            ".......................##########################",
            ".....................############################",
            ".....................############################",
            "...................#.###########################.",
            "...................#.###########################.",
            "...................#.##########################..",
            "...................#.##########################..",
            "...................#.#########################...",
            "...................#.########################.#..",
            "...................#.####.###################.#..",
            "...................#.####.##################..#..",
            "...................######.#####################..",
            "...................#####.######################..",
            "...................#####.######################..",
            "..................######.####################....",
        )
    )
    points = {(x, y) for y, row in enumerate(rows) for x, value in enumerate(row) if value == "#"}
    component_plan = _component_plan(
        tmp_path,
        points=points,
        xyxy=(0, 0, len(rows[0]), len(rows)),
        canvas_edge=1280,
    )

    plan = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=("part/objects",),
    )

    assert validate_mesh_plan(plan) is plan


def test_mesh_plan_never_connects_two_frozen_components(tmp_path: Path) -> None:
    points = _filled_rectangle(1, 1, 5, 5) | _filled_rectangle(12, 10, 17, 14)
    component_plan = _component_plan(tmp_path, points=points)
    sources = load_mesh_component_sources(component_plan, item_root=tmp_path)

    plan = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=("part/objects",),
    )

    assert len(sources) == 2
    assert len(plan.meshes) == 2
    source_by_id = {source.component_id: source for source in sources}
    for mesh in plan.meshes:
        source = source_by_id[mesh.component_id]
        x1, y1, x2, y2 = source.bbox
        assert all(x1 <= vertex.position[0] <= x2 and y1 <= vertex.position[1] <= y2 for vertex in mesh.vertices)
        assert _mesh_uncovered_source_pixels(source, mesh) == ()


def test_mesh_plan_builds_a_thin_diagonal_without_qhull_joggle(
    tmp_path: Path,
) -> None:
    component_plan = _component_plan(
        tmp_path,
        points={(1, 1), (2, 2), (3, 3), (4, 4)},
    )

    plan = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=("part/objects",),
    )

    source = load_mesh_component_sources(component_plan, item_root=tmp_path)[0]
    assert len(plan.meshes) == 1
    assert plan.diagnostics == ()
    assert _mesh_uncovered_source_pixels(source, plan.meshes[0]) == ()
    assert "QJ" not in plan.descriptor.qhull_options.split()


def test_mesh_plan_is_independent_of_component_source_iteration_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = _filled_rectangle(1, 1, 5, 5) | _filled_rectangle(12, 10, 17, 14)
    component_plan = _component_plan(tmp_path, points=points)
    expected = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=("part/objects",),
    )
    original_loader = mesh_builder_module.load_mesh_component_sources

    def reversed_loader(*args, **kwargs):
        return tuple(reversed(original_loader(*args, **kwargs)))

    monkeypatch.setattr(
        mesh_builder_module,
        "load_mesh_component_sources",
        reversed_loader,
    )

    actual = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=("part/objects",),
    )

    assert actual == expected


def test_mesh_plan_is_independent_of_contour_iteration_and_winding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skimage import measure

    points = _filled_rectangle(1, 1, 14, 14) - _filled_rectangle(5, 5, 10, 10)
    component_plan = _component_plan(tmp_path, points=points)
    expected = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=("part/objects",),
    )
    original_find_contours = measure.find_contours

    def reversed_contours(*args, **kwargs):
        import numpy as np

        altered = []
        for contour in reversed(original_find_contours(*args, **kwargs)):
            core = contour[:-1] if np.array_equal(contour[0], contour[-1]) else contour
            core = np.roll(core[::-1], 3, axis=0)
            altered.append(np.vstack((core, core[0])))
        return altered

    monkeypatch.setattr(measure, "find_contours", reversed_contours)

    actual = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=("part/objects",),
    )

    assert actual == expected


def test_mesh_plan_is_independent_of_qhull_simplex_order_and_winding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component_plan = _component_plan(
        tmp_path,
        points=_filled_rectangle(1, 1, 14, 14),
    )
    expected = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=("part/objects",),
    )
    original_run_delaunay = mesh_builder_module._run_delaunay

    def reversed_simplexes(points):
        return original_run_delaunay(points)[::-1, ::-1]

    monkeypatch.setattr(mesh_builder_module, "_run_delaunay", reversed_simplexes)

    actual = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=("part/objects",),
    )

    assert actual == expected


def test_mesh_plan_is_independent_of_interior_sample_iteration_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component_plan = _component_plan(
        tmp_path,
        points=_filled_rectangle(1, 1, 18, 18),
    )
    expected = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=("part/objects",),
    )
    original_interior_points = mesh_builder_module._interior_points

    def reversed_interior_points(*args, **kwargs):
        return tuple(reversed(original_interior_points(*args, **kwargs)))

    monkeypatch.setattr(
        mesh_builder_module,
        "_interior_points",
        reversed_interior_points,
    )

    actual = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=("part/objects",),
    )

    assert actual == expected


def test_mesh_plan_never_rethresholds_or_relabels_stage_a_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skimage import measure

    component_plan = _component_plan(
        tmp_path,
        points=_filled_rectangle(1, 1, 14, 14),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Stage B must consume QCL without thresholding or labeling")

    monkeypatch.setattr(component_plan_module, "_clean_binary_mask", forbidden)
    monkeypatch.setattr(component_plan_module, "_label_components", forbidden)
    monkeypatch.setattr(measure, "label", forbidden)

    plan = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=("part/objects",),
    )

    assert len(plan.meshes) == 1


def test_mesh_validator_rejects_a_rehashed_noncanonical_boundary_start(
    tmp_path: Path,
) -> None:
    component_plan = _component_plan(
        tmp_path,
        points=_filled_rectangle(1, 1, 14, 14),
    )
    plan = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=("part/objects",),
    )
    mesh = plan.meshes[0]
    loop = mesh.boundary_loops[0]
    rotated = (*loop[1:], loop[0])
    vertices = list(mesh.vertices)
    for order, index in enumerate(rotated):
        vertices[index] = replace(vertices[index], boundary_order=order)
    provisional_mesh = replace(
        mesh,
        vertices=tuple(vertices),
        boundary_loops=(rotated, *mesh.boundary_loops[1:]),
        mesh_sha256="",
    )
    tampered_mesh = replace(
        provisional_mesh,
        mesh_sha256=jcs_sha256(provisional_mesh.content_payload()),
    )
    provisional_plan = replace(plan, meshes=(tampered_mesh,), plan_sha256="")
    tampered_plan = replace(
        provisional_plan,
        plan_sha256=jcs_sha256(provisional_plan.semantic_payload()),
    )

    with pytest.raises(mesh_builder_module.MeshBuildError) as captured:
        validate_mesh_plan(tampered_plan)

    assert captured.value.code == "invalid_mesh_plan"
