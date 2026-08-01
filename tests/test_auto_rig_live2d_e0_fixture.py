from __future__ import annotations

import os
from pathlib import Path

import pytest

from module.auto_rig.export.live2d.cubism_core import exercise_moc_with_core
from module.auto_rig.export.live2d.e0_fixture import (
    DEFORMER_E0_CANVAS_SIZE,
    STATIC_E0_CANVAS_SIZE,
    STATIC_E0_CORE_UVS,
    STATIC_E0_MOC_UVS,
    STATIC_E0_MOC_VERTICES,
    STATIC_E0_ROOT_VERTICES,
    STATIC_E0_UVS,
    build_deformer_e0_missing_default_moc,
    build_deformer_e0_moc,
    build_static_e0_moc,
    evaluate_deformer_e0_canvas,
    evaluate_deformer_e0_reversed_stack_canvas,
)
from module.auto_rig.export.live2d.moc3 import parse_moc3_v400_envelope
from module.auto_rig.export.live2d.moc3_codec import decode_moc3_v400


def test_static_e0_fixture_is_deterministic_and_typed() -> None:
    first = build_static_e0_moc()
    second = build_static_e0_moc()
    envelope = parse_moc3_v400_envelope(first)
    document = decode_moc3_v400(first)

    assert first == second
    assert envelope.header.version == 3
    assert envelope.canvas.size == STATIC_E0_CANVAS_SIZE
    assert document.counts == (1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 8, 1, 2, 1, 1, 8, 6, 1, 1, 1, 0, 0, 0)
    assert document.section("part.ids") == ("PartRoot",)
    assert document.section("art_mesh.ids") == ("ArtMeshFixture",)
    assert document.section("parameter.ids") == ("ParamOpacity",)
    assert document.section("keyform_position.xys") == tuple(
        coordinate for point in STATIC_E0_MOC_VERTICES for coordinate in point
    )
    assert document.section("uv.xys") == tuple(coordinate for point in STATIC_E0_MOC_UVS for coordinate in point)


@pytest.mark.optional_runtime
def test_static_e0_fixture_passes_official_core_and_preserves_geometry(tmp_path: Path) -> None:
    core_path = os.environ.get("LIVE2D_CUBISM_CORE_PATH")
    if not core_path:
        pytest.skip("LIVE2D_CUBISM_CORE_PATH is required")
    moc_path = tmp_path / "static-e0.moc3"
    moc_path.write_bytes(build_static_e0_moc())

    result = exercise_moc_with_core(core_path, moc_path, capture_model_state=True)

    assert result.consistency is True
    assert result.parameter_count == 1
    assert result.part_count == 1
    assert result.drawable_count == 1
    assert result.model_state is not None
    assert result.model_state.parameters[0].id == "ParamOpacity"
    assert result.model_state.parameters[0].value == pytest.approx(1.0)
    assert result.model_state.parts[0].id == "PartRoot"
    drawable = result.model_state.drawables[0]
    assert drawable.id == "ArtMeshFixture"
    for actual, expected in zip(drawable.vertex_positions, STATIC_E0_ROOT_VERTICES):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(drawable.vertex_uvs, STATIC_E0_CORE_UVS):
        assert actual == pytest.approx(expected)


def test_deformer_e0_fixture_encodes_warp_rotation_parent_graph() -> None:
    first = build_deformer_e0_moc()
    second = build_deformer_e0_moc()
    document = decode_moc3_v400(first)

    assert first == second
    assert document.canvas.size == DEFORMER_E0_CANVAS_SIZE
    assert document.section("deformer.ids") == (
        "WarpBreath",
        "RotationOuter",
        "RotationInner",
    )
    assert document.section("deformer.types") == (0, 1, 1)
    assert document.section("deformer.specific_indices") == (0, 0, 1)
    assert document.section("deformer.parent_deformer_indices") == (-1, 0, 1)
    assert document.section("art_mesh.ids") == ("ArtMeshDirect", "ArtMeshNested")
    assert document.section("art_mesh.parent_deformer_indices") == (0, 2)
    assert document.section("parameter.ids") == (
        "ParamStatic",
        "ParamBreath",
        "ParamOuter",
        "ParamInner",
    )
    assert document.section("parameter.default_values")[-1] == 6.0
    assert document.section("keys.values")[-3:] == (-30.0, 6.0, 30.0)


@pytest.mark.optional_runtime
def test_deformer_e0_explicit_non_midpoint_default_preserves_rest_pose(tmp_path: Path) -> None:
    core_path = os.environ.get("LIVE2D_CUBISM_CORE_PATH")
    if not core_path:
        pytest.skip("LIVE2D_CUBISM_CORE_PATH is required")
    positive_path = tmp_path / "deformer-e0.moc3"
    negative_path = tmp_path / "deformer-e0-missing-default.moc3"
    positive_path.write_bytes(build_deformer_e0_moc())
    negative_path.write_bytes(build_deformer_e0_missing_default_moc())

    positive = exercise_moc_with_core(core_path, positive_path, capture_model_state=True)
    negative = exercise_moc_with_core(core_path, negative_path, capture_model_state=True)

    assert positive.model_state is not None
    assert negative.model_state is not None
    positive_nested = next(
        drawable for drawable in positive.model_state.drawables if drawable.id == "ArtMeshNested"
    )
    negative_nested = next(
        drawable for drawable in negative.model_state.drawables if drawable.id == "ArtMeshNested"
    )
    positive_parameters = {parameter.id: parameter.value for parameter in positive.model_state.parameters}
    assert positive_parameters["ParamInner"] == pytest.approx(6.0)
    residuals = tuple(
        max(abs(actual[0] - bad[0]), abs(actual[1] - bad[1]))
        * max(DEFORMER_E0_CANVAS_SIZE)
        for actual, bad in zip(
            positive_nested.vertex_positions,
            negative_nested.vertex_positions,
        )
    )
    assert max(residuals) > 0.1


@pytest.mark.optional_runtime
@pytest.mark.parametrize(
    "parameter_values",
    [
        {},
        {"ParamBreath": -1.0},
        {"ParamBreath": 1.0},
        {"ParamOuter": -20.0},
        {"ParamOuter": 20.0},
        {"ParamInner": -30.0},
        {"ParamInner": 30.0},
        {"ParamBreath": 0.375, "ParamOuter": -7.5, "ParamInner": 12.5},
        {"ParamBreath": -0.625, "ParamOuter": 13.25, "ParamInner": -21.75},
    ],
)
def test_deformer_e0_fixture_matches_official_core_geometry_matrix(
    tmp_path: Path,
    parameter_values: dict[str, float],
) -> None:
    core_path = os.environ.get("LIVE2D_CUBISM_CORE_PATH")
    if not core_path:
        pytest.skip("LIVE2D_CUBISM_CORE_PATH is required")
    moc_path = tmp_path / "deformer-e0.moc3"
    moc_path.write_bytes(build_deformer_e0_moc())

    result = exercise_moc_with_core(
        core_path,
        moc_path,
        parameter_values=parameter_values,
        capture_model_state=True,
    )

    assert result.consistency is True
    assert result.model_state is not None
    expected_by_id = evaluate_deformer_e0_canvas(parameter_values)
    assert tuple(drawable.id for drawable in result.model_state.drawables) == tuple(expected_by_id)
    for drawable in result.model_state.drawables:
        actual_canvas = tuple(
            (
                point[0] * max(DEFORMER_E0_CANVAS_SIZE) + DEFORMER_E0_CANVAS_SIZE[0] / 2.0,
                DEFORMER_E0_CANVAS_SIZE[1] / 2.0
                - point[1] * max(DEFORMER_E0_CANVAS_SIZE),
            )
            for point in drawable.vertex_positions
        )
        for actual, expected in zip(actual_canvas, expected_by_id[drawable.id]):
            assert actual == pytest.approx(expected, abs=0.1)


def test_deformer_e0_rotation_order_is_observably_non_commutative() -> None:
    parameters = {"ParamOuter": 20.0, "ParamInner": 30.0}
    expected = evaluate_deformer_e0_canvas(parameters)["ArtMeshNested"]
    reversed_stack = evaluate_deformer_e0_reversed_stack_canvas(parameters)

    residuals = tuple(
        max(abs(actual[0] - reversed_point[0]), abs(actual[1] - reversed_point[1]))
        for actual, reversed_point in zip(expected, reversed_stack)
    )

    assert max(residuals) > 0.1


@pytest.mark.optional_runtime
@pytest.mark.parametrize("outer_value", (-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0))
def test_deformer_e0_angle_and_origin_interpolate_together_at_nine_points(
    tmp_path: Path,
    outer_value: float,
) -> None:
    core_path = os.environ.get("LIVE2D_CUBISM_CORE_PATH")
    if not core_path:
        pytest.skip("LIVE2D_CUBISM_CORE_PATH is required")
    moc_path = tmp_path / "deformer-e0.moc3"
    moc_path.write_bytes(build_deformer_e0_moc())
    parameters = {"ParamOuter": outer_value}

    result = exercise_moc_with_core(
        core_path,
        moc_path,
        parameter_values=parameters,
        capture_model_state=True,
    )

    assert result.model_state is not None
    nested = next(drawable for drawable in result.model_state.drawables if drawable.id == "ArtMeshNested")
    expected = evaluate_deformer_e0_canvas(parameters)["ArtMeshNested"]
    actual_canvas = tuple(
        (
            point[0] * max(DEFORMER_E0_CANVAS_SIZE) + DEFORMER_E0_CANVAS_SIZE[0] / 2.0,
            DEFORMER_E0_CANVAS_SIZE[1] / 2.0
            - point[1] * max(DEFORMER_E0_CANVAS_SIZE),
        )
        for point in nested.vertex_positions
    )
    for actual, expected_point in zip(actual_canvas, expected):
        assert actual == pytest.approx(expected_point, abs=0.1)
