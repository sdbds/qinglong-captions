from __future__ import annotations

from dataclasses import replace

import pytest

from module.auto_rig.export.spine.coordinates import (
    SPINE_COORDINATE_PLAN_VERSION,
    SpineCoordinateError,
    build_spine_coordinate_plan,
    canvas_to_spine,
    spine_to_canvas,
    validate_spine_coordinate_plan,
)


def test_spine_coordinate_plan_freezes_center_origin_and_y_reflection() -> None:
    plan = build_spine_coordinate_plan(
        {"width": 400, "height": 300, "origin": "top_left", "y_axis": "down"},
        input_fingerprint="sha256:" + "1" * 64,
        landmarks=((17.25, 29.5), (200.0, 150.0)),
    )

    assert plan.schema_version == SPINE_COORDINATE_PLAN_VERSION
    assert canvas_to_spine(plan, 0.0, 0.0) == (-200.0, 150.0)
    assert canvas_to_spine(plan, 400.0, 300.0) == (200.0, -150.0)
    assert canvas_to_spine(plan, 200.0, 150.0) == (0.0, 0.0)
    assert spine_to_canvas(plan, -200.0, 150.0) == (0.0, 0.0)
    assert plan.maximum_round_trip_error <= 1e-6
    assert validate_spine_coordinate_plan(plan) is plan


def test_spine_coordinate_plan_rejects_guessing_or_rehashed_mutation() -> None:
    with pytest.raises(SpineCoordinateError, match="top-left/down"):
        build_spine_coordinate_plan(
            {"width": 400, "height": 300, "origin": "bottom_left", "y_axis": "up"},
            input_fingerprint="sha256:" + "2" * 64,
        )
    with pytest.raises(SpineCoordinateError, match="finite"):
        build_spine_coordinate_plan(
            {"width": 400, "height": 300, "origin": "top_left", "y_axis": "down"},
            input_fingerprint="sha256:" + "2" * 64,
            landmarks=((float("nan"), 0.0),),
        )

    plan = build_spine_coordinate_plan(
        {"width": 400, "height": 300, "origin": "top_left", "y_axis": "down"},
        input_fingerprint="sha256:" + "3" * 64,
    )
    with pytest.raises(SpineCoordinateError, match="digest"):
        validate_spine_coordinate_plan(replace(plan, width=401))
