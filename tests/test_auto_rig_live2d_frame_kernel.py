from __future__ import annotations

import pytest

from module.auto_rig.export.live2d.frame_kernel import (
    apply_rotation_stack,
    apply_similarity,
    canvas_to_root,
    invert_rotation_stack,
    invert_similarity,
    root_to_canvas,
    rotation_stack_rank_conflict,
)


def test_canvas_to_root_uses_max_edge_and_flips_y_once() -> None:
    assert canvas_to_root((0.0, 0.0), 1024.0, 512.0) == pytest.approx((-0.5, 0.25))
    assert canvas_to_root((512.0, 256.0), 1024.0, 512.0) == pytest.approx((0.0, 0.0))


@pytest.mark.parametrize(
    ("point", "width", "height"),
    [
        ((0.0, 0.0), 1024.0, 1024.0),
        ((1024.0, 512.0), 1024.0, 512.0),
        ((13.25, 997.75), 768.0, 1280.0),
    ],
)
def test_canvas_root_mapping_round_trips_exact_formula(
    point: tuple[float, float],
    width: float,
    height: float,
) -> None:
    encoded = canvas_to_root(point, width, height)

    assert root_to_canvas(encoded, width, height) == pytest.approx(point, abs=1e-12)


def test_similarity_uses_translate_rotate_scale_order() -> None:
    transformed = apply_similarity(
        (2.0, 0.0),
        origin=(10.0, 20.0),
        angle_degrees=90.0,
        scale=2.0,
    )

    assert transformed == pytest.approx((10.0, 24.0), abs=1e-12)


def test_similarity_inverse_recovers_local_point() -> None:
    local = (-3.5, 8.25)
    transformed = apply_similarity(local, origin=(2.0, -7.0), angle_degrees=-37.0, scale=1.75)

    assert invert_similarity(
        transformed,
        origin=(2.0, -7.0),
        angle_degrees=-37.0,
        scale=1.75,
    ) == pytest.approx(local, abs=1e-12)


def test_rotation_stack_applies_lower_rank_as_outer_node() -> None:
    entries = [
        (20, 1.0, 0.0, 0.0, 1.0),
        (10, 10.0, 0.0, 90.0, 1.0),
    ]

    assert apply_rotation_stack((0.0, 0.0), entries) == pytest.approx((10.0, 1.0), abs=1e-12)


def test_rotation_stack_order_is_observably_non_commutative() -> None:
    lower_rank_outer = [
        (10, 10.0, 0.0, 90.0, 1.0),
        (20, 1.0, 0.0, 0.0, 1.0),
    ]
    reversed_ranks = [
        (20, 10.0, 0.0, 90.0, 1.0),
        (10, 1.0, 0.0, 0.0, 1.0),
    ]

    expected = apply_rotation_stack((0.0, 0.0), lower_rank_outer)
    reversed_result = apply_rotation_stack((0.0, 0.0), reversed_ranks)

    assert expected == pytest.approx((10.0, 1.0), abs=1e-12)
    assert reversed_result == pytest.approx((11.0, 0.0), abs=1e-12)


def test_rotation_stack_detects_duplicate_ranks() -> None:
    assert rotation_stack_rank_conflict(
        [
            (10, 0.0, 0.0, 0.0, 1.0),
            (10, 5.0, 2.0, 15.0, 1.0),
        ]
    )
    assert not rotation_stack_rank_conflict(
        [
            (10, 0.0, 0.0, 0.0, 1.0),
            (20, 5.0, 2.0, 15.0, 1.0),
        ]
    )


def test_rotation_stack_inverse_recovers_child_local_point() -> None:
    entries = [
        (300, -4.0, 5.0, 17.0, 0.8),
        (100, 12.0, -3.0, -42.0, 1.2),
        (200, 2.0, 9.0, 11.0, 1.05),
    ]
    local = (7.25, -1.5)

    parent = apply_rotation_stack(local, entries)

    assert invert_rotation_stack(parent, entries) == pytest.approx(local, abs=1e-10)


def test_empty_rotation_stack_is_identity() -> None:
    assert apply_rotation_stack((3.0, 4.0), []) == (3.0, 4.0)
    assert invert_rotation_stack((3.0, 4.0), []) == (3.0, 4.0)
