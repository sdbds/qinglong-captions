from __future__ import annotations

from math import cos, radians, sin
from typing import Iterable

Point2 = tuple[float, float]
RotationStackEntry = tuple[int, float, float, float, float]

FRAME_KERNEL_VERSION = "live2d-frame-kernel-v1"
ROTATION_STACK_SEMANTICS_VERSION = "rotation-stack-v1"


def canvas_to_root(point: Point2, width: float, height: float) -> Point2:
    ppu = max(width, height)
    return ((point[0] - width / 2.0) / ppu, (height / 2.0 - point[1]) / ppu)


def root_to_canvas(point: Point2, width: float, height: float) -> Point2:
    ppu = max(width, height)
    return (point[0] * ppu + width / 2.0, height / 2.0 - point[1] * ppu)


def apply_similarity(
    point: Point2,
    *,
    origin: Point2,
    angle_degrees: float,
    scale: float,
) -> Point2:
    angle = radians(angle_degrees)
    cosine = cos(angle)
    sine = sin(angle)
    scaled_x = scale * point[0]
    scaled_y = scale * point[1]
    return (
        origin[0] + cosine * scaled_x - sine * scaled_y,
        origin[1] + sine * scaled_x + cosine * scaled_y,
    )


def invert_similarity(
    point: Point2,
    *,
    origin: Point2,
    angle_degrees: float,
    scale: float,
) -> Point2:
    angle = radians(angle_degrees)
    cosine = cos(angle)
    sine = sin(angle)
    translated_x = point[0] - origin[0]
    translated_y = point[1] - origin[1]
    return (
        (cosine * translated_x + sine * translated_y) / scale,
        (-sine * translated_x + cosine * translated_y) / scale,
    )


def rotation_stack_rank_conflict(entries: Iterable[RotationStackEntry]) -> bool:
    ranks = [entry[0] for entry in entries]
    return len(ranks) != len(set(ranks))


def apply_rotation_stack(point: Point2, entries: Iterable[RotationStackEntry]) -> Point2:
    ordered = sorted(entries, key=lambda entry: entry[0])
    transformed = point
    for _, origin_x, origin_y, angle_degrees, scale in reversed(ordered):
        transformed = apply_similarity(
            transformed,
            origin=(origin_x, origin_y),
            angle_degrees=angle_degrees,
            scale=scale,
        )
    return transformed


def invert_rotation_stack(point: Point2, entries: Iterable[RotationStackEntry]) -> Point2:
    ordered = sorted(entries, key=lambda entry: entry[0])
    transformed = point
    for _, origin_x, origin_y, angle_degrees, scale in ordered:
        transformed = invert_similarity(
            transformed,
            origin=(origin_x, origin_y),
            angle_degrees=angle_degrees,
            scale=scale,
        )
    return transformed


__all__ = [
    "FRAME_KERNEL_VERSION",
    "Point2",
    "ROTATION_STACK_SEMANTICS_VERSION",
    "RotationStackEntry",
    "apply_rotation_stack",
    "apply_similarity",
    "canvas_to_root",
    "invert_rotation_stack",
    "invert_similarity",
    "root_to_canvas",
    "rotation_stack_rank_conflict",
]
