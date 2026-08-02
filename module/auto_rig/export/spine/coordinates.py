from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from ...jcs import jcs_sha256

SPINE_COORDINATE_PLAN_VERSION = "spine-coordinate-plan-v1"
SPINE_COORDINATE_TRANSFORM_VERSION = "canvas-center-y-reflection-v1"

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class SpineCoordinateError(ValueError):
    """Raised when the canvas-to-Spine frame contract is invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(f"invalid_spine_coordinate_plan: {message}")


def _error(message: str) -> SpineCoordinateError:
    return SpineCoordinateError(message)


def _number(value: object, *, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise _error(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise _error(f"{field} must be finite")
    return result


def _clean(value: float) -> float:
    return 0.0 if abs(value) < 1e-12 else value


@dataclass(frozen=True, slots=True)
class SpineCoordinatePlan:
    schema_version: str
    transform_version: str
    width: int
    height: int
    input_fingerprint: str
    test_points: tuple[tuple[float, float], ...]
    maximum_round_trip_error: float
    test_points_sha256: str
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "transform_version": self.transform_version,
            "width": self.width,
            "height": self.height,
            "input_fingerprint": self.input_fingerprint,
            "test_points": [list(point) for point in self.test_points],
            "maximum_round_trip_error": self.maximum_round_trip_error,
            "test_points_sha256": self.test_points_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "plan_sha256": self.plan_sha256}


def canvas_to_spine(
    plan: SpineCoordinatePlan,
    x: float,
    y: float,
) -> tuple[float, float]:
    px = _number(x, field="canvas x")
    py = _number(y, field="canvas y")
    return (
        _clean(px - plan.width / 2.0),
        _clean(plan.height / 2.0 - py),
    )


def spine_to_canvas(
    plan: SpineCoordinatePlan,
    x: float,
    y: float,
) -> tuple[float, float]:
    px = _number(x, field="Spine x")
    py = _number(y, field="Spine y")
    return (
        _clean(px + plan.width / 2.0),
        _clean(plan.height / 2.0 - py),
    )


def _test_points(
    width: int,
    height: int,
    landmarks: Iterable[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    values = {
        (0.0, 0.0),
        (float(width), 0.0),
        (0.0, float(height)),
        (float(width), float(height)),
        (width / 2.0, height / 2.0),
    }
    for index, point in enumerate(landmarks):
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise _error(f"landmark {index} must be an x/y pair")
        values.add(
            (
                _number(point[0], field=f"landmark {index} x"),
                _number(point[1], field=f"landmark {index} y"),
            )
        )
    return tuple(sorted(values))


def build_spine_coordinate_plan(
    canvas: Mapping[str, object],
    *,
    input_fingerprint: str,
    landmarks: Iterable[tuple[float, float]] = (),
) -> SpineCoordinatePlan:
    """Freeze the one canvas boundary accepted by the Spine 4.2 exporter."""

    if not isinstance(canvas, Mapping):
        raise _error("canvas must be an object")
    if canvas.get("origin") != "top_left" or canvas.get("y_axis") != "down":
        raise _error("canvas must use the frozen top-left/down coordinate contract")
    width_value = _number(canvas.get("width"), field="canvas width")
    height_value = _number(canvas.get("height"), field="canvas height")
    if not width_value.is_integer() or not height_value.is_integer():
        raise _error("canvas dimensions must be integers")
    width = int(width_value)
    height = int(height_value)
    if width <= 0 or height <= 0:
        raise _error("canvas dimensions must be positive")
    if not isinstance(input_fingerprint, str) or not _SHA256_RE.fullmatch(
        input_fingerprint
    ):
        raise _error("input fingerprint must be a lowercase SHA-256 identity")

    points = _test_points(width, height, landmarks)
    provisional = SpineCoordinatePlan(
        schema_version=SPINE_COORDINATE_PLAN_VERSION,
        transform_version=SPINE_COORDINATE_TRANSFORM_VERSION,
        width=width,
        height=height,
        input_fingerprint=input_fingerprint,
        test_points=points,
        maximum_round_trip_error=0.0,
        test_points_sha256=jcs_sha256([list(point) for point in points]),
        plan_sha256="",
    )
    residuals = []
    for point in points:
        recovered = spine_to_canvas(provisional, *canvas_to_spine(provisional, *point))
        residuals.append(math.hypot(recovered[0] - point[0], recovered[1] - point[1]))
    plan_without_digest = SpineCoordinatePlan(
        schema_version=provisional.schema_version,
        transform_version=provisional.transform_version,
        width=width,
        height=height,
        input_fingerprint=input_fingerprint,
        test_points=points,
        maximum_round_trip_error=max(residuals, default=0.0),
        test_points_sha256=provisional.test_points_sha256,
        plan_sha256="",
    )
    plan = SpineCoordinatePlan(
        schema_version=plan_without_digest.schema_version,
        transform_version=plan_without_digest.transform_version,
        width=plan_without_digest.width,
        height=plan_without_digest.height,
        input_fingerprint=plan_without_digest.input_fingerprint,
        test_points=plan_without_digest.test_points,
        maximum_round_trip_error=plan_without_digest.maximum_round_trip_error,
        test_points_sha256=plan_without_digest.test_points_sha256,
        plan_sha256=jcs_sha256(plan_without_digest.semantic_payload()),
    )
    return validate_spine_coordinate_plan(plan)


def validate_spine_coordinate_plan(
    plan: SpineCoordinatePlan,
) -> SpineCoordinatePlan:
    if not isinstance(plan, SpineCoordinatePlan):
        raise _error("plan has the wrong type")
    if (
        plan.schema_version != SPINE_COORDINATE_PLAN_VERSION
        or plan.transform_version != SPINE_COORDINATE_TRANSFORM_VERSION
    ):
        raise _error("coordinate version is unsupported")
    if plan.width <= 0 or plan.height <= 0:
        raise _error("canvas dimensions must be positive")
    if not _SHA256_RE.fullmatch(plan.input_fingerprint):
        raise _error("input fingerprint is invalid")
    if tuple(sorted(set(plan.test_points))) != plan.test_points:
        raise _error("test points are not canonical")
    if plan.test_points_sha256 != jcs_sha256(
        [list(point) for point in plan.test_points]
    ):
        raise _error("test-point digest mismatch")
    residual = 0.0
    for point in plan.test_points:
        if not all(math.isfinite(value) for value in point):
            raise _error("test points must be finite")
        recovered = spine_to_canvas(plan, *canvas_to_spine(plan, *point))
        residual = max(
            residual,
            math.hypot(recovered[0] - point[0], recovered[1] - point[1]),
        )
    if residual > 1e-6 or plan.maximum_round_trip_error != residual:
        raise _error("round-trip residual violates the coordinate contract")
    if plan.plan_sha256 != jcs_sha256(plan.semantic_payload()):
        raise _error("plan digest mismatch")
    return plan


__all__ = [
    "SPINE_COORDINATE_PLAN_VERSION",
    "SPINE_COORDINATE_TRANSFORM_VERSION",
    "SpineCoordinateError",
    "SpineCoordinatePlan",
    "build_spine_coordinate_plan",
    "canvas_to_spine",
    "spine_to_canvas",
    "validate_spine_coordinate_plan",
]
