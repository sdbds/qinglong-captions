from __future__ import annotations

import math

SPINE_42_UV_ADAPTER_VERSION = "spine-4.2-uv-v1"


class SpineUvError(ValueError):
    """Raised when canonical image-space UVs cannot map to a Spine region."""

    def __init__(self, message: str) -> None:
        super().__init__(f"invalid_spine_uv: {message}")


def _point(value: object, *, field: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SpineUvError(f"{field} must be an x/y pair")
    if any(
        not isinstance(item, (int, float))
        or isinstance(item, bool)
        or not math.isfinite(float(item))
        for item in value
    ):
        raise SpineUvError(f"{field} must contain finite numbers")
    return float(value[0]), float(value[1])


def canvas_to_spine_region_uv(
    position: object,
    source_xyxy: object,
) -> tuple[float, float]:
    """Return Spine region-local UV; v=0 follows the atlas region's top edge."""

    x, y = _point(position, field="canvas position")
    if not isinstance(source_xyxy, (list, tuple)) or len(source_xyxy) != 4:
        raise SpineUvError("source_xyxy must be a four-value rectangle")
    if any(
        not isinstance(item, (int, float))
        or isinstance(item, bool)
        or not math.isfinite(float(item))
        for item in source_xyxy
    ):
        raise SpineUvError("source_xyxy must contain finite numbers")
    x0, y0, x1, y1 = (float(value) for value in source_xyxy)
    if x1 <= x0 or y1 <= y0:
        raise SpineUvError("source_xyxy must have positive area")
    u = (x - x0) / (x1 - x0)
    v = (y - y0) / (y1 - y0)
    if not -1e-6 <= u <= 1.0 + 1e-6 or not -1e-6 <= v <= 1.0 + 1e-6:
        raise SpineUvError("vertex lies outside its source region")
    return min(max(u, 0.0), 1.0), min(max(v, 0.0), 1.0)


__all__ = [
    "SPINE_42_UV_ADAPTER_VERSION",
    "SpineUvError",
    "canvas_to_spine_region_uv",
]
