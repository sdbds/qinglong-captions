from __future__ import annotations

import math
from typing import Sequence

CUBISM_V400_UV_KERNEL_VERSION = "cubism-v400-uv-kernel-v1"


def _uv(value: Sequence[float]) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError("UV must be a two-number sequence")
    u, v = value
    if (
        isinstance(u, bool)
        or isinstance(v, bool)
        or not isinstance(u, (int, float))
        or not isinstance(v, (int, float))
    ):
        raise ValueError("UV coordinates must be numbers")
    point = (float(u), float(v))
    if not all(math.isfinite(coordinate) for coordinate in point):
        raise ValueError("UV coordinates must be finite")
    return point


def canonical_top_left_to_moc_uv(value: Sequence[float]) -> tuple[float, float]:
    """Encode canonical top-left image UVs into the V4.00 MOC section."""

    return _uv(value)


def moc_to_canonical_top_left_uv(value: Sequence[float]) -> tuple[float, float]:
    """Decode a V4.00 MOC UV into canonical top-left image UVs."""

    return _uv(value)


def canonical_top_left_to_core_api_uv(value: Sequence[float]) -> tuple[float, float]:
    """Return the UV exposed by Core for a canonical V4.00 MOC UV."""

    u, v = _uv(value)
    return (u, 1.0 - v)


def core_api_to_canonical_top_left_uv(value: Sequence[float]) -> tuple[float, float]:
    """Recover canonical UVs from the Core drawable API representation."""

    u, v = _uv(value)
    return (u, 1.0 - v)


def core_api_to_d3d11_sample_uv(value: Sequence[float]) -> tuple[float, float]:
    """Model the official D3D11 Framework shader's final V flip."""

    u, v = _uv(value)
    return (u, 1.0 - v)


__all__ = [
    "CUBISM_V400_UV_KERNEL_VERSION",
    "canonical_top_left_to_core_api_uv",
    "canonical_top_left_to_moc_uv",
    "core_api_to_canonical_top_left_uv",
    "core_api_to_d3d11_sample_uv",
    "moc_to_canonical_top_left_uv",
]
