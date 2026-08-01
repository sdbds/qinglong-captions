from __future__ import annotations

import pytest

from module.auto_rig.export.live2d.uv_kernel import (
    CUBISM_V400_UV_KERNEL_VERSION,
    canonical_top_left_to_core_api_uv,
    canonical_top_left_to_moc_uv,
    core_api_to_canonical_top_left_uv,
    core_api_to_d3d11_sample_uv,
    moc_to_canonical_top_left_uv,
)


def test_cubism_v400_uv_path_preserves_canonical_texture_sampling() -> None:
    canonical = (0.125, 0.75)

    moc = canonical_top_left_to_moc_uv(canonical)
    core = canonical_top_left_to_core_api_uv(canonical)
    sampled = core_api_to_d3d11_sample_uv(core)

    assert CUBISM_V400_UV_KERNEL_VERSION == "cubism-v400-uv-kernel-v1"
    assert moc == pytest.approx(canonical)
    assert moc_to_canonical_top_left_uv(moc) == pytest.approx(canonical)
    assert core == pytest.approx((0.125, 0.25))
    assert core_api_to_canonical_top_left_uv(core) == pytest.approx(canonical)
    assert sampled == pytest.approx(canonical)


@pytest.mark.parametrize("point", ((float("nan"), 0.0), (0.0, float("inf"))))
def test_cubism_v400_uv_kernel_rejects_non_finite_values(point: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        canonical_top_left_to_moc_uv(point)
