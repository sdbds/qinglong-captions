from __future__ import annotations

import os
from pathlib import Path

import pytest

from module.auto_rig.export.live2d.cubism_renderer import render_moc_with_offscreen_harness
from module.auto_rig.export.live2d.e0_assets import (
    E0_BASE_PARAMETER_VALUES,
    E0_EXPECTED_PARAMETER_VALUES,
    build_e0_expression_json_bytes,
    build_e0_motion_json_bytes,
    write_e0_orientation_texture,
)
from module.auto_rig.export.live2d.e0_fixture import (
    STATIC_E0_ROOT_VERTICES,
    build_deformer_e0_moc,
    build_static_e0_document,
    build_static_e0_moc,
)
from module.auto_rig.export.live2d.moc3_codec import (
    Moc3V400Document,
    encode_moc3_v400,
)


def test_offscreen_harness_honors_serialized_motion_fade() -> None:
    source = (Path(__file__).parents[1] / "tools" / "auto_rig_live2d_e0" / "main.cpp").read_text(encoding="utf-8")

    assert "motion->SetFadeInTime" not in source
    assert "motion->SetFadeOutTime" not in source


def _sample_near_vertex(
    rgba: bytes,
    width: int,
    height: int,
    vertex: tuple[float, float],
) -> tuple[int, int, int, int]:
    centroid = (
        sum(point[0] for point in STATIC_E0_ROOT_VERTICES) / len(STATIC_E0_ROOT_VERTICES),
        sum(point[1] for point in STATIC_E0_ROOT_VERTICES) / len(STATIC_E0_ROOT_VERTICES),
    )
    point = (
        vertex[0] * 0.9 + centroid[0] * 0.1,
        vertex[1] * 0.9 + centroid[1] * 0.1,
    )
    x = round((point[0] + 1.0) * 0.5 * (width - 1))
    y = round((1.0 - point[1]) * 0.5 * (height - 1))
    offset = (y * width + x) * 4
    return tuple(rgba[offset : offset + 4])  # type: ignore[return-value]


@pytest.mark.optional_runtime
def test_offscreen_harness_renders_uv_orientation_and_straight_alpha(tmp_path: Path) -> None:
    executable = os.environ.get("LIVE2D_E0_RENDERER_PATH")
    if not executable:
        pytest.skip("LIVE2D_E0_RENDERER_PATH is required")
    moc_path = tmp_path / "static-e0.moc3"
    texture_path = tmp_path / "orientation.png"
    moc_path.write_bytes(build_static_e0_moc())
    write_e0_orientation_texture(texture_path)

    evidence = render_moc_with_offscreen_harness(
        executable,
        moc_path,
        texture_path,
        width=512,
        height=512,
    )

    assert evidence.width == 512
    assert evidence.height == 512
    assert evidence.validator_protocol_digest == ("sha256:df69c4a95ab40da12ce05deb7070edd76c58c8ec43a9c9a699cf9917dbfb8a21")
    assert evidence.nonzero_alpha_pixels > 1000
    assert evidence.alpha_bbox is not None
    samples = tuple(
        _sample_near_vertex(evidence.rgba, evidence.width, evidence.height, vertex) for vertex in STATIC_E0_ROOT_VERTICES
    )
    assert samples[0][0] > 200 and samples[0][1] < 30 and samples[0][2] < 30
    assert samples[1][1] > 200 and samples[1][0] < 30 and samples[1][2] < 30
    assert samples[2][0] > 200 and samples[2][1] > 200 and samples[2][2] < 30
    assert samples[3][2] > 200 and samples[3][0] < 30 and samples[3][1] < 30
    alphas = evidence.rgba[3::4]
    assert any(0 < alpha < 255 for alpha in alphas)


@pytest.mark.optional_runtime
def test_offscreen_harness_applies_motion_then_full_weight_expression(tmp_path: Path) -> None:
    executable = os.environ.get("LIVE2D_E0_RENDERER_PATH")
    if not executable:
        pytest.skip("LIVE2D_E0_RENDERER_PATH is required")
    moc_path = tmp_path / "deformer-e0.moc3"
    texture_path = tmp_path / "orientation.png"
    motion_path = tmp_path / "e0.motion3.json"
    expression_path = tmp_path / "e0.exp3.json"
    moc_path.write_bytes(build_deformer_e0_moc())
    write_e0_orientation_texture(texture_path)
    motion_path.write_bytes(build_e0_motion_json_bytes())
    expression_path.write_bytes(build_e0_expression_json_bytes())

    baseline = render_moc_with_offscreen_harness(
        executable,
        moc_path,
        texture_path,
        parameter_values=E0_BASE_PARAMETER_VALUES,
    )
    evidence = render_moc_with_offscreen_harness(
        executable,
        moc_path,
        texture_path,
        parameter_values=E0_BASE_PARAMETER_VALUES,
        motion_path=motion_path,
        expression_path=expression_path,
        evaluation_time=1.0,
        observe_parameter_ids=("ParamBreath", "ParamOuter", "ParamInner"),
    )

    assert evidence.parameter_values == pytest.approx(E0_EXPECTED_PARAMETER_VALUES, abs=1e-6)
    assert evidence.rgba_sha256 != baseline.rgba_sha256
    assert evidence.nonzero_alpha_pixels > 1000


@pytest.mark.optional_runtime
def test_offscreen_harness_binds_ordered_multiple_texture_pages(
    tmp_path: Path,
) -> None:
    executable = os.environ.get("LIVE2D_E0_RENDERER_PATH")
    if not executable:
        pytest.skip("LIVE2D_E0_RENDERER_PATH is required")
    document = build_static_e0_document()
    sections = dict(document.sections)
    sections["art_mesh.texture_indices"] = (1,)
    two_page_document = Moc3V400Document(
        counts=document.counts,
        canvas=document.canvas,
        sections=sections,
    )
    moc_path = tmp_path / "two-page.moc3"
    texture_0 = tmp_path / "page_0.png"
    texture_1 = tmp_path / "page_1.png"
    moc_path.write_bytes(encode_moc3_v400(two_page_document))
    write_e0_orientation_texture(texture_0)
    write_e0_orientation_texture(texture_1)

    evidence = render_moc_with_offscreen_harness(executable, moc_path, (texture_0, texture_1))
    assert evidence.nonzero_alpha_pixels > 1000
    with pytest.raises(RuntimeError, match="texture.*count|failed"):
        render_moc_with_offscreen_harness(executable, moc_path, texture_0)
