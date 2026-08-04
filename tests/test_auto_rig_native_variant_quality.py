from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from module.auto_rig.component_plan import (
    build_mask_component_plan,
    extend_mask_component_plan_with_variants,
)
from module.auto_rig.contracts import AutoRigPartContract, ValidatedPartSource
from module.auto_rig.draw_order import build_ordinary_draw_order
from module.auto_rig.jcs import jcs_sha256
from module.auto_rig.mask_sources import LoadedPartAlpha
from module.auto_rig.native_variant_quality import (
    BASE_FEATURE_SCALE_KIND,
    BUILTIN_NATIVE_VARIANT_ROLE_ENVELOPES,
    NATIVE_VARIANT_ROLE_ENVELOPE_VERSION,
    NativeVariantBranchMetrics,
    NativeVariantQualityError,
    NativeVariantQualityPlan,
    base_context_radius_px,
    build_native_variant_quality_plan,
    evaluate_native_variant_candidates,
    native_variant_role_registry_payload,
    validate_native_variant_role_registry,
)
from module.auto_rig.native_variants import NativeVariantCandidate, NativeVariantSet

_SHA256_ZERO = "sha256:" + "0" * 64


def test_disk_dilation_scales_to_release_canvas_radii() -> None:
    script = """
import numpy as np

from module.auto_rig.native_variant_quality import _disk_dilate

mask = np.zeros((256, 256), dtype=bool)
mask[128, 128] = True
result = _disk_dilate(mask, 128)
assert result.shape == mask.shape
assert result[128, 128]
assert not result[0, 0]
print("ok")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
        timeout=8,
    )

    assert completed.stdout.strip() == "ok"


def _alpha(width: int, height: int, points: set[tuple[int, int]]) -> bytes:
    values = bytearray(width * height)
    for x, y in points:
        values[y * width + x] = 255
    return bytes(values)


def _loaded_part(
    part_id: str,
    base_tag: str,
    xyxy: tuple[int, int, int, int],
    points: set[tuple[int, int]],
    *,
    depth: float,
    side: str | None = None,
) -> LoadedPartAlpha:
    width = xyxy[2] - xyxy[0]
    height = xyxy[3] - xyxy[1]
    return LoadedPartAlpha(
        part=AutoRigPartContract(
            source_tag=(f"{base_tag}-r" if side == "xmin" else f"{base_tag}-l" if side == "xmax" else base_tag),
            base_tag=base_tag,
            semantic_slug=base_tag.replace(" ", "-"),
            side=side,
            part_id=part_id,
            xyxy=xyxy,
            depth_median=depth,
            source=ValidatedPartSource(
                mode="png",
                color_path=Path("unused-color.png"),
                depth_path=Path("unused-depth.png"),
                color_sha256=_SHA256_ZERO,
                depth_sha256=_SHA256_ZERO,
                layer_name=None,
            ),
        ),
        width=width,
        height=height,
        alpha_u8=_alpha(width, height, points),
    )


def _mouth_candidate(
    points: set[tuple[int, int]],
    *,
    xyxy: tuple[int, int, int, int] = (20, 20, 32, 44),
    variant_id: str = "mouth-open",
    role: str = "mouth_open",
    composite_mode: str = "occluding_overlay_v1",
) -> NativeVariantCandidate:
    width = xyxy[2] - xyxy[0]
    height = xyxy[3] - xyxy[1]
    alpha = _alpha(width, height, points)
    return NativeVariantCandidate(
        variant_id=variant_id,
        part_id=f"part/native.{variant_id}",
        semantic_role=role,
        composite_mode=composite_mode,
        base_part_ids=("part/mouth",),
        draw_anchor_part_id="part/mouth",
        anchor_base_tag="mouth",
        anchor_depth_median=0.0,
        xyxy=xyxy,
        relative_path=f"{variant_id}.png",
        png_path=Path(f"{variant_id}.png"),
        file_sha256=_SHA256_ZERO,
        rgba_mode="RGBA",
        alpha_mode="straight",
        color_space="sRGB",
        alpha_mass_u8_sum=sum(alpha),
        alpha_u8=alpha,
    )


def _variant_set(*entries: NativeVariantCandidate) -> NativeVariantSet:
    provisional = NativeVariantSet(
        schema_version="native-variant-set-v1",
        present=bool(entries),
        manifest_path=None,
        entries=tuple(sorted(entries, key=lambda item: item.variant_id)),
        native_variant_set_sha256="",
    )
    return replace(
        provisional,
        native_variant_set_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def _mouth_quality(
    tmp_path: Path,
    candidate: NativeVariantCandidate,
    *,
    extras: tuple[LoadedPartAlpha, ...] = (),
) -> NativeVariantBranchMetrics:
    canvas = 64
    face = _loaded_part(
        "part/face",
        "face",
        (0, 0, canvas, canvas),
        {(x, y) for y in range(canvas) for x in range(canvas)},
        depth=0.5,
    )
    mouth = _loaded_part(
        "part/mouth",
        "mouth",
        (21, 30, 31, 32),
        {(x, y) for y in range(2) for x in range(10)},
        depth=0.0,
    )
    loaded = (face, mouth, *extras)
    base_plan = build_mask_component_plan(
        loaded,
        canvas_edge=canvas,
        item_root=tmp_path,
    )
    draw_order = build_ordinary_draw_order(base_plan.parts)
    variant_set = _variant_set(candidate)
    combined = extend_mask_component_plan_with_variants(
        base_plan,
        variant_set,
        tmp_path,
    )
    results = evaluate_native_variant_candidates(
        loaded,
        combined,
        draw_order,
        variant_set,
        item_root=tmp_path,
    )
    assert len(results) == 1
    assert len(results[0].branches) == 1
    return results[0].branches[0]


def _mouth_quality_plan(
    tmp_path: Path,
    *candidates: NativeVariantCandidate,
) -> NativeVariantQualityPlan:
    tmp_path.mkdir(parents=True, exist_ok=True)
    canvas = 64
    face = _loaded_part(
        "part/face",
        "face",
        (0, 0, canvas, canvas),
        {(x, y) for y in range(canvas) for x in range(canvas)},
        depth=0.5,
    )
    mouth = _loaded_part(
        "part/mouth",
        "mouth",
        (21, 30, 31, 32),
        {(x, y) for y in range(2) for x in range(10)},
        depth=0.0,
    )
    loaded = (face, mouth)
    base_plan = build_mask_component_plan(
        loaded,
        canvas_edge=canvas,
        item_root=tmp_path,
    )
    draw_order = build_ordinary_draw_order(base_plan.parts)
    variant_set = _variant_set(*candidates)
    combined = extend_mask_component_plan_with_variants(
        base_plan,
        variant_set,
        tmp_path,
    )
    return build_native_variant_quality_plan(
        loaded,
        combined,
        draw_order,
        variant_set,
        item_root=tmp_path,
    )


def _eye_quality_result(
    tmp_path: Path,
    *,
    role: str,
    xyxy: tuple[int, int, int, int],
    points: set[tuple[int, int]],
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    canvas = 64
    face = _loaded_part(
        "part/face",
        "face",
        (0, 0, canvas, canvas),
        {(x, y) for y in range(canvas) for x in range(canvas)},
        depth=0.5,
    )
    xmin = _loaded_part(
        "part/eyewhite.xmin",
        "eyewhite",
        (2, 20, 22, 40),
        {(x, y) for y in range(20) for x in range(20)},
        depth=0.0,
        side="xmin",
    )
    xmax = _loaded_part(
        "part/eyewhite.xmax",
        "eyewhite",
        (42, 20, 62, 40),
        {(x, y) for y in range(20) for x in range(20)},
        depth=0.0,
        side="xmax",
    )
    loaded = (face, xmin, xmax)
    base_plan = build_mask_component_plan(
        loaded,
        canvas_edge=canvas,
        item_root=tmp_path,
    )
    draw_order = build_ordinary_draw_order(base_plan.parts)
    if role == "eye_closed.xmin":
        base_ids = ("part/eyewhite.xmin",)
    elif role == "eye_closed.xmax":
        base_ids = ("part/eyewhite.xmax",)
    else:
        base_ids = ("part/eyewhite.xmax", "part/eyewhite.xmin")
    anchor = max(base_ids, key=draw_order.ordinary_part_order.index)
    width = xyxy[2] - xyxy[0]
    height = xyxy[3] - xyxy[1]
    alpha = _alpha(width, height, points)
    candidate = NativeVariantCandidate(
        variant_id=role.replace("eye_closed.", "blink-"),
        part_id=f"part/native.{role.replace('eye_closed.', 'blink-')}",
        semantic_role=role,
        composite_mode="occluding_overlay_v1",
        base_part_ids=base_ids,
        draw_anchor_part_id=anchor,
        anchor_base_tag="eyewhite",
        anchor_depth_median=0.0,
        xyxy=xyxy,
        relative_path="blink.png",
        png_path=Path("blink.png"),
        file_sha256=_SHA256_ZERO,
        rgba_mode="RGBA",
        alpha_mode="straight",
        color_space="sRGB",
        alpha_mass_u8_sum=sum(alpha),
        alpha_u8=alpha,
    )
    variant_set = _variant_set(candidate)
    combined = extend_mask_component_plan_with_variants(
        base_plan,
        variant_set,
        tmp_path,
    )
    return evaluate_native_variant_candidates(
        loaded,
        combined,
        draw_order,
        variant_set,
        item_root=tmp_path,
    )[0]


def _eye_quality_plan(
    tmp_path: Path,
    *specs: tuple[str, tuple[int, int, int, int], set[tuple[int, int]]],
) -> NativeVariantQualityPlan:
    tmp_path.mkdir(parents=True, exist_ok=True)
    canvas = 64
    face = _loaded_part(
        "part/face",
        "face",
        (0, 0, canvas, canvas),
        {(x, y) for y in range(canvas) for x in range(canvas)},
        depth=0.5,
    )
    xmin = _loaded_part(
        "part/eyewhite.xmin",
        "eyewhite",
        (2, 20, 22, 40),
        {(x, y) for y in range(20) for x in range(20)},
        depth=0.0,
        side="xmin",
    )
    xmax = _loaded_part(
        "part/eyewhite.xmax",
        "eyewhite",
        (42, 20, 62, 40),
        {(x, y) for y in range(20) for x in range(20)},
        depth=0.0,
        side="xmax",
    )
    loaded = (face, xmin, xmax)
    base_plan = build_mask_component_plan(
        loaded,
        canvas_edge=canvas,
        item_root=tmp_path,
    )
    draw_order = build_ordinary_draw_order(base_plan.parts)
    candidates = []
    for role, xyxy, points in specs:
        if role == "eye_closed.xmin":
            base_ids = ("part/eyewhite.xmin",)
        elif role == "eye_closed.xmax":
            base_ids = ("part/eyewhite.xmax",)
        else:
            base_ids = ("part/eyewhite.xmax", "part/eyewhite.xmin")
        anchor = max(base_ids, key=draw_order.ordinary_part_order.index)
        width = xyxy[2] - xyxy[0]
        height = xyxy[3] - xyxy[1]
        alpha = _alpha(width, height, points)
        variant_id = role.replace("eye_closed.", "blink-")
        candidates.append(
            NativeVariantCandidate(
                variant_id=variant_id,
                part_id=f"part/native.{variant_id}",
                semantic_role=role,
                composite_mode="occluding_overlay_v1",
                base_part_ids=base_ids,
                draw_anchor_part_id=anchor,
                anchor_base_tag="eyewhite",
                anchor_depth_median=0.0,
                xyxy=xyxy,
                relative_path=f"{variant_id}.png",
                png_path=Path(f"{variant_id}.png"),
                file_sha256=_SHA256_ZERO,
                rgba_mode="RGBA",
                alpha_mode="straight",
                color_space="sRGB",
                alpha_mass_u8_sum=sum(alpha),
                alpha_u8=alpha,
            )
        )
    variant_set = _variant_set(*candidates)
    combined = extend_mask_component_plan_with_variants(
        base_plan,
        variant_set,
        tmp_path,
    )
    return build_native_variant_quality_plan(
        loaded,
        combined,
        draw_order,
        variant_set,
        item_root=tmp_path,
    )


def _rows_by_role():
    return {row.semantic_role: row for row in BUILTIN_NATIVE_VARIANT_ROLE_ENVELOPES}


def test_native_variant_role_envelope_registry_is_exact_and_digestible() -> None:
    rows = _rows_by_role()

    assert set(rows) == {
        "eye_closed.xmin",
        "eye_closed.xmax",
        "eye_closed.coupled",
        "mouth_closed",
        "mouth_open",
        "mouth_smile",
        "mouth_frown",
    }
    assert BASE_FEATURE_SCALE_KIND == "sqrt_base_alpha_mass_v1"
    assert rows["eye_closed.xmin"].to_dict() == {
        "semantic_role": "eye_closed.xmin",
        "support_mode": "support_preserving",
        "k_numerator": 3,
        "k_denominator": 2,
        "c_numerator": 1,
        "c_denominator": 5,
        "radius_min_px": 2,
        "radius_max_px": 12,
    }
    assert rows["mouth_smile"].to_dict() == {
        "semantic_role": "mouth_smile",
        "support_mode": "support_expanding",
        "k_numerator": 4,
        "k_denominator": 1,
        "c_numerator": 1,
        "c_denominator": 1,
        "radius_min_px": 4,
        "radius_max_px": 24,
    }
    assert rows["mouth_open"].to_dict() == {
        "semantic_role": "mouth_open",
        "support_mode": "support_expanding",
        "k_numerator": 12,
        "k_denominator": 1,
        "c_numerator": 2,
        "c_denominator": 1,
        "radius_min_px": 8,
        "radius_max_px": 48,
    }
    payload = native_variant_role_registry_payload()
    assert payload["schema_version"] == NATIVE_VARIANT_ROLE_ENVELOPE_VERSION
    assert payload["base_feature_scale_kind"] == BASE_FEATURE_SCALE_KIND
    assert payload["role_envelopes"] == [row.to_dict() for row in sorted(rows.values(), key=lambda row: row.semantic_role)]
    assert validate_native_variant_role_registry() == jcs_sha256(payload)


@pytest.mark.parametrize(
    ("role", "expected"),
    (
        ("eye_closed.xmin", 3),
        ("eye_closed.xmax", 3),
        ("eye_closed.coupled", 3),
        ("mouth_smile", 14),
        ("mouth_frown", 14),
        ("mouth_open", 28),
    ),
)
def test_base_context_radius_uses_common_mass_scale_and_exact_round_half_up(
    role: str,
    expected: int,
) -> None:
    assert base_context_radius_px(role, base_alpha_mass_u8_sum=200 * 255) == expected


def test_base_context_radius_is_independent_of_canvas_and_bbox_span_by_api_shape() -> None:
    first = base_context_radius_px("mouth_open", base_alpha_mass_u8_sum=200 * 255)
    second = base_context_radius_px("mouth_open", base_alpha_mass_u8_sum=200 * 255)

    assert first == second == 28


@pytest.mark.parametrize(
    ("role", "mass_u8", "expected"),
    (
        ("eye_closed.xmin", 255, 2),
        ("eye_closed.xmin", 10000 * 255, 12),
        ("mouth_smile", 255, 4),
        ("mouth_smile", 10000 * 255, 24),
        ("mouth_open", 255, 8),
        ("mouth_open", 10000 * 255, 48),
    ),
)
def test_base_context_radius_applies_role_pixel_clamps(
    role: str,
    mass_u8: int,
    expected: int,
) -> None:
    assert base_context_radius_px(role, base_alpha_mass_u8_sum=mass_u8) == expected


@pytest.mark.parametrize(
    "rows",
    (
        BUILTIN_NATIVE_VARIANT_ROLE_ENVELOPES[:-1],
        (*BUILTIN_NATIVE_VARIANT_ROLE_ENVELOPES, BUILTIN_NATIVE_VARIANT_ROLE_ENVELOPES[0]),
        (
            replace(
                BUILTIN_NATIVE_VARIANT_ROLE_ENVELOPES[0],
                radius_min_px=13,
                radius_max_px=12,
            ),
            *BUILTIN_NATIVE_VARIANT_ROLE_ENVELOPES[1:],
        ),
        (
            replace(BUILTIN_NATIVE_VARIANT_ROLE_ENVELOPES[0], k_numerator=0),
            *BUILTIN_NATIVE_VARIANT_ROLE_ENVELOPES[1:],
        ),
    ),
)
def test_invalid_native_variant_role_envelope_registry_is_a_startup_error(rows) -> None:
    with pytest.raises(NativeVariantQualityError) as captured:
        validate_native_variant_role_registry(rows)

    assert captured.value.code == "invalid_primitive_registry"


def test_base_context_radius_rejects_unknown_role_and_empty_mass() -> None:
    with pytest.raises(NativeVariantQualityError, match="unknown role"):
        base_context_radius_px("mouth_unknown", base_alpha_mass_u8_sum=255)
    with pytest.raises(NativeVariantQualityError, match="positive"):
        base_context_radius_px("mouth_open", base_alpha_mass_u8_sum=0)


def test_valid_mouth_overlay_passes_coverage_scale_and_base_anchored_face_envelope(
    tmp_path: Path,
) -> None:
    points = {(x, y) for y in range(7, 18) for x in range(12)}

    metrics = _mouth_quality(tmp_path, _mouth_candidate(points))

    assert metrics.eligible is True
    assert metrics.rejection_reason is None
    assert metrics.base_alpha_mass_u8_sum == 20 * 255
    assert metrics.variant_alpha_mass_u8_sum == 132 * 255
    assert metrics.coverage_leak == 0.0
    assert metrics.alpha_mass_ratio == pytest.approx(6.6)
    assert metrics.base_feature_scale_kind == BASE_FEATURE_SCALE_KIND
    assert metrics.base_support_bbox == (21, 30, 31, 32)
    assert metrics.base_support_aspect_ratio == 5.0
    assert metrics.base_context_radius_px == 9
    assert metrics.occlusion_intrusion == 0.0
    assert metrics.intrusion_face_ratio == 0.0
    assert metrics.intrusion_other_parts_ratio == 0.0
    assert [entry.part_id for entry in metrics.intrusion_by_part] == [
        "part/face",
        "part/mouth",
    ]
    assert all(entry.ratio == 0.0 for entry in metrics.intrusion_by_part)


def test_transparent_replacement_is_rejected_by_coverage_leak(tmp_path: Path) -> None:
    points = {(x, y) for y in range(10, 12) for x in range(1, 6)}

    metrics = _mouth_quality(tmp_path, _mouth_candidate(points))

    assert metrics.eligible is False
    assert metrics.rejection_reason == "coverage_leak"
    assert metrics.coverage_leak == pytest.approx(0.5)


def test_crossfade_closed_mouth_does_not_require_replacement_coverage(
    tmp_path: Path,
) -> None:
    candidate = _mouth_candidate(
        {(x, 1) for x in range(10)},
        xyxy=(21, 29, 31, 33),
        variant_id="mouth-closed",
        role="mouth_closed",
        composite_mode="crossfade_overlay_v1",
    )

    metrics = _mouth_quality(tmp_path, candidate)
    plan = _mouth_quality_plan(tmp_path / "plan", candidate)

    assert metrics.coverage_leak >= 0.5
    assert metrics.eligible is True
    assert metrics.rejection_reason is None
    assert plan.bundle_results[0].bundle_id == "mouth_crossfade.native"
    assert plan.bundle_results[0].eligible is True


def test_oversized_mouth_overlay_is_rejected_by_role_mass_ratio(tmp_path: Path) -> None:
    points = {(x, y) for y in range(24) for x in range(12)}

    metrics = _mouth_quality(tmp_path, _mouth_candidate(points))

    assert metrics.eligible is False
    assert metrics.rejection_reason == "alpha_mass_ratio_exceeded"
    assert metrics.alpha_mass_ratio == pytest.approx(14.4)


def test_cheek_patch_inside_mass_limit_but_outside_base_envelope_is_intrusion(
    tmp_path: Path,
) -> None:
    points = {(x, y) for y in range(7, 23) for x in range(12)}

    metrics = _mouth_quality(tmp_path, _mouth_candidate(points))

    assert metrics.alpha_mass_ratio == pytest.approx(9.6)
    assert metrics.eligible is False
    assert metrics.rejection_reason == "occlusion_intrusion"
    assert metrics.occlusion_intrusion > 0.01
    assert metrics.intrusion_face_ratio == metrics.occlusion_intrusion
    assert metrics.intrusion_other_parts_ratio == 0.0


def test_intrusion_reports_every_prefix_part_and_excludes_parts_after_anchor(
    tmp_path: Path,
) -> None:
    nose = _loaded_part(
        "part/nose",
        "nose",
        (22, 27, 28, 31),
        {(x, y) for y in range(4) for x in range(6)},
        depth=1.0,
    )
    front_hair = _loaded_part(
        "part/front-hair",
        "front hair",
        (20, 36, 32, 42),
        {(x, y) for y in range(6) for x in range(12)},
        depth=1.0,
    )
    points = {(x, y) for y in range(7, 22) for x in range(12)}

    metrics = _mouth_quality(
        tmp_path,
        _mouth_candidate(points),
        extras=(nose, front_hair),
    )

    assert [entry.part_id for entry in metrics.intrusion_by_part] == [
        "part/face",
        "part/mouth",
        "part/nose",
    ]
    by_part = {entry.part_id: entry.ratio for entry in metrics.intrusion_by_part}
    assert by_part["part/mouth"] == 0.0
    assert by_part["part/nose"] > 0.0
    assert "part/front-hair" not in by_part
    assert metrics.intrusion_other_parts_ratio == pytest.approx(by_part["part/nose"])
    assert metrics.occlusion_intrusion == pytest.approx(sum(entry.ratio for entry in metrics.intrusion_by_part))
    assert metrics.occlusion_intrusion == pytest.approx(metrics.intrusion_face_ratio + metrics.intrusion_other_parts_ratio)


def test_coupled_eye_uses_two_independent_component_branches_and_mass_scales(
    tmp_path: Path,
) -> None:
    coupled = _eye_quality_result(
        tmp_path / "coupled",
        role="eye_closed.coupled",
        xyxy=(2, 20, 62, 40),
        points=({(x, y) for y in range(20) for x in range(20)} | {(x, y) for y in range(20) for x in range(40, 60)}),
    )

    assert coupled.eligible is True
    assert tuple(branch.side for branch in coupled.branches) == ("xmin", "xmax")
    for branch in coupled.branches:
        assert branch.base_alpha_mass_u8_sum == 400 * 255
        assert branch.variant_alpha_mass_u8_sum == 400 * 255
        assert branch.alpha_mass_ratio == 1.0
        assert branch.base_context_radius_px == 4
        assert branch.coverage_leak == 0.0
        assert branch.occlusion_intrusion == 0.0
        assert len(branch.variant_component_ids) == 1


@pytest.mark.parametrize(
    "points",
    (
        {(x, y) for y in range(20) for x in range(20)},
        (
            {(x, y) for y in range(10) for x in range(10)}
            | {(x, y) for y in range(10) for x in range(25, 35)}
            | {(x, y) for y in range(10) for x in range(50, 60)}
        ),
        ({(x, y) for y in range(20) for x in range(20)} | {(58, 0), (59, 0), (58, 1), (59, 1)}),
    ),
)
def test_coupled_eye_requires_exactly_two_reliable_sided_components(
    tmp_path: Path,
    points: set[tuple[int, int]],
) -> None:
    result = _eye_quality_result(
        tmp_path,
        role="eye_closed.coupled",
        xyxy=(2, 20, 62, 40),
        points=points,
    )

    assert result.eligible is False
    assert result.rejection_reason == "component_partition"
    assert result.branches == ()


def test_empty_native_variant_source_has_versioned_empty_quality_plan(tmp_path: Path) -> None:
    plan = _mouth_quality_plan(tmp_path)

    assert plan.candidate_results == ()
    assert plan.bundle_results == ()
    assert plan.quality_eligible_variant_ids == ()
    assert plan.plan_sha256 == jcs_sha256(plan.semantic_payload())


def test_mouth_open_is_an_independent_atomic_quality_bundle(tmp_path: Path) -> None:
    valid = {(x, y) for y in range(7, 18) for x in range(12)}

    plan = _mouth_quality_plan(tmp_path, _mouth_candidate(valid))

    assert plan.quality_eligible_variant_ids == ("mouth-open",)
    assert len(plan.bundle_results) == 1
    bundle = plan.bundle_results[0]
    assert bundle.bundle_id == "mouth_open.native"
    assert bundle.complete is True
    assert bundle.eligible is True
    assert bundle.rejection_reason is None
    assert bundle.variant_ids == ("mouth-open",)


def test_mouth_form_requires_both_endpoints_as_one_atomic_bundle(tmp_path: Path) -> None:
    valid = {(x, y) for y in range(9, 14) for x in range(12)}
    smile = _mouth_candidate(valid, variant_id="mouth-smile", role="mouth_smile")

    incomplete = _mouth_quality_plan(tmp_path / "incomplete", smile)

    assert incomplete.candidate_results[0].eligible is True
    assert incomplete.quality_eligible_variant_ids == ()
    assert incomplete.bundle_results[0].bundle_id == "mouth_form.native"
    assert incomplete.bundle_results[0].complete is False
    assert incomplete.bundle_results[0].rejection_reason == "bundle_incomplete"

    frown = _mouth_candidate(valid, variant_id="mouth-frown", role="mouth_frown")
    complete = _mouth_quality_plan(tmp_path / "complete", frown, smile)

    assert complete.quality_eligible_variant_ids == ("mouth-frown", "mouth-smile")
    assert complete.bundle_results[0].complete is True
    assert complete.bundle_results[0].eligible is True


def test_bad_mouth_form_endpoint_rejects_the_whole_bundle(tmp_path: Path) -> None:
    valid = {(x, y) for y in range(9, 14) for x in range(12)}
    incomplete_coverage = {(x, y) for y in range(10, 12) for x in range(1, 6)}
    smile = _mouth_candidate(valid, variant_id="mouth-smile", role="mouth_smile")
    frown = _mouth_candidate(
        incomplete_coverage,
        variant_id="mouth-frown",
        role="mouth_frown",
    )

    plan = _mouth_quality_plan(tmp_path, smile, frown)

    assert plan.quality_eligible_variant_ids == ()
    bundle = plan.bundle_results[0]
    assert bundle.complete is True
    assert bundle.eligible is False
    assert bundle.rejection_reason == "coverage_leak"


def test_blink_bundle_accepts_exactly_two_singles_or_one_coupled_candidate(
    tmp_path: Path,
) -> None:
    eye = {(x, y) for y in range(20) for x in range(20)}
    singles = _eye_quality_plan(
        tmp_path / "singles",
        ("eye_closed.xmin", (2, 20, 22, 40), eye),
        ("eye_closed.xmax", (42, 20, 62, 40), eye),
    )
    coupled = _eye_quality_plan(
        tmp_path / "coupled",
        (
            "eye_closed.coupled",
            (2, 20, 62, 40),
            eye | {(x, y) for y in range(20) for x in range(40, 60)},
        ),
    )

    assert singles.bundle_results[0].bundle_id == "blink.native"
    assert singles.bundle_results[0].eligible is True
    assert singles.quality_eligible_variant_ids == ("blink-xmax", "blink-xmin")
    assert coupled.bundle_results[0].bundle_id == "blink.native"
    assert coupled.bundle_results[0].eligible is True
    assert coupled.quality_eligible_variant_ids == ("blink-coupled",)


def test_single_sided_blink_is_candidate_valid_but_atomic_bundle_incomplete(
    tmp_path: Path,
) -> None:
    eye = {(x, y) for y in range(20) for x in range(20)}

    plan = _eye_quality_plan(
        tmp_path,
        ("eye_closed.xmin", (2, 20, 22, 40), eye),
    )

    assert plan.candidate_results[0].eligible is True
    assert plan.bundle_results[0].complete is False
    assert plan.bundle_results[0].rejection_reason == "bundle_incomplete"
    assert plan.quality_eligible_variant_ids == ()


def test_native_variant_quality_plan_is_iteration_independent_and_self_validating(
    tmp_path: Path,
) -> None:
    valid = {(x, y) for y in range(9, 14) for x in range(12)}
    smile = _mouth_candidate(valid, variant_id="mouth-smile", role="mouth_smile")
    frown = _mouth_candidate(valid, variant_id="mouth-frown", role="mouth_frown")

    forward = _mouth_quality_plan(tmp_path / "forward", smile, frown)
    reversed_plan = _mouth_quality_plan(tmp_path / "reversed", frown, smile)

    assert reversed_plan == forward
    assert forward.plan_sha256 == jcs_sha256(forward.semantic_payload())
    for result in forward.candidate_results:
        assert result.result_sha256 == jcs_sha256(result.content_payload())
        for branch in result.branches:
            assert branch.occlusion_intrusion == pytest.approx(sum(entry.ratio for entry in branch.intrusion_by_part))
            assert branch.occlusion_intrusion == pytest.approx(branch.intrusion_face_ratio + branch.intrusion_other_parts_ratio)
    for bundle in forward.bundle_results:
        assert bundle.result_sha256 == jcs_sha256(bundle.content_payload())
