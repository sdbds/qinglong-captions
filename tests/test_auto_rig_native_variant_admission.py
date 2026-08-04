from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from module.auto_rig.artifacts import FileDigest
from module.auto_rig.component_plan import (
    MASK_COMPONENT_PLAN_VERSION,
    NATIVE_VARIANT_PARTITION_VERSION,
    MaskCleanupDescriptor,
    MaskComponentPlan,
    MaskComponentRecord,
    NativeVariantPartitionRecord,
    NormalizedMaskPart,
)
from module.auto_rig.draw_order import build_ordinary_draw_order
from module.auto_rig.jcs import jcs_sha256
from module.auto_rig.native_variant_admission import (
    DRAW_ANCHOR_EXPANDER_VERSION,
    FINAL_DRAW_ORDER_PLAN_VERSION,
    NATIVE_VARIANT_ADMISSION_PRIORITY,
    NATIVE_VARIANT_ELIGIBILITY_PLAN_VERSION,
    NativeVariantAdmissionError,
    admit_native_variant_resources,
    expand_draw_order_with_admitted_variants,
)
from module.auto_rig.native_variant_quality import (
    NATIVE_VARIANT_COMPOSITE_PLAN_VERSION,
    NATIVE_VARIANT_QUALITY_PLAN_VERSION,
    NativeVariantBundleQualityResult,
    NativeVariantQualityPlan,
    NativeVariantQualityResult,
)
from module.auto_rig.native_variants import NativeVariantCandidate, NativeVariantSet
from module.auto_rig.texture_plan import TextureRegionInput

_SHA256_ZERO = "sha256:" + "0" * 64
_SHA256_ONE = "sha256:" + "1" * 64


def _texture_region(
    part_id: str,
    width: int,
    height: int,
    *,
    variant_id: str | None = None,
) -> TextureRegionInput:
    return TextureRegionInput(
        part_id=part_id,
        source_kind="native_variant" if variant_id is not None else "see_through",
        variant_id=variant_id,
        source_xyxy=(0, 0, width, height),
        width=width,
        height=height,
        rgba_sha256=_SHA256_ZERO,
        alpha_mode="straight",
        color_space="srgb_bytes",
        pixel_contract_version="texture-pixel-rgba8-straight-srgb-v1",
    )


def _component(component_id: str, part_id: str, label: int = 1) -> MaskComponentRecord:
    return MaskComponentRecord(
        component_id=component_id,
        label=label,
        part_id=part_id,
        side=None,
        bbox=(0, 0, 1, 1),
        cleaned_binary_mask_sha256=_SHA256_ZERO,
        pixel_count=1,
    )


def _component_plan(
    base_regions: tuple[TextureRegionInput, ...],
    native_regions: tuple[TextureRegionInput, ...],
    *,
    base_component_count: int | None = None,
    native_component_counts: dict[str, int] | None = None,
) -> MaskComponentPlan:
    cleanup = MaskCleanupDescriptor(
        schema_version="mask-cleanup-v1",
        alpha_threshold_u8=1,
        connectivity=8,
        morphology="identity",
        min_component_area_px=4,
        max_hole_area_px=4,
        numpy_version="1.26.4",
        scikit_image_version="0.25.2",
        qcl_codec_version="canonical-label-map-qcl1",
        side_classifier_version="mask-side-classifier-v1",
        side_min_component_fraction_numerator=1,
        side_min_component_fraction_denominator=20,
        component_id_schema="mask-component-id-v1",
    )
    base_count = base_component_count if base_component_count is not None else len(base_regions)
    per_base = [1] * len(base_regions)
    if per_base:
        per_base[0] += base_count - len(base_regions)
    parts = []
    component_serial = 0
    for region, count in zip(base_regions, per_base, strict=True):
        components = []
        for label in range(1, count + 1):
            component_serial += 1
            components.append(
                _component(
                    f"component/c_{component_serial:064x}",
                    region.part_id,
                    label,
                )
            )
        parts.append(
            NormalizedMaskPart(
                part_id=region.part_id,
                source_part_id=region.part_id,
                source_tag=region.part_id.removeprefix("part/").replace("-", " "),
                base_tag=region.part_id.removeprefix("part/").replace("-", " "),
                semantic_slug=region.part_id.removeprefix("part/"),
                side=None,
                side_provenance="none",
                source_xyxy=region.source_xyxy,
                xyxy=(0, 0, 1, 1),
                depth_median=0.5,
                cleaned_binary_mask_sha256=_SHA256_ZERO,
                qcl_file=FileDigest(
                    path=f"rig/cache/A/components/{component_serial:064x}.qcl",
                    size=16,
                    sha256=_SHA256_ZERO,
                ),
                components=tuple(components),
            )
        )
    native_counts = native_component_counts or {region.variant_id: 1 for region in native_regions if region.variant_id is not None}
    partitions = []
    for region in native_regions:
        assert region.variant_id is not None
        components = []
        for label in range(1, native_counts[region.variant_id] + 1):
            component_serial += 1
            components.append(
                _component(
                    f"component/c_{component_serial:064x}",
                    region.part_id,
                    label,
                )
            )
        provisional = NativeVariantPartitionRecord(
            schema_version=NATIVE_VARIANT_PARTITION_VERSION,
            variant_id=region.variant_id,
            part_id=region.part_id,
            semantic_role="synthetic",
            status="ready",
            side_provenance="none",
            source_xyxy=region.source_xyxy,
            xyxy=(0, 0, 1, 1),
            cleaned_binary_mask_sha256=_SHA256_ZERO,
            qcl_file=FileDigest(
                path=f"rig/cache/A/components/{component_serial:064x}.qcl",
                size=16,
                sha256=_SHA256_ZERO,
            ),
            components=tuple(components),
            partition_sha256="",
        )
        partitions.append(
            replace(
                provisional,
                partition_sha256=jcs_sha256(provisional.content_payload()),
            )
        )
    provisional_plan = MaskComponentPlan(
        schema_version=MASK_COMPONENT_PLAN_VERSION,
        canvas_edge=1280,
        cleanup=cleanup,
        parts=tuple(parts),
        native_variant_set_sha256=_SHA256_ONE,
        variant_partitions=tuple(partitions),
        base_projected_component_count=base_count,
        native_variant_projected_component_count=sum(native_counts.values()),
        projected_component_count=base_count + sum(native_counts.values()),
        plan_sha256="",
    )
    return replace(
        provisional_plan,
        plan_sha256=jcs_sha256(provisional_plan.semantic_payload()),
    )


def _quality_result(variant_id: str, role: str, *, eligible: bool = True):
    provisional = NativeVariantQualityResult(
        variant_id=variant_id,
        semantic_role=role,
        eligible=eligible,
        rejection_reason=None if eligible else "coverage_leak",
        branches=(),
        result_sha256="",
    )
    return replace(provisional, result_sha256=jcs_sha256(provisional.content_payload()))


def _bundle(bundle_id: str, results: tuple[NativeVariantQualityResult, ...]):
    provisional = NativeVariantBundleQualityResult(
        bundle_id=bundle_id,
        semantic_roles=tuple(sorted(result.semantic_role for result in results)),
        variant_ids=tuple(sorted(result.variant_id for result in results)),
        complete=True,
        eligible=all(result.eligible for result in results),
        rejection_reason=None if all(result.eligible for result in results) else "coverage_leak",
        result_sha256="",
    )
    return replace(provisional, result_sha256=jcs_sha256(provisional.content_payload()))


def _quality_plan(*bundles: tuple[str, tuple[tuple[str, str], ...]]) -> NativeVariantQualityPlan:
    results = tuple(_quality_result(variant_id, role) for _, members in bundles for variant_id, role in members)
    by_id = {result.variant_id: result for result in results}
    bundle_results = tuple(
        _bundle(bundle_id, tuple(by_id[variant_id] for variant_id, _ in members)) for bundle_id, members in bundles
    )
    provisional = NativeVariantQualityPlan(
        schema_version=NATIVE_VARIANT_QUALITY_PLAN_VERSION,
        native_variant_set_sha256=_SHA256_ONE,
        component_plan_sha256="",
        draw_order_plan_sha256=_SHA256_ZERO,
        role_registry_sha256=_SHA256_ZERO,
        composite_plan_version=NATIVE_VARIANT_COMPOSITE_PLAN_VERSION,
        candidate_results=tuple(sorted(results, key=lambda result: result.variant_id)),
        bundle_results=bundle_results,
        quality_eligible_variant_ids=tuple(sorted(result.variant_id for result in results)),
        plan_sha256="",
    )
    return provisional


def _bind_quality_plan(
    quality: NativeVariantQualityPlan,
    component_plan: MaskComponentPlan,
) -> NativeVariantQualityPlan:
    provisional = replace(
        quality,
        component_plan_sha256=component_plan.plan_sha256,
        plan_sha256="",
    )
    return replace(provisional, plan_sha256=jcs_sha256(provisional.semantic_payload()))


def test_mandatory_base_texture_failure_is_a_hard_item_error() -> None:
    base = tuple(_texture_region(f"part/base{index}", 1192, 1192) for index in range(5))
    components = _component_plan(base, ())
    quality = _bind_quality_plan(_quality_plan(), components)

    with pytest.raises(NativeVariantAdmissionError) as captured:
        admit_native_variant_resources(base, (), components, quality)

    assert captured.value.code == "texture_budget_exceeded"


def test_base_drawable_overflow_is_reported_but_not_misclassified_as_optional_failure() -> None:
    base = (_texture_region("part/base", 64, 64),)
    components = _component_plan(base, (), base_component_count=1002)
    quality = _bind_quality_plan(_quality_plan(), components)

    eligibility = admit_native_variant_resources(base, (), components, quality)

    assert eligibility.schema_version == NATIVE_VARIANT_ELIGIBILITY_PLAN_VERSION
    assert eligibility.base_projected_drawable_count == 1002
    assert eligibility.final_projected_drawable_count == 1002
    assert eligibility.render_variant_ids == ()
    assert eligibility.admission_attempts == ()
    assert eligibility.mandatory_base_texture_plan.fit is True


def test_native_groups_repack_from_empty_and_continue_after_texture_rejection() -> None:
    assert NATIVE_VARIANT_ADMISSION_PRIORITY == (
        "blink.native",
        "mouth_crossfade.native",
        "mouth_open.native",
        "mouth_form.native",
    )
    base = tuple(_texture_region(f"part/base{index}", 1192, 1192) for index in range(4))
    native = (
        _texture_region("part/native.blink", 800, 800, variant_id="blink"),
        _texture_region("part/native.mouth-open", 900, 900, variant_id="mouth-open"),
        _texture_region("part/native.mouth-smile", 300, 300, variant_id="mouth-smile"),
        _texture_region("part/native.mouth-frown", 300, 300, variant_id="mouth-frown"),
    )
    components = _component_plan(base, native)
    quality = _bind_quality_plan(
        _quality_plan(
            ("blink.native", (("blink", "eye_closed.coupled"),)),
            ("mouth_open.native", (("mouth-open", "mouth_open"),)),
            (
                "mouth_form.native",
                (("mouth-smile", "mouth_smile"), ("mouth-frown", "mouth_frown")),
            ),
        ),
        components,
    )

    eligibility = admit_native_variant_resources(base, native, components, quality)

    assert tuple(attempt.bundle_id for attempt in eligibility.admission_attempts) == (
        "blink.native",
        "mouth_open.native",
        "mouth_form.native",
    )
    assert tuple(attempt.status for attempt in eligibility.admission_attempts) == (
        "admitted",
        "rejected_texture_budget",
        "admitted",
    )
    assert eligibility.render_variant_ids == ("blink", "mouth-frown", "mouth-smile")
    assert "part/native.mouth-open" not in eligibility.final_texture_plan.input_part_ids
    assert set(eligibility.final_texture_plan.input_part_ids) == {
        *(region.part_id for region in base),
        "part/native.blink",
        "part/native.mouth-frown",
        "part/native.mouth-smile",
    }
    assert eligibility.plan_sha256 == jcs_sha256(eligibility.semantic_payload())


def test_native_group_is_atomically_rejected_before_packing_when_drawables_exceed_1001() -> None:
    base = (_texture_region("part/base", 64, 64),)
    native = (_texture_region("part/native.blink", 64, 64, variant_id="blink"),)
    components = _component_plan(
        base,
        native,
        base_component_count=999,
        native_component_counts={"blink": 3},
    )
    quality = _bind_quality_plan(
        _quality_plan(("blink.native", (("blink", "eye_closed.coupled"),))),
        components,
    )

    eligibility = admit_native_variant_resources(base, native, components, quality)

    attempt = eligibility.admission_attempts[0]
    assert attempt.status == "rejected_drawable_budget"
    assert attempt.reason_code == "native_variant_drawable_budget"
    assert attempt.projected_drawable_count == 1002
    assert attempt.texture_plan is None
    assert eligibility.render_variant_ids == ()
    assert eligibility.final_texture_plan == eligibility.mandatory_base_texture_plan


def _draw_candidate(
    variant_id: str,
    role: str,
    anchor: str,
) -> NativeVariantCandidate:
    return NativeVariantCandidate(
        variant_id=variant_id,
        part_id=f"part/native.{variant_id}",
        semantic_role=role,
        composite_mode="occluding_overlay_v1",
        base_part_ids=(anchor,),
        draw_anchor_part_id=anchor,
        anchor_base_tag=anchor.removeprefix("part/"),
        anchor_depth_median=0.5,
        xyxy=(0, 0, 1, 1),
        relative_path=f"{variant_id}.png",
        png_path=Path(f"{variant_id}.png"),
        file_sha256=_SHA256_ZERO,
        rgba_mode="RGBA",
        alpha_mode="straight",
        color_space="sRGB",
        alpha_mass_u8_sum=255,
        alpha_u8=b"\xff",
    )


def _draw_variant_set(*candidates: NativeVariantCandidate) -> NativeVariantSet:
    provisional = NativeVariantSet(
        schema_version="native-variant-set-v1",
        present=True,
        manifest_path=None,
        entries=tuple(sorted(candidates, key=lambda item: item.variant_id)),
        native_variant_set_sha256="",
    )
    return replace(
        provisional,
        native_variant_set_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def test_final_draw_order_expands_only_admitted_variants_inside_anchor_bundle() -> None:
    base = tuple(_texture_region(part_id, 64, 64) for part_id in ("part/back-hair", "part/face", "part/front-hair"))
    native = (
        _texture_region("part/native.smile", 32, 32, variant_id="smile"),
        _texture_region("part/native.frown", 32, 32, variant_id="frown"),
    )
    components = _component_plan(base, native)
    quality = _bind_quality_plan(
        _quality_plan(
            (
                "mouth_form.native",
                (("smile", "mouth_smile"), ("frown", "mouth_frown")),
            )
        ),
        components,
    )
    eligibility = admit_native_variant_resources(base, native, components, quality)
    ordinary = build_ordinary_draw_order(components.parts)
    anchor = "part/face"
    variants = _draw_variant_set(
        _draw_candidate("smile", "mouth_smile", anchor),
        _draw_candidate("frown", "mouth_frown", anchor),
        _draw_candidate("rejected", "mouth_open", anchor),
    )

    final = expand_draw_order_with_admitted_variants(ordinary, variants, eligibility)

    anchor_index = final.part_order.index(anchor)
    assert final.schema_version == FINAL_DRAW_ORDER_PLAN_VERSION
    assert final.anchor_expander_version == DRAW_ANCHOR_EXPANDER_VERSION
    assert final.part_order[anchor_index : anchor_index + 3] == (
        anchor,
        "part/native.frown",
        "part/native.smile",
    )
    assert "part/native.rejected" not in final.part_order
    ordinary_in_final = tuple(part_id for part_id in final.part_order if not part_id.startswith("part/native."))
    assert ordinary_in_final == ordinary.ordinary_part_order
    assert tuple(record.part_draw_rank for record in final.records) == tuple(range(len(final.records)))
    assert final.plan_sha256 == jcs_sha256(final.semantic_payload())
