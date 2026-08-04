from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from .component_plan import MaskComponentPlan
from .draw_order import OrdinaryDrawOrderPlan
from .jcs import jcs_sha256
from .native_variant_quality import (
    NativeVariantBundleQualityResult,
    NativeVariantQualityPlan,
)
from .native_variants import NativeVariantCandidate, NativeVariantSet
from .texture_plan import TexturePagePlan, TextureRegionInput, build_texture_page_plan

NATIVE_VARIANT_ELIGIBILITY_PLAN_VERSION = "native-variant-eligibility-plan-v1"
NATIVE_VARIANT_ADMISSION_POLICY_VERSION = "native-variant-admission-policy-v1"
NATIVE_VARIANT_ADMISSION_PRIORITY = (
    "blink.native",
    "mouth_crossfade.native",
    "mouth_open.native",
    "mouth_form.native",
)
LIVE2D_PROJECTED_DRAWABLE_LIMIT = 1001
FINAL_DRAW_ORDER_PLAN_VERSION = "final-draw-order-plan-v1"
DRAW_ANCHOR_EXPANDER_VERSION = "native-variant-anchor-expander-v1"


class NativeVariantAdmissionError(ValueError):
    """Raised when mandatory resources or admission inputs violate their contract."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _texture_plan_payload(plan: TexturePagePlan) -> dict[str, object]:
    return {**plan.semantic_payload(), "plan_sha256": plan.plan_sha256}


@dataclass(frozen=True, slots=True)
class NativeVariantAdmissionAttempt:
    bundle_id: str
    variant_ids: tuple[str, ...]
    status: Literal[
        "admitted",
        "rejected_quality",
        "rejected_texture_budget",
        "rejected_drawable_budget",
    ]
    reason_code: str | None
    projected_drawable_count: int
    quality_result_sha256: str
    texture_plan: TexturePagePlan | None

    def to_dict(self) -> dict[str, object]:
        return {
            "bundle_id": self.bundle_id,
            "variant_ids": list(self.variant_ids),
            "status": self.status,
            "reason_code": self.reason_code,
            "projected_drawable_count": self.projected_drawable_count,
            "quality_result_sha256": self.quality_result_sha256,
            "texture_plan": (_texture_plan_payload(self.texture_plan) if self.texture_plan is not None else None),
        }


@dataclass(frozen=True, slots=True)
class NativeVariantEligibilityPlan:
    schema_version: str
    admission_policy_version: str
    admission_priority: tuple[str, ...]
    component_plan_sha256: str
    native_variant_quality_plan_sha256: str
    base_projected_drawable_count: int
    mandatory_base_texture_plan: TexturePagePlan
    admission_attempts: tuple[NativeVariantAdmissionAttempt, ...]
    render_variant_ids: tuple[str, ...]
    final_projected_drawable_count: int
    final_texture_plan: TexturePagePlan
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "admission_policy_version": self.admission_policy_version,
            "admission_priority": list(self.admission_priority),
            "component_plan_sha256": self.component_plan_sha256,
            "native_variant_quality_plan_sha256": self.native_variant_quality_plan_sha256,
            "base_projected_drawable_count": self.base_projected_drawable_count,
            "mandatory_base_texture_plan": _texture_plan_payload(self.mandatory_base_texture_plan),
            "admission_attempts": [attempt.to_dict() for attempt in self.admission_attempts],
            "render_variant_ids": list(self.render_variant_ids),
            "final_projected_drawable_count": self.final_projected_drawable_count,
            "final_texture_plan": _texture_plan_payload(self.final_texture_plan),
        }


@dataclass(frozen=True, slots=True)
class FinalPartDrawRecord:
    part_id: str
    source_kind: Literal["see_through", "native_variant"]
    part_draw_rank: int
    depth_bucket: int
    draw_anchor_part_id: str | None
    draw_bundle_id: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "part_id": self.part_id,
            "source_kind": self.source_kind,
            "part_draw_rank": self.part_draw_rank,
            "depth_bucket": self.depth_bucket,
            "draw_anchor_part_id": self.draw_anchor_part_id,
            "draw_bundle_id": self.draw_bundle_id,
        }


@dataclass(frozen=True, slots=True)
class FinalDrawOrderPlan:
    schema_version: str
    anchor_expander_version: str
    ordinary_draw_order_plan_sha256: str
    native_variant_set_sha256: str
    native_variant_eligibility_plan_sha256: str
    part_order: tuple[str, ...]
    records: tuple[FinalPartDrawRecord, ...]
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "anchor_expander_version": self.anchor_expander_version,
            "ordinary_draw_order_plan_sha256": self.ordinary_draw_order_plan_sha256,
            "native_variant_set_sha256": self.native_variant_set_sha256,
            "native_variant_eligibility_plan_sha256": self.native_variant_eligibility_plan_sha256,
            "part_order": list(self.part_order),
            "records": [record.to_dict() for record in self.records],
        }


def _error(code: str, message: str) -> NativeVariantAdmissionError:
    return NativeVariantAdmissionError(code, message)


def _validate_quality_plan(plan: NativeVariantQualityPlan) -> None:
    if jcs_sha256(plan.semantic_payload()) != plan.plan_sha256:
        raise _error("invalid_native_variant_admission", "quality-plan digest is invalid")
    candidate_by_id = {result.variant_id: result for result in plan.candidate_results}
    if len(candidate_by_id) != len(plan.candidate_results):
        raise _error("invalid_native_variant_admission", "quality candidate IDs are not unique")
    for result in plan.candidate_results:
        if jcs_sha256(result.content_payload()) != result.result_sha256:
            raise _error("invalid_native_variant_admission", "quality candidate digest is invalid")
    bundle_ids = tuple(bundle.bundle_id for bundle in plan.bundle_results)
    if len(bundle_ids) != len(set(bundle_ids)) or any(
        bundle_id not in NATIVE_VARIANT_ADMISSION_PRIORITY for bundle_id in bundle_ids
    ):
        raise _error("invalid_native_variant_admission", "quality bundle IDs are invalid")
    eligible_from_bundles: set[str] = set()
    for bundle in plan.bundle_results:
        if jcs_sha256(bundle.content_payload()) != bundle.result_sha256:
            raise _error("invalid_native_variant_admission", "quality bundle digest is invalid")
        if any(variant_id not in candidate_by_id for variant_id in bundle.variant_ids):
            raise _error("invalid_native_variant_admission", "quality bundle references unknown candidate")
        if bundle.eligible:
            eligible_from_bundles.update(bundle.variant_ids)
    if tuple(sorted(eligible_from_bundles)) != plan.quality_eligible_variant_ids:
        raise _error("invalid_native_variant_admission", "quality eligible IDs differ from bundles")


def _validate_inputs(
    base_regions: tuple[TextureRegionInput, ...],
    native_regions: tuple[TextureRegionInput, ...],
    component_plan: MaskComponentPlan,
    quality_plan: NativeVariantQualityPlan,
) -> tuple[
    dict[str, TextureRegionInput],
    dict[str, int],
    dict[str, NativeVariantBundleQualityResult],
]:
    if jcs_sha256(component_plan.semantic_payload()) != component_plan.plan_sha256:
        raise _error("invalid_native_variant_admission", "component-plan digest is invalid")
    _validate_quality_plan(quality_plan)
    if quality_plan.component_plan_sha256 != component_plan.plan_sha256:
        raise _error("invalid_native_variant_admission", "quality plan belongs to another component plan")
    if quality_plan.native_variant_set_sha256 != component_plan.native_variant_set_sha256:
        raise _error("invalid_native_variant_admission", "quality/component variant-set digests differ")
    expected_base_ids = {part.part_id for part in component_plan.parts}
    if {region.part_id for region in base_regions} != expected_base_ids or any(
        region.source_kind != "see_through" for region in base_regions
    ):
        raise _error("invalid_native_variant_admission", "base texture regions differ from base payloads")
    native_by_id = {region.variant_id: region for region in native_regions if region.variant_id is not None}
    if len(native_by_id) != len(native_regions) or any(region.source_kind != "native_variant" for region in native_regions):
        raise _error("invalid_native_variant_admission", "native texture regions are invalid")
    expected_native_ids = {result.variant_id for result in quality_plan.candidate_results}
    if set(native_by_id) != expected_native_ids:
        raise _error("invalid_native_variant_admission", "native texture regions differ from quality candidates")
    partition_by_id = {partition.variant_id: partition for partition in component_plan.variant_partitions}
    if set(partition_by_id) != expected_native_ids:
        raise _error("invalid_native_variant_admission", "variant partitions differ from quality candidates")
    for variant_id, region in native_by_id.items():
        if partition_by_id[variant_id].part_id != region.part_id:
            raise _error("invalid_native_variant_admission", "variant region Part differs from partition")
    component_counts = {variant_id: len(partition.components) for variant_id, partition in partition_by_id.items()}
    bundles = {bundle.bundle_id: bundle for bundle in quality_plan.bundle_results}
    return native_by_id, component_counts, bundles


def admit_native_variant_resources(
    base_regions: Iterable[TextureRegionInput],
    native_regions: Iterable[TextureRegionInput],
    component_plan: MaskComponentPlan,
    quality_plan: NativeVariantQualityPlan,
) -> NativeVariantEligibilityPlan:
    """Apply shared texture and drawable budgets to complete quality bundles."""

    base = tuple(base_regions)
    native = tuple(native_regions)
    native_by_id, component_counts, bundles = _validate_inputs(
        base,
        native,
        component_plan,
        quality_plan,
    )
    mandatory_plan = build_texture_page_plan(base)
    if not mandatory_plan.fit:
        raise _error(
            "texture_budget_exceeded",
            f"mandatory base texture set does not fit: {mandatory_plan.failure_reason}",
        )

    accepted: set[str] = set()
    attempts: list[NativeVariantAdmissionAttempt] = []
    base_count = component_plan.base_projected_component_count
    for bundle_id in NATIVE_VARIANT_ADMISSION_PRIORITY:
        bundle = bundles.get(bundle_id)
        if bundle is None:
            continue
        variant_ids = tuple(sorted(bundle.variant_ids))
        projected_count = base_count + sum(component_counts[variant_id] for variant_id in accepted | set(variant_ids))
        if not bundle.eligible:
            attempts.append(
                NativeVariantAdmissionAttempt(
                    bundle_id=bundle_id,
                    variant_ids=variant_ids,
                    status="rejected_quality",
                    reason_code=bundle.rejection_reason,
                    projected_drawable_count=projected_count,
                    quality_result_sha256=bundle.result_sha256,
                    texture_plan=None,
                )
            )
            continue
        if projected_count > LIVE2D_PROJECTED_DRAWABLE_LIMIT:
            attempts.append(
                NativeVariantAdmissionAttempt(
                    bundle_id=bundle_id,
                    variant_ids=variant_ids,
                    status="rejected_drawable_budget",
                    reason_code="native_variant_drawable_budget",
                    projected_drawable_count=projected_count,
                    quality_result_sha256=bundle.result_sha256,
                    texture_plan=None,
                )
            )
            continue
        attempted_ids = accepted | set(variant_ids)
        attempted_regions = (*base, *(native_by_id[variant_id] for variant_id in sorted(attempted_ids)))
        texture_plan = build_texture_page_plan(attempted_regions)
        if not texture_plan.fit:
            attempts.append(
                NativeVariantAdmissionAttempt(
                    bundle_id=bundle_id,
                    variant_ids=variant_ids,
                    status="rejected_texture_budget",
                    reason_code="native_variant_texture_budget",
                    projected_drawable_count=projected_count,
                    quality_result_sha256=bundle.result_sha256,
                    texture_plan=texture_plan,
                )
            )
            continue
        accepted.update(variant_ids)
        attempts.append(
            NativeVariantAdmissionAttempt(
                bundle_id=bundle_id,
                variant_ids=variant_ids,
                status="admitted",
                reason_code=None,
                projected_drawable_count=projected_count,
                quality_result_sha256=bundle.result_sha256,
                texture_plan=texture_plan,
            )
        )

    render_variant_ids = tuple(sorted(accepted))
    final_regions = (*base, *(native_by_id[variant_id] for variant_id in render_variant_ids))
    final_texture_plan = build_texture_page_plan(final_regions)
    if not final_texture_plan.fit:
        raise _error("invalid_native_variant_admission", "final admitted texture plan does not fit")
    last_admitted = next(
        (attempt for attempt in reversed(attempts) if attempt.status == "admitted"),
        None,
    )
    expected_final = last_admitted.texture_plan if last_admitted is not None else mandatory_plan
    if final_texture_plan != expected_final:
        raise _error("invalid_native_variant_admission", "final replay differs from admitted snapshot")
    final_count = base_count + sum(component_counts[variant_id] for variant_id in accepted)
    semantic_payload = {
        "schema_version": NATIVE_VARIANT_ELIGIBILITY_PLAN_VERSION,
        "admission_policy_version": NATIVE_VARIANT_ADMISSION_POLICY_VERSION,
        "admission_priority": list(NATIVE_VARIANT_ADMISSION_PRIORITY),
        "component_plan_sha256": component_plan.plan_sha256,
        "native_variant_quality_plan_sha256": quality_plan.plan_sha256,
        "base_projected_drawable_count": base_count,
        "mandatory_base_texture_plan": _texture_plan_payload(mandatory_plan),
        "admission_attempts": [attempt.to_dict() for attempt in attempts],
        "render_variant_ids": list(render_variant_ids),
        "final_projected_drawable_count": final_count,
        "final_texture_plan": _texture_plan_payload(final_texture_plan),
    }
    return NativeVariantEligibilityPlan(
        schema_version=NATIVE_VARIANT_ELIGIBILITY_PLAN_VERSION,
        admission_policy_version=NATIVE_VARIANT_ADMISSION_POLICY_VERSION,
        admission_priority=NATIVE_VARIANT_ADMISSION_PRIORITY,
        component_plan_sha256=component_plan.plan_sha256,
        native_variant_quality_plan_sha256=quality_plan.plan_sha256,
        base_projected_drawable_count=base_count,
        mandatory_base_texture_plan=mandatory_plan,
        admission_attempts=tuple(attempts),
        render_variant_ids=render_variant_ids,
        final_projected_drawable_count=final_count,
        final_texture_plan=final_texture_plan,
        plan_sha256=jcs_sha256(semantic_payload),
    )


def expand_draw_order_with_admitted_variants(
    ordinary_plan: OrdinaryDrawOrderPlan,
    variant_set: NativeVariantSet,
    eligibility_plan: NativeVariantEligibilityPlan,
) -> FinalDrawOrderPlan:
    """Expand admitted variants immediately above their ordinary draw anchor."""

    if jcs_sha256(ordinary_plan.semantic_payload()) != ordinary_plan.plan_sha256:
        raise _error("invalid_native_variant_admission", "ordinary draw-order digest is invalid")
    if jcs_sha256(variant_set.semantic_payload()) != variant_set.native_variant_set_sha256:
        raise _error("invalid_native_variant_admission", "NativeVariant set digest is invalid")
    if jcs_sha256(eligibility_plan.semantic_payload()) != eligibility_plan.plan_sha256:
        raise _error("invalid_native_variant_admission", "eligibility-plan digest is invalid")
    candidate_by_id = {candidate.variant_id: candidate for candidate in variant_set.entries}
    if len(candidate_by_id) != len(variant_set.entries) or any(
        variant_id not in candidate_by_id for variant_id in eligibility_plan.render_variant_ids
    ):
        raise _error("invalid_native_variant_admission", "render variants differ from candidate set")
    ordinary_ids = set(ordinary_plan.ordinary_part_order)
    ordinary_records = {record.part_id: record for record in ordinary_plan.records}
    if set(ordinary_records) != ordinary_ids:
        raise _error("invalid_native_variant_admission", "ordinary draw records differ from order")
    by_anchor: dict[str, list[NativeVariantCandidate]] = {}
    for variant_id in eligibility_plan.render_variant_ids:
        candidate = candidate_by_id[variant_id]
        if candidate.draw_anchor_part_id not in ordinary_ids:
            raise _error("invalid_native_variant_admission", "variant draw anchor is not ordinary")
        if candidate.draw_anchor_part_id not in candidate.base_part_ids:
            raise _error("invalid_native_variant_admission", "variant draw anchor is outside base set")
        by_anchor.setdefault(candidate.draw_anchor_part_id, []).append(candidate)

    bundle_id_by_anchor = {
        anchor: "draw-bundle/b_"
        + jcs_sha256(
            {
                "schema_version": DRAW_ANCHOR_EXPANDER_VERSION,
                "draw_anchor_part_id": anchor,
                "variant_ids": sorted(candidate.variant_id for candidate in candidates),
            }
        ).removeprefix("sha256:")
        for anchor, candidates in by_anchor.items()
    }
    final_ids: list[str] = []
    variant_by_part_id = {}
    for ordinary_id in ordinary_plan.ordinary_part_order:
        final_ids.append(ordinary_id)
        candidates = sorted(
            by_anchor.get(ordinary_id, ()),
            key=lambda candidate: (candidate.semantic_role, candidate.variant_id),
        )
        for candidate in candidates:
            if candidate.part_id in ordinary_ids or candidate.part_id in variant_by_part_id:
                raise _error("invalid_native_variant_admission", "final draw Part IDs collide")
            variant_by_part_id[candidate.part_id] = candidate
            final_ids.append(candidate.part_id)

    records: list[FinalPartDrawRecord] = []
    for rank, part_id in enumerate(final_ids):
        candidate = variant_by_part_id.get(part_id)
        if candidate is None:
            ordinary = ordinary_records[part_id]
            records.append(
                FinalPartDrawRecord(
                    part_id=part_id,
                    source_kind="see_through",
                    part_draw_rank=rank,
                    depth_bucket=ordinary.depth_bucket,
                    draw_anchor_part_id=None,
                    draw_bundle_id=bundle_id_by_anchor.get(part_id),
                )
            )
        else:
            anchor = candidate.draw_anchor_part_id
            records.append(
                FinalPartDrawRecord(
                    part_id=part_id,
                    source_kind="native_variant",
                    part_draw_rank=rank,
                    depth_bucket=ordinary_records[anchor].depth_bucket,
                    draw_anchor_part_id=anchor,
                    draw_bundle_id=bundle_id_by_anchor[anchor],
                )
            )
    semantic_payload = {
        "schema_version": FINAL_DRAW_ORDER_PLAN_VERSION,
        "anchor_expander_version": DRAW_ANCHOR_EXPANDER_VERSION,
        "ordinary_draw_order_plan_sha256": ordinary_plan.plan_sha256,
        "native_variant_set_sha256": variant_set.native_variant_set_sha256,
        "native_variant_eligibility_plan_sha256": eligibility_plan.plan_sha256,
        "part_order": final_ids,
        "records": [record.to_dict() for record in records],
    }
    return FinalDrawOrderPlan(
        schema_version=FINAL_DRAW_ORDER_PLAN_VERSION,
        anchor_expander_version=DRAW_ANCHOR_EXPANDER_VERSION,
        ordinary_draw_order_plan_sha256=ordinary_plan.plan_sha256,
        native_variant_set_sha256=variant_set.native_variant_set_sha256,
        native_variant_eligibility_plan_sha256=eligibility_plan.plan_sha256,
        part_order=tuple(final_ids),
        records=tuple(records),
        plan_sha256=jcs_sha256(semantic_payload),
    )


__all__ = [
    "LIVE2D_PROJECTED_DRAWABLE_LIMIT",
    "DRAW_ANCHOR_EXPANDER_VERSION",
    "FINAL_DRAW_ORDER_PLAN_VERSION",
    "NATIVE_VARIANT_ADMISSION_POLICY_VERSION",
    "NATIVE_VARIANT_ADMISSION_PRIORITY",
    "NATIVE_VARIANT_ELIGIBILITY_PLAN_VERSION",
    "NativeVariantAdmissionAttempt",
    "NativeVariantAdmissionError",
    "NativeVariantEligibilityPlan",
    "FinalDrawOrderPlan",
    "FinalPartDrawRecord",
    "admit_native_variant_resources",
    "expand_draw_order_with_admitted_variants",
]
