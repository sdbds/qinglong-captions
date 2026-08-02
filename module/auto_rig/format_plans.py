from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .capabilities import CapabilityPlan
from .control_bindings import ControlBinding, ControlBindingPlan
from .control_registry import ControlRegistryPlan, validate_control_registry_plan
from .export_symbols import (
    GlobalExportSymbolTable,
    validate_global_export_symbol_table,
)
from .jcs import jcs_sha256
from .preset_library import PresetLibraryPlan, validate_preset_library_plan
from .primitive_candidates import (
    PrimitiveCandidate,
    PrimitiveCandidateSet,
    validate_primitive_candidate_set,
)
from .rig_geometry import RigGeometryCache, validate_rig_geometry_cache

CAPABILITY_PROFILE_SCHEMA_VERSION = "capability-profile-v1"
CAPABILITY_PROFILE_REGISTRY_VERSION = "capability-profile-registry-v1"
FORMAT_MODEL_PLAN_VERSION = "format-model-plan-v1"
FORMAT_CAPABILITY_PREFLIGHT_VERSION = "format-capability-preflight-v1"
FORMAT_PRESET_SET_PLAN_VERSION = "format-preset-set-plan-v1"
FORMAT_PLAN_SET_VERSION = "format-plan-set-v1"
SPINE_FORMAT_PLANNER_VERSION = "spine-format-planner-v1"
LIVE2D_FORMAT_PLANNER_VERSION = "live2d-format-planner-v1"

_FORMATS = ("live2d_moc3_v4_00", "spine_4_2")
_CORE_REQUIRED = ("breath", "head_nod", "head_shake", "idle")
_AVATAR_REQUIRED = (
    "blink",
    "breath",
    "happy",
    "head_nod",
    "head_shake",
    "idle",
    "sad",
    "surprised",
    "talk",
)
_OPTIONAL_PRIORITY = (
    "body_sway",
    "blink",
    "talk",
    "surprised",
    "happy",
    "sad",
    "wave.xmin",
    "wave.xmax",
)

DecisionStatus = Literal["supported", "omitted"]
ModelStatus = Literal["supported", "unsupported"]


class FormatPlanError(ValueError):
    """Raised when a profile cannot be encoded by its required formats."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _error(code: str, message: str) -> FormatPlanError:
    return FormatPlanError(code, message)


@dataclass(frozen=True, slots=True)
class CapabilityProfile:
    schema_version: str
    registry_version: str
    profile_id: str
    required_formats: tuple[str, ...]
    required_preset_ids: tuple[str, ...]
    optional_preset_priority: tuple[str, ...]
    optional_preset_parity: str
    strict_capabilities: bool
    terminal_delivery: bool
    profile_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "registry_version": self.registry_version,
            "profile_id": self.profile_id,
            "required_formats": list(self.required_formats),
            "required_preset_ids": list(self.required_preset_ids),
            "optional_preset_priority": list(self.optional_preset_priority),
            "optional_preset_parity": self.optional_preset_parity,
            "strict_capabilities": self.strict_capabilities,
            "terminal_delivery": self.terminal_delivery,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "profile_sha256": self.profile_sha256}


@dataclass(frozen=True, slots=True)
class FormatModelPlan:
    schema_version: str
    planner_version: str
    format_id: str
    status: ModelStatus
    reason_codes: tuple[str, ...]
    component_count: int
    component_draw_ranks: tuple[int, ...]
    bone_count: int
    part_count: int
    texture_page_count: int
    candidate_count: int
    symbol_count: int
    planner_input_sha256: str
    planner_output_sha256: str
    plan_sha256: str

    def output_payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "component_count": self.component_count,
            "component_draw_ranks": list(self.component_draw_ranks),
            "bone_count": self.bone_count,
            "part_count": self.part_count,
            "texture_page_count": self.texture_page_count,
            "candidate_count": self.candidate_count,
            "symbol_count": self.symbol_count,
        }

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "planner_version": self.planner_version,
            "format_id": self.format_id,
            **self.output_payload(),
            "planner_input_sha256": self.planner_input_sha256,
            "planner_output_sha256": self.planner_output_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "plan_sha256": self.plan_sha256}


@dataclass(frozen=True, slots=True)
class FormatPresetDecision:
    schema_version: str
    planner_version: str
    format_id: str
    preset_id: str
    required: bool
    status: DecisionStatus
    reason: str | None
    quality_tier: str
    selected_implementation_ids: tuple[str, ...]
    selected_binding_ids: tuple[str, ...]
    selected_candidate_ids: tuple[str, ...]
    selected_primitive_key_sha256: tuple[str, ...]
    artifact_candidate_id: str | None
    artifact_symbol_id: str | None
    artifact_export_name: str | None
    conflicts_with: tuple[str, ...]
    failed_primitive_keys: tuple[str, ...]
    incompatible_with: tuple[str, ...]
    planner_input_sha256: str
    planner_output_sha256: str
    decision_sha256: str

    def output_payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "quality_tier": self.quality_tier,
            "selected_implementation_ids": list(self.selected_implementation_ids),
            "selected_binding_ids": list(self.selected_binding_ids),
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "selected_primitive_key_sha256": list(
                self.selected_primitive_key_sha256
            ),
            "artifact_candidate_id": self.artifact_candidate_id,
            "artifact_symbol_id": self.artifact_symbol_id,
            "artifact_export_name": self.artifact_export_name,
            "conflicts_with": list(self.conflicts_with),
            "failed_primitive_keys": list(self.failed_primitive_keys),
            "incompatible_with": list(self.incompatible_with),
        }

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "planner_version": self.planner_version,
            "format_id": self.format_id,
            "preset_id": self.preset_id,
            "required": self.required,
            **self.output_payload(),
            "planner_input_sha256": self.planner_input_sha256,
            "planner_output_sha256": self.planner_output_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "decision_sha256": self.decision_sha256}


@dataclass(frozen=True, slots=True)
class FormatPresetSetPlan:
    schema_version: str
    planner_version: str
    selection_version: str
    format_id: str
    profile_id: str
    model_plan_sha256: str
    required_preset_ids: tuple[str, ...]
    optional_priority: tuple[str, ...]
    decisions: tuple[FormatPresetDecision, ...]
    selected_preset_ids: tuple[str, ...]
    selected_candidate_ids: tuple[str, ...]
    selected_primitive_key_sha256: tuple[str, ...]
    primitive_union_sha256: str
    planner_input_sha256: str
    planner_output_sha256: str
    plan_sha256: str

    def output_payload(self) -> dict[str, object]:
        return {
            "decisions": [decision.to_dict() for decision in self.decisions],
            "selected_preset_ids": list(self.selected_preset_ids),
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "selected_primitive_key_sha256": list(
                self.selected_primitive_key_sha256
            ),
            "primitive_union_sha256": self.primitive_union_sha256,
        }

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "planner_version": self.planner_version,
            "selection_version": self.selection_version,
            "format_id": self.format_id,
            "profile_id": self.profile_id,
            "model_plan_sha256": self.model_plan_sha256,
            "required_preset_ids": list(self.required_preset_ids),
            "optional_priority": list(self.optional_priority),
            **self.output_payload(),
            "planner_input_sha256": self.planner_input_sha256,
            "planner_output_sha256": self.planner_output_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "plan_sha256": self.plan_sha256}


@dataclass(frozen=True, slots=True)
class FormatPlanSet:
    schema_version: str
    profile: CapabilityProfile
    rig_geometry_cache_sha256: str
    capability_plan_sha256: str
    control_binding_plan_sha256: str
    primitive_candidate_set_sha256: str
    global_symbol_table_sha256: str
    model_plans: tuple[FormatModelPlan, ...]
    preset_set_plans: tuple[FormatPresetSetPlan, ...]
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile.to_dict(),
            "rig_geometry_cache_sha256": self.rig_geometry_cache_sha256,
            "capability_plan_sha256": self.capability_plan_sha256,
            "control_binding_plan_sha256": self.control_binding_plan_sha256,
            "primitive_candidate_set_sha256": self.primitive_candidate_set_sha256,
            "global_symbol_table_sha256": self.global_symbol_table_sha256,
            "model_plans": [plan.to_dict() for plan in self.model_plans],
            "preset_set_plans": [plan.to_dict() for plan in self.preset_set_plans],
        }


def _profile(
    profile_id: str,
    required_formats: tuple[str, ...],
    required_presets: tuple[str, ...],
    *,
    terminal_delivery: bool,
) -> CapabilityProfile:
    values = {
        "schema_version": CAPABILITY_PROFILE_SCHEMA_VERSION,
        "registry_version": CAPABILITY_PROFILE_REGISTRY_VERSION,
        "profile_id": profile_id,
        "required_formats": tuple(sorted(required_formats)),
        "required_preset_ids": tuple(sorted(required_presets)),
        "optional_preset_priority": _OPTIONAL_PRIORITY,
        "optional_preset_parity": "per_format",
        "strict_capabilities": True,
        "terminal_delivery": terminal_delivery,
    }
    provisional = CapabilityProfile(**values, profile_sha256="")
    return CapabilityProfile(
        **values,
        profile_sha256=jcs_sha256(provisional.semantic_payload()),
    )


_PROFILES = {
    profile.profile_id: profile
    for profile in (
        _profile(
            "dual_runtime_core_v1",
            _FORMATS,
            _CORE_REQUIRED,
            terminal_delivery=True,
        ),
        _profile(
            "dual_runtime_avatar_v1",
            _FORMATS,
            _AVATAR_REQUIRED,
            terminal_delivery=True,
        ),
        _profile(
            "spine_4_2_dev",
            ("spine_4_2",),
            _CORE_REQUIRED,
            terminal_delivery=False,
        ),
    )
}


def load_capability_profile(profile_id: str) -> CapabilityProfile:
    try:
        profile = _PROFILES[profile_id]
    except KeyError as exc:
        raise _error(
            "invalid_capability_profile", f"unknown capability profile: {profile_id}"
        ) from exc
    if profile.profile_sha256 != jcs_sha256(profile.semantic_payload()):
        raise _error("invalid_capability_profile", "profile registry digest mismatch")
    return profile


def _live2d_draw_order_capacity(ranks: tuple[int, ...]) -> str | None:
    if ranks != tuple(range(len(ranks))):
        return "invalid_component_draw_rank"
    if len(ranks) > 1001:
        return "draw_order_capacity_exceeded"
    return None


def _validate_inputs(
    cache: RigGeometryCache,
    controls: ControlRegistryPlan,
    presets: PresetLibraryPlan,
    capabilities: CapabilityPlan,
    bindings: ControlBindingPlan,
    candidates: PrimitiveCandidateSet,
    symbols: GlobalExportSymbolTable,
) -> None:
    validate_rig_geometry_cache(cache)
    validate_control_registry_plan(controls)
    validate_preset_library_plan(presets, controls)
    if (
        capabilities.plan_sha256 != jcs_sha256(capabilities.semantic_payload())
        or capabilities.rig_geometry_cache_sha256 != cache.cache_sha256
        or capabilities.preset_library_sha256 != presets.plan_sha256
    ):
        raise _error("format_plan_mismatch", "capability plan provenance is invalid")
    if (
        bindings.plan_sha256 != jcs_sha256(bindings.semantic_payload())
        or bindings.rig_geometry_cache_sha256 != cache.cache_sha256
        or bindings.control_registry_sha256 != controls.registry_sha256
        or bindings.preset_library_sha256 != presets.plan_sha256
        or bindings.capability_plan_sha256 != capabilities.plan_sha256
    ):
        raise _error("format_plan_mismatch", "control binding provenance is invalid")
    validate_primitive_candidate_set(
        candidates,
        cache,
        controls,
        presets,
        bindings,
        texture_page_ids=candidates.texture_page_ids,
    )
    validate_global_export_symbol_table(symbols, candidates, controls)


def _model_plan(
    format_id: str,
    cache: RigGeometryCache,
    candidates: PrimitiveCandidateSet,
    symbols: GlobalExportSymbolTable,
) -> FormatModelPlan:
    planner_version = (
        SPINE_FORMAT_PLANNER_VERSION
        if format_id == "spine_4_2"
        else LIVE2D_FORMAT_PLANNER_VERSION
    )
    format_candidates = tuple(
        item for item in candidates.candidates if item.format_id == format_id
    )
    model_candidates = tuple(
        item for item in format_candidates if item.binding_template is None
    )
    symbol_key_ids = {
        symbol.typed_primitive_key.key_sha256 for symbol in symbols.symbols
    }
    reasons: set[str] = set()
    if any(
        candidate.typed_primitive_key.key_sha256 not in symbol_key_ids
        for candidate in format_candidates
    ):
        reasons.add("missing_export_symbol")
    mesh_count = len(cache.skinning_plan.weighted_meshes)
    ranks = tuple(
        record.component_draw_rank for record in cache.component_draw_order.records
    )
    if format_id == "spine_4_2":
        required_counts = {
            "spine_bone": len(cache.bone_graph.bones),
            "spine_slot": mesh_count,
            "spine_attachment_key": mesh_count,
            "spine_attachment_object": mesh_count,
            "spine_atlas_region": len(cache.parts),
            "spine_skin": 1,
        }
    else:
        required_counts = {
            "live2d_part": len(cache.parts),
            "live2d_artmesh": mesh_count,
        }
        capacity_reason = _live2d_draw_order_capacity(ranks)
        if capacity_reason is not None:
            reasons.add(capacity_reason)
    observed_counts: dict[str, int] = {}
    for candidate in model_candidates:
        observed_counts[candidate.candidate_kind] = (
            observed_counts.get(candidate.candidate_kind, 0) + 1
        )
    for kind, expected_count in required_counts.items():
        if observed_counts.get(kind, 0) != expected_count:
            reasons.add("missing_model_candidate")
    page_count = sum(
        candidate.candidate_kind == "texture_page"
        for candidate in model_candidates
    )
    if page_count != len(candidates.texture_page_ids):
        reasons.add("texture_page_reference_mismatch")
    status: ModelStatus = "unsupported" if reasons else "supported"
    input_payload = {
        "planner_version": planner_version,
        "format_id": format_id,
        "rig_geometry_cache_sha256": cache.cache_sha256,
        "candidate_universe_sha256": candidates.candidate_universe_sha256,
        "global_symbol_table_sha256": symbols.table_sha256,
    }
    values = {
        "schema_version": FORMAT_MODEL_PLAN_VERSION,
        "planner_version": planner_version,
        "format_id": format_id,
        "status": status,
        "reason_codes": tuple(sorted(reasons)),
        "component_count": mesh_count,
        "component_draw_ranks": ranks,
        "bone_count": len(cache.bone_graph.bones),
        "part_count": len(cache.parts),
        "texture_page_count": page_count,
        "candidate_count": len(format_candidates),
        "symbol_count": sum(
            symbol.namespace_key.format_id == format_id for symbol in symbols.symbols
        ),
        "planner_input_sha256": jcs_sha256(input_payload),
    }
    provisional_output = FormatModelPlan(
        **values,
        planner_output_sha256="",
        plan_sha256="",
    )
    values["planner_output_sha256"] = jcs_sha256(
        provisional_output.output_payload()
    )
    provisional = FormatModelPlan(**values, plan_sha256="")
    return FormatModelPlan(
        **values,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def _preset_maps(presets: PresetLibraryPlan):
    descriptor_by_id = {
        clip.preset_id: (clip.clip_id, clip.template_sha256, "motion")
        for clip in presets.clips
    }
    controls_by_id = {
        clip.preset_id: tuple(curve.control_id for curve in clip.control_curves)
        for clip in presets.clips
    }
    descriptor_by_id.update(
        {
            expression.preset_id: (
                expression.expression_id,
                expression.template_sha256,
                "expression",
            )
            for expression in presets.expressions
        }
    )
    controls_by_id.update(
        {
            expression.preset_id: tuple(
                value.control_id for value in expression.values
            )
            for expression in presets.expressions
        }
    )
    return descriptor_by_id, controls_by_id


def _make_decision(
    *,
    planner_version: str,
    format_id: str,
    preset_id: str,
    required: bool,
    quality_tier: str,
    preset_template_sha256: str,
    capability_sha256: str,
    model_plan_sha256: str,
    candidate_universe_sha256: str,
    symbol_table_sha256: str,
    status: DecisionStatus,
    reason: str | None,
    selected_implementations: tuple[str, ...] = (),
    selected_bindings: tuple[str, ...] = (),
    selected_candidates: tuple[PrimitiveCandidate, ...] = (),
    artifact_candidate: PrimitiveCandidate | None = None,
    artifact_symbol_id: str | None = None,
    artifact_export_name: str | None = None,
    conflicts_with: tuple[str, ...] = (),
    failed_keys: tuple[str, ...] = (),
) -> FormatPresetDecision:
    incompatible = {
        "talk": ("happy", "sad"),
        "happy": ("talk",),
        "sad": ("talk",),
    }.get(preset_id, ())
    input_payload = {
        "planner_version": planner_version,
        "format_id": format_id,
        "preset_id": preset_id,
        "required": required,
        "preset_template_sha256": preset_template_sha256,
        "capability_sha256": capability_sha256,
        "model_plan_sha256": model_plan_sha256,
        "candidate_universe_sha256": candidate_universe_sha256,
        "global_symbol_table_sha256": symbol_table_sha256,
    }
    values = {
        "schema_version": FORMAT_CAPABILITY_PREFLIGHT_VERSION,
        "planner_version": planner_version,
        "format_id": format_id,
        "preset_id": preset_id,
        "required": required,
        "status": status,
        "reason": reason,
        "quality_tier": quality_tier,
        "selected_implementation_ids": tuple(sorted(selected_implementations)),
        "selected_binding_ids": tuple(sorted(selected_bindings)),
        "selected_candidate_ids": tuple(
            sorted(candidate.candidate_id for candidate in selected_candidates)
        ),
        "selected_primitive_key_sha256": tuple(
            sorted(
                {
                    candidate.typed_primitive_key.key_sha256
                    for candidate in selected_candidates
                }
            )
        ),
        "artifact_candidate_id": (
            None if artifact_candidate is None else artifact_candidate.candidate_id
        ),
        "artifact_symbol_id": artifact_symbol_id,
        "artifact_export_name": artifact_export_name,
        "conflicts_with": tuple(sorted(conflicts_with)),
        "failed_primitive_keys": tuple(sorted(set(failed_keys))),
        "incompatible_with": incompatible,
        "planner_input_sha256": jcs_sha256(input_payload),
    }
    provisional_output = FormatPresetDecision(
        **values,
        planner_output_sha256="",
        decision_sha256="",
    )
    values["planner_output_sha256"] = jcs_sha256(
        provisional_output.output_payload()
    )
    provisional = FormatPresetDecision(**values, decision_sha256="")
    return FormatPresetDecision(
        **values,
        decision_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def _artifact_candidate(
    preset_id: str,
    descriptor_id: str,
    descriptor_kind: str,
    format_id: str,
    candidates: PrimitiveCandidateSet,
) -> PrimitiveCandidate | None:
    candidate_kind = (
        "spine_animation"
        if format_id == "spine_4_2"
        else "live2d_motion"
        if descriptor_kind == "motion"
        else "live2d_expression"
    )
    matches = tuple(
        candidate
        for candidate in candidates.candidates
        if candidate.candidate_kind == candidate_kind
        and candidate.typed_primitive_key.base_source_internal_id == descriptor_id
        and candidate.typed_primitive_key.preset_id == descriptor_id
    )
    if len(matches) > 1:
        raise _error(
            "format_plan_mismatch", f"artifact candidate is ambiguous: {preset_id}"
        )
    return matches[0] if matches else None


def _individual_decisions(
    format_id: str,
    profile: CapabilityProfile,
    model_plan: FormatModelPlan,
    presets: PresetLibraryPlan,
    capabilities: CapabilityPlan,
    bindings: ControlBindingPlan,
    candidates: PrimitiveCandidateSet,
    symbols: GlobalExportSymbolTable,
) -> tuple[FormatPresetDecision, ...]:
    planner_version = (
        SPINE_FORMAT_PLANNER_VERSION
        if format_id == "spine_4_2"
        else LIVE2D_FORMAT_PLANNER_VERSION
    )
    descriptor_by_id, controls_by_preset = _preset_maps(presets)
    capability_by_id = {
        capability.preset_id: capability for capability in capabilities.capabilities
    }
    binding_candidates = {
        candidate.binding_template.binding_id: candidate
        for candidate in candidates.candidates
        if candidate.format_id == format_id and candidate.binding_template is not None
    }
    symbol_by_key = {
        symbol.typed_primitive_key.key_sha256: symbol for symbol in symbols.symbols
    }
    bindings_by_group: dict[str, list[ControlBinding]] = {}
    for binding in bindings.bindings:
        bindings_by_group.setdefault(binding.binding_group_id, []).append(binding)
    results: list[FormatPresetDecision] = []
    for preset_id in sorted(descriptor_by_id):
        descriptor_id, template_sha256, descriptor_kind = descriptor_by_id[preset_id]
        capability = capability_by_id[preset_id]
        required = preset_id in profile.required_preset_ids
        common = {
            "planner_version": planner_version,
            "format_id": format_id,
            "preset_id": preset_id,
            "required": required,
            "quality_tier": capability.quality_tier,
            "preset_template_sha256": template_sha256,
            "capability_sha256": capability.capability_sha256,
            "model_plan_sha256": model_plan.plan_sha256,
            "candidate_universe_sha256": candidates.candidate_universe_sha256,
            "symbol_table_sha256": symbols.table_sha256,
        }
        if model_plan.status != "supported":
            results.append(
                _make_decision(
                    **common,
                    status="omitted",
                    reason=model_plan.reason_codes[0],
                )
            )
            continue
        if capability.status != "available":
            results.append(
                _make_decision(
                    **common,
                    status="omitted",
                    reason=(
                        capability.reason_codes[0]
                        if capability.reason_codes
                        else "missing_capability"
                    ),
                )
            )
            continue
        if format_id == "live2d_moc3_v4_00" and preset_id.startswith("wave."):
            results.append(
                _make_decision(
                    **common,
                    status="omitted",
                    reason="live2d_joint_bend_requires_glue",
                )
            )
            continue

        preset_control_ids = set(controls_by_preset[preset_id])
        relevant_groups = tuple(
            sorted(
                group_id
                for group_id, group_bindings in bindings_by_group.items()
                if any(
                    binding.control_id in preset_control_ids
                    for binding in group_bindings
                )
            )
        )
        selected_bindings: list[ControlBinding] = []
        selected_candidates: list[PrimitiveCandidate] = []
        selected_implementations: list[str] = []
        missing_reason: str | None = None
        for group_id in relevant_groups:
            group = bindings_by_group[group_id]
            implementations: dict[str, list[ControlBinding]] = {}
            for binding in group:
                implementations.setdefault(binding.implementation_id, []).append(binding)
            eligible = []
            for implementation_id, rows in implementations.items():
                if all(row.binding_id in binding_candidates for row in rows):
                    ranks = {row.implementation_rank for row in rows}
                    digests = {row.implementation_bundle_digest for row in rows}
                    if len(ranks) != 1 or len(digests) != 1:
                        raise _error(
                            "format_plan_mismatch",
                            f"implementation bundle is inconsistent: {implementation_id}",
                        )
                    eligible.append((next(iter(ranks)), implementation_id, rows))
            if not eligible:
                missing_reason = "unsupported_control_binding"
                break
            eligible.sort(key=lambda item: (item[0], item[1]))
            if len(eligible) > 1 and eligible[0][0] == eligible[1][0]:
                raise _error(
                    "format_plan_mismatch",
                    f"implementation ranks collide in {group_id}",
                )
            _rank, implementation_id, rows = eligible[0]
            selected_implementations.append(implementation_id)
            selected_bindings.extend(rows)
            selected_candidates.extend(
                binding_candidates[row.binding_id] for row in rows
            )
        if not relevant_groups or not preset_control_ids <= {
            binding.control_id for binding in selected_bindings
        }:
            missing_reason = missing_reason or "unsupported_control_binding"
        artifact = _artifact_candidate(
            preset_id,
            descriptor_id,
            descriptor_kind,
            format_id,
            candidates,
        )
        artifact_symbol = (
            None
            if artifact is None
            else symbol_by_key.get(artifact.typed_primitive_key.key_sha256)
        )
        if artifact is None or artifact_symbol is None:
            missing_reason = missing_reason or "missing_export_symbol"
        if missing_reason is not None:
            results.append(
                _make_decision(
                    **common,
                    status="omitted",
                    reason=missing_reason,
                )
            )
            continue
        results.append(
            _make_decision(
                **common,
                status="supported",
                reason=None,
                selected_implementations=tuple(selected_implementations),
                selected_bindings=tuple(
                    binding.binding_id for binding in selected_bindings
                ),
                selected_candidates=tuple(selected_candidates),
                artifact_candidate=artifact,
                artifact_symbol_id=artifact_symbol.symbol_id,
                artifact_export_name=artifact_symbol.export_name,
            )
        )
    return tuple(results)


def _non_rigid_conflicts(
    preset_ids: tuple[str, ...],
    decisions: dict[str, FormatPresetDecision],
    candidate_by_id: dict[str, PrimitiveCandidate],
) -> tuple[dict[str, set[str]], tuple[str, ...]]:
    target_controls: dict[str, dict[str, set[str]]] = {}
    failed_keys: set[str] = set()
    for preset_id in preset_ids:
        decision = decisions[preset_id]
        for candidate_id in decision.selected_candidate_ids:
            candidate = candidate_by_id[candidate_id]
            template = candidate.binding_template
            if template is None or template.property != "deform":
                continue
            target_controls.setdefault(template.target_id, {}).setdefault(
                template.control_id, set()
            ).add(preset_id)
    conflicts: dict[str, set[str]] = {}
    for target_id, controls in target_controls.items():
        if len(controls) <= 1:
            continue
        involved = set().union(*controls.values())
        for preset_id in involved:
            conflicts.setdefault(preset_id, set()).update(involved - {preset_id})
        for preset_id in involved:
            for candidate_id in decisions[preset_id].selected_candidate_ids:
                candidate = candidate_by_id[candidate_id]
                if (
                    candidate.binding_template is not None
                    and candidate.binding_template.property == "deform"
                    and candidate.binding_template.target_id == target_id
                ):
                    failed_keys.add(candidate.typed_primitive_key.key_sha256)
    return conflicts, tuple(sorted(failed_keys))


def _replace_omitted(
    decision: FormatPresetDecision,
    *,
    reason: str,
    conflicts_with: tuple[str, ...],
    failed_keys: tuple[str, ...],
) -> FormatPresetDecision:
    provisional_output = replace(
        decision,
        status="omitted",
        reason=reason,
        selected_implementation_ids=(),
        selected_binding_ids=(),
        selected_candidate_ids=(),
        selected_primitive_key_sha256=(),
        artifact_candidate_id=None,
        artifact_symbol_id=None,
        artifact_export_name=None,
        conflicts_with=tuple(sorted(conflicts_with)),
        failed_primitive_keys=tuple(sorted(failed_keys)),
        planner_output_sha256="",
        decision_sha256="",
    )
    provisional = replace(
        provisional_output,
        planner_output_sha256=jcs_sha256(provisional_output.output_payload()),
    )
    return replace(
        provisional,
        decision_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def _preset_set_plan(
    format_id: str,
    profile: CapabilityProfile,
    model_plan: FormatModelPlan,
    presets: PresetLibraryPlan,
    capabilities: CapabilityPlan,
    bindings: ControlBindingPlan,
    candidates: PrimitiveCandidateSet,
    symbols: GlobalExportSymbolTable,
) -> FormatPresetSetPlan:
    planner_version = (
        SPINE_FORMAT_PLANNER_VERSION
        if format_id == "spine_4_2"
        else LIVE2D_FORMAT_PLANNER_VERSION
    )
    individual = _individual_decisions(
        format_id,
        profile,
        model_plan,
        presets,
        capabilities,
        bindings,
        candidates,
        symbols,
    )
    decisions = {decision.preset_id: decision for decision in individual}
    all_preset_ids = set(decisions)
    expected_preset_ids = set(profile.required_preset_ids) | set(
        profile.optional_preset_priority
    )
    if all_preset_ids != expected_preset_ids:
        raise _error(
            "invalid_preset_registry",
            "profile required/optional inputs do not cover the preset registry",
        )
    missing_required = tuple(
        preset_id
        for preset_id in profile.required_preset_ids
        if decisions[preset_id].status != "supported"
    )
    if missing_required:
        reasons = ", ".join(
            f"{preset_id}:{decisions[preset_id].reason}"
            for preset_id in missing_required
        )
        raise _error(
            "missing_required_capability",
            f"{format_id} required presets are unsupported: {reasons}",
        )
    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in candidates.candidates
    }
    selected = list(profile.required_preset_ids)
    if format_id == "live2d_moc3_v4_00":
        conflicts, failed_keys = _non_rigid_conflicts(
            tuple(selected), decisions, candidate_by_id
        )
        if conflicts:
            details = ", ".join(
                f"{preset}:{','.join(sorted(values))}"
                for preset, values in sorted(conflicts.items())
            )
            raise _error(
                "missing_required_capability",
                f"live2d_parameter_conflict among required presets: {details}; "
                f"failed={','.join(failed_keys)}",
            )
    for preset_id in profile.optional_preset_priority:
        decision = decisions[preset_id]
        if decision.status != "supported":
            continue
        tentative = (*selected, preset_id)
        if format_id == "live2d_moc3_v4_00":
            conflicts, failed_keys = _non_rigid_conflicts(
                tentative, decisions, candidate_by_id
            )
            if preset_id in conflicts:
                decisions[preset_id] = _replace_omitted(
                    decision,
                    reason="live2d_parameter_conflict",
                    conflicts_with=tuple(conflicts[preset_id]),
                    failed_keys=failed_keys,
                )
                continue
        selected.append(preset_id)
    ordered_decisions = tuple(decisions[preset_id] for preset_id in sorted(decisions))
    selected_candidate_ids = tuple(
        sorted(
            {
                candidate_id
                for preset_id in selected
                for candidate_id in decisions[preset_id].selected_candidate_ids
            }
        )
    )
    selected_keys = tuple(
        sorted(
            {
                candidate_by_id[candidate_id].typed_primitive_key.key_sha256
                for candidate_id in selected_candidate_ids
            }
        )
    )
    union_payload = {
        "candidate_ids": list(selected_candidate_ids),
        "typed_primitive_key_sha256": list(selected_keys),
    }
    input_payload = {
        "planner_version": planner_version,
        "selection_version": presets.optional_selection_version,
        "format_id": format_id,
        "profile_sha256": profile.profile_sha256,
        "model_plan_sha256": model_plan.plan_sha256,
        "required_preset_ids": list(profile.required_preset_ids),
        "optional_priority": list(profile.optional_preset_priority),
        "individual_decision_sha256": [
            decision.decision_sha256 for decision in individual
        ],
    }
    values = {
        "schema_version": FORMAT_PRESET_SET_PLAN_VERSION,
        "planner_version": planner_version,
        "selection_version": presets.optional_selection_version,
        "format_id": format_id,
        "profile_id": profile.profile_id,
        "model_plan_sha256": model_plan.plan_sha256,
        "required_preset_ids": profile.required_preset_ids,
        "optional_priority": profile.optional_preset_priority,
        "decisions": ordered_decisions,
        "selected_preset_ids": tuple(sorted(selected)),
        "selected_candidate_ids": selected_candidate_ids,
        "selected_primitive_key_sha256": selected_keys,
        "primitive_union_sha256": jcs_sha256(union_payload),
        "planner_input_sha256": jcs_sha256(input_payload),
    }
    provisional_output = FormatPresetSetPlan(
        **values,
        planner_output_sha256="",
        plan_sha256="",
    )
    values["planner_output_sha256"] = jcs_sha256(
        provisional_output.output_payload()
    )
    provisional = FormatPresetSetPlan(**values, plan_sha256="")
    return FormatPresetSetPlan(
        **values,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def _build_format_plan_set(
    cache: RigGeometryCache,
    controls: ControlRegistryPlan,
    presets: PresetLibraryPlan,
    capabilities: CapabilityPlan,
    bindings: ControlBindingPlan,
    candidates: PrimitiveCandidateSet,
    symbols: GlobalExportSymbolTable,
    *,
    profile_id: str,
) -> FormatPlanSet:
    profile = load_capability_profile(profile_id)
    if presets.optional_priority != profile.optional_preset_priority:
        raise _error(
            "invalid_preset_registry", "profile optional priority differs from library"
        )
    model_plans = tuple(
        _model_plan(format_id, cache, candidates, symbols)
        for format_id in profile.required_formats
    )
    unsupported_models = tuple(
        plan for plan in model_plans if plan.status != "supported"
    )
    if unsupported_models:
        details = ", ".join(
            f"{plan.format_id}:{','.join(plan.reason_codes)}"
            for plan in unsupported_models
        )
        raise _error("format_model_unsupported", details)
    preset_sets = tuple(
        _preset_set_plan(
            model.format_id,
            profile,
            model,
            presets,
            capabilities,
            bindings,
            candidates,
            symbols,
        )
        for model in model_plans
    )
    values = {
        "schema_version": FORMAT_PLAN_SET_VERSION,
        "profile": profile,
        "rig_geometry_cache_sha256": cache.cache_sha256,
        "capability_plan_sha256": capabilities.plan_sha256,
        "control_binding_plan_sha256": bindings.plan_sha256,
        "primitive_candidate_set_sha256": candidates.plan_sha256,
        "global_symbol_table_sha256": symbols.table_sha256,
        "model_plans": model_plans,
        "preset_set_plans": preset_sets,
    }
    provisional = FormatPlanSet(**values, plan_sha256="")
    return FormatPlanSet(
        **values,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def validate_format_plan_set(
    plan: FormatPlanSet,
    cache: RigGeometryCache,
    controls: ControlRegistryPlan,
    presets: PresetLibraryPlan,
    capabilities: CapabilityPlan,
    bindings: ControlBindingPlan,
    candidates: PrimitiveCandidateSet,
    symbols: GlobalExportSymbolTable,
) -> FormatPlanSet:
    """Re-run all pure format decisions and require exact equality."""

    _validate_inputs(
        cache, controls, presets, capabilities, bindings, candidates, symbols
    )
    if not isinstance(plan, FormatPlanSet) or plan.schema_version != FORMAT_PLAN_SET_VERSION:
        raise _error("format_plan_mismatch", "format plan-set version is unsupported")
    try:
        expected = _build_format_plan_set(
            cache,
            controls,
            presets,
            capabilities,
            bindings,
            candidates,
            symbols,
            profile_id=plan.profile.profile_id,
        )
    except FormatPlanError as exc:
        if exc.code == "format_plan_mismatch":
            raise
        raise _error("format_plan_mismatch", str(exc)) from exc
    if plan != expected:
        raise _error("format_plan_mismatch", "persisted format decisions drifted")
    if plan.plan_sha256 != jcs_sha256(plan.semantic_payload()):
        raise _error("format_plan_mismatch", "format plan-set digest mismatch")
    return plan


def build_format_plan_set(
    cache: RigGeometryCache,
    controls: ControlRegistryPlan,
    presets: PresetLibraryPlan,
    capabilities: CapabilityPlan,
    bindings: ControlBindingPlan,
    candidates: PrimitiveCandidateSet,
    symbols: GlobalExportSymbolTable,
    *,
    profile_id: str,
) -> FormatPlanSet:
    """Build profile decisions without writing target-format artifacts."""

    _validate_inputs(
        cache, controls, presets, capabilities, bindings, candidates, symbols
    )
    plan = _build_format_plan_set(
        cache,
        controls,
        presets,
        capabilities,
        bindings,
        candidates,
        symbols,
        profile_id=profile_id,
    )
    return validate_format_plan_set(
        plan,
        cache,
        controls,
        presets,
        capabilities,
        bindings,
        candidates,
        symbols,
    )


__all__ = [
    "CAPABILITY_PROFILE_REGISTRY_VERSION",
    "CAPABILITY_PROFILE_SCHEMA_VERSION",
    "FORMAT_CAPABILITY_PREFLIGHT_VERSION",
    "FORMAT_MODEL_PLAN_VERSION",
    "FORMAT_PLAN_SET_VERSION",
    "FORMAT_PRESET_SET_PLAN_VERSION",
    "LIVE2D_FORMAT_PLANNER_VERSION",
    "SPINE_FORMAT_PLANNER_VERSION",
    "CapabilityProfile",
    "FormatModelPlan",
    "FormatPlanError",
    "FormatPlanSet",
    "FormatPresetDecision",
    "FormatPresetSetPlan",
    "build_format_plan_set",
    "load_capability_profile",
    "validate_format_plan_set",
]
