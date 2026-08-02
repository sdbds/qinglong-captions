from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from module.auto_rig.capabilities import derive_capabilities
from module.auto_rig.control_bindings import build_control_binding_plan
from module.auto_rig.export_symbols import build_global_export_symbol_table
from module.auto_rig.format_plans import (
    CAPABILITY_PROFILE_REGISTRY_VERSION,
    FORMAT_PRESET_SET_PLAN_VERSION,
    FormatPlanError,
    _live2d_draw_order_capacity,
    build_format_plan_set,
    load_capability_profile,
    validate_format_plan_set,
)
from module.auto_rig.jcs import jcs_sha256
from module.auto_rig.primitive_candidates import enumerate_primitive_candidates
from tests.test_auto_rig_capabilities import build_capability_fixture


def _fixture(tmp_path: Path, **kwargs):
    cache, anatomy, controls, presets = build_capability_fixture(tmp_path, **kwargs)
    capabilities = derive_capabilities(
        cache,
        anatomy,
        presets,
        native_variant_set=None,
    )
    bindings = build_control_binding_plan(
        cache,
        anatomy,
        controls,
        presets,
        capabilities,
        native_variant_set=None,
    )
    candidates = enumerate_primitive_candidates(
        cache,
        controls,
        presets,
        bindings,
        texture_page_ids=("texture-page/page_0",),
    )
    symbols = build_global_export_symbol_table(candidates, controls)
    return cache, controls, presets, capabilities, bindings, candidates, symbols


def _build(tmp_path: Path, profile_id: str = "dual_runtime_core_v1", **kwargs):
    values = _fixture(tmp_path, **kwargs)
    plan = build_format_plan_set(*values, profile_id=profile_id)
    return (*values, plan)


def test_profile_registry_freezes_required_formats_presets_and_terminal_scope() -> None:
    core = load_capability_profile("dual_runtime_core_v1")
    avatar = load_capability_profile("dual_runtime_avatar_v1")
    dev = load_capability_profile("spine_4_2_dev")

    assert core.registry_version == CAPABILITY_PROFILE_REGISTRY_VERSION
    assert core.required_formats == ("live2d_moc3_v4_00", "spine_4_2")
    assert core.required_preset_ids == (
        "breath",
        "head_nod",
        "head_shake",
        "idle",
    )
    assert avatar.required_preset_ids == (
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
    assert core.optional_preset_parity == avatar.optional_preset_parity == "per_format"
    assert core.terminal_delivery is True
    assert avatar.terminal_delivery is True
    assert dev.required_formats == ("spine_4_2",)
    assert dev.terminal_delivery is False
    assert all(profile.strict_capabilities for profile in (core, avatar, dev))

    with pytest.raises(FormatPlanError, match="unknown capability profile"):
        load_capability_profile("anything_goes")


def test_core_format_plan_selects_atomic_bundles_and_allows_optional_format_split(
    tmp_path: Path,
) -> None:
    (
        cache,
        controls,
        presets,
        capabilities,
        bindings,
        candidates,
        symbols,
        plan,
    ) = _build(tmp_path)

    assert validate_format_plan_set(
        plan,
        cache,
        controls,
        presets,
        capabilities,
        bindings,
        candidates,
        symbols,
    ) is plan
    by_format = {item.format_id: item for item in plan.preset_set_plans}
    spine = {item.preset_id: item for item in by_format["spine_4_2"].decisions}
    live2d = {
        item.preset_id: item
        for item in by_format["live2d_moc3_v4_00"].decisions
    }

    for preset_id in plan.profile.required_preset_ids:
        assert spine[preset_id].status == "supported"
        assert live2d[preset_id].status == "supported"
        assert spine[preset_id].required is True
        assert live2d[preset_id].required is True

    assert spine["wave.xmin"].status == "supported"
    assert spine["wave.xmax"].status == "supported"
    assert live2d["wave.xmin"].status == "omitted"
    assert live2d["wave.xmin"].reason == "live2d_joint_bend_requires_glue"
    assert live2d["talk"].status == "supported"
    assert live2d["surprised"].status == "supported"
    assert live2d["happy"].status == "omitted"
    assert live2d["sad"].status == "omitted"
    assert live2d["happy"].reason == "live2d_parameter_conflict"
    assert "talk" in live2d["happy"].conflicts_with

    blink = live2d["blink"]
    assert blink.status == "supported"
    assert blink.selected_implementation_ids == (
        "binding-impl/blink.procedural-v1",
    )
    blink_bindings = {
        binding.binding_id
        for binding in bindings.bindings
        if binding.implementation_id == "binding-impl/blink.procedural-v1"
    }
    assert set(blink.selected_binding_ids) == blink_bindings

    candidate_ids = {candidate.candidate_id for candidate in candidates.candidates}
    key_ids = {
        candidate.typed_primitive_key.key_sha256
        for candidate in candidates.candidates
    }
    for set_plan in plan.preset_set_plans:
        assert set(set_plan.selected_candidate_ids) <= candidate_ids
        assert set(set_plan.selected_primitive_key_sha256) <= key_ids


def test_live2d_model_plan_preserves_component_rank_and_capacity_boundary(
    tmp_path: Path,
) -> None:
    cache, *_rest, plan = _build(tmp_path)
    live2d = next(
        model for model in plan.model_plans if model.format_id == "live2d_moc3_v4_00"
    )

    assert live2d.status == "supported"
    assert live2d.component_count == len(cache.skinning_plan.weighted_meshes)
    assert live2d.component_draw_ranks == tuple(
        record.component_draw_rank for record in cache.component_draw_order.records
    )
    assert _live2d_draw_order_capacity(tuple(range(1001))) is None
    assert _live2d_draw_order_capacity(tuple(range(1002))) == (
        "draw_order_capacity_exceeded"
    )
    assert _live2d_draw_order_capacity((0, 2)) == "invalid_component_draw_rank"


def test_required_capability_or_required_union_conflict_fails_before_writer(
    tmp_path: Path,
) -> None:
    with pytest.raises(FormatPlanError) as head_only:
        _build(
            tmp_path / "head",
            torso=False,
            limb_mode="missing",
            eyes=False,
            brows=False,
            mouth=False,
        )
    assert head_only.value.code == "missing_required_capability"

    with pytest.raises(FormatPlanError) as avatar:
        _build(tmp_path / "avatar", profile_id="dual_runtime_avatar_v1")
    assert avatar.value.code == "missing_required_capability"
    assert "live2d_parameter_conflict" in str(avatar.value)


def test_spine_dev_plan_is_single_format_and_non_terminal(tmp_path: Path) -> None:
    *_inputs, plan = _build(tmp_path, profile_id="spine_4_2_dev")

    assert plan.profile.terminal_delivery is False
    assert tuple(item.format_id for item in plan.model_plans) == ("spine_4_2",)
    assert tuple(item.format_id for item in plan.preset_set_plans) == ("spine_4_2",)
    assert plan.preset_set_plans[0].schema_version == FORMAT_PRESET_SET_PLAN_VERSION


def test_format_plan_validator_rejects_rehashed_unknown_candidate(tmp_path: Path) -> None:
    *inputs, plan = _build(tmp_path)
    set_plans = list(plan.preset_set_plans)
    selected = set_plans[0]
    changed = replace(
        selected,
        selected_candidate_ids=(*selected.selected_candidate_ids, "candidate/c_" + "f" * 64),
        plan_sha256="",
    )
    changed = replace(changed, plan_sha256=jcs_sha256(changed.semantic_payload()))
    set_plans[0] = changed
    provisional = replace(plan, preset_set_plans=tuple(set_plans), plan_sha256="")
    tampered = replace(
        provisional,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )

    with pytest.raises(FormatPlanError) as exc_info:
        validate_format_plan_set(tampered, *inputs)

    assert exc_info.value.code == "format_plan_mismatch"
