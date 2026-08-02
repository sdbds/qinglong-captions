from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from module.auto_rig.jcs import jcs_sha256
from module.auto_rig.primitive_candidates import (
    PRIMITIVE_CANDIDATE_ENUMERATOR_VERSION,
    PrimitiveCandidateSetError,
    enumerate_primitive_candidates,
    validate_primitive_candidate_set,
)
from tests.test_auto_rig_control_bindings import _build


def _enumerate(tmp_path: Path):
    cache, _anatomy, controls, presets, _capabilities, bindings = _build(tmp_path)
    candidates = enumerate_primitive_candidates(
        cache,
        controls,
        presets,
        bindings,
        texture_page_ids=("texture-page/page_0", "texture-page/page_1"),
    )
    return cache, controls, presets, bindings, candidates


def test_candidate_universe_is_profile_independent_and_reference_closed(
    tmp_path: Path,
) -> None:
    cache, controls, presets, bindings, candidates = _enumerate(tmp_path)

    assert candidates.enumerator_version == PRIMITIVE_CANDIDATE_ENUMERATOR_VERSION
    assert validate_primitive_candidate_set(
        candidates,
        cache,
        controls,
        presets,
        bindings,
        texture_page_ids=("texture-page/page_0", "texture-page/page_1"),
    ) is candidates
    assert len({candidate.candidate_id for candidate in candidates.candidates}) == len(
        candidates.candidates
    )
    assert tuple(candidate.candidate_id for candidate in candidates.candidates) == tuple(
        sorted(candidate.candidate_id for candidate in candidates.candidates)
    )

    kinds = {candidate.candidate_kind for candidate in candidates.candidates}
    assert {
        "spine_skin",
        "spine_bone",
        "spine_slot",
        "spine_attachment_key",
        "spine_attachment_object",
        "spine_atlas_region",
        "spine_animation",
        "spine_binding",
        "live2d_part",
        "live2d_artmesh",
        "live2d_parameter",
        "live2d_rotation_deformer",
        "live2d_motion",
        "live2d_expression",
        "live2d_binding",
        "texture_page",
    } <= kinds

    binding_ids = {binding.binding_id for binding in bindings.bindings}
    spine_bindings = {
        candidate.binding_template.binding_id
        for candidate in candidates.candidates
        if candidate.format_id == "spine_4_2" and candidate.binding_template is not None
    }
    assert spine_bindings == binding_ids

    live2d_control_ids = {
        control.control_id for control in controls.controls if control.live2d is not None
    }
    live2d_bindings = {
        candidate.binding_template.binding_id
        for candidate in candidates.candidates
        if candidate.format_id == "live2d_moc3_v4_00"
        and candidate.binding_template is not None
    }
    assert live2d_bindings == {
        binding.binding_id
        for binding in bindings.bindings
        if binding.control_id in live2d_control_ids
    }

    assert not any(
        candidate.format_id == "live2d_moc3_v4_00"
        and candidate.binding_template is not None
        and candidate.binding_template.control_id.startswith("control/wave_")
        for candidate in candidates.candidates
    )


def test_sampled_mesh_deform_reuses_the_static_artmesh_primitive(
    tmp_path: Path,
) -> None:
    _cache, _controls, _presets, _bindings, candidates = _enumerate(tmp_path)
    model_artmeshes = {
        (
            candidate.typed_primitive_key.base_source_internal_id,
            candidate.typed_primitive_key.component_id,
        ): candidate.primitive_target_id
        for candidate in candidates.candidates
        if candidate.candidate_kind == "live2d_artmesh"
        and candidate.binding_template is None
    }
    sampled_deforms = [
        candidate
        for candidate in candidates.candidates
        if candidate.candidate_kind == "live2d_binding"
        and candidate.binding_template is not None
        and candidate.binding_template.property == "deform"
    ]

    assert sampled_deforms
    for candidate in sampled_deforms:
        key = candidate.typed_primitive_key
        assert key.kind == "live2d_artmesh"
        assert key.control_id is None
        assert key.parameter_id is None
        assert key.derivation_tokens == ()
        assert candidate.primitive_target_id == model_artmeshes[
            (key.base_source_internal_id, key.component_id)
        ]
    assert not any(
        candidate.typed_primitive_key.kind == "live2d_warp_deformer"
        for candidate in candidates.candidates
    )


def test_rotation_instance_key_uses_typed_bone_parameter_identity(
    tmp_path: Path,
) -> None:
    _cache, _controls, _presets, _bindings, candidates = _enumerate(tmp_path)

    matches = {
        candidate.typed_primitive_key
        for candidate in candidates.candidates
        if candidate.typed_primitive_key.kind == "live2d_rotation_deformer"
        and candidate.typed_primitive_key.source_internal_ids == ("bone/torso",)
        and candidate.typed_primitive_key.parameter_id == "parameter/auto_idle"
    }
    assert len(matches) == 1
    key = matches.pop()
    assert key.base_source_internal_id == "bone/torso"
    assert key.derivation_tokens == ("rot", "auto_idle")


def test_candidate_validator_rejects_rehashed_unknown_required_fact(tmp_path: Path) -> None:
    cache, controls, presets, bindings, candidates = _enumerate(tmp_path)
    rows = list(candidates.candidates)
    changed = replace(
        rows[0],
        required_rig_fact_ids=(*rows[0].required_rig_fact_ids, "mesh/missing"),
        candidate_sha256="",
    )
    changed = replace(changed, candidate_sha256=jcs_sha256(changed.semantic_payload()))
    rows[0] = changed
    provisional = replace(candidates, candidates=tuple(rows), plan_sha256="")
    tampered = replace(
        provisional,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )

    with pytest.raises(PrimitiveCandidateSetError) as exc_info:
        validate_primitive_candidate_set(
            tampered,
            cache,
            controls,
            presets,
            bindings,
            texture_page_ids=("texture-page/page_0", "texture-page/page_1"),
        )

    assert exc_info.value.code == "invalid_primitive_candidate_set"
