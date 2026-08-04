from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from module.auto_rig.capabilities import derive_capabilities
from module.auto_rig.control_bindings import (
    CONTROL_BINDING_PLAN_VERSION,
    RIG_TRANSFORM_SEMANTICS_VERSION,
    TARGET_TRANSFER_VERSION,
    ControlBindingPlanError,
    build_control_binding_plan,
    validate_control_binding_plan,
)
from module.auto_rig.jcs import jcs_sha256
from tests.test_auto_rig_capabilities import build_capability_fixture


def _build(tmp_path: Path, **fixture_kwargs):
    cache, anatomy, controls, presets = build_capability_fixture(
        tmp_path,
        **fixture_kwargs,
    )
    capabilities = derive_capabilities(
        cache,
        anatomy,
        presets,
        native_variant_set=None,
    )
    plan = build_control_binding_plan(
        cache,
        anatomy,
        controls,
        presets,
        capabilities,
        native_variant_set=None,
    )
    return cache, anatomy, controls, presets, capabilities, plan


def _bindings(plan, *, control_id=None, target_id=None, property_name=None, kind=None):
    return tuple(
        binding
        for binding in plan.bindings
        if (control_id is None or binding.control_id == control_id)
        and (target_id is None or binding.target_id == target_id)
        and (property_name is None or binding.property == property_name)
        and (kind is None or binding.implementation_kind == kind)
    )


def _metric(anatomy, metric_id):
    return next(metric for metric in anatomy.metrics if metric.metric_id == metric_id)


def test_core_bindings_freeze_canvas_clockwise_geometry_normalized_transfers(
    tmp_path: Path,
) -> None:
    cache, anatomy, controls, presets, capabilities, plan = _build(tmp_path)

    assert plan.schema_version == CONTROL_BINDING_PLAN_VERSION
    assert plan.transform_semantics_version == RIG_TRANSFORM_SEMANTICS_VERSION
    assert (
        validate_control_binding_plan(
            plan,
            cache,
            anatomy,
            controls,
            presets,
            capabilities,
            native_variant_set=None,
        )
        is plan
    )

    idle = _bindings(
        plan,
        control_id="control/idle",
        target_id="bone/torso",
        property_name="rotation",
    )
    assert len(idle) == 1
    assert idle[0].transfer.schema_version == TARGET_TRANSFER_VERSION
    assert idle[0].transfer.kind == "affine_scalar"
    assert idle[0].transfer.output_at_default == (0.0,)
    assert idle[0].transfer.gain == (1.0,)

    head_metric = _metric(anatomy, "mask/head_core")
    nod_y = _bindings(
        plan,
        control_id="control/head_nod",
        target_id="bone/head",
        property_name="translation_y",
    )[0]
    nod_rotation = _bindings(
        plan,
        control_id="control/head_nod",
        target_id="bone/head",
        property_name="rotation",
    )[0]
    shake_x = _bindings(
        plan,
        control_id="control/head_shake",
        target_id="bone/head",
        property_name="translation_x",
    )[0]
    assert nod_y.transfer.gain == (0.06 * head_metric.height / 30.0,)
    assert nod_rotation.transfer.gain == (0.2,)
    assert shake_x.transfer.gain == (0.06 * head_metric.width / 30.0,)

    torso_metric = _metric(anatomy, "mask/torso_core")
    sway_x = _bindings(
        plan,
        control_id="control/body_sway",
        target_id="bone/torso",
        property_name="translation_x",
    )[0]
    assert sway_x.transfer.gain == (0.02 * torso_metric.width / 10.0,)


def test_breath_and_facial_deforms_store_default_rest_and_topology_bound_samples(
    tmp_path: Path,
) -> None:
    cache, _anatomy, _controls, _presets, _capabilities, plan = _build(tmp_path)
    weighted_by_id = {mesh.mesh_id: mesh for mesh in cache.skinning_plan.weighted_meshes}

    breath_bindings = _bindings(
        plan,
        control_id="control/breath",
        property_name="deform",
    )
    assert breath_bindings
    torso_metric = _metric(_anatomy, "mask/torso_core")
    maximum_x_delta = 0.0
    maximum_y_delta = 0.0
    for binding in breath_bindings:
        mesh = weighted_by_id[binding.target_id]
        rest = tuple(value for vertex in mesh.vertices for value in vertex.position)
        transfer = binding.transfer
        assert transfer.kind == "sampled_deform"
        assert transfer.topology_sha256 == mesh.weighted_mesh_sha256
        assert next(sample for sample in transfer.samples if sample.input_value == 0.0).output_values == rest
        expanded = next(sample for sample in transfer.samples if sample.input_value == 1.0).output_values
        maximum_x_delta = max(
            maximum_x_delta,
            max(abs(after - before) for before, after in zip(rest[0::2], expanded[0::2], strict=True)),
        )
        maximum_y_delta = max(
            maximum_y_delta,
            max(abs(after - before) for before, after in zip(rest[1::2], expanded[1::2], strict=True)),
        )
    assert maximum_x_delta >= 0.01 * torso_metric.width
    assert maximum_y_delta >= 0.01 * torso_metric.height

    eye_deforms = _bindings(
        plan,
        control_id="control/eye_open.xmin",
        property_name="deform",
        kind="procedural",
    )
    assert len(eye_deforms) >= 3
    for binding in eye_deforms:
        mesh = weighted_by_id[binding.target_id]
        rest = tuple(value for vertex in mesh.vertices for value in vertex.position)
        assert next(sample for sample in binding.transfer.samples if sample.input_value == 1.0).output_values == rest
    iris_opacity = _bindings(
        plan,
        control_id="control/eye_open.xmin",
        property_name="opacity",
        kind="procedural",
    )
    assert iris_opacity
    assert next(sample for sample in iris_opacity[0].transfer.samples if sample.input_value == 0.0).output_values == (0.0,)

    talk = _bindings(
        plan,
        control_id="control/mouth_open",
        property_name="deform",
        kind="procedural",
    )
    mouth_form = _bindings(
        plan,
        control_id="control/mouth_form",
        property_name="deform",
        kind="procedural",
    )
    assert talk and mouth_form
    for binding in talk:
        rest = next(sample for sample in binding.transfer.samples if sample.input_value == 0.0).output_values
        opened = next(sample for sample in binding.transfer.samples if sample.input_value == 1.0).output_values
        rest_span = max(rest[1::2]) - min(rest[1::2])
        opened_span = max(opened[1::2]) - min(opened[1::2])
        assert opened_span >= rest_span * 2.2 - 1e-6
    assert all(binding.visibility_branch_id is None for binding in plan.bindings)


def test_head_shake_uses_depth_to_move_front_and_back_hair_differently(
    tmp_path: Path,
) -> None:
    cache, _anatomy, _controls, _presets, _capabilities, plan = _build(
        tmp_path,
        hair=True,
    )
    mesh_by_id = {mesh.mesh_id: mesh for mesh in cache.skinning_plan.weighted_meshes}
    part_by_id = {part.part_id: part for part in cache.parts}
    offsets_by_tag: dict[str, set[float]] = {}

    for binding in _bindings(
        plan,
        control_id="control/head_shake",
        property_name="deform",
        kind="canonical",
    ):
        mesh = mesh_by_id[binding.target_id]
        tag = part_by_id[mesh.part_id].base_tag
        rest = next(sample for sample in binding.transfer.samples if sample.input_value == 0.0).output_values
        maximum = max(
            binding.transfer.samples,
            key=lambda sample: sample.input_value,
        ).output_values
        offsets_by_tag.setdefault(tag, set()).add(maximum[0] - rest[0])
        assert maximum[1::2] == rest[1::2]

    assert set(offsets_by_tag) == {"back hair", "front hair"}
    assert min(offsets_by_tag["front hair"]) > 0.0
    assert max(offsets_by_tag["back hair"]) < 0.0
    assert offsets_by_tag["front hair"] != offsets_by_tag["back hair"]


def test_wave_bindings_use_visual_side_sign_and_complete_three_bone_chain(
    tmp_path: Path,
) -> None:
    _cache, _anatomy, _controls, _presets, _capabilities, plan = _build(tmp_path)

    expected = {
        ("control/wave_lift.xmin", "bone/upper_arm.xmin"): 35.0,
        ("control/wave_lift.xmin", "bone/forearm.xmin"): 20.0,
        ("control/wave_osc.xmin", "bone/forearm.xmin"): 15.0,
        ("control/wave_osc.xmin", "bone/hand.xmin"): 5.0,
        ("control/wave_lift.xmax", "bone/upper_arm.xmax"): -35.0,
        ("control/wave_lift.xmax", "bone/forearm.xmax"): -20.0,
        ("control/wave_osc.xmax", "bone/forearm.xmax"): -15.0,
        ("control/wave_osc.xmax", "bone/hand.xmax"): -5.0,
    }
    observed = {
        (binding.control_id, binding.target_id): binding.transfer.gain[0]
        for binding in plan.bindings
        if binding.control_id.startswith("control/wave_")
    }
    assert observed == expected


def test_missing_wrist_removes_the_entire_matching_wave_bundle(tmp_path: Path) -> None:
    _cache, _anatomy, _controls, _presets, _capabilities, plan = _build(
        tmp_path,
        omitted_joints=("joint/wrist.xmin",),
    )

    assert not _bindings(plan, control_id="control/wave_lift.xmin")
    assert not _bindings(plan, control_id="control/wave_osc.xmin")
    assert _bindings(plan, control_id="control/wave_lift.xmax")


def test_coarse_limb_sways_deform_merged_parts_without_claiming_joint_chains(
    tmp_path: Path,
) -> None:
    omitted = (
        "joint/shoulder.xmin",
        "joint/elbow.xmin",
        "joint/wrist.xmin",
        "joint/hand_tip.xmin",
        "joint/shoulder.xmax",
        "joint/elbow.xmax",
        "joint/wrist.xmax",
        "joint/hand_tip.xmax",
        "joint/hip.xmin",
        "joint/knee.xmin",
        "joint/ankle.xmin",
        "joint/toe.xmin",
        "joint/hip.xmax",
        "joint/knee.xmax",
        "joint/ankle.xmax",
        "joint/toe.xmax",
    )
    cache, _anatomy, _controls, _presets, _capabilities, plan = _build(
        tmp_path,
        limb_mode="merged",
        omitted_joints=omitted,
    )
    mesh_by_id = {mesh.mesh_id: mesh for mesh in cache.skinning_plan.weighted_meshes}
    part_by_id = {part.part_id: part for part in cache.parts}

    for control_id, allowed_tags in (
        ("control/arm_sway", {"handwear"}),
        ("control/leg_sway", {"legwear", "footwear"}),
    ):
        bindings = _bindings(
            plan,
            control_id=control_id,
            property_name="deform",
            kind="procedural",
        )
        assert bindings
        for binding in bindings:
            mesh = mesh_by_id[binding.target_id]
            assert part_by_id[mesh.part_id].base_tag in allowed_tags
            assert tuple(sample.input_value for sample in binding.transfer.samples) == (
                -1.0,
                0.0,
                1.0,
            )
            rest = tuple(value for vertex in mesh.vertices for value in vertex.position)
            samples = {sample.input_value: sample.output_values for sample in binding.transfer.samples}
            assert samples[0.0] == rest
            assert samples[-1.0] != rest
            assert samples[1.0] != rest

    assert not _bindings(plan, control_id="control/wave_lift.xmin")
    assert not _bindings(plan, control_id="control/wave_lift.xmax")


def test_binding_ids_bundles_and_ranks_are_typed_canonical_and_reference_closed(
    tmp_path: Path,
) -> None:
    _cache, _anatomy, _controls, _presets, _capabilities, plan = _build(tmp_path)

    binding_ids = tuple(binding.binding_id for binding in plan.bindings)
    assert len(binding_ids) == len(set(binding_ids))
    assert all(binding_id.startswith("binding/b_") for binding_id in binding_ids)
    assert all("clip/" not in binding_id and "expression/" not in binding_id for binding_id in binding_ids)
    implementation_rows = {}
    for binding in plan.bindings:
        implementation_rows.setdefault(binding.implementation_id, []).append(binding)
    assert all(
        len({binding.implementation_bundle_digest for binding in bindings}) == 1 for bindings in implementation_rows.values()
    )
    ranks_by_group = {}
    for binding in plan.bindings:
        ranks_by_group.setdefault(binding.binding_group_id, {})[binding.implementation_id] = binding.implementation_rank
    assert all(
        len(set(implementation_ranks.values())) == len(implementation_ranks) for implementation_ranks in ranks_by_group.values()
    )


def test_blink_sides_form_one_atomic_procedural_implementation_bundle(
    tmp_path: Path,
) -> None:
    _cache, _anatomy, _controls, _presets, _capabilities, plan = _build(tmp_path)

    blink_bindings = tuple(binding for binding in plan.bindings if binding.control_id.startswith("control/eye_open."))
    assert {binding.binding_group_id for binding in blink_bindings} == {"binding-group/blink"}
    assert {binding.implementation_id for binding in blink_bindings} == {"binding-impl/blink.procedural-v1"}
    assert len({binding.implementation_bundle_digest for binding in blink_bindings}) == 1


@pytest.mark.parametrize(
    "mutation",
    ("delete_bundle_member", "wrong_target", "wrong_default", "illegal_visibility"),
)
def test_binding_validator_rejects_rehashed_reference_and_bundle_mutations(
    tmp_path: Path,
    mutation: str,
) -> None:
    cache, anatomy, controls, presets, capabilities, plan = _build(tmp_path)
    bindings = list(plan.bindings)
    if mutation == "delete_bundle_member":
        implementation_id = next(
            binding.implementation_id
            for binding in bindings
            if sum(item.implementation_id == binding.implementation_id for item in bindings) > 1
        )
        delete_index = next(index for index, binding in enumerate(bindings) if binding.implementation_id == implementation_id)
        del bindings[delete_index]
    elif mutation == "wrong_target":
        bindings[0] = replace(bindings[0], target_id="bone/missing")
    elif mutation == "wrong_default":
        transfer = replace(
            bindings[0].transfer,
            output_at_default=(99.0,),
            transfer_sha256="",
        )
        transfer = replace(
            transfer,
            transfer_sha256=jcs_sha256(transfer.semantic_payload()),
        )
        bindings[0] = replace(bindings[0], transfer=transfer)
    else:
        bindings[0] = replace(
            bindings[0],
            visibility_branch_id="visibility-branch/not-native",
        )
    provisional = replace(plan, bindings=tuple(bindings), plan_sha256="")
    tampered = replace(
        provisional,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )

    with pytest.raises(ControlBindingPlanError) as exc_info:
        validate_control_binding_plan(
            tampered,
            cache,
            anatomy,
            controls,
            presets,
            capabilities,
            native_variant_set=None,
        )

    assert exc_info.value.code == "invalid_control_binding_plan"
