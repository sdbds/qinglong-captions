from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from module.auto_rig.anatomy import build_anatomy_mask_geometry
from module.auto_rig.bone_graph import build_bone_graph
from module.auto_rig.capabilities import (
    CAPABILITY_PLAN_VERSION,
    CapabilityPlanError,
    derive_capabilities,
    validate_capability_plan,
)
from module.auto_rig.component_geometry import (
    load_component_geometry,
    load_mesh_component_sources,
)
from module.auto_rig.component_plan import build_mask_component_plan
from module.auto_rig.control_registry import build_control_registry_plan
from module.auto_rig.jcs import jcs_sha256
from module.auto_rig.joint_pipeline import build_stage_a_joint_plan
from module.auto_rig.mesh_builder import build_mesh_plan
from module.auto_rig.overrides import (
    load_rig_override_source,
    validate_rig_override_source,
)
from module.auto_rig.preset_library import build_preset_library_plan
from module.auto_rig.rig_geometry import build_rig_geometry_cache
from module.auto_rig.skinning import build_skinning_plan
from tests.test_auto_rig_bone_graph import _JOINT_COORDINATES
from tests.test_auto_rig_component_plan import _loaded_part
from tests.test_auto_rig_overrides import _payload, _target, _write_override
from tests.test_auto_rig_rig_geometry import _final_draw_plan


def _rect(x1: int, y1: int, x2: int, y2: int) -> set[tuple[int, int]]:
    return {(x, y) for y in range(y1, y2) for x in range(x1, x2)}


def _part(
    base_tag: str,
    *,
    side: str | None = None,
    xyxy: tuple[int, int, int, int],
    depth: float,
):
    suffix = "" if side is None else f".{side}"
    token = base_tag.replace(" ", "-")
    width = xyxy[2] - xyxy[0]
    height = xyxy[3] - xyxy[1]
    return _loaded_part(
        source_tag=f"{base_tag}{suffix}",
        base_tag=base_tag,
        semantic_slug=f"{token}{suffix}",
        part_id=f"part/{token}{suffix}",
        side=side,
        xyxy=xyxy,
        points=_rect(1, 1, max(2, width - 1), max(2, height - 1)),
        depth=depth,
    )


def _scenario_parts(
    *,
    eyes: bool = True,
    brows: bool = True,
    mouth: bool = True,
    torso: bool = True,
    hair: bool = False,
    limb_mode: str = "split",
    lower_limbs: bool = True,
):
    parts = [
        _part("face", xyxy=(80, 40, 120, 100), depth=0.2),
    ]
    if torso:
        parts.append(_part("topwear", xyxy=(60, 110, 140, 230), depth=0.5))
    if hair:
        parts.extend(
            (
                _part("back hair", xyxy=(72, 32, 128, 108), depth=0.3),
                _part("front hair", xyxy=(76, 34, 124, 88), depth=0.0),
            )
        )
    if mouth:
        parts.append(_part("mouth", xyxy=(92, 80, 108, 89), depth=0.1))
    if eyes:
        for side, x1, x2 in (("xmin", 84, 97), ("xmax", 103, 116)):
            parts.extend(
                (
                    _part("eyewhite", side=side, xyxy=(x1, 58, x2, 68), depth=0.1),
                    _part("irides", side=side, xyxy=(x1 + 3, 59, x2 - 3, 67), depth=0.05),
                    _part("eyelash", side=side, xyxy=(x1, 56, x2, 61), depth=0.0),
                )
            )
    if brows:
        parts.extend(
            (
                _part("eyebrow", side="xmin", xyxy=(84, 49, 97, 54), depth=0.0),
                _part("eyebrow", side="xmax", xyxy=(103, 49, 116, 54), depth=0.0),
            )
        )
    if limb_mode == "split":
        parts.extend(
            (
                _part("handwear", side="xmin", xyxy=(25, 115, 58, 225), depth=0.6),
                _part("handwear", side="xmax", xyxy=(142, 115, 175, 225), depth=0.6),
            )
        )
    elif limb_mode == "merged":
        parts.append(_part("handwear", xyxy=(25, 115, 175, 225), depth=0.6))
    if lower_limbs:
        parts.extend(
            (
                _part("legwear", xyxy=(66, 220, 134, 330), depth=0.7),
                _part("footwear", side="xmin", xyxy=(48, 315, 82, 345), depth=0.8),
                _part("footwear", side="xmax", xyxy=(118, 315, 152, 345), depth=0.8),
            )
        )
    return tuple(parts)


def build_capability_fixture(
    tmp_path: Path,
    *,
    eyes: bool = True,
    brows: bool = True,
    mouth: bool = True,
    torso: bool = True,
    hair: bool = False,
    limb_mode: str = "split",
    lower_limbs: bool = True,
    omitted_joints: tuple[str, ...] = (),
):
    target = _target(tmp_path)
    component_plan = build_mask_component_plan(
        _scenario_parts(
            eyes=eyes,
            brows=brows,
            mouth=mouth,
            torso=torso,
            hair=hair,
            limb_mode=limb_mode,
            lower_limbs=lower_limbs,
        ),
        canvas_edge=768,
        item_root=tmp_path,
    )
    loaded_geometry = load_component_geometry(component_plan, item_root=tmp_path)
    anatomy = build_anatomy_mask_geometry(loaded_geometry, canvas_edge=768)
    payload = _payload(target.target_input_fingerprint)
    payload["joints"] = {
        joint_id: {"x": point[0], "y": point[1], "allow_outside": True}
        for joint_id, point in _JOINT_COORDINATES.items()
        if joint_id not in omitted_joints
    }
    payload["tag_aliases"] = {}
    _write_override(tmp_path, payload)
    overrides = validate_rig_override_source(
        load_rig_override_source(tmp_path),
        target,
    )
    joints = build_stage_a_joint_plan(
        anatomy,
        target=target,
        overrides=overrides,
    )
    bones = build_bone_graph(joints)
    final_draw = _final_draw_plan(component_plan.parts)
    mesh_plan = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=final_draw.part_order,
    )
    component_sources = load_mesh_component_sources(
        component_plan,
        item_root=tmp_path,
    )
    skinning = build_skinning_plan(
        mesh_plan,
        bones,
        joints,
        normalized_parts=component_plan.parts,
        component_sources=component_sources,
        native_variant_set=None,
    )
    cache = build_rig_geometry_cache(
        target=target,
        component_plan=component_plan,
        final_draw=final_draw,
        joints=joints,
        bone_graph=bones,
        mesh_plan=mesh_plan,
        skinning_plan=skinning,
        native_variant_set=None,
        stage_a_fingerprint="sha256:" + "a" * 64,
        stage_b_fingerprint="sha256:" + "b" * 64,
    )
    controls = build_control_registry_plan()
    presets = build_preset_library_plan(controls)
    return cache, anatomy.plan, controls, presets


def _by_preset(plan):
    return {capability.preset_id: capability for capability in plan.capabilities}


def test_full_geometry_derives_core_facial_and_side_specific_capabilities(
    tmp_path: Path,
) -> None:
    cache, anatomy, _controls, presets = build_capability_fixture(tmp_path)

    plan = derive_capabilities(cache, anatomy, presets, native_variant_set=None)
    by_preset = _by_preset(plan)

    assert plan.schema_version == CAPABILITY_PLAN_VERSION
    assert (
        validate_capability_plan(
            plan,
            cache,
            anatomy,
            presets,
            native_variant_set=None,
        )
        is plan
    )
    assert {preset for preset, capability in by_preset.items() if capability.status == "available"} == {
        "idle",
        "breath",
        "head_nod",
        "head_shake",
        "body_sway",
        "arm_sway",
        "leg_sway",
        "blink",
        "talk",
        "happy",
        "unimpressed",
        "sad",
        "surprised",
        "wink_screen_left",
        "wink_screen_right",
        "wave.xmin",
        "wave.xmax",
    }
    assert by_preset["blink"].quality_tier == "procedural"
    assert by_preset["talk"].quality_tier == "procedural_silhouette"
    assert set(by_preset["wave.xmin"].evidence_ids) >= {
        "bone/forearm.xmin",
        "bone/hand.xmin",
        "bone/upper_arm.xmin",
    }


def test_missing_wrist_only_removes_the_matching_wave_capability(tmp_path: Path) -> None:
    cache, anatomy, _controls, presets = build_capability_fixture(
        tmp_path,
        omitted_joints=("joint/wrist.xmin",),
    )

    by_preset = _by_preset(derive_capabilities(cache, anatomy, presets, native_variant_set=None))

    assert by_preset["wave.xmin"].status == "unavailable"
    assert "missing_complete_limb_chain" in by_preset["wave.xmin"].reason_codes
    assert by_preset["wave.xmax"].status == "available"


def test_merged_limb_does_not_claim_two_side_specific_waves(tmp_path: Path) -> None:
    cache, anatomy, _controls, presets = build_capability_fixture(
        tmp_path,
        limb_mode="merged",
    )

    by_preset = _by_preset(derive_capabilities(cache, anatomy, presets, native_variant_set=None))

    assert by_preset["wave.xmin"].status == "unavailable"
    assert by_preset["wave.xmax"].status == "unavailable"
    assert "missing_sided_limb_part" in by_preset["wave.xmin"].reason_codes
    assert by_preset["arm_sway"].status == "available"
    assert by_preset["arm_sway"].quality_tier == "procedural_silhouette"


def test_coarse_limb_sways_survive_missing_articulated_joint_chains(tmp_path: Path) -> None:
    omitted = tuple(
        joint_id
        for joint_id in _JOINT_COORDINATES
        if any(
            token in joint_id
            for token in (
                "shoulder",
                "elbow",
                "wrist",
                "hand_tip",
                "hip.",
                "knee",
                "ankle",
                "toe",
            )
        )
    )
    cache, anatomy, _controls, presets = build_capability_fixture(
        tmp_path,
        limb_mode="merged",
        omitted_joints=omitted,
    )

    by_preset = _by_preset(derive_capabilities(cache, anatomy, presets, native_variant_set=None))

    assert by_preset["arm_sway"].status == "available"
    assert by_preset["arm_sway"].quality_tier == "procedural_silhouette"
    mesh_part = {mesh.mesh_id: mesh.part_id for mesh in cache.skinning_plan.weighted_meshes}
    part_base = {part.part_id: part.base_tag for part in cache.parts}
    assert any(item in mesh_part and part_base[mesh_part[item]] == "handwear" for item in by_preset["arm_sway"].evidence_ids)
    assert by_preset["leg_sway"].status == "available"
    assert by_preset["leg_sway"].quality_tier == "procedural_silhouette"
    assert "joint/pelvis" in by_preset["leg_sway"].evidence_ids
    assert any(
        item in mesh_part and part_base[mesh_part[item]] in {"legwear", "footwear"} for item in by_preset["leg_sway"].evidence_ids
    )
    assert by_preset["wave.xmin"].status == "unavailable"
    assert by_preset["wave.xmax"].status == "unavailable"


def test_head_only_geometry_never_promotes_synthetic_root_to_torso_capability(
    tmp_path: Path,
) -> None:
    omitted = tuple(
        joint_id for joint_id in _JOINT_COORDINATES if joint_id not in {"joint/neck", "joint/head_base", "joint/head_top"}
    )
    cache, anatomy, _controls, presets = build_capability_fixture(
        tmp_path,
        eyes=False,
        brows=False,
        mouth=False,
        torso=False,
        limb_mode="missing",
        lower_limbs=False,
        omitted_joints=omitted,
    )

    by_preset = _by_preset(derive_capabilities(cache, anatomy, presets, native_variant_set=None))

    assert {bone.bone_id for bone in cache.bone_graph.bones} >= {
        "bone/root",
        "bone/neck",
        "bone/head",
    }
    assert by_preset["head_nod"].status == "available"
    assert by_preset["head_shake"].status == "available"
    assert by_preset["idle"].status == "unavailable"
    assert by_preset["breath"].status == "unavailable"
    assert "bone/root" not in by_preset["idle"].evidence_ids


def test_missing_eye_layers_remove_only_presets_that_actually_drive_eyes(tmp_path: Path) -> None:
    cache, anatomy, _controls, presets = build_capability_fixture(
        tmp_path,
        eyes=False,
    )

    by_preset = _by_preset(derive_capabilities(cache, anatomy, presets, native_variant_set=None))

    assert by_preset["blink"].status == "unavailable"
    assert by_preset["wink_screen_left"].status == "unavailable"
    assert by_preset["wink_screen_right"].status == "unavailable"
    assert by_preset["happy"].status == "available"
    assert by_preset["unimpressed"].status == "available"
    assert by_preset["sad"].status == "available"
    assert by_preset["talk"].status == "available"


def test_capability_validator_rederives_records_after_outer_rehash(tmp_path: Path) -> None:
    cache, anatomy, _controls, presets = build_capability_fixture(tmp_path)
    plan = derive_capabilities(cache, anatomy, presets, native_variant_set=None)
    changed = replace(plan.capabilities[0], status="unavailable", capability_sha256="")
    changed = replace(
        changed,
        capability_sha256=jcs_sha256(changed.semantic_payload()),
    )
    provisional = replace(
        plan,
        capabilities=(changed, *plan.capabilities[1:]),
        plan_sha256="",
    )
    tampered = replace(
        provisional,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )

    with pytest.raises(CapabilityPlanError) as exc_info:
        validate_capability_plan(
            tampered,
            cache,
            anatomy,
            presets,
            native_variant_set=None,
        )

    assert exc_info.value.code == "invalid_capability_plan"
