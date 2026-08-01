from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from module.auto_rig.artifacts import atomic_write_bytes
from module.auto_rig.bone_graph import build_bone_graph
from module.auto_rig.component_geometry import load_mesh_component_sources
from module.auto_rig.component_plan import build_mask_component_plan
from module.auto_rig.contracts import load_auto_rig_input_contract
from module.auto_rig.draw_order import build_ordinary_draw_order
from module.auto_rig.input_identity import build_target_input_identity
from module.auto_rig.jcs import jcs_sha256
from module.auto_rig.manifests import build_stage_fingerprint, build_stage_manifest
from module.auto_rig.mesh_builder import build_mesh_plan
from module.auto_rig.native_variant_admission import (
    DRAW_ANCHOR_EXPANDER_VERSION,
    FINAL_DRAW_ORDER_PLAN_VERSION,
    FinalDrawOrderPlan,
    FinalPartDrawRecord,
)
from module.auto_rig.rig_geometry import (
    COMPONENT_DRAW_ORDER_EXPANDER_VERSION,
    RIG_GEOMETRY_CACHE_PATH,
    RIG_GEOMETRY_CACHE_VERSION,
    ComponentDrawOrderError,
    RigGeometryCacheError,
    build_rig_geometry_cache,
    expand_component_draw_order,
    rig_geometry_cache_bytes,
    validate_component_draw_order,
    validate_rig_geometry_cache,
    validate_rig_geometry_cache_payload,
)
from module.auto_rig.skinning import build_skinning_plan
from tests.test_auto_rig_bone_graph import _joint_stage
from tests.test_auto_rig_component_plan import _loaded_part

_EMPTY_VARIANT_SET_SHA256 = jcs_sha256(
    {"schema_version": "native-variant-set-v1", "present": False, "entries": []}
)
_EMPTY_ELIGIBILITY_SHA256 = jcs_sha256(
    {"schema_version": "native-variant-eligibility-plan-v1", "render_variant_ids": []}
)


def _final_draw_plan(parts) -> FinalDrawOrderPlan:
    ordinary = build_ordinary_draw_order(parts)
    records = tuple(
        FinalPartDrawRecord(
            part_id=record.part_id,
            source_kind="see_through",
            part_draw_rank=record.part_draw_rank,
            depth_bucket=record.depth_bucket,
            draw_anchor_part_id=None,
            draw_bundle_id=None,
        )
        for record in ordinary.records
    )
    content = {
        "schema_version": FINAL_DRAW_ORDER_PLAN_VERSION,
        "anchor_expander_version": DRAW_ANCHOR_EXPANDER_VERSION,
        "ordinary_draw_order_plan_sha256": ordinary.plan_sha256,
        "native_variant_set_sha256": _EMPTY_VARIANT_SET_SHA256,
        "native_variant_eligibility_plan_sha256": _EMPTY_ELIGIBILITY_SHA256,
        "part_order": list(ordinary.ordinary_part_order),
        "records": [record.to_dict() for record in records],
    }
    return FinalDrawOrderPlan(
        schema_version=FINAL_DRAW_ORDER_PLAN_VERSION,
        anchor_expander_version=DRAW_ANCHOR_EXPANDER_VERSION,
        ordinary_draw_order_plan_sha256=ordinary.plan_sha256,
        native_variant_set_sha256=_EMPTY_VARIANT_SET_SHA256,
        native_variant_eligibility_plan_sha256=_EMPTY_ELIGIBILITY_SHA256,
        part_order=ordinary.ordinary_part_order,
        records=records,
        plan_sha256=jcs_sha256(content),
    )


def _parts():
    face = _loaded_part(
        source_tag="face",
        base_tag="face",
        semantic_slug="face",
        part_id="part/face",
        side=None,
        xyxy=(10, 10, 30, 30),
        points={
            (1, 1),
            (2, 1),
            (1, 2),
            (2, 2),
            (12, 12),
            (13, 12),
            (12, 13),
            (13, 13),
        },
        depth=0.2,
    )
    handwear = _loaded_part(
        source_tag="handwear",
        base_tag="handwear",
        semantic_slug="handwear",
        part_id="part/handwear",
        side=None,
        xyxy=(40, 10, 50, 20),
        points={(x, y) for y in range(2, 7) for x in range(2, 7)},
        depth=0.8,
    )
    return face, handwear


def _plans(tmp_path: Path, *, reverse_parts: bool = False):
    parts = _parts()
    component_plan = build_mask_component_plan(
        tuple(reversed(parts)) if reverse_parts else parts,
        canvas_edge=768,
        item_root=tmp_path,
    )
    final_draw = _final_draw_plan(component_plan.parts)
    mesh_plan = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=final_draw.part_order,
    )
    return component_plan, final_draw, mesh_plan


def test_component_draw_expander_assigns_gapless_contiguous_stable_ranks(
    tmp_path: Path,
) -> None:
    _component_plan, final_draw, mesh_plan = _plans(tmp_path)

    plan = expand_component_draw_order(final_draw, mesh_plan)

    assert COMPONENT_DRAW_ORDER_EXPANDER_VERSION == "component-draw-order-expander-v1"
    assert validate_component_draw_order(plan, final_draw=final_draw, mesh_plan=mesh_plan) is plan
    assert tuple(record.component_draw_rank for record in plan.records) == tuple(
        range(len(plan.records))
    )
    records_by_part = {
        part_id: tuple(record for record in plan.records if record.part_id == part_id)
        for part_id in final_draw.part_order
    }
    for part_id, records in records_by_part.items():
        assert tuple(record.component_id for record in records) == tuple(
            sorted(record.component_id for record in records)
        )
        ranks = tuple(record.component_draw_rank for record in records)
        assert ranks == tuple(range(min(ranks), max(ranks) + 1)), part_id


def test_component_draw_ranks_ignore_part_and_mesh_construction_order(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    _first_components, first_draw, first_mesh = _plans(first_root)
    _second_components, second_draw, second_mesh = _plans(
        second_root,
        reverse_parts=True,
    )

    first = expand_component_draw_order(first_draw, first_mesh)
    second = expand_component_draw_order(second_draw, second_mesh)

    assert first == second


def test_component_draw_expander_rejects_a_mesh_outside_the_a_component_table(
    tmp_path: Path,
) -> None:
    _component_plan, final_draw, mesh_plan = _plans(tmp_path)
    mesh = mesh_plan.meshes[0]
    external = replace(mesh, component_id="component/c_" + "f" * 64, mesh_sha256="")
    external = replace(external, mesh_sha256=jcs_sha256(external.content_payload()))
    provisional = replace(mesh_plan, meshes=(external, *mesh_plan.meshes[1:]), plan_sha256="")
    tampered = replace(
        provisional,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )

    with pytest.raises(ComponentDrawOrderError):
        expand_component_draw_order(final_draw, tampered)


def _full_geometry_inputs(tmp_path: Path):
    joints = _joint_stage(tmp_path)
    target = build_target_input_identity(load_auto_rig_input_contract(tmp_path))
    component_plan, final_draw, mesh_plan = _plans(tmp_path)
    component_sources = load_mesh_component_sources(component_plan, item_root=tmp_path)
    bones = build_bone_graph(joints)
    skinning = build_skinning_plan(
        mesh_plan,
        bones,
        joints,
        normalized_parts=component_plan.parts,
        component_sources=component_sources,
        native_variant_set=None,
    )
    return target, component_plan, final_draw, joints, bones, mesh_plan, skinning


def _rehash_cache(cache, **changes):
    provisional = replace(cache, **changes, cache_sha256="")
    return replace(
        provisional,
        cache_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def test_rig_geometry_cache_is_complete_b_owned_geometry_not_a_partial_rig_document(
    tmp_path: Path,
) -> None:
    target, components, final_draw, joints, bones, meshes, skinning = (
        _full_geometry_inputs(tmp_path)
    )

    cache = build_rig_geometry_cache(
        target=target,
        component_plan=components,
        final_draw=final_draw,
        joints=joints,
        bone_graph=bones,
        mesh_plan=meshes,
        skinning_plan=skinning,
        native_variant_set=None,
        stage_a_fingerprint="sha256:" + "a" * 64,
        stage_b_fingerprint="sha256:" + "b" * 64,
    )
    payload = json.loads(rig_geometry_cache_bytes(cache))

    assert RIG_GEOMETRY_CACHE_VERSION == "rig-geometry-cache-v1"
    assert validate_rig_geometry_cache(cache) is cache
    assert payload["cache_schema_version"] == RIG_GEOMETRY_CACHE_VERSION
    assert payload["target_input_fingerprint"] == target.target_input_fingerprint
    assert tuple(part["part_draw_rank"] for part in payload["parts"]) == tuple(
        range(len(payload["parts"]))
    )
    assert tuple(
        record["component_draw_rank"] for record in payload["component_draw_order"]["records"]
    ) == tuple(range(len(meshes.sources)))
    assert payload["degradation_state"] == "degraded"
    assert "rigid_fallback_applied" in payload["degradation_codes"]
    forbidden = {
        "capabilities",
        "control_specs",
        "control_bindings",
        "clips",
        "expressions",
        "format_plans",
        "primitive_candidates",
        "export_symbols",
        "texture_pages",
    }
    assert not (forbidden & set(payload))


def test_rig_geometry_cache_validator_rejects_rehashed_rank_drift_and_c_fields(
    tmp_path: Path,
) -> None:
    target, components, final_draw, joints, bones, meshes, skinning = (
        _full_geometry_inputs(tmp_path)
    )
    cache = build_rig_geometry_cache(
        target=target,
        component_plan=components,
        final_draw=final_draw,
        joints=joints,
        bone_graph=bones,
        mesh_plan=meshes,
        skinning_plan=skinning,
        native_variant_set=None,
        stage_a_fingerprint="sha256:" + "a" * 64,
        stage_b_fingerprint="sha256:" + "b" * 64,
    )
    draw = cache.component_draw_order
    changed_record = replace(draw.records[0], component_draw_rank=99)
    changed_draw = replace(
        draw,
        records=(changed_record, *draw.records[1:]),
        plan_sha256="",
    )
    changed_draw = replace(
        changed_draw,
        plan_sha256=jcs_sha256(changed_draw.semantic_payload()),
    )
    tampered = _rehash_cache(cache, component_draw_order=changed_draw)

    with pytest.raises(RigGeometryCacheError):
        validate_rig_geometry_cache(tampered)

    payload = cache.to_dict()
    payload["capabilities"] = []
    with pytest.raises(RigGeometryCacheError):
        validate_rig_geometry_cache_payload(payload)


def test_rig_geometry_cache_validator_rejects_rehashed_reference_drift(
    tmp_path: Path,
) -> None:
    target, components, final_draw, joints, bones, meshes, skinning = (
        _full_geometry_inputs(tmp_path)
    )
    cache = build_rig_geometry_cache(
        target=target,
        component_plan=components,
        final_draw=final_draw,
        joints=joints,
        bone_graph=bones,
        mesh_plan=meshes,
        skinning_plan=skinning,
        native_variant_set=None,
        stage_a_fingerprint="sha256:" + "a" * 64,
        stage_b_fingerprint="sha256:" + "b" * 64,
    )

    changed_part = replace(
        cache.parts[0],
        component_ids=("mask-component/external",),
    )
    with pytest.raises(RigGeometryCacheError):
        validate_rig_geometry_cache(
            _rehash_cache(cache, parts=(changed_part, *cache.parts[1:]))
        )

    child_index = next(
        index for index, bone in enumerate(cache.bone_graph.bones) if bone.parent_id
    )
    changed_bone = replace(
        cache.bone_graph.bones[child_index],
        parent_id="bone/missing",
    )
    changed_bones = list(cache.bone_graph.bones)
    changed_bones[child_index] = changed_bone
    provisional_graph = replace(
        cache.bone_graph,
        bones=tuple(changed_bones),
        plan_sha256="",
    )
    changed_graph = replace(
        provisional_graph,
        plan_sha256=jcs_sha256(provisional_graph.semantic_payload()),
    )
    with pytest.raises(RigGeometryCacheError):
        validate_rig_geometry_cache(
            _rehash_cache(cache, bone_graph=changed_graph)
        )


def test_stage_b_owns_only_the_private_geometry_cache_payload(tmp_path: Path) -> None:
    target, components, final_draw, joints, bones, meshes, skinning = (
        _full_geometry_inputs(tmp_path)
    )
    cache = build_rig_geometry_cache(
        target=target,
        component_plan=components,
        final_draw=final_draw,
        joints=joints,
        bone_graph=bones,
        mesh_plan=meshes,
        skinning_plan=skinning,
        native_variant_set=None,
        stage_a_fingerprint="sha256:" + "a" * 64,
        stage_b_fingerprint="sha256:" + "b" * 64,
    )
    cache_path = tmp_path / Path(*RIG_GEOMETRY_CACHE_PATH.split("/"))
    atomic_write_bytes(cache_path, rig_geometry_cache_bytes(cache))

    manifest = build_stage_manifest(
        tmp_path,
        stage_name="B",
        stage_schema_version=1,
        algorithm_version=RIG_GEOMETRY_CACHE_VERSION,
        upstream_manifests={"A": "sha256:" + "c" * 64},
        input_file_sha256=(),
        target_input_fingerprint=target.target_input_fingerprint,
        native_variant_set_sha256=final_draw.native_variant_set_sha256,
        native_variant_eligibility_sha256=(
            final_draw.native_variant_eligibility_plan_sha256
        ),
        relevant_config_fingerprint="sha256:" + "d" * 64,
        rig_overrides_sha256=joints.rig_overrides_sha256,
        output_paths=(RIG_GEOMETRY_CACHE_PATH,),
        status=(
            "stage_validated_with_degradation"
            if cache.degradation_state == "degraded"
            else "stage_validated"
        ),
    )

    assert tuple(item.path for item in manifest.output_file_sha256) == (
        RIG_GEOMETRY_CACHE_PATH,
    )
    assert not (tmp_path / "rig" / "rig.json").exists()


def test_mesh_descriptor_change_invalidates_b_and_c_but_not_a(tmp_path: Path) -> None:
    target, _components, final_draw, joints, _bones, meshes, _skinning = (
        _full_geometry_inputs(tmp_path)
    )
    common = {
        "stage_schema_version": 1,
        "input_file_sha256": (),
        "target_input_fingerprint": target.target_input_fingerprint,
        "native_variant_set_sha256": final_draw.native_variant_set_sha256,
        "native_variant_eligibility_sha256": (
            final_draw.native_variant_eligibility_plan_sha256
        ),
        "rig_overrides_sha256": joints.rig_overrides_sha256,
        "status": "stage_validated",
    }
    a_fingerprint = build_stage_fingerprint(
        **common,
        stage_name="A",
        algorithm_version="stage-a-v1",
        upstream_manifests={},
        relevant_config_fingerprint=jcs_sha256({"joint_pipeline": "v1"}),
    )
    changed_a_fingerprint = build_stage_fingerprint(
        **common,
        stage_name="A",
        algorithm_version="stage-a-v1",
        upstream_manifests={},
        relevant_config_fingerprint=jcs_sha256({"joint_pipeline": "v1"}),
    )
    baseline_mesh_config = jcs_sha256(
        {"mesh_descriptor_sha256": meshes.descriptor.descriptor_sha256}
    )
    changed_mesh_config = jcs_sha256(
        {
            "mesh_descriptor_sha256": meshes.descriptor.descriptor_sha256,
            "sampling_revision": 2,
        }
    )
    b_fingerprint = build_stage_fingerprint(
        **common,
        stage_name="B",
        algorithm_version=RIG_GEOMETRY_CACHE_VERSION,
        upstream_manifests={"A": a_fingerprint},
        relevant_config_fingerprint=baseline_mesh_config,
    )
    changed_b_fingerprint = build_stage_fingerprint(
        **common,
        stage_name="B",
        algorithm_version=RIG_GEOMETRY_CACHE_VERSION,
        upstream_manifests={"A": a_fingerprint},
        relevant_config_fingerprint=changed_mesh_config,
    )
    c_fingerprint = build_stage_fingerprint(
        **common,
        stage_name="C",
        algorithm_version="rig-document-v1",
        upstream_manifests={"A": a_fingerprint, "B": b_fingerprint},
        relevant_config_fingerprint=jcs_sha256({"profile": "dual-runtime-core-v1"}),
    )
    changed_c_fingerprint = build_stage_fingerprint(
        **common,
        stage_name="C",
        algorithm_version="rig-document-v1",
        upstream_manifests={"A": a_fingerprint, "B": changed_b_fingerprint},
        relevant_config_fingerprint=jcs_sha256({"profile": "dual-runtime-core-v1"}),
    )

    assert changed_a_fingerprint == a_fingerprint
    assert changed_b_fingerprint != b_fingerprint
    assert changed_c_fingerprint != c_fingerprint
