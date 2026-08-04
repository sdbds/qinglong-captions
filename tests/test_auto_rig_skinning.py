from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

import module.auto_rig.skinning as skinning_module
from module.auto_rig.bone_graph import build_bone_graph
from module.auto_rig.component_geometry import load_mesh_component_sources
from module.auto_rig.component_plan import (
    build_mask_component_plan,
    extend_mask_component_plan_with_variants,
)
from module.auto_rig.jcs import jcs_sha256
from module.auto_rig.mesh_builder import build_mesh_plan
from module.auto_rig.skinning import (
    PART_BONE_BINDING_REGISTRY_VERSION,
    SKINNING_PLAN_VERSION,
    build_part_bone_bindings,
    build_skinning_plan,
    compute_arc_length_influences,
    validate_skinning_plan,
)
from tests.test_auto_rig_bone_graph import _JOINT_COORDINATES, _joint_stage
from tests.test_auto_rig_component_plan import _loaded_part, _variant, _variant_set


def _solid_part(
    *,
    source_tag: str,
    base_tag: str,
    part_id: str,
    side: str | None,
    x: int,
):
    return _loaded_part(
        source_tag=source_tag,
        base_tag=base_tag,
        semantic_slug=base_tag.replace(" ", "-"),
        part_id=part_id,
        side=side,
        xyxy=(x, 20, x + 8, 28),
        points={(px, py) for py in range(8) for px in range(8)},
    )


def _semantic_mesh_plan(tmp_path: Path):
    loaded = (
        _solid_part(
            source_tag="face",
            base_tag="face",
            part_id="part/face",
            side=None,
            x=10,
        ),
        _solid_part(
            source_tag="front hair",
            base_tag="front hair",
            part_id="part/front-hair",
            side=None,
            x=30,
        ),
        _solid_part(
            source_tag="handwear-r",
            base_tag="handwear",
            part_id="part/handwear.xmin",
            side="xmin",
            x=50,
        ),
        _solid_part(
            source_tag="handwear-l",
            base_tag="handwear",
            part_id="part/handwear.xmax",
            side="xmax",
            x=70,
        ),
        _solid_part(
            source_tag="legwear",
            base_tag="legwear",
            part_id="part/legwear.xmin",
            side="xmin",
            x=90,
        ),
        _solid_part(
            source_tag="objects",
            base_tag="objects",
            part_id="part/objects",
            side=None,
            x=110,
        ),
    )
    component_plan = build_mask_component_plan(
        loaded,
        canvas_edge=768,
        item_root=tmp_path,
    )
    final_part_ids = tuple(part.part_id for part in component_plan.parts)
    mesh_plan = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=final_part_ids,
    )
    return component_plan, mesh_plan


def test_part_bone_registry_exposes_both_limb_chains_for_geometry_selection(
    tmp_path: Path,
) -> None:
    component_plan, mesh_plan = _semantic_mesh_plan(tmp_path)
    bone_graph = build_bone_graph(_joint_stage(tmp_path))

    bindings = build_part_bone_bindings(
        mesh_plan,
        bone_graph,
        normalized_parts=component_plan.parts,
        native_variant_set=None,
    )
    by_part = {binding.part_id: binding for binding in bindings}

    assert PART_BONE_BINDING_REGISTRY_VERSION == "part-bone-binding-registry-v2"
    assert SKINNING_PLAN_VERSION == "skinning-plan-v3"
    assert by_part["part/face"].semantic_candidate_bone_ids == ("bone/head",)
    assert by_part["part/front-hair"].semantic_candidate_bone_ids == ("bone/head",)
    assert by_part["part/handwear.xmin"].semantic_candidate_bone_ids == (
        "bone/upper_arm.xmin",
        "bone/forearm.xmin",
        "bone/hand.xmin",
        "bone/upper_arm.xmax",
        "bone/forearm.xmax",
        "bone/hand.xmax",
    )
    assert by_part["part/handwear.xmax"].semantic_candidate_bone_ids == (
        "bone/upper_arm.xmax",
        "bone/forearm.xmax",
        "bone/hand.xmax",
        "bone/upper_arm.xmin",
        "bone/forearm.xmin",
        "bone/hand.xmin",
    )
    assert by_part["part/legwear.xmin"].semantic_candidate_bone_ids == (
        "bone/thigh.xmin",
        "bone/shin.xmin",
        "bone/foot.xmin",
        "bone/thigh.xmax",
        "bone/shin.xmax",
        "bone/foot.xmax",
    )
    assert by_part["part/objects"].dynamic_candidate is True
    assert not (
        set(by_part["part/face"].semantic_candidate_bone_ids)
        & {
            "bone/upper_arm.xmin",
            "bone/upper_arm.xmax",
            "bone/thigh.xmin",
            "bone/thigh.xmax",
        }
    )


def test_native_variant_inherits_its_admitted_anchor_binding(tmp_path: Path) -> None:
    mouth = _solid_part(
        source_tag="mouth",
        base_tag="mouth",
        part_id="part/mouth",
        side=None,
        x=10,
    )
    base_plan = build_mask_component_plan((mouth,), canvas_edge=768, item_root=tmp_path)
    variants = _variant_set(_variant("mouth-open"))
    component_plan = extend_mask_component_plan_with_variants(
        base_plan,
        variants,
        tmp_path,
    )
    mesh_plan = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=("mouth-open",),
        final_part_ids=("part/mouth", "part/native.mouth-open"),
    )
    bone_graph = build_bone_graph(_joint_stage(tmp_path))

    bindings = build_part_bone_bindings(
        mesh_plan,
        bone_graph,
        normalized_parts=component_plan.parts,
        native_variant_set=variants,
    )
    by_part = {binding.part_id: binding for binding in bindings}

    base = by_part["part/mouth"]
    variant = by_part["part/native.mouth-open"]
    assert variant.anchor_part_id == base.part_id
    assert variant.semantic_candidate_bone_ids == base.semantic_candidate_bone_ids
    assert variant.fallback_bone_ids == base.fallback_bone_ids
    assert variant.dynamic_candidate == base.dynamic_candidate


def _single_part_mesh(
    tmp_path: Path,
    *,
    source_tag: str,
    base_tag: str,
    part_id: str,
    side: str | None,
):
    loaded = _solid_part(
        source_tag=source_tag,
        base_tag=base_tag,
        part_id=part_id,
        side=side,
        x=10,
    )
    component_plan = build_mask_component_plan(
        (loaded,),
        canvas_edge=768,
        item_root=tmp_path,
    )
    mesh_plan = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=(part_id,),
    )
    sources = load_mesh_component_sources(component_plan, item_root=tmp_path)
    return component_plan, mesh_plan, sources


def test_skinning_plan_assigns_non_limb_meshes_one_semantic_bone(
    tmp_path: Path,
) -> None:
    component_plan, mesh_plan, sources = _single_part_mesh(
        tmp_path,
        source_tag="face",
        base_tag="face",
        part_id="part/face",
        side=None,
    )
    joints = _joint_stage(tmp_path)
    bones = build_bone_graph(joints)

    plan = build_skinning_plan(
        mesh_plan,
        bones,
        joints,
        normalized_parts=component_plan.parts,
        component_sources=sources,
        native_variant_set=None,
    )

    assert validate_skinning_plan(plan) is plan
    assert plan.diagnostics == ()
    assert len(plan.weighted_meshes) == 1
    assert all(
        tuple((item.bone_id, item.weight) for item in vertex.influences) == (("bone/head", 1.0),)
        for vertex in plan.weighted_meshes[0].vertices
    )


def test_skinning_plan_records_rigid_fallback_for_missing_semantic_bone(
    tmp_path: Path,
) -> None:
    component_plan, mesh_plan, sources = _single_part_mesh(
        tmp_path,
        source_tag="face",
        base_tag="face",
        part_id="part/face",
        side=None,
    )
    joints = _joint_stage(
        tmp_path,
        omitted=("joint/head_base", "joint/head_top"),
    )
    bones = build_bone_graph(joints)

    plan = build_skinning_plan(
        mesh_plan,
        bones,
        joints,
        normalized_parts=component_plan.parts,
        component_sources=sources,
        native_variant_set=None,
    )

    assert {diagnostic.code for diagnostic in plan.diagnostics} == {"rigid_fallback_applied"}
    assert all(vertex.influences[0].bone_id == "bone/torso" for vertex in plan.weighted_meshes[0].vertices)


def test_skinning_plan_degrades_a_one_bone_limb_to_rigid(tmp_path: Path) -> None:
    component_plan, mesh_plan, sources = _single_part_mesh(
        tmp_path,
        source_tag="handwear-r",
        base_tag="handwear",
        part_id="part/handwear.xmin",
        side="xmin",
    )
    joints = _joint_stage(tmp_path, omitted=("joint/wrist.xmin",))
    bones = build_bone_graph(joints)

    plan = build_skinning_plan(
        mesh_plan,
        bones,
        joints,
        normalized_parts=component_plan.parts,
        component_sources=sources,
        native_variant_set=None,
    )

    assert tuple(item.code for item in plan.diagnostics) == ("rigid_fallback_applied",)
    assert all(vertex.influences[0].bone_id == "bone/upper_arm.xmin" for vertex in plan.weighted_meshes[0].vertices)


def test_skinning_plan_keeps_unknown_parts_visible_and_diagnosed(
    tmp_path: Path,
) -> None:
    component_plan, mesh_plan, sources = _single_part_mesh(
        tmp_path,
        source_tag="mystery",
        base_tag="mystery",
        part_id="part/mystery",
        side=None,
    )
    joints = _joint_stage(tmp_path)
    bones = build_bone_graph(joints)

    plan = build_skinning_plan(
        mesh_plan,
        bones,
        joints,
        normalized_parts=component_plan.parts,
        component_sources=sources,
        native_variant_set=None,
    )

    assert {item.code for item in plan.diagnostics} == {
        "unknown_part_semantics",
        "rigid_fallback_applied",
    }
    assert all(
        vertex.influences == (plan.weighted_meshes[0].vertices[0].influences[0],) and vertex.influences[0].bone_id == "bone/root"
        for vertex in plan.weighted_meshes[0].vertices
    )


def _influence_map(influences) -> dict[str, float]:
    return {influence.bone_id: influence.weight for influence in influences}


def test_arc_length_weights_follow_a_bent_chain_and_blend_only_adjacent_bones() -> None:
    bones = ("bone/upper", "bone/middle", "bone/lower")
    points = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
    radii = (2.0, 2.0)

    proximal = compute_arc_length_influences(
        position=(4.0, 0.0),
        bone_ids=bones,
        joint_points=points,
        internal_joint_radii=radii,
    )
    first_joint = compute_arc_length_influences(
        position=(10.0, 0.0),
        bone_ids=bones,
        joint_points=points,
        internal_joint_radii=radii,
    )
    distal = compute_arc_length_influences(
        position=(1.0, 10.0),
        bone_ids=bones,
        joint_points=points,
        internal_joint_radii=radii,
    )

    assert proximal == (proximal[0],)
    assert _influence_map(proximal) == {"bone/upper": 1.0}
    assert _influence_map(first_joint) == {
        "bone/middle": 0.5,
        "bone/upper": 0.5,
    }
    assert _influence_map(distal) == {"bone/lower": 1.0}
    assert all(len(influences) <= 2 for influences in (proximal, first_joint, distal))


def test_arc_length_weights_are_scale_invariant_and_prune_zero_width_tails() -> None:
    bones = ("bone/a", "bone/b")
    points = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0))
    scale = 1280.0 / 768.0
    base = compute_arc_length_influences(
        position=(9.0, 0.0),
        bone_ids=bones,
        joint_points=points,
        internal_joint_radii=(2.0,),
    )
    scaled = compute_arc_length_influences(
        position=(9.0 * scale, 0.0),
        bone_ids=bones,
        joint_points=tuple((x * scale, y * scale) for x, y in points),
        internal_joint_radii=(2.0 * scale,),
    )
    band_edge = compute_arc_length_influences(
        position=(8.0, 0.0),
        bone_ids=bones,
        joint_points=points,
        internal_joint_radii=(2.0,),
    )

    assert base == scaled
    assert _influence_map(base) == {"bone/a": 0.75, "bone/b": 0.25}
    assert _influence_map(band_edge) == {"bone/a": 1.0}
    assert all(influence.weight > 0.0 for influence in (*base, *band_edge))
    assert sum(influence.weight for influence in base) == 1.0


def _thick_polyline_points(
    points: tuple[tuple[float, float], ...],
    *,
    bbox: tuple[int, int, int, int],
    radius: int,
) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    x1, y1, x2, y2 = bbox
    for start, end in zip(points[:-1], points[1:], strict=True):
        for step in range(101):
            fraction = step / 100.0
            center_x = start[0] + fraction * (end[0] - start[0])
            center_y = start[1] + fraction * (end[1] - start[1])
            for y in range(math.floor(center_y - radius), math.ceil(center_y + radius) + 1):
                for x in range(math.floor(center_x - radius), math.ceil(center_x + radius) + 1):
                    if x1 <= x < x2 and y1 <= y < y2 and (x + 0.5 - center_x) ** 2 + (y + 0.5 - center_y) ** 2 <= radius**2:
                        result.add((x - x1, y - y1))
    return result


def test_skinning_plan_uses_component_radius_and_joint_chain_for_bent_limb_weights(
    tmp_path: Path,
) -> None:
    bbox = (20, 110, 85, 235)
    chain = ((70.0, 125.0), (50.0, 160.0), (40.0, 195.0), (35.0, 220.0))
    loaded = _loaded_part(
        source_tag="handwear-r",
        base_tag="handwear",
        semantic_slug="handwear",
        part_id="part/handwear.xmin",
        side="xmin",
        xyxy=bbox,
        points=_thick_polyline_points(chain, bbox=bbox, radius=7),
    )
    component_plan = build_mask_component_plan(
        (loaded,),
        canvas_edge=768,
        item_root=tmp_path,
    )
    mesh_plan = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=("part/handwear.xmin",),
    )
    sources = load_mesh_component_sources(component_plan, item_root=tmp_path)
    joints = _joint_stage(tmp_path)
    bones = build_bone_graph(joints)

    plan = build_skinning_plan(
        mesh_plan,
        bones,
        joints,
        normalized_parts=component_plan.parts,
        component_sources=sources,
        native_variant_set=None,
    )

    assert plan.diagnostics == ()
    mesh = plan.weighted_meshes[0]
    assert mesh.allowed_bone_ids == (
        "bone/upper_arm.xmin",
        "bone/forearm.xmin",
        "bone/hand.xmin",
        "bone/upper_arm.xmax",
        "bone/forearm.xmax",
        "bone/hand.xmax",
    )
    assert any(len(vertex.influences) == 2 for vertex in mesh.vertices)
    assert {influence.bone_id for vertex in mesh.vertices for influence in vertex.influences} == {
        "bone/upper_arm.xmin",
        "bone/forearm.xmin",
        "bone/hand.xmin",
    }
    assert all(abs(sum(influence.weight for influence in vertex.influences) - 1.0) <= 1e-12 for vertex in mesh.vertices)
    assert all(".xmax" not in influence.bone_id for vertex in mesh.vertices for influence in vertex.influences)


def test_skinning_plan_partitions_an_unsided_merged_limb_across_both_pose_chains(
    tmp_path: Path,
) -> None:
    bbox = (20, 110, 180, 230)
    left_chain = ((70.0, 125.0), (50.0, 160.0), (40.0, 195.0), (35.0, 220.0))
    right_chain = (
        (130.0, 125.0),
        (150.0, 160.0),
        (160.0, 195.0),
        (165.0, 220.0),
    )
    loaded = _loaded_part(
        source_tag="handwear",
        base_tag="handwear",
        semantic_slug="handwear",
        part_id="part/handwear",
        side=None,
        xyxy=bbox,
        points=(
            _thick_polyline_points(left_chain, bbox=bbox, radius=7)
            | _thick_polyline_points(right_chain, bbox=bbox, radius=7)
            | {(x - bbox[0], y - bbox[1]) for y in range(123, 129) for x in range(70, 131)}
        ),
    )
    component_plan = build_mask_component_plan(
        (loaded,),
        canvas_edge=768,
        item_root=tmp_path,
    )
    mesh_plan = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=tuple(part.part_id for part in component_plan.parts),
    )
    sources = load_mesh_component_sources(component_plan, item_root=tmp_path)
    joints = _joint_stage(
        tmp_path,
        omitted=("joint/hand_tip.xmin", "joint/hand_tip.xmax"),
    )
    bones = build_bone_graph(joints)

    plan = build_skinning_plan(
        mesh_plan,
        bones,
        joints,
        normalized_parts=component_plan.parts,
        component_sources=sources,
        native_variant_set=None,
    )

    assert plan.diagnostics == ()
    binding = plan.part_bindings[0]
    assert binding.side is None
    assert binding.usable_bone_ids == (
        "bone/upper_arm.xmin",
        "bone/forearm.xmin",
        "bone/upper_arm.xmax",
        "bone/forearm.xmax",
    )
    all_influences = {
        influence.bone_id for mesh in plan.weighted_meshes for vertex in mesh.vertices for influence in vertex.influences
    }
    assert all_influences == set(binding.usable_bone_ids)
    assert "bone/root" not in all_influences

    left_vertices = [vertex for mesh in plan.weighted_meshes for vertex in mesh.vertices if vertex.position[0] < 80.0]
    right_vertices = [vertex for mesh in plan.weighted_meshes for vertex in mesh.vertices if vertex.position[0] > 120.0]
    assert left_vertices and right_vertices
    assert {influence.bone_id.rsplit(".", 1)[-1] for vertex in left_vertices for influence in vertex.influences} == {"xmin"}
    assert {influence.bone_id.rsplit(".", 1)[-1] for vertex in right_vertices for influence in vertex.influences} == {"xmax"}


def test_skinning_plan_uses_geometry_not_part_suffix_for_crossed_distal_limbs(
    tmp_path: Path,
) -> None:
    bbox = (52, 258, 88, 344)
    loaded = _loaded_part(
        source_tag="footwear-r",
        base_tag="footwear",
        semantic_slug="footwear",
        part_id="part/footwear.xmin",
        side="xmin",
        xyxy=bbox,
        points={(x, y) for y in range(1, 85) for x in range(1, 35)},
    )
    component_plan = build_mask_component_plan(
        (loaded,),
        canvas_edge=768,
        item_root=tmp_path,
    )
    mesh_plan = build_mesh_plan(
        component_plan,
        item_root=tmp_path,
        render_variant_ids=(),
        final_part_ids=tuple(part.part_id for part in component_plan.parts),
    )
    sources = load_mesh_component_sources(component_plan, item_root=tmp_path)
    coordinates = {
        **_JOINT_COORDINATES,
        "joint/knee.xmin": (55, 260),
        "joint/ankle.xmin": (130, 320),
        "joint/toe.xmin": (145, 335),
        "joint/knee.xmax": (115, 260),
        "joint/ankle.xmax": (70, 320),
        "joint/toe.xmax": (55, 335),
    }
    joints = _joint_stage(tmp_path, coordinates=coordinates)
    bones = build_bone_graph(joints)

    plan = build_skinning_plan(
        mesh_plan,
        bones,
        joints,
        normalized_parts=component_plan.parts,
        component_sources=sources,
        native_variant_set=None,
    )

    influence_ids = {
        influence.bone_id for mesh in plan.weighted_meshes for vertex in mesh.vertices for influence in vertex.influences
    }
    assert all(bone_id.endswith(".xmax") for bone_id in influence_ids)
    assert influence_ids


def _rehash_weighted_mesh(mesh):
    provisional = replace(mesh, weighted_mesh_sha256="")
    return replace(
        provisional,
        weighted_mesh_sha256=jcs_sha256(provisional.content_payload()),
    )


def _rehash_skinning_plan(plan, *, weighted_mesh):
    provisional = replace(plan, weighted_meshes=(weighted_mesh,), plan_sha256="")
    return replace(
        provisional,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def test_skinning_validator_rejects_rehashed_unknown_bones_and_geometry_drift(
    tmp_path: Path,
) -> None:
    component_plan, mesh_plan, sources = _single_part_mesh(
        tmp_path,
        source_tag="face",
        base_tag="face",
        part_id="part/face",
        side=None,
    )
    joints = _joint_stage(tmp_path)
    bones = build_bone_graph(joints)
    plan = build_skinning_plan(
        mesh_plan,
        bones,
        joints,
        normalized_parts=component_plan.parts,
        component_sources=sources,
        native_variant_set=None,
    )

    assert plan.bone_ids == tuple(bone.bone_id for bone in bones.bones)
    mesh = plan.weighted_meshes[0]
    vertex = mesh.vertices[0]
    unknown_influence = replace(vertex.influences[0], bone_id="bone/not-in-graph")
    unknown_vertex = replace(vertex, influences=(unknown_influence,))
    unknown_mesh = _rehash_weighted_mesh(
        replace(
            mesh,
            allowed_bone_ids=(*mesh.allowed_bone_ids, "bone/not-in-graph"),
            vertices=(unknown_vertex, *mesh.vertices[1:]),
        )
    )
    unknown_plan = _rehash_skinning_plan(plan, weighted_mesh=unknown_mesh)

    with pytest.raises(skinning_module.SkinningError):
        validate_skinning_plan(unknown_plan)

    moved_vertex = replace(
        vertex,
        position=(vertex.position[0] + 1 / 256.0, vertex.position[1]),
    )
    moved_mesh = _rehash_weighted_mesh(replace(mesh, vertices=(moved_vertex, *mesh.vertices[1:])))
    moved_plan = _rehash_skinning_plan(plan, weighted_mesh=moved_mesh)

    with pytest.raises(skinning_module.SkinningError):
        validate_skinning_plan(moved_plan)
