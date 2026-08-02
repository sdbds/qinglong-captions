from __future__ import annotations

from dataclasses import replace

import pytest

from module.auto_rig.export.spine.bind_plan import (
    SPINE_BIND_PLAN_VERSION,
    SpineBindPlanError,
    build_spine_bind_plan,
    reconstruct_spine_point,
    validate_spine_bind_plan,
)
from module.auto_rig.export.spine.coordinates import build_spine_coordinate_plan


def _coordinate_plan():
    return build_spine_coordinate_plan(
        {"width": 400, "height": 300, "origin": "top_left", "y_axis": "down"},
        input_fingerprint="sha256:" + "4" * 64,
    )


def _bones():
    return (
        {
            "bone_id": "bone/root",
            "parent_id": None,
            "role": "synthetic_root",
            "head": None,
            "tail": None,
            "length": 0.0,
        },
        {
            "bone_id": "bone/a",
            "parent_id": "bone/root",
            "role": "torso",
            "head": [200.0, 150.0],
            "tail": [240.0, 150.0],
            "length": 40.0,
        },
        {
            "bone_id": "bone/b",
            "parent_id": "bone/a",
            "role": "forearm",
            "head": [240.0, 150.0],
            "tail": [240.0, 110.0],
            "length": 40.0,
        },
        {
            "bone_id": "bone/c",
            "parent_id": "bone/b",
            "role": "hand",
            "head": [240.0, 110.0],
            "tail": [220.0, 90.0],
            "length": 28.284271247461902,
        },
    )


def _meshes():
    return (
        {
            "mesh_id": "mesh/test",
            "part_id": "part/test",
            "component_id": "component/test",
            "vertices": [
                {
                    "position": [235.0, 120.0],
                    "uv": [0.7, 0.42857142857142855],
                    "influences": [
                        {"bone_id": "bone/a", "weight": 0.25},
                        {"bone_id": "bone/b", "weight": 0.75},
                    ],
                },
                {
                    "position": [225.0, 100.0],
                    "uv": [0.5, 0.14285714285714285],
                    "influences": [
                        {"bone_id": "bone/b", "weight": 0.5},
                        {"bone_id": "bone/c", "weight": 0.5},
                    ],
                },
                {
                    "position": [210.0, 130.0],
                    "uv": [0.2, 0.5714285714285714],
                    "influences": [
                        {"bone_id": "bone/a", "weight": 1.0},
                    ],
                },
            ],
            "triangles": [0, 1, 2],
        },
    )


def test_spine_bind_plan_reconstructs_rotated_hierarchy_and_each_influence() -> None:
    plan = build_spine_bind_plan(_bones(), _meshes(), _coordinate_plan())

    assert plan.schema_version == SPINE_BIND_PLAN_VERSION
    assert [bone.bone_id for bone in plan.bones] == [
        "bone/root",
        "bone/a",
        "bone/b",
        "bone/c",
    ]
    root, a, b, c = plan.bones
    assert (root.x, root.y, root.rotation, root.length) == (0.0, 0.0, 0.0, 0.0)
    assert (a.x, a.y, a.rotation, a.length) == pytest.approx((0.0, 0.0, 0.0, 40.0))
    assert (b.x, b.y, b.rotation, b.length) == pytest.approx((40.0, 0.0, 90.0, 40.0))
    assert (c.x, c.y, c.rotation, c.length) == pytest.approx((40.0, 0.0, 45.0, 28.284271247461902))
    assert plan.meshes[0].slot_bone_id == "bone/a"
    assert plan.maximum_reconstruction_error <= 1e-9

    for vertex in plan.meshes[0].vertices:
        for influence in vertex.influences:
            point = reconstruct_spine_point(plan, influence.bone_id, (influence.x, influence.y))
            assert point == pytest.approx(vertex.spine_position, abs=1e-9)
    assert validate_spine_bind_plan(plan, _bones(), _meshes(), _coordinate_plan()) is plan


def test_spine_bind_plan_rejects_unknown_bone_and_tampered_local_point() -> None:
    meshes = list(_meshes())
    meshes[0] = {**meshes[0], "vertices": [dict(meshes[0]["vertices"][0])]}
    meshes[0]["vertices"][0]["influences"] = [
        {"bone_id": "bone/missing", "weight": 1.0}
    ]
    with pytest.raises(SpineBindPlanError, match="unknown bone"):
        build_spine_bind_plan(_bones(), tuple(meshes), _coordinate_plan())

    plan = build_spine_bind_plan(_bones(), _meshes(), _coordinate_plan())
    vertex = plan.meshes[0].vertices[0]
    changed_influence = replace(vertex.influences[0], x=vertex.influences[0].x + 1.0)
    changed_vertex = replace(vertex, influences=(changed_influence, *vertex.influences[1:]))
    changed_mesh = replace(plan.meshes[0], vertices=(changed_vertex, *plan.meshes[0].vertices[1:]))
    tampered = replace(plan, meshes=(changed_mesh,), plan_sha256=plan.plan_sha256)
    with pytest.raises(SpineBindPlanError, match="digest|reconstruction"):
        validate_spine_bind_plan(tampered, _bones(), _meshes(), _coordinate_plan())
