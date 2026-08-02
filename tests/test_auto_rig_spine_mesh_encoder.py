from __future__ import annotations

from copy import deepcopy

import pytest

from module.auto_rig.export.spine.bind_plan import build_spine_bind_plan
from module.auto_rig.export.spine.coordinates import build_spine_coordinate_plan
from module.auto_rig.export.spine.mesh_encoder import (
    SpineMeshEncodingError,
    encode_spine_mesh,
    parse_weighted_vertices,
    validate_spine_encoded_mesh,
)
from tests.test_auto_rig_spine_bind_plan import _bones, _meshes


def _plan(meshes=None):
    coordinate = build_spine_coordinate_plan(
        {"width": 400, "height": 300, "origin": "top_left", "y_axis": "down"},
        input_fingerprint="sha256:" + "5" * 64,
    )
    source_meshes = _meshes() if meshes is None else meshes
    return source_meshes, build_spine_bind_plan(_bones(), source_meshes, coordinate)


def test_weighted_mesh_encoder_uses_official_flat_variable_length_vertices() -> None:
    meshes, bind = _plan()
    encoded = encode_spine_mesh(
        meshes[0],
        bind.meshes[0],
        source_xyxy=(200.0, 90.0, 250.0, 160.0),
        attachment_key_name="test",
        attachment_object_name="test_object",
        atlas_region_name="test_region",
    )

    assert encoded.weighted is True
    assert encoded.payload["type"] == "mesh"
    assert encoded.payload["name"] == "test_object"
    assert encoded.payload["path"] == "test_region"
    assert encoded.payload["triangles"] == [0, 1, 2]
    assert all(not isinstance(value, list) for value in encoded.payload["vertices"])
    parsed = parse_weighted_vertices(
        encoded.payload["vertices"], vertex_count=len(meshes[0]["vertices"])
    )
    assert [len(vertex) for vertex in parsed] == [2, 2, 1]
    assert sum(influence[3] for influence in parsed[0]) == pytest.approx(1.0)
    assert validate_spine_encoded_mesh(encoded, meshes[0], bind.meshes[0]) is encoded


def test_rigid_mesh_encoder_uses_plain_slot_local_xy_pairs() -> None:
    source = deepcopy(_meshes()[0])
    for vertex in source["vertices"]:
        vertex["influences"] = [{"bone_id": "bone/a", "weight": 1.0}]
    meshes, bind = _plan((source,))
    assert bind.meshes[0].slot_bone_id == "bone/a"

    encoded = encode_spine_mesh(
        meshes[0],
        bind.meshes[0],
        source_xyxy=(200.0, 90.0, 250.0, 160.0),
        attachment_key_name="same",
        attachment_object_name="same",
        atlas_region_name="same",
    )

    assert encoded.weighted is False
    assert len(encoded.payload["vertices"]) == 2 * len(source["vertices"])
    assert "name" not in encoded.payload
    assert "path" not in encoded.payload


def test_mesh_encoder_rejects_nested_or_out_of_range_topology() -> None:
    meshes, bind = _plan()
    changed = deepcopy(meshes[0])
    changed["triangles"] = [[0, 1, 2]]
    with pytest.raises(SpineMeshEncodingError, match="triangles"):
        encode_spine_mesh(
            changed,
            bind.meshes[0],
            source_xyxy=(200.0, 90.0, 250.0, 160.0),
            attachment_key_name="test",
            attachment_object_name="test",
            atlas_region_name="test",
        )
