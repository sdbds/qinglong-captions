from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from module.auto_rig.export.live2d.animations import build_live2d_animation_plan
from module.auto_rig.export.live2d.artmesh import build_live2d_artmesh_plan
from module.auto_rig.export.live2d.keyforms import build_live2d_keyform_plan
from module.auto_rig.export.live2d.runtime_assets import (
    LIVE2D_ARTIFACT_BASENAME,
    LIVE2D_RUNTIME_ASSET_PLAN_VERSION,
    _hit_area_entries,
    build_live2d_runtime_asset_plan,
    encode_live2d_runtime_asset,
    validate_cubism_runtime_json_bytes,
    validate_live2d_runtime_asset_plan,
)
from module.auto_rig.jcs import jcs_bytes
from module.auto_rig.rig_document import RigDocument
from tests.test_auto_rig_live2d_coordinates import _coordinate_plans


@pytest.fixture(scope="module")
def runtime_fixture(tmp_path_factory: pytest.TempPathFactory):
    root = Path(tmp_path_factory.mktemp("live2d-runtime-assets"))
    rig, symbols, _registry, bindings, coordinates = _coordinate_plans(root)
    artmeshes = build_live2d_artmesh_plan(rig, symbols, bindings, coordinates)
    keyforms = build_live2d_keyform_plan(rig, bindings, coordinates, artmeshes)
    animations = build_live2d_animation_plan(rig, symbols, bindings, keyforms)
    plan = build_live2d_runtime_asset_plan(rig, symbols, bindings, artmeshes, animations)
    return rig, symbols, bindings, artmeshes, animations, plan


def test_model3_references_exact_contiguous_runtime_inventory(runtime_fixture) -> None:
    rig, symbols, bindings, artmeshes, animations, plan = runtime_fixture
    assert LIVE2D_ARTIFACT_BASENAME == "model"
    assert plan.schema_version == LIVE2D_RUNTIME_ASSET_PLAN_VERSION
    assert plan.model3.relative_path == "model.model3.json"
    assert plan.cdi3.relative_path == "model.cdi3.json"
    assert validate_live2d_runtime_asset_plan(plan, rig, symbols, bindings, artmeshes, animations) is plan

    model = plan.model3.payload
    assert model["Version"] == 3
    references = model["FileReferences"]
    assert references["Moc"] == "model.moc3"
    assert references["DisplayInfo"] == "model.cdi3.json"
    assert references["Textures"] == [f"textures/page_{index}.png" for index in range(artmeshes.texture_page_count)]
    assert list(references["Motions"]) == ["Presets"]
    assert {item["File"] for item in references["Motions"]["Presets"]} == {
        asset.relative_path for asset in animations.motion_assets
    }
    motion_by_path = {item["File"]: item for item in references["Motions"]["Presets"]}
    for asset in animations.motion_assets:
        entry = motion_by_path[asset.relative_path]
        assert entry["Name"] == asset.artifact_export_name
        inlined_entry = {
            **entry,
            "File": "data:application/json;base64,ZXhhbXBsZQ==",
        }
        assert inlined_entry.get("Name") == asset.artifact_export_name
    assert {item["File"] for item in references["Expressions"]} == {asset.relative_path for asset in animations.expression_assets}
    assert all(item["FadeInTime"] == 0.0 and item["FadeOutTime"] == 0.0 for item in references["Motions"]["Presets"])
    assert not ({"Physics", "Pose", "UserData"} & set(references))
    assert "PremultipliedAlpha" not in model

    hit_areas = {item["Name"]: item["Id"] for item in model["HitAreas"]}
    assert set(hit_areas) == {
        "ArmScreenLeft",
        "ArmScreenRight",
        "Body",
        "Head",
        "LegScreenLeft",
        "LegScreenRight",
    }
    artmesh_by_export_name = {artmesh.export_name: artmesh for artmesh in artmeshes.artmeshes}
    part_by_id = {part["part_id"]: part for part in rig.to_dict()["parts"]}
    assert part_by_id[artmesh_by_export_name[hit_areas["Head"]].part_id]["base_tag"] == "face"
    assert part_by_id[artmesh_by_export_name[hit_areas["Body"]].part_id]["base_tag"] in {"topwear", "bottomwear"}
    for name, side in (
        ("ArmScreenLeft", "xmin"),
        ("ArmScreenRight", "xmax"),
    ):
        part = part_by_id[artmesh_by_export_name[hit_areas[name]].part_id]
        assert part["base_tag"] == "handwear"
        assert part["side"] == side
    for name, side in (
        ("LegScreenLeft", "xmin"),
        ("LegScreenRight", "xmax"),
    ):
        part = part_by_id[artmesh_by_export_name[hit_areas[name]].part_id]
        assert part["base_tag"] in {"footwear", "legwear"}
        if part["base_tag"] == "footwear":
            assert part["side"] == side

    group_by_name = {group["Name"]: group for group in model["Groups"]}
    mouth = next(parameter for parameter in bindings.parameters if parameter.control_id == "control/mouth_open")
    # Blink is an explicit motion. Advertising these parameters as EyeBlink
    # asks runtimes to inject random blinking into every other motion and can
    # overwrite the two one-eye expressions.
    assert "EyeBlink" not in group_by_name
    assert group_by_name["LipSync"]["Ids"] == [mouth.export_name]

    expression_by_name = {item["Name"]: item for item in references["Expressions"]}
    assert expression_by_name["wink_screen_left"]["File"] == ("expressions/wink_screen_left.exp3.json")
    assert expression_by_name["wink_screen_right"]["File"] == ("expressions/wink_screen_right.exp3.json")


def test_cdi3_closes_parameter_and_part_id_sets(runtime_fixture) -> None:
    _rig, _symbols, bindings, artmeshes, _animations, plan = runtime_fixture
    display = plan.cdi3.payload
    assert display["Version"] == 3
    assert {item["Id"] for item in display["Parameters"]} == {parameter.export_name for parameter in bindings.parameters}
    assert {item["Id"] for item in display["Parts"]} == {part.export_name for part in artmeshes.parts}
    assert display["ParameterGroups"] == []

    for asset in (plan.model3, plan.cdi3):
        encoded = encode_live2d_runtime_asset(asset)
        assert encoded.endswith(b"\n")
        assert b'\n  "' in encoded
        assert validate_cubism_runtime_json_bytes(encoded, asset.payload) == encoded
        minified = json.dumps(asset.payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("ascii")
        with pytest.raises(ValueError, match="runtime JSON encoding"):
            validate_cubism_runtime_json_bytes(minified, asset.payload)


def test_model3_uses_one_honest_combined_hit_area_for_a_connected_merged_limb(
    tmp_path: Path,
) -> None:
    rig, symbols, _registry, bindings, coordinates = _coordinate_plans(
        tmp_path,
        limb_mode="merged",
    )
    artmeshes = build_live2d_artmesh_plan(rig, symbols, bindings, coordinates)
    keyforms = build_live2d_keyform_plan(rig, bindings, coordinates, artmeshes)
    animations = build_live2d_animation_plan(rig, symbols, bindings, keyforms)
    plan = build_live2d_runtime_asset_plan(
        rig,
        symbols,
        bindings,
        artmeshes,
        animations,
    )

    hit_areas = {item["Name"]: item["Id"] for item in plan.model3.payload["HitAreas"]}
    assert "Arms" in hit_areas
    assert "ArmScreenLeft" not in hit_areas
    assert "ArmScreenRight" not in hit_areas


def test_split_limb_hit_areas_never_fall_back_to_a_false_combined_label(
    tmp_path: Path,
) -> None:
    rig, symbols, _registry, bindings, coordinates = _coordinate_plans(tmp_path)
    artmeshes = build_live2d_artmesh_plan(rig, symbols, bindings, coordinates)
    payload = rig.to_dict()
    for joint in payload["joints"]:
        if any(str(joint["joint_id"]).startswith(f"joint/{stem}.") for stem in ("shoulder", "elbow", "wrist")):
            joint["x"] = 0.0
            joint["y"] = 0.0
    displaced = RigDocument(jcs_bytes(payload))

    hit_areas = {item["Name"]: item["Id"] for item in _hit_area_entries(displaced, artmeshes)}

    assert "Arms" not in hit_areas
    assert {"ArmScreenLeft", "ArmScreenRight"} <= set(hit_areas)


def test_runtime_asset_validator_rejects_stale_motion_reference(
    runtime_fixture,
) -> None:
    rig, symbols, bindings, artmeshes, animations, plan = runtime_fixture
    payload = dict(plan.model3.payload)
    references = dict(payload["FileReferences"])
    motions = {name: [dict(item) for item in items] for name, items in references["Motions"].items()}
    motions["Presets"][0]["File"] = "motions/stale.motion3.json"
    references["Motions"] = motions
    payload["FileReferences"] = references
    broken_model = replace(plan.model3, payload=payload)
    broken = replace(plan, model3=broken_model)

    with pytest.raises(ValueError, match="digest|differs|reference"):
        validate_live2d_runtime_asset_plan(broken, rig, symbols, bindings, artmeshes, animations)
