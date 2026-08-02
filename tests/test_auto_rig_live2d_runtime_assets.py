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
    build_live2d_runtime_asset_plan,
    encode_live2d_runtime_asset,
    validate_cubism_runtime_json_bytes,
    validate_live2d_runtime_asset_plan,
)
from tests.test_auto_rig_live2d_coordinates import _coordinate_plans


@pytest.fixture(scope="module")
def runtime_fixture(tmp_path_factory: pytest.TempPathFactory):
    root = Path(tmp_path_factory.mktemp("live2d-runtime-assets"))
    rig, symbols, _registry, bindings, coordinates = _coordinate_plans(root)
    artmeshes = build_live2d_artmesh_plan(
        rig, symbols, bindings, coordinates
    )
    keyforms = build_live2d_keyform_plan(
        rig, bindings, coordinates, artmeshes
    )
    animations = build_live2d_animation_plan(
        rig, symbols, bindings, keyforms
    )
    plan = build_live2d_runtime_asset_plan(
        rig, symbols, bindings, artmeshes, animations
    )
    return rig, symbols, bindings, artmeshes, animations, plan


def test_model3_references_exact_contiguous_runtime_inventory(runtime_fixture) -> None:
    rig, symbols, bindings, artmeshes, animations, plan = runtime_fixture
    assert LIVE2D_ARTIFACT_BASENAME == "model"
    assert plan.schema_version == LIVE2D_RUNTIME_ASSET_PLAN_VERSION
    assert plan.model3.relative_path == "model.model3.json"
    assert plan.cdi3.relative_path == "model.cdi3.json"
    assert validate_live2d_runtime_asset_plan(
        plan, rig, symbols, bindings, artmeshes, animations
    ) is plan

    model = plan.model3.payload
    assert model["Version"] == 3
    references = model["FileReferences"]
    assert references["Moc"] == "model.moc3"
    assert references["DisplayInfo"] == "model.cdi3.json"
    assert references["Textures"] == [
        f"textures/page_{index}.png"
        for index in range(artmeshes.texture_page_count)
    ]
    assert list(references["Motions"]) == ["Presets"]
    assert {
        item["File"] for item in references["Motions"]["Presets"]
    } == {asset.relative_path for asset in animations.motion_assets}
    assert {
        item["File"] for item in references["Expressions"]
    } == {asset.relative_path for asset in animations.expression_assets}
    assert all(
        item["FadeInTime"] == 0.0 and item["FadeOutTime"] == 0.0
        for item in references["Motions"]["Presets"]
    )
    assert not ({"Physics", "Pose", "UserData"} & set(references))
    assert "PremultipliedAlpha" not in model

    group_by_name = {group["Name"]: group for group in model["Groups"]}
    blink = sorted(
        parameter.export_name
        for parameter in bindings.parameters
        if parameter.control_id
        in {"control/eye_open.xmin", "control/eye_open.xmax"}
    )
    mouth = next(
        parameter
        for parameter in bindings.parameters
        if parameter.control_id == "control/mouth_open"
    )
    assert group_by_name["EyeBlink"]["Ids"] == blink
    assert group_by_name["LipSync"]["Ids"] == [mouth.export_name]


def test_cdi3_closes_parameter_and_part_id_sets(runtime_fixture) -> None:
    _rig, _symbols, bindings, artmeshes, _animations, plan = runtime_fixture
    display = plan.cdi3.payload
    assert display["Version"] == 3
    assert {item["Id"] for item in display["Parameters"]} == {
        parameter.export_name for parameter in bindings.parameters
    }
    assert {item["Id"] for item in display["Parts"]} == {
        part.export_name for part in artmeshes.parts
    }
    assert display["ParameterGroups"] == []

    for asset in (plan.model3, plan.cdi3):
        encoded = encode_live2d_runtime_asset(asset)
        assert encoded.endswith(b"\n")
        assert b"\n  \"" in encoded
        assert validate_cubism_runtime_json_bytes(encoded, asset.payload) == encoded
        minified = json.dumps(
            asset.payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
        with pytest.raises(ValueError, match="runtime JSON encoding"):
            validate_cubism_runtime_json_bytes(minified, asset.payload)


def test_runtime_asset_validator_rejects_stale_motion_reference(
    runtime_fixture,
) -> None:
    rig, symbols, bindings, artmeshes, animations, plan = runtime_fixture
    payload = dict(plan.model3.payload)
    references = dict(payload["FileReferences"])
    motions = {
        name: [dict(item) for item in items]
        for name, items in references["Motions"].items()
    }
    motions["Presets"][0]["File"] = "motions/stale.motion3.json"
    references["Motions"] = motions
    payload["FileReferences"] = references
    broken_model = replace(plan.model3, payload=payload)
    broken = replace(plan, model3=broken_model)

    with pytest.raises(ValueError, match="digest|differs|reference"):
        validate_live2d_runtime_asset_plan(
            broken, rig, symbols, bindings, artmeshes, animations
        )
