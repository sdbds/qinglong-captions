from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from module.auto_rig.export.live2d.animations import (
    LIVE2D_ANIMATION_PLAN_VERSION,
    build_live2d_animation_plan,
    encode_live2d_animation_asset,
    validate_live2d_animation_plan,
)
from module.auto_rig.export.live2d.artmesh import build_live2d_artmesh_plan
from module.auto_rig.export.live2d.keyforms import build_live2d_keyform_plan
from tests.test_auto_rig_live2d_coordinates import _coordinate_plans


@pytest.fixture(scope="module")
def animation_fixture(tmp_path_factory: pytest.TempPathFactory):
    root = Path(tmp_path_factory.mktemp("live2d-animations"))
    rig, symbols, _registry, bindings, coordinates = _coordinate_plans(root)
    artmeshes = build_live2d_artmesh_plan(rig, symbols, bindings, coordinates)
    keyforms = build_live2d_keyform_plan(rig, bindings, coordinates, artmeshes)
    plan = build_live2d_animation_plan(rig, symbols, bindings, keyforms)
    return rig, symbols, bindings, keyforms, plan


def test_motion_assets_preserve_exact_linear_control_curves(animation_fixture) -> None:
    rig, symbols, bindings, keyforms, plan = animation_fixture
    assert plan.schema_version == LIVE2D_ANIMATION_PLAN_VERSION
    assert validate_live2d_animation_plan(plan, rig, symbols, bindings, keyforms) is plan
    assert {record.preset_id for record in plan.motion_assets} == {
        "arm_sway",
        "blink",
        "body_sway",
        "breath",
        "head_nod",
        "head_shake",
        "idle",
        "leg_sway",
        "talk",
    }
    nod = next(record for record in plan.motion_assets if record.preset_id == "head_nod")
    assert nod.relative_path == "motions/head_nod.motion3.json"
    payload = nod.payload
    assert payload["Version"] == 3
    assert payload["Meta"]["Duration"] == 1.0
    assert payload["Meta"]["Fps"] == 30.0
    assert payload["Meta"]["Loop"] is False
    assert payload["Meta"]["FadeInTime"] == 0.0
    assert payload["Meta"]["FadeOutTime"] == 0.0
    assert payload["Meta"]["CurveCount"] == len(payload["Curves"])
    curve = payload["Curves"][0]
    assert curve["Target"] == "Parameter"
    assert curve["Segments"] == [0.0, 0.0, 0, 0.5, 15.0, 0, 1.0, 0.0]
    assert "FadeInTime" not in curve and "FadeOutTime" not in curve

    blink = next(record for record in plan.motion_assets if record.preset_id == "blink")
    assert blink.payload["Meta"]["Duration"] == 0.8
    assert all(
        curve["Segments"]
        == [
            0.0,
            1.0,
            0,
            0.2,
            0.0,
            0,
            0.3,
            0.0,
            0,
            16 / 30,
            1.0,
            0,
            0.8,
            1.0,
        ]
        for curve in blink.payload["Curves"]
    )

    encoded = encode_live2d_animation_asset(nod)
    assert encoded.endswith(b"\n")
    assert b'\n  "Curves"' in encoded
    assert encoded == encode_live2d_animation_asset(nod)


def test_expression_assets_multiply_eye_open_and_overwrite_other_targets(
    animation_fixture,
) -> None:
    _rig, _symbols, bindings, _keyforms, plan = animation_fixture
    eye_ids = {
        parameter.export_name
        for parameter in bindings.parameters
        if parameter.control_id in {"control/eye_open.xmin", "control/eye_open.xmax"}
    }
    assert [record.preset_id for record in plan.expression_assets] == [
        "surprised",
        "wink_screen_left",
        "wink_screen_right",
    ]
    expression = plan.expression_assets[0]
    assert expression.relative_path == "expressions/surprised.exp3.json"
    assert expression.payload["Type"] == "Live2D Expression"
    assert expression.payload["FadeInTime"] == 0.0
    assert expression.payload["FadeOutTime"] == 0.0
    assert expression.payload["Parameters"]
    for asset in plan.expression_assets:
        for parameter in asset.payload["Parameters"]:
            expected_blend = "Multiply" if parameter["Id"] in eye_ids else "Overwrite"
            assert parameter["Blend"] == expected_blend
            assert "Time" not in parameter and "Segments" not in parameter


def test_only_explicit_blink_motion_drives_eye_open_parameters(
    animation_fixture,
) -> None:
    _rig, _symbols, bindings, _keyforms, plan = animation_fixture
    eye_ids = {
        parameter.export_name
        for parameter in bindings.parameters
        if parameter.control_id in {"control/eye_open.xmin", "control/eye_open.xmax"}
    }
    motion_by_id = {asset.preset_id: asset for asset in plan.motion_assets}

    assert set(motion_by_id["blink"].parameter_ids) == eye_ids
    for preset_id, asset in motion_by_id.items():
        if preset_id != "blink":
            assert eye_ids.isdisjoint(asset.parameter_ids), preset_id


def test_screen_wink_expressions_close_exactly_one_eye(
    animation_fixture,
) -> None:
    _rig, _symbols, bindings, _keyforms, plan = animation_fixture
    eye_id_by_control = {
        parameter.control_id: parameter.export_name
        for parameter in bindings.parameters
        if parameter.control_id in {"control/eye_open.xmin", "control/eye_open.xmax"}
    }
    expression_by_id = {asset.preset_id: asset for asset in plan.expression_assets}

    expected = {
        "wink_screen_left": {
            eye_id_by_control["control/eye_open.xmin"]: 0.0,
            eye_id_by_control["control/eye_open.xmax"]: 1.0,
        },
        "wink_screen_right": {
            eye_id_by_control["control/eye_open.xmin"]: 1.0,
            eye_id_by_control["control/eye_open.xmax"]: 0.0,
        },
    }
    for preset_id, values in expected.items():
        parameters = expression_by_id[preset_id].payload["Parameters"]
        assert {parameter["Id"]: parameter["Value"] for parameter in parameters} == values
        assert {parameter["Id"]: parameter["Blend"] for parameter in parameters} == {
            parameter_id: "Multiply" for parameter_id in values
        }


def test_animation_validator_rejects_duplicate_parameter_curve(
    animation_fixture,
) -> None:
    rig, symbols, bindings, keyforms, plan = animation_fixture
    first = plan.motion_assets[0]
    payload = dict(first.payload)
    payload["Curves"] = [*payload["Curves"], dict(payload["Curves"][0])]
    broken_asset = replace(first, payload=payload)
    broken = replace(plan, motion_assets=(broken_asset, *plan.motion_assets[1:]))

    with pytest.raises(ValueError, match="duplicate|digest|differs"):
        validate_live2d_animation_plan(broken, rig, symbols, bindings, keyforms)


def test_animation_validator_rejects_overwrite_for_eye_expression(
    animation_fixture,
) -> None:
    rig, symbols, bindings, keyforms, plan = animation_fixture
    wink = next(asset for asset in plan.expression_assets if asset.preset_id == "wink_screen_left")
    payload = dict(wink.payload)
    parameters = [dict(parameter) for parameter in payload["Parameters"]]
    parameters[0]["Blend"] = "Overwrite"
    payload["Parameters"] = parameters
    broken_asset = replace(wink, payload=payload)
    broken = replace(
        plan,
        expression_assets=tuple(broken_asset if asset is wink else asset for asset in plan.expression_assets),
    )

    with pytest.raises(ValueError, match="per-control blend"):
        validate_live2d_animation_plan(broken, rig, symbols, bindings, keyforms)


def test_animation_validator_rejects_blink_without_open_recovery_hold(
    animation_fixture,
) -> None:
    rig, symbols, bindings, keyforms, plan = animation_fixture
    blink = next(asset for asset in plan.motion_assets if asset.preset_id == "blink")
    payload = dict(blink.payload)
    curves = [dict(curve) for curve in payload["Curves"]]
    curves[0]["Segments"] = list(curves[0]["Segments"])
    curves[0]["Segments"][-5] = 22 / 30
    payload["Curves"] = curves
    broken_asset = replace(blink, payload=payload)
    broken = replace(
        plan,
        motion_assets=tuple(broken_asset if asset is blink else asset for asset in plan.motion_assets),
    )

    with pytest.raises(ValueError, match="recovery hold"):
        validate_live2d_animation_plan(broken, rig, symbols, bindings, keyforms)
