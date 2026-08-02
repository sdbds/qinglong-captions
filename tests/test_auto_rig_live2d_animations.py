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
    artmeshes = build_live2d_artmesh_plan(
        rig, symbols, bindings, coordinates
    )
    keyforms = build_live2d_keyform_plan(
        rig, bindings, coordinates, artmeshes
    )
    plan = build_live2d_animation_plan(rig, symbols, bindings, keyforms)
    return rig, symbols, bindings, keyforms, plan


def test_motion_assets_preserve_exact_linear_control_curves(animation_fixture) -> None:
    rig, symbols, bindings, keyforms, plan = animation_fixture
    assert plan.schema_version == LIVE2D_ANIMATION_PLAN_VERSION
    assert validate_live2d_animation_plan(
        plan, rig, symbols, bindings, keyforms
    ) is plan
    assert {record.preset_id for record in plan.motion_assets} == {
        "blink",
        "body_sway",
        "breath",
        "head_nod",
        "head_shake",
        "idle",
        "talk",
    }
    nod = next(record for record in plan.motion_assets if record.preset_id == "head_nod")
    assert nod.relative_path == "motions/head_nod.motion3.json"
    payload = nod.payload
    assert payload["Version"] == 3
    assert payload["Meta"]["Duration"] == 1.0
    assert payload["Meta"]["Fps"] == 30.0
    assert payload["Meta"]["Loop"] is False
    assert payload["Meta"]["CurveCount"] == len(payload["Curves"])
    curve = payload["Curves"][0]
    assert curve["Target"] == "Parameter"
    assert curve["Segments"] == [0.0, 0.0, 0, 0.5, 15.0, 0, 1.0, 0.0]
    assert "FadeInTime" not in curve and "FadeOutTime" not in curve

    encoded = encode_live2d_animation_asset(nod)
    assert encoded.endswith(b"\n")
    assert b"\n  \"Curves\"" in encoded
    assert encoded == encode_live2d_animation_asset(nod)


def test_expression_assets_are_full_weight_overwrite_without_time_axis(
    animation_fixture,
) -> None:
    _rig, _symbols, _bindings, _keyforms, plan = animation_fixture
    assert [record.preset_id for record in plan.expression_assets] == ["surprised"]
    expression = plan.expression_assets[0]
    assert expression.relative_path == "expressions/surprised.exp3.json"
    assert expression.payload["Type"] == "Live2D Expression"
    assert expression.payload["FadeInTime"] == 0.0
    assert expression.payload["FadeOutTime"] == 0.0
    assert expression.payload["Parameters"]
    assert all(
        parameter["Blend"] == "Overwrite"
        for parameter in expression.payload["Parameters"]
    )
    assert all(
        "Time" not in parameter and "Segments" not in parameter
        for parameter in expression.payload["Parameters"]
    )


def test_animation_validator_rejects_duplicate_parameter_curve(
    animation_fixture,
) -> None:
    rig, symbols, bindings, keyforms, plan = animation_fixture
    first = plan.motion_assets[0]
    payload = dict(first.payload)
    payload["Curves"] = [*payload["Curves"], dict(payload["Curves"][0])]
    broken_asset = replace(first, payload=payload)
    broken = replace(
        plan, motion_assets=(broken_asset, *plan.motion_assets[1:])
    )

    with pytest.raises(ValueError, match="duplicate|digest|differs"):
        validate_live2d_animation_plan(
            broken, rig, symbols, bindings, keyforms
        )
