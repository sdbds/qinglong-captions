from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from module.auto_rig.export.spine.animations import (
    SPINE_ANIMATION_PLAN_VERSION,
    SpineAnimationError,
    build_spine_animation_plan,
    validate_spine_animation_plan,
)
from module.auto_rig.export.spine.atlas import build_spine_atlas_plan
from module.auto_rig.export.spine.bind_plan import build_spine_bind_plan
from module.auto_rig.export.spine.coordinates import build_spine_coordinate_plan
from module.auto_rig.export.spine.model import build_spine_document
from module.auto_rig.export.spine.symbols import build_spine_symbol_view, require_spine_symbol
from tests.test_auto_rig_rig_document import _build


@pytest.fixture(scope="module")
def spine_fixture(tmp_path_factory: pytest.TempPathFactory):
    root = Path(tmp_path_factory.mktemp("spine-animations"))
    *_, rig = _build(root)
    payload = rig.to_dict()
    coordinates = build_spine_coordinate_plan(
        payload["canvas"], input_fingerprint=payload["input_fingerprint"]
    )
    bind = build_spine_bind_plan(payload["bones"], payload["meshes"], coordinates)
    symbols = build_spine_symbol_view(payload["export_symbols"])
    atlas = build_spine_atlas_plan(
        payload["texture_pages"], payload["parts"], symbols
    )
    setup = build_spine_document(rig, coordinates, bind, symbols, atlas)
    plan = build_spine_animation_plan(rig, coordinates, bind, symbols, setup)
    return rig, coordinates, bind, symbols, setup, plan


def _walk_keyframes(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_keyframes(child)
    elif isinstance(value, list):
        if value and all(isinstance(item, dict) and "time" in item for item in value):
            yield value
        for child in value:
            yield from _walk_keyframes(child)


def test_spine_motion_plan_encodes_selected_linear_curves_and_deforms(
    spine_fixture,
) -> None:
    rig, coordinates, bind, symbols, setup, plan = spine_fixture
    assert plan.schema_version == SPINE_ANIMATION_PLAN_VERSION
    payload = rig.to_dict()
    spine_set = next(
        item
        for item in payload["format_plans"]["preset_set_plans"]
        if item["format_id"] == "spine_4_2"
    )
    supported_names = {
        item["artifact_export_name"]
        for item in spine_set["decisions"]
        if item["status"] == "supported"
    }
    assert set(plan.animations) == supported_names
    assert {"breath", "head_nod", "head_shake", "idle", "wave_xmin", "wave_xmax"} <= set(plan.animations)
    assert "attachments" in plan.animations["breath"]
    assert "attachments" in plan.animations["talk"]
    assert "slots" in plan.animations["blink"]

    head_name = require_spine_symbol(
        symbols, kind="spine_bone", source_internal_ids=("bone/head",)
    ).export_name
    nod_keys = plan.animations["head_nod"]["bones"][head_name]["rotate"]
    assert [key["time"] for key in nod_keys] == [0.0, 0.5, 1.0]
    assert nod_keys[1]["value"] < 0
    for keyframes in _walk_keyframes(plan.animations):
        assert all("curve" not in key for key in keyframes)
    assert validate_spine_animation_plan(
        plan, rig, coordinates, bind, symbols, setup
    ) is plan


def test_spine_expression_hold_has_exact_two_equal_keys(spine_fixture) -> None:
    _rig, _coordinates, _bind, _symbols, _setup, plan = spine_fixture
    for expression_name in ("happy", "sad", "surprised"):
        animation = plan.animations[expression_name]
        keyframe_groups = tuple(_walk_keyframes(animation))
        assert keyframe_groups
        for keys in keyframe_groups:
            assert [key["time"] for key in keys] == [0.0, 1.0 / 30.0]
            first = {key: value for key, value in keys[0].items() if key != "time"}
            second = {key: value for key, value in keys[1].items() if key != "time"}
            assert first == second


def test_spine_animation_validator_rejects_target_curve_field(spine_fixture) -> None:
    rig, coordinates, bind, symbols, setup, plan = spine_fixture
    changed = {name: dict(animation) for name, animation in plan.animations.items()}
    changed["idle"] = {
        **changed["idle"],
        "bones": {
            **changed["idle"]["bones"],
            next(iter(changed["idle"]["bones"])): {
                "rotate": [{"time": 0.0, "value": 0.0, "curve": "linear"}]
            },
        },
    }
    tampered = replace(plan, animations=changed)
    with pytest.raises(SpineAnimationError, match="curve|digest"):
        validate_spine_animation_plan(
            tampered, rig, coordinates, bind, symbols, setup
        )
