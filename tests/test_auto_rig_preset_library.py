from __future__ import annotations

from dataclasses import replace

import pytest

from module.auto_rig.control_registry import build_control_registry_plan
from module.auto_rig.jcs import jcs_sha256
from module.auto_rig.preset_library import (
    EXPRESSION_PRESET_SCHEMA_VERSION,
    MOTION_CLIP_SCHEMA_VERSION,
    MOTION_RUNTIME_APPLICATION_VERSION,
    OPTIONAL_PRESET_SELECTION_VERSION,
    PRESET_LIBRARY_VERSION,
    PresetLibraryError,
    build_preset_library_plan,
    validate_preset_library_plan,
)

EXPECTED_CLIPS = {
    "arm_sway": (
        90,
        True,
        {
            "control/arm_sway": (
                (0, 0.0),
                (22, 1.0),
                (45, 0.0),
                (67, -1.0),
                (90, 0.0),
            )
        },
    ),
    "idle": (120, True, {"control/idle": ((0, 0.0), (30, 1.0), (60, 0.0), (90, -1.0), (120, 0.0))}),
    "leg_sway": (
        120,
        True,
        {
            "control/leg_sway": (
                (0, 0.0),
                (30, 1.0),
                (60, 0.0),
                (90, -1.0),
                (120, 0.0),
            )
        },
    ),
    "breath": (90, True, {"control/breath": ((0, 0.0), (45, 1.0), (90, 0.0))}),
    "head_nod": (30, False, {"control/head_nod": ((0, 0.0), (15, 15.0), (30, 0.0))}),
    "head_shake": (45, False, {"control/head_shake": ((0, 0.0), (10, -20.0), (25, 20.0), (45, 0.0))}),
    "body_sway": (120, True, {"control/body_sway": ((0, 0.0), (30, 5.0), (60, 0.0), (90, -5.0), (120, 0.0))}),
    "blink": (
        24,
        False,
        {
            "control/eye_open.xmax": (
                (0, 1.0),
                (6, 0.0),
                (9, 0.0),
                (16, 1.0),
                (24, 1.0),
            ),
            "control/eye_open.xmin": (
                (0, 1.0),
                (6, 0.0),
                (9, 0.0),
                (16, 1.0),
                (24, 1.0),
            ),
        },
    ),
    "talk": (30, True, {"control/mouth_open": ((0, 0.0), (10, 1.0), (20, 0.35), (30, 0.0))}),
    "wave.xmax": (
        60,
        False,
        {
            "control/wave_lift.xmax": ((0, 0.0), (15, 1.0), (45, 1.0), (60, 0.0)),
            "control/wave_osc.xmax": ((0, 0.0), (15, 0.0), (25, 1.0), (35, -1.0), (45, 1.0), (60, 0.0)),
        },
    ),
    "wave.xmin": (
        60,
        False,
        {
            "control/wave_lift.xmin": ((0, 0.0), (15, 1.0), (45, 1.0), (60, 0.0)),
            "control/wave_osc.xmin": ((0, 0.0), (15, 0.0), (25, 1.0), (35, -1.0), (45, 1.0), (60, 0.0)),
        },
    ),
}
EXPECTED_EXPRESSIONS = {
    "happy": {
        "control/brow_y.xmax": 0.15,
        "control/brow_y.xmin": 0.15,
        "control/mouth_form": 0.7,
    },
    "sad": {
        "control/brow_y.xmax": 0.25,
        "control/brow_y.xmin": 0.25,
        "control/mouth_form": -0.7,
    },
    "surprised": {
        "control/brow_y.xmax": 0.8,
        "control/brow_y.xmin": 0.8,
        "control/mouth_open": 0.8,
    },
    "unimpressed": {
        "control/brow_y.xmax": -0.35,
        "control/brow_y.xmin": -0.35,
        "control/mouth_form": -0.15,
    },
    "wink_screen_left": {
        "control/eye_open.xmax": 1.0,
        "control/eye_open.xmin": 0.0,
    },
    "wink_screen_right": {
        "control/eye_open.xmax": 0.0,
        "control/eye_open.xmin": 1.0,
    },
}
EXPECTED_OPTIONAL_PRIORITY = (
    "body_sway",
    "arm_sway",
    "leg_sway",
    "blink",
    "talk",
    "surprised",
    "happy",
    "sad",
    "unimpressed",
    "wink_screen_left",
    "wink_screen_right",
    "wave.xmin",
    "wave.xmax",
)


def _rehash(plan, **changes):
    provisional = replace(plan, **changes, plan_sha256="")
    return replace(
        provisional,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def _curves(clip):
    return {curve.control_id: tuple((key.frame, key.value) for key in curve.keys) for curve in clip.control_curves}


def test_preset_library_freezes_exact_motion_and_expression_semantics() -> None:
    controls = build_control_registry_plan()
    plan = build_preset_library_plan(controls)

    assert validate_preset_library_plan(plan, controls) is plan
    assert plan.library_version == PRESET_LIBRARY_VERSION
    assert plan.motion_schema_version == MOTION_CLIP_SCHEMA_VERSION
    assert plan.expression_schema_version == EXPRESSION_PRESET_SCHEMA_VERSION
    assert plan.runtime_application.version == MOTION_RUNTIME_APPLICATION_VERSION
    assert plan.optional_selection_version == OPTIONAL_PRESET_SELECTION_VERSION
    assert {clip.preset_id: (clip.duration_frames, clip.loop, _curves(clip)) for clip in plan.clips} == EXPECTED_CLIPS
    assert {
        expression.preset_id: {value.control_id: value.absolute_value for value in expression.values}
        for expression in plan.expressions
    } == EXPECTED_EXPRESSIONS
    assert plan.optional_priority == EXPECTED_OPTIONAL_PRIORITY


def test_preset_kinds_and_runtime_application_are_not_inferred_from_names() -> None:
    controls = build_control_registry_plan()
    plan = build_preset_library_plan(controls)

    assert {clip.preset_id for clip in plan.clips} >= {"blink", "talk"}
    assert {expression.preset_id for expression in plan.expressions} == {
        "happy",
        "unimpressed",
        "sad",
        "surprised",
        "wink_screen_left",
        "wink_screen_right",
    }
    assert all(clip.sample_rate_hz == 30 for clip in plan.clips)
    assert all(clip.interpolation == "linear" for clip in plan.clips)
    assert plan.runtime_application.motion_fade_in_seconds == 0.0
    assert plan.runtime_application.motion_fade_out_seconds == 0.0
    assert plan.runtime_application.motion_track_index == 0
    assert plan.runtime_application.expression_track_index == 1
    assert plan.runtime_application.expression_mode == "overwrite_full_weight"
    assert plan.runtime_application.apply_expression_after == "base_motion_track_0"


def test_blink_has_closed_and_open_holds_for_runtime_safe_restoration() -> None:
    controls = build_control_registry_plan()
    plan = build_preset_library_plan(controls)
    blink = next(clip for clip in plan.clips if clip.preset_id == "blink")

    assert blink.duration_frames == 24
    for curve in blink.control_curves:
        values = tuple((key.frame, key.value) for key in curve.keys)
        assert values[1:3] == ((6, 0.0), (9, 0.0))
        assert values[-2:] == ((16, 1.0), (24, 1.0))


def test_held_expressions_never_request_intermediate_eye_crossfade() -> None:
    controls = build_control_registry_plan()
    plan = build_preset_library_plan(controls)
    expressions = {
        expression.preset_id: {value.control_id: value.absolute_value for value in expression.values}
        for expression in plan.expressions
    }

    eye_controls = {"control/eye_open.xmin", "control/eye_open.xmax"}
    assert eye_controls.isdisjoint(expressions["happy"])
    assert eye_controls.isdisjoint(expressions["unimpressed"])
    for preset_id in ("wink_screen_left", "wink_screen_right"):
        assert set(expressions[preset_id]) == eye_controls
        assert set(expressions[preset_id].values()) == {0.0, 1.0}


@pytest.mark.parametrize(
    "mutation",
    (
        "unknown_control",
        "duplicate_curve",
        "non_increasing_frame",
        "open_loop",
        "out_of_domain",
        "bad_optional_priority",
        "expression_default_placeholder",
    ),
)
def test_preset_library_rejects_rehashed_registry_mutations(mutation: str) -> None:
    controls = build_control_registry_plan()
    plan = build_preset_library_plan(controls)
    clips = list(plan.clips)
    expressions = list(plan.expressions)
    changes = {}
    if mutation == "unknown_control":
        curve = replace(clips[0].control_curves[0], control_id="control/missing")
        clips[0] = replace(clips[0], control_curves=(curve,))
        changes["clips"] = tuple(clips)
    elif mutation == "duplicate_curve":
        clips[0] = replace(
            clips[0],
            control_curves=(clips[0].control_curves[0],) * 2,
        )
        changes["clips"] = tuple(clips)
    elif mutation == "non_increasing_frame":
        curve = clips[0].control_curves[0]
        keys = list(curve.keys)
        keys[1] = replace(keys[1], frame=keys[0].frame)
        clips[0] = replace(clips[0], control_curves=(replace(curve, keys=tuple(keys)),))
        changes["clips"] = tuple(clips)
    elif mutation == "open_loop":
        clip_index = next(index for index, clip in enumerate(clips) if clip.loop)
        curve = clips[clip_index].control_curves[0]
        keys = list(curve.keys)
        keys[-1] = replace(keys[-1], value=keys[-1].value + 0.25)
        clips[clip_index] = replace(
            clips[clip_index],
            control_curves=(replace(curve, keys=tuple(keys)),),
        )
        changes["clips"] = tuple(clips)
    elif mutation == "out_of_domain":
        clip_index = next(index for index, clip in enumerate(clips) if clip.preset_id == "breath")
        curve = clips[clip_index].control_curves[0]
        keys = list(curve.keys)
        keys[1] = replace(keys[1], value=2.0)
        clips[clip_index] = replace(
            clips[clip_index],
            control_curves=(replace(curve, keys=tuple(keys)),),
        )
        changes["clips"] = tuple(clips)
    elif mutation == "bad_optional_priority":
        changes["optional_priority"] = plan.optional_priority[:-1]
    else:
        expression = expressions[0]
        value = expression.values[0]
        control = next(item for item in controls.controls if item.control_id == value.control_id)
        expressions[0] = replace(
            expression,
            values=(replace(value, absolute_value=control.default), *expression.values[1:]),
        )
        changes["expressions"] = tuple(expressions)

    with pytest.raises(PresetLibraryError) as exc_info:
        validate_preset_library_plan(_rehash(plan, **changes), controls)

    assert exc_info.value.code == "invalid_preset_registry"
