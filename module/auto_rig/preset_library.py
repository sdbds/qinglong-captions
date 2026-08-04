from __future__ import annotations

import math
from dataclasses import dataclass

from .control_registry import (
    ControlRegistryPlan,
    ControlSpec,
    validate_control_registry_plan,
)
from .jcs import jcs_sha256

PRESET_LIBRARY_PLAN_VERSION = "preset-library-plan-v1"
PRESET_LIBRARY_VERSION = "motion-core-v6"
MOTION_CLIP_SCHEMA_VERSION = "motion-clip-v1"
EXPRESSION_PRESET_SCHEMA_VERSION = "expression-preset-v1"
MOTION_RUNTIME_APPLICATION_VERSION = "motion-runtime-application-v1"
OPTIONAL_PRESET_SELECTION_VERSION = "optional-preset-selection-v4"
MOTION_SAMPLE_RATE_HZ = 30
_BLINK_DURATION_FRAMES = 24
_BLINK_KEYS = ((0, 1.0), (6, 0.0), (9, 0.0), (16, 1.0), (24, 1.0))
CORE_REQUIRED_PRESET_IDS = ("breath", "head_nod", "head_shake", "idle")
OPTIONAL_PRESET_PRIORITY = (
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


class PresetLibraryError(ValueError):
    """Raised when the packaged preset registry is invalid."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _error(message: str) -> PresetLibraryError:
    return PresetLibraryError("invalid_preset_registry", message)


@dataclass(frozen=True, slots=True)
class ControlKey:
    frame: int
    value: float

    def to_dict(self) -> dict[str, object]:
        return {"frame": self.frame, "value": self.value}


@dataclass(frozen=True, slots=True)
class ControlCurve:
    control_id: str
    keys: tuple[ControlKey, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "control_id": self.control_id,
            "keys": [key.to_dict() for key in self.keys],
        }


@dataclass(frozen=True, slots=True)
class MotionClipTemplate:
    preset_id: str
    clip_id: str
    preset_version: str
    sample_rate_hz: int
    duration_frames: int
    interpolation: str
    loop: bool
    control_curves: tuple[ControlCurve, ...]
    template_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "preset_id": self.preset_id,
            "clip_id": self.clip_id,
            "preset_version": self.preset_version,
            "sample_rate_hz": self.sample_rate_hz,
            "duration_frames": self.duration_frames,
            "interpolation": self.interpolation,
            "loop": self.loop,
            "control_curves": [curve.to_dict() for curve in self.control_curves],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "template_sha256": self.template_sha256}


@dataclass(frozen=True, slots=True)
class ExpressionValue:
    control_id: str
    absolute_value: float

    def to_dict(self) -> dict[str, object]:
        return {
            "control_id": self.control_id,
            "absolute_value": self.absolute_value,
        }


@dataclass(frozen=True, slots=True)
class ExpressionPresetTemplate:
    preset_id: str
    expression_id: str
    preset_version: str
    application_mode: str
    values: tuple[ExpressionValue, ...]
    template_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "preset_id": self.preset_id,
            "expression_id": self.expression_id,
            "preset_version": self.preset_version,
            "application_mode": self.application_mode,
            "values": [value.to_dict() for value in self.values],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "template_sha256": self.template_sha256}


@dataclass(frozen=True, slots=True)
class MotionRuntimeApplication:
    version: str
    motion_weight: float
    motion_fade_in_seconds: float
    motion_fade_out_seconds: float
    motion_track_index: int
    motion_alpha: float
    motion_mix_duration_seconds: float
    crossfade: bool
    expression_mode: str
    expression_weight: float
    expression_fade_in_seconds: float
    expression_fade_out_seconds: float
    expression_track_index: int
    expression_loop: bool
    expression_mix_blend: str
    expression_mix_duration_seconds: float
    hold_until_cleared: bool
    apply_expression_after: str

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "motion": {
                "weight": self.motion_weight,
                "fade_in_seconds": self.motion_fade_in_seconds,
                "fade_out_seconds": self.motion_fade_out_seconds,
                "track_index": self.motion_track_index,
                "alpha": self.motion_alpha,
                "mix_duration_seconds": self.motion_mix_duration_seconds,
                "crossfade": self.crossfade,
            },
            "expression": {
                "mode": self.expression_mode,
                "weight": self.expression_weight,
                "fade_in_seconds": self.expression_fade_in_seconds,
                "fade_out_seconds": self.expression_fade_out_seconds,
                "track_index": self.expression_track_index,
                "loop": self.expression_loop,
                "mix_blend": self.expression_mix_blend,
                "mix_duration_seconds": self.expression_mix_duration_seconds,
                "hold_until_cleared": self.hold_until_cleared,
                "apply_after": self.apply_expression_after,
            },
        }


@dataclass(frozen=True, slots=True)
class PresetLibraryPlan:
    schema_version: str
    library_version: str
    motion_schema_version: str
    expression_schema_version: str
    control_registry_sha256: str
    runtime_application: MotionRuntimeApplication
    clips: tuple[MotionClipTemplate, ...]
    expressions: tuple[ExpressionPresetTemplate, ...]
    core_required_preset_ids: tuple[str, ...]
    optional_selection_version: str
    optional_priority: tuple[str, ...]
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "library_version": self.library_version,
            "motion_schema_version": self.motion_schema_version,
            "expression_schema_version": self.expression_schema_version,
            "control_registry_sha256": self.control_registry_sha256,
            "runtime_application": self.runtime_application.to_dict(),
            "clips": [clip.to_dict() for clip in self.clips],
            "expressions": [expression.to_dict() for expression in self.expressions],
            "core_required_preset_ids": list(self.core_required_preset_ids),
            "optional_selection_version": self.optional_selection_version,
            "optional_priority": list(self.optional_priority),
        }


def _keys(values: tuple[tuple[int, float], ...]) -> tuple[ControlKey, ...]:
    return tuple(ControlKey(frame=frame, value=value) for frame, value in values)


def _curve(control_id: str, values: tuple[tuple[int, float], ...]) -> ControlCurve:
    return ControlCurve(control_id=control_id, keys=_keys(values))


def _clip(
    preset_id: str,
    duration_frames: int,
    loop: bool,
    curves: tuple[ControlCurve, ...],
) -> MotionClipTemplate:
    values = {
        "preset_id": preset_id,
        "clip_id": f"clip/{preset_id}",
        "preset_version": PRESET_LIBRARY_VERSION,
        "sample_rate_hz": MOTION_SAMPLE_RATE_HZ,
        "duration_frames": duration_frames,
        "interpolation": "linear",
        "loop": loop,
        "control_curves": tuple(sorted(curves, key=lambda item: item.control_id)),
    }
    provisional = MotionClipTemplate(**values, template_sha256="")
    return MotionClipTemplate(
        **values,
        template_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def _expression(
    preset_id: str,
    values_by_control: tuple[tuple[str, float], ...],
) -> ExpressionPresetTemplate:
    values = {
        "preset_id": preset_id,
        "expression_id": f"expression/{preset_id}",
        "preset_version": PRESET_LIBRARY_VERSION,
        "application_mode": "overwrite_full_weight",
        "values": tuple(
            ExpressionValue(control_id=control_id, absolute_value=value) for control_id, value in sorted(values_by_control)
        ),
    }
    provisional = ExpressionPresetTemplate(**values, template_sha256="")
    return ExpressionPresetTemplate(
        **values,
        template_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def _runtime_application() -> MotionRuntimeApplication:
    return MotionRuntimeApplication(
        version=MOTION_RUNTIME_APPLICATION_VERSION,
        motion_weight=1.0,
        motion_fade_in_seconds=0.0,
        motion_fade_out_seconds=0.0,
        motion_track_index=0,
        motion_alpha=1.0,
        motion_mix_duration_seconds=0.0,
        crossfade=False,
        expression_mode="overwrite_full_weight",
        expression_weight=1.0,
        expression_fade_in_seconds=0.0,
        expression_fade_out_seconds=0.0,
        expression_track_index=1,
        expression_loop=True,
        expression_mix_blend="replace",
        expression_mix_duration_seconds=0.0,
        hold_until_cleared=True,
        apply_expression_after="base_motion_track_0",
    )


def _control_map(plan: ControlRegistryPlan) -> dict[str, ControlSpec]:
    return {control.control_id: control for control in plan.controls}


def validate_preset_library_plan(
    plan: PresetLibraryPlan,
    controls: ControlRegistryPlan,
) -> PresetLibraryPlan:
    """Validate preset identity, curve semantics, and control-domain closure."""

    validate_control_registry_plan(controls)
    if not isinstance(plan, PresetLibraryPlan):
        raise _error("preset library must use PresetLibraryPlan")
    if (
        plan.schema_version != PRESET_LIBRARY_PLAN_VERSION
        or plan.library_version != PRESET_LIBRARY_VERSION
        or plan.motion_schema_version != MOTION_CLIP_SCHEMA_VERSION
        or plan.expression_schema_version != EXPRESSION_PRESET_SCHEMA_VERSION
        or plan.optional_selection_version != OPTIONAL_PRESET_SELECTION_VERSION
    ):
        raise _error("preset library version is unsupported")
    if plan.control_registry_sha256 != controls.registry_sha256:
        raise _error("preset library references a different ControlRegistry")
    if plan.runtime_application != _runtime_application():
        raise _error("runtime application contract differs from v1")
    if tuple(sorted(plan.clips, key=lambda item: item.preset_id)) != plan.clips:
        raise _error("motion clips must be sorted by preset ID")
    if tuple(sorted(plan.expressions, key=lambda item: item.preset_id)) != plan.expressions:
        raise _error("expressions must be sorted by preset ID")
    control_by_id = _control_map(controls)
    preset_ids: list[str] = []
    for clip in plan.clips:
        preset_ids.append(clip.preset_id)
        if (
            clip.clip_id != f"clip/{clip.preset_id}"
            or clip.preset_version != PRESET_LIBRARY_VERSION
            or clip.sample_rate_hz != MOTION_SAMPLE_RATE_HZ
            or clip.duration_frames <= 0
            or clip.interpolation != "linear"
        ):
            raise _error(f"invalid clip descriptor: {clip.preset_id}")
        if tuple(sorted(clip.control_curves, key=lambda item: item.control_id)) != clip.control_curves:
            raise _error(f"curves are not canonical for {clip.preset_id}")
        curve_ids = tuple(curve.control_id for curve in clip.control_curves)
        if not curve_ids or len(curve_ids) != len(set(curve_ids)):
            raise _error(f"clip has duplicate or empty curves: {clip.preset_id}")
        for curve in clip.control_curves:
            control = control_by_id.get(curve.control_id)
            if control is None:
                raise _error(f"clip references unknown control: {curve.control_id}")
            if len(curve.keys) < 2:
                raise _error(f"curve has fewer than two keys: {clip.preset_id}")
            frames = tuple(key.frame for key in curve.keys)
            if (
                any(isinstance(frame, bool) or not isinstance(frame, int) for frame in frames)
                or any(left >= right for left, right in zip(frames, frames[1:]))
                or frames[0] != 0
                or frames[-1] != clip.duration_frames
            ):
                raise _error(f"curve frame grid is invalid: {clip.preset_id}")
            for key in curve.keys:
                if not isinstance(key.value, (int, float)) or not math.isfinite(key.value):
                    raise _error(f"curve contains a non-finite value: {clip.preset_id}")
                if not control.minimum <= key.value <= control.maximum:
                    raise _error(f"curve exceeds the control domain: {clip.preset_id}")
            if clip.loop and curve.keys[0].value != curve.keys[-1].value:
                raise _error(f"loop endpoints differ: {clip.preset_id}")
        if clip.preset_id == "blink":
            blink_curves = {curve.control_id: tuple((key.frame, key.value) for key in curve.keys) for curve in clip.control_curves}
            if (
                clip.loop
                or clip.duration_frames != _BLINK_DURATION_FRAMES
                or blink_curves
                != {
                    "control/eye_open.xmax": _BLINK_KEYS,
                    "control/eye_open.xmin": _BLINK_KEYS,
                }
            ):
                raise _error("blink lacks the frozen closed/open recovery holds")
        if clip.template_sha256 != jcs_sha256(clip.semantic_payload()):
            raise _error(f"clip digest mismatch: {clip.preset_id}")
    for expression in plan.expressions:
        preset_ids.append(expression.preset_id)
        if (
            expression.expression_id != f"expression/{expression.preset_id}"
            or expression.preset_version != PRESET_LIBRARY_VERSION
            or expression.application_mode != "overwrite_full_weight"
        ):
            raise _error(f"invalid expression descriptor: {expression.preset_id}")
        if tuple(sorted(expression.values, key=lambda item: item.control_id)) != expression.values:
            raise _error(f"expression values are not canonical: {expression.preset_id}")
        value_ids = tuple(value.control_id for value in expression.values)
        if not value_ids or len(value_ids) != len(set(value_ids)):
            raise _error(f"expression has duplicate or empty values: {expression.preset_id}")
        for value in expression.values:
            control = control_by_id.get(value.control_id)
            if control is None:
                raise _error(f"expression references unknown control: {value.control_id}")
            if not math.isfinite(value.absolute_value) or not (control.minimum <= value.absolute_value <= control.maximum):
                raise _error(f"expression value exceeds the domain: {expression.preset_id}")
            intentional_wink_reset = (
                expression.preset_id in {"wink_screen_left", "wink_screen_right"}
                and value.control_id in {"control/eye_open.xmin", "control/eye_open.xmax"}
                and value.absolute_value == 1.0
            )
            if value.absolute_value == control.default and not intentional_wink_reset:
                raise _error(f"expression contains a default-value placeholder: {expression.preset_id}")
        if expression.template_sha256 != jcs_sha256(expression.semantic_payload()):
            raise _error(f"expression digest mismatch: {expression.preset_id}")
    if len(preset_ids) != len(set(preset_ids)):
        raise _error("preset IDs must be unique across clips and expressions")
    all_presets = set(preset_ids)
    if plan.core_required_preset_ids != CORE_REQUIRED_PRESET_IDS:
        raise _error("core required preset registry differs from v1")
    if plan.optional_priority != OPTIONAL_PRESET_PRIORITY:
        raise _error("optional priority registry differs from v1")
    if set(plan.core_required_preset_ids) | set(plan.optional_priority) != all_presets:
        raise _error("required and optional preset registries do not cover the library")
    if set(plan.core_required_preset_ids) & set(plan.optional_priority):
        raise _error("a preset cannot be both required-core and optional")
    if plan.plan_sha256 != jcs_sha256(plan.semantic_payload()):
        raise _error("preset library plan digest mismatch")
    return plan


def build_preset_library_plan(
    controls: ControlRegistryPlan,
) -> PresetLibraryPlan:
    """Build the complete exporter-neutral PresetLibrary motion-core-v1."""

    validate_control_registry_plan(controls)
    clips = (
        _clip(
            "arm_sway",
            90,
            True,
            (
                _curve(
                    "control/arm_sway",
                    ((0, 0.0), (22, 1.0), (45, 0.0), (67, -1.0), (90, 0.0)),
                ),
            ),
        ),
        _clip(
            "blink",
            _BLINK_DURATION_FRAMES,
            False,
            (
                _curve("control/eye_open.xmax", _BLINK_KEYS),
                _curve("control/eye_open.xmin", _BLINK_KEYS),
            ),
        ),
        _clip(
            "body_sway",
            120,
            True,
            (
                _curve(
                    "control/body_sway",
                    ((0, 0.0), (30, 5.0), (60, 0.0), (90, -5.0), (120, 0.0)),
                ),
            ),
        ),
        _clip(
            "breath",
            90,
            True,
            (_curve("control/breath", ((0, 0.0), (45, 1.0), (90, 0.0))),),
        ),
        _clip(
            "head_nod",
            30,
            False,
            (_curve("control/head_nod", ((0, 0.0), (15, 15.0), (30, 0.0))),),
        ),
        _clip(
            "head_shake",
            45,
            False,
            (
                _curve(
                    "control/head_shake",
                    ((0, 0.0), (10, -20.0), (25, 20.0), (45, 0.0)),
                ),
            ),
        ),
        _clip(
            "idle",
            120,
            True,
            (
                _curve(
                    "control/idle",
                    ((0, 0.0), (30, 1.0), (60, 0.0), (90, -1.0), (120, 0.0)),
                ),
            ),
        ),
        _clip(
            "leg_sway",
            120,
            True,
            (
                _curve(
                    "control/leg_sway",
                    ((0, 0.0), (30, 1.0), (60, 0.0), (90, -1.0), (120, 0.0)),
                ),
            ),
        ),
        _clip(
            "talk",
            30,
            True,
            (
                _curve(
                    "control/mouth_open",
                    ((0, 0.0), (10, 1.0), (20, 0.35), (30, 0.0)),
                ),
            ),
        ),
        _clip(
            "wave.xmax",
            60,
            False,
            (
                _curve(
                    "control/wave_lift.xmax",
                    ((0, 0.0), (15, 1.0), (45, 1.0), (60, 0.0)),
                ),
                _curve(
                    "control/wave_osc.xmax",
                    ((0, 0.0), (15, 0.0), (25, 1.0), (35, -1.0), (45, 1.0), (60, 0.0)),
                ),
            ),
        ),
        _clip(
            "wave.xmin",
            60,
            False,
            (
                _curve(
                    "control/wave_lift.xmin",
                    ((0, 0.0), (15, 1.0), (45, 1.0), (60, 0.0)),
                ),
                _curve(
                    "control/wave_osc.xmin",
                    ((0, 0.0), (15, 0.0), (25, 1.0), (35, -1.0), (45, 1.0), (60, 0.0)),
                ),
            ),
        ),
    )
    expressions = (
        _expression(
            "happy",
            (
                ("control/mouth_form", 0.7),
                ("control/brow_y.xmin", 0.15),
                ("control/brow_y.xmax", 0.15),
            ),
        ),
        _expression(
            "unimpressed",
            (
                ("control/mouth_form", -0.15),
                ("control/brow_y.xmin", -0.35),
                ("control/brow_y.xmax", -0.35),
            ),
        ),
        _expression(
            "sad",
            (
                ("control/mouth_form", -0.7),
                ("control/brow_y.xmin", 0.25),
                ("control/brow_y.xmax", 0.25),
            ),
        ),
        _expression(
            "surprised",
            (
                ("control/mouth_open", 0.8),
                ("control/brow_y.xmin", 0.8),
                ("control/brow_y.xmax", 0.8),
            ),
        ),
        _expression(
            "wink_screen_right",
            (
                ("control/eye_open.xmin", 1.0),
                ("control/eye_open.xmax", 0.0),
            ),
        ),
        _expression(
            "wink_screen_left",
            (
                ("control/eye_open.xmin", 0.0),
                ("control/eye_open.xmax", 1.0),
            ),
        ),
    )
    values = {
        "schema_version": PRESET_LIBRARY_PLAN_VERSION,
        "library_version": PRESET_LIBRARY_VERSION,
        "motion_schema_version": MOTION_CLIP_SCHEMA_VERSION,
        "expression_schema_version": EXPRESSION_PRESET_SCHEMA_VERSION,
        "control_registry_sha256": controls.registry_sha256,
        "runtime_application": _runtime_application(),
        "clips": tuple(sorted(clips, key=lambda item: item.preset_id)),
        "expressions": tuple(sorted(expressions, key=lambda item: item.preset_id)),
        "core_required_preset_ids": CORE_REQUIRED_PRESET_IDS,
        "optional_selection_version": OPTIONAL_PRESET_SELECTION_VERSION,
        "optional_priority": OPTIONAL_PRESET_PRIORITY,
    }
    provisional = PresetLibraryPlan(**values, plan_sha256="")
    plan = PresetLibraryPlan(
        **values,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )
    return validate_preset_library_plan(plan, controls)


__all__ = [
    "CORE_REQUIRED_PRESET_IDS",
    "EXPRESSION_PRESET_SCHEMA_VERSION",
    "MOTION_CLIP_SCHEMA_VERSION",
    "MOTION_RUNTIME_APPLICATION_VERSION",
    "MOTION_SAMPLE_RATE_HZ",
    "OPTIONAL_PRESET_PRIORITY",
    "OPTIONAL_PRESET_SELECTION_VERSION",
    "PRESET_LIBRARY_PLAN_VERSION",
    "PRESET_LIBRARY_VERSION",
    "ControlCurve",
    "ControlKey",
    "ExpressionPresetTemplate",
    "ExpressionValue",
    "MotionClipTemplate",
    "MotionRuntimeApplication",
    "PresetLibraryError",
    "PresetLibraryPlan",
    "build_preset_library_plan",
    "validate_preset_library_plan",
]
