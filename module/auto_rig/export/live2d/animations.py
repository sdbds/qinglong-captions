from __future__ import annotations

import math
from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping

from ...jcs import jcs_sha256
from ...rig_document import RigDocument, validate_rig_document
from .binding_plan import Live2DBindingPlan
from .e0_assets import cubism_runtime_json_bytes
from .keyforms import Live2DKeyformPlan
from .symbols import (
    Live2DSymbolView,
    require_live2d_symbol,
    validate_live2d_symbol_view,
)

LIVE2D_ANIMATION_PLAN_VERSION = "live2d-animation-plan-v3"
LIVE2D_MOTION_VERSION = 3
LIVE2D_MOTION_SAMPLE_RATE_HZ = 30
_EYE_OPEN_CONTROL_IDS = frozenset({"control/eye_open.xmin", "control/eye_open.xmax"})
_BLINK_RUNTIME_SEGMENTS = (
    0.0,
    1.0,
    0,
    0.2,
    0.0,
    0,
    0.3,
    0.0,
    0,
    16 / LIVE2D_MOTION_SAMPLE_RATE_HZ,
    1.0,
    0,
    0.8,
    1.0,
)


class Live2DAnimationError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(f"invalid_live2d_animation: {message}")


def _error(message: str) -> Live2DAnimationError:
    return Live2DAnimationError(message)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _error(f"{field} must be an object")
    return value


def _list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise _error(f"{field} must be a list")
    return value


def _number(value: object, *, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise _error(f"{field} must be finite numeric data")
    result = round(float(value), 12)
    return 0.0 if abs(result) < 1e-12 else result


def _runtime_sha256(payload: Mapping[str, object]) -> str:
    return f"sha256:{sha256(cubism_runtime_json_bytes(payload)).hexdigest()}"


@dataclass(frozen=True, slots=True)
class Live2DAnimationAsset:
    preset_id: str
    source_kind: str
    source_id: str
    artifact_export_name: str
    artifact_symbol_id: str
    relative_path: str
    required: bool
    selected_binding_ids: tuple[str, ...]
    parameter_ids: tuple[str, ...]
    payload: dict[str, object]
    runtime_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "preset_id": self.preset_id,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "artifact_export_name": self.artifact_export_name,
            "artifact_symbol_id": self.artifact_symbol_id,
            "relative_path": self.relative_path,
            "required": self.required,
            "selected_binding_ids": list(self.selected_binding_ids),
            "parameter_ids": list(self.parameter_ids),
            "payload": self.payload,
            "runtime_sha256": self.runtime_sha256,
        }


@dataclass(frozen=True, slots=True)
class Live2DAnimationPlan:
    schema_version: str
    runtime_json_encoder_version: str
    sample_rate_hz: int
    rig_document_sha256: str
    symbol_view_sha256: str
    binding_plan_sha256: str
    keyform_plan_sha256: str
    format_plan_sha256: str
    control_binding_set_sha256: str
    clip_set_sha256: str
    expression_set_sha256: str
    motion_assets: tuple[Live2DAnimationAsset, ...]
    expression_assets: tuple[Live2DAnimationAsset, ...]
    asset_set_sha256: str
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "runtime_json_encoder_version": self.runtime_json_encoder_version,
            "sample_rate_hz": self.sample_rate_hz,
            "rig_document_sha256": self.rig_document_sha256,
            "symbol_view_sha256": self.symbol_view_sha256,
            "binding_plan_sha256": self.binding_plan_sha256,
            "keyform_plan_sha256": self.keyform_plan_sha256,
            "format_plan_sha256": self.format_plan_sha256,
            "control_binding_set_sha256": self.control_binding_set_sha256,
            "clip_set_sha256": self.clip_set_sha256,
            "expression_set_sha256": self.expression_set_sha256,
            "motion_assets": [asset.to_dict() for asset in self.motion_assets],
            "expression_assets": [asset.to_dict() for asset in self.expression_assets],
            "asset_set_sha256": self.asset_set_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "plan_sha256": self.plan_sha256}


def encode_live2d_animation_asset(asset: Live2DAnimationAsset) -> bytes:
    if not isinstance(asset, Live2DAnimationAsset):
        raise _error("animation asset has the wrong type")
    encoded = cubism_runtime_json_bytes(asset.payload)
    if f"sha256:{sha256(encoded).hexdigest()}" != asset.runtime_sha256:
        raise _error("animation runtime JSON digest mismatch")
    return encoded


def _format_set(payload: Mapping[str, object]) -> Mapping[str, object]:
    formats = _mapping(payload.get("format_plans"), field="format plans")
    matches = [
        _mapping(raw, field="format preset-set plan")
        for raw in _list(formats.get("preset_set_plans"), field="format preset-set plans")
        if isinstance(raw, Mapping) and raw.get("format_id") == "live2d_moc3_v4_00"
    ]
    if len(matches) != 1:
        raise _error("Rig must contain exactly one Live2D preset-set plan")
    return matches[0]


def _motion_payload(
    clip: Mapping[str, object],
    control_to_parameter: Mapping[str, object],
) -> tuple[dict[str, object], tuple[str, ...]]:
    if clip.get("sample_rate_hz") != LIVE2D_MOTION_SAMPLE_RATE_HZ:
        raise _error("Live2D v1 accepts only 30 Hz MotionClip input")
    duration_frames = clip.get("duration_frames")
    loop = clip.get("loop")
    if (
        not isinstance(duration_frames, int)
        or isinstance(duration_frames, bool)
        or duration_frames <= 0
        or not isinstance(loop, bool)
    ):
        raise _error("MotionClip duration/loop contract is invalid")
    curves = []
    parameter_ids = []
    total_segments = 0
    total_points = 0
    for raw_curve in _list(clip.get("control_curves"), field="motion control curves"):
        curve = _mapping(raw_curve, field="motion control curve")
        control_id = curve.get("control_id")
        parameter = control_to_parameter.get(str(control_id))
        if not isinstance(control_id, str) or parameter is None:
            raise _error("motion curve references an unemitted control")
        export_name = getattr(parameter, "export_name", None)
        minimum = getattr(parameter, "minimum", None)
        maximum = getattr(parameter, "maximum", None)
        if not isinstance(export_name, str):
            raise _error("motion curve parameter contract is invalid")
        keys = []
        for raw_key in _list(curve.get("keys"), field="motion control keys"):
            key = _mapping(raw_key, field="motion control key")
            frame = _number(key.get("frame"), field="motion frame")
            value = _number(key.get("value"), field="motion value")
            if frame < 0 or frame > duration_frames:
                raise _error("motion frame lies outside duration")
            if value < float(minimum) or value > float(maximum):
                raise _error("motion value lies outside parameter min/max")
            keys.append((frame, value))
        if (
            len(keys) < 2
            or keys != sorted(keys)
            or len({frame for frame, _value in keys}) != len(keys)
            or keys[0][0] != 0.0
            or keys[-1][0] != float(duration_frames)
        ):
            raise _error("motion keys are not a complete strictly ordered curve")
        segments: list[float | int] = [
            keys[0][0] / LIVE2D_MOTION_SAMPLE_RATE_HZ,
            keys[0][1],
        ]
        for frame, value in keys[1:]:
            segments.extend((0, frame / LIVE2D_MOTION_SAMPLE_RATE_HZ, value))
        curves.append({"Target": "Parameter", "Id": export_name, "Segments": segments})
        parameter_ids.append(export_name)
        total_segments += len(keys) - 1
        total_points += len(keys)
    if not curves or len(set(parameter_ids)) != len(parameter_ids):
        raise _error("motion contains no curves or duplicate parameters")
    ordered = sorted(zip(parameter_ids, curves, strict=True))
    parameter_ids = [identity for identity, _curve in ordered]
    curves = [curve for _identity, curve in ordered]
    payload = {
        "Version": LIVE2D_MOTION_VERSION,
        "Meta": {
            "Duration": duration_frames / LIVE2D_MOTION_SAMPLE_RATE_HZ,
            "Fps": float(LIVE2D_MOTION_SAMPLE_RATE_HZ),
            "Loop": loop,
            "FadeInTime": 0.0,
            "FadeOutTime": 0.0,
            "AreBeziersRestricted": True,
            "CurveCount": len(curves),
            "TotalSegmentCount": total_segments,
            "TotalPointCount": total_points,
            "UserDataCount": 0,
            "TotalUserDataSize": 0,
        },
        "Curves": curves,
        "UserData": [],
    }
    return payload, tuple(parameter_ids)


def _expression_payload(
    expression: Mapping[str, object],
    control_to_parameter: Mapping[str, object],
    keyforms: Live2DKeyformPlan,
) -> tuple[dict[str, object], tuple[str, ...]]:
    if expression.get("application_mode") != "overwrite_full_weight":
        raise _error("Live2D formal expressions require full-weight absolute values")
    keyform_ranges: dict[str, tuple[float, float]] = {}
    for record in keyforms.artmesh_keyforms:
        if record.parameter_id is not None:
            current = keyform_ranges.get(record.parameter_id)
            low = record.parameter_values[0]
            high = record.parameter_values[-1]
            keyform_ranges[record.parameter_id] = (
                low if current is None else min(low, current[0]),
                high if current is None else max(high, current[1]),
            )
    parameters = []
    ids = []
    for raw in _list(expression.get("values"), field="expression values"):
        value = _mapping(raw, field="expression value")
        control_id = value.get("control_id")
        parameter = control_to_parameter.get(str(control_id))
        if not isinstance(control_id, str) or parameter is None:
            raise _error("expression references an unemitted control")
        parameter_id = getattr(parameter, "parameter_id", None)
        export_name = getattr(parameter, "export_name", None)
        minimum = getattr(parameter, "minimum", None)
        maximum = getattr(parameter, "maximum", None)
        default = getattr(parameter, "default", None)
        absolute = _number(value.get("absolute_value"), field="expression absolute value")
        if (
            not isinstance(parameter_id, str)
            or not isinstance(export_name, str)
            or absolute < float(minimum)
            or absolute > float(maximum)
        ):
            raise _error("expression parameter/value contract is invalid")
        keyform_range = keyform_ranges.get(parameter_id)
        if keyform_range is None or absolute < keyform_range[0] or absolute > keyform_range[1]:
            raise _error("expression value lacks an ArtMesh keyform interval")
        if control_id in _EYE_OPEN_CONTROL_IDS and default != 1.0:
            raise _error("multiplicative eye-open expressions require a default of 1")
        # Cubism applies a persistent expression after the active motion. Eye-open
        # values therefore multiply the motion result so an explicit blink can
        # still close both eyes; all other absolute expression targets overwrite.
        blend = "Multiply" if control_id in _EYE_OPEN_CONTROL_IDS else "Overwrite"
        parameters.append({"Id": export_name, "Value": absolute, "Blend": blend})
        ids.append(export_name)
    if not parameters or len(set(ids)) != len(ids):
        raise _error("expression contains no targets or duplicate parameters")
    parameters.sort(key=lambda item: str(item["Id"]))
    ids.sort()
    return (
        {
            "Type": "Live2D Expression",
            "FadeInTime": 0.0,
            "FadeOutTime": 0.0,
            "Parameters": parameters,
        },
        tuple(ids),
    )


def _assemble(
    rig: RigDocument,
    symbols: Live2DSymbolView,
    bindings: Live2DBindingPlan,
    keyforms: Live2DKeyformPlan,
) -> Live2DAnimationPlan:
    payload = rig.to_dict()
    format_set = _format_set(payload)
    decisions = [
        _mapping(raw, field="Live2D format decision") for raw in _list(format_set.get("decisions"), field="Live2D decisions")
    ]
    supported = [item for item in decisions if item.get("status") == "supported"]
    if tuple(sorted(str(item.get("preset_id")) for item in supported)) != tuple(sorted(bindings.supported_preset_ids)):
        raise _error("binding plan and format decisions support different presets")
    raw_bindings = _list(payload.get("control_bindings"), field="control bindings")
    binding_by_id = {}
    for raw in raw_bindings:
        binding = _mapping(raw, field="control binding")
        identity = binding.get("binding_id")
        if not isinstance(identity, str) or identity in binding_by_id:
            raise _error("control binding identity is invalid or duplicated")
        binding_by_id[identity] = binding
    control_to_parameter = {parameter.control_id: parameter for parameter in bindings.parameters}
    clips = {
        str(item["preset_id"]): item
        for raw in _list(payload.get("clips"), field="clips")
        for item in (_mapping(raw, field="clip"),)
    }
    expressions = {
        str(item["preset_id"]): item
        for raw in _list(payload.get("expressions"), field="expressions")
        for item in (_mapping(raw, field="expression"),)
    }

    motions = []
    expression_assets = []
    used_paths: set[str] = set()
    for decision in sorted(supported, key=lambda item: str(item.get("preset_id"))):
        preset_id = decision.get("preset_id")
        artifact_name = decision.get("artifact_export_name")
        artifact_symbol_id = decision.get("artifact_symbol_id")
        selected_ids = decision.get("selected_binding_ids")
        if (
            not isinstance(preset_id, str)
            or not isinstance(artifact_name, str)
            or not artifact_name
            or "/" in artifact_name
            or "\\" in artifact_name
            or not isinstance(artifact_symbol_id, str)
            or not isinstance(selected_ids, list)
            or not selected_ids
            or any(not isinstance(value, str) for value in selected_ids)
        ):
            raise _error("supported decision has an invalid artifact contract")
        selected = []
        for identity in selected_ids:
            binding = binding_by_id.get(identity)
            if binding is None:
                raise _error("supported decision selects an absent binding")
            selected.append(binding)
        selected_controls = {str(item.get("control_id")) for item in selected}
        if preset_id in clips:
            source_kind = "motion"
            source = clips[preset_id]
            source_id = source.get("clip_id")
            curve_controls = {
                str(_mapping(raw, field="motion curve").get("control_id"))
                for raw in _list(source.get("control_curves"), field="motion control curves")
            }
            if selected_controls != curve_controls:
                raise _error("motion controls differ from selected bindings")
            asset_payload, parameter_ids = _motion_payload(source, control_to_parameter)
            kind = "live2d_motion"
            relative_path = f"motions/{artifact_name}.motion3.json"
        elif preset_id in expressions:
            source_kind = "expression"
            source = expressions[preset_id]
            source_id = source.get("expression_id")
            value_controls = {
                str(_mapping(raw, field="expression value").get("control_id"))
                for raw in _list(source.get("values"), field="expression values")
            }
            if selected_controls != value_controls:
                raise _error("expression controls differ from selected bindings")
            asset_payload, parameter_ids = _expression_payload(source, control_to_parameter, keyforms)
            kind = "live2d_expression"
            relative_path = f"expressions/{artifact_name}.exp3.json"
        else:
            raise _error("supported preset has no clip or expression source")
        if not isinstance(source_id, str):
            raise _error("supported source identity is invalid")
        symbol = require_live2d_symbol(
            symbols,
            kind=kind,
            source_internal_ids=(source_id,),
            preset_id=source_id,
        )
        if symbol.export_name != artifact_name or symbol.symbol_id != artifact_symbol_id:
            raise _error("runtime artifact differs from the global symbol")
        if relative_path in used_paths:
            raise _error("two supported presets share one runtime path")
        used_paths.add(relative_path)
        asset = Live2DAnimationAsset(
            preset_id=preset_id,
            source_kind=source_kind,
            source_id=source_id,
            artifact_export_name=artifact_name,
            artifact_symbol_id=artifact_symbol_id,
            relative_path=relative_path,
            required=bool(decision.get("required")),
            selected_binding_ids=tuple(selected_ids),
            parameter_ids=parameter_ids,
            payload=asset_payload,
            runtime_sha256=_runtime_sha256(asset_payload),
        )
        if source_kind == "motion":
            motions.append(asset)
        else:
            expression_assets.append(asset)
    if not motions:
        raise _error("Live2D preset plan has no supported motions")
    motions_tuple = tuple(sorted(motions, key=lambda item: item.preset_id))
    expressions_tuple = tuple(sorted(expression_assets, key=lambda item: item.preset_id))
    asset_payload = {
        "motions": [asset.to_dict() for asset in motions_tuple],
        "expressions": [asset.to_dict() for asset in expressions_tuple],
    }
    values = {
        "schema_version": LIVE2D_ANIMATION_PLAN_VERSION,
        "runtime_json_encoder_version": "cubism-runtime-json-v1",
        "sample_rate_hz": LIVE2D_MOTION_SAMPLE_RATE_HZ,
        "rig_document_sha256": rig.document_sha256,
        "symbol_view_sha256": symbols.view_sha256,
        "binding_plan_sha256": bindings.plan_sha256,
        "keyform_plan_sha256": keyforms.plan_sha256,
        "format_plan_sha256": str(payload["format_plans"]["plan_sha256"]),
        "control_binding_set_sha256": jcs_sha256(raw_bindings),
        "clip_set_sha256": jcs_sha256(payload["clips"]),
        "expression_set_sha256": jcs_sha256(payload["expressions"]),
        "motion_assets": motions_tuple,
        "expression_assets": expressions_tuple,
        "asset_set_sha256": jcs_sha256(asset_payload),
    }
    provisional = Live2DAnimationPlan(**values, plan_sha256="")
    return Live2DAnimationPlan(**values, plan_sha256=jcs_sha256(provisional.semantic_payload()))


def build_live2d_animation_plan(
    rig: RigDocument,
    symbols: Live2DSymbolView,
    bindings: Live2DBindingPlan,
    keyforms: Live2DKeyformPlan,
) -> Live2DAnimationPlan:
    validate_rig_document(rig)
    validate_live2d_symbol_view(symbols)
    return _assemble(rig, symbols, bindings, keyforms)


def _validate_motion(asset: Live2DAnimationAsset) -> None:
    payload = asset.payload
    meta = _mapping(payload.get("Meta"), field="motion Meta")
    curves = _list(payload.get("Curves"), field="motion Curves")
    ids = []
    segment_count = 0
    point_count = 0
    for raw in curves:
        curve = _mapping(raw, field="motion curve")
        identity = curve.get("Id")
        segments = _list(curve.get("Segments"), field="motion Segments")
        if (
            curve.get("Target") != "Parameter"
            or not isinstance(identity, str)
            or len(segments) < 5
            or (len(segments) - 2) % 3
            or any(segments[index] != 0 for index in range(2, len(segments), 3))
            or "FadeInTime" in curve
            or "FadeOutTime" in curve
        ):
            raise _error("motion curve is not an exact linear parameter curve")
        ids.append(identity)
        segment_count += (len(segments) - 2) // 3
        point_count += 1 + (len(segments) - 2) // 3
    if len(ids) != len(set(ids)):
        raise _error("motion has duplicate parameter curves")
    if asset.preset_id == "blink" and (
        meta.get("Duration") != 0.8
        or meta.get("Loop") is not False
        or len(curves) != 2
        or any(tuple(_mapping(raw, field="blink curve").get("Segments", ())) != _BLINK_RUNTIME_SEGMENTS for raw in curves)
    ):
        raise _error("blink motion lacks its closed/open recovery hold")
    if (
        meta.get("CurveCount") != len(curves)
        or meta.get("TotalSegmentCount") != segment_count
        or meta.get("TotalPointCount") != point_count
        or meta.get("FadeInTime") != 0.0
        or meta.get("FadeOutTime") != 0.0
        or payload.get("UserData") != []
    ):
        raise _error("motion Meta counts/fade differ from its curves")


def _validate_expression(
    asset: Live2DAnimationAsset,
    *,
    eye_parameter_ids: frozenset[str],
) -> None:
    payload = asset.payload
    parameters = _list(payload.get("Parameters"), field="expression Parameters")
    ids = []
    for raw in parameters:
        parameter = _mapping(raw, field="expression parameter")
        identity = parameter.get("Id")
        expected_blend = "Multiply" if identity in eye_parameter_ids else "Overwrite"
        if not isinstance(identity, str) or parameter.get("Blend") != expected_blend or set(parameter) != {"Id", "Value", "Blend"}:
            raise _error("expression target has an invalid per-control blend mode")
        ids.append(identity)
    if not parameters or len(ids) != len(set(ids)) or payload.get("FadeInTime") != 0.0 or payload.get("FadeOutTime") != 0.0:
        raise _error("expression has duplicate targets or nonzero fade")


def validate_live2d_animation_plan(
    plan: Live2DAnimationPlan,
    rig: RigDocument,
    symbols: Live2DSymbolView,
    bindings: Live2DBindingPlan,
    keyforms: Live2DKeyformPlan,
) -> Live2DAnimationPlan:
    if not isinstance(plan, Live2DAnimationPlan):
        raise _error("animation plan has the wrong type")
    for asset in plan.motion_assets:
        _validate_motion(asset)
        encode_live2d_animation_asset(asset)
    eye_parameter_ids = frozenset(
        parameter.export_name for parameter in bindings.parameters if parameter.control_id in _EYE_OPEN_CONTROL_IDS
    )
    for asset in plan.expression_assets:
        _validate_expression(asset, eye_parameter_ids=eye_parameter_ids)
        encode_live2d_animation_asset(asset)
    expected = _assemble(rig, symbols, bindings, keyforms)
    if plan != expected:
        raise _error("animation plan differs from canonical Stage C decisions")
    if plan.plan_sha256 != jcs_sha256(plan.semantic_payload()):
        raise _error("animation plan digest mismatch")
    return plan


__all__ = [
    "LIVE2D_ANIMATION_PLAN_VERSION",
    "LIVE2D_MOTION_SAMPLE_RATE_HZ",
    "LIVE2D_MOTION_VERSION",
    "Live2DAnimationAsset",
    "Live2DAnimationError",
    "Live2DAnimationPlan",
    "build_live2d_animation_plan",
    "encode_live2d_animation_asset",
    "validate_live2d_animation_plan",
]
