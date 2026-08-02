from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from ...jcs import jcs_sha256
from ...rig_document import RigDocument, validate_rig_document
from .bind_plan import (
    SpineBindPlan,
    SpineMeshBind,
    spine_local_point,
    spine_parent_local_vector,
    validate_spine_bind_plan,
)
from .coordinates import (
    SpineCoordinatePlan,
    canvas_to_spine,
    validate_spine_coordinate_plan,
)
from .model import SPINE_DOCUMENT_VERSION, SpineDocument
from .symbols import (
    SpineSymbolView,
    require_spine_symbol,
    validate_spine_symbol_view,
)

SPINE_ANIMATION_PLAN_VERSION = "spine-animation-plan-v1"
SPINE_EXPRESSION_HOLD_VERSION = "spine-expression-hold-v1"
SPINE_ANIMATION_SAMPLE_RATE_HZ = 30


class SpineAnimationError(ValueError):
    """Raised when frozen controls cannot be encoded as Spine 4.2 timelines."""

    def __init__(self, message: str) -> None:
        super().__init__(f"invalid_spine_animation: {message}")


def _error(message: str) -> SpineAnimationError:
    return SpineAnimationError(message)


def _clean(value: float) -> float:
    if not math.isfinite(value):
        raise _error("timeline contains a non-finite number")
    rounded = round(float(value), 12)
    return 0.0 if abs(rounded) < 1e-12 else rounded


@dataclass(frozen=True, slots=True)
class SpineAnimationRecord:
    preset_id: str
    source_kind: str
    source_id: str
    artifact_export_name: str
    required: bool
    selected_binding_ids: tuple[str, ...]
    control_ids: tuple[str, ...]
    target_count: int
    animation_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "preset_id": self.preset_id,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "artifact_export_name": self.artifact_export_name,
            "required": self.required,
            "selected_binding_ids": list(self.selected_binding_ids),
            "control_ids": list(self.control_ids),
            "target_count": self.target_count,
            "animation_sha256": self.animation_sha256,
        }


@dataclass(frozen=True, slots=True)
class SpineAnimationPlan:
    schema_version: str
    expression_hold_version: str
    sample_rate_hz: int
    rig_document_sha256: str
    coordinate_plan_sha256: str
    bind_plan_sha256: str
    symbol_view_sha256: str
    setup_document_sha256: str
    format_plan_sha256: str
    control_binding_set_sha256: str
    clip_set_sha256: str
    expression_set_sha256: str
    animations: dict[str, object]
    records: tuple[SpineAnimationRecord, ...]
    animation_set_sha256: str
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "expression_hold_version": self.expression_hold_version,
            "sample_rate_hz": self.sample_rate_hz,
            "rig_document_sha256": self.rig_document_sha256,
            "coordinate_plan_sha256": self.coordinate_plan_sha256,
            "bind_plan_sha256": self.bind_plan_sha256,
            "symbol_view_sha256": self.symbol_view_sha256,
            "setup_document_sha256": self.setup_document_sha256,
            "format_plan_sha256": self.format_plan_sha256,
            "control_binding_set_sha256": self.control_binding_set_sha256,
            "clip_set_sha256": self.clip_set_sha256,
            "expression_set_sha256": self.expression_set_sha256,
            "animations": self.animations,
            "records": [record.to_dict() for record in self.records],
            "animation_set_sha256": self.animation_set_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "plan_sha256": self.plan_sha256}


def _require_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _error(f"{field} must be an object")
    return value


def _require_list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise _error(f"{field} must be a list")
    return value


def _number(value: object, *, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise _error(f"{field} must be numeric")
    return _clean(float(value))


def _control_value(keys: Sequence[Mapping[str, object]], frame: float) -> float:
    if not keys:
        raise _error("control curve has no keys")
    points = [
        (
            _number(key.get("frame"), field="control frame"),
            _number(key.get("value"), field="control value"),
        )
        for key in keys
    ]
    if points != sorted(points) or len({point[0] for point in points}) != len(points):
        raise _error("control curve keys are not strictly ordered")
    if frame < points[0][0] - 1e-9 or frame > points[-1][0] + 1e-9:
        raise _error("timeline frame is outside its control curve")
    for key_frame, value in points:
        if abs(frame - key_frame) <= 1e-9:
            return value
    for (left_frame, left_value), (right_frame, right_value) in zip(
        points, points[1:], strict=False
    ):
        if left_frame < frame < right_frame:
            ratio = (frame - left_frame) / (right_frame - left_frame)
            return _clean(left_value + (right_value - left_value) * ratio)
    raise _error("control curve cannot evaluate the requested frame")


def _transfer_output(binding: Mapping[str, object], control_value: float) -> tuple[float, ...]:
    transfer = _require_mapping(binding.get("transfer"), field="binding transfer")
    kind = transfer.get("kind")
    control_default = _number(
        transfer.get("control_default"), field="transfer control_default"
    )
    output_default = tuple(
        _number(value, field="transfer default output")
        for value in _require_list(
            transfer.get("output_at_default"), field="transfer output_at_default"
        )
    )
    if kind in {"affine_scalar", "affine_vector"}:
        gain = tuple(
            _number(value, field="transfer gain")
            for value in _require_list(transfer.get("gain"), field="transfer gain")
        )
        if not output_default or len(gain) != len(output_default):
            raise _error("affine transfer dimensions differ")
        return tuple(
            _clean(default + scale * (control_value - control_default))
            for default, scale in zip(output_default, gain, strict=True)
        )
    if kind not in {"sampled_property", "sampled_deform"}:
        raise _error(f"unsupported transfer kind: {kind}")
    raw_samples = _require_list(transfer.get("samples"), field="transfer samples")
    samples = []
    for raw in raw_samples:
        sample = _require_mapping(raw, field="transfer sample")
        samples.append(
            (
                _number(sample.get("input_value"), field="sample input"),
                tuple(
                    _number(value, field="sample output")
                    for value in _require_list(
                        sample.get("output_values"), field="sample outputs"
                    )
                ),
            )
        )
    if not samples or samples != sorted(samples) or len({item[0] for item in samples}) != len(samples):
        raise _error("sampled transfer points are not strictly ordered")
    if any(len(output) != len(output_default) for _value, output in samples):
        raise _error("sampled transfer output dimensions differ")
    for sample_input, output in samples:
        if abs(control_value - sample_input) <= 1e-9:
            return output
    for (left_input, left_output), (right_input, right_output) in zip(
        samples, samples[1:], strict=False
    ):
        if left_input < control_value < right_input:
            ratio = (control_value - left_input) / (right_input - left_input)
            return tuple(
                _clean(left + (right - left) * ratio)
                for left, right in zip(left_output, right_output, strict=True)
            )
    raise _error("control value lies outside the sampled transfer domain")


def _binding_frames(
    binding: Mapping[str, object],
    keys: Sequence[Mapping[str, object]],
) -> tuple[float, ...]:
    points = [
        (
            _number(key.get("frame"), field="control frame"),
            _number(key.get("value"), field="control value"),
        )
        for key in keys
    ]
    frames = {point[0] for point in points}
    transfer = _require_mapping(binding.get("transfer"), field="binding transfer")
    if transfer.get("kind") not in {"sampled_property", "sampled_deform"}:
        return tuple(sorted(frames))
    knots = tuple(
        _number(
            _require_mapping(raw, field="transfer sample").get("input_value"),
            field="sample input",
        )
        for raw in _require_list(transfer.get("samples"), field="transfer samples")
    )
    for (left_frame, left_value), (right_frame, right_value) in zip(
        points, points[1:], strict=False
    ):
        if left_value == right_value:
            continue
        low, high = sorted((left_value, right_value))
        for knot in knots:
            if low < knot < high:
                ratio = (knot - left_value) / (right_value - left_value)
                frames.add(_clean(left_frame + (right_frame - left_frame) * ratio))
    return tuple(sorted(frames))


def _group_frames(
    bindings: Sequence[Mapping[str, object]],
    curves: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[float, ...]:
    frames: set[float] = set()
    for binding in bindings:
        control_id = binding.get("control_id")
        if not isinstance(control_id, str) or control_id not in curves:
            raise _error("selected binding lacks a selected control curve")
        frames.update(_binding_frames(binding, curves[control_id]))
    if not frames:
        raise _error("timeline target has no frames")
    return tuple(sorted(frames))


def _output_at_frame(
    binding: Mapping[str, object],
    curves: Mapping[str, Sequence[Mapping[str, object]]],
    frame: float,
) -> tuple[float, ...]:
    control_id = binding.get("control_id")
    if not isinstance(control_id, str) or control_id not in curves:
        raise _error("binding references a missing control curve")
    return _transfer_output(binding, _control_value(curves[control_id], frame))


def _time(frame: float) -> float:
    value = frame / SPINE_ANIMATION_SAMPLE_RATE_HZ
    if not math.isfinite(value):
        raise _error("timeline time is not finite")
    return 0.0 if value == 0.0 else value


def _mesh_deform_offsets(
    outputs: tuple[float, ...],
    mesh_bind: SpineMeshBind,
    coordinates: SpineCoordinatePlan,
    bind: SpineBindPlan,
    *,
    weighted: bool,
) -> list[float]:
    if len(outputs) != len(mesh_bind.vertices) * 2:
        raise _error(f"deform output has the wrong topology: {mesh_bind.mesh_id}")
    result: list[float] = []
    for vertex_index, vertex in enumerate(mesh_bind.vertices):
        canvas_point = outputs[vertex_index * 2], outputs[vertex_index * 2 + 1]
        spine_point = canvas_to_spine(coordinates, *canvas_point)
        if weighted:
            for influence in vertex.influences:
                local = spine_local_point(
                    bind,
                    influence.bone_id,
                    spine_point,
                )
                result.extend(
                    (
                        _clean(local[0] - influence.x),
                        _clean(local[1] - influence.y),
                    )
                )
        else:
            if (
                len(vertex.influences) != 1
                or vertex.influences[0].bone_id != mesh_bind.slot_bone_id
            ):
                raise _error("unweighted deform does not use exactly the slot bone")
            influence = vertex.influences[0]
            local = spine_local_point(
                bind,
                influence.bone_id,
                spine_point,
            )
            result.extend(
                (
                    _clean(local[0] - influence.x),
                    _clean(local[1] - influence.y),
                )
            )
    return result


def _compile_animation(
    *,
    bindings: Sequence[Mapping[str, object]],
    curves: Mapping[str, Sequence[Mapping[str, object]]],
    coordinates: SpineCoordinatePlan,
    bind: SpineBindPlan,
    symbols: SpineSymbolView,
    setup: SpineDocument,
) -> tuple[dict[str, object], int]:
    bone_names = {
        bone.bone_id: require_spine_symbol(
            symbols,
            kind="spine_bone",
            source_internal_ids=(bone.bone_id,),
        ).export_name
        for bone in bind.bones
    }
    bind_mesh_by_id = {mesh.mesh_id: mesh for mesh in bind.meshes}
    component_by_mesh = {
        str(record["mesh_id"]): record for record in setup.component_records
    }
    skin_name = require_spine_symbol(
        symbols,
        kind="spine_skin",
        source_internal_ids=("skin/default",),
    ).export_name

    rotations: dict[str, list[Mapping[str, object]]] = {}
    translations: dict[str, list[Mapping[str, object]]] = {}
    deforms: dict[str, list[Mapping[str, object]]] = {}
    opacities: dict[str, list[Mapping[str, object]]] = {}
    for binding in bindings:
        target_id = binding.get("target_id")
        property_name = binding.get("property")
        if not isinstance(target_id, str) or not isinstance(property_name, str):
            raise _error("binding target or property is invalid")
        if property_name == "rotation" and target_id in bone_names:
            rotations.setdefault(target_id, []).append(binding)
        elif property_name in {"translation_x", "translation_y"} and target_id in bone_names:
            translations.setdefault(target_id, []).append(binding)
        elif property_name == "deform" and target_id in bind_mesh_by_id:
            deforms.setdefault(target_id, []).append(binding)
        elif property_name == "opacity" and target_id in bind_mesh_by_id:
            opacities.setdefault(target_id, []).append(binding)
        else:
            raise _error(f"unsupported or mismatched binding target: {target_id} {property_name}")

    animation: dict[str, object] = {}
    bone_payload: dict[str, dict[str, object]] = {}
    for bone_id in sorted(rotations):
        group = rotations[bone_id]
        keys = []
        for frame in _group_frames(group, curves):
            value = sum(_output_at_frame(item, curves, frame)[0] for item in group)
            keys.append({"time": _time(frame), "value": _clean(-value)})
        bone_payload.setdefault(bone_names[bone_id], {})["rotate"] = keys
    for bone_id in sorted(translations):
        group = translations[bone_id]
        keys = []
        for frame in _group_frames(group, curves):
            canvas_x = 0.0
            canvas_y = 0.0
            for item in group:
                value = _output_at_frame(item, curves, frame)[0]
                if item["property"] == "translation_x":
                    canvas_x += value
                else:
                    canvas_y += value
            local_x, local_y = spine_parent_local_vector(
                bind, bone_id, (_clean(canvas_x), _clean(-canvas_y))
            )
            keys.append({"time": _time(frame), "x": local_x, "y": local_y})
        bone_payload.setdefault(bone_names[bone_id], {})["translate"] = keys
    if bone_payload:
        animation["bones"] = {
            name: bone_payload[name] for name in sorted(bone_payload)
        }

    if any(len(group) != 1 for group in deforms.values()):
        raise _error("a non-rigid target has multiple selected control drivers")
    attachment_payload: dict[str, dict[str, dict[str, object]]] = {}
    for mesh_id in sorted(deforms):
        binding = deforms[mesh_id][0]
        mesh_bind = bind_mesh_by_id[mesh_id]
        component = component_by_mesh.get(mesh_id)
        if component is None:
            raise _error(f"deform target lacks a setup component: {mesh_id}")
        weighted = component.get("weighted")
        if not isinstance(weighted, bool):
            raise _error("setup component lacks weighted state")
        default_output = _transfer_output(
            binding,
            _number(
                _require_mapping(
                    binding.get("transfer"), field="binding transfer"
                ).get("control_default"),
                field="transfer control_default",
            ),
        )
        default_offsets = _mesh_deform_offsets(
            default_output, mesh_bind, coordinates, bind, weighted=weighted
        )
        if max((abs(value) for value in default_offsets), default=0.0) > 0.1:
            raise _error(f"deform default does not restore setup: {mesh_id}")
        keys = []
        for frame in _group_frames((binding,), curves):
            output = _output_at_frame(binding, curves, frame)
            keys.append(
                {
                    "time": _time(frame),
                    "vertices": _mesh_deform_offsets(
                        output,
                        mesh_bind,
                        coordinates,
                        bind,
                        weighted=weighted,
                    ),
                }
            )
        slot_name = str(component["slot_name"])
        attachment_name = str(component["attachment_key_name"])
        attachment_payload.setdefault(skin_name, {}).setdefault(
            slot_name, {}
        )[attachment_name] = {"deform": keys}
    if attachment_payload:
        animation["attachments"] = attachment_payload

    if any(len(group) != 1 for group in opacities.values()):
        raise _error("a slot opacity target has multiple selected control drivers")
    slot_payload: dict[str, dict[str, object]] = {}
    for mesh_id in sorted(opacities):
        binding = opacities[mesh_id][0]
        component = component_by_mesh.get(mesh_id)
        if component is None:
            raise _error(f"opacity target lacks a setup component: {mesh_id}")
        keys = []
        for frame in _group_frames((binding,), curves):
            output = _output_at_frame(binding, curves, frame)
            if len(output) != 1 or not 0.0 <= output[0] <= 1.0:
                raise _error("slot opacity lies outside [0, 1]")
            keys.append({"time": _time(frame), "value": output[0]})
        slot_payload[str(component["slot_name"])] = {"alpha": keys}
    if slot_payload:
        animation["slots"] = slot_payload
    if not animation:
        raise _error("supported preset produced no visible timeline")
    target_count = sum(
        len(group) for group in (rotations, translations, deforms, opacities)
    )
    return animation, target_count


def _contains_curve(value: object) -> bool:
    if isinstance(value, Mapping):
        return "curve" in value or any(_contains_curve(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_curve(child) for child in value)
    return False


def _validate_setup(
    setup: SpineDocument,
    rig: RigDocument,
    coordinates: SpineCoordinatePlan,
    bind: SpineBindPlan,
    symbols: SpineSymbolView,
) -> None:
    if not isinstance(setup, SpineDocument) or setup.schema_version != SPINE_DOCUMENT_VERSION:
        raise _error("setup document has the wrong type or version")
    if (
        setup.rig_document_sha256 != rig.document_sha256
        or setup.coordinate_plan_sha256 != coordinates.plan_sha256
        or setup.bind_plan_sha256 != bind.plan_sha256
        or setup.symbol_view_sha256 != symbols.view_sha256
    ):
        raise _error("setup document references different Stage C or bind facts")
    if setup.to_dict().get("animations") != {}:
        raise _error("animation compiler requires an animation-free setup document")


def build_spine_animation_plan(
    rig: RigDocument,
    coordinates: SpineCoordinatePlan,
    bind: SpineBindPlan,
    symbols: SpineSymbolView,
    setup: SpineDocument,
) -> SpineAnimationPlan:
    validate_rig_document(rig)
    validate_spine_coordinate_plan(coordinates)
    validate_spine_symbol_view(symbols)
    payload = rig.to_dict()
    validate_spine_bind_plan(bind, payload["bones"], payload["meshes"], coordinates)
    _validate_setup(setup, rig, coordinates, bind, symbols)

    format_plans = _require_mapping(payload["format_plans"], field="format_plans")
    set_plans = _require_list(
        format_plans.get("preset_set_plans"), field="format preset-set plans"
    )
    spine_sets = [
        _require_mapping(item, field="format preset-set plan")
        for item in set_plans
        if isinstance(item, Mapping) and item.get("format_id") == "spine_4_2"
    ]
    if len(spine_sets) != 1:
        raise _error("Rig must contain exactly one Spine 4.2 preset-set plan")
    spine_set = spine_sets[0]
    decisions = _require_list(spine_set.get("decisions"), field="Spine decisions")
    binding_records = _require_list(
        payload["control_bindings"], field="control bindings"
    )
    bindings_by_id = {}
    for raw in binding_records:
        binding = _require_mapping(raw, field="control binding")
        binding_id = binding.get("binding_id")
        if not isinstance(binding_id, str) or binding_id in bindings_by_id:
            raise _error("control binding IDs are invalid or duplicated")
        bindings_by_id[binding_id] = binding
    clips = {
        str(item["preset_id"]): item
        for raw in _require_list(payload["clips"], field="clips")
        for item in (_require_mapping(raw, field="clip"),)
    }
    expressions = {
        str(item["preset_id"]): item
        for raw in _require_list(payload["expressions"], field="expressions")
        for item in (_require_mapping(raw, field="expression"),)
    }

    animations: dict[str, object] = {}
    records = []
    for raw_decision in decisions:
        decision = _require_mapping(raw_decision, field="Spine decision")
        if decision.get("status") != "supported":
            continue
        preset_id = decision.get("preset_id")
        artifact_name = decision.get("artifact_export_name")
        selected_ids = decision.get("selected_binding_ids")
        if (
            not isinstance(preset_id, str)
            or not isinstance(artifact_name, str)
            or not artifact_name
            or not isinstance(selected_ids, list)
            or not selected_ids
            or any(not isinstance(value, str) for value in selected_ids)
        ):
            raise _error("supported decision lacks an artifact or selected bindings")
        if artifact_name in animations:
            raise _error("two supported presets share one animation artifact")
        selected_bindings = []
        for binding_id in selected_ids:
            binding = bindings_by_id.get(binding_id)
            if binding is None:
                raise _error(f"selected binding is absent: {binding_id}")
            selected_bindings.append(binding)
        if preset_id in clips:
            source_kind = "clip"
            source = clips[preset_id]
            source_id = source.get("clip_id")
            if source.get("sample_rate_hz") != SPINE_ANIMATION_SAMPLE_RATE_HZ:
                raise _error("Spine v1 only accepts 30 Hz MotionClip input")
            raw_curves = _require_list(
                source.get("control_curves"), field="clip control curves"
            )
            curves = {}
            for raw_curve in raw_curves:
                curve = _require_mapping(raw_curve, field="control curve")
                control_id = curve.get("control_id")
                if not isinstance(control_id, str) or control_id in curves:
                    raise _error("clip control IDs are invalid or duplicated")
                curves[control_id] = tuple(
                    _require_mapping(key, field="control key")
                    for key in _require_list(curve.get("keys"), field="control keys")
                )
        elif preset_id in expressions:
            source_kind = "expression"
            source = expressions[preset_id]
            source_id = source.get("expression_id")
            if source.get("application_mode") != "overwrite_full_weight":
                raise _error("Spine expressions require full-weight overwrite")
            curves = {}
            for raw_value in _require_list(
                source.get("values"), field="expression values"
            ):
                value = _require_mapping(raw_value, field="expression value")
                control_id = value.get("control_id")
                if not isinstance(control_id, str) or control_id in curves:
                    raise _error("expression control IDs are invalid or duplicated")
                absolute = _number(
                    value.get("absolute_value"), field="expression absolute value"
                )
                curves[control_id] = (
                    {"frame": 0, "value": absolute},
                    {"frame": 1, "value": absolute},
                )
        else:
            raise _error(f"supported preset has no source descriptor: {preset_id}")
        if not isinstance(source_id, str):
            raise _error("preset source identity is invalid")
        selected_control_ids = {str(binding.get("control_id")) for binding in selected_bindings}
        if selected_control_ids != set(curves):
            raise _error("selected binding controls differ from the preset descriptor")
        artifact_symbol = require_spine_symbol(
            symbols,
            kind="spine_animation",
            source_internal_ids=(source_id,),
            preset_id=source_id,
        )
        if (
            artifact_symbol.export_name != artifact_name
            or artifact_symbol.symbol_id != decision.get("artifact_symbol_id")
        ):
            raise _error("animation artifact differs from the frozen global symbol")
        animation, target_count = _compile_animation(
            bindings=selected_bindings,
            curves=curves,
            coordinates=coordinates,
            bind=bind,
            symbols=symbols,
            setup=setup,
        )
        if _contains_curve(animation):
            raise _error("Spine linear timelines must omit the curve field")
        animations[artifact_name] = animation
        records.append(
            SpineAnimationRecord(
                preset_id=preset_id,
                source_kind=source_kind,
                source_id=source_id,
                artifact_export_name=artifact_name,
                required=bool(decision.get("required")),
                selected_binding_ids=tuple(selected_ids),
                control_ids=tuple(sorted(curves)),
                target_count=target_count,
                animation_sha256=jcs_sha256(animation),
            )
        )
    if not animations:
        raise _error("Spine preset-set plan has no supported artifacts")
    animations = {name: animations[name] for name in sorted(animations)}
    records_tuple = tuple(sorted(records, key=lambda item: item.preset_id))
    provisional = SpineAnimationPlan(
        schema_version=SPINE_ANIMATION_PLAN_VERSION,
        expression_hold_version=SPINE_EXPRESSION_HOLD_VERSION,
        sample_rate_hz=SPINE_ANIMATION_SAMPLE_RATE_HZ,
        rig_document_sha256=rig.document_sha256,
        coordinate_plan_sha256=coordinates.plan_sha256,
        bind_plan_sha256=bind.plan_sha256,
        symbol_view_sha256=symbols.view_sha256,
        setup_document_sha256=setup.document_sha256,
        format_plan_sha256=str(format_plans["plan_sha256"]),
        control_binding_set_sha256=jcs_sha256(binding_records),
        clip_set_sha256=jcs_sha256(payload["clips"]),
        expression_set_sha256=jcs_sha256(payload["expressions"]),
        animations=animations,
        records=records_tuple,
        animation_set_sha256=jcs_sha256(animations),
        plan_sha256="",
    )
    return replace(
        provisional,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def validate_spine_animation_plan(
    plan: SpineAnimationPlan,
    rig: RigDocument,
    coordinates: SpineCoordinatePlan,
    bind: SpineBindPlan,
    symbols: SpineSymbolView,
    setup: SpineDocument,
) -> SpineAnimationPlan:
    if not isinstance(plan, SpineAnimationPlan):
        raise _error("animation plan has the wrong type")
    if (
        plan.schema_version != SPINE_ANIMATION_PLAN_VERSION
        or plan.expression_hold_version != SPINE_EXPRESSION_HOLD_VERSION
        or plan.sample_rate_hz != SPINE_ANIMATION_SAMPLE_RATE_HZ
    ):
        raise _error("animation plan version is unsupported")
    if _contains_curve(plan.animations):
        raise _error("Spine linear timelines must omit the curve field")
    if plan.animation_set_sha256 != jcs_sha256(plan.animations):
        raise _error("animation-set digest mismatch")
    if plan.plan_sha256 != jcs_sha256(plan.semantic_payload()):
        raise _error("animation plan digest mismatch")
    expected = build_spine_animation_plan(rig, coordinates, bind, symbols, setup)
    if plan != expected:
        raise _error("animation plan differs from recomputed Stage C decisions")
    return plan


__all__ = [
    "SPINE_ANIMATION_PLAN_VERSION",
    "SPINE_ANIMATION_SAMPLE_RATE_HZ",
    "SPINE_EXPRESSION_HOLD_VERSION",
    "SpineAnimationError",
    "SpineAnimationPlan",
    "SpineAnimationRecord",
    "build_spine_animation_plan",
    "validate_spine_animation_plan",
]
