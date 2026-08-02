from __future__ import annotations

import math
from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping, Sequence

from ...jcs import jcs_sha256
from ...rig_document import RigDocument
from .artmesh import Live2DArtMeshPlan
from .binding_plan import Live2DBindingPlan
from .coordinates import Live2DCoordinatePlan, live2d_local_to_canvas
from .document import (
    LIVE2D_DRAW_ORDER_BASE,
    LIVE2D_MOC3_MAX_BYTES,
    build_live2d_moc3_document,
)
from .frame_kernel import apply_similarity
from .keyforms import Live2DKeyformPlan
from .moc3_codec import (
    Moc3V400Document,
    decode_moc3_v400,
    encode_moc3_v400,
)

LIVE2D_MOC3_VALIDATOR_VERSION = "live2d-moc3-validator-v1"
LIVE2D_STRUCTURE_REPORT_VERSION = "live2d-structure-report-v1"


class Live2DMoc3ValidationError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(f"invalid_live2d_moc3_payload: {message}")


def _error(message: str) -> Live2DMoc3ValidationError:
    return Live2DMoc3ValidationError(message)


def _ints(document: Moc3V400Document, name: str) -> tuple[int, ...]:
    values = document.section(name)
    if any(type(value) is not int for value in values):
        raise _error(f"{name} must contain integers")
    return tuple(int(value) for value in values)


def _floats(document: Moc3V400Document, name: str) -> tuple[float, ...]:
    values = document.section(name)
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in values
    ):
        raise _error(f"{name} must contain finite numbers")
    return tuple(float(value) for value in values)


def _strings(document: Moc3V400Document, name: str) -> tuple[str, ...]:
    values = document.section(name)
    if any(not isinstance(value, str) or not value for value in values):
        raise _error(f"{name} must contain non-empty IDs")
    return tuple(str(value) for value in values)


def _lerp(left: float, right: float, amount: float) -> float:
    return left + (right - left) * amount


def _selected_keyform(
    document: Moc3V400Document,
    *,
    band_index: int,
    keyform_begin: int,
    keyform_count: int,
    parameter_values: Sequence[float],
) -> tuple[int, int, float]:
    band_begins = _ints(document, "keyform_binding_band.begin_indices")
    band_counts = _ints(document, "keyform_binding_band.counts")
    if not 0 <= band_index < len(band_begins):
        raise _error("keyform binding band index is out of range")
    association_begin = band_begins[band_index]
    association_count = band_counts[band_index]
    if association_count == 0:
        if keyform_count != 1:
            raise _error("unbound object must have exactly one static keyform")
        return keyform_begin, keyform_begin, 0.0
    if association_count != 1:
        raise _error("v1 object bands must contain exactly one binding")
    associations = _ints(document, "keyform_binding_index.indices")
    if not 0 <= association_begin < len(associations):
        raise _error("keyform binding association is out of range")
    binding_index = associations[association_begin]
    key_begins = _ints(document, "keyform_binding.keys_begin_indices")
    key_counts = _ints(document, "keyform_binding.keys_counts")
    if not 0 <= binding_index < len(key_begins):
        raise _error("keyform binding index is out of range")
    key_begin = key_begins[binding_index]
    key_count = key_counts[binding_index]
    if key_count != keyform_count or key_count <= 0:
        raise _error("binding key count differs from object keyform count")
    keys = _floats(document, "keys.values")
    if key_begin < 0 or key_begin + key_count > len(keys):
        raise _error("binding key span is out of range")

    parameter_index = None
    parameter_begins = _ints(
        document, "parameter.keyform_binding_begin_indices"
    )
    parameter_counts = _ints(document, "parameter.keyform_binding_counts")
    for index, (begin, count) in enumerate(
        zip(parameter_begins, parameter_counts, strict=True)
    ):
        if begin < 0 or begin + count > len(key_begins):
            raise _error("parameter binding span is out of range")
        if begin <= binding_index < begin + count:
            if parameter_index is not None:
                raise _error("binding belongs to multiple parameters")
            parameter_index = index
    if parameter_index is None or parameter_index >= len(parameter_values):
        raise _error("binding has no parameter owner")
    value = parameter_values[parameter_index]
    binding_keys = keys[key_begin : key_begin + key_count]
    if any(right <= left for left, right in zip(binding_keys, binding_keys[1:])):
        raise _error("binding keys are not strictly increasing")
    if value < binding_keys[0] or value > binding_keys[-1]:
        raise _error("parameter value lies outside its binding keys")
    for index, key in enumerate(binding_keys):
        if abs(value - key) <= 1e-7:
            selected = keyform_begin + index
            return selected, selected, 0.0
    for index, (left, right) in enumerate(zip(binding_keys, binding_keys[1:])):
        if left < value < right:
            return (
                keyform_begin + index,
                keyform_begin + index + 1,
                (value - left) / (right - left),
            )
    raise _error("parameter value has no interpolation interval")


@dataclass(frozen=True, slots=True)
class Live2DEvaluatedArtMesh:
    artmesh_id: str
    canvas_positions: tuple[tuple[float, float], ...]
    opacity: float
    draw_order: float


@dataclass(frozen=True, slots=True)
class Live2DMoc3State:
    parameters: tuple[tuple[str, float], ...]
    artmeshes: tuple[Live2DEvaluatedArtMesh, ...]


@dataclass(frozen=True, slots=True)
class Live2DParameterTargetClosure:
    parameter_id: str
    parameter_export_name: str
    target_export_names: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "parameter_id": self.parameter_id,
            "parameter_export_name": self.parameter_export_name,
            "target_export_names": list(self.target_export_names),
        }


@dataclass(frozen=True, slots=True)
class Live2DStructureValidationReport:
    schema_version: str
    validator_version: str
    moc_sha256: str
    moc_file_size: int
    rig_document_sha256: str
    binding_plan_sha256: str
    coordinate_plan_sha256: str
    artmesh_plan_sha256: str
    keyform_plan_sha256: str
    counts: tuple[int, ...]
    texture_page_count: int
    total_baked_vertex_positions: int
    maximum_default_position_residual: float
    maximum_default_opacity_residual: float
    emitted_part_ids: tuple[str, ...]
    emitted_deformer_ids: tuple[str, ...]
    emitted_artmesh_ids: tuple[str, ...]
    emitted_parameter_ids: tuple[str, ...]
    pruned_bone_ids: tuple[str, ...]
    parameter_target_closure: tuple[Live2DParameterTargetClosure, ...]
    report_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "validator_version": self.validator_version,
            "moc_sha256": self.moc_sha256,
            "moc_file_size": self.moc_file_size,
            "rig_document_sha256": self.rig_document_sha256,
            "binding_plan_sha256": self.binding_plan_sha256,
            "coordinate_plan_sha256": self.coordinate_plan_sha256,
            "artmesh_plan_sha256": self.artmesh_plan_sha256,
            "keyform_plan_sha256": self.keyform_plan_sha256,
            "counts": list(self.counts),
            "texture_page_count": self.texture_page_count,
            "total_baked_vertex_positions": self.total_baked_vertex_positions,
            "maximum_default_position_residual": (
                self.maximum_default_position_residual
            ),
            "maximum_default_opacity_residual": (
                self.maximum_default_opacity_residual
            ),
            "emitted_part_ids": list(self.emitted_part_ids),
            "emitted_deformer_ids": list(self.emitted_deformer_ids),
            "emitted_artmesh_ids": list(self.emitted_artmesh_ids),
            "emitted_parameter_ids": list(self.emitted_parameter_ids),
            "pruned_bone_ids": list(self.pruned_bone_ids),
            "parameter_target_closure": [
                record.to_dict() for record in self.parameter_target_closure
            ],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "report_sha256": self.report_sha256}


@dataclass(frozen=True, slots=True)
class _RotationState:
    origin_x: float
    origin_y: float
    angle: float
    scale: float


def _parameter_state(
    document: Moc3V400Document,
    overrides: Mapping[str, float] | None,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    ids = _strings(document, "parameter.ids")
    minimums = _floats(document, "parameter.min_values")
    defaults = _floats(document, "parameter.default_values")
    maximums = _floats(document, "parameter.max_values")
    if not (len(ids) == len(minimums) == len(defaults) == len(maximums)):
        raise _error("parameter section lengths differ")
    values = list(defaults)
    index_by_id = {identity: index for index, identity in enumerate(ids)}
    if len(index_by_id) != len(ids):
        raise _error("parameter IDs are duplicated")
    if overrides:
        for identity, raw_value in overrides.items():
            index = index_by_id.get(identity)
            if index is None:
                raise _error(f"unknown parameter override: {identity}")
            if (
                not isinstance(raw_value, (int, float))
                or isinstance(raw_value, bool)
                or not math.isfinite(float(raw_value))
            ):
                raise _error("parameter override is not finite")
            value = float(raw_value)
            if value < minimums[index] or value > maximums[index]:
                raise _error("parameter override lies outside min/max")
            values[index] = value
    return ids, tuple(values)


def _rotation_states(
    document: Moc3V400Document,
    parameter_values: tuple[float, ...],
) -> tuple[_RotationState, ...]:
    bands = _ints(
        document, "rotation_deformer.keyform_binding_band_indices"
    )
    begins = _ints(document, "rotation_deformer.keyform_begin_indices")
    counts = _ints(document, "rotation_deformer.keyform_counts")
    angles = _floats(document, "rotation_deformer_keyform.angles")
    origins_x = _floats(document, "rotation_deformer_keyform.origin_xs")
    origins_y = _floats(document, "rotation_deformer_keyform.origin_ys")
    scales = _floats(document, "rotation_deformer_keyform.scales")
    reflects_x = document.section("rotation_deformer_keyform.reflect_xs")
    reflects_y = document.section("rotation_deformer_keyform.reflect_ys")
    if any(reflects_x) or any(reflects_y):
        raise _error("rotation_deformer_keyform.reflect_xs/ys must be false")
    if not (
        len(angles)
        == len(origins_x)
        == len(origins_y)
        == len(scales)
        == len(reflects_x)
        == len(reflects_y)
    ):
        raise _error("RotationDeformer keyform section lengths differ")
    result = []
    for band, begin, count in zip(bands, begins, counts, strict=True):
        left, right, amount = _selected_keyform(
            document,
            band_index=band,
            keyform_begin=begin,
            keyform_count=count,
            parameter_values=parameter_values,
        )
        result.append(
            _RotationState(
                origin_x=_lerp(origins_x[left], origins_x[right], amount),
                origin_y=_lerp(origins_y[left], origins_y[right], amount),
                angle=_lerp(angles[left], angles[right], amount),
                scale=_lerp(scales[left], scales[right], amount),
            )
        )
    return tuple(result)


def _to_canvas(
    document: Moc3V400Document, point: tuple[float, float]
) -> tuple[float, float]:
    return (
        point[0] * document.canvas.pixels_per_unit + document.canvas.origin_x,
        point[1] * document.canvas.pixels_per_unit + document.canvas.origin_y,
    )


def evaluate_live2d_moc3_state(
    document: Moc3V400Document,
    parameter_overrides: Mapping[str, float] | None = None,
) -> Live2DMoc3State:
    if not isinstance(document, Moc3V400Document):
        raise _error("state evaluator requires a typed MOC3 document")
    # The codec is also the self-contained section/count/type validator.
    encode_moc3_v400(document)
    parameter_ids, parameter_values = _parameter_state(
        document, parameter_overrides
    )
    rotations = _rotation_states(document, parameter_values)
    generic_types = _ints(document, "deformer.types")
    generic_specific = _ints(document, "deformer.specific_indices")
    generic_parents = _ints(document, "deformer.parent_deformer_indices")
    if not (
        len(generic_types) == len(generic_specific) == len(generic_parents)
    ):
        raise _error("generic deformer section lengths differ")
    if any(kind != 1 for kind in generic_types):
        raise _error("v1 production document may contain only RotationDeformers")
    if any(index < 0 or index >= len(rotations) for index in generic_specific):
        raise _error("deformer specific index is out of range")

    ids = _strings(document, "art_mesh.ids")
    bands = _ints(document, "art_mesh.keyform_binding_band_indices")
    begins = _ints(document, "art_mesh.keyform_begin_indices")
    counts = _ints(document, "art_mesh.keyform_counts")
    parent_deformers = _ints(document, "art_mesh.parent_deformer_indices")
    vertex_counts = _ints(document, "art_mesh.position_index_counts")
    key_positions = _ints(
        document, "art_mesh_keyform.keyform_position_begin_indices"
    )
    positions = _floats(document, "keyform_position.xys")
    opacities = _floats(document, "art_mesh_keyform.opacities")
    draw_orders = _floats(document, "art_mesh_keyform.draw_orders")
    if not (
        len(ids)
        == len(bands)
        == len(begins)
        == len(counts)
        == len(parent_deformers)
        == len(vertex_counts)
    ):
        raise _error("ArtMesh object section lengths differ")
    result = []
    for (
        identity,
        band,
        begin,
        count,
        parent,
        vertex_count,
    ) in zip(
        ids,
        bands,
        begins,
        counts,
        parent_deformers,
        vertex_counts,
        strict=True,
    ):
        left, right, amount = _selected_keyform(
            document,
            band_index=band,
            keyform_begin=begin,
            keyform_count=count,
            parameter_values=parameter_values,
        )
        if not (0 <= left < len(key_positions) and 0 <= right < len(key_positions)):
            raise _error("ArtMesh keyform index is out of range")
        left_begin = key_positions[left]
        right_begin = key_positions[right]
        if (
            vertex_count <= 0
            or left_begin < 0
            or right_begin < 0
            or left_begin + vertex_count * 2 > len(positions)
            or right_begin + vertex_count * 2 > len(positions)
        ):
            raise _error("ArtMesh keyform position span is out of range")
        local_points = tuple(
            (
                _lerp(
                    positions[left_begin + index * 2],
                    positions[right_begin + index * 2],
                    amount,
                ),
                _lerp(
                    positions[left_begin + index * 2 + 1],
                    positions[right_begin + index * 2 + 1],
                    amount,
                ),
            )
            for index in range(vertex_count)
        )
        transformed = []
        for local in local_points:
            point = local
            cursor = parent
            seen: set[int] = set()
            while cursor != -1:
                if cursor in seen or not 0 <= cursor < len(generic_parents):
                    raise _error("deformer parent chain is cyclic or out of range")
                seen.add(cursor)
                rotation = rotations[generic_specific[cursor]]
                point = apply_similarity(
                    point,
                    origin=(rotation.origin_x, rotation.origin_y),
                    angle_degrees=rotation.angle,
                    scale=rotation.scale,
                )
                cursor = generic_parents[cursor]
            transformed.append(_to_canvas(document, point))
        result.append(
            Live2DEvaluatedArtMesh(
                artmesh_id=identity,
                canvas_positions=tuple(transformed),
                opacity=_lerp(opacities[left], opacities[right], amount),
                draw_order=_lerp(draw_orders[left], draw_orders[right], amount),
            )
        )
    return Live2DMoc3State(
        parameters=tuple(zip(parameter_ids, parameter_values, strict=True)),
        artmeshes=tuple(result),
    )


def _compare_document(
    actual: Moc3V400Document, expected: Moc3V400Document
) -> None:
    if actual.counts != expected.counts:
        raise _error("count_info differs from the canonical compiler output")
    if actual.canvas != expected.canvas:
        raise _error("canvas info differs from the canonical compiler output")
    for name, expected_values in expected.sections.items():
        if actual.section(name) != expected_values:
            raise _error(f"section {name} differs from the canonical compiler output")


def validate_live2d_moc3_document(
    document: Moc3V400Document,
    rig: RigDocument,
    bindings: Live2DBindingPlan,
    coordinates: Live2DCoordinatePlan,
    artmeshes: Live2DArtMeshPlan,
    keyforms: Live2DKeyformPlan,
) -> Moc3V400Document:
    if not isinstance(document, Moc3V400Document):
        raise _error("validator requires a typed MOC3 document")
    encoded = encode_moc3_v400(document)
    if len(encoded) > LIVE2D_MOC3_MAX_BYTES:
        raise _error("MOC3 exceeds the 64 MiB capacity guard")
    expected = build_live2d_moc3_document(
        rig, bindings, coordinates, artmeshes, keyforms
    )
    _compare_document(document, expected)
    state = evaluate_live2d_moc3_state(document)
    state_by_name = {record.artmesh_id: record for record in state.artmeshes}
    if len(state_by_name) != len(artmeshes.artmeshes):
        raise _error("evaluated ArtMesh IDs are duplicated")
    for setup in artmeshes.artmeshes:
        actual = state_by_name.get(setup.export_name)
        if actual is None:
            raise _error("evaluated state lacks a canonical ArtMesh")
        expected_points = tuple(
            live2d_local_to_canvas(
                coordinates, setup.parent_instance_id, point
            )
            for point in setup.positions
        )
        residual = max(
            (
                math.hypot(left[0] - right[0], left[1] - right[1])
                for left, right in zip(
                    actual.canvas_positions, expected_points, strict=True
                )
            ),
            default=0.0,
        )
        if residual > 0.1:
            raise _error("default parameter state does not reproduce rest geometry")
        if abs(actual.opacity - setup.setup_opacity) > 1 / 255:
            raise _error("default parameter state does not reproduce rest opacity")
        if actual.draw_order != LIVE2D_DRAW_ORDER_BASE + setup.draw_order:
            raise _error("default parameter state does not reproduce draw order")
    return document


def validate_live2d_moc3_payload(
    payload: bytes,
    rig: RigDocument,
    bindings: Live2DBindingPlan,
    coordinates: Live2DCoordinatePlan,
    artmeshes: Live2DArtMeshPlan,
    keyforms: Live2DKeyformPlan,
) -> Moc3V400Document:
    if not isinstance(payload, bytes):
        raise _error("MOC3 payload must be bytes")
    if len(payload) > LIVE2D_MOC3_MAX_BYTES:
        raise _error("MOC3 exceeds the 64 MiB capacity guard")
    document = decode_moc3_v400(payload)
    if encode_moc3_v400(document) != payload:
        raise _error("MOC3 decode/encode bytes are not identical")
    return validate_live2d_moc3_document(
        document, rig, bindings, coordinates, artmeshes, keyforms
    )


def build_live2d_structure_validation_report(
    payload: bytes,
    rig: RigDocument,
    bindings: Live2DBindingPlan,
    coordinates: Live2DCoordinatePlan,
    artmeshes: Live2DArtMeshPlan,
    keyforms: Live2DKeyformPlan,
) -> Live2DStructureValidationReport:
    document = validate_live2d_moc3_payload(
        payload, rig, bindings, coordinates, artmeshes, keyforms
    )
    state = evaluate_live2d_moc3_state(document)
    state_by_name = {record.artmesh_id: record for record in state.artmeshes}
    maximum_position = 0.0
    maximum_opacity = 0.0
    for setup in artmeshes.artmeshes:
        actual = state_by_name[setup.export_name]
        expected = tuple(
            live2d_local_to_canvas(
                coordinates, setup.parent_instance_id, point
            )
            for point in setup.positions
        )
        maximum_position = max(
            maximum_position,
            max(
                (
                    math.hypot(left[0] - right[0], left[1] - right[1])
                    for left, right in zip(
                        actual.canvas_positions, expected, strict=True
                    )
                ),
                default=0.0,
            ),
        )
        maximum_opacity = max(
            maximum_opacity, abs(actual.opacity - setup.setup_opacity)
        )
    target_names: dict[str, set[str]] = {
        parameter.parameter_id: set() for parameter in bindings.parameters
    }
    for binding in bindings.bindings:
        target_names.setdefault(binding.parameter_id, set()).add(
            binding.primitive_export_name
        )
    closure = tuple(
        Live2DParameterTargetClosure(
            parameter_id=parameter.parameter_id,
            parameter_export_name=parameter.export_name,
            target_export_names=tuple(sorted(target_names[parameter.parameter_id])),
        )
        for parameter in bindings.parameters
    )
    if any(not record.target_export_names for record in closure):
        raise _error("parameter-to-visible-target closure is incomplete")
    values = {
        "schema_version": LIVE2D_STRUCTURE_REPORT_VERSION,
        "validator_version": LIVE2D_MOC3_VALIDATOR_VERSION,
        "moc_sha256": f"sha256:{sha256(payload).hexdigest()}",
        "moc_file_size": len(payload),
        "rig_document_sha256": rig.document_sha256,
        "binding_plan_sha256": bindings.plan_sha256,
        "coordinate_plan_sha256": coordinates.plan_sha256,
        "artmesh_plan_sha256": artmeshes.plan_sha256,
        "keyform_plan_sha256": keyforms.plan_sha256,
        "counts": document.counts,
        "texture_page_count": artmeshes.texture_page_count,
        "total_baked_vertex_positions": keyforms.total_baked_vertex_positions,
        "maximum_default_position_residual": maximum_position,
        "maximum_default_opacity_residual": maximum_opacity,
        "emitted_part_ids": _strings(document, "part.ids"),
        "emitted_deformer_ids": _strings(document, "deformer.ids"),
        "emitted_artmesh_ids": _strings(document, "art_mesh.ids"),
        "emitted_parameter_ids": _strings(document, "parameter.ids"),
        "pruned_bone_ids": bindings.pruned_bone_ids,
        "parameter_target_closure": closure,
    }
    provisional = Live2DStructureValidationReport(**values, report_sha256="")
    return Live2DStructureValidationReport(
        **values, report_sha256=jcs_sha256(provisional.semantic_payload())
    )


__all__ = [
    "LIVE2D_MOC3_VALIDATOR_VERSION",
    "LIVE2D_STRUCTURE_REPORT_VERSION",
    "Live2DEvaluatedArtMesh",
    "Live2DMoc3State",
    "Live2DMoc3ValidationError",
    "Live2DParameterTargetClosure",
    "Live2DStructureValidationReport",
    "build_live2d_structure_validation_report",
    "evaluate_live2d_moc3_state",
    "validate_live2d_moc3_document",
    "validate_live2d_moc3_payload",
]
