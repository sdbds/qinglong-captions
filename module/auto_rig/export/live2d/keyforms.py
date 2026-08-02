from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from ...jcs import jcs_sha256
from ...rig_document import RigDocument, validate_rig_document
from .artmesh import Live2DArtMeshPlan, Live2DArtMeshPlanRecord
from .binding_plan import Live2DBindingPlan
from .coordinates import Live2DCoordinatePlan, canvas_to_artmesh_local

LIVE2D_KEYFORM_PLAN_VERSION = "live2d-keyform-plan-v1"
LIVE2D_MAX_STOPS_PER_BINDING = 17
LIVE2D_MAX_BAKED_VERTEX_POSITIONS = 1_000_000


class Live2DKeyformError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(f"invalid_live2d_keyform_plan: {message}")


def _error(message: str) -> Live2DKeyformError:
    return Live2DKeyformError(message)


@dataclass(frozen=True, slots=True)
class Live2DArtMeshKeyforms:
    mesh_id: str
    primitive_target_id: str
    parameter_id: str | None
    parameter_export_name: str | None
    parameter_default: float
    parameter_values: tuple[float, ...]
    positions: tuple[tuple[tuple[float, float], ...], ...]
    opacities: tuple[float, ...]
    has_deform: bool
    has_opacity: bool
    binding_ids: tuple[str, ...]
    maximum_position_residual: float
    maximum_opacity_residual: float

    def to_dict(self) -> dict[str, object]:
        return {
            "mesh_id": self.mesh_id,
            "primitive_target_id": self.primitive_target_id,
            "parameter_id": self.parameter_id,
            "parameter_export_name": self.parameter_export_name,
            "parameter_default": self.parameter_default,
            "parameter_values": list(self.parameter_values),
            "positions": [
                [list(point) for point in keyform] for keyform in self.positions
            ],
            "opacities": list(self.opacities),
            "has_deform": self.has_deform,
            "has_opacity": self.has_opacity,
            "binding_ids": list(self.binding_ids),
            "maximum_position_residual": self.maximum_position_residual,
            "maximum_opacity_residual": self.maximum_opacity_residual,
        }


@dataclass(frozen=True, slots=True)
class Live2DKeyformPlan:
    schema_version: str
    rig_document_sha256: str
    binding_plan_sha256: str
    coordinate_plan_sha256: str
    artmesh_plan_sha256: str
    max_stops_per_binding: int
    max_baked_vertex_positions: int
    total_baked_vertex_positions: int
    artmesh_keyforms: tuple[Live2DArtMeshKeyforms, ...]
    maximum_position_residual: float
    maximum_opacity_residual: float
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "rig_document_sha256": self.rig_document_sha256,
            "binding_plan_sha256": self.binding_plan_sha256,
            "coordinate_plan_sha256": self.coordinate_plan_sha256,
            "artmesh_plan_sha256": self.artmesh_plan_sha256,
            "max_stops_per_binding": self.max_stops_per_binding,
            "max_baked_vertex_positions": self.max_baked_vertex_positions,
            "total_baked_vertex_positions": self.total_baked_vertex_positions,
            "artmesh_keyforms": [
                record.to_dict() for record in self.artmesh_keyforms
            ],
            "maximum_position_residual": self.maximum_position_residual,
            "maximum_opacity_residual": self.maximum_opacity_residual,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "plan_sha256": self.plan_sha256}


def _finite(value: object, *, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise _error(f"{field} must be finite")
    return float(value)


def _sample_table(
    transfer: Mapping[str, object], *, expected_size: int
) -> tuple[tuple[float, tuple[float, ...]], ...]:
    raw_samples = transfer.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise _error("sampled transfer has no samples")
    values = []
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, Mapping):
            raise _error("sampled transfer row is not an object")
        input_value = _finite(raw.get("input_value"), field="sample input")
        output = raw.get("output_values")
        if not isinstance(output, list) or len(output) != expected_size:
            raise _error("sampled transfer output size differs from its target")
        values.append(
            (
                input_value,
                tuple(
                    _finite(value, field=f"sample output {index}") for value in output
                ),
            )
        )
    values.sort(key=lambda item: item[0])
    if len({value for value, _output in values}) != len(values):
        raise _error("sampled transfer input values are duplicated")
    return tuple(values)


def _evaluate_samples(
    samples: tuple[tuple[float, tuple[float, ...]], ...], value: float
) -> tuple[float, ...]:
    if value < samples[0][0] or value > samples[-1][0]:
        raise _error("parameter value is outside sampled transfer bounds")
    for input_value, output in samples:
        if value == input_value:
            return output
    for (left_value, left), (right_value, right) in zip(samples, samples[1:]):
        if left_value < value < right_value:
            amount = (value - left_value) / (right_value - left_value)
            return tuple(
                left_item + (right_item - left_item) * amount
                for left_item, right_item in zip(left, right)
            )
    raise _error("sample interpolation interval is missing")


def _area(
    points: Sequence[tuple[float, float]], indices: tuple[int, int, int]
) -> float:
    a, b, c = (points[index] for index in indices)
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (
        c[0] - a[0]
    )


def _validate_winding(
    setup: Live2DArtMeshPlanRecord,
    keyform: tuple[tuple[float, float], ...],
) -> None:
    indices = setup.triangle_indices
    for offset in range(0, len(indices), 3):
        triangle = (indices[offset], indices[offset + 1], indices[offset + 2])
        setup_area = _area(setup.positions, triangle)
        key_area = _area(keyform, triangle)
        if abs(setup_area) <= 1e-12 or abs(key_area) <= 1e-12:
            raise _error("ArtMesh keyform contains a degenerate triangle")
        if math.copysign(1.0, setup_area) != math.copysign(1.0, key_area):
            raise _error("ArtMesh keyform flips a triangle")


def _distance(
    left: tuple[tuple[float, float], ...], right: tuple[tuple[float, float], ...]
) -> float:
    return max(
        (
            math.hypot(a[0] - b[0], a[1] - b[1])
            for a, b in zip(left, right)
        ),
        default=0.0,
    )


def evaluate_artmesh_keyform(
    record: Live2DArtMeshKeyforms,
    parameter_value: float,
) -> tuple[tuple[tuple[float, float], ...], float]:
    value = _finite(parameter_value, field="parameter value")
    values = record.parameter_values
    if value < values[0] or value > values[-1]:
        raise _error("parameter value lies outside ArtMesh keyforms")
    for index, stored in enumerate(values):
        if value == stored:
            return record.positions[index], record.opacities[index]
    for index, (left, right) in enumerate(zip(values, values[1:])):
        if left < value < right:
            amount = (value - left) / (right - left)
            positions = tuple(
                (
                    a[0] + (b[0] - a[0]) * amount,
                    a[1] + (b[1] - a[1]) * amount,
                )
                for a, b in zip(record.positions[index], record.positions[index + 1])
            )
            opacity = record.opacities[index] + (
                record.opacities[index + 1] - record.opacities[index]
            ) * amount
            return positions, opacity
    raise _error("ArtMesh keyform interpolation interval is missing")


def _assemble(
    rig: RigDocument,
    bindings: Live2DBindingPlan,
    coordinates: Live2DCoordinatePlan,
    artmeshes: Live2DArtMeshPlan,
) -> Live2DKeyformPlan:
    payload = rig.to_dict()
    full_bindings = {
        binding["binding_id"]: binding for binding in payload["control_bindings"]
    }
    parameter_by_id = {parameter.parameter_id: parameter for parameter in bindings.parameters}
    dynamic_by_target = {
        record.primitive_target_id: record
        for record in bindings.bindings
        if record.primitive_kind == "live2d_artmesh"
    }
    records = []
    total_baked = 0
    maximum_position_residual = 0.0
    maximum_opacity_residual = 0.0
    for setup in artmeshes.artmeshes:
        dynamic = dynamic_by_target.get(setup.primitive_target_id)
        if dynamic is None:
            records.append(
                Live2DArtMeshKeyforms(
                    mesh_id=setup.mesh_id,
                    primitive_target_id=setup.primitive_target_id,
                    parameter_id=None,
                    parameter_export_name=None,
                    parameter_default=0.0,
                    parameter_values=(0.0,),
                    positions=(setup.positions,),
                    opacities=(setup.setup_opacity,),
                    has_deform=False,
                    has_opacity=False,
                    binding_ids=(),
                    maximum_position_residual=0.0,
                    maximum_opacity_residual=0.0,
                )
            )
            continue
        parameter = parameter_by_id.get(dynamic.parameter_id)
        if parameter is None:
            raise _error("non-rigid ArtMesh binding lacks an emitted parameter")
        deform_samples = None
        opacity_samples = None
        for binding_id in dynamic.binding_ids:
            binding = full_bindings.get(binding_id)
            if binding is None or binding.get("target_id") != setup.mesh_id:
                raise _error("ArtMesh binding target differs from its plan")
            transfer = binding.get("transfer")
            if not isinstance(transfer, Mapping):
                raise _error("ArtMesh binding transfer is invalid")
            if binding.get("property") == "deform":
                if transfer.get("kind") != "sampled_deform" or deform_samples is not None:
                    raise _error("ArtMesh deform transfer is unsupported or duplicated")
                deform_samples = _sample_table(
                    transfer, expected_size=len(setup.positions) * 2
                )
            elif binding.get("property") == "opacity":
                if transfer.get("kind") != "sampled_property" or opacity_samples is not None:
                    raise _error("ArtMesh opacity transfer is unsupported or duplicated")
                opacity_samples = _sample_table(transfer, expected_size=1)
            else:
                raise _error("non-rigid ArtMesh binding has an unsupported property")
        values = {parameter.default}
        if deform_samples is not None:
            values.update(value for value, _output in deform_samples)
        if opacity_samples is not None:
            values.update(value for value, _output in opacity_samples)
        parameter_values = tuple(sorted(values))
        if len(parameter_values) > LIVE2D_MAX_STOPS_PER_BINDING:
            raise _error("ArtMesh binding exceeds the 17-stop capacity guard")
        positions = []
        opacities = []
        for value in parameter_values:
            if deform_samples is None:
                local_positions = setup.positions
            else:
                absolute = _evaluate_samples(deform_samples, value)
                local_positions = tuple(
                    canvas_to_artmesh_local(
                        coordinates,
                        setup.parent_instance_id,
                        (absolute[index], absolute[index + 1]),
                    )
                    for index in range(0, len(absolute), 2)
                )
            opacity = (
                setup.setup_opacity
                if opacity_samples is None
                else _evaluate_samples(opacity_samples, value)[0]
            )
            if not 0.0 <= opacity <= 1.0:
                raise _error("ArtMesh keyform opacity lies outside [0,1]")
            _validate_winding(setup, local_positions)
            positions.append(local_positions)
            opacities.append(opacity)
        default_index = parameter_values.index(parameter.default)
        position_residual = _distance(positions[default_index], setup.positions)
        opacity_residual = abs(opacities[default_index] - setup.setup_opacity)
        if position_residual > 0.1 or opacity_residual > 1 / 255:
            raise _error("ArtMesh default keyform does not reproduce setup state")
        record = Live2DArtMeshKeyforms(
            mesh_id=setup.mesh_id,
            primitive_target_id=setup.primitive_target_id,
            parameter_id=parameter.parameter_id,
            parameter_export_name=parameter.export_name,
            parameter_default=parameter.default,
            parameter_values=parameter_values,
            positions=tuple(positions),
            opacities=tuple(opacities),
            has_deform=deform_samples is not None,
            has_opacity=opacity_samples is not None,
            binding_ids=dynamic.binding_ids,
            maximum_position_residual=position_residual,
            maximum_opacity_residual=opacity_residual,
        )
        records.append(record)
        total_baked += len(parameter_values) * len(setup.positions)
        maximum_position_residual = max(maximum_position_residual, position_residual)
        maximum_opacity_residual = max(maximum_opacity_residual, opacity_residual)
    if total_baked > LIVE2D_MAX_BAKED_VERTEX_POSITIONS:
        raise _error("ArtMesh keyforms exceed the baked-position capacity guard")
    values = {
        "schema_version": LIVE2D_KEYFORM_PLAN_VERSION,
        "rig_document_sha256": rig.document_sha256,
        "binding_plan_sha256": bindings.plan_sha256,
        "coordinate_plan_sha256": coordinates.plan_sha256,
        "artmesh_plan_sha256": artmeshes.plan_sha256,
        "max_stops_per_binding": LIVE2D_MAX_STOPS_PER_BINDING,
        "max_baked_vertex_positions": LIVE2D_MAX_BAKED_VERTEX_POSITIONS,
        "total_baked_vertex_positions": total_baked,
        "artmesh_keyforms": tuple(records),
        "maximum_position_residual": maximum_position_residual,
        "maximum_opacity_residual": maximum_opacity_residual,
    }
    provisional = Live2DKeyformPlan(**values, plan_sha256="")
    return Live2DKeyformPlan(
        **values, plan_sha256=jcs_sha256(provisional.semantic_payload())
    )


def build_live2d_keyform_plan(
    rig: RigDocument,
    bindings: Live2DBindingPlan,
    coordinates: Live2DCoordinatePlan,
    artmeshes: Live2DArtMeshPlan,
) -> Live2DKeyformPlan:
    validate_rig_document(rig)
    return _assemble(rig, bindings, coordinates, artmeshes)


def validate_live2d_keyform_plan(
    plan: Live2DKeyformPlan,
    rig: RigDocument,
    bindings: Live2DBindingPlan,
    coordinates: Live2DCoordinatePlan,
    artmeshes: Live2DArtMeshPlan,
) -> Live2DKeyformPlan:
    if not isinstance(plan, Live2DKeyformPlan):
        raise _error("keyform plan has the wrong type")
    validate_rig_document(rig)
    if plan != _assemble(rig, bindings, coordinates, artmeshes):
        raise _error("keyform plan differs from the canonical Rig projection")
    if plan.plan_sha256 != jcs_sha256(plan.semantic_payload()):
        raise _error("keyform plan digest mismatch")
    return plan


__all__ = [
    "LIVE2D_KEYFORM_PLAN_VERSION",
    "LIVE2D_MAX_BAKED_VERTEX_POSITIONS",
    "LIVE2D_MAX_STOPS_PER_BINDING",
    "Live2DArtMeshKeyforms",
    "Live2DKeyformError",
    "Live2DKeyformPlan",
    "build_live2d_keyform_plan",
    "evaluate_artmesh_keyform",
    "validate_live2d_keyform_plan",
]
