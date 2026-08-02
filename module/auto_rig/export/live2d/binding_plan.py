from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ...jcs import jcs_sha256
from ...rig_document import RigDocument, validate_rig_document
from .rigid_drivers import (
    RigidDriverRegistry,
    validate_rigid_driver_registry,
)
from .symbols import (
    Live2DSymbolView,
    require_live2d_symbol,
    symbol_by_typed_key,
    validate_live2d_symbol_view,
)

LIVE2D_BINDING_PLAN_VERSION = "live2d-binding-plan-v1"
LIVE2D_DRIVER_LIVENESS_VERSION = "live2d-driver-liveness-v1"


class Live2DBindingPlanError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(f"invalid_live2d_binding_plan: {message}")


def _error(message: str) -> Live2DBindingPlanError:
    return Live2DBindingPlanError(message)


@dataclass(frozen=True, slots=True)
class Live2DParameterPlan:
    control_id: str
    parameter_id: str
    export_name: str
    minimum: float
    default: float
    maximum: float
    unit: str
    symbol_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "control_id": self.control_id,
            "parameter_id": self.parameter_id,
            "export_name": self.export_name,
            "minimum": self.minimum,
            "default": self.default,
            "maximum": self.maximum,
            "unit": self.unit,
            "symbol_sha256": self.symbol_sha256,
        }


@dataclass(frozen=True, slots=True)
class Live2DBindingRecord:
    control_id: str
    parameter_id: str
    parameter_export_name: str
    primitive_kind: str
    primitive_target_id: str
    primitive_export_name: str
    rig_target_id: str
    properties: tuple[str, ...]
    binding_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    preset_ids: tuple[str, ...]
    transfer_sha256: tuple[str, ...]
    stack_rank: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "control_id": self.control_id,
            "parameter_id": self.parameter_id,
            "parameter_export_name": self.parameter_export_name,
            "primitive_kind": self.primitive_kind,
            "primitive_target_id": self.primitive_target_id,
            "primitive_export_name": self.primitive_export_name,
            "rig_target_id": self.rig_target_id,
            "properties": list(self.properties),
            "binding_ids": list(self.binding_ids),
            "candidate_ids": list(self.candidate_ids),
            "preset_ids": list(self.preset_ids),
            "transfer_sha256": list(self.transfer_sha256),
            "stack_rank": self.stack_rank,
        }


@dataclass(frozen=True, slots=True)
class Live2DRotationInstance:
    instance_id: str
    export_name: str
    bone_id: str
    control_id: str
    parameter_id: str
    stack_rank: int
    parent_instance_id: str | None
    binding_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "export_name": self.export_name,
            "bone_id": self.bone_id,
            "control_id": self.control_id,
            "parameter_id": self.parameter_id,
            "stack_rank": self.stack_rank,
            "parent_instance_id": self.parent_instance_id,
            "binding_ids": list(self.binding_ids),
        }


@dataclass(frozen=True, slots=True)
class Live2DArtMeshAttachment:
    mesh_id: str
    part_id: str
    component_id: str
    primitive_target_id: str
    export_name: str
    dominant_bone_id: str
    parent_instance_id: str | None
    folded_bone_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "mesh_id": self.mesh_id,
            "part_id": self.part_id,
            "component_id": self.component_id,
            "primitive_target_id": self.primitive_target_id,
            "export_name": self.export_name,
            "dominant_bone_id": self.dominant_bone_id,
            "parent_instance_id": self.parent_instance_id,
            "folded_bone_ids": list(self.folded_bone_ids),
        }


@dataclass(frozen=True, slots=True)
class Live2DBindingPlan:
    schema_version: str
    liveness_version: str
    rig_document_sha256: str
    global_symbol_table_sha256: str
    symbol_view_sha256: str
    rigid_driver_registry_sha256: str
    format_preset_set_plan_sha256: str
    supported_preset_ids: tuple[str, ...]
    parameters: tuple[Live2DParameterPlan, ...]
    bindings: tuple[Live2DBindingRecord, ...]
    rotation_instances: tuple[Live2DRotationInstance, ...]
    artmesh_attachments: tuple[Live2DArtMeshAttachment, ...]
    pruned_bone_ids: tuple[str, ...]
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "liveness_version": self.liveness_version,
            "rig_document_sha256": self.rig_document_sha256,
            "global_symbol_table_sha256": self.global_symbol_table_sha256,
            "symbol_view_sha256": self.symbol_view_sha256,
            "rigid_driver_registry_sha256": self.rigid_driver_registry_sha256,
            "format_preset_set_plan_sha256": self.format_preset_set_plan_sha256,
            "supported_preset_ids": list(self.supported_preset_ids),
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "bindings": [binding.to_dict() for binding in self.bindings],
            "rotation_instances": [
                instance.to_dict() for instance in self.rotation_instances
            ],
            "artmesh_attachments": [
                attachment.to_dict() for attachment in self.artmesh_attachments
            ],
            "pruned_bone_ids": list(self.pruned_bone_ids),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "plan_sha256": self.plan_sha256}


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _error(f"{field} must be an object")
    return value


def _list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise _error(f"{field} must be a list")
    return value


def _id_map(values: list[object], field: str) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for index, raw in enumerate(values):
        record = _mapping(raw, field=f"{field}[{index}]")
        identity = record.get(field)
        if not isinstance(identity, str) or not identity or identity in result:
            raise _error(f"{field} identity is invalid or duplicated")
        result[identity] = record
    return result


def _live2d_set_plan(payload: Mapping[str, object]) -> Mapping[str, object]:
    formats = _mapping(payload.get("format_plans"), field="format_plans")
    matches = [
        _mapping(value, field="preset_set_plan")
        for value in _list(formats.get("preset_set_plans"), field="preset_set_plans")
        if isinstance(value, Mapping)
        and value.get("format_id") == "live2d_moc3_v4_00"
    ]
    if len(matches) != 1:
        raise _error("RigDocument lacks one Live2D preset-set plan")
    return matches[0]


def _control_parameter(
    control: Mapping[str, object],
) -> tuple[str, str, str, float, float, float, str]:
    control_id = control.get("control_id")
    formats = _mapping(control.get("format_bindings"), field="format_bindings")
    live2d = _mapping(
        formats.get("live2d_moc3_v4_00"), field="Live2D parameter binding"
    )
    values = (
        control_id,
        live2d.get("parameter_id"),
        live2d.get("export_name"),
        control.get("minimum"),
        control.get("default"),
        control.get("maximum"),
        control.get("unit"),
    )
    if (
        not all(isinstance(value, str) and value for value in values[:3])
        or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values[3:6]
        )
        or not isinstance(values[6], str)
    ):
        raise _error("control has an invalid Live2D parameter contract")
    return (
        str(values[0]),
        str(values[1]),
        str(values[2]),
        float(values[3]),
        float(values[4]),
        float(values[5]),
        str(values[6]),
    )


def _dominant_bone(mesh: Mapping[str, object]) -> str:
    totals: dict[str, float] = {}
    for vertex in _list(mesh.get("vertices"), field="mesh.vertices"):
        record = _mapping(vertex, field="mesh vertex")
        for influence in _list(record.get("influences"), field="vertex influences"):
            row = _mapping(influence, field="vertex influence")
            bone_id = row.get("bone_id")
            weight = row.get("weight")
            if not isinstance(bone_id, str) or not isinstance(weight, (int, float)):
                raise _error("mesh influence is invalid")
            totals[bone_id] = totals.get(bone_id, 0.0) + float(weight)
    if not totals:
        raise _error("mesh has no bone influence")
    return min(totals, key=lambda bone_id: (-totals[bone_id], bone_id))


def _assemble(
    rig: RigDocument,
    symbols: Live2DSymbolView,
    registry: RigidDriverRegistry,
) -> Live2DBindingPlan:
    payload = rig.to_dict()
    candidates_section = _mapping(
        payload.get("primitive_candidates"), field="primitive_candidates"
    )
    candidates = _id_map(
        _list(candidates_section.get("candidates"), field="candidates"),
        "candidate_id",
    )
    full_bindings = _id_map(
        _list(payload.get("control_bindings"), field="control_bindings"),
        "binding_id",
    )
    controls = _id_map(
        _list(payload.get("control_specs"), field="control_specs"), "control_id"
    )
    set_plan = _live2d_set_plan(payload)
    decisions = [
        _mapping(value, field="format decision")
        for value in _list(set_plan.get("decisions"), field="format decisions")
    ]
    supported = tuple(
        sorted(
            str(decision["preset_id"])
            for decision in decisions
            if decision.get("status") == "supported"
        )
    )
    preset_by_candidate: dict[str, set[str]] = {}
    for decision in decisions:
        if decision.get("status") != "supported":
            continue
        preset_id = decision.get("preset_id")
        if not isinstance(preset_id, str):
            raise _error("supported decision lacks a preset ID")
        for candidate_id in _list(
            decision.get("selected_candidate_ids"), field="selected candidate IDs"
        ):
            if not isinstance(candidate_id, str) or candidate_id not in candidates:
                raise _error("supported decision selects an unknown candidate")
            preset_by_candidate.setdefault(candidate_id, set()).add(preset_id)

    registry_by_control = {row.control_id: row for row in registry.rows}
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for candidate_id, preset_ids in sorted(preset_by_candidate.items()):
        candidate = candidates[candidate_id]
        if candidate.get("format_id") != "live2d_moc3_v4_00":
            raise _error("Live2D decision selected a non-Live2D candidate")
        template = _mapping(candidate.get("binding_template"), field="binding template")
        binding_id = template.get("binding_id")
        if not isinstance(binding_id, str) or binding_id not in full_bindings:
            raise _error("binding candidate references an unknown binding")
        binding = full_bindings[binding_id]
        control_id = binding.get("control_id")
        if not isinstance(control_id, str) or control_id not in controls:
            raise _error("binding references an unknown control")
        (
            _control_id,
            parameter_id,
            parameter_name,
            _minimum,
            _default,
            _maximum,
            _unit,
        ) = _control_parameter(controls[control_id])
        typed = _mapping(candidate.get("typed_primitive_key"), field="typed key")
        typed_digest = typed.get("key_sha256")
        if not isinstance(typed_digest, str):
            raise _error("binding candidate lacks a typed-key digest")
        symbol = symbol_by_typed_key(symbols, typed_digest)
        primitive_kind = typed.get("kind")
        primitive_target_id = candidate.get("primitive_target_id")
        rig_target_id = binding.get("target_id")
        property_name = binding.get("property")
        transfer = _mapping(binding.get("transfer"), field="target transfer")
        transfer_digest = transfer.get("transfer_sha256")
        if (
            primitive_kind
            not in {"live2d_artmesh", "live2d_rotation_deformer"}
            or not isinstance(primitive_target_id, str)
            or not isinstance(rig_target_id, str)
            or not isinstance(property_name, str)
            or not isinstance(transfer_digest, str)
        ):
            raise _error("binding candidate has an unsupported primitive contract")
        rank = None
        if primitive_kind == "live2d_rotation_deformer":
            row = registry_by_control.get(control_id)
            if row is None or row.parameter_id != parameter_id:
                raise _error("rotation binding is absent from RigidDriverRegistry")
            rank = row.stack_rank
        key = (parameter_id, primitive_target_id)
        group = grouped.setdefault(
            key,
            {
                "control_id": control_id,
                "parameter_id": parameter_id,
                "parameter_export_name": parameter_name,
                "primitive_kind": primitive_kind,
                "primitive_target_id": primitive_target_id,
                "primitive_export_name": symbol.export_name,
                "rig_target_id": rig_target_id,
                "properties": set(),
                "binding_ids": set(),
                "candidate_ids": set(),
                "preset_ids": set(),
                "transfer_sha256": set(),
                "stack_rank": rank,
            },
        )
        identity_fields = {
            "control_id": control_id,
            "parameter_id": parameter_id,
            "parameter_export_name": parameter_name,
            "primitive_kind": primitive_kind,
            "primitive_target_id": primitive_target_id,
            "primitive_export_name": symbol.export_name,
            "rig_target_id": rig_target_id,
            "stack_rank": rank,
        }
        if any(group[field] != value for field, value in identity_fields.items()):
            raise _error("one primitive/parameter binding has conflicting identity")
        group["properties"].add(property_name)  # type: ignore[union-attr]
        group["binding_ids"].add(binding_id)  # type: ignore[union-attr]
        group["candidate_ids"].add(candidate_id)  # type: ignore[union-attr]
        group["preset_ids"].update(preset_ids)  # type: ignore[union-attr]
        group["transfer_sha256"].add(transfer_digest)  # type: ignore[union-attr]

    non_rigid_parameters: dict[str, set[str]] = {}
    records = []
    for group in grouped.values():
        if group["primitive_kind"] == "live2d_artmesh":
            non_rigid_parameters.setdefault(
                str(group["primitive_target_id"]), set()
            ).add(str(group["parameter_id"]))
        records.append(
            Live2DBindingRecord(
                control_id=str(group["control_id"]),
                parameter_id=str(group["parameter_id"]),
                parameter_export_name=str(group["parameter_export_name"]),
                primitive_kind=str(group["primitive_kind"]),
                primitive_target_id=str(group["primitive_target_id"]),
                primitive_export_name=str(group["primitive_export_name"]),
                rig_target_id=str(group["rig_target_id"]),
                properties=tuple(sorted(group["properties"])),  # type: ignore[arg-type]
                binding_ids=tuple(sorted(group["binding_ids"])),  # type: ignore[arg-type]
                candidate_ids=tuple(sorted(group["candidate_ids"])),  # type: ignore[arg-type]
                preset_ids=tuple(sorted(group["preset_ids"])),  # type: ignore[arg-type]
                transfer_sha256=tuple(sorted(group["transfer_sha256"])),  # type: ignore[arg-type]
                stack_rank=group["stack_rank"],  # type: ignore[arg-type]
            )
        )
    if any(len(parameter_ids) > 1 for parameter_ids in non_rigid_parameters.values()):
        raise _error("two non-rigid parameters target one ArtMesh")
    bindings = tuple(
        sorted(records, key=lambda item: (item.parameter_id, item.primitive_target_id))
    )

    parameter_records = []
    for parameter_id in sorted({record.parameter_id for record in bindings}):
        matching_controls = {
            record.control_id for record in bindings if record.parameter_id == parameter_id
        }
        if len(matching_controls) != 1:
            raise _error("one parameter resolves to multiple controls")
        control_id = matching_controls.pop()
        (
            _control_id,
            _parameter_id,
            export_name,
            minimum,
            default,
            maximum,
            unit,
        ) = _control_parameter(controls[control_id])
        symbol = require_live2d_symbol(
            symbols, kind="live2d_parameter", parameter_id=parameter_id
        )
        if symbol.export_name != export_name:
            raise _error("parameter symbol differs from ControlRegistry")
        parameter_records.append(
            Live2DParameterPlan(
                control_id=control_id,
                parameter_id=parameter_id,
                export_name=export_name,
                minimum=minimum,
                default=default,
                maximum=maximum,
                unit=unit,
                symbol_sha256=symbol.symbol_sha256,
            )
        )

    bones = _id_map(_list(payload.get("bones"), field="bones"), "bone_id")
    parent_by_bone = {
        bone_id: bone.get("parent_id") for bone_id, bone in bones.items()
    }
    by_bone: dict[str, list[Live2DBindingRecord]] = {}
    for record in bindings:
        if record.primitive_kind == "live2d_rotation_deformer":
            by_bone.setdefault(record.rig_target_id, []).append(record)
    depth_cache: dict[str, int] = {}

    def depth(bone_id: str) -> int:
        if bone_id in depth_cache:
            return depth_cache[bone_id]
        parent = parent_by_bone.get(bone_id)
        value = 0 if parent is None else depth(str(parent)) + 1
        depth_cache[bone_id] = value
        return value

    inner_by_bone: dict[str, str] = {}
    rotation_instances = []
    for bone_id in sorted(by_bone, key=lambda value: (depth(value), value)):
        parent_bone = parent_by_bone.get(bone_id)
        outer_parent = None
        while isinstance(parent_bone, str):
            outer_parent = inner_by_bone.get(parent_bone)
            if outer_parent is not None:
                break
            parent_bone = parent_by_bone.get(parent_bone)
        previous = outer_parent
        for record in sorted(
            by_bone[bone_id], key=lambda value: (int(value.stack_rank or 0), value.control_id)
        ):
            if record.stack_rank is None:
                raise _error("rotation instance lacks a stack rank")
            rotation_instances.append(
                Live2DRotationInstance(
                    instance_id=record.primitive_target_id,
                    export_name=record.primitive_export_name,
                    bone_id=bone_id,
                    control_id=record.control_id,
                    parameter_id=record.parameter_id,
                    stack_rank=record.stack_rank,
                    parent_instance_id=previous,
                    binding_ids=record.binding_ids,
                )
            )
            previous = record.primitive_target_id
        inner_by_bone[bone_id] = str(previous)

    model_artmeshes: dict[tuple[str, str], Mapping[str, object]] = {}
    for candidate in candidates.values():
        if (
            candidate.get("candidate_kind") != "live2d_artmesh"
            or candidate.get("binding_template") is not None
        ):
            continue
        typed = _mapping(candidate.get("typed_primitive_key"), field="ArtMesh key")
        part_id = typed.get("base_source_internal_id")
        component_id = typed.get("component_id")
        if not isinstance(part_id, str) or not isinstance(component_id, str):
            raise _error("ArtMesh model candidate lacks Part/component identity")
        identity = (part_id, component_id)
        if identity in model_artmeshes:
            raise _error("ArtMesh model candidate is duplicated")
        model_artmeshes[identity] = candidate

    attachments = []
    meshes = _id_map(_list(payload.get("meshes"), field="meshes"), "mesh_id")
    for mesh_id, mesh in sorted(meshes.items()):
        part_id = mesh.get("part_id")
        component_id = mesh.get("component_id")
        if not isinstance(part_id, str) or not isinstance(component_id, str):
            raise _error("mesh lacks Part/component identity")
        candidate = model_artmeshes.get((part_id, component_id))
        if candidate is None:
            raise _error("mesh lacks a static ArtMesh candidate")
        typed = _mapping(candidate.get("typed_primitive_key"), field="ArtMesh key")
        symbol = symbol_by_typed_key(symbols, str(typed.get("key_sha256")))
        dominant = _dominant_bone(mesh)
        cursor: str | None = dominant
        folded = []
        parent_instance = None
        while cursor is not None:
            parent_instance = inner_by_bone.get(cursor)
            if parent_instance is not None:
                break
            if cursor != "bone/root":
                folded.append(cursor)
            parent = parent_by_bone.get(cursor)
            cursor = parent if isinstance(parent, str) else None
        attachments.append(
            Live2DArtMeshAttachment(
                mesh_id=mesh_id,
                part_id=part_id,
                component_id=component_id,
                primitive_target_id=str(candidate["primitive_target_id"]),
                export_name=symbol.export_name,
                dominant_bone_id=dominant,
                parent_instance_id=parent_instance,
                folded_bone_ids=tuple(folded),
            )
        )

    live_bones = set(by_bone)
    pruned = tuple(
        sorted(
            bone_id
            for bone_id in bones
            if bone_id != "bone/root" and bone_id not in live_bones
        )
    )
    values = {
        "schema_version": LIVE2D_BINDING_PLAN_VERSION,
        "liveness_version": LIVE2D_DRIVER_LIVENESS_VERSION,
        "rig_document_sha256": rig.document_sha256,
        "global_symbol_table_sha256": str(
            _mapping(payload["export_symbols"], field="export_symbols")["table_sha256"]
        ),
        "symbol_view_sha256": symbols.view_sha256,
        "rigid_driver_registry_sha256": registry.registry_sha256,
        "format_preset_set_plan_sha256": str(set_plan["plan_sha256"]),
        "supported_preset_ids": supported,
        "parameters": tuple(parameter_records),
        "bindings": bindings,
        "rotation_instances": tuple(rotation_instances),
        "artmesh_attachments": tuple(attachments),
        "pruned_bone_ids": pruned,
    }
    provisional = Live2DBindingPlan(**values, plan_sha256="")
    return Live2DBindingPlan(
        **values, plan_sha256=jcs_sha256(provisional.semantic_payload())
    )


def build_live2d_binding_plan(
    rig: RigDocument,
    symbols: Live2DSymbolView,
    registry: RigidDriverRegistry,
) -> Live2DBindingPlan:
    validate_rig_document(rig)
    validate_live2d_symbol_view(symbols)
    validate_rigid_driver_registry(registry, rig.to_dict()["control_specs"])
    return validate_live2d_binding_plan(
        _assemble(rig, symbols, registry), rig, symbols, registry
    )


def validate_live2d_binding_plan(
    plan: Live2DBindingPlan,
    rig: RigDocument,
    symbols: Live2DSymbolView,
    registry: RigidDriverRegistry,
) -> Live2DBindingPlan:
    if not isinstance(plan, Live2DBindingPlan):
        raise _error("binding plan has the wrong type")
    validate_rig_document(rig)
    validate_live2d_symbol_view(symbols)
    validate_rigid_driver_registry(registry, rig.to_dict()["control_specs"])
    if (
        plan.schema_version != LIVE2D_BINDING_PLAN_VERSION
        or plan.liveness_version != LIVE2D_DRIVER_LIVENESS_VERSION
    ):
        raise _error("binding/liveness version is unsupported")
    expected = _assemble(rig, symbols, registry)
    if plan != expected:
        raise _error("binding plan differs from the canonical Rig projection")
    if plan.plan_sha256 != jcs_sha256(plan.semantic_payload()):
        raise _error("binding plan digest mismatch")
    return plan


__all__ = [
    "LIVE2D_BINDING_PLAN_VERSION",
    "LIVE2D_DRIVER_LIVENESS_VERSION",
    "Live2DArtMeshAttachment",
    "Live2DBindingPlan",
    "Live2DBindingPlanError",
    "Live2DBindingRecord",
    "Live2DParameterPlan",
    "Live2DRotationInstance",
    "build_live2d_binding_plan",
    "validate_live2d_binding_plan",
]
