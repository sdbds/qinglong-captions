from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from ...rig_document import RigDocument, validate_rig_document
from .artmesh import Live2DArtMeshPlan
from .binding_plan import Live2DBindingPlan
from .coordinates import Live2DCoordinatePlan, Live2DRotationFrame
from .keyforms import Live2DKeyformPlan, validate_live2d_keyform_plan
from .moc3 import Moc3CanvasInfo
from .moc3_codec import (
    Moc3V400Document,
    decode_moc3_v400,
    empty_v400_sections,
    encode_moc3_v400,
)

LIVE2D_MOC3_COMPILER_VERSION = "live2d-moc3-compiler-v1"
LIVE2D_MOC3_MAX_BYTES = 64 * 1024 * 1024
LIVE2D_DRAW_ORDER_BASE = 500


class Live2DMoc3CompileError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(f"invalid_live2d_moc3_document: {message}")


def _error(message: str) -> Live2DMoc3CompileError:
    return Live2DMoc3CompileError(message)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _error(f"{field} must be an object")
    return value


def _number(value: object, *, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise _error(f"{field} must be finite numeric data")
    return float(value)


def _clean(value: float) -> float:
    return 0.0 if abs(value) < 1e-12 else value


def _affine_output(transfer: Mapping[str, object], control_value: float) -> float:
    if transfer.get("kind") != "affine_scalar":
        raise _error("RotationDeformer requires affine-scalar transfers")
    control_default = _number(transfer.get("control_default"), field="transfer control_default")
    defaults = transfer.get("output_at_default")
    gains = transfer.get("gain")
    if not isinstance(defaults, list) or not isinstance(gains, list):
        raise _error("affine transfer default/gain must be lists")
    if len(defaults) != 1 or len(gains) != 1:
        raise _error("RotationDeformer affine transfer must be scalar")
    return _clean(
        _number(defaults[0], field="transfer output default")
        + _number(gains[0], field="transfer gain") * (control_value - control_default)
    )


@dataclass(frozen=True, slots=True)
class _ObjectBinding:
    object_kind: str
    object_index: int
    parameter_index: int
    keys: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _RotationKeyforms:
    values: tuple[float, ...]
    angles: tuple[float, ...]
    origin_xs: tuple[float, ...]
    origin_ys: tuple[float, ...]
    scales: tuple[float, ...]


def _rotation_keyforms(
    *,
    rig_payload: Mapping[str, object],
    bindings: Live2DBindingPlan,
    frame: Live2DRotationFrame,
    instance_index: int,
) -> _RotationKeyforms:
    instance = bindings.rotation_instances[instance_index]
    parameter_by_id = {parameter.parameter_id: parameter for parameter in bindings.parameters}
    parameter = parameter_by_id.get(instance.parameter_id)
    if parameter is None:
        raise _error("RotationDeformer parameter is absent")
    values = tuple(sorted({parameter.minimum, parameter.default, parameter.maximum}))
    if not values:
        raise _error("RotationDeformer has no parameter values")
    raw_bindings = rig_payload.get("control_bindings")
    if not isinstance(raw_bindings, list):
        raise _error("Rig control bindings are absent")
    full_by_id = {
        str(record.get("binding_id")): record
        for record in raw_bindings
        if isinstance(record, Mapping) and isinstance(record.get("binding_id"), str)
    }
    property_bindings: dict[str, Mapping[str, object]] = {}
    for binding_id in instance.binding_ids:
        record = full_by_id.get(binding_id)
        if record is None:
            raise _error("RotationDeformer references an unknown binding")
        if record.get("control_id") != instance.control_id or record.get("target_id") != instance.bone_id:
            raise _error("RotationDeformer binding identity differs from its instance")
        property_name = record.get("property")
        if property_name not in {"rotation", "translation_x", "translation_y"}:
            raise _error("RotationDeformer binding has an unsupported property")
        if str(property_name) in property_bindings:
            raise _error("RotationDeformer property is duplicated")
        property_bindings[str(property_name)] = record

    angles = []
    origin_xs = []
    origin_ys = []
    scales = []
    for value in values:
        outputs = {"rotation": 0.0, "translation_x": 0.0, "translation_y": 0.0}
        for property_name, record in property_bindings.items():
            outputs[property_name] = _affine_output(_mapping(record.get("transfer"), field="binding transfer"), value)
        # Root origins are MOC root units; nested origins are parent-local pixels.
        origin_xs.append(_clean(frame.origin[0] + outputs["translation_x"] * frame.scale))
        origin_ys.append(_clean(frame.origin[1] + outputs["translation_y"] * frame.scale))
        angles.append(outputs["rotation"])
        scales.append(frame.scale)
    return _RotationKeyforms(
        values=values,
        angles=tuple(angles),
        origin_xs=tuple(origin_xs),
        origin_ys=tuple(origin_ys),
        scales=tuple(scales),
    )


def _build(
    rig: RigDocument,
    bindings: Live2DBindingPlan,
    coordinates: Live2DCoordinatePlan,
    artmeshes: Live2DArtMeshPlan,
    keyforms: Live2DKeyformPlan,
) -> Moc3V400Document:
    payload = rig.to_dict()
    parameter_index = {parameter.parameter_id: index for index, parameter in enumerate(bindings.parameters)}
    frame_by_id = {frame.instance_id: frame for frame in coordinates.rotation_frames}
    rotation_index = {instance.instance_id: index for index, instance in enumerate(bindings.rotation_instances)}
    if set(frame_by_id) != set(rotation_index):
        raise _error("coordinate and binding RotationDeformer sets differ")
    keyforms_by_mesh = {record.mesh_id: record for record in keyforms.artmesh_keyforms}
    if set(keyforms_by_mesh) != {record.mesh_id for record in artmeshes.artmeshes}:
        raise _error("ArtMesh and keyform sets differ")
    if not artmeshes.parts or not artmeshes.artmeshes:
        raise _error("MOC3 requires at least one Part and one ArtMesh")
    for mesh in artmeshes.artmeshes:
        if len(mesh.positions) > 32767 or len(mesh.triangle_indices) > 32767:
            raise _error("ArtMesh exceeds the signed-int16 vertex/index guard")

    object_bindings: list[_ObjectBinding] = []
    artmesh_dynamic_binding: dict[int, int] = {}
    for index, artmesh in enumerate(artmeshes.artmeshes):
        record = keyforms_by_mesh[artmesh.mesh_id]
        if record.parameter_id is None:
            continue
        parameter = parameter_index.get(record.parameter_id)
        if parameter is None:
            raise _error("ArtMesh keyforms reference an unknown parameter")
        artmesh_dynamic_binding[index] = len(object_bindings)
        object_bindings.append(_ObjectBinding("art_mesh", index, parameter, record.parameter_values))

    rotation_records: list[_RotationKeyforms] = []
    rotation_dynamic_binding: dict[int, int] = {}
    for index, instance in enumerate(bindings.rotation_instances):
        parameter = parameter_index.get(instance.parameter_id)
        if parameter is None:
            raise _error("RotationDeformer references an unknown parameter")
        record = _rotation_keyforms(
            rig_payload=payload,
            bindings=bindings,
            frame=frame_by_id[instance.instance_id],
            instance_index=index,
        )
        rotation_records.append(record)
        rotation_dynamic_binding[index] = len(object_bindings)
        object_bindings.append(_ObjectBinding("rotation", index, parameter, record.values))

    binding_order = sorted(
        range(len(object_bindings)),
        key=lambda index: (
            object_bindings[index].parameter_index,
            object_bindings[index].object_kind,
            object_bindings[index].object_index,
        ),
    )
    binding_index_by_object = {object_index: binding_index for binding_index, object_index in enumerate(binding_order)}
    ordered_bindings = [object_bindings[index] for index in binding_order]
    binding_key_begins = []
    binding_key_counts = []
    all_keys: list[float] = []
    for record in ordered_bindings:
        binding_key_begins.append(len(all_keys))
        binding_key_counts.append(len(record.keys))
        all_keys.extend(record.keys)

    parameter_binding_begins = []
    parameter_binding_counts = []
    cursor = 0
    for index in range(len(bindings.parameters)):
        count = sum(record.parameter_index == index for record in ordered_bindings)
        parameter_binding_begins.append(cursor)
        parameter_binding_counts.append(count)
        cursor += count
    if cursor != len(ordered_bindings):
        raise _error("parameter binding spans are not closed")
    if any(count <= 0 for count in parameter_binding_counts):
        raise _error("MOC3 contains a parameter without a visible target")

    artmesh_count = len(artmeshes.artmeshes)
    part_count = len(artmeshes.parts)
    deformer_count = len(bindings.rotation_instances)
    rotation_count = deformer_count
    artmesh_band_indices = tuple(range(artmesh_count))
    part_band_indices = tuple(range(artmesh_count, artmesh_count + part_count))
    deformer_band_indices = tuple(
        range(
            artmesh_count + part_count,
            artmesh_count + part_count + deformer_count,
        )
    )
    rotation_band_indices = tuple(
        range(
            artmesh_count + part_count + deformer_count,
            artmesh_count + part_count + deformer_count + rotation_count,
        )
    )
    band_count = artmesh_count + part_count + deformer_count + rotation_count
    band_begins = [0] * band_count
    band_counts = [0] * band_count
    associations: list[int] = []
    for object_index, object_binding_index in artmesh_dynamic_binding.items():
        band_index = artmesh_band_indices[object_index]
        band_begins[band_index] = len(associations)
        band_counts[band_index] = 1
        associations.append(binding_index_by_object[object_binding_index])
    for object_index, object_binding_index in rotation_dynamic_binding.items():
        band_index = rotation_band_indices[object_index]
        band_begins[band_index] = len(associations)
        band_counts[band_index] = 1
        associations.append(binding_index_by_object[object_binding_index])

    sections = empty_v400_sections()
    sections.update(
        {
            "part.ids": tuple(part.export_name for part in artmeshes.parts),
            "part.keyform_binding_band_indices": part_band_indices,
            "part.keyform_begin_indices": tuple(range(part_count)),
            "part.keyform_counts": (1,) * part_count,
            # `visibles` is the object's runtime availability flag, not its
            # default keyform opacity. A drawable hidden at rest still has to
            # remain available so an expression can reveal it later.
            "part.visibles": (True,) * part_count,
            "part.enables": (True,) * part_count,
            "part.parent_part_indices": (-1,) * part_count,
            "deformer.ids": tuple(instance.export_name for instance in bindings.rotation_instances),
            "deformer.keyform_binding_band_indices": deformer_band_indices,
            "deformer.visibles": (True,) * deformer_count,
            "deformer.enables": (True,) * deformer_count,
            "deformer.parent_part_indices": (0,) * deformer_count,
            "deformer.parent_deformer_indices": tuple(
                -1 if instance.parent_instance_id is None else rotation_index[instance.parent_instance_id]
                for instance in bindings.rotation_instances
            ),
            "deformer.types": (1,) * deformer_count,
            "deformer.specific_indices": tuple(range(rotation_count)),
            "rotation_deformer.keyform_binding_band_indices": rotation_band_indices,
            "rotation_deformer.keyform_begin_indices": tuple(
                sum(len(item.values) for item in rotation_records[:index]) for index in range(rotation_count)
            ),
            "rotation_deformer.keyform_counts": tuple(len(record.values) for record in rotation_records),
            "rotation_deformer.base_angles": (0.0,) * rotation_count,
            "art_mesh.ids": tuple(mesh.export_name for mesh in artmeshes.artmeshes),
            "art_mesh.keyform_binding_band_indices": artmesh_band_indices,
            "art_mesh.visibles": (True,) * artmesh_count,
            "art_mesh.enables": (True,) * artmesh_count,
            "art_mesh.parent_part_indices": tuple(
                next(index for index, part in enumerate(artmeshes.parts) if part.part_id == mesh.part_id)
                for mesh in artmeshes.artmeshes
            ),
            "art_mesh.parent_deformer_indices": tuple(
                -1 if mesh.parent_instance_id is None else rotation_index[mesh.parent_instance_id] for mesh in artmeshes.artmeshes
            ),
            "art_mesh.texture_indices": tuple(mesh.texture_index for mesh in artmeshes.artmeshes),
            "art_mesh.drawable_flags": (4,) * artmesh_count,
            "art_mesh.position_index_counts": tuple(len(mesh.positions) for mesh in artmeshes.artmeshes),
            "art_mesh.uv_begin_indices": tuple(
                sum(len(item.uvs) * 2 for item in artmeshes.artmeshes[:index]) for index in range(artmesh_count)
            ),
            "art_mesh.position_index_begin_indices": tuple(
                sum(len(item.triangle_indices) for item in artmeshes.artmeshes[:index]) for index in range(artmesh_count)
            ),
            "art_mesh.vertex_counts": tuple(len(mesh.triangle_indices) for mesh in artmeshes.artmeshes),
            "art_mesh.mask_begin_indices": (0,) * artmesh_count,
            "art_mesh.mask_counts": (0,) * artmesh_count,
            "parameter.ids": tuple(parameter.export_name for parameter in bindings.parameters),
            "parameter.max_values": tuple(parameter.maximum for parameter in bindings.parameters),
            "parameter.min_values": tuple(parameter.minimum for parameter in bindings.parameters),
            "parameter.default_values": tuple(parameter.default for parameter in bindings.parameters),
            "parameter.repeats": (False,) * len(bindings.parameters),
            "parameter.decimal_places": (1,) * len(bindings.parameters),
            "parameter.keyform_binding_begin_indices": tuple(parameter_binding_begins),
            "parameter.keyform_binding_counts": tuple(parameter_binding_counts),
            "part_keyform.draw_orders": tuple(float(LIVE2D_DRAW_ORDER_BASE + part.draw_rank) for part in artmeshes.parts),
            "rotation_deformer_keyform.opacities": tuple(1.0 for record in rotation_records for _value in record.values),
            "rotation_deformer_keyform.angles": tuple(value for record in rotation_records for value in record.angles),
            "rotation_deformer_keyform.origin_xs": tuple(value for record in rotation_records for value in record.origin_xs),
            "rotation_deformer_keyform.origin_ys": tuple(value for record in rotation_records for value in record.origin_ys),
            "rotation_deformer_keyform.scales": tuple(value for record in rotation_records for value in record.scales),
            "rotation_deformer_keyform.reflect_xs": tuple(False for record in rotation_records for _value in record.values),
            "rotation_deformer_keyform.reflect_ys": tuple(False for record in rotation_records for _value in record.values),
            "keyform_binding_index.indices": tuple(associations),
            "keyform_binding_band.begin_indices": tuple(band_begins),
            "keyform_binding_band.counts": tuple(band_counts),
            "keyform_binding.keys_begin_indices": tuple(binding_key_begins),
            "keyform_binding.keys_counts": tuple(binding_key_counts),
            "keys.values": tuple(all_keys),
            "drawable_mask.art_mesh_indices": (-1,),
            "draw_order_group.object_begin_indices": (0,),
            "draw_order_group.object_counts": (artmesh_count,),
            "draw_order_group.object_total_counts": (artmesh_count,),
            "draw_order_group.min_draw_orders": (1000,),
            "draw_order_group.max_draw_orders": (200,),
            "draw_order_group_object.types": (0,) * artmesh_count,
            "draw_order_group_object.indices": tuple(range(artmesh_count - 1, -1, -1)),
            "draw_order_group_object.group_indices": (-1,) * artmesh_count,
        }
    )

    artmesh_keyform_begins = []
    artmesh_keyform_counts = []
    artmesh_opacities: list[float] = []
    artmesh_draw_orders: list[float] = []
    artmesh_position_begins: list[int] = []
    keyform_positions: list[float] = []
    for mesh in artmeshes.artmeshes:
        record = keyforms_by_mesh[mesh.mesh_id]
        artmesh_keyform_begins.append(len(artmesh_opacities))
        artmesh_keyform_counts.append(len(record.parameter_values))
        for positions, opacity in zip(record.positions, record.opacities, strict=True):
            artmesh_opacities.append(opacity)
            artmesh_draw_orders.append(float(LIVE2D_DRAW_ORDER_BASE + mesh.draw_order))
            artmesh_position_begins.append(len(keyform_positions))
            keyform_positions.extend(coordinate for point in positions for coordinate in point)
    sections.update(
        {
            "art_mesh.keyform_begin_indices": tuple(artmesh_keyform_begins),
            "art_mesh.keyform_counts": tuple(artmesh_keyform_counts),
            "art_mesh_keyform.opacities": tuple(artmesh_opacities),
            "art_mesh_keyform.draw_orders": tuple(artmesh_draw_orders),
            "art_mesh_keyform.keyform_position_begin_indices": tuple(artmesh_position_begins),
            "keyform_position.xys": tuple(keyform_positions),
            "uv.xys": tuple(coordinate for mesh in artmeshes.artmeshes for point in mesh.uvs for coordinate in point),
            "position_index.indices": tuple(index for mesh in artmeshes.artmeshes for index in mesh.triangle_indices),
        }
    )
    rotation_keyform_count = sum(len(record.values) for record in rotation_records)
    artmesh_keyform_count = len(artmesh_opacities)
    counts = (
        part_count,
        deformer_count,
        0,
        rotation_count,
        artmesh_count,
        len(bindings.parameters),
        part_count,
        0,
        rotation_keyform_count,
        artmesh_keyform_count,
        len(keyform_positions),
        len(associations),
        band_count,
        len(object_bindings),
        len(all_keys),
        sum(len(mesh.uvs) * 2 for mesh in artmeshes.artmeshes),
        sum(len(mesh.triangle_indices) for mesh in artmeshes.artmeshes),
        1,
        1,
        artmesh_count,
        0,
        0,
        0,
    )
    document = Moc3V400Document(
        counts=counts,
        canvas=Moc3CanvasInfo(
            pixels_per_unit=coordinates.ppu,
            origin_x=coordinates.width / 2.0,
            origin_y=coordinates.height / 2.0,
            width=float(coordinates.width),
            height=float(coordinates.height),
            flag=0,
        ),
        sections=sections,
    )
    encoded = encode_moc3_v400(document)
    if len(encoded) > LIVE2D_MOC3_MAX_BYTES:
        raise _error("MOC3 exceeds the 64 MiB capacity guard")
    # The public typed document is float32-canonical, exactly as it will be
    # observed after a binary reload. This makes encode/decode identity explicit.
    return decode_moc3_v400(encoded)


def build_live2d_moc3_document(
    rig: RigDocument,
    bindings: Live2DBindingPlan,
    coordinates: Live2DCoordinatePlan,
    artmeshes: Live2DArtMeshPlan,
    keyforms: Live2DKeyformPlan,
) -> Moc3V400Document:
    validate_rig_document(rig)
    validate_live2d_keyform_plan(keyforms, rig, bindings, coordinates, artmeshes)
    return _build(rig, bindings, coordinates, artmeshes, keyforms)


__all__ = [
    "LIVE2D_DRAW_ORDER_BASE",
    "LIVE2D_MOC3_COMPILER_VERSION",
    "LIVE2D_MOC3_MAX_BYTES",
    "Live2DMoc3CompileError",
    "build_live2d_moc3_document",
]
