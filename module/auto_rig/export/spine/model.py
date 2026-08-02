from __future__ import annotations

import json
from dataclasses import dataclass

from ...jcs import jcs_bytes, jcs_sha256
from ...rig_document import RigDocument, validate_rig_document
from .atlas import SpineAtlasPlan, validate_spine_atlas_plan
from .bind_plan import SpineBindPlan, validate_spine_bind_plan
from .coordinates import SpineCoordinatePlan, validate_spine_coordinate_plan
from .mesh_encoder import encode_spine_mesh, validate_spine_encoded_mesh
from .symbols import (
    SpineSymbolView,
    require_spine_symbol,
    validate_spine_symbol_view,
)

SPINE_DOCUMENT_VERSION = "spine-document-v1"
SPINE_JSON_VERSION = "4.2"


class SpineDocumentError(ValueError):
    """Raised when the disposable Spine document is incomplete or inconsistent."""

    def __init__(self, message: str) -> None:
        super().__init__(f"invalid_spine_document: {message}")


def _error(message: str) -> SpineDocumentError:
    return SpineDocumentError(message)


@dataclass(frozen=True, slots=True)
class SpineDocument:
    schema_version: str
    rig_document_sha256: str
    coordinate_plan_sha256: str
    bind_plan_sha256: str
    symbol_view_sha256: str
    atlas_plan_sha256: str
    component_records: tuple[dict[str, object], ...]
    _canonical_json: bytes

    def to_dict(self) -> dict[str, object]:
        value = json.loads(self._canonical_json.decode("utf-8"))
        if not isinstance(value, dict):
            raise _error("Spine JSON root is not an object")
        return value

    @property
    def document_sha256(self) -> str:
        return jcs_sha256(self.to_dict())

    @property
    def slot_names(self) -> list[str]:
        return [str(record["slot_name"]) for record in self.component_records]

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "rig_document_sha256": self.rig_document_sha256,
            "coordinate_plan_sha256": self.coordinate_plan_sha256,
            "bind_plan_sha256": self.bind_plan_sha256,
            "symbol_view_sha256": self.symbol_view_sha256,
            "atlas_plan_sha256": self.atlas_plan_sha256,
            "component_records": list(self.component_records),
            "document_sha256": self.document_sha256,
        }


def _bone_payloads(bind: SpineBindPlan, symbols: SpineSymbolView):
    result = []
    names: dict[str, str] = {}
    for bone in bind.bones:
        name = require_spine_symbol(
            symbols,
            kind="spine_bone",
            source_internal_ids=(bone.bone_id,),
        ).export_name
        names[bone.bone_id] = name
        if bone.bone_id == "bone/root":
            result.append({"name": name})
            continue
        if bone.parent_id not in names:
            raise _error(f"bone parent name is unavailable: {bone.bone_id}")
        result.append(
            {
                "name": name,
                "parent": names[bone.parent_id],
                "length": bone.length,
                "transform": "normal",
                "x": bone.x,
                "y": bone.y,
                "rotation": bone.rotation,
            }
        )
    return result, names


def build_spine_document(
    rig: RigDocument,
    coordinates: SpineCoordinatePlan,
    bind: SpineBindPlan,
    symbols: SpineSymbolView,
    atlas: SpineAtlasPlan,
    *,
    animations: dict[str, object] | None = None,
) -> SpineDocument:
    validate_rig_document(rig)
    validate_spine_coordinate_plan(coordinates)
    validate_spine_symbol_view(symbols)
    payload = rig.to_dict()
    validate_spine_bind_plan(bind, payload["bones"], payload["meshes"], coordinates)
    validate_spine_atlas_plan(
        atlas, payload["texture_pages"], payload["parts"], symbols
    )
    if animations is not None and not isinstance(animations, dict):
        raise _error("animations must be an object")

    bones, bone_names = _bone_payloads(bind, symbols)
    bind_mesh_by_id = {mesh.mesh_id: mesh for mesh in bind.meshes}
    part_by_id = {part["part_id"]: part for part in payload["parts"]}
    region_by_part = {
        region.part_id: region.name
        for page in atlas.pages
        for region in page.regions
    }
    meshes = sorted(payload["meshes"], key=lambda item: item["component_draw_rank"])
    if [mesh["component_draw_rank"] for mesh in meshes] != list(range(len(meshes))):
        raise _error("component draw ranks must be gapless from zero")

    slots = []
    attachments: dict[str, dict[str, object]] = {}
    component_records = []
    for mesh in meshes:
        mesh_id = mesh["mesh_id"]
        bind_mesh = bind_mesh_by_id.get(mesh_id)
        if bind_mesh is None:
            raise _error(f"mesh lacks a Spine BindPlan record: {mesh_id}")
        part_id = mesh["part_id"]
        component_id = mesh["component_id"]
        part = part_by_id.get(part_id)
        if part is None:
            raise _error(f"mesh references an unknown Part: {part_id}")
        sources = (part_id, component_id)
        slot_symbol = require_spine_symbol(
            symbols, kind="spine_slot", source_internal_ids=sources
        )
        attachment_key = require_spine_symbol(
            symbols,
            kind="spine_attachment_key",
            source_internal_ids=sources,
            skin_id="skin/default",
            slot_id=component_id,
        )
        attachment_object = require_spine_symbol(
            symbols, kind="spine_attachment_object", source_internal_ids=sources
        )
        region_name = region_by_part.get(part_id)
        if region_name is None:
            raise _error(f"Part lacks an atlas region: {part_id}")
        encoded = encode_spine_mesh(
            mesh,
            bind_mesh,
            source_xyxy=part["xyxy"],
            attachment_key_name=attachment_key.export_name,
            attachment_object_name=attachment_object.export_name,
            atlas_region_name=region_name,
        )
        validate_spine_encoded_mesh(encoded, mesh, bind_mesh)
        slot: dict[str, object] = {
            "name": slot_symbol.export_name,
            "bone": bone_names[bind_mesh.slot_bone_id],
            "attachment": attachment_key.export_name,
        }
        if part["setup_visibility"] == 0:
            slot["color"] = "ffffff00"
        elif part["setup_visibility"] != 1:
            raise _error(f"Part has an invalid setup visibility: {part_id}")
        slots.append(slot)
        attachments[slot_symbol.export_name] = {
            attachment_key.export_name: encoded.payload
        }
        component_records.append(
            {
                "component_draw_rank": mesh["component_draw_rank"],
                "part_id": part_id,
                "component_id": component_id,
                "mesh_id": mesh_id,
                "slot_name": slot_symbol.export_name,
                "slot_bone_id": bind_mesh.slot_bone_id,
                "slot_bone_name": bone_names[bind_mesh.slot_bone_id],
                "attachment_key_name": attachment_key.export_name,
                "attachment_object_name": attachment_object.export_name,
                "atlas_region_name": region_name,
                "weighted": encoded.weighted,
                "encoded_mesh_sha256": encoded.payload_sha256,
            }
        )

    skin_name = require_spine_symbol(
        symbols,
        kind="spine_skin",
        source_internal_ids=("skin/default",),
    ).export_name
    skeleton = {
        "skeleton": {
            "spine": SPINE_JSON_VERSION,
            "x": -coordinates.width / 2.0,
            "y": -coordinates.height / 2.0,
            "width": coordinates.width,
            "height": coordinates.height,
            "fps": 30,
            "images": "./textures/",
        },
        "bones": bones,
        "slots": slots,
        "skins": [{"name": skin_name, "attachments": attachments}],
        "animations": {} if animations is None else animations,
    }
    canonical = jcs_bytes(skeleton)
    document = SpineDocument(
        schema_version=SPINE_DOCUMENT_VERSION,
        rig_document_sha256=rig.document_sha256,
        coordinate_plan_sha256=coordinates.plan_sha256,
        bind_plan_sha256=bind.plan_sha256,
        symbol_view_sha256=symbols.view_sha256,
        atlas_plan_sha256=atlas.plan_sha256,
        component_records=tuple(component_records),
        _canonical_json=canonical,
    )
    return document


def validate_spine_document(
    document: SpineDocument,
    rig: RigDocument,
    coordinates: SpineCoordinatePlan,
    bind: SpineBindPlan,
    symbols: SpineSymbolView,
    atlas: SpineAtlasPlan,
) -> SpineDocument:
    if not isinstance(document, SpineDocument):
        raise _error("Spine document has the wrong type")
    if document.schema_version != SPINE_DOCUMENT_VERSION:
        raise _error("Spine document version is unsupported")
    payload = document.to_dict()
    if jcs_bytes(payload) != document._canonical_json:
        raise _error("Spine JSON bytes are not canonical JCS")
    expected = build_spine_document(
        rig,
        coordinates,
        bind,
        symbols,
        atlas,
        animations=payload.get("animations"),
    )
    if document != expected:
        raise _error("Spine document differs from recomputed Stage C facts")
    return document


__all__ = [
    "SPINE_DOCUMENT_VERSION",
    "SPINE_JSON_VERSION",
    "SpineDocument",
    "SpineDocumentError",
    "build_spine_document",
    "validate_spine_document",
]
