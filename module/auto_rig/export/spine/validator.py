from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from ...jcs import jcs_sha256
from ...rig_document import RigDocument, validate_rig_document
from .animations import (
    SPINE_ANIMATION_PLAN_VERSION,
    SpineAnimationPlan,
    validate_spine_animation_plan,
)
from .atlas import (
    SPINE_ATLAS_PLAN_VERSION,
    SpineAtlasPlan,
    parse_spine_atlas,
    serialize_spine_atlas,
    validate_spine_atlas_plan,
)
from .bind_plan import (
    SPINE_BIND_PLAN_VERSION,
    SpineBindPlan,
    validate_spine_bind_plan,
)
from .coordinates import (
    SPINE_COORDINATE_PLAN_VERSION,
    SpineCoordinatePlan,
    validate_spine_coordinate_plan,
)
from .mesh_encoder import (
    SPINE_MESH_ENCODER_VERSION,
    parse_weighted_vertices,
)
from .model import SPINE_JSON_VERSION, build_spine_document
from .serializer import (
    SPINE_SERIALIZATION_VERSION,
    SpineSerializationError,
    build_spine_encoding_descriptor,
    parse_spine_document,
    serialize_spine_document,
)
from .symbols import (
    SPINE_SYMBOL_VIEW_VERSION,
    SpineSymbolView,
    validate_spine_symbol_view,
)
from .uv import SPINE_42_UV_ADAPTER_VERSION

SPINE_BUNDLE_VALIDATOR_VERSION = "spine-bundle-validator-v1"
SPINE_SETUP_RECONSTRUCTION_TOLERANCE_PX = 0.1

Affine = tuple[float, float, float, float, float, float]


class SpineBundleValidationError(ValueError):
    """Raised when public Spine artifacts differ from frozen Stage C facts."""

    def __init__(self, message: str) -> None:
        super().__init__(f"invalid_spine_bundle: {message}")


def _error(message: str) -> SpineBundleValidationError:
    return SpineBundleValidationError(message)


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _finite(value: object, *, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise _error(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise _error(f"{field} must be finite")
    return result


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _error(f"{field} must be an object")
    return value


def _list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise _error(f"{field} must be a list")
    return value


def _mul(left: Affine, right: Affine) -> Affine:
    la, lb, lc, ld, ltx, lty = left
    ra, rb, rc, rd, rtx, rty = right
    return (
        la * ra + lc * rb,
        lb * ra + ld * rb,
        la * rc + lc * rd,
        lb * rc + ld * rd,
        la * rtx + lc * rty + ltx,
        lb * rtx + ld * rty + lty,
    )


def _trs(x: float, y: float, rotation: float) -> Affine:
    angle = math.radians(rotation)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return cosine, sine, -sine, cosine, x, y


def _apply(matrix: Affine, point: tuple[float, float]) -> tuple[float, float]:
    a, b, c, d, tx, ty = matrix
    x, y = point
    return a * x + c * y + tx, b * x + d * y + ty


@dataclass(frozen=True, slots=True)
class SpineBundleValidationReport:
    schema_version: str
    validated: bool
    validator_fingerprint: str
    encoding_descriptor_sha256: str
    rig_document_sha256: str
    coordinate_plan_sha256: str
    bind_plan_sha256: str
    symbol_view_sha256: str
    atlas_plan_sha256: str
    animation_plan_sha256: str
    skeleton_sha256: str
    atlas_sha256: str
    texture_page_set_sha256: str
    bone_count: int
    slot_count: int
    attachment_count: int
    animation_count: int
    page_count: int
    region_count: int
    maximum_setup_residual_px: float
    report_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "validated": self.validated,
            "validator_fingerprint": self.validator_fingerprint,
            "encoding_descriptor_sha256": self.encoding_descriptor_sha256,
            "rig_document_sha256": self.rig_document_sha256,
            "coordinate_plan_sha256": self.coordinate_plan_sha256,
            "bind_plan_sha256": self.bind_plan_sha256,
            "symbol_view_sha256": self.symbol_view_sha256,
            "atlas_plan_sha256": self.atlas_plan_sha256,
            "animation_plan_sha256": self.animation_plan_sha256,
            "skeleton_sha256": self.skeleton_sha256,
            "atlas_sha256": self.atlas_sha256,
            "texture_page_set_sha256": self.texture_page_set_sha256,
            "bone_count": self.bone_count,
            "slot_count": self.slot_count,
            "attachment_count": self.attachment_count,
            "animation_count": self.animation_count,
            "page_count": self.page_count,
            "region_count": self.region_count,
            "maximum_setup_residual_px": self.maximum_setup_residual_px,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "report_sha256": self.report_sha256}


def _validator_fingerprint() -> str:
    descriptor = build_spine_encoding_descriptor()
    return jcs_sha256(
        {
            "schema_version": SPINE_BUNDLE_VALIDATOR_VERSION,
            "serialization_version": SPINE_SERIALIZATION_VERSION,
            "encoding_descriptor_sha256": descriptor.descriptor_sha256,
            "spine_json_version": SPINE_JSON_VERSION,
            "coordinate_plan_version": SPINE_COORDINATE_PLAN_VERSION,
            "bind_plan_version": SPINE_BIND_PLAN_VERSION,
            "symbol_view_version": SPINE_SYMBOL_VIEW_VERSION,
            "mesh_encoder_version": SPINE_MESH_ENCODER_VERSION,
            "uv_adapter_version": SPINE_42_UV_ADAPTER_VERSION,
            "atlas_plan_version": SPINE_ATLAS_PLAN_VERSION,
            "animation_plan_version": SPINE_ANIMATION_PLAN_VERSION,
            "setup_tolerance_px": SPINE_SETUP_RECONSTRUCTION_TOLERANCE_PX,
        }
    )


def _world_matrices(
    parsed: Mapping[str, object],
    bind: SpineBindPlan,
) -> tuple[dict[str, Affine], float]:
    raw_bones = _list(parsed.get("bones"), field="bones")
    if len(raw_bones) != len(bind.bones):
        raise _error("bone count differs from the BindPlan")
    matrices: dict[str, Affine] = {}
    maximum_residual = 0.0
    for index, (raw, expected_bind) in enumerate(zip(raw_bones, bind.bones, strict=True)):
        bone = _mapping(raw, field="bone")
        name = bone.get("name")
        if not isinstance(name, str) or not name or name in matrices:
            raise _error("bone names must be non-empty and unique")
        if index == 0:
            if "parent" in bone:
                raise _error("root bone must not have a parent")
            matrix = _trs(
                _finite(bone.get("x", 0), field="root x"),
                _finite(bone.get("y", 0), field="root y"),
                _finite(bone.get("rotation", 0), field="root rotation"),
            )
        else:
            parent_name = bone.get("parent")
            if not isinstance(parent_name, str) or parent_name not in matrices:
                raise _error("bone parent is absent or not topological")
            transform = bone.get("transform", "normal")
            if transform != "normal":
                raise _error("Spine v1 only accepts normal bone inheritance")
            if any(
                _finite(bone.get(field, default), field=f"bone {field}") != default
                for field, default in (
                    ("scaleX", 1),
                    ("scaleY", 1),
                    ("shearX", 0),
                    ("shearY", 0),
                )
            ):
                raise _error("Spine v1 bones must use identity scale and shear")
            local = _trs(
                _finite(bone.get("x", 0), field="bone x"),
                _finite(bone.get("y", 0), field="bone y"),
                _finite(bone.get("rotation", 0), field="bone rotation"),
            )
            matrix = _mul(matrices[parent_name], local)
        matrices[name] = matrix
        maximum_residual = max(
            maximum_residual,
            max(
                abs(left - right)
                for left, right in zip(matrix, expected_bind.world_matrix, strict=True)
            ),
        )
    return matrices, maximum_residual


def _flat_numbers(value: object, *, field: str) -> list[float]:
    values = _list(value, field=field)
    if any(isinstance(item, (list, dict)) for item in values):
        raise _error(f"{field} must be a flat numeric array")
    return [_finite(item, field=field) for item in values]


def _contains_curve(value: object) -> bool:
    if isinstance(value, Mapping):
        return "curve" in value or any(_contains_curve(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_curve(child) for child in value)
    return False


def _validate_timeline_arrays(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"uvs", "triangles", "vertices"} and isinstance(child, list):
                if any(isinstance(item, (list, dict)) for item in child):
                    raise _error(f"{key} must be a flat array")
            _validate_timeline_arrays(child)
        return
    if not isinstance(value, list):
        return
    if value and all(isinstance(item, Mapping) and "time" in item for item in value):
        times = [
            _finite(_mapping(item, field="timeline key").get("time"), field="timeline time")
            for item in value
        ]
        if any(left >= right for left, right in zip(times, times[1:], strict=False)):
            raise _error("timeline times must be strictly increasing")
    for child in value:
        _validate_timeline_arrays(child)


def _setup_residual(
    parsed: Mapping[str, object],
    bind: SpineBindPlan,
    matrices: Mapping[str, Affine],
    setup_records: Sequence[Mapping[str, object]],
    atlas_regions: set[str],
) -> tuple[float, int]:
    raw_slots = _list(parsed.get("slots"), field="slots")
    expected_slot_names = [str(record["slot_name"]) for record in setup_records]
    actual_slot_names = [
        _mapping(slot, field="slot").get("name") for slot in raw_slots
    ]
    if actual_slot_names != expected_slot_names:
        raise _error("slot order differs from canonical component draw order")
    slot_by_name = {
        str(_mapping(slot, field="slot")["name"]): _mapping(slot, field="slot")
        for slot in raw_slots
    }
    raw_skins = _list(parsed.get("skins"), field="skins")
    if len(raw_skins) != 1:
        raise _error("Spine v1 requires exactly one default skin")
    skin = _mapping(raw_skins[0], field="skin")
    attachments = _mapping(skin.get("attachments"), field="skin attachments")
    bind_meshes = {mesh.mesh_id: mesh for mesh in bind.meshes}
    maximum_residual = 0.0
    attachment_count = 0
    if set(attachments) != set(expected_slot_names):
        raise _error("skin attachment slots are incomplete or stale")
    for record in setup_records:
        slot_name = str(record["slot_name"])
        attachment_key = str(record["attachment_key_name"])
        mesh_id = str(record["mesh_id"])
        slot = slot_by_name[slot_name]
        bone_name = slot.get("bone")
        if not isinstance(bone_name, str) or bone_name not in matrices:
            raise _error("slot references an unknown bone")
        if slot.get("attachment") != attachment_key:
            raise _error("slot setup attachment differs from the component record")
        slot_attachments = _mapping(
            attachments.get(slot_name), field="slot attachments"
        )
        if set(slot_attachments) != {attachment_key}:
            raise _error("slot must contain exactly its setup mesh attachment")
        mesh = _mapping(
            slot_attachments[attachment_key], field="mesh attachment"
        )
        if mesh.get("type") != "mesh":
            raise _error("setup attachment is not a mesh")
        actual_name = mesh.get("name", attachment_key)
        if not isinstance(actual_name, str) or actual_name != record["attachment_object_name"]:
            raise _error("actual attachment name differs from the global symbol")
        effective_path = mesh.get("path", actual_name)
        if (
            not isinstance(effective_path, str)
            or effective_path != record["atlas_region_name"]
            or effective_path not in atlas_regions
        ):
            raise _error("mesh attachment resolves to an unknown atlas region")
        mesh_bind = bind_meshes.get(mesh_id)
        if mesh_bind is None:
            raise _error("setup attachment has no BindPlan mesh")
        vertex_count = len(mesh_bind.vertices)
        uvs = _flat_numbers(mesh.get("uvs"), field="mesh uvs")
        if len(uvs) != vertex_count * 2:
            raise _error("mesh UV count differs from the BindPlan")
        raw_triangles = _list(mesh.get("triangles"), field="mesh triangles")
        if any(not isinstance(item, int) or isinstance(item, bool) for item in raw_triangles):
            raise _error("mesh triangles must be flat integer indices")
        if len(raw_triangles) % 3 or any(
            not 0 <= item < vertex_count for item in raw_triangles
        ):
            raise _error("mesh triangle topology is invalid")
        weighted = record.get("weighted")
        if not isinstance(weighted, bool):
            raise _error("component weighted state is invalid")
        if weighted:
            try:
                parsed_vertices = parse_weighted_vertices(
                    mesh.get("vertices"), vertex_count=vertex_count
                )
            except ValueError as exc:
                raise _error("weighted mesh vertices are invalid") from exc
            for expected_vertex, influences in zip(
                mesh_bind.vertices, parsed_vertices, strict=True
            ):
                world_x = 0.0
                world_y = 0.0
                for bone_index, x, y, weight in influences:
                    if not 0 <= bone_index < len(bind.bones):
                        raise _error("weighted mesh references an unknown bone index")
                    bone_name_for_index = next(
                        name
                        for name, expected_bone in zip(
                            matrices, bind.bones, strict=True
                        )
                        if expected_bone.index == bone_index
                    )
                    point = _apply(matrices[bone_name_for_index], (x, y))
                    world_x += point[0] * weight
                    world_y += point[1] * weight
                maximum_residual = max(
                    maximum_residual,
                    math.dist((world_x, world_y), expected_vertex.spine_position),
                )
        else:
            vertices = _flat_numbers(mesh.get("vertices"), field="mesh vertices")
            if len(vertices) != vertex_count * 2:
                raise _error("unweighted mesh vertex count differs from the BindPlan")
            for index, expected_vertex in enumerate(mesh_bind.vertices):
                point = _apply(
                    matrices[bone_name],
                    (vertices[index * 2], vertices[index * 2 + 1]),
                )
                maximum_residual = max(
                    maximum_residual,
                    math.dist(point, expected_vertex.spine_position),
                )
        attachment_count += 1
    return maximum_residual, attachment_count


def _validate_atlas_bytes(payload: bytes, plan: SpineAtlasPlan) -> tuple[int, int]:
    expected = serialize_spine_atlas(plan)
    if payload != expected:
        raise _error("atlas bytes differ from the canonical SpineAtlasPlan")
    try:
        parsed = parse_spine_atlas(payload)
    except ValueError as exc:
        raise _error("atlas bytes are not valid Spine ASCII/LF") from exc
    if len(parsed.pages) != len(plan.pages):
        raise _error("atlas page count differs from the SpineAtlasPlan")
    region_count = 0
    for parsed_page, expected_page in zip(parsed.pages, plan.pages, strict=True):
        if (
            parsed_page.path != expected_page.path
            or parsed_page.width != expected_page.width
            or parsed_page.height != expected_page.height
            or parsed_page.pma is not False
        ):
            raise _error("atlas page metadata differs from the SpineAtlasPlan")
        expected_regions = {
            region.name: (
                region.x,
                region.y,
                region.width,
                region.height,
            )
            for region in expected_page.regions
        }
        actual_regions = {region.name: region.bounds for region in parsed_page.regions}
        if actual_regions != expected_regions:
            raise _error("atlas region inventory or bounds differ from the plan")
        region_count += len(parsed_page.regions)
    return len(parsed.pages), region_count


def _validate_pages(
    pages: Mapping[str, bytes],
    atlas: SpineAtlasPlan,
) -> str:
    expected_paths = {page.path for page in atlas.pages}
    if set(pages) != expected_paths:
        raise _error("texture page inventory differs from the atlas")
    records = []
    for page in atlas.pages:
        payload = pages.get(page.path)
        if not isinstance(payload, bytes):
            raise _error("texture page payload must be bytes")
        digest = _sha256(payload)
        if digest != page.source_encoded_png_sha256:
            raise _error("texture page bytes differ from Stage C canonical PNG")
        records.append({"path": page.path, "size": len(payload), "sha256": digest})
    return jcs_sha256(records)


def validate_spine_bundle(
    skeleton_bytes: bytes,
    atlas_bytes: bytes,
    texture_pages: Mapping[str, bytes],
    rig: RigDocument,
    coordinates: SpineCoordinatePlan,
    bind: SpineBindPlan,
    symbols: SpineSymbolView,
    atlas: SpineAtlasPlan,
    animations: SpineAnimationPlan,
) -> SpineBundleValidationReport:
    try:
        parsed = parse_spine_document(skeleton_bytes)
    except SpineSerializationError as exc:
        raise _error(str(exc)) from exc
    if _contains_curve(parsed):
        raise _error("Spine linear timeline contains a forbidden curve field")
    _validate_timeline_arrays(parsed)
    metadata = _mapping(parsed.get("skeleton"), field="skeleton metadata")
    if metadata.get("spine") != SPINE_JSON_VERSION:
        raise _error("skeleton metadata is not exact Spine 4.2")

    validate_rig_document(rig)
    validate_spine_coordinate_plan(coordinates)
    validate_spine_symbol_view(symbols)
    rig_payload = rig.to_dict()
    validate_spine_bind_plan(bind, rig_payload["bones"], rig_payload["meshes"], coordinates)
    validate_spine_atlas_plan(
        atlas, rig_payload["texture_pages"], rig_payload["parts"], symbols
    )
    setup = build_spine_document(rig, coordinates, bind, symbols, atlas)
    validate_spine_animation_plan(
        animations, rig, coordinates, bind, symbols, setup
    )
    expected_document = build_spine_document(
        rig,
        coordinates,
        bind,
        symbols,
        atlas,
        animations=animations.animations,
    )
    expected_bytes = serialize_spine_document(expected_document)
    if skeleton_bytes != expected_bytes or parsed != expected_document.to_dict():
        raise _error("skeleton JSON differs from recomputed Stage C facts")

    page_count, region_count = _validate_atlas_bytes(atlas_bytes, atlas)
    texture_page_set_sha256 = _validate_pages(texture_pages, atlas)
    matrices, bone_residual = _world_matrices(parsed, bind)
    atlas_regions = {
        region.name for page in atlas.pages for region in page.regions
    }
    mesh_residual, attachment_count = _setup_residual(
        parsed,
        bind,
        matrices,
        expected_document.component_records,
        atlas_regions,
    )
    maximum_residual = max(bone_residual, mesh_residual)
    if maximum_residual > SPINE_SETUP_RECONSTRUCTION_TOLERANCE_PX:
        raise _error("setup reconstruction exceeds 0.1 px")

    descriptor = build_spine_encoding_descriptor()
    raw_slots = _list(parsed["slots"], field="slots")
    raw_bones = _list(parsed["bones"], field="bones")
    raw_animations = _mapping(parsed["animations"], field="animations")
    provisional = SpineBundleValidationReport(
        schema_version=SPINE_BUNDLE_VALIDATOR_VERSION,
        validated=True,
        validator_fingerprint=_validator_fingerprint(),
        encoding_descriptor_sha256=descriptor.descriptor_sha256,
        rig_document_sha256=rig.document_sha256,
        coordinate_plan_sha256=coordinates.plan_sha256,
        bind_plan_sha256=bind.plan_sha256,
        symbol_view_sha256=symbols.view_sha256,
        atlas_plan_sha256=atlas.plan_sha256,
        animation_plan_sha256=animations.plan_sha256,
        skeleton_sha256=_sha256(skeleton_bytes),
        atlas_sha256=_sha256(atlas_bytes),
        texture_page_set_sha256=texture_page_set_sha256,
        bone_count=len(raw_bones),
        slot_count=len(raw_slots),
        attachment_count=attachment_count,
        animation_count=len(raw_animations),
        page_count=page_count,
        region_count=region_count,
        maximum_setup_residual_px=maximum_residual,
        report_sha256="",
    )
    return replace(
        provisional,
        report_sha256=jcs_sha256(provisional.semantic_payload()),
    )


__all__ = [
    "SPINE_BUNDLE_VALIDATOR_VERSION",
    "SPINE_SETUP_RECONSTRUCTION_TOLERANCE_PX",
    "SpineBundleValidationError",
    "SpineBundleValidationReport",
    "validate_spine_bundle",
]
