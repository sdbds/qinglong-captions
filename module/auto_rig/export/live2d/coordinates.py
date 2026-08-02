from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ...jcs import jcs_sha256
from ...rig_document import RigDocument, validate_rig_document
from .attestation import load_live2d_frame_attestation
from .binding_plan import (
    Live2DBindingPlan,
    validate_live2d_binding_plan,
)
from .frame_kernel import FRAME_KERNEL_VERSION, ROTATION_STACK_SEMANTICS_VERSION

LIVE2D_COORDINATE_PLAN_VERSION = "live2d-coordinate-plan-v1"
LIVE2D_COORDINATE_SCHEMA_VERSION = "live2d-frames-v1"


class Live2DCoordinateError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(f"invalid_live2d_coordinate_plan: {message}")


def _error(message: str) -> Live2DCoordinateError:
    return Live2DCoordinateError(message)


def _point(value: Sequence[object], *, field: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise _error(f"{field} must be an x/y pair")
    result = []
    for coordinate in value:
        if (
            not isinstance(coordinate, (int, float))
            or isinstance(coordinate, bool)
            or not math.isfinite(float(coordinate))
        ):
            raise _error(f"{field} must contain finite numbers")
        result.append(float(coordinate))
    return result[0], result[1]


def _clean(value: float) -> float:
    return 0.0 if abs(value) < 1e-12 else value


@dataclass(frozen=True, slots=True)
class Live2DRotationFrame:
    instance_id: str
    bone_id: str
    parent_instance_id: str | None
    input_frame: str
    output_frame: str
    pivot_canvas: tuple[float, float]
    origin: tuple[float, float]
    scale: float
    stack_rank: int

    def to_dict(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "bone_id": self.bone_id,
            "parent_instance_id": self.parent_instance_id,
            "input_frame": self.input_frame,
            "output_frame": self.output_frame,
            "pivot_canvas": list(self.pivot_canvas),
            "origin": list(self.origin),
            "scale": self.scale,
            "stack_rank": self.stack_rank,
        }


@dataclass(frozen=True, slots=True)
class Live2DCoordinatePlan:
    schema_version: str
    coordinate_schema_version: str
    frame_kernel_version: str
    rotation_stack_semantics_version: str
    binding_plan_sha256: str
    frame_contract_sha256: str
    width: int
    height: int
    ppu: float
    rotation_frames: tuple[Live2DRotationFrame, ...]
    tested_point_count: int
    maximum_round_trip_error: float
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "coordinate_schema_version": self.coordinate_schema_version,
            "frame_kernel_version": self.frame_kernel_version,
            "rotation_stack_semantics_version": self.rotation_stack_semantics_version,
            "binding_plan_sha256": self.binding_plan_sha256,
            "frame_contract_sha256": self.frame_contract_sha256,
            "width": self.width,
            "height": self.height,
            "ppu": self.ppu,
            "rotation_frames": [frame.to_dict() for frame in self.rotation_frames],
            "tested_point_count": self.tested_point_count,
            "maximum_round_trip_error": self.maximum_round_trip_error,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "plan_sha256": self.plan_sha256}


def canvas_to_moc_root(
    plan: Live2DCoordinatePlan, point: Sequence[object]
) -> tuple[float, float]:
    x, y = _point(point, field="canvas point")
    return (
        _clean((x - plan.width / 2.0) / plan.ppu),
        _clean((y - plan.height / 2.0) / plan.ppu),
    )


def moc_root_to_canvas(
    plan: Live2DCoordinatePlan, point: Sequence[object]
) -> tuple[float, float]:
    x, y = _point(point, field="MOC root point")
    return (
        _clean(x * plan.ppu + plan.width / 2.0),
        _clean(y * plan.ppu + plan.height / 2.0),
    )


def _frames(plan: Live2DCoordinatePlan) -> dict[str, Live2DRotationFrame]:
    return {frame.instance_id: frame for frame in plan.rotation_frames}


def _frame(
    plan: Live2DCoordinatePlan,
    instance_id: str,
) -> Live2DRotationFrame:
    value = _frames(plan).get(instance_id)
    if value is None:
        raise _error(f"unknown RotationDeformer instance: {instance_id}")
    return value


def _local_to_canvas(
    plan: Live2DCoordinatePlan,
    instance_id: str,
    point: tuple[float, float],
) -> tuple[float, float]:
    frame = _frame(plan, instance_id)
    parent_point = (
        frame.origin[0] + frame.scale * point[0],
        frame.origin[1] + frame.scale * point[1],
    )
    if frame.parent_instance_id is None:
        return moc_root_to_canvas(plan, parent_point)
    return _local_to_canvas(plan, frame.parent_instance_id, parent_point)


def _canvas_to_local(
    plan: Live2DCoordinatePlan,
    instance_id: str,
    point: tuple[float, float],
) -> tuple[float, float]:
    frame = _frame(plan, instance_id)
    parent_point = (
        canvas_to_moc_root(plan, point)
        if frame.parent_instance_id is None
        else _canvas_to_local(plan, frame.parent_instance_id, point)
    )
    return (
        _clean((parent_point[0] - frame.origin[0]) / frame.scale),
        _clean((parent_point[1] - frame.origin[1]) / frame.scale),
    )


def canvas_to_artmesh_local(
    plan: Live2DCoordinatePlan,
    parent_instance_id: str | None,
    point: Sequence[object],
) -> tuple[float, float]:
    normalized = _point(point, field="ArtMesh canvas point")
    if parent_instance_id is None:
        return canvas_to_moc_root(plan, normalized)
    return _canvas_to_local(plan, parent_instance_id, normalized)


def live2d_local_to_canvas(
    plan: Live2DCoordinatePlan,
    parent_instance_id: str | None,
    point: Sequence[object],
) -> tuple[float, float]:
    normalized = _point(point, field="ArtMesh local point")
    if parent_instance_id is None:
        return moc_root_to_canvas(plan, normalized)
    return _local_to_canvas(plan, parent_instance_id, normalized)


def _attestation_digest() -> str:
    path = Path(__file__).with_name("attestations") / "live2d-frames-v1.json"
    attestation = load_live2d_frame_attestation(
        path.read_bytes()
    )
    digest = attestation.get("live2d_frame_contract_digest")
    if not isinstance(digest, str):
        raise _error("packaged frame attestation has no contract digest")
    return digest


def _assemble(
    rig: RigDocument,
    bindings: Live2DBindingPlan,
) -> Live2DCoordinatePlan:
    payload = rig.to_dict()
    canvas = payload["canvas"]
    if not isinstance(canvas, Mapping):
        raise _error("Rig canvas is invalid")
    width = canvas.get("width")
    height = canvas.get("height")
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or isinstance(height, bool)
        or width <= 0
        or height <= 0
        or canvas.get("origin") != "top_left"
        or canvas.get("y_axis") != "down"
    ):
        raise _error("Rig canvas differs from the frozen LayerDiff contract")
    ppu = float(max(width, height))
    bones = {
        bone["bone_id"]: bone
        for bone in payload["bones"]
        if isinstance(bone, Mapping) and isinstance(bone.get("bone_id"), str)
    }
    instances = {instance.instance_id: instance for instance in bindings.rotation_instances}
    frames: list[Live2DRotationFrame] = []
    frame_by_id: dict[str, Live2DRotationFrame] = {}
    for instance in bindings.rotation_instances:
        bone = bones.get(instance.bone_id)
        if bone is None or bone.get("head") is None:
            raise _error("RotationDeformer bone has no finite pivot")
        pivot = _point(bone["head"], field="bone head")
        if instance.parent_instance_id is None:
            origin = (
                (pivot[0] - width / 2.0) / ppu,
                (pivot[1] - height / 2.0) / ppu,
            )
            scale = 1.0 / ppu
            output_frame = "MOC_ROOT"
        else:
            parent_instance = instances.get(instance.parent_instance_id)
            parent_frame = frame_by_id.get(instance.parent_instance_id)
            if parent_instance is None or parent_frame is None:
                raise _error("RotationDeformer parent is absent or out of order")
            parent_bone = bones.get(parent_instance.bone_id)
            if parent_bone is None or parent_bone.get("head") is None:
                raise _error("parent RotationDeformer bone has no pivot")
            parent_pivot = _point(parent_bone["head"], field="parent bone head")
            origin = (
                _clean(pivot[0] - parent_pivot[0]),
                _clean(pivot[1] - parent_pivot[1]),
            )
            scale = 1.0
            output_frame = "ROTATION_LOCAL"
        frame = Live2DRotationFrame(
            instance_id=instance.instance_id,
            bone_id=instance.bone_id,
            parent_instance_id=instance.parent_instance_id,
            input_frame="ROTATION_LOCAL",
            output_frame=output_frame,
            pivot_canvas=pivot,
            origin=(_clean(origin[0]), _clean(origin[1])),
            scale=scale,
            stack_rank=instance.stack_rank,
        )
        frames.append(frame)
        frame_by_id[frame.instance_id] = frame

    provisional = Live2DCoordinatePlan(
        schema_version=LIVE2D_COORDINATE_PLAN_VERSION,
        coordinate_schema_version=LIVE2D_COORDINATE_SCHEMA_VERSION,
        frame_kernel_version=FRAME_KERNEL_VERSION,
        rotation_stack_semantics_version=ROTATION_STACK_SEMANTICS_VERSION,
        binding_plan_sha256=bindings.plan_sha256,
        frame_contract_sha256=_attestation_digest(),
        width=width,
        height=height,
        ppu=ppu,
        rotation_frames=tuple(frames),
        tested_point_count=0,
        maximum_round_trip_error=0.0,
        plan_sha256="",
    )
    attachment_by_mesh = {
        attachment.mesh_id: attachment for attachment in bindings.artmesh_attachments
    }
    maximum = 0.0
    count = 0
    for mesh in payload["meshes"]:
        attachment = attachment_by_mesh.get(mesh["mesh_id"])
        if attachment is None:
            raise _error("coordinate plan lacks an ArtMesh attachment")
        for vertex in mesh["vertices"]:
            point = _point(vertex["position"], field="mesh vertex")
            local = canvas_to_artmesh_local(
                provisional, attachment.parent_instance_id, point
            )
            recovered = live2d_local_to_canvas(
                provisional, attachment.parent_instance_id, local
            )
            maximum = max(
                maximum, math.hypot(recovered[0] - point[0], recovered[1] - point[1])
            )
            count += 1
    values = {
        **{
            field: getattr(provisional, field)
            for field in (
                "schema_version",
                "coordinate_schema_version",
                "frame_kernel_version",
                "rotation_stack_semantics_version",
                "binding_plan_sha256",
                "frame_contract_sha256",
                "width",
                "height",
                "ppu",
                "rotation_frames",
            )
        },
        "tested_point_count": count,
        "maximum_round_trip_error": maximum,
    }
    without_digest = Live2DCoordinatePlan(**values, plan_sha256="")
    return Live2DCoordinatePlan(
        **values, plan_sha256=jcs_sha256(without_digest.semantic_payload())
    )


def build_live2d_coordinate_plan(
    rig: RigDocument,
    bindings: Live2DBindingPlan,
) -> Live2DCoordinatePlan:
    validate_rig_document(rig)
    plan = _assemble(rig, bindings)
    if plan.maximum_round_trip_error > 0.1:
        raise _error("coordinate round-trip residual exceeds 0.1 px")
    return plan


def validate_live2d_coordinate_plan(
    plan: Live2DCoordinatePlan,
    rig: RigDocument,
    bindings: Live2DBindingPlan,
) -> Live2DCoordinatePlan:
    if not isinstance(plan, Live2DCoordinatePlan):
        raise _error("coordinate plan has the wrong type")
    validate_rig_document(rig)
    if plan != _assemble(rig, bindings):
        raise _error("coordinate plan differs from the canonical binding projection")
    if plan.maximum_round_trip_error > 0.1:
        raise _error("coordinate round-trip residual exceeds 0.1 px")
    if plan.plan_sha256 != jcs_sha256(plan.semantic_payload()):
        raise _error("coordinate plan digest mismatch")
    return plan


__all__ = [
    "LIVE2D_COORDINATE_PLAN_VERSION",
    "LIVE2D_COORDINATE_SCHEMA_VERSION",
    "Live2DCoordinateError",
    "Live2DCoordinatePlan",
    "Live2DRotationFrame",
    "build_live2d_coordinate_plan",
    "canvas_to_artmesh_local",
    "canvas_to_moc_root",
    "live2d_local_to_canvas",
    "moc_root_to_canvas",
    "validate_live2d_coordinate_plan",
]
