from __future__ import annotations

import math
from dataclasses import dataclass

from .jcs import jcs_sha256
from .joint_pipeline import StageAJointPlan, validate_stage_a_joint_plan

BONE_SPEC_REGISTRY_VERSION = "bone-spec-registry-v1"
BONE_GRAPH_PLAN_VERSION = "bone-graph-plan-v1"
BONE_MIN_LENGTH_PX = 1.0 / 256.0
BONE_MAX_CANVAS_DIAGONALS = 2.0


class BoneGraphError(ValueError):
    """Raised when resolved joints cannot form the declarative bone graph."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _error(code: str, message: str) -> BoneGraphError:
    return BoneGraphError(code, message)


@dataclass(frozen=True, slots=True)
class BoneSpec:
    spec_id: str
    bone_id: str
    declared_parent_id: str | None
    head_joint_id: str | None
    tail_joint_id: str | None
    role: str

    def to_dict(self) -> dict[str, object]:
        return {
            "spec_id": self.spec_id,
            "bone_id": self.bone_id,
            "declared_parent_id": self.declared_parent_id,
            "head_joint_id": self.head_joint_id,
            "tail_joint_id": self.tail_joint_id,
            "role": self.role,
        }


def _spec(
    slug: str,
    parent: str | None,
    head: str | None,
    tail: str | None,
    role: str,
) -> BoneSpec:
    return BoneSpec(
        spec_id=f"bone-spec/{slug}",
        bone_id=f"bone/{slug}",
        declared_parent_id=None if parent is None else f"bone/{parent}",
        head_joint_id=None if head is None else f"joint/{head}",
        tail_joint_id=None if tail is None else f"joint/{tail}",
        role=role,
    )


BONE_SPECS = (
    _spec("root", None, None, None, "synthetic_root"),
    _spec("lower_torso", "root", "pelvis", "spine", "lower_torso"),
    _spec("torso", "lower_torso", "spine", "neck", "torso"),
    _spec("neck", "torso", "neck", "head_base", "neck"),
    _spec("head", "neck", "head_base", "head_top", "head"),
    _spec(
        "upper_arm.xmin",
        "torso",
        "shoulder.xmin",
        "elbow.xmin",
        "upper_arm",
    ),
    _spec(
        "forearm.xmin",
        "upper_arm.xmin",
        "elbow.xmin",
        "wrist.xmin",
        "forearm",
    ),
    _spec(
        "hand.xmin",
        "forearm.xmin",
        "wrist.xmin",
        "hand_tip.xmin",
        "hand",
    ),
    _spec(
        "upper_arm.xmax",
        "torso",
        "shoulder.xmax",
        "elbow.xmax",
        "upper_arm",
    ),
    _spec(
        "forearm.xmax",
        "upper_arm.xmax",
        "elbow.xmax",
        "wrist.xmax",
        "forearm",
    ),
    _spec(
        "hand.xmax",
        "forearm.xmax",
        "wrist.xmax",
        "hand_tip.xmax",
        "hand",
    ),
    _spec("thigh.xmin", "lower_torso", "hip.xmin", "knee.xmin", "thigh"),
    _spec("shin.xmin", "thigh.xmin", "knee.xmin", "ankle.xmin", "shin"),
    _spec("foot.xmin", "shin.xmin", "ankle.xmin", "toe.xmin", "foot"),
    _spec("thigh.xmax", "lower_torso", "hip.xmax", "knee.xmax", "thigh"),
    _spec("shin.xmax", "thigh.xmax", "knee.xmax", "ankle.xmax", "shin"),
    _spec("foot.xmax", "shin.xmax", "ankle.xmax", "toe.xmax", "foot"),
)
BONE_IDS = tuple(spec.bone_id for spec in BONE_SPECS)


@dataclass(frozen=True, slots=True)
class RigBone:
    bone_id: str
    spec_id: str
    parent_id: str | None
    declared_parent_id: str | None
    head_joint_id: str | None
    tail_joint_id: str | None
    role: str
    head: tuple[float, float] | None
    tail: tuple[float, float] | None
    length: float
    parent_promotion: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "bone_id": self.bone_id,
            "spec_id": self.spec_id,
            "parent_id": self.parent_id,
            "declared_parent_id": self.declared_parent_id,
            "head_joint_id": self.head_joint_id,
            "tail_joint_id": self.tail_joint_id,
            "role": self.role,
            "head": list(self.head) if self.head is not None else None,
            "tail": list(self.tail) if self.tail is not None else None,
            "length": self.length,
            "parent_promotion": self.parent_promotion,
        }


@dataclass(frozen=True, slots=True)
class BoneGraphDiagnostic:
    bone_id: str
    code: str
    joint_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "bone_id": self.bone_id,
            "code": self.code,
            "joint_ids": list(self.joint_ids),
        }


@dataclass(frozen=True, slots=True)
class BoneGraphPlan:
    schema_version: str
    registry_version: str
    stage_a_joint_plan_sha256: str
    canvas_width: int
    canvas_height: int
    minimum_non_root_length_px: float
    maximum_canvas_diagonals: float
    specs: tuple[BoneSpec, ...]
    bones: tuple[RigBone, ...]
    diagnostics: tuple[BoneGraphDiagnostic, ...]
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "registry_version": self.registry_version,
            "stage_a_joint_plan_sha256": self.stage_a_joint_plan_sha256,
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "minimum_non_root_length_px": self.minimum_non_root_length_px,
            "maximum_canvas_diagonals": self.maximum_canvas_diagonals,
            "specs": [spec.to_dict() for spec in self.specs],
            "bones": [bone.to_dict() for bone in self.bones],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


def _effective_parent(
    spec: BoneSpec,
    *,
    emitted: set[str],
    spec_by_id: dict[str, BoneSpec],
) -> str:
    candidate = spec.declared_parent_id
    while candidate is not None:
        if candidate in emitted:
            return candidate
        candidate = spec_by_id[candidate].declared_parent_id
    return "bone/root"


def _root_bone() -> RigBone:
    spec = BONE_SPECS[0]
    return RigBone(
        bone_id=spec.bone_id,
        spec_id=spec.spec_id,
        parent_id=None,
        declared_parent_id=None,
        head_joint_id=None,
        tail_joint_id=None,
        role=spec.role,
        head=None,
        tail=None,
        length=0.0,
        parent_promotion=False,
    )


def validate_bone_graph(plan: BoneGraphPlan) -> BoneGraphPlan:
    """Validate the structural and numeric invariants of a B bone plan."""

    if not isinstance(plan, BoneGraphPlan):
        raise _error("invalid_bone_graph", "bone graph must use BoneGraphPlan")
    if plan.schema_version != BONE_GRAPH_PLAN_VERSION:
        raise _error("invalid_bone_graph", "unsupported bone graph version")
    if plan.registry_version != BONE_SPEC_REGISTRY_VERSION or plan.specs != BONE_SPECS:
        raise _error("invalid_bone_graph", "bone registry differs from the built-in contract")
    if (
        plan.minimum_non_root_length_px != BONE_MIN_LENGTH_PX
        or plan.maximum_canvas_diagonals != BONE_MAX_CANVAS_DIAGONALS
    ):
        raise _error("invalid_bone_graph", "bone length limits differ from v1")
    if (
        type(plan.canvas_width) is not int
        or type(plan.canvas_height) is not int
        or plan.canvas_width <= 0
        or plan.canvas_height <= 0
    ):
        raise _error("invalid_bone_graph", "bone graph canvas is invalid")
    if plan.plan_sha256 != jcs_sha256(plan.semantic_payload()):
        raise _error("invalid_bone_graph", "bone graph digest mismatch")
    if not plan.bones or plan.bones[0] != _root_bone():
        raise _error("invalid_bone_graph", "bone graph lacks the canonical synthetic root")
    bone_ids = tuple(bone.bone_id for bone in plan.bones)
    if len(bone_ids) != len(set(bone_ids)) or any(item not in BONE_IDS for item in bone_ids):
        raise _error("invalid_bone_graph", "bone identities are invalid")
    expected_order = tuple(bone_id for bone_id in BONE_IDS if bone_id in set(bone_ids))
    if bone_ids != expected_order:
        raise _error("invalid_bone_graph", "bones are not in canonical topological order")
    spec_by_bone = {spec.bone_id: spec for spec in BONE_SPECS}
    emitted: set[str] = set()
    maximum = math.hypot(plan.canvas_width, plan.canvas_height) * plan.maximum_canvas_diagonals
    for bone in plan.bones:
        spec = spec_by_bone[bone.bone_id]
        if bone.spec_id != spec.spec_id or bone.role != spec.role:
            raise _error("invalid_bone_graph", "bone differs from its declarative spec")
        if bone.bone_id == "bone/root":
            emitted.add(bone.bone_id)
            continue
        if bone.parent_id not in emitted:
            raise _error("invalid_bone_graph", "bone parent is absent or follows its child")
        if bone.head_joint_id != spec.head_joint_id or bone.tail_joint_id != spec.tail_joint_id:
            raise _error("invalid_bone_graph", "bone joint references differ from its spec")
        if bone.head is None or bone.tail is None:
            raise _error("invalid_bone_graph", "non-root bone lacks rest endpoints")
        expected_length = math.dist(bone.head, bone.tail)
        if not math.isfinite(bone.length) or abs(bone.length - expected_length) > 1e-9:
            raise _error("invalid_bone_graph", "bone length differs from its endpoints")
        if not BONE_MIN_LENGTH_PX <= bone.length <= maximum:
            raise _error("invalid_bone_length", f"bone length is outside v1 limits: {bone.bone_id}")
        expected_parent = _effective_parent(
            spec,
            emitted=emitted,
            spec_by_id=spec_by_bone,
        )
        if bone.parent_id != expected_parent:
            raise _error("invalid_bone_graph", "bone parent promotion is not canonical")
        if bone.parent_promotion != (bone.parent_id != bone.declared_parent_id):
            raise _error("invalid_bone_graph", "bone parent-promotion flag is inconsistent")
        emitted.add(bone.bone_id)
    return plan


def build_bone_graph(joints: StageAJointPlan) -> BoneGraphPlan:
    """Build the frozen BoneSpec subset supported by resolved Stage A joints."""

    validate_stage_a_joint_plan(joints)
    resolutions = {item.joint_id: item for item in joints.joints.resolutions}
    spec_by_bone = {spec.bone_id: spec for spec in BONE_SPECS}
    maximum = math.hypot(joints.joints.canvas_width, joints.joints.canvas_height)
    maximum *= BONE_MAX_CANVAS_DIAGONALS
    bones = [_root_bone()]
    emitted = {"bone/root"}
    diagnostics: list[BoneGraphDiagnostic] = []
    for spec in BONE_SPECS[1:]:
        head = resolutions[spec.head_joint_id]
        tail = resolutions[spec.tail_joint_id]
        if head.status != "resolved" or tail.status != "resolved":
            diagnostics.append(
                BoneGraphDiagnostic(
                    bone_id=spec.bone_id,
                    code="bone_joint_unresolved",
                    joint_ids=(spec.head_joint_id, spec.tail_joint_id),
                )
            )
            continue
        head_point = (float(head.x), float(head.y))
        tail_point = (float(tail.x), float(tail.y))
        length = math.dist(head_point, tail_point)
        if not BONE_MIN_LENGTH_PX <= length <= maximum:
            if head.source == "override" or tail.source == "override":
                raise _error(
                    "invalid_bone_length",
                    f"override produces an invalid bone length: {spec.bone_id}",
                )
            diagnostics.append(
                BoneGraphDiagnostic(
                    bone_id=spec.bone_id,
                    code="invalid_bone_length",
                    joint_ids=(spec.head_joint_id, spec.tail_joint_id),
                )
            )
            continue
        parent_id = _effective_parent(
            spec,
            emitted=emitted,
            spec_by_id=spec_by_bone,
        )
        bones.append(
            RigBone(
                bone_id=spec.bone_id,
                spec_id=spec.spec_id,
                parent_id=parent_id,
                declared_parent_id=spec.declared_parent_id,
                head_joint_id=spec.head_joint_id,
                tail_joint_id=spec.tail_joint_id,
                role=spec.role,
                head=head_point,
                tail=tail_point,
                length=length,
                parent_promotion=parent_id != spec.declared_parent_id,
            )
        )
        emitted.add(spec.bone_id)
    values = {
        "schema_version": BONE_GRAPH_PLAN_VERSION,
        "registry_version": BONE_SPEC_REGISTRY_VERSION,
        "stage_a_joint_plan_sha256": joints.plan_sha256,
        "canvas_width": joints.joints.canvas_width,
        "canvas_height": joints.joints.canvas_height,
        "minimum_non_root_length_px": BONE_MIN_LENGTH_PX,
        "maximum_canvas_diagonals": BONE_MAX_CANVAS_DIAGONALS,
        "specs": BONE_SPECS,
        "bones": tuple(bones),
        "diagnostics": tuple(diagnostics),
    }
    provisional = BoneGraphPlan(**values, plan_sha256="")
    plan = BoneGraphPlan(
        **values,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )
    return validate_bone_graph(plan)


__all__ = [
    "BONE_GRAPH_PLAN_VERSION",
    "BONE_IDS",
    "BONE_MAX_CANVAS_DIAGONALS",
    "BONE_MIN_LENGTH_PX",
    "BONE_SPEC_REGISTRY_VERSION",
    "BONE_SPECS",
    "BoneGraphDiagnostic",
    "BoneGraphError",
    "BoneGraphPlan",
    "BoneSpec",
    "RigBone",
    "build_bone_graph",
    "validate_bone_graph",
]
