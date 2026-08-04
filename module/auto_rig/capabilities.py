from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .anatomy import ANATOMY_MASK_PLAN_VERSION, AnatomyMaskPlan
from .jcs import jcs_sha256
from .native_variants import NativeVariantSet
from .preset_library import PresetLibraryPlan, validate_preset_library_plan
from .rig_geometry import RigGeometryCache, validate_rig_geometry_cache

CAPABILITY_PLAN_VERSION = "capability-plan-v1"

CapabilityStatus = Literal["available", "unavailable"]
CapabilityTier = Literal[
    "structural",
    "native",
    "procedural",
    "procedural_silhouette",
    "unavailable",
]


class CapabilityPlanError(ValueError):
    """Raised when capability facts do not match frozen geometry inputs."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _error(message: str) -> CapabilityPlanError:
    return CapabilityPlanError("invalid_capability_plan", message)


@dataclass(frozen=True, slots=True)
class RigCapability:
    capability_id: str
    preset_id: str
    status: CapabilityStatus
    quality_tier: CapabilityTier
    required_control_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    capability_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "capability_id": self.capability_id,
            "preset_id": self.preset_id,
            "status": self.status,
            "quality_tier": self.quality_tier,
            "required_control_ids": list(self.required_control_ids),
            "evidence_ids": list(self.evidence_ids),
            "reason_codes": list(self.reason_codes),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.semantic_payload(),
            "capability_sha256": self.capability_sha256,
        }


@dataclass(frozen=True, slots=True)
class CapabilityPlan:
    schema_version: str
    rig_geometry_cache_sha256: str
    anatomy_plan_sha256: str
    preset_library_sha256: str
    native_variant_set_sha256: str
    capabilities: tuple[RigCapability, ...]
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "rig_geometry_cache_sha256": self.rig_geometry_cache_sha256,
            "anatomy_plan_sha256": self.anatomy_plan_sha256,
            "preset_library_sha256": self.preset_library_sha256,
            "native_variant_set_sha256": self.native_variant_set_sha256,
            "capabilities": [item.to_dict() for item in self.capabilities],
        }


def _validate_anatomy_plan(cache: RigGeometryCache, anatomy: AnatomyMaskPlan) -> None:
    if not isinstance(anatomy, AnatomyMaskPlan):
        raise _error("capability derivation requires AnatomyMaskPlan")
    if anatomy.schema_version != ANATOMY_MASK_PLAN_VERSION:
        raise _error("anatomy plan version is unsupported")
    if anatomy.plan_sha256 != jcs_sha256(anatomy.semantic_payload()):
        raise _error("anatomy plan digest mismatch")
    if anatomy.canvas_edge != cache.target.canvas_width:
        raise _error("anatomy canvas differs from geometry cache")
    if anatomy.plan_sha256 != cache.joint_plan.anatomy_plan_sha256:
        raise _error("anatomy plan differs from Stage A joint evidence")


def _validate_native_set(
    cache: RigGeometryCache,
    native_variant_set: NativeVariantSet | None,
) -> None:
    native_parts = tuple(part for part in cache.parts if part.source_kind == "native_variant")
    if native_variant_set is None:
        if native_parts:
            raise _error("native render Parts require the authenticated NativeVariant set")
        return
    if (
        native_variant_set.native_variant_set_sha256 != jcs_sha256(native_variant_set.semantic_payload())
        or native_variant_set.native_variant_set_sha256 != cache.native_variant_set_sha256
    ):
        raise _error("NativeVariant set differs from the geometry cache")
    entry_by_id = {entry.variant_id: entry for entry in native_variant_set.entries}
    if len(entry_by_id) != len(native_variant_set.entries):
        raise _error("NativeVariant set has duplicate IDs")
    for part in native_parts:
        entry = entry_by_id.get(part.variant_id or "")
        if (
            entry is None
            or entry.part_id != part.part_id
            or entry.semantic_role != part.semantic_role
            or entry.base_part_ids != part.base_part_ids
            or entry.draw_anchor_part_id != part.draw_anchor_part_id
        ):
            raise _error("native render Part differs from its authenticated entry")


def _preset_controls(presets: PresetLibraryPlan) -> dict[str, tuple[str, ...]]:
    result = {clip.preset_id: tuple(curve.control_id for curve in clip.control_curves) for clip in presets.clips}
    result.update(
        {expression.preset_id: tuple(value.control_id for value in expression.values) for expression in presets.expressions}
    )
    return result


def _capability(
    preset_id: str,
    controls: tuple[str, ...],
    *,
    status: CapabilityStatus,
    tier: CapabilityTier,
    evidence: set[str] | tuple[str, ...] = (),
    reasons: set[str] | tuple[str, ...] = (),
) -> RigCapability:
    values = {
        "capability_id": f"capability/preset.{preset_id}",
        "preset_id": preset_id,
        "status": status,
        "quality_tier": tier,
        "required_control_ids": tuple(sorted(controls)),
        "evidence_ids": tuple(sorted(evidence)),
        "reason_codes": tuple(sorted(reasons)),
    }
    provisional = RigCapability(**values, capability_sha256="")
    return RigCapability(
        **values,
        capability_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def _derive_records(
    cache: RigGeometryCache,
    anatomy: AnatomyMaskPlan,
    presets: PresetLibraryPlan,
) -> tuple[RigCapability, ...]:
    control_ids = _preset_controls(presets)
    bone_ids = {bone.bone_id for bone in cache.bone_graph.bones}
    mesh_by_part: dict[str, set[str]] = {}
    for mesh in cache.skinning_plan.weighted_meshes:
        mesh_by_part.setdefault(mesh.part_id, set()).add(mesh.mesh_id)
    metric_by_id = {metric.metric_id: metric for metric in anatomy.metrics}
    part_by_base_side: dict[tuple[str, str | None], list[object]] = {}
    for part in cache.parts:
        if part.source_kind == "see_through" and part.part_id in mesh_by_part:
            part_by_base_side.setdefault((part.base_tag, part.side), []).append(part)
    native_roles = {
        part.semantic_role for part in cache.parts if part.source_kind == "native_variant" and part.semantic_role is not None
    }
    native_part_ids_by_role = {
        role: {part.part_id for part in cache.parts if part.source_kind == "native_variant" and part.semantic_role == role}
        for role in native_roles
    }

    torso_metric = metric_by_id.get("mask/torso_core")
    torso_parts = tuple(
        part
        for part in cache.parts
        if part.source_kind == "see_through" and part.base_tag in {"topwear", "bottomwear"} and part.part_id in mesh_by_part
    )
    torso_reasons: set[str] = set()
    if torso_metric is None or torso_metric.status != "available":
        torso_reasons.add("missing_torso_metric")
    if "bone/torso" not in bone_ids:
        torso_reasons.add("missing_torso_bone")
    if not torso_parts:
        torso_reasons.add("missing_torso_mesh")
    torso_evidence = {
        "mask/torso_core",
        "bone/torso",
        *(mesh_id for part in torso_parts for mesh_id in mesh_by_part[part.part_id]),
    } - ({"mask/torso_core"} if "missing_torso_metric" in torso_reasons else set())
    torso_available = not torso_reasons

    head_metric = metric_by_id.get("mask/head_core")
    head_reasons: set[str] = set()
    if head_metric is None or head_metric.status != "available":
        head_reasons.add("missing_head_metric")
    if not {"bone/neck", "bone/head"} <= bone_ids:
        head_reasons.add("missing_head_chain")
    head_evidence = (
        {"mask/head_core", "bone/neck", "bone/head"}
        & ({"mask/head_core"} if head_metric and head_metric.status == "available" else set())
    ) | ({"bone/neck", "bone/head"} & bone_ids)
    head_available = not head_reasons

    arm_parts = tuple(
        part
        for part in cache.parts
        if part.source_kind == "see_through" and part.base_tag == "handwear" and part.part_id in mesh_by_part
    )
    arm_reasons: set[str] = set()
    if torso_metric is None or torso_metric.status != "available":
        arm_reasons.add("missing_torso_metric")
    if not arm_parts:
        arm_reasons.add("missing_arm_mesh")
    arm_evidence = {mesh_id for part in arm_parts for mesh_id in mesh_by_part[part.part_id]}
    if "missing_torso_metric" not in arm_reasons:
        arm_evidence.add("mask/torso_core")

    pelvis = next(
        (resolution for resolution in cache.joint_plan.joints.resolutions if resolution.joint_id == "joint/pelvis"),
        None,
    )
    leg_parts = tuple(
        part
        for part in cache.parts
        if part.source_kind == "see_through" and part.base_tag in {"legwear", "footwear"} and part.part_id in mesh_by_part
    )
    leg_reasons: set[str] = set()
    if pelvis is None or pelvis.status != "resolved":
        leg_reasons.add("missing_pelvis_anchor")
    if not leg_parts:
        leg_reasons.add("missing_leg_mesh")
    leg_evidence = {mesh_id for part in leg_parts for mesh_id in mesh_by_part[part.part_id]}
    if "missing_pelvis_anchor" not in leg_reasons:
        leg_evidence.add("joint/pelvis")

    records: dict[str, RigCapability] = {}
    for preset_id in ("idle", "body_sway"):
        records[preset_id] = _capability(
            preset_id,
            control_ids[preset_id],
            status="available" if torso_available else "unavailable",
            tier="structural" if torso_available else "unavailable",
            evidence=torso_evidence,
            reasons=torso_reasons,
        )
    records["breath"] = _capability(
        "breath",
        control_ids["breath"],
        status="available" if torso_available else "unavailable",
        tier="procedural" if torso_available else "unavailable",
        evidence=torso_evidence,
        reasons=torso_reasons,
    )
    records["arm_sway"] = _capability(
        "arm_sway",
        control_ids["arm_sway"],
        status="available" if not arm_reasons else "unavailable",
        tier="procedural_silhouette" if not arm_reasons else "unavailable",
        evidence=arm_evidence,
        reasons=arm_reasons,
    )
    records["leg_sway"] = _capability(
        "leg_sway",
        control_ids["leg_sway"],
        status="available" if not leg_reasons else "unavailable",
        tier="procedural_silhouette" if not leg_reasons else "unavailable",
        evidence=leg_evidence,
        reasons=leg_reasons,
    )
    for preset_id in ("head_nod", "head_shake"):
        records[preset_id] = _capability(
            preset_id,
            control_ids[preset_id],
            status="available" if head_available else "unavailable",
            tier="structural" if head_available else "unavailable",
            evidence=head_evidence,
            reasons=head_reasons,
        )

    for side in ("xmin", "xmax"):
        preset_id = f"wave.{side}"
        chain = {
            f"bone/upper_arm.{side}",
            f"bone/forearm.{side}",
            f"bone/hand.{side}",
        }
        sided_parts = part_by_base_side.get(("handwear", side), [])
        reasons: set[str] = set()
        if not chain <= bone_ids:
            reasons.add("missing_complete_limb_chain")
        if not sided_parts:
            reasons.add("missing_sided_limb_part")
        evidence = (chain & bone_ids) | {mesh_id for part in sided_parts for mesh_id in mesh_by_part[part.part_id]}
        records[preset_id] = _capability(
            preset_id,
            control_ids[preset_id],
            status="available" if not reasons else "unavailable",
            tier="structural" if not reasons else "unavailable",
            evidence=evidence,
            reasons=reasons,
        )

    native_blink = (
        "eye_closed.coupled" in native_roles
        or {
            "eye_closed.xmin",
            "eye_closed.xmax",
        }
        <= native_roles
    )
    procedural_eye_parts = {
        (base_tag, side)
        for base_tag in ("eyewhite", "irides", "eyelash")
        for side in ("xmin", "xmax")
        if part_by_base_side.get((base_tag, side))
    }
    required_eye_parts = {(base_tag, side) for base_tag in ("eyewhite", "irides", "eyelash") for side in ("xmin", "xmax")}
    procedural_blink = procedural_eye_parts == required_eye_parts
    blink_evidence: set[str] = set()
    if native_blink:
        for role in ("eye_closed.coupled", "eye_closed.xmin", "eye_closed.xmax"):
            blink_evidence.update(native_part_ids_by_role.get(role, ()))
    elif procedural_blink:
        for key in required_eye_parts:
            for part in part_by_base_side[key]:
                blink_evidence.update(mesh_by_part[part.part_id])
    blink_available = native_blink or procedural_blink
    records["blink"] = _capability(
        "blink",
        control_ids["blink"],
        status="available" if blink_available else "unavailable",
        tier=("native" if native_blink else "procedural" if procedural_blink else "unavailable"),
        evidence=blink_evidence,
        reasons=() if blink_available else ("missing_layered_eye_parts",),
    )

    mouth_parts = part_by_base_side.get(("mouth", None), [])
    mouth_mesh_ids = {mesh_id for part in mouth_parts for mesh_id in mesh_by_part[part.part_id]}
    native_mouth_open = bool({"mouth_open", "mouth_closed"} & native_roles)
    procedural_mouth = bool(mouth_parts)
    talk_available = native_mouth_open or procedural_mouth
    talk_evidence = (
        set().union(
            native_part_ids_by_role.get("mouth_open", set()),
            native_part_ids_by_role.get("mouth_closed", set()),
        )
        if native_mouth_open
        else mouth_mesh_ids
    )
    records["talk"] = _capability(
        "talk",
        control_ids["talk"],
        status="available" if talk_available else "unavailable",
        tier=("native" if native_mouth_open else "procedural_silhouette" if procedural_mouth else "unavailable"),
        evidence=talk_evidence,
        reasons=() if talk_available else ("missing_mouth_part",),
    )

    native_mouth_form = {"mouth_smile", "mouth_frown"} <= native_roles
    mouth_form_available = native_mouth_form or procedural_mouth
    mouth_form_evidence: set[str] = set(mouth_mesh_ids)
    if native_mouth_form:
        mouth_form_evidence = set(native_part_ids_by_role["mouth_smile"]) | set(native_part_ids_by_role["mouth_frown"])
    brow_parts = {side: part_by_base_side.get(("eyebrow", side), []) for side in ("xmin", "xmax")}
    brows_available = all(brow_parts.values())
    brow_evidence = {mesh_id for parts in brow_parts.values() for part in parts for mesh_id in mesh_by_part[part.part_id]}
    expression_requirements = {
        "happy": (mouth_form_available and brows_available),
        "sad": (mouth_form_available and brows_available),
        "surprised": (talk_available and brows_available),
        "unimpressed": (mouth_form_available and brows_available),
        "wink_screen_left": blink_available,
        "wink_screen_right": blink_available,
    }
    for preset_id, available in expression_requirements.items():
        reasons: set[str] = set()
        if preset_id in {"happy", "sad", "unimpressed"} and not mouth_form_available:
            reasons.add("missing_mouth_form_capability")
        if preset_id in {"wink_screen_left", "wink_screen_right"} and not blink_available:
            reasons.add("missing_eye_open_capability")
        if preset_id == "surprised" and not talk_available:
            reasons.add("missing_mouth_open_capability")
        if preset_id not in {"wink_screen_left", "wink_screen_right"} and not brows_available:
            reasons.add("missing_brow_parts")
        evidence = set() if preset_id in {"wink_screen_left", "wink_screen_right"} else set(brow_evidence)
        if preset_id in {"happy", "sad", "unimpressed"}:
            evidence.update(mouth_form_evidence)
        if preset_id in {"wink_screen_left", "wink_screen_right"}:
            evidence.update(blink_evidence)
        if preset_id == "surprised":
            evidence.update(talk_evidence)
        native_expression = (
            (preset_id in {"happy", "sad", "unimpressed"} and native_mouth_form)
            or (preset_id == "surprised" and native_mouth_open)
            or (preset_id in {"wink_screen_left", "wink_screen_right"} and native_blink)
        )
        records[preset_id] = _capability(
            preset_id,
            control_ids[preset_id],
            status="available" if available else "unavailable",
            tier=("native" if available and native_expression else "procedural" if available else "unavailable"),
            evidence=evidence,
            reasons=reasons,
        )

    if set(records) != set(control_ids):
        missing = sorted(set(control_ids) - set(records))
        raise _error(f"capability derivation lacks preset handlers: {missing}")
    return tuple(records[preset_id] for preset_id in sorted(records))


def validate_capability_plan(
    plan: CapabilityPlan,
    cache: RigGeometryCache,
    anatomy: AnatomyMaskPlan,
    presets: PresetLibraryPlan,
    *,
    native_variant_set: NativeVariantSet | None,
) -> CapabilityPlan:
    """Re-derive and validate every capability against immutable A/B facts."""

    validate_rig_geometry_cache(cache)
    _validate_anatomy_plan(cache, anatomy)
    _validate_native_set(cache, native_variant_set)
    if not isinstance(plan, CapabilityPlan) or plan.schema_version != CAPABILITY_PLAN_VERSION:
        raise _error("capability plan version is unsupported")
    if plan.rig_geometry_cache_sha256 != cache.cache_sha256:
        raise _error("capability plan references another geometry cache")
    if plan.anatomy_plan_sha256 != anatomy.plan_sha256:
        raise _error("capability plan references another anatomy plan")
    if plan.preset_library_sha256 != presets.plan_sha256:
        raise _error("capability plan references another preset library")
    if plan.native_variant_set_sha256 != cache.native_variant_set_sha256:
        raise _error("capability plan references another NativeVariant set")
    expected = _derive_records(cache, anatomy, presets)
    if plan.capabilities != expected:
        raise _error("capability records differ from frozen geometry facts")
    if tuple(item.preset_id for item in plan.capabilities) != tuple(sorted(item.preset_id for item in plan.capabilities)):
        raise _error("capabilities are not in canonical preset order")
    if any(
        item.capability_sha256 != jcs_sha256(item.semantic_payload())
        or (item.status == "available") != (item.quality_tier != "unavailable")
        or (item.status == "available") == bool(item.reason_codes)
        for item in plan.capabilities
    ):
        raise _error("capability status, reasons, tier, or digest is invalid")
    if plan.plan_sha256 != jcs_sha256(plan.semantic_payload()):
        raise _error("capability plan digest mismatch")
    return plan


def derive_capabilities(
    cache: RigGeometryCache,
    anatomy: AnatomyMaskPlan,
    presets: PresetLibraryPlan,
    *,
    native_variant_set: NativeVariantSet | None,
) -> CapabilityPlan:
    """Derive preset capabilities without reopening masks, QCL, or source images."""

    validate_rig_geometry_cache(cache)
    _validate_anatomy_plan(cache, anatomy)
    _validate_native_set(cache, native_variant_set)
    records = _derive_records(cache, anatomy, presets)
    values = {
        "schema_version": CAPABILITY_PLAN_VERSION,
        "rig_geometry_cache_sha256": cache.cache_sha256,
        "anatomy_plan_sha256": anatomy.plan_sha256,
        "preset_library_sha256": presets.plan_sha256,
        "native_variant_set_sha256": cache.native_variant_set_sha256,
        "capabilities": records,
    }
    provisional = CapabilityPlan(**values, plan_sha256="")
    plan = CapabilityPlan(
        **values,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )
    return validate_capability_plan(
        plan,
        cache,
        anatomy,
        presets,
        native_variant_set=native_variant_set,
    )


__all__ = [
    "CAPABILITY_PLAN_VERSION",
    "CapabilityPlan",
    "CapabilityPlanError",
    "RigCapability",
    "derive_capabilities",
    "validate_capability_plan",
]
