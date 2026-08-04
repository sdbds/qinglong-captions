from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

from .anatomy import AnatomyMaskMetric, AnatomyMaskPlan
from .capabilities import (
    CapabilityPlan,
    RigCapability,
    validate_capability_plan,
)
from .control_registry import (
    ControlRegistryPlan,
    ControlSpec,
    validate_control_registry_plan,
)
from .jcs import jcs_sha256
from .native_variants import NativeVariantSet
from .preset_library import PresetLibraryPlan, validate_preset_library_plan
from .rig_geometry import RigGeometryCache, validate_rig_geometry_cache
from .skinning import WeightedMeshRecord

CONTROL_BINDING_PLAN_VERSION = "control-binding-plan-v3"
CONTROL_BINDING_ID_VERSION = "control-binding-id-v1"
TARGET_TRANSFER_VERSION = "target-transfer-v1"
RIG_TRANSFORM_SEMANTICS_VERSION = "rig-transform-semantics-v3"

HEAD_DEPTH_PARALLAX_TAGS = frozenset({"back hair", "earwear", "eyewear", "front hair", "headwear"})
HEAD_DEPTH_PARALLAX_MAX_CANVAS_PIXELS = 8.0
HEAD_DEPTH_PARALLAX_HEAD_WIDTH_RATIO = 0.03
PROCEDURAL_MOUTH_OPEN_SCALE = 2.2
BREATH_HORIZONTAL_SCALE = 1.03
BREATH_VERTICAL_SCALE = 1.03
PROCEDURAL_ARM_SWAY_DEGREES = 4.0
PROCEDURAL_LEG_SWAY_DEGREES = 2.0

ImplementationKind = Literal["canonical", "native", "procedural"]
TransferKind = Literal[
    "affine_scalar",
    "affine_vector",
    "sampled_property",
    "sampled_deform",
]

_IMPLEMENTATION_RANKS = {"canonical": 0, "native": 10, "procedural": 100}


class ControlBindingPlanError(ValueError):
    """Raised when control-to-model bindings violate the frozen item facts."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _error(message: str) -> ControlBindingPlanError:
    return ControlBindingPlanError("invalid_control_binding_plan", message)


@dataclass(frozen=True, slots=True)
class TransferMetricInput:
    metric_id: str
    anatomy_plan_sha256: str
    bbox: tuple[int, int, int, int]
    width: int
    height: int

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_id": self.metric_id,
            "anatomy_plan_sha256": self.anatomy_plan_sha256,
            "bbox": list(self.bbox),
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class TransferSample:
    input_value: float
    output_values: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "input_value": self.input_value,
            "output_values": list(self.output_values),
        }


@dataclass(frozen=True, slots=True)
class TargetTransfer:
    schema_version: str
    kind: TransferKind
    evaluator_version: str
    control_default: float
    output_at_default: tuple[float, ...]
    gain: tuple[float, ...]
    samples: tuple[TransferSample, ...]
    metric_inputs: tuple[TransferMetricInput, ...]
    topology_sha256: str | None
    input_sha256: str
    output_sha256: str
    transfer_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "evaluator_version": self.evaluator_version,
            "control_default": self.control_default,
            "output_at_default": list(self.output_at_default),
            "gain": list(self.gain),
            "samples": [sample.to_dict() for sample in self.samples],
            "metric_inputs": [metric.to_dict() for metric in self.metric_inputs],
            "topology_sha256": self.topology_sha256,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "transfer_sha256": self.transfer_sha256}


@dataclass(frozen=True, slots=True)
class ControlBinding:
    binding_group_id: str
    implementation_id: str
    implementation_kind: ImplementationKind
    implementation_rank: int
    implementation_bundle_digest: str
    binding_id: str
    control_id: str
    target_id: str
    property: str
    visibility_branch_id: str | None
    required_rig_facts: tuple[str, ...]
    transfer: TargetTransfer
    binding_sha256: str

    def bundle_record(self) -> dict[str, object]:
        return {
            "binding_id": self.binding_id,
            "control_id": self.control_id,
            "target_id": self.target_id,
            "property": self.property,
            "visibility_branch_id": self.visibility_branch_id,
            "required_rig_facts": list(self.required_rig_facts),
            "transfer_sha256": self.transfer.transfer_sha256,
        }

    def semantic_payload(self) -> dict[str, object]:
        return {
            "binding_group_id": self.binding_group_id,
            "implementation_id": self.implementation_id,
            "implementation_kind": self.implementation_kind,
            "implementation_rank": self.implementation_rank,
            "implementation_bundle_digest": self.implementation_bundle_digest,
            "binding_id": self.binding_id,
            "control_id": self.control_id,
            "target_id": self.target_id,
            "property": self.property,
            "visibility_branch_id": self.visibility_branch_id,
            "required_rig_facts": list(self.required_rig_facts),
            "transfer": self.transfer.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "binding_sha256": self.binding_sha256}


@dataclass(frozen=True, slots=True)
class ControlBindingPlan:
    schema_version: str
    transfer_version: str
    transform_semantics_version: str
    rig_geometry_cache_sha256: str
    anatomy_plan_sha256: str
    control_registry_sha256: str
    preset_library_sha256: str
    capability_plan_sha256: str
    native_variant_set_sha256: str
    bindings: tuple[ControlBinding, ...]
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "transfer_version": self.transfer_version,
            "transform_semantics_version": self.transform_semantics_version,
            "rig_geometry_cache_sha256": self.rig_geometry_cache_sha256,
            "anatomy_plan_sha256": self.anatomy_plan_sha256,
            "control_registry_sha256": self.control_registry_sha256,
            "preset_library_sha256": self.preset_library_sha256,
            "capability_plan_sha256": self.capability_plan_sha256,
            "native_variant_set_sha256": self.native_variant_set_sha256,
            "bindings": [binding.to_dict() for binding in self.bindings],
        }


def _round(value: float) -> float:
    rounded = round(float(value), 9)
    return 0.0 if rounded == 0 else rounded


def _metric_input(
    metric: AnatomyMaskMetric,
    anatomy: AnatomyMaskPlan,
) -> TransferMetricInput:
    if metric.status != "available" or metric.bbox is None:
        raise _error(f"required metric is not available: {metric.metric_id}")
    return TransferMetricInput(
        metric_id=metric.metric_id,
        anatomy_plan_sha256=anatomy.plan_sha256,
        bbox=metric.bbox,
        width=metric.width,
        height=metric.height,
    )


def _make_transfer(
    *,
    kind: TransferKind,
    evaluator_version: str,
    control_default: float,
    output_at_default: tuple[float, ...],
    gain: tuple[float, ...] = (),
    samples: tuple[TransferSample, ...] = (),
    metric_inputs: tuple[TransferMetricInput, ...] = (),
    topology_sha256: str | None = None,
) -> TargetTransfer:
    normalized_default = tuple(_round(value) for value in output_at_default)
    normalized_gain = tuple(_round(value) for value in gain)
    normalized_samples = tuple(
        TransferSample(
            input_value=_round(sample.input_value),
            output_values=tuple(_round(value) for value in sample.output_values),
        )
        for sample in sorted(samples, key=lambda item: item.input_value)
    )
    input_sha256 = jcs_sha256(
        {
            "schema_version": TARGET_TRANSFER_VERSION,
            "kind": kind,
            "evaluator_version": evaluator_version,
            "control_default": control_default,
            "metric_inputs": [metric.to_dict() for metric in metric_inputs],
            "topology_sha256": topology_sha256,
        }
    )
    output_sha256 = jcs_sha256(
        {
            "output_at_default": list(normalized_default),
            "gain": list(normalized_gain),
            "samples": [sample.to_dict() for sample in normalized_samples],
        }
    )
    values = {
        "schema_version": TARGET_TRANSFER_VERSION,
        "kind": kind,
        "evaluator_version": evaluator_version,
        "control_default": _round(control_default),
        "output_at_default": normalized_default,
        "gain": normalized_gain,
        "samples": normalized_samples,
        "metric_inputs": metric_inputs,
        "topology_sha256": topology_sha256,
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
    }
    provisional = TargetTransfer(**values, transfer_sha256="")
    return TargetTransfer(
        **values,
        transfer_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def _affine(
    control: ControlSpec,
    gain: float,
    *,
    metric_inputs: tuple[TransferMetricInput, ...] = (),
) -> TargetTransfer:
    return _make_transfer(
        kind="affine_scalar",
        evaluator_version="affine-scalar-v1",
        control_default=control.default,
        output_at_default=(0.0,),
        gain=(gain,),
        metric_inputs=metric_inputs,
    )


def _rest_values(mesh: WeightedMeshRecord) -> tuple[float, ...]:
    return tuple(value for vertex in mesh.vertices for value in vertex.position)


def _deformed_values(
    mesh: WeightedMeshRecord,
    transform,
) -> tuple[float, ...]:
    values: list[float] = []
    for vertex in mesh.vertices:
        x, y = transform(*vertex.position)
        values.extend((_round(x), _round(y)))
    return tuple(values)


def _rotated_values(
    mesh: WeightedMeshRecord,
    *,
    pivot_x: float,
    pivot_y: float,
    degrees: float,
) -> tuple[float, ...]:
    radians = math.radians(degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return _deformed_values(
        mesh,
        lambda x, y: (
            pivot_x + (x - pivot_x) * cosine - (y - pivot_y) * sine,
            pivot_y + (x - pivot_x) * sine + (y - pivot_y) * cosine,
        ),
    )


def _sampled_deform(
    control: ControlSpec,
    mesh: WeightedMeshRecord,
    samples: tuple[tuple[float, tuple[float, ...]], ...],
    *,
    evaluator_version: str,
    metric_inputs: tuple[TransferMetricInput, ...] = (),
) -> TargetTransfer:
    sample_records = tuple(TransferSample(input_value=value, output_values=output) for value, output in samples)
    default_matches = [sample.output_values for sample in sample_records if sample.input_value == control.default]
    if len(default_matches) != 1:
        raise _error(f"sampled deform lacks one default sample for {control.control_id}")
    return _make_transfer(
        kind="sampled_deform",
        evaluator_version=evaluator_version,
        control_default=control.default,
        output_at_default=default_matches[0],
        samples=sample_records,
        metric_inputs=metric_inputs,
        topology_sha256=mesh.weighted_mesh_sha256,
    )


def _sampled_property(
    control: ControlSpec,
    samples: tuple[tuple[float, float], ...],
    *,
    evaluator_version: str,
) -> TargetTransfer:
    sample_records = tuple(TransferSample(input_value=value, output_values=(_round(output),)) for value, output in samples)
    default_matches = [sample.output_values for sample in sample_records if sample.input_value == control.default]
    if len(default_matches) != 1:
        raise _error(f"sampled property lacks one default sample for {control.control_id}")
    return _make_transfer(
        kind="sampled_property",
        evaluator_version=evaluator_version,
        control_default=control.default,
        output_at_default=default_matches[0],
        samples=sample_records,
    )


def _binding_id(
    implementation_id: str,
    control_id: str,
    target_id: str,
    property_name: str,
) -> str:
    digest = jcs_sha256(
        {
            "schema_version": CONTROL_BINDING_ID_VERSION,
            "implementation_id": implementation_id,
            "control_id": control_id,
            "target_id": target_id,
            "property": property_name,
        }
    ).removeprefix("sha256:")
    return f"binding/b_{digest}"


def _draft_binding(
    *,
    group_id: str,
    implementation_id: str,
    implementation_kind: ImplementationKind,
    control_id: str,
    target_id: str,
    property_name: str,
    required_facts: tuple[str, ...],
    transfer: TargetTransfer,
    visibility_branch_id: str | None = None,
) -> ControlBinding:
    if implementation_kind not in _IMPLEMENTATION_RANKS:
        raise _error(f"unknown implementation kind: {implementation_kind}")
    return ControlBinding(
        binding_group_id=group_id,
        implementation_id=implementation_id,
        implementation_kind=implementation_kind,
        implementation_rank=_IMPLEMENTATION_RANKS[implementation_kind],
        implementation_bundle_digest="",
        binding_id=_binding_id(
            implementation_id,
            control_id,
            target_id,
            property_name,
        ),
        control_id=control_id,
        target_id=target_id,
        property=property_name,
        visibility_branch_id=visibility_branch_id,
        required_rig_facts=tuple(sorted(set(required_facts))),
        transfer=transfer,
        binding_sha256="",
    )


def _finalize_bindings(drafts: list[ControlBinding]) -> tuple[ControlBinding, ...]:
    by_implementation: dict[str, list[ControlBinding]] = {}
    for draft in drafts:
        by_implementation.setdefault(draft.implementation_id, []).append(draft)
    finalized: list[ControlBinding] = []
    for implementation_id in sorted(by_implementation):
        members = sorted(
            by_implementation[implementation_id],
            key=lambda item: item.binding_id,
        )
        bundle_digest = jcs_sha256(
            {
                "schema_version": "control-binding-bundle-v1",
                "implementation_id": implementation_id,
                "bindings": [member.bundle_record() for member in members],
            }
        )
        for member in members:
            with_bundle = replace(
                member,
                implementation_bundle_digest=bundle_digest,
            )
            finalized.append(
                replace(
                    with_bundle,
                    binding_sha256=jcs_sha256(with_bundle.semantic_payload()),
                )
            )
    return tuple(sorted(finalized, key=lambda item: item.binding_id))


def _capability_map(plan: CapabilityPlan) -> dict[str, RigCapability]:
    return {item.preset_id: item for item in plan.capabilities}


def _control_map(plan: ControlRegistryPlan) -> dict[str, ControlSpec]:
    return {item.control_id: item for item in plan.controls}


def _mesh_maps(
    cache: RigGeometryCache,
) -> tuple[dict[str, WeightedMeshRecord], dict[str, list[WeightedMeshRecord]]]:
    by_id = {mesh.mesh_id: mesh for mesh in cache.skinning_plan.weighted_meshes}
    by_part: dict[str, list[WeightedMeshRecord]] = {}
    for mesh in cache.skinning_plan.weighted_meshes:
        by_part.setdefault(mesh.part_id, []).append(mesh)
    for meshes in by_part.values():
        meshes.sort(key=lambda item: item.component_id)
    return by_id, by_part


def _triangle_area(values: tuple[float, ...], ia: int, ib: int, ic: int) -> float:
    ax, ay = values[2 * ia], values[2 * ia + 1]
    bx, by = values[2 * ib], values[2 * ib + 1]
    cx, cy = values[2 * ic], values[2 * ic + 1]
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _validate_deform_orientation(
    mesh: WeightedMeshRecord,
    values: tuple[float, ...],
) -> None:
    rest = _rest_values(mesh)
    for offset in range(0, len(mesh.triangles), 3):
        indices = mesh.triangles[offset : offset + 3]
        rest_area = _triangle_area(rest, *indices)
        deformed_area = _triangle_area(values, *indices)
        if rest_area == 0 or deformed_area <= 0:
            raise _error(f"sampled deform flips or collapses triangle in {mesh.mesh_id}")


def _derive_bindings(
    cache: RigGeometryCache,
    anatomy: AnatomyMaskPlan,
    controls: ControlRegistryPlan,
    capabilities: CapabilityPlan,
) -> tuple[ControlBinding, ...]:
    control_by_id = _control_map(controls)
    capability_by_id = _capability_map(capabilities)
    mesh_by_id, meshes_by_part = _mesh_maps(cache)
    part_by_id = {part.part_id: part for part in cache.parts}
    metric_by_id = {metric.metric_id: metric for metric in anatomy.metrics}
    drafts: list[ControlBinding] = []

    def available(preset_id: str) -> bool:
        return capability_by_id[preset_id].status == "available"

    def add_affine(
        group: str,
        implementation: str,
        control_id: str,
        target_id: str,
        property_name: str,
        gain: float,
        *,
        facts: tuple[str, ...],
        metrics: tuple[TransferMetricInput, ...] = (),
    ) -> None:
        drafts.append(
            _draft_binding(
                group_id=group,
                implementation_id=implementation,
                implementation_kind="canonical",
                control_id=control_id,
                target_id=target_id,
                property_name=property_name,
                required_facts=facts,
                transfer=_affine(
                    control_by_id[control_id],
                    gain,
                    metric_inputs=metrics,
                ),
            )
        )

    torso_metric_input: tuple[TransferMetricInput, ...] = ()
    if metric_by_id["mask/torso_core"].status == "available":
        torso_metric_input = (_metric_input(metric_by_id["mask/torso_core"], anatomy),)
    head_metric_input: tuple[TransferMetricInput, ...] = ()
    if metric_by_id["mask/head_core"].status == "available":
        head_metric_input = (_metric_input(metric_by_id["mask/head_core"], anatomy),)

    if available("idle"):
        add_affine(
            "binding-group/idle.torso",
            "binding-impl/idle.torso.canonical-v1",
            "control/idle",
            "bone/torso",
            "rotation",
            1.0,
            facts=("bone/torso", "mask/torso_core"),
            metrics=torso_metric_input,
        )
    if available("body_sway"):
        torso_width = torso_metric_input[0].width
        add_affine(
            "binding-group/body_sway.torso",
            "binding-impl/body_sway.torso.canonical-v1",
            "control/body_sway",
            "bone/torso",
            "translation_x",
            0.02 * torso_width / 10.0,
            facts=("bone/torso", "mask/torso_core"),
            metrics=torso_metric_input,
        )
        add_affine(
            "binding-group/body_sway.torso",
            "binding-impl/body_sway.torso.canonical-v1",
            "control/body_sway",
            "bone/torso",
            "rotation",
            0.2,
            facts=("bone/torso", "mask/torso_core"),
            metrics=torso_metric_input,
        )
    if available("head_nod"):
        head_height = head_metric_input[0].height
        add_affine(
            "binding-group/head_nod.head",
            "binding-impl/head_nod.head.canonical-v1",
            "control/head_nod",
            "bone/head",
            "translation_y",
            0.06 * head_height / 30.0,
            facts=("bone/head", "bone/neck", "mask/head_core"),
            metrics=head_metric_input,
        )
        add_affine(
            "binding-group/head_nod.head",
            "binding-impl/head_nod.head.canonical-v1",
            "control/head_nod",
            "bone/head",
            "rotation",
            0.2,
            facts=("bone/head", "bone/neck", "mask/head_core"),
            metrics=head_metric_input,
        )
    if available("head_shake"):
        head_width = head_metric_input[0].width
        add_affine(
            "binding-group/head_shake.head",
            "binding-impl/head_shake.head.canonical-v1",
            "control/head_shake",
            "bone/head",
            "translation_x",
            0.06 * head_width / 30.0,
            facts=("bone/head", "bone/neck", "mask/head_core"),
            metrics=head_metric_input,
        )
        add_affine(
            "binding-group/head_shake.head",
            "binding-impl/head_shake.head.canonical-v1",
            "control/head_shake",
            "bone/head",
            "rotation",
            0.1,
            facts=("bone/head", "bone/neck", "mask/head_core"),
            metrics=head_metric_input,
        )
        face_parts = tuple(part for part in cache.parts if part.source_kind == "see_through" and part.base_tag == "face")
        parallax_parts = tuple(
            sorted(
                (part for part in cache.parts if part.source_kind == "see_through" and part.base_tag in HEAD_DEPTH_PARALLAX_TAGS),
                key=lambda part: part.part_id,
            )
        )
        if face_parts and parallax_parts:
            face_depths = sorted(part.depth_median for part in face_parts)
            face_depth = face_depths[len(face_depths) // 2]
            depth_deltas = {part.part_id: face_depth - part.depth_median for part in parallax_parts}
            depth_span = max(abs(value) for value in depth_deltas.values())
            if depth_span > 1e-9:
                maximum_offset = min(
                    HEAD_DEPTH_PARALLAX_MAX_CANVAS_PIXELS,
                    head_width * HEAD_DEPTH_PARALLAX_HEAD_WIDTH_RATIO,
                )
                control = control_by_id["control/head_shake"]
                control_span = max(
                    abs(control.minimum - control.default),
                    abs(control.maximum - control.default),
                )
                if control_span <= 0.0:
                    raise _error("head-shake control has no non-default range")
                for part in parallax_parts:
                    endpoint_offset = maximum_offset * depth_deltas[part.part_id] / depth_span
                    if abs(endpoint_offset) < 0.25:
                        continue
                    for mesh in meshes_by_part.get(part.part_id, ()):
                        rest = _rest_values(mesh)

                        def shifted(value: float) -> tuple[float, ...]:
                            amount = endpoint_offset * (value - control.default) / control_span
                            return _deformed_values(
                                mesh,
                                lambda x, y: (x + amount, y),
                            )

                        samples = tuple(
                            (value, rest if value == control.default else shifted(value))
                            for value in sorted({control.minimum, control.default, control.maximum})
                        )
                        for _value, output in samples:
                            _validate_deform_orientation(mesh, output)
                        drafts.append(
                            _draft_binding(
                                group_id="binding-group/head_shake.depth-parallax",
                                implementation_id=("binding-impl/head_shake.depth-parallax-v1"),
                                implementation_kind="canonical",
                                control_id=control.control_id,
                                target_id=mesh.mesh_id,
                                property_name="deform",
                                required_facts=(
                                    "mask/head_core",
                                    mesh.mesh_id,
                                    part.part_id,
                                    *(face.part_id for face in face_parts),
                                ),
                                transfer=_sampled_deform(
                                    control,
                                    mesh,
                                    samples,
                                    evaluator_version=("head-depth-parallax-horizontal-v1"),
                                    metric_inputs=head_metric_input,
                                ),
                            )
                        )

    if available("arm_sway"):
        torso_bbox = torso_metric_input[0].bbox
        torso_x1, torso_y1, torso_x2, torso_y2 = torso_bbox
        torso_width = torso_x2 - torso_x1
        torso_height = torso_y2 - torso_y1
        torso_center_x = (torso_x1 + torso_x2) / 2.0
        shoulder_y = torso_y1 + 0.15 * torso_height
        control = control_by_id["control/arm_sway"]
        arm_parts = tuple(part for part in cache.parts if part.source_kind == "see_through" and part.base_tag == "handwear")
        for part in arm_parts:
            for mesh in meshes_by_part.get(part.part_id, ()):
                rest = _rest_values(mesh)
                center_x = (mesh.component_bbox[0] + mesh.component_bbox[2]) / 2.0
                if center_x < torso_center_x - 0.05 * torso_width:
                    pivot_x = torso_x1 + 0.15 * torso_width
                    direction = 1.0
                elif center_x > torso_center_x + 0.05 * torso_width:
                    pivot_x = torso_x2 - 0.15 * torso_width
                    direction = -1.0
                else:
                    pivot_x = torso_center_x
                    direction = 1.0
                samples = tuple(
                    (
                        value,
                        rest
                        if value == control.default
                        else _rotated_values(
                            mesh,
                            pivot_x=pivot_x,
                            pivot_y=shoulder_y,
                            degrees=value * direction * PROCEDURAL_ARM_SWAY_DEGREES,
                        ),
                    )
                    for value in (control.minimum, control.default, control.maximum)
                )
                for _value, output in samples:
                    _validate_deform_orientation(mesh, output)
                drafts.append(
                    _draft_binding(
                        group_id="binding-group/arm_sway.coarse",
                        implementation_id="binding-impl/arm_sway.coarse-v1",
                        implementation_kind="procedural",
                        control_id=control.control_id,
                        target_id=mesh.mesh_id,
                        property_name="deform",
                        required_facts=(
                            "mask/torso_core",
                            part.part_id,
                            mesh.mesh_id,
                        ),
                        transfer=_sampled_deform(
                            control,
                            mesh,
                            samples,
                            evaluator_version="coarse-arm-component-rotation-v1",
                            metric_inputs=torso_metric_input,
                        ),
                    )
                )

    if available("leg_sway"):
        pelvis = next(resolution for resolution in cache.joint_plan.joints.resolutions if resolution.joint_id == "joint/pelvis")
        if pelvis.status != "resolved" or pelvis.x is None or pelvis.y is None:
            raise _error("leg-sway capability lacks a resolved pelvis anchor")
        control = control_by_id["control/leg_sway"]
        leg_parts = tuple(
            part for part in cache.parts if part.source_kind == "see_through" and part.base_tag in {"legwear", "footwear"}
        )
        for part in leg_parts:
            for mesh in meshes_by_part.get(part.part_id, ()):
                rest = _rest_values(mesh)
                samples = tuple(
                    (
                        value,
                        rest
                        if value == control.default
                        else _rotated_values(
                            mesh,
                            pivot_x=pelvis.x,
                            pivot_y=pelvis.y,
                            degrees=value * PROCEDURAL_LEG_SWAY_DEGREES,
                        ),
                    )
                    for value in (control.minimum, control.default, control.maximum)
                )
                for _value, output in samples:
                    _validate_deform_orientation(mesh, output)
                drafts.append(
                    _draft_binding(
                        group_id="binding-group/leg_sway.coarse",
                        implementation_id="binding-impl/leg_sway.coarse-v1",
                        implementation_kind="procedural",
                        control_id=control.control_id,
                        target_id=mesh.mesh_id,
                        property_name="deform",
                        required_facts=(
                            "joint/pelvis",
                            part.part_id,
                            mesh.mesh_id,
                        ),
                        transfer=_sampled_deform(
                            control,
                            mesh,
                            samples,
                            evaluator_version="coarse-leg-group-rotation-v1",
                        ),
                    )
                )

    if available("breath"):
        spine = next(resolution for resolution in cache.joint_plan.joints.resolutions if resolution.joint_id == "joint/spine")
        if spine.status != "resolved" or spine.x is None or spine.y is None:
            raise _error("breath capability lacks a resolved spine anchor")
        torso_anchor_y = float(torso_metric_input[0].bbox[3])
        torso_parts = tuple(
            part for part in cache.parts if part.source_kind == "see_through" and part.base_tag in {"topwear", "bottomwear"}
        )
        for part in torso_parts:
            for mesh in meshes_by_part.get(part.part_id, ()):
                rest = _rest_values(mesh)
                expanded = _deformed_values(
                    mesh,
                    lambda x, y, ax=spine.x, ay=torso_anchor_y: (
                        ax + (x - ax) * BREATH_HORIZONTAL_SCALE,
                        ay + (y - ay) * BREATH_VERTICAL_SCALE,
                    ),
                )
                _validate_deform_orientation(mesh, expanded)
                drafts.append(
                    _draft_binding(
                        group_id="binding-group/breath.torso",
                        implementation_id="binding-impl/breath.torso.canonical-v1",
                        implementation_kind="canonical",
                        control_id="control/breath",
                        target_id=mesh.mesh_id,
                        property_name="deform",
                        required_facts=(
                            "joint/spine",
                            "mask/torso_core",
                            mesh.mesh_id,
                        ),
                        transfer=_sampled_deform(
                            control_by_id["control/breath"],
                            mesh,
                            ((0.0, rest), (1.0, expanded)),
                            evaluator_version="breath-torso-deform-v2",
                            metric_inputs=torso_metric_input,
                        ),
                    )
                )

    for side, sign in (("xmin", 1.0), ("xmax", -1.0)):
        preset_id = f"wave.{side}"
        if not available(preset_id):
            continue
        implementation = f"binding-impl/wave.{side}.canonical-v1"
        group = f"binding-group/wave.{side}"
        facts = (
            f"bone/upper_arm.{side}",
            f"bone/forearm.{side}",
            f"bone/hand.{side}",
        )
        for control_id, target_id, gain in (
            (f"control/wave_lift.{side}", f"bone/upper_arm.{side}", sign * 35.0),
            (f"control/wave_lift.{side}", f"bone/forearm.{side}", sign * 20.0),
            (f"control/wave_osc.{side}", f"bone/forearm.{side}", sign * 15.0),
            (f"control/wave_osc.{side}", f"bone/hand.{side}", sign * 5.0),
        ):
            add_affine(
                group,
                implementation,
                control_id,
                target_id,
                "rotation",
                gain,
                facts=facts,
            )

    def ordinary_parts(base_tag: str, side: str | None = None):
        return tuple(
            part for part in cache.parts if part.source_kind == "see_through" and part.base_tag == base_tag and part.side == side
        )

    if available("blink"):
        for side in ("xmin", "xmax"):
            eye_meshes_by_tag = {
                base_tag: tuple(mesh for part in ordinary_parts(base_tag, side) for mesh in meshes_by_part.get(part.part_id, ()))
                for base_tag in ("eyewhite", "irides", "eyelash")
            }
            if all(eye_meshes_by_tag.values()):
                all_vertices = [
                    vertex.position for meshes in eye_meshes_by_tag.values() for mesh in meshes for vertex in mesh.vertices
                ]
                closure_y = sum(y for _x, y in all_vertices) / len(all_vertices)
                implementation = "binding-impl/blink.procedural-v1"
                for base_tag, meshes in eye_meshes_by_tag.items():
                    for mesh in meshes:
                        rest = _rest_values(mesh)
                        center_y = sum(vertex.position[1] for vertex in mesh.vertices) / len(mesh.vertices)
                        if base_tag == "eyelash":
                            closed = _deformed_values(
                                mesh,
                                lambda x, y, delta=closure_y - center_y: (x, y + delta),
                            )
                        else:
                            closed = _deformed_values(
                                mesh,
                                lambda x, y, center=center_y: (
                                    x,
                                    closure_y + (y - center) * 0.05,
                                ),
                            )
                        _validate_deform_orientation(mesh, closed)
                        drafts.append(
                            _draft_binding(
                                group_id="binding-group/blink",
                                implementation_id=implementation,
                                implementation_kind="procedural",
                                control_id=f"control/eye_open.{side}",
                                target_id=mesh.mesh_id,
                                property_name="deform",
                                required_facts=(mesh.mesh_id, part_by_id[mesh.part_id].part_id),
                                transfer=_sampled_deform(
                                    control_by_id[f"control/eye_open.{side}"],
                                    mesh,
                                    ((0.0, closed), (1.0, rest)),
                                    evaluator_version=f"layered-blink-{base_tag}-v1",
                                ),
                            )
                        )
                        if base_tag == "irides":
                            drafts.append(
                                _draft_binding(
                                    group_id="binding-group/blink",
                                    implementation_id=implementation,
                                    implementation_kind="procedural",
                                    control_id=f"control/eye_open.{side}",
                                    target_id=mesh.mesh_id,
                                    property_name="opacity",
                                    required_facts=(mesh.mesh_id,),
                                    transfer=_sampled_property(
                                        control_by_id[f"control/eye_open.{side}"],
                                        ((0.0, 0.0), (1.0, 1.0)),
                                        evaluator_version="layered-blink-iris-opacity-v1",
                                    ),
                                )
                            )

    mouth_meshes = tuple(mesh for part in ordinary_parts("mouth") for mesh in meshes_by_part.get(part.part_id, ()))
    mouth_open_samples: dict[str, tuple[tuple[float, tuple[float, ...]], ...]] = {}
    if available("talk") and mouth_meshes:
        for mesh in mouth_meshes:
            rest = _rest_values(mesh)
            center_y = sum(vertex.position[1] for vertex in mesh.vertices) / len(mesh.vertices)
            opened = _deformed_values(
                mesh,
                lambda x, y: (
                    x,
                    center_y + (y - center_y) * PROCEDURAL_MOUTH_OPEN_SCALE,
                ),
            )
            _validate_deform_orientation(mesh, opened)
            samples = ((0.0, rest), (1.0, opened))
            mouth_open_samples[mesh.mesh_id] = samples
            drafts.append(
                _draft_binding(
                    group_id="binding-group/mouth_open",
                    implementation_id="binding-impl/mouth_open.procedural-v1",
                    implementation_kind="procedural",
                    control_id="control/mouth_open",
                    target_id=mesh.mesh_id,
                    property_name="deform",
                    required_facts=(mesh.mesh_id,),
                    transfer=_sampled_deform(
                        control_by_id["control/mouth_open"],
                        mesh,
                        samples,
                        evaluator_version="mouth-open-silhouette-v2",
                    ),
                )
            )
    if any(available(preset_id) for preset_id in ("happy", "sad", "unimpressed")) and mouth_meshes:
        for mesh in mouth_meshes:
            rest = _rest_values(mesh)
            xs = [vertex.position[0] for vertex in mesh.vertices]
            ys = [vertex.position[1] for vertex in mesh.vertices]
            center_x = (min(xs) + max(xs)) / 2.0
            half_width = max((max(xs) - min(xs)) / 2.0, 1e-6)
            height = max(max(ys) - min(ys), 1.0)

            def mouth_form(value: float):
                return _deformed_values(
                    mesh,
                    lambda x, y: (
                        x,
                        y - value * 0.15 * height * abs(x - center_x) / half_width,
                    ),
                )

            frown = mouth_form(-1.0)
            smile = mouth_form(1.0)
            _validate_deform_orientation(mesh, frown)
            _validate_deform_orientation(mesh, smile)
            drafts.append(
                _draft_binding(
                    group_id="binding-group/mouth_form",
                    implementation_id="binding-impl/mouth_form.procedural-v1",
                    implementation_kind="procedural",
                    control_id="control/mouth_form",
                    target_id=mesh.mesh_id,
                    property_name="deform",
                    required_facts=(mesh.mesh_id,),
                    transfer=_sampled_deform(
                        control_by_id["control/mouth_form"],
                        mesh,
                        ((-1.0, frown), (0.0, rest), (1.0, smile)),
                        evaluator_version="mouth-form-deform-v1",
                    ),
                )
            )

    if any(available(preset_id) for preset_id in ("happy", "sad", "surprised", "unimpressed")):
        head_height = head_metric_input[0].height if head_metric_input else 0
        for side in ("xmin", "xmax"):
            for part in ordinary_parts("eyebrow", side):
                for mesh in meshes_by_part.get(part.part_id, ()):
                    rest = _rest_values(mesh)
                    control = control_by_id[f"control/brow_y.{side}"]
                    lowered = _deformed_values(
                        mesh,
                        lambda x, y: (x, y + 0.03 * head_height),
                    )
                    raised = _deformed_values(
                        mesh,
                        lambda x, y: (x, y - 0.03 * head_height),
                    )
                    drafts.append(
                        _draft_binding(
                            group_id=f"binding-group/brow_y.{side}",
                            implementation_id=f"binding-impl/brow_y.{side}.procedural-v1",
                            implementation_kind="procedural",
                            control_id=control.control_id,
                            target_id=mesh.mesh_id,
                            property_name="deform",
                            required_facts=(mesh.mesh_id, "mask/head_core"),
                            transfer=_sampled_deform(
                                control,
                                mesh,
                                ((-1.0, lowered), (0.0, rest), (1.0, raised)),
                                evaluator_version="brow-y-deform-v1",
                                metric_inputs=head_metric_input,
                            ),
                        )
                    )

    native_parts = tuple(part for part in cache.parts if part.source_kind == "native_variant")
    native_by_role = {
        role: tuple(part for part in native_parts if part.semantic_role == role)
        for role in {part.semantic_role for part in native_parts}
        if role is not None
    }

    if available("blink") and native_by_role:
        blink_targets: dict[str, list[WeightedMeshRecord]] = {"xmin": [], "xmax": []}
        for side in ("xmin", "xmax"):
            for part in native_by_role.get(f"eye_closed.{side}", ()):
                blink_targets[side].extend(meshes_by_part.get(part.part_id, ()))
        for part in native_by_role.get("eye_closed.coupled", ()):
            meshes = sorted(
                meshes_by_part.get(part.part_id, ()),
                key=lambda mesh: (
                    (mesh.component_bbox[0] + mesh.component_bbox[2]) / 2.0,
                    mesh.component_id,
                ),
            )
            if len(meshes) != 2:
                raise _error("coupled eye native variant must contain exactly two meshes")
            blink_targets["xmin"].append(meshes[0])
            blink_targets["xmax"].append(meshes[1])
        if all(blink_targets.values()):
            for side, meshes in blink_targets.items():
                for base_tag in ("eyewhite", "irides", "eyelash"):
                    for base_part in ordinary_parts(base_tag, side):
                        for base_mesh in meshes_by_part.get(base_part.part_id, ()):
                            drafts.append(
                                _draft_binding(
                                    group_id="binding-group/blink",
                                    implementation_id="binding-impl/blink.native-v1",
                                    implementation_kind="native",
                                    control_id=f"control/eye_open.{side}",
                                    target_id=base_mesh.mesh_id,
                                    property_name="opacity",
                                    required_facts=(base_mesh.mesh_id, base_part.part_id),
                                    transfer=_sampled_property(
                                        control_by_id[f"control/eye_open.{side}"],
                                        ((0.0, 0.0), (1.0, 1.0)),
                                        evaluator_version="native-eye-open-crossfade-opacity-v1",
                                    ),
                                )
                            )
                for mesh in meshes:
                    variant_id = part_by_id[mesh.part_id].variant_id
                    drafts.append(
                        _draft_binding(
                            group_id="binding-group/blink",
                            implementation_id="binding-impl/blink.native-v1",
                            implementation_kind="native",
                            control_id=f"control/eye_open.{side}",
                            target_id=mesh.mesh_id,
                            property_name="opacity",
                            required_facts=(mesh.mesh_id, mesh.part_id),
                            visibility_branch_id=f"visibility-branch/{variant_id}",
                            transfer=_sampled_property(
                                control_by_id[f"control/eye_open.{side}"],
                                ((0.0, 1.0), (1.0, 0.0)),
                                evaluator_version="native-eye-closed-opacity-v1",
                            ),
                        )
                    )

    if available("talk") and native_by_role.get("mouth_closed"):
        implementation_id = "binding-impl/mouth_open.crossfade-v1"
        for mesh in mouth_meshes:
            drafts.append(
                _draft_binding(
                    group_id="binding-group/mouth_open",
                    implementation_id=implementation_id,
                    implementation_kind="native",
                    control_id="control/mouth_open",
                    target_id=mesh.mesh_id,
                    property_name="deform",
                    required_facts=(mesh.mesh_id, mesh.part_id),
                    transfer=_sampled_deform(
                        control_by_id["control/mouth_open"],
                        mesh,
                        mouth_open_samples[mesh.mesh_id],
                        evaluator_version="native-mouth-open-base-deform-v1",
                    ),
                )
            )
            drafts.append(
                _draft_binding(
                    group_id="binding-group/mouth_open",
                    implementation_id=implementation_id,
                    implementation_kind="native",
                    control_id="control/mouth_open",
                    target_id=mesh.mesh_id,
                    property_name="opacity",
                    required_facts=(mesh.mesh_id, mesh.part_id),
                    transfer=_sampled_property(
                        control_by_id["control/mouth_open"],
                        ((0.0, 0.0), (1.0, 1.0)),
                        evaluator_version="native-mouth-open-base-crossfade-opacity-v1",
                    ),
                )
            )
        for part in native_by_role["mouth_closed"]:
            for mesh in meshes_by_part.get(part.part_id, ()):
                drafts.append(
                    _draft_binding(
                        group_id="binding-group/mouth_open",
                        implementation_id=implementation_id,
                        implementation_kind="native",
                        control_id="control/mouth_open",
                        target_id=mesh.mesh_id,
                        property_name="opacity",
                        required_facts=(mesh.mesh_id, part.part_id),
                        visibility_branch_id=f"visibility-branch/{part.variant_id}",
                        transfer=_sampled_property(
                            control_by_id["control/mouth_open"],
                            ((0.0, 1.0), (1.0, 0.0)),
                            evaluator_version="native-mouth-closed-crossfade-opacity-v1",
                        ),
                    )
                )

    if available("talk"):
        for part in native_by_role.get("mouth_open", ()):
            for mesh in meshes_by_part.get(part.part_id, ()):
                drafts.append(
                    _draft_binding(
                        group_id="binding-group/mouth_open",
                        implementation_id="binding-impl/mouth_open.native-v1",
                        implementation_kind="native",
                        control_id="control/mouth_open",
                        target_id=mesh.mesh_id,
                        property_name="opacity",
                        required_facts=(mesh.mesh_id, part.part_id),
                        visibility_branch_id=f"visibility-branch/{part.variant_id}",
                        transfer=_sampled_property(
                            control_by_id["control/mouth_open"],
                            ((0.0, 0.0), (1.0, 1.0)),
                            evaluator_version="native-mouth-open-opacity-v1",
                        ),
                    )
                )
    if any(available(preset_id) for preset_id in ("happy", "sad", "unimpressed")):
        role_samples = {
            "mouth_smile": ((-1.0, 0.0), (0.0, 0.0), (1.0, 1.0)),
            "mouth_frown": ((-1.0, 1.0), (0.0, 0.0), (1.0, 0.0)),
        }
        if all(native_by_role.get(role) for role in role_samples):
            for role, samples in role_samples.items():
                for part in native_by_role[role]:
                    for mesh in meshes_by_part.get(part.part_id, ()):
                        drafts.append(
                            _draft_binding(
                                group_id="binding-group/mouth_form",
                                implementation_id="binding-impl/mouth_form.native-v1",
                                implementation_kind="native",
                                control_id="control/mouth_form",
                                target_id=mesh.mesh_id,
                                property_name="opacity",
                                required_facts=(mesh.mesh_id, part.part_id),
                                visibility_branch_id=f"visibility-branch/{part.variant_id}",
                                transfer=_sampled_property(
                                    control_by_id["control/mouth_form"],
                                    samples,
                                    evaluator_version=f"native-{role}-opacity-v1",
                                ),
                            )
                        )

    return _finalize_bindings(drafts)


def _validate_transfer(transfer: TargetTransfer) -> None:
    if transfer.schema_version != TARGET_TRANSFER_VERSION:
        raise _error("target transfer version is unsupported")
    numeric = (
        transfer.control_default,
        *transfer.output_at_default,
        *transfer.gain,
        *(value for sample in transfer.samples for value in (sample.input_value, *sample.output_values)),
    )
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in numeric):
        raise _error("target transfer contains a non-finite value")
    if transfer.kind.startswith("affine"):
        if transfer.samples or not transfer.gain:
            raise _error("affine transfer must use gain and no samples")
    else:
        if transfer.gain or not transfer.samples:
            raise _error("sampled transfer must use samples and no gain")
        inputs = tuple(sample.input_value for sample in transfer.samples)
        if inputs != tuple(sorted(set(inputs))):
            raise _error("sample inputs must be unique and sorted")
        defaults = [sample.output_values for sample in transfer.samples if sample.input_value == transfer.control_default]
        if defaults != [transfer.output_at_default]:
            raise _error("sampled transfer does not restore the default output")
    if transfer.transfer_sha256 != jcs_sha256(transfer.semantic_payload()):
        raise _error("target transfer digest mismatch")


def validate_control_binding_plan(
    plan: ControlBindingPlan,
    cache: RigGeometryCache,
    anatomy: AnatomyMaskPlan,
    controls: ControlRegistryPlan,
    presets: PresetLibraryPlan,
    capabilities: CapabilityPlan,
    *,
    native_variant_set: NativeVariantSet | None,
) -> ControlBindingPlan:
    """Rebuild and validate every item-eligible atomic binding bundle."""

    validate_rig_geometry_cache(cache)
    validate_control_registry_plan(controls)
    validate_preset_library_plan(presets, controls)
    validate_capability_plan(
        capabilities,
        cache,
        anatomy,
        presets,
        native_variant_set=native_variant_set,
    )
    if not isinstance(plan, ControlBindingPlan) or (
        plan.schema_version != CONTROL_BINDING_PLAN_VERSION
        or plan.transfer_version != TARGET_TRANSFER_VERSION
        or plan.transform_semantics_version != RIG_TRANSFORM_SEMANTICS_VERSION
    ):
        raise _error("control binding plan version is unsupported")
    expected_identities = (
        cache.cache_sha256,
        anatomy.plan_sha256,
        controls.registry_sha256,
        presets.plan_sha256,
        capabilities.plan_sha256,
        cache.native_variant_set_sha256,
    )
    observed_identities = (
        plan.rig_geometry_cache_sha256,
        plan.anatomy_plan_sha256,
        plan.control_registry_sha256,
        plan.preset_library_sha256,
        plan.capability_plan_sha256,
        plan.native_variant_set_sha256,
    )
    if observed_identities != expected_identities:
        raise _error("binding plan references different registry or geometry inputs")
    if tuple(sorted(plan.bindings, key=lambda item: item.binding_id)) != plan.bindings:
        raise _error("bindings are not in canonical ID order")
    binding_ids = tuple(binding.binding_id for binding in plan.bindings)
    if len(binding_ids) != len(set(binding_ids)):
        raise _error("binding IDs are not unique")
    bone_ids = {bone.bone_id for bone in cache.bone_graph.bones}
    mesh_by_id, _meshes_by_part = _mesh_maps(cache)
    control_ids = {control.control_id for control in controls.controls}
    native_mesh_ids = {
        mesh.mesh_id
        for mesh in cache.skinning_plan.weighted_meshes
        if next(part for part in cache.parts if part.part_id == mesh.part_id).source_kind == "native_variant"
    }
    ranks_by_group: dict[str, dict[str, int]] = {}
    by_implementation: dict[str, list[ControlBinding]] = {}
    for binding in plan.bindings:
        _validate_transfer(binding.transfer)
        if binding.control_id not in control_ids:
            raise _error("binding references an unknown control")
        if binding.property in {"rotation", "translation_x", "translation_y"}:
            if binding.target_id not in bone_ids:
                raise _error("rigid binding references an unknown bone")
        elif binding.property in {"deform", "opacity"}:
            if binding.target_id not in mesh_by_id:
                raise _error("drawable binding references an unknown mesh")
        else:
            raise _error(f"unsupported canonical property: {binding.property}")
        if (binding.visibility_branch_id is not None) != (binding.target_id in native_mesh_ids):
            raise _error("visibility branch is allowed only for native drawable targets")
        if binding.implementation_rank != _IMPLEMENTATION_RANKS.get(binding.implementation_kind):
            raise _error("implementation kind/rank differs from v1")
        ranks_by_group.setdefault(binding.binding_group_id, {})[binding.implementation_id] = binding.implementation_rank
        by_implementation.setdefault(binding.implementation_id, []).append(binding)
        if binding.binding_sha256 != jcs_sha256(binding.semantic_payload()):
            raise _error("binding digest mismatch")
    if any(len(set(ranks.values())) != len(ranks) for ranks in ranks_by_group.values()):
        raise _error("a binding group contains duplicate implementation ranks")
    for implementation_id, members in by_implementation.items():
        expected_bundle = jcs_sha256(
            {
                "schema_version": "control-binding-bundle-v1",
                "implementation_id": implementation_id,
                "bindings": [member.bundle_record() for member in sorted(members, key=lambda item: item.binding_id)],
            }
        )
        if any(member.implementation_bundle_digest != expected_bundle for member in members):
            raise _error("implementation bundle digest mismatch")
    expected = _derive_bindings(cache, anatomy, controls, capabilities)
    if plan.bindings != expected:
        raise _error("bindings differ from frozen geometry and capability facts")
    if plan.plan_sha256 != jcs_sha256(plan.semantic_payload()):
        raise _error("control binding plan digest mismatch")
    return plan


def build_control_binding_plan(
    cache: RigGeometryCache,
    anatomy: AnatomyMaskPlan,
    controls: ControlRegistryPlan,
    presets: PresetLibraryPlan,
    capabilities: CapabilityPlan,
    *,
    native_variant_set: NativeVariantSet | None,
) -> ControlBindingPlan:
    """Materialize complete item-eligible control implementations and transfers."""

    validate_capability_plan(
        capabilities,
        cache,
        anatomy,
        presets,
        native_variant_set=native_variant_set,
    )
    bindings = _derive_bindings(cache, anatomy, controls, capabilities)
    values = {
        "schema_version": CONTROL_BINDING_PLAN_VERSION,
        "transfer_version": TARGET_TRANSFER_VERSION,
        "transform_semantics_version": RIG_TRANSFORM_SEMANTICS_VERSION,
        "rig_geometry_cache_sha256": cache.cache_sha256,
        "anatomy_plan_sha256": anatomy.plan_sha256,
        "control_registry_sha256": controls.registry_sha256,
        "preset_library_sha256": presets.plan_sha256,
        "capability_plan_sha256": capabilities.plan_sha256,
        "native_variant_set_sha256": cache.native_variant_set_sha256,
        "bindings": bindings,
    }
    provisional = ControlBindingPlan(**values, plan_sha256="")
    plan = ControlBindingPlan(
        **values,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )
    return validate_control_binding_plan(
        plan,
        cache,
        anatomy,
        controls,
        presets,
        capabilities,
        native_variant_set=native_variant_set,
    )


__all__ = [
    "CONTROL_BINDING_ID_VERSION",
    "CONTROL_BINDING_PLAN_VERSION",
    "RIG_TRANSFORM_SEMANTICS_VERSION",
    "TARGET_TRANSFER_VERSION",
    "ControlBinding",
    "ControlBindingPlan",
    "ControlBindingPlanError",
    "TargetTransfer",
    "TransferMetricInput",
    "TransferSample",
    "build_control_binding_plan",
    "validate_control_binding_plan",
]
