from __future__ import annotations

import re
from dataclasses import dataclass

from .control_bindings import ControlBinding, ControlBindingPlan
from .control_registry import (
    ControlRegistryPlan,
    validate_control_registry_plan,
)
from .jcs import jcs_sha256
from .preset_library import PresetLibraryPlan, validate_preset_library_plan
from .rig_geometry import RigGeometryCache, validate_rig_geometry_cache

PRIMITIVE_CANDIDATE_SET_VERSION = "primitive-candidate-set-v1"
PRIMITIVE_CANDIDATE_ENUMERATOR_VERSION = "primitive-candidate-enumerator-v2"
TYPED_PRIMITIVE_KEY_VERSION = "typed-primitive-key-v1"

_INTERNAL_ID_RE = re.compile(
    r"^[a-z][a-z0-9_-]*/[a-z0-9][a-z0-9._-]{0,126}$"
)
_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,31}$")
_FORMATS = ("live2d_moc3_v4_00", "spine_4_2")


class PrimitiveCandidateSetError(ValueError):
    """Raised when the immutable exporter primitive universe is inconsistent."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _error(message: str) -> PrimitiveCandidateSetError:
    return PrimitiveCandidateSetError("invalid_primitive_candidate_set", message)


@dataclass(frozen=True, slots=True)
class TypedPrimitiveKey:
    schema_version: str
    format_id: str
    kind: str
    source_internal_ids: tuple[str, ...]
    base_source_internal_id: str
    derivation_tokens: tuple[str, ...]
    parameter_id: str | None
    control_id: str | None
    preset_id: str | None
    component_id: str | None
    skin_id: str | None
    slot_id: str | None
    directory: str | None
    key_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "format_id": self.format_id,
            "kind": self.kind,
            "source_internal_ids": list(self.source_internal_ids),
            "base_source_internal_id": self.base_source_internal_id,
            "derivation_tokens": list(self.derivation_tokens),
            "parameter_id": self.parameter_id,
            "control_id": self.control_id,
            "preset_id": self.preset_id,
            "component_id": self.component_id,
            "skin_id": self.skin_id,
            "slot_id": self.slot_id,
            "directory": self.directory,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "key_sha256": self.key_sha256}


@dataclass(frozen=True, slots=True)
class PrimitiveBindingTemplate:
    binding_id: str
    binding_group_id: str
    implementation_id: str
    implementation_bundle_digest: str
    control_id: str
    target_id: str
    property: str
    visibility_branch_id: str | None
    transfer_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "binding_id": self.binding_id,
            "binding_group_id": self.binding_group_id,
            "implementation_id": self.implementation_id,
            "implementation_bundle_digest": self.implementation_bundle_digest,
            "control_id": self.control_id,
            "target_id": self.target_id,
            "property": self.property,
            "visibility_branch_id": self.visibility_branch_id,
            "transfer_sha256": self.transfer_sha256,
        }


@dataclass(frozen=True, slots=True)
class PrimitiveCandidate:
    candidate_id: str
    format_id: str
    candidate_kind: str
    typed_primitive_key: TypedPrimitiveKey
    primitive_target_id: str
    binding_template: PrimitiveBindingTemplate | None
    required_rig_fact_ids: tuple[str, ...]
    driver_registry_entry_id: str
    candidate_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "format_id": self.format_id,
            "candidate_kind": self.candidate_kind,
            "typed_primitive_key": self.typed_primitive_key.to_dict(),
            "primitive_target_id": self.primitive_target_id,
            "binding_template": (
                None if self.binding_template is None else self.binding_template.to_dict()
            ),
            "required_rig_fact_ids": list(self.required_rig_fact_ids),
            "driver_registry_entry_id": self.driver_registry_entry_id,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "candidate_sha256": self.candidate_sha256}


@dataclass(frozen=True, slots=True)
class PrimitiveCandidateSet:
    schema_version: str
    enumerator_version: str
    rig_geometry_cache_sha256: str
    control_registry_sha256: str
    preset_library_sha256: str
    control_binding_plan_sha256: str
    texture_page_ids: tuple[str, ...]
    deterministic_absences: tuple[str, ...]
    candidates: tuple[PrimitiveCandidate, ...]
    candidate_universe_sha256: str
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "enumerator_version": self.enumerator_version,
            "rig_geometry_cache_sha256": self.rig_geometry_cache_sha256,
            "control_registry_sha256": self.control_registry_sha256,
            "preset_library_sha256": self.preset_library_sha256,
            "control_binding_plan_sha256": self.control_binding_plan_sha256,
            "texture_page_ids": list(self.texture_page_ids),
            "deterministic_absences": list(self.deterministic_absences),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "candidate_universe_sha256": self.candidate_universe_sha256,
        }


def _id_token(internal_id: str) -> str:
    token = re.sub(r"[.-]+", "_", internal_id.rsplit("/", 1)[-1]).strip("_")
    if not _TOKEN_RE.fullmatch(token):
        raise _error(f"internal ID cannot form a derivation token: {internal_id}")
    return token


def _component_token(component_id: str) -> str:
    return "c_" + jcs_sha256(component_id).removeprefix("sha256:")[:16]


def _validate_internal_id(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not _INTERNAL_ID_RE.fullmatch(value):
        raise _error(f"{field} is not an InternalId v1 value: {value!r}")


def _key(
    format_id: str,
    kind: str,
    source_internal_ids: tuple[str, ...],
    base_source_internal_id: str,
    *,
    derivation_tokens: tuple[str, ...] = (),
    parameter_id: str | None = None,
    control_id: str | None = None,
    preset_id: str | None = None,
    component_id: str | None = None,
    skin_id: str | None = None,
    slot_id: str | None = None,
    directory: str | None = None,
) -> TypedPrimitiveKey:
    if format_id not in _FORMATS:
        raise _error(f"unknown exporter family: {format_id}")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", kind):
        raise _error(f"primitive kind is invalid: {kind}")
    if not source_internal_ids or len(set(source_internal_ids)) != len(source_internal_ids):
        raise _error("typed primitive source IDs must be non-empty and unique")
    for value in source_internal_ids:
        _validate_internal_id(value, field="source_internal_id")
    if base_source_internal_id not in source_internal_ids:
        raise _error("base source ID is not one of the typed primitive sources")
    for token in derivation_tokens:
        if not _TOKEN_RE.fullmatch(token):
            raise _error(f"invalid derivation token: {token}")
    for field_name, value in (
        ("parameter_id", parameter_id),
        ("control_id", control_id),
        ("preset_id", preset_id),
        ("component_id", component_id),
        ("skin_id", skin_id),
        ("slot_id", slot_id),
    ):
        if value is not None:
            _validate_internal_id(value, field=field_name)
    values = {
        "schema_version": TYPED_PRIMITIVE_KEY_VERSION,
        "format_id": format_id,
        "kind": kind,
        "source_internal_ids": source_internal_ids,
        "base_source_internal_id": base_source_internal_id,
        "derivation_tokens": derivation_tokens,
        "parameter_id": parameter_id,
        "control_id": control_id,
        "preset_id": preset_id,
        "component_id": component_id,
        "skin_id": skin_id,
        "slot_id": slot_id,
        "directory": directory,
    }
    provisional = TypedPrimitiveKey(**values, key_sha256="")
    return TypedPrimitiveKey(
        **values,
        key_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def _candidate(
    *,
    candidate_kind: str,
    key: TypedPrimitiveKey,
    binding: ControlBinding | None = None,
    required_facts: tuple[str, ...] = (),
    driver_registry_entry_id: str,
) -> PrimitiveCandidate:
    template = None
    if binding is not None:
        template = PrimitiveBindingTemplate(
            binding_id=binding.binding_id,
            binding_group_id=binding.binding_group_id,
            implementation_id=binding.implementation_id,
            implementation_bundle_digest=binding.implementation_bundle_digest,
            control_id=binding.control_id,
            target_id=binding.target_id,
            property=binding.property,
            visibility_branch_id=binding.visibility_branch_id,
            transfer_sha256=binding.transfer.transfer_sha256,
        )
    identity = {
        "schema_version": PRIMITIVE_CANDIDATE_ENUMERATOR_VERSION,
        "format_id": key.format_id,
        "candidate_kind": candidate_kind,
        "typed_primitive_key_sha256": key.key_sha256,
        "binding_id": None if template is None else template.binding_id,
    }
    candidate_id = "candidate/c_" + jcs_sha256(identity).removeprefix("sha256:")
    values = {
        "candidate_id": candidate_id,
        "format_id": key.format_id,
        "candidate_kind": candidate_kind,
        "typed_primitive_key": key,
        "primitive_target_id": "primitive/p_"
        + key.key_sha256.removeprefix("sha256:"),
        "binding_template": template,
        "required_rig_fact_ids": tuple(sorted(set(required_facts))),
        "driver_registry_entry_id": driver_registry_entry_id,
    }
    provisional = PrimitiveCandidate(**values, candidate_sha256="")
    return PrimitiveCandidate(
        **values,
        candidate_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def _validate_inputs(
    cache: RigGeometryCache,
    controls: ControlRegistryPlan,
    presets: PresetLibraryPlan,
    bindings: ControlBindingPlan,
    texture_page_ids: tuple[str, ...],
) -> tuple[str, ...]:
    validate_rig_geometry_cache(cache)
    validate_control_registry_plan(controls)
    validate_preset_library_plan(presets, controls)
    if not isinstance(bindings, ControlBindingPlan):
        raise _error("control binding plan has the wrong type")
    if (
        bindings.plan_sha256 != jcs_sha256(bindings.semantic_payload())
        or bindings.rig_geometry_cache_sha256 != cache.cache_sha256
        or bindings.control_registry_sha256 != controls.registry_sha256
        or bindings.preset_library_sha256 != presets.plan_sha256
    ):
        raise _error("control binding plan provenance is invalid")
    if any(
        binding.binding_sha256 != jcs_sha256(binding.semantic_payload())
        for binding in bindings.bindings
    ):
        raise _error("control binding digest is invalid")
    normalized_pages = tuple(sorted(texture_page_ids))
    if normalized_pages != texture_page_ids or len(set(normalized_pages)) != len(
        normalized_pages
    ):
        raise _error("texture page IDs must be unique and canonical")
    for page_id in normalized_pages:
        _validate_internal_id(page_id, field="texture_page_id")
    return normalized_pages


def _enumerate(
    cache: RigGeometryCache,
    controls: ControlRegistryPlan,
    presets: PresetLibraryPlan,
    bindings: ControlBindingPlan,
    texture_page_ids: tuple[str, ...],
) -> tuple[tuple[PrimitiveCandidate, ...], tuple[str, ...]]:
    candidates: list[PrimitiveCandidate] = []
    dynamic_keys: dict[str, TypedPrimitiveKey] = {}
    bone_ids = tuple(bone.bone_id for bone in cache.bone_graph.bones)
    mesh_by_id = {
        mesh.mesh_id: mesh for mesh in cache.skinning_plan.weighted_meshes
    }
    part_by_id = {part.part_id: part for part in cache.parts}
    control_by_id = {control.control_id: control for control in controls.controls}

    def add_model(candidate_kind: str, key: TypedPrimitiveKey, *facts: str) -> None:
        candidates.append(
            _candidate(
                candidate_kind=candidate_kind,
                key=key,
                required_facts=tuple(facts),
                driver_registry_entry_id="driver/static-model-v1",
            )
        )

    skin_id = "skin/default"
    add_model(
        "spine_skin",
        _key("spine_4_2", "spine_skin", (skin_id,), skin_id),
        skin_id,
    )
    for bone_id in bone_ids:
        add_model(
            "spine_bone",
            _key("spine_4_2", "spine_bone", (bone_id,), bone_id),
            bone_id,
        )

    meshes_by_part: dict[str, list[object]] = {}
    for mesh in mesh_by_id.values():
        meshes_by_part.setdefault(mesh.part_id, []).append(mesh)
    for part_id in sorted(part_by_id):
        part_meshes = sorted(
            meshes_by_part.get(part_id, ()), key=lambda item: item.mesh_id
        )
        add_model(
            "spine_atlas_region",
            _key("spine_4_2", "spine_atlas_region", (part_id,), part_id),
            part_id,
        )
        add_model(
            "live2d_part",
            _key("live2d_moc3_v4_00", "live2d_part", (part_id,), part_id),
            part_id,
        )
        for mesh in part_meshes:
            component_tokens = (
                ()
                if len(part_meshes) == 1
                else (_component_token(mesh.component_id),)
            )
            source_ids = (part_id, mesh.component_id)
            common = {
                "source_internal_ids": source_ids,
                "base_source_internal_id": part_id,
                "derivation_tokens": component_tokens,
                "component_id": mesh.component_id,
            }
            slot_key = _key(
                "spine_4_2",
                "spine_slot",
                **common,
            )
            add_model("spine_slot", slot_key, part_id, mesh.component_id, mesh.mesh_id)
            add_model(
                "spine_attachment_key",
                _key(
                    "spine_4_2",
                    "spine_attachment_key",
                    **common,
                    skin_id=skin_id,
                    slot_id=mesh.component_id,
                ),
                part_id,
                mesh.component_id,
                mesh.mesh_id,
            )
            attachment_key = _key(
                "spine_4_2",
                "spine_attachment_object",
                **common,
            )
            add_model(
                "spine_attachment_object",
                attachment_key,
                part_id,
                mesh.component_id,
                mesh.mesh_id,
            )
            artmesh_key = _key(
                "live2d_moc3_v4_00",
                "live2d_artmesh",
                **common,
            )
            add_model(
                "live2d_artmesh",
                artmesh_key,
                part_id,
                mesh.component_id,
                mesh.mesh_id,
            )

    for control in controls.controls:
        if control.live2d is None:
            continue
        parameter_id = control.live2d.parameter_id
        add_model(
            "live2d_parameter",
            _key(
                "live2d_moc3_v4_00",
                "live2d_parameter",
                (parameter_id, control.control_id),
                parameter_id,
                parameter_id=parameter_id,
                control_id=control.control_id,
            ),
            parameter_id,
            control.control_id,
        )

    for clip in presets.clips:
        add_model(
            "spine_animation",
            _key(
                "spine_4_2",
                "spine_animation",
                (clip.clip_id,),
                clip.clip_id,
                preset_id=clip.clip_id,
            ),
            clip.clip_id,
        )
        add_model(
            "live2d_motion",
            _key(
                "live2d_moc3_v4_00",
                "live2d_motion",
                (clip.clip_id,),
                clip.clip_id,
                preset_id=clip.clip_id,
                directory="motions",
            ),
            clip.clip_id,
        )
    for expression in presets.expressions:
        add_model(
            "spine_animation",
            _key(
                "spine_4_2",
                "spine_animation",
                (expression.expression_id,),
                expression.expression_id,
                preset_id=expression.expression_id,
            ),
            expression.expression_id,
        )
        add_model(
            "live2d_expression",
            _key(
                "live2d_moc3_v4_00",
                "live2d_expression",
                (expression.expression_id,),
                expression.expression_id,
                preset_id=expression.expression_id,
                directory="expressions",
            ),
            expression.expression_id,
        )

    for page_id in texture_page_ids:
        for format_id in _FORMATS:
            add_model(
                "texture_page",
                _key(format_id, "texture_page", (page_id,), page_id),
                page_id,
            )

    deterministic_absences: list[str] = []
    for binding in bindings.bindings:
        target_mesh = mesh_by_id.get(binding.target_id)
        if binding.target_id in bone_ids:
            spine_target_key = _key(
                "spine_4_2",
                "spine_bone",
                (binding.target_id,),
                binding.target_id,
            )
        elif target_mesh is not None:
            part_meshes = meshes_by_part[target_mesh.part_id]
            tokens = (
                ()
                if len(part_meshes) == 1
                else (_component_token(target_mesh.component_id),)
            )
            spine_target_key = _key(
                "spine_4_2",
                "spine_attachment_object",
                (target_mesh.part_id, target_mesh.component_id),
                target_mesh.part_id,
                derivation_tokens=tokens,
                component_id=target_mesh.component_id,
            )
        else:
            raise _error(f"binding target is not a bone or weighted mesh: {binding.target_id}")
        candidates.append(
            _candidate(
                candidate_kind="spine_binding",
                key=spine_target_key,
                binding=binding,
                required_facts=binding.required_rig_facts,
                driver_registry_entry_id=binding.implementation_id,
            )
        )

        control = control_by_id.get(binding.control_id)
        if control is None:
            raise _error(f"binding references an unknown control: {binding.control_id}")
        if control.live2d is None:
            deterministic_absences.append(
                f"live2d_parameter_absent:{binding.control_id}:{binding.binding_id}"
            )
            continue
        parameter_id = control.live2d.parameter_id
        if binding.target_id in bone_ids:
            live_key = _key(
                "live2d_moc3_v4_00",
                "live2d_rotation_deformer",
                (binding.target_id,),
                binding.target_id,
                derivation_tokens=("rot", _id_token(parameter_id)),
                parameter_id=parameter_id,
                control_id=binding.control_id,
            )
        else:
            assert target_mesh is not None
            part_meshes = meshes_by_part[target_mesh.part_id]
            tokens = (
                ()
                if len(part_meshes) == 1
                else (_component_token(target_mesh.component_id),)
            )
            live_key = _key(
                "live2d_moc3_v4_00",
                "live2d_artmesh",
                (target_mesh.part_id, target_mesh.component_id),
                target_mesh.part_id,
                derivation_tokens=tokens,
                component_id=target_mesh.component_id,
            )
        dynamic_keys[live_key.key_sha256] = live_key
        candidates.append(
            _candidate(
                candidate_kind="live2d_binding",
                key=live_key,
                binding=binding,
                required_facts=binding.required_rig_facts,
                driver_registry_entry_id=binding.implementation_id,
            )
        )

    existing_key_kinds = {
        (candidate.typed_primitive_key.key_sha256, candidate.candidate_kind)
        for candidate in candidates
    }
    for key in dynamic_keys.values():
        if (key.key_sha256, key.kind) not in existing_key_kinds:
            add_model(key.kind, key, *key.source_internal_ids)

    ordered = tuple(sorted(candidates, key=lambda item: item.candidate_id))
    if len({candidate.candidate_id for candidate in ordered}) != len(ordered):
        raise _error("candidate identities collide")
    return ordered, tuple(sorted(deterministic_absences))


def validate_primitive_candidate_set(
    plan: PrimitiveCandidateSet,
    cache: RigGeometryCache,
    controls: ControlRegistryPlan,
    presets: PresetLibraryPlan,
    bindings: ControlBindingPlan,
    *,
    texture_page_ids: tuple[str, ...] = (),
) -> PrimitiveCandidateSet:
    """Re-enumerate the complete universe and reject profile/pruning drift."""

    normalized_pages = _validate_inputs(
        cache, controls, presets, bindings, texture_page_ids
    )
    if (
        not isinstance(plan, PrimitiveCandidateSet)
        or plan.schema_version != PRIMITIVE_CANDIDATE_SET_VERSION
        or plan.enumerator_version != PRIMITIVE_CANDIDATE_ENUMERATOR_VERSION
    ):
        raise _error("candidate set version is unsupported")
    if (
        plan.rig_geometry_cache_sha256 != cache.cache_sha256
        or plan.control_registry_sha256 != controls.registry_sha256
        or plan.preset_library_sha256 != presets.plan_sha256
        or plan.control_binding_plan_sha256 != bindings.plan_sha256
        or plan.texture_page_ids != normalized_pages
    ):
        raise _error("candidate set provenance differs from its inputs")
    expected, expected_absences = _enumerate(
        cache, controls, presets, bindings, normalized_pages
    )
    if plan.candidates != expected or plan.deterministic_absences != expected_absences:
        raise _error("candidate records differ from the complete enumerator output")
    if any(
        candidate.candidate_sha256 != jcs_sha256(candidate.semantic_payload())
        or candidate.typed_primitive_key.key_sha256
        != jcs_sha256(candidate.typed_primitive_key.semantic_payload())
        or candidate.format_id != candidate.typed_primitive_key.format_id
        for candidate in plan.candidates
    ):
        raise _error("candidate or typed-key digest is invalid")
    universe_payload = [
        candidate.to_dict() for candidate in plan.candidates
    ] + [{"deterministic_absence": value} for value in plan.deterministic_absences]
    if plan.candidate_universe_sha256 != jcs_sha256(universe_payload):
        raise _error("candidate universe digest mismatch")
    if plan.plan_sha256 != jcs_sha256(plan.semantic_payload()):
        raise _error("candidate set digest mismatch")
    return plan


def enumerate_primitive_candidates(
    cache: RigGeometryCache,
    controls: ControlRegistryPlan,
    presets: PresetLibraryPlan,
    bindings: ControlBindingPlan,
    *,
    texture_page_ids: tuple[str, ...] = (),
) -> PrimitiveCandidateSet:
    """Enumerate both exporter families without profile or capability pruning."""

    normalized_pages = _validate_inputs(
        cache, controls, presets, bindings, texture_page_ids
    )
    candidates, absences = _enumerate(
        cache, controls, presets, bindings, normalized_pages
    )
    universe_payload = [candidate.to_dict() for candidate in candidates] + [
        {"deterministic_absence": value} for value in absences
    ]
    values = {
        "schema_version": PRIMITIVE_CANDIDATE_SET_VERSION,
        "enumerator_version": PRIMITIVE_CANDIDATE_ENUMERATOR_VERSION,
        "rig_geometry_cache_sha256": cache.cache_sha256,
        "control_registry_sha256": controls.registry_sha256,
        "preset_library_sha256": presets.plan_sha256,
        "control_binding_plan_sha256": bindings.plan_sha256,
        "texture_page_ids": normalized_pages,
        "deterministic_absences": absences,
        "candidates": candidates,
        "candidate_universe_sha256": jcs_sha256(universe_payload),
    }
    provisional = PrimitiveCandidateSet(**values, plan_sha256="")
    plan = PrimitiveCandidateSet(
        **values,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )
    return validate_primitive_candidate_set(
        plan,
        cache,
        controls,
        presets,
        bindings,
        texture_page_ids=normalized_pages,
    )


__all__ = [
    "PRIMITIVE_CANDIDATE_ENUMERATOR_VERSION",
    "PRIMITIVE_CANDIDATE_SET_VERSION",
    "TYPED_PRIMITIVE_KEY_VERSION",
    "PrimitiveBindingTemplate",
    "PrimitiveCandidate",
    "PrimitiveCandidateSet",
    "PrimitiveCandidateSetError",
    "TypedPrimitiveKey",
    "enumerate_primitive_candidates",
    "validate_primitive_candidate_set",
]
