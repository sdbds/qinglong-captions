from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .capabilities import CapabilityPlan
from .control_bindings import ControlBindingPlan
from .control_registry import ControlRegistryPlan, validate_control_registry_plan
from .export_symbols import (
    GlobalExportSymbolTable,
    validate_global_export_symbol_table,
)
from .format_plans import FormatPlanSet, validate_format_plan_set
from .jcs import JcsContractError, jcs_bytes, jcs_sha256
from .preset_library import PresetLibraryPlan, validate_preset_library_plan
from .primitive_candidates import (
    PrimitiveCandidateSet,
    validate_primitive_candidate_set,
)
from .rig_geometry import RigGeometryCache, validate_rig_geometry_cache
from .texture_plan import CanonicalTexturePageSet, TexturePagePlan

RIG_DOCUMENT_SCHEMA_VERSION = 1
RIG_DOCUMENT_GENERATOR_NAME = "qinglong-auto-rig"
RIG_DOCUMENT_ALGORITHM_VERSION = "rig-document-v1"

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "generator",
        "input_fingerprint",
        "input",
        "canvas",
        "parts",
        "joint_observations",
        "joints",
        "bones",
        "meshes",
        "capabilities",
        "control_specs",
        "control_bindings",
        "clips",
        "expressions",
        "runtime_application",
        "format_plans",
        "primitive_candidates",
        "export_symbols",
        "texture_pages",
        "diagnostics",
        "provenance",
    }
)


class RigDocumentError(ValueError):
    """Raised when a public C-complete RigDocument is partial or inconsistent."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _error(message: str) -> RigDocumentError:
    return RigDocumentError("invalid_rig_document", message)


def _decode_canonical(payload: bytes) -> dict[str, object]:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise _error(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error("RigDocument is not valid UTF-8 JSON") from exc
    if type(value) is not dict:
        raise _error("RigDocument root must be an object")
    try:
        if jcs_bytes(value) != payload:
            raise _error("RigDocument bytes are not canonical JCS")
    except JcsContractError as exc:
        raise _error("RigDocument is outside the JCS value domain") from exc
    return value


@dataclass(frozen=True, slots=True)
class RigDocument:
    """Immutable canonical bytes with typed read-only section projections."""

    _canonical_json: bytes

    def to_dict(self) -> dict[str, object]:
        return _decode_canonical(self._canonical_json)

    @property
    def schema_version(self) -> int:
        return int(self.to_dict()["schema_version"])

    @property
    def input_fingerprint(self) -> str:
        return str(self.to_dict()["input_fingerprint"])

    @property
    def parts(self) -> tuple[dict[str, object], ...]:
        return tuple(self.to_dict()["parts"])  # type: ignore[arg-type]

    @property
    def joints(self) -> tuple[dict[str, object], ...]:
        return tuple(self.to_dict()["joints"])  # type: ignore[arg-type]

    @property
    def bones(self) -> tuple[dict[str, object], ...]:
        return tuple(self.to_dict()["bones"])  # type: ignore[arg-type]

    @property
    def meshes(self) -> tuple[dict[str, object], ...]:
        return tuple(self.to_dict()["meshes"])  # type: ignore[arg-type]

    @property
    def profile_id(self) -> str:
        section = self.to_dict()["format_plans"]
        assert isinstance(section, dict)
        profile = section["profile"]
        assert isinstance(profile, dict)
        return str(profile["profile_id"])

    @property
    def document_sha256(self) -> str:
        return jcs_sha256(self.to_dict())


def _with_plan_digest(plan) -> dict[str, object]:
    return {**plan.semantic_payload(), "plan_sha256": plan.plan_sha256}


def _with_table_digest(table: GlobalExportSymbolTable) -> dict[str, object]:
    return {**table.semantic_payload(), "table_sha256": table.table_sha256}


def _native_part_quality(
    part_id: str,
    *,
    native_quality_by_part: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    return native_quality_by_part.get(part_id)


def _public_parts(
    cache: RigGeometryCache,
    *,
    native_quality_by_part: dict[str, dict[str, object]],
    native_composite_mode_by_part: dict[str, str],
) -> list[dict[str, object]]:
    result = []
    for part in cache.parts:
        payload = part.to_dict()
        payload.update(
            {
                "setup_visibility": 0.0 if part.source_kind == "native_variant" else 1.0,
                "composite_mode": native_composite_mode_by_part.get(part.part_id),
                "draw_policy_sha256": cache.final_draw_order.plan_sha256,
                "native_quality": _native_part_quality(
                    part.part_id,
                    native_quality_by_part=native_quality_by_part,
                ),
            }
        )
        if part.source_kind == "native_variant" and payload["composite_mode"] is None:
            raise _error("native Part lacks its authenticated composite mode")
        result.append(payload)
    return result


def _public_meshes(cache: RigGeometryCache) -> list[dict[str, object]]:
    rank_by_component = {
        record.component_id: record.component_draw_rank
        for record in cache.component_draw_order.records
    }
    mesh_by_component = {
        mesh.component_id: mesh for mesh in cache.skinning_plan.weighted_meshes
    }
    return [
        {
            **mesh_by_component[record.component_id].to_dict(),
            "component_draw_rank": record.component_draw_rank,
            "component_draw_order_plan_sha256": cache.component_draw_order.plan_sha256,
        }
        for record in cache.component_draw_order.records
        if record.component_id in rank_by_component
    ]


def _texture_page_records(
    plan: TexturePagePlan,
    page_set: CanonicalTexturePageSet,
) -> list[dict[str, object]]:
    if (
        plan.plan_sha256 != jcs_sha256(plan.semantic_payload())
        or not plan.fit
        or plan.failure_reason is not None
        or plan.unplaced_part_ids
    ):
        raise _error("TexturePagePlan is not a complete fit")
    if (
        page_set.set_sha256 != jcs_sha256(page_set.semantic_payload())
        or page_set.texture_page_plan_sha256 != plan.plan_sha256
    ):
        raise _error("canonical texture page set differs from TexturePagePlan")
    plan_pages = {page.index: page for page in plan.pages}
    materialized = {page.index: page for page in page_set.pages}
    if set(plan_pages) != set(materialized) or tuple(sorted(plan_pages)) != tuple(
        range(len(plan_pages))
    ):
        raise _error("materialized texture pages differ from the page plan")
    placements: dict[int, list[dict[str, object]]] = {}
    for placement in plan.placements:
        placements.setdefault(placement.page_index, []).append(placement.to_dict())
    records = []
    for index in sorted(plan_pages):
        page = plan_pages[index]
        material = materialized[index]
        expected_path = f"rig/shared/textures/{page.relative_path}"
        if (
            material.relative_path != expected_path
            or material.file.path != expected_path
            or material.file.sha256 != material.encoded_png_sha256
            or material.width != plan.page_width
            or material.height != plan.page_height
        ):
            raise _error("materialized texture page violates the canonical path contract")
        records.append(
            {
                "texture_page_id": f"texture-page/page_{index}",
                "index": index,
                "width": material.width,
                "height": material.height,
                "relative_path": material.relative_path,
                "region_part_ids": list(page.region_part_ids),
                "packed_area": page.packed_area,
                "placements": sorted(
                    placements.get(index, ()), key=lambda item: item["part_id"]
                ),
                "canonical_uv_space": "page_top_left_v_down",
                "alpha_mode": "straight",
                "color_space": "srgb_bytes",
                "texture_page_plan_sha256": plan.plan_sha256,
                "canonical_texture_page_set_sha256": page_set.set_sha256,
                "encoder": page_set.encoder.to_dict(),
                "rgba_sha256": material.rgba_sha256,
                "encoded_png_sha256": material.encoded_png_sha256,
                "file": material.file.to_dict(),
            }
        )
    return records


def _as_list(value: object, field: str, *, nonempty: bool = False) -> list[object]:
    if type(value) is not list or (nonempty and not value):
        raise _error(f"{field} must be a{' non-empty' if nonempty else ''} array")
    return value


def _as_dict(value: object, field: str) -> dict[str, object]:
    if type(value) is not dict:
        raise _error(f"{field} must be an object")
    return value


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise _error(f"{field} must be a SHA-256 digest")
    return value


def _ids(
    records: list[object],
    field: str,
    *,
    group: str,
) -> dict[str, dict[str, object]]:
    result = {}
    for raw in records:
        record = _as_dict(raw, group)
        value = record.get(field)
        if not isinstance(value, str) or not value:
            raise _error(f"{group}.{field} must be a non-empty string")
        if value in result:
            raise _error(f"duplicate {group} ID: {value}")
        result[value] = record
    return result


def _verify_removed_digest(
    record: dict[str, object],
    digest_field: str,
    *,
    group: str,
    extra_removed: tuple[str, ...] = (),
) -> None:
    observed = _digest(record.get(digest_field), f"{group}.{digest_field}")
    payload = {
        key: value
        for key, value in record.items()
        if key != digest_field and key not in extra_removed
    }
    if observed != jcs_sha256(payload):
        raise _error(f"{group} content digest mismatch")


def _validate_geometry(payload: dict[str, object]) -> tuple[set[str], set[str], set[str]]:
    parts = _as_list(payload["parts"], "parts", nonempty=True)
    part_by_id = _ids(parts, "part_id", group="part")
    part_ranks = sorted(int(part["part_draw_rank"]) for part in part_by_id.values())
    if part_ranks != list(range(len(parts))):
        raise _error("Part draw ranks are not gapless")
    component_owner = {}
    for part_id, part in part_by_id.items():
        components = _as_list(part.get("component_ids"), "part.component_ids", nonempty=True)
        for component_id in components:
            if not isinstance(component_id, str) or component_id in component_owner:
                raise _error("component IDs are invalid or duplicated across Parts")
            component_owner[component_id] = part_id
        source_kind = part.get("source_kind")
        if source_kind not in {"see_through", "native_variant"}:
            raise _error("Part source_kind is invalid")
        expected_visibility = 0.0 if source_kind == "native_variant" else 1.0
        if part.get("setup_visibility") != expected_visibility:
            raise _error("Part setup visibility differs from its source kind")

    observations = _as_list(payload["joint_observations"], "joint_observations")
    observation_by_id = _ids(observations, "observation_id", group="joint_observation")
    joints = _as_list(payload["joints"], "joints", nonempty=True)
    joint_by_id = _ids(joints, "joint_id", group="joint")
    for joint in joint_by_id.values():
        selected = joint.get("selected_observation_id")
        candidates = _as_list(
            joint.get("candidate_observation_ids"), "joint.candidate_observation_ids"
        )
        if selected is not None and selected not in observation_by_id:
            raise _error("joint selects an unknown observation")
        if any(candidate not in observation_by_id for candidate in candidates):
            raise _error("joint candidate list contains an unknown observation")

    bones = _as_list(payload["bones"], "bones", nonempty=True)
    bone_by_id = _ids(bones, "bone_id", group="bone")
    root = bone_by_id.get("bone/root")
    if root is None or root.get("role") != "synthetic_root":
        raise _error("RigDocument lacks its synthetic root bone")
    for bone_id, bone in bone_by_id.items():
        parent = bone.get("parent_id")
        head = bone.get("head_joint_id")
        tail = bone.get("tail_joint_id")
        if bone_id == "bone/root":
            if any(value is not None for value in (parent, head, tail)):
                raise _error("synthetic root has non-null topology references")
            continue
        if parent not in bone_by_id or head not in joint_by_id or tail not in joint_by_id:
            raise _error("bone topology references an unknown parent or joint")

    meshes = _as_list(payload["meshes"], "meshes", nonempty=True)
    mesh_by_id = _ids(meshes, "mesh_id", group="mesh")
    ranks = sorted(int(mesh["component_draw_rank"]) for mesh in mesh_by_id.values())
    if ranks != list(range(len(meshes))):
        raise _error("component draw ranks are not gapless")
    observed_components = set()
    for mesh in mesh_by_id.values():
        _verify_removed_digest(
            mesh,
            "weighted_mesh_sha256",
            group="mesh",
            extra_removed=("component_draw_rank", "component_draw_order_plan_sha256"),
        )
        part_id = mesh.get("part_id")
        component_id = mesh.get("component_id")
        if part_id not in part_by_id or component_owner.get(component_id) != part_id:
            raise _error("mesh references an unknown Part/component pair")
        if component_id in observed_components:
            raise _error("multiple public meshes reference one component")
        observed_components.add(component_id)
        vertices = _as_list(mesh.get("vertices"), "mesh.vertices", nonempty=True)
        triangles = _as_list(mesh.get("triangles"), "mesh.triangles", nonempty=True)
        if len(triangles) % 3 or any(
            not isinstance(index, int) or not 0 <= index < len(vertices)
            for index in triangles
        ):
            raise _error("mesh triangle indices are invalid")
        for vertex in vertices:
            vertex_record = _as_dict(vertex, "mesh.vertex")
            uv = _as_list(vertex_record.get("uv"), "mesh.vertex.uv")
            if len(uv) != 2 or any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
                for value in uv
            ):
                raise _error("mesh UV lies outside canonical top-left space")
            influences = _as_list(
                vertex_record.get("influences"), "mesh.vertex.influences", nonempty=True
            )
            if not 1 <= len(influences) <= 4:
                raise _error("mesh vertex influence count is invalid")
            total = 0.0
            for influence in influences:
                influence_record = _as_dict(influence, "mesh.influence")
                if influence_record.get("bone_id") not in bone_by_id:
                    raise _error("mesh influence references an unknown bone")
                weight = influence_record.get("weight")
                if not isinstance(weight, (int, float)) or float(weight) < 0:
                    raise _error("mesh influence weight is invalid")
                total += float(weight)
            if abs(total - 1.0) > 1e-6:
                raise _error("mesh influence weights do not sum to one")
    if observed_components != set(component_owner):
        raise _error("public mesh set does not cover every Part component")
    return set(part_by_id), set(bone_by_id), set(mesh_by_id)


def _validate_motion_and_bindings(
    payload: dict[str, object],
    *,
    target_ids: set[str],
) -> tuple[set[str], set[str]]:
    capabilities = _as_list(payload["capabilities"], "capabilities", nonempty=True)
    capability_by_preset = _ids(capabilities, "preset_id", group="capability")
    for capability in capability_by_preset.values():
        _verify_removed_digest(
            capability, "capability_sha256", group="capability"
        )
    controls = _as_list(payload["control_specs"], "control_specs", nonempty=True)
    control_by_id = _ids(controls, "control_id", group="control")
    registry_digests = {
        _digest(control.get("registry_sha256"), "control.registry_sha256")
        for control in control_by_id.values()
    }
    if len(registry_digests) != 1:
        raise _error("control specs disagree on registry digest")
    bindings = _as_list(payload["control_bindings"], "control_bindings", nonempty=True)
    binding_by_id = _ids(bindings, "binding_id", group="control_binding")
    by_implementation: dict[str, list[dict[str, object]]] = {}
    for binding in binding_by_id.values():
        transfer = _as_dict(binding.get("transfer"), "control_binding.transfer")
        _verify_removed_digest(transfer, "transfer_sha256", group="target_transfer")
        _verify_removed_digest(binding, "binding_sha256", group="control_binding")
        if binding.get("control_id") not in control_by_id:
            raise _error("control binding references an unknown control")
        if binding.get("target_id") not in target_ids:
            raise _error("control binding references an unknown target")
        implementation_id = binding.get("implementation_id")
        if not isinstance(implementation_id, str):
            raise _error("control binding implementation ID is invalid")
        by_implementation.setdefault(implementation_id, []).append(binding)
    for implementation_id, members in by_implementation.items():
        bundle_payload = {
            "schema_version": "control-binding-bundle-v1",
            "implementation_id": implementation_id,
            "bindings": [
                {
                    "binding_id": member["binding_id"],
                    "control_id": member["control_id"],
                    "target_id": member["target_id"],
                    "property": member["property"],
                    "visibility_branch_id": member["visibility_branch_id"],
                    "required_rig_facts": member["required_rig_facts"],
                    "transfer_sha256": member["transfer"]["transfer_sha256"],  # type: ignore[index]
                }
                for member in sorted(members, key=lambda row: str(row["binding_id"]))
            ],
        }
        expected = jcs_sha256(bundle_payload)
        if any(member.get("implementation_bundle_digest") != expected for member in members):
            raise _error("control binding bundle digest mismatch")

    clips = _as_list(payload["clips"], "clips", nonempty=True)
    expressions = _as_list(payload["expressions"], "expressions", nonempty=True)
    runtime_application = _as_dict(
        payload["runtime_application"], "runtime_application"
    )
    if set(runtime_application) != {"version", "motion", "expression"}:
        raise _error("motion runtime-application contract is invalid")
    preset_records = []
    for clip in clips:
        record = _as_dict(clip, "clip")
        _verify_removed_digest(record, "template_sha256", group="clip")
        for curve in _as_list(record.get("control_curves"), "clip.control_curves"):
            if _as_dict(curve, "control_curve").get("control_id") not in control_by_id:
                raise _error("clip references an unknown control")
        preset_records.append(record)
    for expression in expressions:
        record = _as_dict(expression, "expression")
        _verify_removed_digest(record, "template_sha256", group="expression")
        for value in _as_list(record.get("values"), "expression.values"):
            if _as_dict(value, "expression_value").get("control_id") not in control_by_id:
                raise _error("expression references an unknown control")
        preset_records.append(record)
    preset_by_id = _ids(preset_records, "preset_id", group="preset")
    if set(preset_by_id) != set(capability_by_preset):
        raise _error("capability and preset universes differ")
    return set(binding_by_id), set(preset_by_id)


def _validate_candidates_and_symbols(
    payload: dict[str, object],
    *,
    binding_ids: set[str],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    candidate_section = _as_dict(
        payload["primitive_candidates"], "primitive_candidates"
    )
    _verify_removed_digest(
        candidate_section, "plan_sha256", group="primitive_candidates"
    )
    candidates = _as_list(
        candidate_section.get("candidates"), "primitive_candidates.candidates", nonempty=True
    )
    candidate_by_id = _ids(candidates, "candidate_id", group="primitive_candidate")
    keys: dict[str, dict[str, object]] = {}
    for candidate in candidate_by_id.values():
        _verify_removed_digest(candidate, "candidate_sha256", group="primitive_candidate")
        key = _as_dict(candidate.get("typed_primitive_key"), "typed_primitive_key")
        _verify_removed_digest(key, "key_sha256", group="typed_primitive_key")
        key_sha = str(key["key_sha256"])
        previous = keys.get(key_sha)
        if previous is not None and previous != key:
            raise _error("typed primitive key digest collision")
        keys[key_sha] = key
        if candidate.get("primitive_target_id") != "primitive/p_" + key_sha.removeprefix(
            "sha256:"
        ):
            raise _error("primitive target ID differs from typed key")
        template = candidate.get("binding_template")
        if template is not None:
            binding_id = _as_dict(template, "binding_template").get("binding_id")
            if binding_id not in binding_ids:
                raise _error("primitive candidate references an unknown binding")

    symbol_section = _as_dict(payload["export_symbols"], "export_symbols")
    _verify_removed_digest(symbol_section, "table_sha256", group="export_symbols")
    symbols = _as_list(symbol_section.get("symbols"), "export_symbols.symbols", nonempty=True)
    symbol_by_id = _ids(symbols, "symbol_id", group="export_symbol")
    symbol_keys = set()
    namespace_names = set()
    for symbol in symbol_by_id.values():
        _verify_removed_digest(symbol, "symbol_sha256", group="export_symbol")
        key = _as_dict(symbol.get("typed_primitive_key"), "export_symbol.typed_key")
        key_sha = key.get("key_sha256")
        if key_sha not in keys or keys[key_sha] != key:
            raise _error("export symbol references an unknown typed key")
        namespace = _as_dict(symbol.get("namespace_key"), "export_namespace")
        _verify_removed_digest(namespace, "key_sha256", group="export_namespace")
        export_name = symbol.get("export_name")
        if (
            not isinstance(export_name, str)
            or not export_name
            or not export_name.isascii()
            or len(export_name.encode("ascii")) >= 64
        ):
            raise _error("export symbol name violates the ASCII length contract")
        uniqueness = (namespace["key_sha256"], export_name)
        if uniqueness in namespace_names:
            raise _error("export names collide inside a namespace")
        namespace_names.add(uniqueness)
        symbol_keys.add(key_sha)
    if symbol_keys != set(keys):
        raise _error("global symbol table does not cover the candidate key universe")
    return candidate_by_id, symbol_by_id


def _validate_format_plans(
    payload: dict[str, object],
    *,
    preset_ids: set[str],
    candidates: dict[str, dict[str, object]],
    symbols: dict[str, dict[str, object]],
) -> None:
    section = _as_dict(payload["format_plans"], "format_plans")
    _verify_removed_digest(section, "plan_sha256", group="format_plans")
    profile = _as_dict(section.get("profile"), "format_plans.profile")
    _verify_removed_digest(profile, "profile_sha256", group="capability_profile")
    required_formats = set(
        _as_list(profile.get("required_formats"), "profile.required_formats", nonempty=True)
    )
    required_presets = set(
        _as_list(
            profile.get("required_preset_ids"), "profile.required_preset_ids", nonempty=True
        )
    )
    if not required_presets <= preset_ids:
        raise _error("profile requires an unknown preset")
    model_plans = _as_list(section.get("model_plans"), "format_plans.model_plans", nonempty=True)
    model_by_format = _ids(model_plans, "format_id", group="format_model_plan")
    set_plans = _as_list(
        section.get("preset_set_plans"), "format_plans.preset_set_plans", nonempty=True
    )
    sets_by_format = _ids(set_plans, "format_id", group="format_preset_set_plan")
    if set(model_by_format) != required_formats or set(sets_by_format) != required_formats:
        raise _error("format plans do not cover required formats exactly")
    symbol_by_key = {
        symbol["typed_primitive_key"]["key_sha256"]: symbol  # type: ignore[index]
        for symbol in symbols.values()
    }
    for format_id in required_formats:
        model = model_by_format[format_id]
        _verify_removed_digest(model, "plan_sha256", group="format_model_plan")
        if model.get("status") != "supported":
            raise _error("required format model plan is unsupported")
        set_plan = sets_by_format[format_id]
        _verify_removed_digest(set_plan, "plan_sha256", group="format_preset_set_plan")
        decisions = _as_list(
            set_plan.get("decisions"), "format_preset_set_plan.decisions", nonempty=True
        )
        decision_by_preset = _ids(decisions, "preset_id", group="format_preset_decision")
        if set(decision_by_preset) != preset_ids:
            raise _error("format preset decisions do not cover the preset universe")
        for preset_id, decision in decision_by_preset.items():
            _verify_removed_digest(
                decision, "decision_sha256", group="format_preset_decision"
            )
            required = preset_id in required_presets
            if bool(decision.get("required")) != required:
                raise _error("format decision required flag differs from profile")
            if required and decision.get("status") != "supported":
                raise _error("required preset is omitted")
            candidate_ids = _as_list(
                decision.get("selected_candidate_ids"),
                "format_decision.selected_candidate_ids",
            )
            if any(candidate_id not in candidates for candidate_id in candidate_ids):
                raise _error("format decision selects an unknown candidate")
            expected_keys = sorted(
                {
                    candidates[candidate_id]["typed_primitive_key"]["key_sha256"]  # type: ignore[index]
                    for candidate_id in candidate_ids
                }
            )
            if decision.get("selected_primitive_key_sha256") != expected_keys:
                raise _error("format decision primitive-key projection is stale")
            artifact_candidate_id = decision.get("artifact_candidate_id")
            artifact_symbol_id = decision.get("artifact_symbol_id")
            artifact_name = decision.get("artifact_export_name")
            if decision.get("status") == "supported":
                if (
                    artifact_candidate_id not in candidates
                    or artifact_symbol_id not in symbols
                ):
                    raise _error("supported preset lacks an artifact candidate/symbol")
                artifact_key = candidates[artifact_candidate_id]["typed_primitive_key"][  # type: ignore[index]
                    "key_sha256"
                ]
                if (
                    symbol_by_key.get(artifact_key) != symbols[artifact_symbol_id]
                    or symbols[artifact_symbol_id].get("export_name") != artifact_name
                ):
                    raise _error("format artifact symbol projection is inconsistent")


def _validate_texture_pages(
    payload: dict[str, object],
    *,
    part_ids: set[str],
    candidate_keys: dict[str, dict[str, object]],
) -> None:
    pages = _as_list(payload["texture_pages"], "texture_pages", nonempty=True)
    page_by_id = _ids(pages, "texture_page_id", group="texture_page")
    if sorted(int(page["index"]) for page in page_by_id.values()) != list(
        range(len(pages))
    ):
        raise _error("texture page indices are not gapless")
    placed_parts = set()
    for page_id, page in page_by_id.items():
        expected_path = f"rig/shared/textures/page_{page['index']}.png"
        if page.get("relative_path") != expected_path:
            raise _error("texture page path is outside the shared namespace")
        if page.get("canonical_uv_space") != "page_top_left_v_down":
            raise _error("texture page canonical UV space is invalid")
        if page.get("alpha_mode") != "straight" or page.get("color_space") != "srgb_bytes":
            raise _error("texture page pixel contract is invalid")
        file_record = _as_dict(page.get("file"), "texture_page.file")
        if (
            file_record.get("path") != expected_path
            or file_record.get("sha256") != page.get("encoded_png_sha256")
        ):
            raise _error("texture page file digest/path mismatch")
        for placement in _as_list(page.get("placements"), "texture_page.placements"):
            placement_record = _as_dict(placement, "texture_placement")
            part_id = placement_record.get("part_id")
            if part_id not in part_ids or part_id in placed_parts:
                raise _error("texture placement Part is unknown or duplicated")
            if placement_record.get("page_index") != page.get("index"):
                raise _error("texture placement page index mismatch")
            placed_parts.add(part_id)
    if placed_parts != part_ids:
        raise _error("texture placements do not cover every render Part")
    page_source_ids = {
        key["base_source_internal_id"]
        for key in candidate_keys.values()
        if key.get("kind") == "texture_page"
    }
    if page_source_ids != set(page_by_id):
        raise _error("texture page candidates differ from materialized pages")


def validate_rig_document_payload(payload: object) -> RigDocument:
    """Validate a decoded payload and return its immutable canonical form."""

    root = _as_dict(payload, "RigDocument")
    if set(root) != _ROOT_FIELDS:
        missing = sorted(_ROOT_FIELDS - set(root))
        extra = sorted(set(root) - _ROOT_FIELDS)
        raise _error(f"RigDocument fields differ; missing={missing}, extra={extra}")
    if root.get("schema_version") != RIG_DOCUMENT_SCHEMA_VERSION:
        raise _error("RigDocument schema version is unsupported")
    generator = _as_dict(root["generator"], "generator")
    if set(generator) != {"name", "algorithm_version"}:
        raise _error("generator descriptor is invalid")
    _digest(root.get("input_fingerprint"), "input_fingerprint")
    input_record = _as_dict(root["input"], "input")
    canvas = _as_dict(root["canvas"], "canvas")
    if (
        input_record.get("coordinate_space") != "layerdiff_canvas"
        or canvas.get("origin") != "top_left"
        or canvas.get("y_axis") != "down"
        or canvas.get("width") != canvas.get("height")
    ):
        raise _error("input/canvas coordinate contract is invalid")
    _digest(input_record.get("native_variant_set_sha256"), "native_variant_set_sha256")
    _digest(
        input_record.get("native_variant_eligibility_sha256"),
        "native_variant_eligibility_sha256",
    )
    part_ids, bone_ids, mesh_ids = _validate_geometry(root)
    binding_ids, preset_ids = _validate_motion_and_bindings(
        root,
        target_ids=bone_ids | mesh_ids,
    )
    candidate_by_id, symbol_by_id = _validate_candidates_and_symbols(
        root,
        binding_ids=binding_ids,
    )
    _validate_format_plans(
        root,
        preset_ids=preset_ids,
        candidates=candidate_by_id,
        symbols=symbol_by_id,
    )
    candidate_keys = {
        candidate["typed_primitive_key"]["key_sha256"]: candidate["typed_primitive_key"]  # type: ignore[index]
        for candidate in candidate_by_id.values()
    }
    _validate_texture_pages(root, part_ids=part_ids, candidate_keys=candidate_keys)
    _as_list(root["diagnostics"], "diagnostics")

    provenance = _as_dict(root["provenance"], "provenance")
    for field in (
        "rig_geometry_cache_sha256",
        "stage_a_joint_plan_sha256",
        "bone_graph_plan_sha256",
        "skinning_plan_sha256",
        "component_draw_order_plan_sha256",
        "capability_plan_sha256",
        "control_registry_sha256",
        "preset_library_sha256",
        "control_binding_plan_sha256",
        "format_plan_set_sha256",
        "primitive_candidate_set_sha256",
        "global_export_symbol_table_sha256",
        "texture_page_plan_sha256",
        "canonical_texture_page_set_sha256",
    ):
        _digest(provenance.get(field), f"provenance.{field}")
    if provenance.get("degradation_state") not in {"clean", "degraded"}:
        raise _error("provenance degradation state is invalid")
    degradation_codes = _as_list(
        provenance.get("degradation_codes"), "provenance.degradation_codes"
    )
    if degradation_codes != sorted(set(degradation_codes)):
        raise _error("provenance degradation codes are not canonical")
    if provenance["format_plan_set_sha256"] != root["format_plans"]["plan_sha256"]:  # type: ignore[index]
        raise _error("format plan provenance mismatch")
    if provenance["primitive_candidate_set_sha256"] != root["primitive_candidates"][  # type: ignore[index]
        "plan_sha256"
    ]:
        raise _error("candidate-set provenance mismatch")
    if provenance["global_export_symbol_table_sha256"] != root["export_symbols"][  # type: ignore[index]
        "table_sha256"
    ]:
        raise _error("symbol-table provenance mismatch")
    texture_plan_digests = {
        page["texture_page_plan_sha256"]  # type: ignore[index]
        for page in root["texture_pages"]  # type: ignore[union-attr]
    }
    texture_set_digests = {
        page["canonical_texture_page_set_sha256"]  # type: ignore[index]
        for page in root["texture_pages"]  # type: ignore[union-attr]
    }
    if texture_plan_digests != {provenance["texture_page_plan_sha256"]}:
        raise _error("texture-plan provenance mismatch")
    if texture_set_digests != {provenance["canonical_texture_page_set_sha256"]}:
        raise _error("canonical texture-set provenance mismatch")
    try:
        canonical = jcs_bytes(root)
    except JcsContractError as exc:
        raise _error("RigDocument contains a non-JCS value") from exc
    return RigDocument(canonical)


def validate_rig_document(rig: RigDocument) -> RigDocument:
    if not isinstance(rig, RigDocument):
        raise _error("RigDocument has the wrong runtime type")
    validated = validate_rig_document_payload(rig.to_dict())
    if validated != rig:
        raise _error("RigDocument canonical bytes differ from validated payload")
    return rig


def build_rig_document(
    cache: RigGeometryCache,
    controls: ControlRegistryPlan,
    presets: PresetLibraryPlan,
    capabilities: CapabilityPlan,
    bindings: ControlBindingPlan,
    formats: FormatPlanSet,
    candidates: PrimitiveCandidateSet,
    symbols: GlobalExportSymbolTable,
    texture_plan: TexturePagePlan,
    texture_page_set: CanonicalTexturePageSet,
    *,
    native_composite_mode_by_part: dict[str, str] | None = None,
    native_quality_by_part: dict[str, dict[str, object]] | None = None,
) -> RigDocument:
    """Assemble the only public Rig type from immutable A/B/C facts."""

    validate_rig_geometry_cache(cache)
    validate_control_registry_plan(controls)
    validate_preset_library_plan(presets, controls)
    validate_primitive_candidate_set(
        candidates,
        cache,
        controls,
        presets,
        bindings,
        texture_page_ids=candidates.texture_page_ids,
    )
    validate_global_export_symbol_table(symbols, candidates, controls)
    validate_format_plan_set(
        formats,
        cache,
        controls,
        presets,
        capabilities,
        bindings,
        candidates,
        symbols,
    )
    texture_pages = _texture_page_records(texture_plan, texture_page_set)
    expected_page_ids = tuple(
        f"texture-page/page_{page.index}" for page in texture_page_set.pages
    )
    if candidates.texture_page_ids != expected_page_ids:
        raise _error("candidate universe differs from canonical texture pages")
    parts = _public_parts(
        cache,
        native_quality_by_part=native_quality_by_part or {},
        native_composite_mode_by_part=native_composite_mode_by_part or {},
    )
    meshes = _public_meshes(cache)
    payload = {
        "schema_version": RIG_DOCUMENT_SCHEMA_VERSION,
        "generator": {
            "name": RIG_DOCUMENT_GENERATOR_NAME,
            "algorithm_version": RIG_DOCUMENT_ALGORITHM_VERSION,
        },
        "input_fingerprint": cache.target_input_fingerprint,
        "input": {
            "tag_version": cache.target.tag_version,
            "coordinate_space": "layerdiff_canvas",
            "native_variant_set_sha256": cache.native_variant_set_sha256,
            "native_variant_eligibility_sha256": cache.native_variant_eligibility_sha256,
        },
        "canvas": {
            "width": cache.target.canvas_width,
            "height": cache.target.canvas_height,
            "origin": "top_left",
            "y_axis": "down",
        },
        "parts": parts,
        "joint_observations": [
            observation.to_dict()
            for observation in cache.joint_plan.joints.observations
        ],
        "joints": [
            resolution.to_dict() for resolution in cache.joint_plan.joints.resolutions
        ],
        "bones": [bone.to_dict() for bone in cache.bone_graph.bones],
        "meshes": meshes,
        "capabilities": [item.to_dict() for item in capabilities.capabilities],
        "control_specs": [control.to_dict() for control in controls.controls],
        "control_bindings": [binding.to_dict() for binding in bindings.bindings],
        "clips": [clip.to_dict() for clip in presets.clips],
        "expressions": [expression.to_dict() for expression in presets.expressions],
        "runtime_application": presets.runtime_application.to_dict(),
        "format_plans": _with_plan_digest(formats),
        "primitive_candidates": _with_plan_digest(candidates),
        "export_symbols": _with_table_digest(symbols),
        "texture_pages": texture_pages,
        "diagnostics": [diagnostic.to_dict() for diagnostic in cache.diagnostics],
        "provenance": {
            "rig_geometry_cache_sha256": cache.cache_sha256,
            "stage_a_joint_plan_sha256": cache.joint_plan.plan_sha256,
            "bone_graph_plan_sha256": cache.bone_graph.plan_sha256,
            "skinning_plan_sha256": cache.skinning_plan.plan_sha256,
            "component_draw_order_plan_sha256": cache.component_draw_order.plan_sha256,
            "capability_plan_sha256": capabilities.plan_sha256,
            "control_registry_sha256": controls.registry_sha256,
            "preset_library_sha256": presets.plan_sha256,
            "control_binding_plan_sha256": bindings.plan_sha256,
            "format_plan_set_sha256": formats.plan_sha256,
            "primitive_candidate_set_sha256": candidates.plan_sha256,
            "global_export_symbol_table_sha256": symbols.table_sha256,
            "texture_page_plan_sha256": texture_plan.plan_sha256,
            "canonical_texture_page_set_sha256": texture_page_set.set_sha256,
            "degradation_state": cache.degradation_state,
            "degradation_codes": list(cache.degradation_codes),
        },
    }
    return validate_rig_document_payload(payload)


def rig_document_bytes(rig: RigDocument) -> bytes:
    return validate_rig_document(rig)._canonical_json


def load_rig_document(path: Path) -> RigDocument:
    payload = path.read_bytes()
    decoded = _decode_canonical(payload)
    rig = validate_rig_document_payload(decoded)
    if rig._canonical_json != payload:
        raise _error("loaded RigDocument changed during validation")
    return rig


__all__ = [
    "RIG_DOCUMENT_ALGORITHM_VERSION",
    "RIG_DOCUMENT_GENERATOR_NAME",
    "RIG_DOCUMENT_SCHEMA_VERSION",
    "RigDocument",
    "RigDocumentError",
    "build_rig_document",
    "load_rig_document",
    "rig_document_bytes",
    "validate_rig_document",
    "validate_rig_document_payload",
]
