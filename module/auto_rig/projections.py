from __future__ import annotations

import json
from dataclasses import dataclass

from .jcs import JcsContractError, jcs_bytes, jcs_sha256
from .rig_document import RigDocument, validate_rig_document

MOTION_MANIFEST_PROJECTOR_VERSION = "motion-manifest-projector-v1"
RIG_REPORT_PROJECTOR_VERSION = "rig-report-projector-v1"


class ProjectionError(ValueError):
    """Raised when a public projection differs from its RigDocument source."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _error(message: str) -> ProjectionError:
    return ProjectionError("input_contract_mismatch", message)


def _canonical_payload(payload: object, *, projection: str) -> bytes:
    if type(payload) is not dict:
        raise _error(f"{projection} projection root must be an object")
    try:
        return jcs_bytes(payload)
    except JcsContractError as exc:
        raise _error(f"{projection} projection is outside the JCS domain") from exc


@dataclass(frozen=True, slots=True)
class MotionManifestProjection:
    _canonical_json: bytes

    @classmethod
    def from_payload(cls, payload: object) -> "MotionManifestProjection":
        return cls(_canonical_payload(payload, projection="motion manifest"))

    def to_dict(self) -> dict[str, object]:
        value = json.loads(self._canonical_json.decode("utf-8"))
        assert isinstance(value, dict)
        return value

    @property
    def projector_version(self) -> str:
        return str(self.to_dict()["projector_version"])

    @property
    def projection_sha256(self) -> str:
        return jcs_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class RigReportProjection:
    _canonical_json: bytes

    @classmethod
    def from_payload(cls, payload: object) -> "RigReportProjection":
        return cls(_canonical_payload(payload, projection="rig report"))

    def to_dict(self) -> dict[str, object]:
        value = json.loads(self._canonical_json.decode("utf-8"))
        assert isinstance(value, dict)
        return value

    @property
    def projector_version(self) -> str:
        return str(self.to_dict()["projector_version"])

    @property
    def projection_sha256(self) -> str:
        return jcs_sha256(self.to_dict())


def _preset_descriptors(rig_payload: dict[str, object]):
    result = {}
    for clip in rig_payload["clips"]:
        result[clip["preset_id"]] = {
            "kind": "motion",
            "internal_id": clip["clip_id"],
            "duration_seconds": clip["duration_frames"] / clip["sample_rate_hz"],
            "loop": clip["loop"],
            "control_ids": sorted(
                curve["control_id"] for curve in clip["control_curves"]
            ),
            "template_sha256": clip["template_sha256"],
        }
    for expression in rig_payload["expressions"]:
        result[expression["preset_id"]] = {
            "kind": "expression",
            "internal_id": expression["expression_id"],
            "duration_seconds": None,
            "loop": True,
            "control_ids": sorted(
                value["control_id"] for value in expression["values"]
            ),
            "template_sha256": expression["template_sha256"],
        }
    return result


def _artifact_path(
    format_id: str,
    descriptor_kind: str,
    export_name: str,
) -> str:
    if format_id == "spine_4_2":
        return f"spine/skeleton.json#animations/{export_name}"
    if descriptor_kind == "motion":
        return f"live2d/motions/{export_name}.motion3.json"
    return f"live2d/expressions/{export_name}.exp3.json"


def _motion_payload(rig: RigDocument) -> dict[str, object]:
    validate_rig_document(rig)
    payload = rig.to_dict()
    format_section = payload["format_plans"]
    profile = format_section["profile"]
    required_presets = set(profile["required_preset_ids"])
    descriptors = _preset_descriptors(payload)
    capabilities = {
        capability["preset_id"]: capability for capability in payload["capabilities"]
    }
    decisions_by_format = {
        set_plan["format_id"]: {
            decision["preset_id"]: decision for decision in set_plan["decisions"]
        }
        for set_plan in format_section["preset_set_plans"]
    }
    presets = []
    for preset_id in sorted(descriptors):
        descriptor = descriptors[preset_id]
        format_records = {}
        supported_formats = []
        for format_id in sorted(decisions_by_format):
            decision = decisions_by_format[format_id][preset_id]
            supported = decision["status"] == "supported"
            artifact = (
                _artifact_path(
                    format_id,
                    descriptor["kind"],
                    decision["artifact_export_name"],
                )
                if supported
                else None
            )
            if supported:
                supported_formats.append(format_id)
            format_records[format_id] = {
                "status": decision["status"],
                "reason": decision["reason"],
                "incompatible_with": decision["incompatible_with"],
                "artifact": artifact,
            }
        presets.append(
            {
                "preset_id": preset_id,
                "kind": descriptor["kind"],
                "internal_id": descriptor["internal_id"],
                "required": preset_id in required_presets,
                "quality_tier": capabilities[preset_id]["quality_tier"],
                "duration_seconds": descriptor["duration_seconds"],
                "loop": descriptor["loop"],
                "control_ids": descriptor["control_ids"],
                "template_sha256": descriptor["template_sha256"],
                "supported_formats": supported_formats,
                "formats": format_records,
            }
        )
    motion_semantics_payload = {
        "control_specs": payload["control_specs"],
        "control_bindings": payload["control_bindings"],
        "clips": payload["clips"],
        "expressions": payload["expressions"],
        "capabilities": payload["capabilities"],
        "format_plans": payload["format_plans"],
        "runtime_application": payload["runtime_application"],
        "global_symbol_table_sha256": payload["export_symbols"]["table_sha256"],
    }
    return {
        "schema_version": 1,
        "projector_version": MOTION_MANIFEST_PROJECTOR_VERSION,
        "rig_json_sha256": rig.document_sha256,
        "motion_semantics_sha256": jcs_sha256(motion_semantics_payload),
        "global_symbol_table_sha256": payload["export_symbols"]["table_sha256"],
        "profile": profile["profile_id"],
        "required_formats": profile["required_formats"],
        "optional_preset_parity": profile["optional_preset_parity"],
        "default_clip": "idle",
        "default_expression": None,
        "runtime_application": payload["runtime_application"],
        "presets": presets,
    }


def project_motion_manifest(rig: RigDocument) -> MotionManifestProjection:
    return MotionManifestProjection.from_payload(_motion_payload(rig))


def validate_motion_manifest_projection(
    projection: MotionManifestProjection,
    rig: RigDocument,
) -> MotionManifestProjection:
    if not isinstance(projection, MotionManifestProjection):
        raise _error("motion projection has the wrong type")
    expected = project_motion_manifest(rig)
    if projection != expected:
        raise _error("motion_manifest.json differs from its RigDocument projection")
    return projection


def motion_manifest_bytes(projection: MotionManifestProjection) -> bytes:
    if not isinstance(projection, MotionManifestProjection):
        raise _error("motion projection has the wrong type")
    return projection._canonical_json


def _report_payload(rig: RigDocument) -> dict[str, object]:
    validate_rig_document(rig)
    payload = rig.to_dict()
    format_section = payload["format_plans"]
    format_status = {}
    for model in format_section["model_plans"]:
        set_plan = next(
            item
            for item in format_section["preset_set_plans"]
            if item["format_id"] == model["format_id"]
        )
        format_status[model["format_id"]] = {
            "model_status": model["status"],
            "model_reason_codes": model["reason_codes"],
            "supported_presets": sorted(
                decision["preset_id"]
                for decision in set_plan["decisions"]
                if decision["status"] == "supported"
            ),
            "omitted_presets": {
                decision["preset_id"]: decision["reason"]
                for decision in set_plan["decisions"]
                if decision["status"] == "omitted"
            },
        }
    return {
        "schema_version": 1,
        "projector_version": RIG_REPORT_PROJECTOR_VERSION,
        "rig_json_sha256": rig.document_sha256,
        "input_fingerprint": payload["input_fingerprint"],
        "profile": format_section["profile"]["profile_id"],
        "degradation_state": payload["provenance"]["degradation_state"],
        "degradation_codes": payload["provenance"]["degradation_codes"],
        "counts": {
            "parts": len(payload["parts"]),
            "joints": len(payload["joints"]),
            "bones": len(payload["bones"]),
            "meshes": len(payload["meshes"]),
            "texture_pages": len(payload["texture_pages"]),
        },
        "capabilities": [
            {
                "preset_id": capability["preset_id"],
                "status": capability["status"],
                "quality_tier": capability["quality_tier"],
                "reason_codes": capability["reason_codes"],
            }
            for capability in payload["capabilities"]
        ],
        "format_status": format_status,
        "diagnostics": payload["diagnostics"],
    }


def project_rig_report(rig: RigDocument) -> RigReportProjection:
    return RigReportProjection.from_payload(_report_payload(rig))


def validate_rig_report_projection(
    projection: RigReportProjection,
    rig: RigDocument,
) -> RigReportProjection:
    if not isinstance(projection, RigReportProjection):
        raise _error("report projection has the wrong type")
    expected = project_rig_report(rig)
    if projection != expected:
        raise _error("rig/report.json differs from its RigDocument projection")
    return projection


def rig_report_bytes(projection: RigReportProjection) -> bytes:
    if not isinstance(projection, RigReportProjection):
        raise _error("report projection has the wrong type")
    return projection._canonical_json


__all__ = [
    "MOTION_MANIFEST_PROJECTOR_VERSION",
    "RIG_REPORT_PROJECTOR_VERSION",
    "MotionManifestProjection",
    "ProjectionError",
    "RigReportProjection",
    "motion_manifest_bytes",
    "project_motion_manifest",
    "project_rig_report",
    "rig_report_bytes",
    "validate_motion_manifest_projection",
    "validate_rig_report_projection",
]
