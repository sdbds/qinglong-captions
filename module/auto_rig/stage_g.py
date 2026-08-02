from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Mapping

from .artifacts import ArtifactContractError, FileDigest
from .export.live2d.cubism_renderer import (
    LIVE2D_E0_VALIDATOR_PROTOCOL_DIGEST,
)
from .export.live2d.release_validator import (
    LIVE2D_RELEASE_REPORT_VERSION,
    LIVE2D_RELEASE_VALIDATOR_VERSION,
)
from .export.live2d.validator import (
    LIVE2D_MOC3_VALIDATOR_VERSION,
    LIVE2D_STRUCTURE_REPORT_VERSION,
)
from .export.spine.validator import SPINE_BUNDLE_VALIDATOR_VERSION
from .jcs import JcsContractError, jcs_bytes, jcs_sha256
from .manifests import StageManifest
from .projections import motion_manifest_bytes, project_motion_manifest
from .rig_document import RigDocument, load_rig_document
from .stage_d import SPINE_EXPORT_REPORT_VERSION, STAGE_D_ALGORITHM_VERSION
from .stage_e import LIVE2D_EXPORT_REPORT_VERSION, STAGE_E_ALGORITHM_VERSION
from .terminal import (
    FORMAL_REQUIRED_FORMATS,
    FormatValidation,
    TerminalFinalizationError,
    TerminalFinalizationResult,
    TextureRuntimeContract,
    _validate_preterminal_graph,
    finalize_success,
    invalidate_terminal,
)

STAGE_G_INTEGRATION_VERSION = "stage-g-terminal-integration-v1"
LIVE2D_VALIDATOR_FINGERPRINT_VERSION = (
    "live2d-validator-fingerprint-v1"
)
SPINE_REPORT_PATH = "rig/spine/export_report.json"
LIVE2D_REPORT_PATH = "rig/live2d/export_report.json"
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class StageGError(RuntimeError):
    """Raised when current exporter evidence cannot form a terminal release."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _error(code: str, message: str) -> StageGError:
    return StageGError(code, message)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _error("invalid_export_report", f"{field} must be an object")
    return value


def _list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise _error("invalid_export_report", f"{field} must be a list")
    return value


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise _error(
            "invalid_export_report",
            f"{field} must be a lowercase sha256:<hex> digest",
        )
    return value


def _finite(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise _error("invalid_export_report", f"{field} must be finite")
    return float(value)


def _load_canonical_object(path: Path) -> dict[str, object]:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise StageGError(
                    "invalid_export_report",
                    f"duplicate JSON key in {path.name}: {key}",
                )
            result[key] = value
        return result

    try:
        payload = path.read_bytes()
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageGError(
            "invalid_export_report",
            f"unable to read canonical report {path.name}",
        ) from exc
    if type(decoded) is not dict:
        raise StageGError(
            "invalid_export_report",
            f"report {path.name} must contain a JSON object",
        )
    try:
        if jcs_bytes(decoded) != payload:
            raise StageGError(
                "invalid_export_report",
                f"report {path.name} is not canonical JCS",
            )
    except JcsContractError as exc:
        raise StageGError(
            "invalid_export_report",
            f"report {path.name} is outside the JCS value domain",
        ) from exc
    return decoded


def _report_digest(payload: Mapping[str, object]) -> str:
    return jcs_sha256(
        {key: value for key, value in payload.items() if key != "report_sha256"}
    )


def _require_report_digest(
    payload: Mapping[str, object],
    *,
    field: str,
) -> None:
    if payload.get("report_sha256") != _report_digest(payload):
        raise _error(
            "invalid_export_report_digest",
            f"{field} digest is invalid",
        )


def _validate_live2d_release_record(report: Mapping[str, object]) -> None:
    release = report.get("release_validation")
    if not isinstance(release, Mapping) or release.get("status") != "passed":
        raise StageGError(
            "live2d_release_validation_missing",
            "formal completion requires per-item official Cubism Core/SDK evidence",
        )
    evidence = release.get("report")
    structure = report.get("structure_validation")
    if not isinstance(evidence, Mapping) or not isinstance(structure, Mapping):
        raise StageGError(
            "live2d_release_validation_invalid",
            "Live2D release or structure evidence is malformed",
        )
    required = {
        "schema_version",
        "validator_version",
        "structure_report_sha256",
        "core_sha256",
        "core_version",
        "renderer_sha256",
        "renderer_protocol_digest",
        "moc_sha256",
        "core_consistency",
        "default_rest_maximum_residual",
        "baseline_rgba_sha256",
        "parameter_evidence",
        "motion_evidence",
        "expression_evidence",
        "report_sha256",
    }
    if set(evidence) != required:
        raise StageGError(
            "live2d_release_validation_invalid",
            "Live2D release evidence has an unexpected schema",
        )
    _require_report_digest(evidence, field="Live2D release report")
    _digest(evidence.get("core_sha256"), field="release.core_sha256")
    _digest(evidence.get("renderer_sha256"), field="release.renderer_sha256")
    _digest(
        evidence.get("baseline_rgba_sha256"),
        field="release.baseline_rgba_sha256",
    )
    if not isinstance(evidence.get("core_version"), str) or not evidence.get(
        "core_version"
    ):
        raise _error(
            "live2d_release_validation_invalid",
            "Live2D release evidence has no Core version",
        )
    _finite(
        evidence.get("default_rest_maximum_residual"),
        field="release.default_rest_maximum_residual",
    )
    for field in (
        "parameter_evidence",
        "motion_evidence",
        "expression_evidence",
    ):
        if not isinstance(evidence.get(field), list):
            raise _error(
                "live2d_release_validation_invalid",
                f"Live2D release {field} must be a list",
            )
    if (
        evidence.get("schema_version") != LIVE2D_RELEASE_REPORT_VERSION
        or evidence.get("validator_version")
        != LIVE2D_RELEASE_VALIDATOR_VERSION
        or evidence.get("renderer_protocol_digest")
        != LIVE2D_E0_VALIDATOR_PROTOCOL_DIGEST
        or evidence.get("core_consistency") is not True
        or evidence.get("structure_report_sha256")
        != structure.get("report_sha256")
        or evidence.get("moc_sha256") != structure.get("moc_sha256")
    ):
        raise StageGError(
            "live2d_release_validation_invalid",
            "Live2D release evidence failed its signed runtime contract",
        )


def _manifest_artifacts_without_report(
    manifest: StageManifest,
    report_path: str,
) -> tuple[FileDigest, ...]:
    records = tuple(
        record
        for record in manifest.output_file_sha256
        if record.path != report_path
    )
    if len(records) + 1 != len(manifest.output_file_sha256):
        raise _error(
            "invalid_export_report",
            f"stage {manifest.stage_name} does not own exactly one report",
        )
    return records


def _validate_report_artifacts(
    report: Mapping[str, object],
    *,
    manifest: StageManifest,
    report_path: str,
) -> None:
    raw = _list(report.get("artifacts"), field="artifacts")
    try:
        records = tuple(FileDigest.from_dict(record) for record in raw)
    except (ArtifactContractError, TypeError) as exc:
        raise _error(
            "invalid_export_report",
            f"stage {manifest.stage_name} artifact inventory is malformed",
        ) from exc
    if len({record.path for record in records}) != len(records):
        raise _error(
            "invalid_export_report",
            f"stage {manifest.stage_name} artifact inventory has duplicate paths",
        )
    expected = _manifest_artifacts_without_report(manifest, report_path)
    if tuple(sorted(records, key=lambda record: record.path)) != expected:
        raise _error(
            "export_report_inventory_mismatch",
            f"stage {manifest.stage_name} report artifacts differ from its manifest",
        )


def _validate_common_report(
    report: Mapping[str, object],
    *,
    manifest: StageManifest,
    report_path: str,
    schema_version: str,
    algorithm_version: str,
    format_id: str,
    rig: RigDocument,
    rig_payload: Mapping[str, object],
) -> None:
    _require_report_digest(report, field=f"stage {manifest.stage_name} export report")
    format_plans = _mapping(
        rig_payload.get("format_plans"), field="rig.format_plans"
    )
    profile = _mapping(format_plans.get("profile"), field="rig.profile")
    export_symbols = _mapping(
        rig_payload.get("export_symbols"), field="rig.export_symbols"
    )
    if (
        report.get("schema_version") != schema_version
        or report.get("algorithm_version") != algorithm_version
        or report.get("status") != manifest.status
        or report.get("format_id") != format_id
        or report.get("profile_id") != profile.get("profile_id")
        or report.get("rig_json_sha256") != rig.document_sha256
        or report.get("format_plan_set_sha256")
        != format_plans.get("plan_sha256")
        or report.get("global_symbol_table_sha256")
        != export_symbols.get("table_sha256")
    ):
        raise _error(
            "export_report_identity_mismatch",
            f"stage {manifest.stage_name} report differs from the committed Rig",
        )
    _validate_report_artifacts(
        report,
        manifest=manifest,
        report_path=report_path,
    )


def _validate_spine_report(
    report: Mapping[str, object],
    *,
    manifest: StageManifest,
    rig: RigDocument,
    rig_payload: Mapping[str, object],
) -> str:
    _validate_common_report(
        report,
        manifest=manifest,
        report_path=SPINE_REPORT_PATH,
        schema_version=SPINE_EXPORT_REPORT_VERSION,
        algorithm_version=STAGE_D_ALGORITHM_VERSION,
        format_id="spine_4_2",
        rig=rig,
        rig_payload=rig_payload,
    )
    validation = _mapping(
        report.get("validation"), field="spine.validation"
    )
    _require_report_digest(validation, field="Spine validation report")
    artifacts = {
        record.path: record
        for record in _manifest_artifacts_without_report(
            manifest, SPINE_REPORT_PATH
        )
    }
    skeleton = artifacts.get("rig/spine/skeleton.json")
    atlas = artifacts.get("rig/spine/skeleton.atlas")
    if (
        validation.get("schema_version") != SPINE_BUNDLE_VALIDATOR_VERSION
        or validation.get("validated") is not True
        or validation.get("rig_document_sha256") != rig.document_sha256
        or skeleton is None
        or atlas is None
        or validation.get("skeleton_sha256") != skeleton.sha256
        or validation.get("atlas_sha256") != atlas.sha256
        or _finite(
            validation.get("maximum_setup_residual_px"),
            field="spine.maximum_setup_residual_px",
        )
        > 0.1
    ):
        raise _error(
            "spine_validation_invalid",
            "Spine structural validation evidence is incomplete or inconsistent",
        )
    return _digest(
        validation.get("validator_fingerprint"),
        field="spine.validator_fingerprint",
    )


def _validate_live2d_report(
    report: Mapping[str, object],
    *,
    manifest: StageManifest,
    rig: RigDocument,
    rig_payload: Mapping[str, object],
) -> tuple[str, str]:
    _validate_common_report(
        report,
        manifest=manifest,
        report_path=LIVE2D_REPORT_PATH,
        schema_version=LIVE2D_EXPORT_REPORT_VERSION,
        algorithm_version=STAGE_E_ALGORITHM_VERSION,
        format_id="live2d_moc3_v4_00",
        rig=rig,
        rig_payload=rig_payload,
    )
    release = report.get("release_validation")
    if (
        report.get("validation_tier") != "release"
        or report.get("formal_release_eligible") is not True
        or not isinstance(release, Mapping)
        or release.get("status") != "passed"
    ):
        raise _error(
            "live2d_release_validation_missing",
            "formal completion requires per-item official Cubism Core/SDK evidence",
        )
    structure = _mapping(
        report.get("structure_validation"),
        field="live2d.structure_validation",
    )
    _require_report_digest(structure, field="Live2D structure report")
    artifacts = {
        record.path: record
        for record in _manifest_artifacts_without_report(
            manifest, LIVE2D_REPORT_PATH
        )
    }
    moc = artifacts.get("rig/live2d/model.moc3")
    if (
        structure.get("schema_version") != LIVE2D_STRUCTURE_REPORT_VERSION
        or structure.get("validator_version") != LIVE2D_MOC3_VALIDATOR_VERSION
        or structure.get("rig_document_sha256") != rig.document_sha256
        or moc is None
        or structure.get("moc_sha256") != moc.sha256
        or _finite(
            structure.get("maximum_default_position_residual"),
            field="live2d.maximum_default_position_residual",
        )
        > 0.1
        or _finite(
            structure.get("maximum_default_opacity_residual"),
            field="live2d.maximum_default_opacity_residual",
        )
        > (1 / 255)
    ):
        raise _error(
            "live2d_structure_validation_invalid",
            "Live2D structure evidence is incomplete or inconsistent",
        )
    _validate_live2d_release_record(report)
    evidence = _mapping(release.get("report"), field="live2d.release.report")
    loader_contract = _digest(
        evidence.get("renderer_protocol_digest"),
        field="live2d.renderer_protocol_digest",
    )
    validator_fingerprint = jcs_sha256(
        {
            "schema_version": LIVE2D_VALIDATOR_FINGERPRINT_VERSION,
            "structure_validator_version": structure["validator_version"],
            "release_validator_version": evidence["validator_version"],
            "core_sha256": evidence["core_sha256"],
            "core_version": evidence["core_version"],
            "renderer_sha256": evidence["renderer_sha256"],
            "renderer_protocol_digest": loader_contract,
        }
    )
    return validator_fingerprint, loader_contract


def _require_terminal_profile(
    rig_payload: Mapping[str, object],
) -> tuple[str, str]:
    format_plans = _mapping(
        rig_payload.get("format_plans"), field="rig.format_plans"
    )
    profile = _mapping(format_plans.get("profile"), field="rig.profile")
    required_formats = _list(
        profile.get("required_formats"), field="rig.profile.required_formats"
    )
    if (
        len(required_formats) != len(FORMAL_REQUIRED_FORMATS)
        or set(required_formats) != set(FORMAL_REQUIRED_FORMATS)
        or profile.get("terminal_delivery") is not True
    ):
        raise _error(
            "terminal_profile_mismatch",
            "Stage G success requires a terminal dual-runtime profile",
        )
    profile_id = profile.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        raise _error("terminal_profile_mismatch", "profile ID is missing")
    return profile_id, _digest(
        profile.get("profile_sha256"), field="profile.profile_sha256"
    )


def _validate_shared_textures(
    rig_payload: Mapping[str, object],
    *,
    manifests: Mapping[str, StageManifest],
) -> None:
    raw_pages = _list(rig_payload.get("texture_pages"), field="rig.texture_pages")
    c_outputs = {record.path: record for record in manifests["C"].output_file_sha256}
    d_outputs = {record.path: record for record in manifests["D"].output_file_sha256}
    e_outputs = {record.path: record for record in manifests["E"].output_file_sha256}
    if not raw_pages:
        raise _error("shared_texture_mismatch", "Rig contains no texture pages")
    for index, raw in enumerate(raw_pages):
        page = _mapping(raw, field=f"rig.texture_pages[{index}]")
        if page.get("index") != index:
            raise _error(
                "shared_texture_mismatch",
                "texture page indices must be contiguous from zero",
            )
        source_path = page.get("relative_path")
        if not isinstance(source_path, str):
            raise _error(
                "shared_texture_mismatch", "texture page path is missing"
            )
        expected_sha = page.get("encoded_png_sha256")
        copies = (
            c_outputs.get(source_path),
            d_outputs.get(f"rig/spine/textures/page_{index}.png"),
            e_outputs.get(f"rig/live2d/textures/page_{index}.png"),
        )
        if any(record is None for record in copies) or any(
            record.sha256 != expected_sha for record in copies if record is not None
        ):
            raise _error(
                "shared_texture_mismatch",
                f"canonical texture page {index} differs across C/D/E",
            )


def _load_release_inputs(
    root: Path,
    *,
    expected_stage_fingerprints: Mapping[str, str],
) -> tuple[
    Mapping[str, StageManifest],
    RigDocument,
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
]:
    try:
        manifests = _validate_preterminal_graph(
            root, expected_stage_fingerprints
        )
    except TerminalFinalizationError as exc:
        raise _error("preterminal_graph_invalid", str(exc)) from exc
    rig = load_rig_document(root / "rig/rig.json")
    rig_payload = rig.to_dict()
    c_outputs = {record.path: record for record in manifests["C"].output_file_sha256}
    rig_record = c_outputs.get("rig/rig.json")
    if rig_record is None or rig_record.sha256 != rig.document_sha256:
        raise _error(
            "rig_identity_mismatch",
            "committed RigDocument differs from Stage C",
        )
    projected_motion = motion_manifest_bytes(project_motion_manifest(rig))
    try:
        committed_motion = (root / "rig/motion_manifest.json").read_bytes()
    except OSError as exc:
        raise _error(
            "motion_manifest_mismatch", "motion manifest is unavailable"
        ) from exc
    if committed_motion != projected_motion:
        raise _error(
            "motion_manifest_mismatch",
            "motion manifest differs from the committed Rig projection",
        )
    spine_report = _load_canonical_object(root / SPINE_REPORT_PATH)
    live2d_report = _load_canonical_object(root / LIVE2D_REPORT_PATH)
    return manifests, rig, rig_payload, spine_report, live2d_report


def execute_stage_g_success(
    item_root: str | Path,
    *,
    config_fingerprint: str,
    expected_stage_fingerprints: Mapping[str, str],
) -> TerminalFinalizationResult:
    """Validate exporter evidence and publish the formal terminal state."""

    root = Path(item_root).resolve(strict=True)
    invalidate_terminal(root)
    manifests, rig, rig_payload, spine_report, live2d_report = (
        _load_release_inputs(
            root,
            expected_stage_fingerprints=expected_stage_fingerprints,
        )
    )
    profile_id, profile_sha256 = _require_terminal_profile(rig_payload)
    spine_validator = _validate_spine_report(
        spine_report,
        manifest=manifests["D"],
        rig=rig,
        rig_payload=rig_payload,
    )
    live2d_validator, loader_contract = _validate_live2d_report(
        live2d_report,
        manifest=manifests["E"],
        rig=rig,
        rig_payload=rig_payload,
    )
    _validate_shared_textures(rig_payload, manifests=manifests)
    input_section = _mapping(rig_payload.get("input"), field="rig.input")
    symbols = _mapping(
        rig_payload.get("export_symbols"), field="rig.export_symbols"
    )
    texture_contract = TextureRuntimeContract(
        schema_version="shared-texture-v1",
        canonical_uv_space="page_top_left_v_down",
        alpha_mode="straight",
        color_space="srgb_bytes",
        spine_uv_adapter="spine-4.2-uv-v1",
        spine_atlas_pma=False,
        live2d_uv_adapter="cubism-v4.00-uv-v1",
        live2d_runtime_loader_contract=loader_contract,
    )
    formats = (
        FormatValidation(
            format_id="spine_4_2",
            stage_name="D",
            status="validated",
            files=tuple(
                record.path for record in manifests["D"].output_file_sha256
            ),
            validator_fingerprint=spine_validator,
        ),
        FormatValidation(
            format_id="live2d_moc3_v4_00",
            stage_name="E",
            status="validated",
            files=tuple(
                record.path for record in manifests["E"].output_file_sha256
            ),
            validator_fingerprint=live2d_validator,
        ),
    )
    try:
        return finalize_success(
            root,
            input_fingerprint=rig.input_fingerprint,
            native_variant_set_sha256=_digest(
                input_section.get("native_variant_set_sha256"),
                field="rig.input.native_variant_set_sha256",
            ),
            native_variant_eligibility_sha256=_digest(
                input_section.get("native_variant_eligibility_sha256"),
                field="rig.input.native_variant_eligibility_sha256",
            ),
            config_fingerprint=config_fingerprint,
            profile=profile_id,
            profile_fingerprint=profile_sha256,
            rig_overrides_sha256=manifests["C"].rig_overrides_sha256,
            expected_stage_fingerprints=expected_stage_fingerprints,
            formats=formats,
            validation_tier="release",
            motion_runtime_contract_sha256=jcs_sha256(
                rig_payload["runtime_application"]
            ),
            global_symbol_table_sha256=_digest(
                symbols.get("table_sha256"),
                field="rig.export_symbols.table_sha256",
            ),
            texture_contract=texture_contract,
        )
    except TerminalFinalizationError as exc:
        raise _error("terminal_finalization_failed", str(exc)) from exc


__all__ = [
    "STAGE_G_INTEGRATION_VERSION",
    "StageGError",
    "execute_stage_g_success",
]
