from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifacts import (
    ArtifactContractError,
    FileDigest,
    atomic_write_json,
    canonical_json_bytes,
    canonical_json_sha256,
    describe_file,
    normalize_relative_path,
    sha256_file,
)
from .manifests import (
    StageManifest,
    StageManifestError,
    build_stage_manifest,
    manifest_relative_path,
    read_stage_manifest,
    write_stage_manifest,
)
from .stage_graph import StageGraphValidator


EXPORT_MANIFEST_PATH = "rig/export_manifest.json"
ERROR_RECORD_PATH = "rig/error.json"
TERMINAL_FINALIZER_VERSION = "terminal-finalizer-v1"
FORMAL_REQUIRED_FORMATS = ("spine_4_2", "live2d_moc3_v4_00")
_FORMAT_STAGE = {"spine_4_2": "D", "live2d_moc3_v4_00": "E"}
_RELEASE_STAGES = ("A", "B", "C", "D", "E")
_STAGE_ORDER = {stage: index for index, stage in enumerate((*_RELEASE_STAGES, "G"))}
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class TerminalFinalizationError(RuntimeError):
    """Raised when an item cannot enter a valid terminal state."""


def _require_digest(value: Any, *, field: str) -> str:
    normalized = str(value)
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise TerminalFinalizationError(f"{field} must be a lowercase sha256:<hex> digest")
    return normalized


def _require_nonempty(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TerminalFinalizationError(f"{field} must be a non-empty string")
    return value.strip()


def _item_path(item_root: str | Path, relative_path: str) -> Path:
    return Path(item_root) / Path(*relative_path.split("/"))


@dataclass(frozen=True)
class FormatValidation:
    format_id: str
    stage_name: str
    status: str
    files: tuple[str, ...]
    validator_fingerprint: str

    def __post_init__(self) -> None:
        format_id = _require_nonempty(self.format_id, field="format_id")
        stage_name = _require_nonempty(self.stage_name, field="stage_name")
        status = _require_nonempty(self.status, field="format status")
        try:
            files = tuple(sorted(normalize_relative_path(path) for path in self.files))
        except (ArtifactContractError, TypeError) as exc:
            raise TerminalFinalizationError(f"invalid format file path: {exc}") from exc
        if not files:
            raise TerminalFinalizationError(f"format {format_id} must declare at least one file")
        if len(files) != len(set(files)):
            raise TerminalFinalizationError(f"format {format_id} contains duplicate file paths")
        validator_fingerprint = _require_digest(
            self.validator_fingerprint,
            field=f"{format_id} validator_fingerprint",
        )
        object.__setattr__(self, "format_id", format_id)
        object.__setattr__(self, "stage_name", stage_name)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "validator_fingerprint", validator_fingerprint)


@dataclass(frozen=True)
class StageFailureRecord:
    stage_name: str
    record_path: str
    diagnostics: tuple[Mapping[str, Any], ...]
    retryable: bool

    def __post_init__(self) -> None:
        if self.stage_name not in _RELEASE_STAGES:
            raise TerminalFinalizationError(f"failure stage must be one of {_RELEASE_STAGES}")
        try:
            record_path = normalize_relative_path(self.record_path)
        except ArtifactContractError as exc:
            raise TerminalFinalizationError(f"invalid failure record path: {exc}") from exc
        expected_prefix = f"rig/cache/{self.stage_name}/"
        if not record_path.startswith(expected_prefix) or record_path == manifest_relative_path(self.stage_name):
            raise TerminalFinalizationError(
                f"failure record for stage {self.stage_name} must be a stage-local cache file"
            )
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(record, Mapping) for record in diagnostics):
            raise TerminalFinalizationError("failure diagnostics must be JSON objects")
        try:
            canonical_json_bytes([dict(record) for record in diagnostics])
        except (TypeError, ValueError) as exc:
            raise TerminalFinalizationError(f"failure diagnostics are not canonical JSON: {exc}") from exc
        if not isinstance(self.retryable, bool):
            raise TerminalFinalizationError("failure retryable must be a boolean")
        object.__setattr__(self, "record_path", record_path)
        object.__setattr__(self, "diagnostics", diagnostics)


@dataclass(frozen=True)
class TerminalFinalizationResult:
    terminal_path: Path
    g_manifest: StageManifest
    payload: Mapping[str, Any]


def _validate_preterminal_graph(
    item_root: Path,
    expected_stage_fingerprints: Mapping[str, str],
) -> dict[str, StageManifest]:
    missing_expected = set(_RELEASE_STAGES) - set(expected_stage_fingerprints)
    extra_expected = set(expected_stage_fingerprints) - set(_RELEASE_STAGES)
    if missing_expected or extra_expected:
        raise TerminalFinalizationError(
            "expected_stage_fingerprints must contain exactly A, B, C, D, and E"
        )

    validator = StageGraphValidator(item_root)
    results = (
        validator.validate(target_stage="D", expected_fingerprints=expected_stage_fingerprints),
        validator.validate(target_stage="E", expected_fingerprints=expected_stage_fingerprints),
    )
    issues = {
        (issue.code, issue.stage_name, issue.path, issue.detail)
        for result in results
        for issue in result.issues
    }
    if issues:
        summary = ", ".join(f"{stage}:{code}" for code, stage, _, _ in sorted(issues))
        raise TerminalFinalizationError(f"preterminal stage graph is not reusable: {summary}")

    manifests: dict[str, StageManifest] = {}
    for result in results:
        manifests.update(dict(result.manifests))
    if set(manifests) != set(_RELEASE_STAGES):
        raise TerminalFinalizationError("preterminal stage graph did not yield all A-E manifests")

    owners: dict[str, str] = {}
    for stage in _RELEASE_STAGES:
        for output in manifests[stage].output_file_sha256:
            previous = owners.setdefault(output.path, stage)
            if previous != stage:
                raise TerminalFinalizationError(
                    f"preterminal stage graph has output ownership conflict: {output.path} ({previous}, {stage})"
                )
    return manifests


def _normalize_formats(formats: Iterable[FormatValidation]) -> dict[str, FormatValidation]:
    normalized = tuple(formats)
    if any(not isinstance(item, FormatValidation) for item in normalized):
        raise TerminalFinalizationError("formats must contain FormatValidation records")
    by_id = {item.format_id: item for item in normalized}
    if len(by_id) != len(normalized) or set(by_id) != set(FORMAL_REQUIRED_FORMATS):
        raise TerminalFinalizationError(
            f"formal delivery requires exactly formats {list(FORMAL_REQUIRED_FORMATS)}"
        )
    for format_id, item in by_id.items():
        expected_stage = _FORMAT_STAGE[format_id]
        if item.stage_name != expected_stage:
            raise TerminalFinalizationError(
                f"format {format_id} must be produced by stage {expected_stage}"
            )
        if item.status != "validated":
            raise TerminalFinalizationError(f"format {format_id} is not validated")
    return by_id


def _artifact_set_digest(files: Iterable[FileDigest]) -> str:
    records = sorted(files, key=lambda item: item.path)
    return canonical_json_sha256([record.to_dict() for record in records])


def _terminal_config_fingerprint(payload: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {
            "finalizer_version": payload["finalizer_version"],
            "input_fingerprint": payload["input_fingerprint"],
            "profile": payload["profile"],
            "profile_fingerprint": payload["profile_fingerprint"],
            "required_formats": payload["required_formats"],
            "upstream_stage_manifests": payload["upstream_stage_manifests"],
            "upstream_artifact_sets": payload["upstream_artifact_sets"],
            "motion_manifest_sha256": payload["motion_manifest_sha256"],
            "global_symbol_table_sha256": payload["global_symbol_table_sha256"],
            "validation": payload["validation"],
            "formats": payload["formats"],
            "status": payload["status"],
        }
    )


def finalize_success(
    item_root: str | Path,
    *,
    input_fingerprint: str,
    profile: str,
    profile_fingerprint: str,
    rig_overrides_sha256: str,
    expected_stage_fingerprints: Mapping[str, str],
    formats: Iterable[FormatValidation],
    validation_tier: str,
    global_symbol_table_sha256: str,
) -> TerminalFinalizationResult:
    root = Path(item_root)
    input_digest = _require_digest(input_fingerprint, field="input_fingerprint")
    profile_name = _require_nonempty(profile, field="profile")
    profile_digest = _require_digest(profile_fingerprint, field="profile_fingerprint")
    overrides_digest = _require_digest(rig_overrides_sha256, field="rig_overrides_sha256")
    symbols_digest = _require_digest(
        global_symbol_table_sha256,
        field="global_symbol_table_sha256",
    )
    tier = _require_nonempty(validation_tier, field="validation_tier")
    if tier != "release":
        raise TerminalFinalizationError("formal dual-runtime completion requires validation_tier='release'")

    manifests = _validate_preterminal_graph(root, expected_stage_fingerprints)
    format_map = _normalize_formats(formats)
    format_payloads: dict[str, dict[str, Any]] = {}
    artifact_sets: dict[str, str] = {}
    for format_id in FORMAL_REQUIRED_FORMATS:
        validation = format_map[format_id]
        stage_outputs = {record.path: record for record in manifests[validation.stage_name].output_file_sha256}
        if set(validation.files) != set(stage_outputs):
            raise TerminalFinalizationError(
                f"format {format_id} files must exactly match stage {validation.stage_name} outputs"
            )
        records = tuple(stage_outputs[path] for path in sorted(validation.files))
        format_payloads[format_id] = {
            "status": "validated",
            "files": [record.to_dict() for record in records],
        }
        artifact_sets[format_id] = _artifact_set_digest(records)

    c_outputs = {record.path: record for record in manifests["C"].output_file_sha256}
    motion_path = "rig/motion_manifest.json"
    if motion_path not in c_outputs:
        raise TerminalFinalizationError("stage C does not own rig/motion_manifest.json")
    canonical_textures = tuple(
        record
        for path, record in sorted(c_outputs.items())
        if path.startswith("rig/shared/textures/") and path.endswith(".png")
    )
    if not canonical_textures:
        raise TerminalFinalizationError("stage C does not own any canonical texture pages")

    upstream_manifests = {
        stage: sha256_file(_item_path(root, manifest_relative_path(stage)))
        for stage in ("C", "D", "E")
    }
    validation_payload = {
        "tier": tier,
        "spine_validator_fingerprint": format_map["spine_4_2"].validator_fingerprint,
        "live2d_validator_fingerprint": format_map["live2d_moc3_v4_00"].validator_fingerprint,
    }
    export_payload: dict[str, Any] = {
        "schema_version": 1,
        "producer_stage": "G",
        "finalizer_version": TERMINAL_FINALIZER_VERSION,
        "input_fingerprint": input_digest,
        "profile": profile_name,
        "profile_fingerprint": profile_digest,
        "required_formats": list(FORMAL_REQUIRED_FORMATS),
        "upstream_stage_manifests": upstream_manifests,
        "upstream_artifact_sets": {
            "canonical_textures": _artifact_set_digest(canonical_textures),
            **artifact_sets,
        },
        "motion_manifest_sha256": c_outputs[motion_path].sha256,
        "global_symbol_table_sha256": symbols_digest,
        "validation": validation_payload,
        "formats": format_payloads,
        "status": "completed",
    }

    g_marker = _item_path(root, manifest_relative_path("G"))
    g_marker.unlink(missing_ok=True)
    _item_path(root, ERROR_RECORD_PATH).unlink(missing_ok=True)
    export_path = _item_path(root, EXPORT_MANIFEST_PATH)
    atomic_write_json(export_path, export_payload)

    marker_inputs = tuple(
        describe_file(root, manifest_relative_path(stage)) for stage in ("C", "D", "E")
    )
    g_manifest = build_stage_manifest(
        root,
        stage_name="G",
        stage_schema_version=1,
        algorithm_version=TERMINAL_FINALIZER_VERSION,
        upstream_manifests=upstream_manifests,
        input_file_sha256=marker_inputs,
        relevant_config_fingerprint=_terminal_config_fingerprint(export_payload),
        rig_overrides_sha256=overrides_digest,
        output_paths=[EXPORT_MANIFEST_PATH],
        status="completed",
    )
    write_stage_manifest(root, g_manifest)
    return TerminalFinalizationResult(
        terminal_path=export_path,
        g_manifest=g_manifest,
        payload=export_payload,
    )


def _completed_upstream_manifests(
    root: Path,
    completed_stage_names: Iterable[str],
) -> tuple[dict[str, str], tuple[FileDigest, ...]]:
    names = tuple(sorted(set(completed_stage_names), key=lambda stage: _STAGE_ORDER.get(stage, 999)))
    if any(stage not in _RELEASE_STAGES for stage in names):
        raise TerminalFinalizationError("completed_stage_names may contain only A-E")
    upstream: dict[str, str] = {}
    inputs: list[FileDigest] = []
    for stage in names:
        try:
            manifest = read_stage_manifest(root, stage)
        except StageManifestError as exc:
            raise TerminalFinalizationError(f"completed stage {stage} has no valid manifest: {exc}") from exc
        if manifest.status != "completed":
            raise TerminalFinalizationError(f"completed stage {stage} manifest is not completed")
        for expected in manifest.output_file_sha256:
            try:
                actual = describe_file(root, expected.path)
            except ArtifactContractError as exc:
                raise TerminalFinalizationError(f"completed stage {stage} output is invalid: {exc}") from exc
            if actual != expected:
                raise TerminalFinalizationError(f"completed stage {stage} output changed: {expected.path}")
        relative = manifest_relative_path(stage)
        upstream[stage] = sha256_file(_item_path(root, relative))
        inputs.append(describe_file(root, relative))
    return upstream, tuple(inputs)


def finalize_failure(
    item_root: str | Path,
    *,
    item_id: str,
    input_fingerprint: str,
    config_fingerprint: str,
    rig_overrides_sha256: str,
    failure_records: Iterable[StageFailureRecord],
    completed_stage_names: Iterable[str] = (),
) -> TerminalFinalizationResult:
    root = Path(item_root)
    normalized_item_id = _require_nonempty(item_id, field="item_id")
    input_digest = _require_digest(input_fingerprint, field="input_fingerprint")
    config_digest = _require_digest(config_fingerprint, field="config_fingerprint")
    overrides_digest = _require_digest(rig_overrides_sha256, field="rig_overrides_sha256")
    records = tuple(failure_records)
    if not records or any(not isinstance(record, StageFailureRecord) for record in records):
        raise TerminalFinalizationError("failure_records must contain at least one StageFailureRecord")
    stages = [record.stage_name for record in records]
    if len(stages) != len(set(stages)):
        raise TerminalFinalizationError("failure_records contains duplicate stages")
    ordered = tuple(sorted(records, key=lambda record: _STAGE_ORDER[record.stage_name]))

    public_records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    failure_inputs: list[FileDigest] = []
    for record in ordered:
        try:
            described = describe_file(root, record.record_path)
        except ArtifactContractError as exc:
            raise TerminalFinalizationError(f"invalid failure record for stage {record.stage_name}: {exc}") from exc
        failure_inputs.append(described)
        public_records.append(
            {
                "stage": record.stage_name,
                "path": described.path,
                "size": described.size,
                "sha256": described.sha256,
            }
        )
        diagnostics.extend(dict(diagnostic) for diagnostic in record.diagnostics)

    failure_set_digest = canonical_json_sha256(
        {"failure_records": public_records, "diagnostics": diagnostics}
    )
    error_payload: dict[str, Any] = {
        "schema_version": 1,
        "producer_stage": "G",
        "terminal_state": "failed",
        "item_id": normalized_item_id,
        "input_fingerprint": input_digest,
        "config_fingerprint": config_digest,
        "rig_overrides_sha256": overrides_digest,
        "failed_stages": [record.stage_name for record in ordered],
        "failure_records": public_records,
        "failure_set_sha256": failure_set_digest,
        "diagnostics": diagnostics,
        "retryable": all(record.retryable for record in ordered),
    }
    completed_upstream, completed_inputs = _completed_upstream_manifests(root, completed_stage_names)

    g_marker = _item_path(root, manifest_relative_path("G"))
    g_marker.unlink(missing_ok=True)
    _item_path(root, EXPORT_MANIFEST_PATH).unlink(missing_ok=True)
    error_path = _item_path(root, ERROR_RECORD_PATH)
    atomic_write_json(error_path, error_payload)

    g_config_fingerprint = canonical_json_sha256(
        {
            "finalizer_version": TERMINAL_FINALIZER_VERSION,
            "input_fingerprint": input_digest,
            "config_fingerprint": config_digest,
            "failed_stages": error_payload["failed_stages"],
            "failure_set_sha256": failure_set_digest,
            "completed_upstream_manifests": completed_upstream,
            "status": "failed",
        }
    )
    g_manifest = build_stage_manifest(
        root,
        stage_name="G",
        stage_schema_version=1,
        algorithm_version=TERMINAL_FINALIZER_VERSION,
        upstream_manifests=completed_upstream,
        input_file_sha256=(*completed_inputs, *failure_inputs),
        relevant_config_fingerprint=g_config_fingerprint,
        rig_overrides_sha256=overrides_digest,
        output_paths=[ERROR_RECORD_PATH],
        status="failed",
    )
    write_stage_manifest(root, g_manifest)
    return TerminalFinalizationResult(
        terminal_path=error_path,
        g_manifest=g_manifest,
        payload=error_payload,
    )


def invalidate_terminal(item_root: str | Path) -> None:
    root = Path(item_root)
    for relative in (manifest_relative_path("G"), EXPORT_MANIFEST_PATH, ERROR_RECORD_PATH):
        _item_path(root, relative).unlink(missing_ok=True)


def _load_export_payload(root: Path) -> dict[str, Any] | None:
    export_path = _item_path(root, EXPORT_MANIFEST_PATH)
    try:
        payload = json.loads(export_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _export_payload_matches_graph(
    root: Path,
    payload: Mapping[str, Any],
    manifests: Mapping[str, StageManifest],
) -> bool:
    required_fields = {
        "schema_version",
        "producer_stage",
        "finalizer_version",
        "input_fingerprint",
        "profile",
        "profile_fingerprint",
        "required_formats",
        "upstream_stage_manifests",
        "upstream_artifact_sets",
        "motion_manifest_sha256",
        "global_symbol_table_sha256",
        "validation",
        "formats",
        "status",
    }
    if set(payload) != required_fields:
        return False
    if (
        payload.get("schema_version") != 1
        or payload.get("producer_stage") != "G"
        or payload.get("finalizer_version") != TERMINAL_FINALIZER_VERSION
        or payload.get("required_formats") != list(FORMAL_REQUIRED_FORMATS)
        or payload.get("status") != "completed"
    ):
        return False
    if not _SHA256_PATTERN.fullmatch(str(payload.get("input_fingerprint"))):
        return False
    if not _SHA256_PATTERN.fullmatch(str(payload.get("profile_fingerprint"))):
        return False
    if not _SHA256_PATTERN.fullmatch(str(payload.get("global_symbol_table_sha256"))):
        return False

    expected_upstream = {
        stage: sha256_file(_item_path(root, manifest_relative_path(stage)))
        for stage in ("C", "D", "E")
    }
    if payload.get("upstream_stage_manifests") != expected_upstream:
        return False

    formats_payload = payload.get("formats")
    if not isinstance(formats_payload, dict) or set(formats_payload) != set(FORMAL_REQUIRED_FORMATS):
        return False
    artifact_sets: dict[str, str] = {}
    for format_id in FORMAL_REQUIRED_FORMATS:
        format_payload = formats_payload.get(format_id)
        if not isinstance(format_payload, dict) or set(format_payload) != {"status", "files"}:
            return False
        if format_payload.get("status") != "validated" or not isinstance(format_payload.get("files"), list):
            return False
        try:
            records = tuple(FileDigest.from_dict(record) for record in format_payload["files"])
        except ArtifactContractError:
            return False
        paths = [record.path for record in records]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            return False
        stage = _FORMAT_STAGE[format_id]
        if records != manifests[stage].output_file_sha256:
            return False
        artifact_sets[format_id] = _artifact_set_digest(records)

    c_outputs = {record.path: record for record in manifests["C"].output_file_sha256}
    motion = c_outputs.get("rig/motion_manifest.json")
    if motion is None or payload.get("motion_manifest_sha256") != motion.sha256:
        return False
    canonical_textures = tuple(
        record
        for path, record in sorted(c_outputs.items())
        if path.startswith("rig/shared/textures/") and path.endswith(".png")
    )
    expected_artifact_sets = {
        "canonical_textures": _artifact_set_digest(canonical_textures),
        **artifact_sets,
    }
    if payload.get("upstream_artifact_sets") != expected_artifact_sets:
        return False

    validation = payload.get("validation")
    if not isinstance(validation, dict) or set(validation) != {
        "tier",
        "spine_validator_fingerprint",
        "live2d_validator_fingerprint",
    }:
        return False
    if validation.get("tier") != "release":
        return False
    if not all(
        _SHA256_PATTERN.fullmatch(str(validation.get(field)))
        for field in ("spine_validator_fingerprint", "live2d_validator_fingerprint")
    ):
        return False

    g_manifest = manifests.get("G")
    if g_manifest is None:
        return False
    return g_manifest.relevant_config_fingerprint == _terminal_config_fingerprint(payload)


def is_item_completed(
    item_root: str | Path,
    *,
    expected_stage_fingerprints: Mapping[str, str],
) -> bool:
    root = Path(item_root)
    if not _item_path(root, EXPORT_MANIFEST_PATH).is_file():
        return False
    if _item_path(root, ERROR_RECORD_PATH).exists():
        return False
    try:
        result = StageGraphValidator(root).validate(
            target_stage="G",
            expected_fingerprints=expected_stage_fingerprints,
        )
    except (OSError, ValueError):
        return False
    if not result.reusable:
        return False
    payload = _load_export_payload(root)
    if payload is None:
        return False
    try:
        return _export_payload_matches_graph(root, payload, dict(result.manifests))
    except (ArtifactContractError, OSError, ValueError):
        return False
