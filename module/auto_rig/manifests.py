from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifacts import (
    ArtifactContractError,
    FileDigest,
    atomic_write_json,
    canonical_json_sha256,
    describe_file,
    normalize_relative_path,
)

STAGE_MANIFEST_SCHEMA_VERSION = 3
VALID_STAGE_NAMES = frozenset({"A", "B", "C", "D", "E", "F", "G"})
VALID_PRODUCTION_STAGE_STATUSES = frozenset(
    {"stage_validated", "stage_validated_with_degradation"}
)
VALID_TERMINAL_STAGE_STATUSES = frozenset(
    {"completed", "completed_with_degradation", "failed"}
)
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "stage_name",
        "stage_schema_version",
        "algorithm_version",
        "stage_fingerprint",
        "upstream_manifests",
        "input_file_sha256",
        "target_input_fingerprint",
        "native_variant_set_sha256",
        "native_variant_eligibility_sha256",
        "relevant_config_fingerprint",
        "rig_overrides_sha256",
        "output_file_sha256",
        "output_inventory_sha256",
        "status",
    }
)

_PUBLIC_EXACT_PATHS = {
    "C": frozenset({"rig/rig.json", "rig/report.json", "rig/motion_manifest.json"}),
    "G": frozenset({"rig/export_manifest.json", "rig/error.json"}),
}
_PUBLIC_DIRECTORY_PREFIXES = {
    "C": ("rig/shared/textures/",),
    "D": ("rig/spine/",),
    "E": ("rig/live2d/",),
}


class StageManifestError(ValueError):
    """Raised when a stage manifest violates its serialized contract."""


def _require_stage_name(stage_name: Any) -> str:
    normalized = str(stage_name)
    if normalized not in VALID_STAGE_NAMES:
        raise StageManifestError(f"unknown auto-rig stage: {stage_name!r}")
    return normalized


def _require_sha256(value: Any, *, field: str) -> str:
    normalized = str(value)
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise StageManifestError(f"{field} must be a lowercase sha256:<hex> SHA-256 value")
    return normalized


def _require_nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StageManifestError(f"{field} must be a non-empty string")
    return value.strip()


def _require_optional_sha256(value: Any, *, field: str) -> str | None:
    return None if value is None else _require_sha256(value, field=field)


def _require_stage_status(stage_name: str, status: Any) -> str:
    if not isinstance(status, str):
        raise StageManifestError("stage status must be a string")
    allowed = (
        VALID_TERMINAL_STAGE_STATUSES
        if stage_name == "G"
        else VALID_PRODUCTION_STAGE_STATUSES
    )
    if status not in allowed:
        raise StageManifestError(
            f"unsupported status for stage {stage_name}: {status!r}"
        )
    return status


def _normalize_semantic_identities(
    *,
    stage_name: str,
    status: str,
    target_input_fingerprint: Any,
    native_variant_set_sha256: Any,
    native_variant_eligibility_sha256: Any,
) -> tuple[str | None, str | None, str | None]:
    target = _require_optional_sha256(
        target_input_fingerprint,
        field="target_input_fingerprint",
    )
    native_set = _require_optional_sha256(
        native_variant_set_sha256,
        field="native_variant_set_sha256",
    )
    native_eligibility = _require_optional_sha256(
        native_variant_eligibility_sha256,
        field="native_variant_eligibility_sha256",
    )
    if stage_name != "G" or status != "failed":
        missing = [
            name
            for name, value in (
                ("target_input_fingerprint", target),
                ("native_variant_set_sha256", native_set),
                ("native_variant_eligibility_sha256", native_eligibility),
            )
            if value is None
        ]
        if missing:
            raise StageManifestError(
                f"successful stage manifest requires {', '.join(missing)}"
            )
    return target, native_set, native_eligibility


def manifest_relative_path(stage_name: str) -> str:
    stage = _require_stage_name(stage_name)
    return f"rig/cache/{stage}/manifest.json"


def _normalize_upstream_manifests(
    stage_name: str,
    upstream_manifests: Mapping[str, str] | Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    try:
        entries = list(upstream_manifests.items()) if isinstance(upstream_manifests, Mapping) else list(upstream_manifests)
    except (AttributeError, TypeError, ValueError) as exc:
        raise StageManifestError("upstream_manifests must be a stage-to-digest mapping") from exc

    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise StageManifestError("upstream_manifests entries must be stage/digest pairs")
        upstream_stage = _require_stage_name(entry[0])
        if upstream_stage == stage_name:
            raise StageManifestError(f"stage {stage_name} cannot list itself as an upstream manifest")
        if upstream_stage in seen:
            raise StageManifestError(f"duplicate upstream manifest: {upstream_stage}")
        seen.add(upstream_stage)
        normalized.append((upstream_stage, _require_sha256(entry[1], field=f"upstream_manifests.{upstream_stage}")))
    return tuple(sorted(normalized))


def _normalize_file_digests(values: Iterable[FileDigest], *, field: str) -> tuple[FileDigest, ...]:
    try:
        normalized = tuple(values)
    except TypeError as exc:
        raise StageManifestError(f"{field} must be a list of file digests") from exc
    if any(not isinstance(value, FileDigest) for value in normalized):
        raise StageManifestError(f"{field} must contain only FileDigest records")
    paths = [value.path for value in normalized]
    if len(paths) != len(set(paths)):
        raise StageManifestError(f"{field} contains a duplicate path")
    return tuple(sorted(normalized, key=lambda value: value.path))


def _parse_file_digests(payload: Any, *, field: str) -> tuple[FileDigest, ...]:
    if not isinstance(payload, list):
        raise StageManifestError(f"{field} must be a list")
    try:
        parsed = tuple(FileDigest.from_dict(item) for item in payload)
    except ArtifactContractError as exc:
        raise StageManifestError(f"invalid {field}: {exc}") from exc
    paths = [item.path for item in parsed]
    if len(paths) != len(set(paths)):
        raise StageManifestError(f"{field} contains a duplicate path")
    if paths != sorted(paths):
        raise StageManifestError(f"{field} must be sorted by path")
    return parsed


def _output_inventory_sha256(outputs: Iterable[FileDigest]) -> str:
    normalized = _normalize_file_digests(outputs, field="output_file_sha256")
    return canonical_json_sha256([item.to_dict() for item in normalized])


def public_output_owner(relative_path: str | Path) -> str | None:
    path = normalize_relative_path(relative_path)
    for stage, exact_paths in _PUBLIC_EXACT_PATHS.items():
        if path in exact_paths:
            return stage
    for stage, prefixes in _PUBLIC_DIRECTORY_PREFIXES.items():
        if any(path.startswith(prefix) for prefix in prefixes):
            return stage
    return None


def scan_stage_public_output_paths(root: str | Path, stage_name: str) -> tuple[str, ...]:
    stage = _require_stage_name(stage_name)
    item_root = Path(root)
    found: set[str] = set()

    for relative_path in _PUBLIC_EXACT_PATHS.get(stage, ()):
        candidate = item_root / Path(*relative_path.split("/"))
        if os.path.lexists(candidate):
            found.add(relative_path)

    for prefix in _PUBLIC_DIRECTORY_PREFIXES.get(stage, ()):
        relative_directory = prefix.rstrip("/")
        directory = item_root / Path(*relative_directory.split("/"))
        if not os.path.lexists(directory):
            continue
        if directory.is_symlink() or not directory.is_dir():
            found.add(relative_directory)
            continue
        pending = [(directory, relative_directory)]
        while pending:
            current, current_relative = pending.pop()
            try:
                with os.scandir(current) as iterator:
                    entries = sorted(iterator, key=lambda entry: entry.name)
            except OSError as exc:
                raise StageManifestError(f"unable to scan public output namespace {current_relative}: {exc}") from exc
            for entry in entries:
                relative = f"{current_relative}/{entry.name}"
                if entry.is_symlink():
                    found.add(relative)
                elif entry.is_dir(follow_symlinks=False):
                    pending.append((Path(entry.path), relative))
                else:
                    found.add(relative)
    return tuple(sorted(found))


def _remove_obsolete_public_outputs(
    root: str | Path,
    stage_name: str,
    declared_paths: Iterable[str],
) -> None:
    stage = _require_stage_name(stage_name)
    item_root = Path(root)
    declared_public = {
        path for path in declared_paths if public_output_owner(path) == stage
    }
    obsolete = set(scan_stage_public_output_paths(item_root, stage)) - declared_public
    for relative_path in sorted(obsolete, reverse=True):
        candidate = item_root / Path(*relative_path.split("/"))
        if candidate.is_dir() and not candidate.is_symlink():
            raise StageManifestError(
                f"owner public namespace contains an unexpected directory artifact: {relative_path}"
            )
        try:
            candidate.unlink(missing_ok=True)
        except OSError as exc:
            raise StageManifestError(f"unable to remove obsolete stage output {relative_path}: {exc}") from exc


def build_stage_fingerprint(
    *,
    stage_name: str,
    stage_schema_version: int,
    algorithm_version: str,
    upstream_manifests: Mapping[str, str] | Iterable[tuple[str, str]],
    input_file_sha256: Iterable[FileDigest],
    target_input_fingerprint: str | None,
    native_variant_set_sha256: str | None,
    native_variant_eligibility_sha256: str | None,
    relevant_config_fingerprint: str,
    rig_overrides_sha256: str,
    status: str,
) -> str:
    stage = _require_stage_name(stage_name)
    if isinstance(stage_schema_version, bool) or not isinstance(stage_schema_version, int) or stage_schema_version < 1:
        raise StageManifestError("stage_schema_version must be a positive integer")
    algorithm = _require_nonempty_string(algorithm_version, field="algorithm_version")
    normalized_status = _require_stage_status(stage, status)
    target, native_set, native_eligibility = _normalize_semantic_identities(
        stage_name=stage,
        status=normalized_status,
        target_input_fingerprint=target_input_fingerprint,
        native_variant_set_sha256=native_variant_set_sha256,
        native_variant_eligibility_sha256=native_variant_eligibility_sha256,
    )
    upstream = _normalize_upstream_manifests(stage, upstream_manifests)
    inputs = _normalize_file_digests(input_file_sha256, field="input_file_sha256")
    config_fingerprint = _require_sha256(relevant_config_fingerprint, field="relevant_config_fingerprint")
    overrides_sha256 = _require_sha256(rig_overrides_sha256, field="rig_overrides_sha256")
    # Resume fingerprints must be derivable before the stage decides whether it degraded.
    payload = {
        "schema_version": STAGE_MANIFEST_SCHEMA_VERSION,
        "stage_name": stage,
        "stage_schema_version": stage_schema_version,
        "algorithm_version": algorithm,
        "upstream_manifests": dict(upstream),
        "input_file_sha256": [item.to_dict() for item in inputs],
        "target_input_fingerprint": target,
        "native_variant_set_sha256": native_set,
        "native_variant_eligibility_sha256": native_eligibility,
        "relevant_config_fingerprint": config_fingerprint,
        "rig_overrides_sha256": overrides_sha256,
    }
    return canonical_json_sha256(payload)


@dataclass(frozen=True)
class StageManifest:
    stage_name: str
    stage_schema_version: int
    algorithm_version: str
    stage_fingerprint: str
    upstream_manifests: tuple[tuple[str, str], ...]
    input_file_sha256: tuple[FileDigest, ...]
    target_input_fingerprint: str | None
    native_variant_set_sha256: str | None
    native_variant_eligibility_sha256: str | None
    relevant_config_fingerprint: str
    rig_overrides_sha256: str
    output_file_sha256: tuple[FileDigest, ...]
    output_inventory_sha256: str
    status: str = "stage_validated"
    schema_version: int = STAGE_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE_MANIFEST_SCHEMA_VERSION:
            raise StageManifestError(f"schema_version must be {STAGE_MANIFEST_SCHEMA_VERSION}, got {self.schema_version!r}")
        stage = _require_stage_name(self.stage_name)
        if (
            isinstance(self.stage_schema_version, bool)
            or not isinstance(self.stage_schema_version, int)
            or self.stage_schema_version < 1
        ):
            raise StageManifestError("stage_schema_version must be a positive integer")
        algorithm = _require_nonempty_string(self.algorithm_version, field="algorithm_version")
        status = _require_stage_status(stage, self.status)
        target, native_set, native_eligibility = _normalize_semantic_identities(
            stage_name=stage,
            status=status,
            target_input_fingerprint=self.target_input_fingerprint,
            native_variant_set_sha256=self.native_variant_set_sha256,
            native_variant_eligibility_sha256=self.native_variant_eligibility_sha256,
        )
        upstream = _normalize_upstream_manifests(stage, self.upstream_manifests)
        inputs = _normalize_file_digests(self.input_file_sha256, field="input_file_sha256")
        outputs = _normalize_file_digests(self.output_file_sha256, field="output_file_sha256")
        inventory_sha256 = _require_sha256(
            self.output_inventory_sha256,
            field="output_inventory_sha256",
        )
        config_fingerprint = _require_sha256(
            self.relevant_config_fingerprint,
            field="relevant_config_fingerprint",
        )
        overrides_sha256 = _require_sha256(self.rig_overrides_sha256, field="rig_overrides_sha256")
        fingerprint = _require_sha256(self.stage_fingerprint, field="stage_fingerprint")

        input_paths = {item.path for item in inputs}
        output_paths = {item.path for item in outputs}
        overlap = input_paths & output_paths
        if overlap:
            raise StageManifestError(f"path cannot be both input and output: {sorted(overlap)[0]}")
        marker_path = manifest_relative_path(stage)
        if marker_path in output_paths:
            raise StageManifestError(f"stage output cannot include its own commit marker: {marker_path}")
        for output_path in output_paths:
            owner = public_output_owner(output_path)
            if owner is not None and owner != stage:
                raise StageManifestError(
                    f"stage {stage} cannot own {owner}-owned public output: {output_path}"
                )

        expected_inventory_sha256 = _output_inventory_sha256(outputs)
        if inventory_sha256 != expected_inventory_sha256:
            raise StageManifestError("output_inventory_sha256 does not match output_file_sha256")

        expected_fingerprint = build_stage_fingerprint(
            stage_name=stage,
            stage_schema_version=self.stage_schema_version,
            algorithm_version=algorithm,
            upstream_manifests=upstream,
            input_file_sha256=inputs,
            target_input_fingerprint=target,
            native_variant_set_sha256=native_set,
            native_variant_eligibility_sha256=native_eligibility,
            relevant_config_fingerprint=config_fingerprint,
            rig_overrides_sha256=overrides_sha256,
            status=status,
        )
        if fingerprint != expected_fingerprint:
            raise StageManifestError("stage_fingerprint does not match the manifest inputs")

        object.__setattr__(self, "stage_name", stage)
        object.__setattr__(self, "algorithm_version", algorithm)
        object.__setattr__(self, "stage_fingerprint", fingerprint)
        object.__setattr__(self, "upstream_manifests", upstream)
        object.__setattr__(self, "input_file_sha256", inputs)
        object.__setattr__(self, "target_input_fingerprint", target)
        object.__setattr__(self, "native_variant_set_sha256", native_set)
        object.__setattr__(self, "native_variant_eligibility_sha256", native_eligibility)
        object.__setattr__(self, "relevant_config_fingerprint", config_fingerprint)
        object.__setattr__(self, "rig_overrides_sha256", overrides_sha256)
        object.__setattr__(self, "output_file_sha256", outputs)
        object.__setattr__(self, "output_inventory_sha256", inventory_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage_name": self.stage_name,
            "stage_schema_version": self.stage_schema_version,
            "algorithm_version": self.algorithm_version,
            "stage_fingerprint": self.stage_fingerprint,
            "upstream_manifests": dict(self.upstream_manifests),
            "input_file_sha256": [item.to_dict() for item in self.input_file_sha256],
            "target_input_fingerprint": self.target_input_fingerprint,
            "native_variant_set_sha256": self.native_variant_set_sha256,
            "native_variant_eligibility_sha256": self.native_variant_eligibility_sha256,
            "relevant_config_fingerprint": self.relevant_config_fingerprint,
            "rig_overrides_sha256": self.rig_overrides_sha256,
            "output_file_sha256": [item.to_dict() for item in self.output_file_sha256],
            "output_inventory_sha256": self.output_inventory_sha256,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "StageManifest":
        if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
            raise StageManifestError(f"stage manifest must contain exactly {sorted(_MANIFEST_FIELDS)}")
        if payload["schema_version"] != STAGE_MANIFEST_SCHEMA_VERSION:
            raise StageManifestError(f"schema_version must be {STAGE_MANIFEST_SCHEMA_VERSION}, got {payload['schema_version']!r}")
        stage = _require_stage_name(payload["stage_name"])
        if not isinstance(payload["upstream_manifests"], dict):
            raise StageManifestError("upstream_manifests must be an object")
        inputs = _parse_file_digests(payload["input_file_sha256"], field="input_file_sha256")
        outputs = _parse_file_digests(payload["output_file_sha256"], field="output_file_sha256")
        return cls(
            schema_version=payload["schema_version"],
            stage_name=stage,
            stage_schema_version=payload["stage_schema_version"],
            algorithm_version=payload["algorithm_version"],
            stage_fingerprint=payload["stage_fingerprint"],
            upstream_manifests=tuple(payload["upstream_manifests"].items()),
            input_file_sha256=inputs,
            target_input_fingerprint=payload["target_input_fingerprint"],
            native_variant_set_sha256=payload["native_variant_set_sha256"],
            native_variant_eligibility_sha256=payload[
                "native_variant_eligibility_sha256"
            ],
            relevant_config_fingerprint=payload["relevant_config_fingerprint"],
            rig_overrides_sha256=payload["rig_overrides_sha256"],
            output_file_sha256=outputs,
            output_inventory_sha256=payload["output_inventory_sha256"],
            status=payload["status"],
        )


def build_stage_manifest(
    root: str | Path,
    *,
    stage_name: str,
    stage_schema_version: int,
    algorithm_version: str,
    upstream_manifests: Mapping[str, str] | Iterable[tuple[str, str]],
    input_file_sha256: Iterable[FileDigest],
    target_input_fingerprint: str | None,
    native_variant_set_sha256: str | None,
    native_variant_eligibility_sha256: str | None,
    relevant_config_fingerprint: str,
    rig_overrides_sha256: str,
    output_paths: Iterable[str | Path],
    status: str | None = None,
) -> StageManifest:
    stage = _require_stage_name(stage_name)
    normalized_status = status or ("completed" if stage == "G" else "stage_validated")
    try:
        normalized_output_paths = [normalize_relative_path(path) for path in output_paths]
    except (ArtifactContractError, TypeError) as exc:
        raise StageManifestError(f"invalid output path: {exc}") from exc
    if len(normalized_output_paths) != len(set(normalized_output_paths)):
        raise StageManifestError("output_file_sha256 contains a duplicate path")
    marker_path = manifest_relative_path(stage)
    if marker_path in normalized_output_paths:
        raise StageManifestError(f"stage output cannot include its own commit marker: {marker_path}")
    for output_path in normalized_output_paths:
        owner = public_output_owner(output_path)
        if owner is not None and owner != stage:
            raise StageManifestError(f"stage {stage} cannot own {owner}-owned public output: {output_path}")
    try:
        outputs = tuple(describe_file(root, path) for path in sorted(normalized_output_paths))
    except ArtifactContractError as exc:
        raise StageManifestError(str(exc)) from exc
    inputs = _normalize_file_digests(input_file_sha256, field="input_file_sha256")
    fingerprint = build_stage_fingerprint(
        stage_name=stage,
        stage_schema_version=stage_schema_version,
        algorithm_version=algorithm_version,
        upstream_manifests=upstream_manifests,
        input_file_sha256=inputs,
        target_input_fingerprint=target_input_fingerprint,
        native_variant_set_sha256=native_variant_set_sha256,
        native_variant_eligibility_sha256=native_variant_eligibility_sha256,
        relevant_config_fingerprint=relevant_config_fingerprint,
        rig_overrides_sha256=rig_overrides_sha256,
        status=normalized_status,
    )
    return StageManifest(
        stage_name=stage,
        stage_schema_version=stage_schema_version,
        algorithm_version=algorithm_version,
        stage_fingerprint=fingerprint,
        upstream_manifests=tuple(upstream_manifests.items() if isinstance(upstream_manifests, Mapping) else upstream_manifests),
        input_file_sha256=inputs,
        target_input_fingerprint=target_input_fingerprint,
        native_variant_set_sha256=native_variant_set_sha256,
        native_variant_eligibility_sha256=native_variant_eligibility_sha256,
        relevant_config_fingerprint=relevant_config_fingerprint,
        rig_overrides_sha256=rig_overrides_sha256,
        output_file_sha256=outputs,
        output_inventory_sha256=_output_inventory_sha256(outputs),
        status=normalized_status,
    )


def write_stage_manifest(root: str | Path, manifest: StageManifest) -> Path:
    item_root = Path(root)
    marker = item_root / Path(*manifest_relative_path(manifest.stage_name).split("/"))
    marker.unlink(missing_ok=True)
    declared_paths = tuple(item.path for item in manifest.output_file_sha256)
    _remove_obsolete_public_outputs(item_root, manifest.stage_name, declared_paths)
    for expected in manifest.output_file_sha256:
        try:
            actual = describe_file(item_root, expected.path)
        except ArtifactContractError as exc:
            raise StageManifestError(str(exc)) from exc
        if actual != expected:
            raise StageManifestError(f"stage output changed before commit: {expected.path}")
    actual_public = set(scan_stage_public_output_paths(item_root, manifest.stage_name))
    declared_public = {
        path for path in declared_paths if public_output_owner(path) == manifest.stage_name
    }
    if actual_public != declared_public:
        raise StageManifestError("stage public output namespace does not match the declared inventory")
    atomic_write_json(marker, manifest.to_dict())
    return marker


def read_stage_manifest(root: str | Path, stage_name: str) -> StageManifest:
    stage = _require_stage_name(stage_name)
    marker = Path(root) / Path(*manifest_relative_path(stage).split("/"))
    if not marker.is_file():
        raise StageManifestError(f"stage manifest is not a regular file: {manifest_relative_path(stage)}")
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageManifestError(f"unable to parse stage manifest {manifest_relative_path(stage)}: {exc}") from exc
    manifest = StageManifest.from_dict(payload)
    if manifest.stage_name != stage:
        raise StageManifestError(f"stage manifest path is for {stage}, but payload declares {manifest.stage_name}")
    return manifest
