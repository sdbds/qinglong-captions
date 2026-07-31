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
    canonical_json_sha256,
    describe_file,
    normalize_relative_path,
)


STAGE_MANIFEST_SCHEMA_VERSION = 1
VALID_STAGE_NAMES = frozenset({"A", "B", "C", "D", "E", "F", "G"})
VALID_STAGE_STATUSES = frozenset({"completed", "failed"})
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
        "relevant_config_fingerprint",
        "rig_overrides_sha256",
        "output_file_sha256",
        "status",
    }
)


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


def build_stage_fingerprint(
    *,
    stage_name: str,
    stage_schema_version: int,
    algorithm_version: str,
    upstream_manifests: Mapping[str, str] | Iterable[tuple[str, str]],
    input_file_sha256: Iterable[FileDigest],
    relevant_config_fingerprint: str,
    rig_overrides_sha256: str,
    status: str = "completed",
) -> str:
    stage = _require_stage_name(stage_name)
    if isinstance(stage_schema_version, bool) or not isinstance(stage_schema_version, int) or stage_schema_version < 1:
        raise StageManifestError("stage_schema_version must be a positive integer")
    algorithm = _require_nonempty_string(algorithm_version, field="algorithm_version")
    if status not in VALID_STAGE_STATUSES:
        raise StageManifestError(f"unsupported stage manifest status: {status!r}")
    upstream = _normalize_upstream_manifests(stage, upstream_manifests)
    inputs = _normalize_file_digests(input_file_sha256, field="input_file_sha256")
    config_fingerprint = _require_sha256(relevant_config_fingerprint, field="relevant_config_fingerprint")
    overrides_sha256 = _require_sha256(rig_overrides_sha256, field="rig_overrides_sha256")
    payload = {
        "schema_version": STAGE_MANIFEST_SCHEMA_VERSION,
        "stage_name": stage,
        "stage_schema_version": stage_schema_version,
        "algorithm_version": algorithm,
        "upstream_manifests": dict(upstream),
        "input_file_sha256": [item.to_dict() for item in inputs],
        "relevant_config_fingerprint": config_fingerprint,
        "rig_overrides_sha256": overrides_sha256,
        "status": status,
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
    relevant_config_fingerprint: str
    rig_overrides_sha256: str
    output_file_sha256: tuple[FileDigest, ...]
    status: str = "completed"
    schema_version: int = STAGE_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STAGE_MANIFEST_SCHEMA_VERSION:
            raise StageManifestError(
                f"schema_version must be {STAGE_MANIFEST_SCHEMA_VERSION}, got {self.schema_version!r}"
            )
        stage = _require_stage_name(self.stage_name)
        if isinstance(self.stage_schema_version, bool) or not isinstance(self.stage_schema_version, int) or self.stage_schema_version < 1:
            raise StageManifestError("stage_schema_version must be a positive integer")
        algorithm = _require_nonempty_string(self.algorithm_version, field="algorithm_version")
        if self.status not in VALID_STAGE_STATUSES:
            raise StageManifestError(f"unsupported stage manifest status: {self.status!r}")
        upstream = _normalize_upstream_manifests(stage, self.upstream_manifests)
        inputs = _normalize_file_digests(self.input_file_sha256, field="input_file_sha256")
        outputs = _normalize_file_digests(self.output_file_sha256, field="output_file_sha256")
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

        expected_fingerprint = build_stage_fingerprint(
            stage_name=stage,
            stage_schema_version=self.stage_schema_version,
            algorithm_version=algorithm,
            upstream_manifests=upstream,
            input_file_sha256=inputs,
            relevant_config_fingerprint=config_fingerprint,
            rig_overrides_sha256=overrides_sha256,
            status=self.status,
        )
        if fingerprint != expected_fingerprint:
            raise StageManifestError("stage_fingerprint does not match the manifest inputs")

        object.__setattr__(self, "stage_name", stage)
        object.__setattr__(self, "algorithm_version", algorithm)
        object.__setattr__(self, "stage_fingerprint", fingerprint)
        object.__setattr__(self, "upstream_manifests", upstream)
        object.__setattr__(self, "input_file_sha256", inputs)
        object.__setattr__(self, "relevant_config_fingerprint", config_fingerprint)
        object.__setattr__(self, "rig_overrides_sha256", overrides_sha256)
        object.__setattr__(self, "output_file_sha256", outputs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage_name": self.stage_name,
            "stage_schema_version": self.stage_schema_version,
            "algorithm_version": self.algorithm_version,
            "stage_fingerprint": self.stage_fingerprint,
            "upstream_manifests": dict(self.upstream_manifests),
            "input_file_sha256": [item.to_dict() for item in self.input_file_sha256],
            "relevant_config_fingerprint": self.relevant_config_fingerprint,
            "rig_overrides_sha256": self.rig_overrides_sha256,
            "output_file_sha256": [item.to_dict() for item in self.output_file_sha256],
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "StageManifest":
        if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
            raise StageManifestError(f"stage manifest must contain exactly {sorted(_MANIFEST_FIELDS)}")
        if payload["schema_version"] != STAGE_MANIFEST_SCHEMA_VERSION:
            raise StageManifestError(
                f"schema_version must be {STAGE_MANIFEST_SCHEMA_VERSION}, got {payload['schema_version']!r}"
            )
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
            relevant_config_fingerprint=payload["relevant_config_fingerprint"],
            rig_overrides_sha256=payload["rig_overrides_sha256"],
            output_file_sha256=outputs,
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
    relevant_config_fingerprint: str,
    rig_overrides_sha256: str,
    output_paths: Iterable[str | Path],
    status: str = "completed",
) -> StageManifest:
    stage = _require_stage_name(stage_name)
    try:
        normalized_output_paths = [normalize_relative_path(path) for path in output_paths]
    except (ArtifactContractError, TypeError) as exc:
        raise StageManifestError(f"invalid output path: {exc}") from exc
    if len(normalized_output_paths) != len(set(normalized_output_paths)):
        raise StageManifestError("output_file_sha256 contains a duplicate path")
    marker_path = manifest_relative_path(stage)
    if marker_path in normalized_output_paths:
        raise StageManifestError(f"stage output cannot include its own commit marker: {marker_path}")
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
        relevant_config_fingerprint=relevant_config_fingerprint,
        rig_overrides_sha256=rig_overrides_sha256,
        status=status,
    )
    return StageManifest(
        stage_name=stage,
        stage_schema_version=stage_schema_version,
        algorithm_version=algorithm_version,
        stage_fingerprint=fingerprint,
        upstream_manifests=tuple(
            upstream_manifests.items() if isinstance(upstream_manifests, Mapping) else upstream_manifests
        ),
        input_file_sha256=inputs,
        relevant_config_fingerprint=relevant_config_fingerprint,
        rig_overrides_sha256=rig_overrides_sha256,
        output_file_sha256=outputs,
        status=status,
    )


def write_stage_manifest(root: str | Path, manifest: StageManifest) -> Path:
    item_root = Path(root)
    for expected in manifest.output_file_sha256:
        try:
            actual = describe_file(item_root, expected.path)
        except ArtifactContractError as exc:
            raise StageManifestError(str(exc)) from exc
        if actual != expected:
            raise StageManifestError(f"stage output changed before commit: {expected.path}")
    marker = item_root / Path(*manifest_relative_path(manifest.stage_name).split("/"))
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
        raise StageManifestError(
            f"stage manifest path is for {stage}, but payload declares {manifest.stage_name}"
        )
    return manifest
