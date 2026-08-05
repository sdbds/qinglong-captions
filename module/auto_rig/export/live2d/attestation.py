from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ...jcs import JcsContractError, jcs_bytes, jcs_sha256
from .frame_kernel import (
    RotationStackEntry,
    apply_rotation_stack,
    apply_similarity,
    canvas_to_root,
    invert_rotation_stack,
    invert_similarity,
    root_to_canvas,
    rotation_stack_rank_conflict,
)
from .uv_kernel import (
    canonical_top_left_to_core_api_uv,
    canonical_top_left_to_moc_uv,
    core_api_to_canonical_top_left_uv,
    core_api_to_d3d11_sample_uv,
    moc_to_canonical_top_left_uv,
)

ATTESTATION_SCHEMA_VERSION = "live2d-frame-attestation-v3"
COORDINATE_SCHEMA_VERSION = "live2d-frames-v1"
KERNEL_SOURCE_DIGEST_VERSION = "kernel-source-digest-v1"

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "contract_descriptor",
    "live2d_frame_contract_digest",
    "provenance",
    "runtime_attestations",
}
_DESCRIPTOR_FIELDS = {
    "coordinate_schema_version",
    "frame_kinds",
    "semantic_kernels",
    "layout_descriptors",
    "pure_vectors",
    "invariants",
}
_FRAME_FIELDS = {"frame_kind_id", "ordinal", "semantics"}
_KERNEL_FIELDS = {
    "kernel_id",
    "kernel_version",
    "semantics",
    "source_digest_version",
    "source_sha256",
}
_LAYOUT_FIELDS = {"descriptor_id", "descriptor_version", "section_index", "payload"}
_VECTOR_FIELDS = {"vector_id", "kernel_id", "operation", "payload"}
_FIXTURE_FIELDS = {"fixture_id", "payload"}
_INVARIANT_FIELDS = {"invariant_id", "payload"}
_RUNTIME_ATTESTATION_FIELDS = {
    "backend_id",
    "core_sha256",
    "core_version",
    "e0_fixtures",
    "platform_id",
    "runtime_provenance",
    "validator_protocol_digest",
    "validator_source_sha256",
}
_VECTOR_PAYLOAD_FIELDS = {"input", "expected", "tolerance"}
_EXPECTED_FRAME_KINDS = (
    (0, "CANVAS_PIXEL"),
    (1, "ROOT_MODEL"),
    (2, "WARP_LOCAL"),
    (3, "ROTATION_LOCAL"),
    (4, "ARTMESH_PARENT_LOCAL"),
)


class Live2DAttestationError(ValueError):
    """Raised when a Live2D frame attestation fails its structural gate."""


@dataclass(frozen=True, slots=True)
class Live2DStructuralAttestation:
    schema_version: str
    coordinate_schema_version: str
    contract_digest: str
    kernel_ids: tuple[str, ...]
    vector_ids: tuple[str, ...]
    runtime_keys: tuple[tuple[str, str, str, str, str], ...]


@dataclass(frozen=True, slots=True)
class Live2DRuntimeAttestation:
    platform_id: str
    backend_id: str
    core_sha256: str
    validator_protocol_digest: str
    validator_source_sha256: str
    record_sha256: str
    payload: Mapping[str, object]


def _require_object(value: Any, *, field: str, fields: set[str] | None = None) -> dict[str, Any]:
    if type(value) is not dict:
        raise Live2DAttestationError(f"{field} must be a JSON object")
    if fields is not None and set(value) != fields:
        raise Live2DAttestationError(f"{field} fields must be exactly {sorted(fields)}")
    return value


def _require_list(value: Any, *, field: str, allow_empty: bool = False) -> list[Any]:
    if type(value) is not list or (not allow_empty and not value):
        qualifier = "a JSON array" if allow_empty else "a non-empty JSON array"
        raise Live2DAttestationError(f"{field} must be {qualifier}")
    return value


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise Live2DAttestationError(f"{field} must be a non-empty string")
    return value


def _require_id(value: Any, *, field: str) -> str:
    identifier = _require_string(value, field=field)
    if not _ID_PATTERN.fullmatch(identifier):
        raise Live2DAttestationError(f"{field} must be a stable ASCII identifier")
    return identifier


def _require_digest(value: Any, *, field: str) -> str:
    digest = _require_string(value, field=field)
    if not _DIGEST_PATTERN.fullmatch(digest):
        raise Live2DAttestationError(f"{field} must be a lowercase sha256:<hex> digest")
    return digest


def _require_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Live2DAttestationError(f"{field} must be a JSON number")
    number = float(value)
    if not math.isfinite(number):
        raise Live2DAttestationError(f"{field} must be finite")
    return number


def _require_point(value: Any, *, field: str) -> tuple[float, float]:
    if type(value) is not list or len(value) != 2:
        raise Live2DAttestationError(f"{field} must be a two-number array")
    return (
        _require_number(value[0], field=f"{field}[0]"),
        _require_number(value[1], field=f"{field}[1]"),
    )


def _require_payload_object(value: Any, *, field: str) -> dict[str, Any]:
    payload = _require_object(value, field=field)
    try:
        jcs_bytes(payload)
    except JcsContractError as exc:
        raise Live2DAttestationError(f"{field} is not an I-JSON object") from exc
    return payload


def _require_unique(values: list[Any], *, field: str) -> None:
    if len(values) != len(set(values)):
        raise Live2DAttestationError(f"{field} contains duplicate identity keys")


def _require_sorted(values: list[Any], *, field: str) -> None:
    if values != sorted(values):
        raise Live2DAttestationError(f"{field} is not in canonical order")


def _validate_frame_kinds(value: Any) -> tuple[str, ...]:
    records = _require_list(value, field="contract_descriptor.frame_kinds")
    identities: list[tuple[int, str]] = []
    for index, raw in enumerate(records):
        record = _require_object(raw, field=f"frame_kinds[{index}]", fields=_FRAME_FIELDS)
        ordinal = record["ordinal"]
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise Live2DAttestationError(f"frame_kinds[{index}].ordinal must be a non-negative integer")
        frame_id = _require_id(record["frame_kind_id"], field=f"frame_kinds[{index}].frame_kind_id")
        _require_payload_object(record["semantics"], field=f"frame_kinds[{index}].semantics")
        identities.append((ordinal, frame_id))
    _require_unique([identity[0] for identity in identities], field="frame_kinds.ordinal")
    _require_unique([identity[1] for identity in identities], field="frame_kinds.frame_kind_id")
    _require_sorted([identity[0] for identity in identities], field="frame_kinds.ordinal")
    if tuple(identities) != _EXPECTED_FRAME_KINDS:
        raise Live2DAttestationError("live2d-frames-v1 frame kinds do not match the frozen table")
    return tuple(identity[1] for identity in identities)


def _validate_semantic_kernels(value: Any) -> tuple[dict[str, Any], ...]:
    records = _require_list(value, field="contract_descriptor.semantic_kernels")
    normalized: list[dict[str, Any]] = []
    identifiers: list[str] = []
    for index, raw in enumerate(records):
        record = _require_object(raw, field=f"semantic_kernels[{index}]", fields=_KERNEL_FIELDS)
        kernel_id = _require_id(record["kernel_id"], field=f"semantic_kernels[{index}].kernel_id")
        _require_id(record["kernel_version"], field=f"semantic_kernels[{index}].kernel_version")
        if record["source_digest_version"] != KERNEL_SOURCE_DIGEST_VERSION:
            raise Live2DAttestationError("semantic kernel source digest version mismatch")
        _require_digest(record["source_sha256"], field=f"semantic_kernels[{index}].source_sha256")
        _require_payload_object(record["semantics"], field=f"semantic_kernels[{index}].semantics")
        identifiers.append(kernel_id)
        normalized.append(record)
    _require_unique(identifiers, field="semantic_kernels.kernel_id")
    _require_sorted(identifiers, field="semantic_kernels.kernel_id")
    return tuple(normalized)


def _validate_layout_descriptors(value: Any) -> tuple[str, ...]:
    records = _require_list(value, field="contract_descriptor.layout_descriptors")
    sort_keys: list[tuple[int, str]] = []
    identifiers: list[str] = []
    for index, raw in enumerate(records):
        record = _require_object(raw, field=f"layout_descriptors[{index}]", fields=_LAYOUT_FIELDS)
        section_index = record["section_index"]
        if isinstance(section_index, bool) or not isinstance(section_index, int) or section_index < 0:
            raise Live2DAttestationError(f"layout_descriptors[{index}].section_index must be non-negative")
        descriptor_id = _require_id(record["descriptor_id"], field=f"layout_descriptors[{index}].descriptor_id")
        _require_id(
            record["descriptor_version"],
            field=f"layout_descriptors[{index}].descriptor_version",
        )
        _require_payload_object(record["payload"], field=f"layout_descriptors[{index}].payload")
        sort_keys.append((section_index, descriptor_id))
        identifiers.append(descriptor_id)
    _require_unique(identifiers, field="layout_descriptors.descriptor_id")
    _require_sorted(sort_keys, field="layout_descriptors")
    return tuple(identifiers)


def _validate_stack_entries(value: Any, *, field: str) -> list[RotationStackEntry]:
    records = _require_list(value, field=field, allow_empty=True)
    entries: list[RotationStackEntry] = []
    expected_fields = {"rank", "origin", "angle_degrees", "scale"}
    for index, raw in enumerate(records):
        record = _require_object(raw, field=f"{field}[{index}]", fields=expected_fields)
        rank = record["rank"]
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise Live2DAttestationError(f"{field}[{index}].rank must be an integer")
        origin = _require_point(record["origin"], field=f"{field}[{index}].origin")
        angle = _require_number(record["angle_degrees"], field=f"{field}[{index}].angle_degrees")
        scale = _require_number(record["scale"], field=f"{field}[{index}].scale")
        if scale <= 0.0:
            raise Live2DAttestationError(f"{field}[{index}].scale must be positive")
        entries.append((rank, origin[0], origin[1], angle, scale))
    return entries


def _compare_vector_value(actual: Any, expected: Any, tolerance: float, *, field: str) -> None:
    if isinstance(expected, bool):
        if actual is not expected:
            raise Live2DAttestationError(f"{field} boolean mismatch")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        actual_number = _require_number(actual, field=field)
        expected_number = _require_number(expected, field=field)
        if abs(actual_number - expected_number) > tolerance:
            raise Live2DAttestationError(f"{field} numeric mismatch")
        return
    if type(expected) is list:
        if type(actual) is not list or len(actual) != len(expected):
            raise Live2DAttestationError(f"{field} array mismatch")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _compare_vector_value(actual_item, expected_item, tolerance, field=f"{field}[{index}]")
        return
    if type(expected) is dict:
        if type(actual) is not dict or set(actual) != set(expected):
            raise Live2DAttestationError(f"{field} object mismatch")
        for key in expected:
            _compare_vector_value(actual[key], expected[key], tolerance, field=f"{field}.{key}")
        return
    if actual != expected:
        raise Live2DAttestationError(f"{field} value mismatch")


def _run_pure_vector(record: dict[str, Any]) -> None:
    vector_id = record["vector_id"]
    payload = _require_object(record["payload"], field=f"pure vector {vector_id}.payload", fields=_VECTOR_PAYLOAD_FIELDS)
    vector_input = _require_payload_object(payload["input"], field=f"pure vector {vector_id}.input")
    expected = _require_payload_object(payload["expected"], field=f"pure vector {vector_id}.expected")
    tolerance = _require_number(payload["tolerance"], field=f"pure vector {vector_id}.tolerance")
    if tolerance < 0.0:
        raise Live2DAttestationError(f"pure vector {vector_id}.tolerance must be non-negative")

    operation = record["operation"]
    if record["kernel_id"] == "uv-kernel-v1":
        if operation != "cubism-v400-uv-path-v1" or set(vector_input) != {
            "canonical_top_left_uv"
        }:
            raise Live2DAttestationError(f"pure vector {vector_id} UV operation mismatch")
        canonical = _require_point(
            vector_input["canonical_top_left_uv"],
            field=f"pure vector {vector_id}.canonical_top_left_uv",
        )
        moc = canonical_top_left_to_moc_uv(canonical)
        core = canonical_top_left_to_core_api_uv(canonical)
        actual = {
            "canonical_from_core": list(core_api_to_canonical_top_left_uv(core)),
            "canonical_from_moc": list(moc_to_canonical_top_left_uv(moc)),
            "core_api_uv": list(core),
            "d3d11_sample_uv": list(core_api_to_d3d11_sample_uv(core)),
            "moc_uv": list(moc),
        }
    elif record["kernel_id"] != "frame-kernel-v1":
        raise Live2DAttestationError(f"pure vector {vector_id} uses an unsupported kernel")
    elif operation == "canvas-root-round-trip-v1":
        if set(vector_input) != {"canvas", "point"}:
            raise Live2DAttestationError(f"pure vector {vector_id} canvas input fields mismatch")
        canvas = _require_point(vector_input["canvas"], field=f"pure vector {vector_id}.canvas")
        if canvas[0] <= 0.0 or canvas[1] <= 0.0:
            raise Live2DAttestationError(f"pure vector {vector_id} canvas edges must be positive")
        point = _require_point(vector_input["point"], field=f"pure vector {vector_id}.point")
        root = canvas_to_root(point, canvas[0], canvas[1])
        decoded = root_to_canvas(root, canvas[0], canvas[1])
        actual = {"canvas": list(decoded), "root": list(root)}
    elif operation == "similarity-round-trip-v1":
        if set(vector_input) != {"point", "origin", "angle_degrees", "scale"}:
            raise Live2DAttestationError(f"pure vector {vector_id} similarity input fields mismatch")
        point = _require_point(vector_input["point"], field=f"pure vector {vector_id}.point")
        origin = _require_point(vector_input["origin"], field=f"pure vector {vector_id}.origin")
        angle = _require_number(vector_input["angle_degrees"], field=f"pure vector {vector_id}.angle_degrees")
        scale = _require_number(vector_input["scale"], field=f"pure vector {vector_id}.scale")
        if scale <= 0.0:
            raise Live2DAttestationError(f"pure vector {vector_id}.scale must be positive")
        parent = apply_similarity(point, origin=origin, angle_degrees=angle, scale=scale)
        local = invert_similarity(parent, origin=origin, angle_degrees=angle, scale=scale)
        actual = {"local": list(local), "parent": list(parent)}
    elif operation == "rotation-stack-round-trip-v1":
        if set(vector_input) != {"point", "entries"}:
            raise Live2DAttestationError(f"pure vector {vector_id} stack input fields mismatch")
        point = _require_point(vector_input["point"], field=f"pure vector {vector_id}.point")
        entries = _validate_stack_entries(vector_input["entries"], field=f"pure vector {vector_id}.entries")
        if rotation_stack_rank_conflict(entries):
            raise Live2DAttestationError(f"pure vector {vector_id} stack contains duplicate ranks")
        parent = apply_rotation_stack(point, entries)
        local = invert_rotation_stack(parent, entries)
        actual = {"local": list(local), "parent": list(parent)}
    elif operation == "rotation-stack-rank-conflict-v1":
        if set(vector_input) != {"entries"}:
            raise Live2DAttestationError(f"pure vector {vector_id} conflict input fields mismatch")
        entries = _validate_stack_entries(vector_input["entries"], field=f"pure vector {vector_id}.entries")
        actual = {"conflict": rotation_stack_rank_conflict(entries)}
    else:
        raise Live2DAttestationError(f"pure vector {vector_id} uses an unsupported operation")

    _compare_vector_value(actual, expected, tolerance, field=f"pure vector {vector_id}.expected")


def _validate_pure_vectors(
    value: Any,
    kernel_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
    records = _require_list(value, field="contract_descriptor.pure_vectors")
    identifiers: list[str] = []
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        record = _require_object(raw, field=f"pure_vectors[{index}]", fields=_VECTOR_FIELDS)
        vector_id = _require_id(record["vector_id"], field=f"pure_vectors[{index}].vector_id")
        kernel_id = _require_id(record["kernel_id"], field=f"pure_vectors[{index}].kernel_id")
        if kernel_id not in kernel_ids:
            raise Live2DAttestationError(f"pure_vectors[{index}] references an unknown kernel")
        _require_id(record["operation"], field=f"pure_vectors[{index}].operation")
        _require_payload_object(record["payload"], field=f"pure_vectors[{index}].payload")
        identifiers.append(vector_id)
        normalized.append(record)
    _require_unique(identifiers, field="pure_vectors.vector_id")
    _require_sorted(identifiers, field="pure_vectors.vector_id")
    return tuple(identifiers), tuple(normalized)


def _validate_payload_records(
    value: Any,
    *,
    field: str,
    id_field: str,
    expected_fields: set[str],
) -> tuple[str, ...]:
    records = _require_list(value, field=f"contract_descriptor.{field}")
    identifiers: list[str] = []
    for index, raw in enumerate(records):
        record = _require_object(raw, field=f"{field}[{index}]", fields=expected_fields)
        identifier = _require_id(record[id_field], field=f"{field}[{index}].{id_field}")
        _require_payload_object(record["payload"], field=f"{field}[{index}].payload")
        identifiers.append(identifier)
    _require_unique(identifiers, field=f"{field}.{id_field}")
    _require_sorted(identifiers, field=f"{field}.{id_field}")
    return tuple(identifiers)


def _validate_runtime_attestations(
    value: Any,
) -> tuple[tuple[tuple[str, str, str, str, str], ...], tuple[dict[str, Any], ...]]:
    records = _require_list(value, field="runtime_attestations")
    keys: list[tuple[str, str, str, str, str]] = []
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        record = _require_object(
            raw,
            field=f"runtime_attestations[{index}]",
            fields=_RUNTIME_ATTESTATION_FIELDS,
        )
        platform_id = _require_id(
            record["platform_id"],
            field=f"runtime_attestations[{index}].platform_id",
        )
        backend_id = _require_id(
            record["backend_id"],
            field=f"runtime_attestations[{index}].backend_id",
        )
        _require_string(record["core_version"], field=f"runtime_attestations[{index}].core_version")
        core_sha256 = _require_digest(
            record["core_sha256"],
            field=f"runtime_attestations[{index}].core_sha256",
        )
        protocol_digest = _require_digest(
            record["validator_protocol_digest"],
            field=f"runtime_attestations[{index}].validator_protocol_digest",
        )
        source_digest = _require_digest(
            record["validator_source_sha256"],
            field=f"runtime_attestations[{index}].validator_source_sha256",
        )
        _require_payload_object(
            record["runtime_provenance"],
            field=f"runtime_attestations[{index}].runtime_provenance",
        )
        fixtures = _require_list(
            record["e0_fixtures"],
            field=f"runtime_attestations[{index}].e0_fixtures",
        )
        fixture_ids: list[str] = []
        for fixture_index, raw_fixture in enumerate(fixtures):
            fixture = _require_object(
                raw_fixture,
                field=f"runtime_attestations[{index}].e0_fixtures[{fixture_index}]",
                fields=_FIXTURE_FIELDS,
            )
            fixture_id = _require_id(
                fixture["fixture_id"],
                field=f"runtime_attestations[{index}].e0_fixtures[{fixture_index}].fixture_id",
            )
            _require_payload_object(
                fixture["payload"],
                field=f"runtime_attestations[{index}].e0_fixtures[{fixture_index}].payload",
            )
            fixture_ids.append(fixture_id)
        _require_unique(fixture_ids, field=f"runtime_attestations[{index}].e0_fixtures")
        _require_sorted(fixture_ids, field=f"runtime_attestations[{index}].e0_fixtures")
        keys.append((platform_id, backend_id, core_sha256, protocol_digest, source_digest))
        normalized.append(record)
    _require_unique(keys, field="runtime_attestations")
    _require_sorted(keys, field="runtime_attestations")
    return tuple(keys), tuple(normalized)


def kernel_source_sha256(source: bytes | str) -> str:
    """Apply KernelSourceDigest v1 and return the repository digest form."""

    if isinstance(source, bytes):
        try:
            text = source.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise Live2DAttestationError("semantic kernel source must be UTF-8") from exc
    elif isinstance(source, str):
        text = source
    else:
        raise Live2DAttestationError("semantic kernel source must be bytes or text")
    if text.startswith("\ufeff"):
        text = text[1:]
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    try:
        encoded = normalized.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise Live2DAttestationError("semantic kernel source contains invalid Unicode") from exc
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Live2DAttestationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def load_live2d_frame_attestation(source: bytes | str) -> Mapping[str, Any]:
    """Parse a byte-for-byte JCS attestation while rejecting duplicate keys."""

    if isinstance(source, bytes):
        raw = source
        try:
            text = source.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise Live2DAttestationError("attestation must be UTF-8") from exc
    elif isinstance(source, str):
        text = source
        try:
            raw = source.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise Live2DAttestationError("attestation contains invalid Unicode") from exc
    else:
        raise Live2DAttestationError("attestation source must be bytes or text")

    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(Live2DAttestationError(f"invalid JSON number constant: {value}")),
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise Live2DAttestationError("attestation is not valid JSON") from exc
    if type(payload) is not dict:
        raise Live2DAttestationError("attestation root must be a JSON object")
    try:
        canonical = jcs_bytes(payload)
    except JcsContractError as exc:
        raise Live2DAttestationError("attestation is not valid I-JSON") from exc
    if raw != canonical:
        raise Live2DAttestationError("attestation bytes are not RFC 8785 canonical JSON")
    return payload


def load_packaged_live2d_frame_attestation() -> Mapping[str, Any]:
    """Load and fully validate the repository-shipped frame attestation."""

    from . import frame_kernel, moc3_layout_kernel, moc3_sections_kernel, uv_kernel

    path = Path(__file__).with_name("attestations") / "live2d-frames-v1.json"
    payload = load_live2d_frame_attestation(path.read_bytes())
    validate_live2d_frame_attestation(
        payload,
        kernel_sources={
            "frame-kernel-v1": Path(frame_kernel.__file__).read_bytes(),
            "moc3-layout-kernel-v1": Path(moc3_layout_kernel.__file__).read_bytes(),
            "moc3-sections-kernel-v1": Path(moc3_sections_kernel.__file__).read_bytes(),
            "uv-kernel-v1": Path(uv_kernel.__file__).read_bytes(),
        },
    )
    return payload


def validate_live2d_frame_attestation(
    payload: Mapping[str, Any],
    *,
    kernel_sources: Mapping[str, bytes | str],
) -> Live2DStructuralAttestation:
    """Validate the Core-independent portion of a Live2D frame attestation."""

    root = _require_object(payload, field="attestation", fields=_TOP_LEVEL_FIELDS)
    if root["schema_version"] != ATTESTATION_SCHEMA_VERSION:
        raise Live2DAttestationError("attestation schema version mismatch")
    descriptor = _require_object(
        root["contract_descriptor"],
        field="contract_descriptor",
        fields=_DESCRIPTOR_FIELDS,
    )
    if descriptor["coordinate_schema_version"] != COORDINATE_SCHEMA_VERSION:
        raise Live2DAttestationError("coordinate schema version mismatch")
    _require_payload_object(root["provenance"], field="provenance")
    declared_digest = _require_digest(
        root["live2d_frame_contract_digest"],
        field="live2d_frame_contract_digest",
    )
    _validate_frame_kinds(descriptor["frame_kinds"])
    kernels = _validate_semantic_kernels(descriptor["semantic_kernels"])
    kernel_ids = tuple(record["kernel_id"] for record in kernels)
    _validate_layout_descriptors(descriptor["layout_descriptors"])
    vector_ids, pure_vectors = _validate_pure_vectors(descriptor["pure_vectors"], kernel_ids)
    invariant_records = descriptor["invariants"]
    _validate_payload_records(
        invariant_records,
        field="invariants",
        id_field="invariant_id",
        expected_fields=_INVARIANT_FIELDS,
    )
    runtime_keys, _runtime_records = _validate_runtime_attestations(root["runtime_attestations"])

    try:
        computed_digest = jcs_sha256(descriptor)
    except JcsContractError as exc:
        raise Live2DAttestationError("contract descriptor is not valid I-JSON") from exc
    if computed_digest != declared_digest:
        raise Live2DAttestationError("live2d frame contract digest mismatch")

    if not isinstance(kernel_sources, Mapping) or set(kernel_sources) != set(kernel_ids):
        raise Live2DAttestationError("kernel source set does not match semantic_kernels")
    for record in kernels:
        kernel_id = record["kernel_id"]
        if kernel_source_sha256(kernel_sources[kernel_id]) != record["source_sha256"]:
            raise Live2DAttestationError(f"semantic kernel source mismatch: {kernel_id}")
    for record in pure_vectors:
        _run_pure_vector(record)

    return Live2DStructuralAttestation(
        schema_version=ATTESTATION_SCHEMA_VERSION,
        coordinate_schema_version=COORDINATE_SCHEMA_VERSION,
        contract_digest=computed_digest,
        kernel_ids=kernel_ids,
        vector_ids=vector_ids,
        runtime_keys=runtime_keys,
    )


def select_runtime_attestation(
    payload: Mapping[str, Any],
    *,
    platform_id: str,
    backend_id: str,
    core_sha256: str,
    validator_protocol_digest: str,
    validator_source_sha256: str,
) -> Live2DRuntimeAttestation:
    """Select one exact platform/backend/Core/protocol/source E0 record.

    Full callers must run :func:`validate_live2d_frame_attestation` first so the
    semantic-kernel sources are checked. Selection repeats the signed envelope
    and runtime-record checks to prevent a malformed or ambiguous lookup.
    """

    root = _require_object(payload, field="attestation", fields=_TOP_LEVEL_FIELDS)
    if root["schema_version"] != ATTESTATION_SCHEMA_VERSION:
        raise Live2DAttestationError("attestation schema version mismatch")
    descriptor = _require_object(
        root["contract_descriptor"],
        field="contract_descriptor",
        fields=_DESCRIPTOR_FIELDS,
    )
    declared_digest = _require_digest(
        root["live2d_frame_contract_digest"],
        field="live2d_frame_contract_digest",
    )
    if jcs_sha256(descriptor) != declared_digest:
        raise Live2DAttestationError("live2d frame contract digest mismatch")
    requested_key = (
        _require_id(platform_id, field="platform_id"),
        _require_id(backend_id, field="backend_id"),
        _require_digest(core_sha256, field="core_sha256"),
        _require_digest(validator_protocol_digest, field="validator_protocol_digest"),
        _require_digest(validator_source_sha256, field="validator_source_sha256"),
    )
    _keys, records = _validate_runtime_attestations(root["runtime_attestations"])
    matches = [
        record
        for record in records
        if (
            record["platform_id"],
            record["backend_id"],
            record["core_sha256"],
            record["validator_protocol_digest"],
            record["validator_source_sha256"],
        )
        == requested_key
    ]
    if len(matches) != 1:
        raise Live2DAttestationError("attestation has no unique exact runtime tuple")
    record = matches[0]
    return Live2DRuntimeAttestation(
        platform_id=requested_key[0],
        backend_id=requested_key[1],
        core_sha256=requested_key[2],
        validator_protocol_digest=requested_key[3],
        validator_source_sha256=requested_key[4],
        record_sha256=jcs_sha256(record),
        payload=record,
    )


__all__ = [
    "ATTESTATION_SCHEMA_VERSION",
    "COORDINATE_SCHEMA_VERSION",
    "KERNEL_SOURCE_DIGEST_VERSION",
    "Live2DAttestationError",
    "Live2DRuntimeAttestation",
    "Live2DStructuralAttestation",
    "kernel_source_sha256",
    "load_live2d_frame_attestation",
    "load_packaged_live2d_frame_attestation",
    "select_runtime_attestation",
    "validate_live2d_frame_attestation",
]
