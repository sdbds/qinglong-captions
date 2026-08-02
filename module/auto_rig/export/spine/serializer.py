from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Mapping

from ...jcs import JcsContractError, jcs_bytes, jcs_sha256
from .atlas import SPINE_ATLAS_ENCODING_VERSION, SpineAtlasPlan, serialize_spine_atlas
from .model import SPINE_JSON_VERSION, SpineDocument

SPINE_SERIALIZATION_VERSION = "spine-serialization-v1"
SPINE_JSON_ENCODING_VERSION = "rfc8785-jcs-utf8-no-bom-v1"
SPINE_PAGE_COPY_VERSION = "canonical-page-byte-copy-v1"


class SpineSerializationError(ValueError):
    """Raised when a Spine public artifact is not canonical or parseable."""

    def __init__(self, message: str) -> None:
        super().__init__(f"invalid_spine_serialization: {message}")


def _error(message: str) -> SpineSerializationError:
    return SpineSerializationError(message)


@dataclass(frozen=True, slots=True)
class SpineEncodingDescriptor:
    schema_version: str
    spine_json_version: str
    skeleton_encoding: str
    report_encoding: str
    atlas_encoding: str
    texture_page_encoding: str
    descriptor_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "spine_json_version": self.spine_json_version,
            "skeleton_encoding": self.skeleton_encoding,
            "report_encoding": self.report_encoding,
            "atlas_encoding": self.atlas_encoding,
            "texture_page_encoding": self.texture_page_encoding,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "descriptor_sha256": self.descriptor_sha256}


def build_spine_encoding_descriptor() -> SpineEncodingDescriptor:
    provisional = SpineEncodingDescriptor(
        schema_version=SPINE_SERIALIZATION_VERSION,
        spine_json_version=SPINE_JSON_VERSION,
        skeleton_encoding=SPINE_JSON_ENCODING_VERSION,
        report_encoding=SPINE_JSON_ENCODING_VERSION,
        atlas_encoding=SPINE_ATLAS_ENCODING_VERSION,
        texture_page_encoding=SPINE_PAGE_COPY_VERSION,
        descriptor_sha256="",
    )
    return replace(
        provisional,
        descriptor_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _error(f"JSON object repeats a key: {key}")
        result[key] = value
    return result


def parse_spine_document(payload: bytes) -> dict[str, object]:
    if not isinstance(payload, bytes) or not payload:
        raise _error("Spine JSON payload must be non-empty bytes")
    if payload.startswith(b"\xef\xbb\xbf"):
        raise _error("Spine JSON must not contain a UTF-8 BOM")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _error("Spine JSON is not UTF-8") from exc
    try:
        value = json.loads(text, object_pairs_hook=_object_without_duplicates)
    except SpineSerializationError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise _error("Spine JSON cannot be parsed") from exc
    if not isinstance(value, dict):
        raise _error("Spine JSON root must be an object")
    try:
        canonical = jcs_bytes(value)
    except JcsContractError as exc:
        raise _error("Spine JSON is outside the RFC 8785 value domain") from exc
    if canonical != payload:
        raise _error("Spine JSON bytes are not canonical RFC 8785 JCS")
    return value


def serialize_spine_document(document: SpineDocument) -> bytes:
    if not isinstance(document, SpineDocument):
        raise _error("document must use SpineDocument")
    payload = jcs_bytes(document.to_dict())
    if parse_spine_document(payload) != document.to_dict():
        raise _error("serialized Spine document did not round-trip")
    return payload


def serialize_spine_report(report: Mapping[str, object]) -> bytes:
    if not isinstance(report, Mapping):
        raise _error("report must be an object")
    try:
        return jcs_bytes(dict(report))
    except JcsContractError as exc:
        raise _error("report is outside the RFC 8785 value domain") from exc


def serialize_spine_atlas_plan(plan: SpineAtlasPlan) -> bytes:
    return serialize_spine_atlas(plan)


__all__ = [
    "SPINE_JSON_ENCODING_VERSION",
    "SPINE_PAGE_COPY_VERSION",
    "SPINE_SERIALIZATION_VERSION",
    "SpineEncodingDescriptor",
    "SpineSerializationError",
    "build_spine_encoding_descriptor",
    "parse_spine_document",
    "serialize_spine_atlas_plan",
    "serialize_spine_document",
    "serialize_spine_report",
]
