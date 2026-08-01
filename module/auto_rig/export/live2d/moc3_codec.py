from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .moc3 import Moc3CanvasInfo, parse_moc3_v400_envelope
from .moc3_layout_kernel import (
    MOC3_BODY_OFFSET,
    MOC3_CANVAS_INFO_OFFSET,
    MOC3_COUNT_INFO_ENTRY_COUNT,
    MOC3_COUNT_INFO_OFFSET,
    MOC3_FILE_ALIGNMENT,
    MOC3_HEADER_SIZE,
    MOC3_LITTLE_ENDIAN,
    MOC3_MAGIC,
    MOC3_SOT_ENTRY_COUNT,
    MOC3_SOT_OFFSET,
    MOC3_V400_VERSION,
)
from .moc3_sections_kernel import (
    MOC3_SECTION_KERNEL_VERSION,
    MOC3_V400_SECTION_SPECS,
    Moc3SectionSpec,
    moc3_element_size,
)


class Moc3CodecError(ValueError):
    """Raised when typed V4.00 section data is malformed or unencodable."""


Moc3Scalar = bool | float | int | str
Moc3SectionValues = tuple[Moc3Scalar, ...]


@dataclass(frozen=True)
class Moc3V400Document:
    counts: tuple[int, ...]
    canvas: Moc3CanvasInfo
    sections: Mapping[str, Moc3SectionValues]

    def __post_init__(self) -> None:
        if len(self.counts) != MOC3_COUNT_INFO_ENTRY_COUNT or any(
            type(count) is not int or count < 0 for count in self.counts
        ):
            raise Moc3CodecError("MOC3 V4.00 counts must contain 23 non-negative integers")
        if not isinstance(self.canvas, Moc3CanvasInfo):
            raise Moc3CodecError("MOC3 V4.00 canvas must be Moc3CanvasInfo")
        if not isinstance(self.sections, Mapping):
            raise Moc3CodecError("MOC3 V4.00 sections must be a mapping")
        expected_names = {spec.name for spec in MOC3_V400_SECTION_SPECS}
        if set(self.sections) != expected_names:
            missing = sorted(expected_names - set(self.sections))
            extra = sorted(set(self.sections) - expected_names)
            raise Moc3CodecError(f"MOC3 V4.00 section set mismatch; missing={missing}, extra={extra}")
        normalized: dict[str, Moc3SectionValues] = {}
        for spec in MOC3_V400_SECTION_SPECS:
            values = self.sections[spec.name]
            if not isinstance(values, (list, tuple)):
                raise Moc3CodecError(f"MOC3 section {spec.name} must be a sequence")
            normalized[spec.name] = tuple(values)
        object.__setattr__(self, "sections", MappingProxyType(normalized))

    def section(self, name: str) -> Moc3SectionValues:
        try:
            return self.sections[name]
        except KeyError as exc:
            raise Moc3CodecError(f"unknown MOC3 V4.00 section: {name}") from exc


def empty_v400_sections() -> dict[str, Moc3SectionValues]:
    return {spec.name: () for spec in MOC3_V400_SECTION_SPECS}


def moc3_v400_sections_descriptor() -> dict[str, Any]:
    """Return the JCS-ready identity of every typed V4.00 section."""

    return {
        "descriptor_id": "moc3-v400-sections-v1",
        "descriptor_version": MOC3_SECTION_KERNEL_VERSION,
        "section_index": 1,
        "payload": {
            "section_count": len(MOC3_V400_SECTION_SPECS),
            "sections": [
                {
                    "count_index": spec.count_index,
                    "element_kind": spec.element_kind,
                    "element_size": moc3_element_size(spec.element_kind),
                    "name": spec.name,
                    "sot_index": spec.sot_index,
                    "writer_alignment": spec.writer_alignment,
                }
                for spec in MOC3_V400_SECTION_SPECS
            ],
        },
    }


def _align_size(size: int, alignment: int) -> int:
    remainder = size % alignment
    return size if remainder == 0 else size + alignment - remainder


def _expected_count(document: Moc3V400Document, spec: Moc3SectionSpec) -> int:
    return document.counts[spec.count_index]


def _decode_string(payload: bytes, start: int, index: int, section_name: str) -> str:
    raw = payload[start + index * 64 : start + (index + 1) * 64]
    terminator = raw.find(b"\0")
    if terminator <= 0 or any(raw[terminator + 1 :]):
        raise Moc3CodecError(f"MOC3 section {section_name}[{index}] is not a canonical fixed ID")
    try:
        return raw[:terminator].decode("ascii")
    except UnicodeDecodeError as exc:
        raise Moc3CodecError(f"MOC3 section {section_name}[{index}] is not ASCII") from exc


def _decode_values(payload: bytes, start: int, count: int, spec: Moc3SectionSpec) -> Moc3SectionValues:
    kind = spec.element_kind
    if kind == "runtime":
        end = start + count * moc3_element_size(kind)
        if any(payload[start:end]):
            raise Moc3CodecError(f"MOC3 runtime section {spec.name} must be zero-filled before revive")
        return ()
    if kind == "str64":
        return tuple(_decode_string(payload, start, index, spec.name) for index in range(count))
    if not count:
        return ()
    if kind == "i32":
        return tuple(struct.unpack_from(f"<{count}i", payload, start))
    if kind == "f32":
        values = tuple(struct.unpack_from(f"<{count}f", payload, start))
        if not all(math.isfinite(value) for value in values):
            raise Moc3CodecError(f"MOC3 float section {spec.name} contains a non-finite value")
        return values
    if kind == "i16":
        return tuple(struct.unpack_from(f"<{count}h", payload, start))
    if kind == "u8":
        return tuple(payload[start : start + count])
    if kind == "bool32":
        raw_values = struct.unpack_from(f"<{count}i", payload, start)
        if any(value not in (0, 1) for value in raw_values):
            raise Moc3CodecError(f"MOC3 bool section {spec.name} contains a value other than 0 or 1")
        return tuple(bool(value) for value in raw_values)
    raise Moc3CodecError(f"unsupported MOC3 element kind: {kind}")


def decode_moc3_v400(payload: bytes) -> Moc3V400Document:
    envelope = parse_moc3_v400_envelope(payload)
    sections: dict[str, Moc3SectionValues] = {}
    for spec_index, spec in enumerate(MOC3_V400_SECTION_SPECS):
        count = envelope.counts[spec.count_index]
        start = envelope.sot_offsets[spec.sot_index]
        end = start + count * moc3_element_size(spec.element_kind)
        next_start = (
            envelope.sot_offsets[MOC3_V400_SECTION_SPECS[spec_index + 1].sot_index]
            if spec_index + 1 < len(MOC3_V400_SECTION_SPECS)
            else len(payload)
        )
        if end > next_start or end > len(payload):
            raise Moc3CodecError(f"MOC3 section {spec.name} exceeds its typed extent")
        if any(payload[end:next_start]):
            raise Moc3CodecError(f"MOC3 padding after section {spec.name} must be zero-filled")
        sections[spec.name] = _decode_values(payload, start, count, spec)
    return Moc3V400Document(counts=envelope.counts, canvas=envelope.canvas, sections=sections)


def _require_int(value: object, minimum: int, maximum: int, *, field: str) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise Moc3CodecError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def _encode_values(values: Moc3SectionValues, count: int, spec: Moc3SectionSpec) -> bytes:
    kind = spec.element_kind
    if kind == "runtime":
        if values:
            raise Moc3CodecError(f"MOC3 runtime section {spec.name} cannot contain serialized values")
        return bytes(count * moc3_element_size(kind))
    if len(values) != count:
        raise Moc3CodecError(f"MOC3 section {spec.name} must contain exactly {count} values")
    if kind == "str64":
        encoded = bytearray()
        for index, value in enumerate(values):
            if type(value) is not str or not value:
                raise Moc3CodecError(f"MOC3 section {spec.name}[{index}] must be a non-empty ASCII ID")
            try:
                raw = value.encode("ascii")
            except UnicodeEncodeError as exc:
                raise Moc3CodecError(f"MOC3 section {spec.name}[{index}] must be ASCII") from exc
            if len(raw) >= 64:
                raise Moc3CodecError(f"MOC3 section {spec.name}[{index}] exceeds the 63-byte ID limit")
            encoded.extend(raw)
            encoded.extend(bytes(64 - len(raw)))
        return bytes(encoded)
    if kind == "bool32":
        if any(type(value) is not bool for value in values):
            raise Moc3CodecError(f"MOC3 section {spec.name} must contain bool values")
        return struct.pack(f"<{count}i", *(int(value) for value in values)) if count else b""
    if kind == "f32":
        normalized: list[float] = []
        for index, value in enumerate(values):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise Moc3CodecError(f"MOC3 section {spec.name}[{index}] must be finite numeric data")
            normalized.append(float(value))
        try:
            return struct.pack(f"<{count}f", *normalized) if count else b""
        except OverflowError as exc:
            raise Moc3CodecError(f"MOC3 section {spec.name} contains a value outside float32") from exc
    if kind == "i32":
        normalized = [
            _require_int(value, -(2**31), 2**31 - 1, field=f"MOC3 section {spec.name}[{index}]")
            for index, value in enumerate(values)
        ]
        return struct.pack(f"<{count}i", *normalized) if count else b""
    if kind == "i16":
        normalized = [
            _require_int(value, -(2**15), 2**15 - 1, field=f"MOC3 section {spec.name}[{index}]")
            for index, value in enumerate(values)
        ]
        return struct.pack(f"<{count}h", *normalized) if count else b""
    if kind == "u8":
        return bytes(
            _require_int(value, 0, 255, field=f"MOC3 section {spec.name}[{index}]")
            for index, value in enumerate(values)
        )
    raise Moc3CodecError(f"unsupported MOC3 element kind: {kind}")


def _validate_canvas(canvas: Moc3CanvasInfo) -> None:
    numbers = (canvas.pixels_per_unit, canvas.origin_x, canvas.origin_y, canvas.width, canvas.height)
    if not all(type(value) in (int, float) and math.isfinite(value) for value in numbers):
        raise Moc3CodecError("MOC3 canvas values must be finite numbers")
    if canvas.pixels_per_unit <= 0 or canvas.width <= 0 or canvas.height <= 0:
        raise Moc3CodecError("MOC3 canvas PPU, width, and height must be positive")
    _require_int(canvas.flag, 0, 255, field="MOC3 canvas flag")


def encode_moc3_v400(document: Moc3V400Document) -> bytes:
    if not isinstance(document, Moc3V400Document):
        raise Moc3CodecError("encode_moc3_v400 requires a Moc3V400Document")
    _validate_canvas(document.canvas)
    payload = bytearray(MOC3_BODY_OFFSET)
    payload[:4] = MOC3_MAGIC
    payload[4] = MOC3_V400_VERSION
    payload[5] = MOC3_LITTLE_ENDIAN
    struct.pack_into(f"<{MOC3_COUNT_INFO_ENTRY_COUNT}i", payload, MOC3_COUNT_INFO_OFFSET, *document.counts)
    struct.pack_into(
        "<5fB",
        payload,
        MOC3_CANVAS_INFO_OFFSET,
        document.canvas.pixels_per_unit,
        document.canvas.origin_x,
        document.canvas.origin_y,
        document.canvas.width,
        document.canvas.height,
        document.canvas.flag,
    )

    offsets = [0] * MOC3_SOT_ENTRY_COUNT
    offsets[0] = MOC3_COUNT_INFO_OFFSET
    offsets[1] = MOC3_CANVAS_INFO_OFFSET
    for spec in MOC3_V400_SECTION_SPECS:
        aligned_size = _align_size(len(payload), spec.writer_alignment)
        if aligned_size > len(payload):
            payload.extend(bytes(aligned_size - len(payload)))
        offsets[spec.sot_index] = len(payload)
        count = _expected_count(document, spec)
        payload.extend(_encode_values(document.section(spec.name), count, spec))

    final_size = max(_align_size(len(payload), MOC3_FILE_ALIGNMENT), MOC3_BODY_OFFSET + MOC3_FILE_ALIGNMENT)
    payload.extend(bytes(final_size - len(payload)))
    struct.pack_into(f"<{MOC3_SOT_ENTRY_COUNT}I", payload, MOC3_SOT_OFFSET, *offsets)
    if len(payload) < MOC3_HEADER_SIZE:
        raise Moc3CodecError("internal MOC3 encoder error")
    return bytes(payload)


__all__ = [
    "Moc3CodecError",
    "Moc3SectionValues",
    "Moc3V400Document",
    "decode_moc3_v400",
    "empty_v400_sections",
    "encode_moc3_v400",
    "moc3_v400_sections_descriptor",
]
