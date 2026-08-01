from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .moc3_layout_kernel import (
    MOC3_BODY_OFFSET,
    MOC3_BUFFER_ALIGNMENT,
    MOC3_CANVAS_INFO_OFFSET,
    MOC3_CANVAS_INFO_SIZE,
    MOC3_COUNT_INFO_ENTRY_COUNT,
    MOC3_COUNT_INFO_OFFSET,
    MOC3_COUNT_INFO_SIZE,
    MOC3_FILE_ALIGNMENT,
    MOC3_HEADER_SIZE,
    MOC3_LAYOUT_KERNEL_VERSION,
    MOC3_LITTLE_ENDIAN,
    MOC3_MAGIC,
    MOC3_RESERVED_OFFSET,
    MOC3_SOT_ENTRY_COUNT,
    MOC3_SOT_OFFSET,
    MOC3_SOT_SIZE,
    MOC3_V400_REQUIRED_SOT_COUNT,
    MOC3_V400_VERSION,
    canvas_is_valid,
    counts_are_valid,
    decode_canvas_info,
    decode_count_info,
    decode_header,
    decode_sot,
    file_extent_is_valid,
    header_is_v400,
    required_sot_offsets_are_valid,
    unused_sot_offsets_are_valid,
    zero_filled,
)


class Moc3EnvelopeError(ValueError):
    """Raised when a MOC3 payload violates the frozen V4.00 envelope."""


@dataclass(frozen=True)
class Moc3Header:
    magic: bytes
    version: int
    endian: int


@dataclass(frozen=True)
class Moc3CanvasInfo:
    pixels_per_unit: float
    origin_x: float
    origin_y: float
    width: float
    height: float
    flag: int

    @property
    def origin(self) -> tuple[float, float]:
        return (self.origin_x, self.origin_y)

    @property
    def size(self) -> tuple[float, float]:
        return (self.width, self.height)


@dataclass(frozen=True)
class Moc3Envelope:
    file_size: int
    header: Moc3Header
    sot_offsets: tuple[int, ...]
    counts: tuple[int, ...]
    canvas: Moc3CanvasInfo

    @property
    def required_sot_offsets(self) -> tuple[int, ...]:
        return self.sot_offsets[:MOC3_V400_REQUIRED_SOT_COUNT]


def _require_zeroes(payload: bytes, start: int, end: int, *, field: str) -> None:
    if not zero_filled(payload[start:end]):
        raise Moc3EnvelopeError(f"{field} must be zero-filled")


def _validate_required_offsets(offsets: tuple[int, ...], file_size: int) -> None:
    if not required_sot_offsets_are_valid(offsets, file_size):
        raise Moc3EnvelopeError("required V4.00 SOT offsets violate the attested layout")


def _validate_unused_offsets(offsets: tuple[int, ...], file_size: int) -> None:
    if not unused_sot_offsets_are_valid(offsets, file_size):
        raise Moc3EnvelopeError("unused V4.00 SOT offsets violate the attested layout")


def parse_moc3_v400_envelope(payload: bytes) -> Moc3Envelope:
    """Parse and validate the fixed V4.00 header/SOT/count/canvas envelope only."""

    if type(payload) is not bytes:
        raise Moc3EnvelopeError("MOC3 payload must be bytes")
    if not file_extent_is_valid(len(payload)):
        raise Moc3EnvelopeError("MOC3 payload must contain an aligned body after the fixed envelope")

    magic, version, endian, header_padding = decode_header(payload)
    if not header_is_v400(magic, version, endian, header_padding):
        raise Moc3EnvelopeError("MOC3 header violates the attested V4.00 layout")

    _require_zeroes(
        payload,
        MOC3_RESERVED_OFFSET,
        MOC3_COUNT_INFO_OFFSET,
        field="MOC3 reserved runtime area",
    )

    offsets = decode_sot(payload)
    _validate_required_offsets(offsets, len(payload))
    _validate_unused_offsets(offsets, len(payload))

    counts = decode_count_info(payload)
    if not counts_are_valid(counts):
        raise Moc3EnvelopeError("MOC3 count-info entries must be non-negative")
    count_payload_end = MOC3_COUNT_INFO_OFFSET + MOC3_COUNT_INFO_ENTRY_COUNT * 4
    _require_zeroes(
        payload,
        count_payload_end,
        MOC3_COUNT_INFO_OFFSET + MOC3_COUNT_INFO_SIZE,
        field="MOC3 count-info padding",
    )

    pixels_per_unit, origin_x, origin_y, width, height, flag = decode_canvas_info(payload)
    if not canvas_is_valid(pixels_per_unit, origin_x, origin_y, width, height):
        raise Moc3EnvelopeError("MOC3 canvas values violate the attested layout")
    canvas_payload_end = MOC3_CANVAS_INFO_OFFSET + 5 * 4 + 1
    _require_zeroes(
        payload,
        canvas_payload_end,
        MOC3_CANVAS_INFO_OFFSET + MOC3_CANVAS_INFO_SIZE,
        field="MOC3 canvas-info padding",
    )

    return Moc3Envelope(
        file_size=len(payload),
        header=Moc3Header(magic=magic, version=version, endian=endian),
        sot_offsets=offsets,
        counts=counts,
        canvas=Moc3CanvasInfo(
            pixels_per_unit=pixels_per_unit,
            origin_x=origin_x,
            origin_y=origin_y,
            width=width,
            height=height,
            flag=flag,
        ),
    )


def moc3_v400_layout_descriptor() -> dict[str, Any]:
    """Return the JCS-ready descriptor for the validated envelope scope."""

    return {
        "descriptor_id": "moc3-v400-envelope-v1",
        "descriptor_version": MOC3_LAYOUT_KERNEL_VERSION,
        "section_index": 0,
        "payload": {
            "magic_hex": MOC3_MAGIC.hex(),
            "version_byte": MOC3_V400_VERSION,
            "endian_byte": MOC3_LITTLE_ENDIAN,
            "header_size": MOC3_HEADER_SIZE,
            "sot_offset": MOC3_SOT_OFFSET,
            "sot_entry_count": MOC3_SOT_ENTRY_COUNT,
            "required_sot_first": 0,
            "required_sot_last": MOC3_V400_REQUIRED_SOT_COUNT - 1,
            "count_info_offset": MOC3_COUNT_INFO_OFFSET,
            "count_info_entry_count": MOC3_COUNT_INFO_ENTRY_COUNT,
            "count_info_size": MOC3_COUNT_INFO_SIZE,
            "canvas_info_offset": MOC3_CANVAS_INFO_OFFSET,
            "canvas_info_size": MOC3_CANVAS_INFO_SIZE,
            "moc_buffer_alignment": MOC3_BUFFER_ALIGNMENT,
            "final_file_alignment": MOC3_FILE_ALIGNMENT,
            "padding_policy": "zero-filled",
            "required_sot_policy": "nonzero-at-or-before-eof-nondecreasing",
            "section_offset_alignment": "per-section-codec-no-global-rule",
            "scope": "header-sot-count-canvas-envelope-only",
            "unused_sot_policy": "zero-or-at-or-before-eof",
        },
    }


__all__ = [
    "Moc3CanvasInfo",
    "Moc3Envelope",
    "Moc3EnvelopeError",
    "Moc3Header",
    "moc3_v400_layout_descriptor",
    "parse_moc3_v400_envelope",
]
