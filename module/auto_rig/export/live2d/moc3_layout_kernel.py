from __future__ import annotations

import math
import struct

MOC3_MAGIC = b"MOC3"
MOC3_V400_VERSION = 3
MOC3_LITTLE_ENDIAN = 0
MOC3_HEADER_SIZE = 64
MOC3_SOT_OFFSET = 64
MOC3_SOT_ENTRY_COUNT = 160
MOC3_SOT_SIZE = MOC3_SOT_ENTRY_COUNT * 4
MOC3_RESERVED_OFFSET = MOC3_SOT_OFFSET + MOC3_SOT_SIZE
MOC3_COUNT_INFO_OFFSET = 1984
MOC3_COUNT_INFO_ENTRY_COUNT = 23
MOC3_COUNT_INFO_SIZE = 128
MOC3_CANVAS_INFO_OFFSET = MOC3_COUNT_INFO_OFFSET + MOC3_COUNT_INFO_SIZE
MOC3_CANVAS_INFO_SIZE = 64
MOC3_BODY_OFFSET = MOC3_CANVAS_INFO_OFFSET + MOC3_CANVAS_INFO_SIZE
MOC3_BUFFER_ALIGNMENT = 64
MOC3_FILE_ALIGNMENT = 64
MOC3_V400_REQUIRED_SOT_COUNT = 102

MOC3_LAYOUT_KERNEL_VERSION = "moc3-v400-envelope-layout-v3"


def decode_header(payload: bytes) -> tuple[bytes, int, int, bytes]:
    return payload[:4], payload[4], payload[5], payload[6:MOC3_HEADER_SIZE]


def decode_sot(payload: bytes) -> tuple[int, ...]:
    return struct.unpack_from(f"<{MOC3_SOT_ENTRY_COUNT}I", payload, MOC3_SOT_OFFSET)


def decode_count_info(payload: bytes) -> tuple[int, ...]:
    return struct.unpack_from(
        f"<{MOC3_COUNT_INFO_ENTRY_COUNT}i",
        payload,
        MOC3_COUNT_INFO_OFFSET,
    )


def decode_canvas_info(payload: bytes) -> tuple[float, float, float, float, float, int]:
    return struct.unpack_from("<5fB", payload, MOC3_CANVAS_INFO_OFFSET)


def file_extent_is_valid(file_size: int) -> bool:
    return file_size > MOC3_BODY_OFFSET and file_size % MOC3_FILE_ALIGNMENT == 0


def zero_filled(payload: bytes) -> bool:
    return not any(payload)


def header_is_v400(magic: bytes, version: int, endian: int, padding: bytes) -> bool:
    return magic == MOC3_MAGIC and version == MOC3_V400_VERSION and endian == MOC3_LITTLE_ENDIAN and zero_filled(padding)


def required_sot_offsets_are_valid(offsets: tuple[int, ...], file_size: int) -> bool:
    if len(offsets) < MOC3_V400_REQUIRED_SOT_COUNT:
        return False
    required = offsets[:MOC3_V400_REQUIRED_SOT_COUNT]
    if required[0] != MOC3_COUNT_INFO_OFFSET or required[1] != MOC3_CANVAS_INFO_OFFSET:
        return False
    for index, offset in enumerate(required):
        if offset == 0 or offset > file_size:
            return False
        if index >= 2 and offset < MOC3_BODY_OFFSET:
            return False
    return all(left <= right for left, right in zip(required, required[1:]))


def unused_sot_offsets_are_valid(offsets: tuple[int, ...], file_size: int) -> bool:
    for offset in offsets[MOC3_V400_REQUIRED_SOT_COUNT:]:
        if offset == 0:
            continue
        if offset < MOC3_BODY_OFFSET or offset > file_size:
            return False
    return True


def counts_are_valid(counts: tuple[int, ...]) -> bool:
    return len(counts) == MOC3_COUNT_INFO_ENTRY_COUNT and all(count >= 0 for count in counts)


def canvas_is_valid(
    pixels_per_unit: float,
    origin_x: float,
    origin_y: float,
    width: float,
    height: float,
) -> bool:
    values = (pixels_per_unit, origin_x, origin_y, width, height)
    return all(math.isfinite(value) for value in values) and pixels_per_unit > 0.0 and width > 0.0 and height > 0.0


__all__ = [
    "MOC3_BODY_OFFSET",
    "MOC3_BUFFER_ALIGNMENT",
    "MOC3_CANVAS_INFO_OFFSET",
    "MOC3_CANVAS_INFO_SIZE",
    "MOC3_COUNT_INFO_ENTRY_COUNT",
    "MOC3_COUNT_INFO_OFFSET",
    "MOC3_COUNT_INFO_SIZE",
    "MOC3_HEADER_SIZE",
    "MOC3_FILE_ALIGNMENT",
    "MOC3_LAYOUT_KERNEL_VERSION",
    "MOC3_LITTLE_ENDIAN",
    "MOC3_MAGIC",
    "MOC3_RESERVED_OFFSET",
    "MOC3_SOT_ENTRY_COUNT",
    "MOC3_SOT_OFFSET",
    "MOC3_SOT_SIZE",
    "MOC3_V400_REQUIRED_SOT_COUNT",
    "MOC3_V400_VERSION",
    "canvas_is_valid",
    "counts_are_valid",
    "decode_canvas_info",
    "decode_count_info",
    "decode_header",
    "decode_sot",
    "file_extent_is_valid",
    "header_is_v400",
    "required_sot_offsets_are_valid",
    "unused_sot_offsets_are_valid",
    "zero_filled",
]
