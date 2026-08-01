from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path

import pytest

from module.auto_rig.export.live2d.moc3 import (
    Moc3EnvelopeError,
    moc3_v400_layout_descriptor,
    parse_moc3_v400_envelope,
)

_FILE_SIZE = 2240
_SOT_OFFSET = 64
_COUNT_OFFSET = 1984
_CANVAS_OFFSET = 2112
_BODY_OFFSET = 2176
_OFFICIAL_V400_GOLDEN_SHA256 = {
    "0323f5f377b9afef54318be09e9b4eea2a8cb266190ff9d2f843f95ecb77716c",  # Hiyori, CubismWebSamples 4-r.4
    "be56f2d656d279d8db9fe302ec744fa5992ff884b2ebb233f202772d5dd7e4dc",  # Rice, CubismWebSamples 4-r.4
    "f6351429bd3dd655ac95d1027e463ae415e99e2b1b43431a9662fcd316373e9e",  # Mark, CubismWebSamples 4-r.4
}


def _synthetic_v400_envelope() -> bytes:
    payload = bytearray(_FILE_SIZE)
    payload[0:4] = b"MOC3"
    payload[4] = 3
    payload[5] = 0

    offsets = [_BODY_OFFSET] * 160
    offsets[0] = _COUNT_OFFSET
    offsets[1] = _CANVAS_OFFSET
    struct.pack_into("<160I", payload, _SOT_OFFSET, *offsets)
    struct.pack_into("<23i", payload, _COUNT_OFFSET, *([0] * 23))
    struct.pack_into("<5fB", payload, _CANVAS_OFFSET, 1024.0, 512.0, 512.0, 1024.0, 1024.0, 0)
    return bytes(payload)


def _replace_u32(payload: bytes, offset: int, value: int) -> bytes:
    mutated = bytearray(payload)
    struct.pack_into("<I", mutated, offset, value)
    return bytes(mutated)


def _replace_i32(payload: bytes, offset: int, value: int) -> bytes:
    mutated = bytearray(payload)
    struct.pack_into("<i", mutated, offset, value)
    return bytes(mutated)


def _replace_f32(payload: bytes, offset: int, value: float) -> bytes:
    mutated = bytearray(payload)
    struct.pack_into("<f", mutated, offset, value)
    return bytes(mutated)


def test_parse_moc3_v400_envelope_reads_frozen_blocks() -> None:
    envelope = parse_moc3_v400_envelope(_synthetic_v400_envelope())

    assert envelope.file_size == _FILE_SIZE
    assert envelope.header.magic == b"MOC3"
    assert envelope.header.version == 3
    assert envelope.header.endian == 0
    assert len(envelope.sot_offsets) == 160
    assert envelope.required_sot_offsets == tuple(envelope.sot_offsets[:102])
    assert envelope.sot_offsets[0:3] == (_COUNT_OFFSET, _CANVAS_OFFSET, _BODY_OFFSET)
    assert envelope.counts == (0,) * 23
    assert envelope.canvas.pixels_per_unit == 1024.0
    assert envelope.canvas.origin == (512.0, 512.0)
    assert envelope.canvas.size == (1024.0, 1024.0)
    assert envelope.canvas.flag == 0


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        _synthetic_v400_envelope()[:-64],
        _synthetic_v400_envelope() + b"\x00",
    ],
)
def test_parse_moc3_v400_envelope_rejects_invalid_file_extent(payload: bytes) -> None:
    with pytest.raises(Moc3EnvelopeError):
        parse_moc3_v400_envelope(payload)


@pytest.mark.parametrize(
    ("offset", "value"),
    [
        (0, ord("X")),
        (4, 2),
        (5, 1),
        (6, 1),
        (703, 1),
        (704, 1),
        (1983, 1),
        (_COUNT_OFFSET + 23 * 4, 1),
        (_CANVAS_OFFSET + 21, 1),
        (_CANVAS_OFFSET + 63, 1),
    ],
)
def test_parse_moc3_v400_envelope_rejects_noncanonical_header_or_padding(offset: int, value: int) -> None:
    payload = bytearray(_synthetic_v400_envelope())
    payload[offset] = value

    with pytest.raises(Moc3EnvelopeError):
        parse_moc3_v400_envelope(bytes(payload))


@pytest.mark.parametrize(
    ("index", "value"),
    [
        (0, 1920),
        (1, 2048),
        (2, 0),
        (2, _FILE_SIZE),
        (2, _BODY_OFFSET + 1),
        (2, _CANVAS_OFFSET),
        (50, _CANVAS_OFFSET),
    ],
)
def test_parse_moc3_v400_envelope_rejects_invalid_required_sot_offsets(index: int, value: int) -> None:
    payload = _replace_u32(_synthetic_v400_envelope(), _SOT_OFFSET + index * 4, value)

    with pytest.raises(Moc3EnvelopeError):
        parse_moc3_v400_envelope(payload)


def test_parse_moc3_v400_envelope_allows_zero_unused_sot_entries() -> None:
    payload = _replace_u32(_synthetic_v400_envelope(), _SOT_OFFSET + 102 * 4, 0)

    envelope = parse_moc3_v400_envelope(payload)

    assert envelope.sot_offsets[102] == 0


def test_parse_moc3_v400_envelope_does_not_apply_moc_buffer_alignment_to_unused_offsets() -> None:
    payload = _replace_u32(_synthetic_v400_envelope(), _SOT_OFFSET + 102 * 4, _BODY_OFFSET + 1)

    envelope = parse_moc3_v400_envelope(payload)

    assert envelope.sot_offsets[102] == _BODY_OFFSET + 1


def test_parse_moc3_v400_envelope_does_not_apply_moc_buffer_alignment_to_sections() -> None:
    payload = bytearray(_synthetic_v400_envelope())
    offsets = list(struct.unpack_from("<160I", payload, _SOT_OFFSET))
    offsets[3:102] = [_BODY_OFFSET + 1] * 99
    struct.pack_into("<160I", payload, _SOT_OFFSET, *offsets)

    envelope = parse_moc3_v400_envelope(bytes(payload))

    assert envelope.sot_offsets[3] == _BODY_OFFSET + 1


def test_parse_moc3_v400_envelope_allows_zero_length_sections_at_eof() -> None:
    payload = bytearray(_synthetic_v400_envelope())
    offsets = list(struct.unpack_from("<160I", payload, _SOT_OFFSET))
    offsets[2:] = [_FILE_SIZE] * 158
    struct.pack_into("<160I", payload, _SOT_OFFSET, *offsets)

    envelope = parse_moc3_v400_envelope(bytes(payload))

    assert envelope.required_sot_offsets[2:] == (_FILE_SIZE,) * 100
    assert envelope.sot_offsets[102:] == (_FILE_SIZE,) * 58


@pytest.mark.parametrize("value", [_BODY_OFFSET - 1, _FILE_SIZE + 1])
def test_parse_moc3_v400_envelope_rejects_invalid_nonzero_unused_sot_entries(value: int) -> None:
    payload = _replace_u32(_synthetic_v400_envelope(), _SOT_OFFSET + 102 * 4, value)

    with pytest.raises(Moc3EnvelopeError):
        parse_moc3_v400_envelope(payload)


def test_parse_moc3_v400_envelope_rejects_negative_count() -> None:
    payload = _replace_i32(_synthetic_v400_envelope(), _COUNT_OFFSET + 3 * 4, -1)

    with pytest.raises(Moc3EnvelopeError):
        parse_moc3_v400_envelope(payload)


@pytest.mark.parametrize(
    ("field_offset", "value"),
    [
        (0, 0.0),
        (0, float("inf")),
        (4, float("nan")),
        (12, 0.0),
        (16, -1.0),
    ],
)
def test_parse_moc3_v400_envelope_rejects_invalid_canvas_numbers(field_offset: int, value: float) -> None:
    payload = _replace_f32(_synthetic_v400_envelope(), _CANVAS_OFFSET + field_offset, value)

    with pytest.raises(Moc3EnvelopeError):
        parse_moc3_v400_envelope(payload)


def test_parse_moc3_v400_envelope_requires_bytes() -> None:
    with pytest.raises(Moc3EnvelopeError):
        parse_moc3_v400_envelope(bytearray(_synthetic_v400_envelope()))  # type: ignore[arg-type]


def test_moc3_v400_layout_descriptor_freezes_envelope_semantics() -> None:
    descriptor = moc3_v400_layout_descriptor()

    assert descriptor == {
        "descriptor_id": "moc3-v400-envelope-v1",
        "descriptor_version": "moc3-v400-envelope-layout-v3",
        "section_index": 0,
        "payload": {
            "magic_hex": "4d4f4333",
            "version_byte": 3,
            "endian_byte": 0,
            "header_size": 64,
            "sot_offset": 64,
            "sot_entry_count": 160,
            "required_sot_first": 0,
            "required_sot_last": 101,
            "count_info_offset": 1984,
            "count_info_entry_count": 23,
            "count_info_size": 128,
            "canvas_info_offset": 2112,
            "canvas_info_size": 64,
            "moc_buffer_alignment": 64,
            "final_file_alignment": 64,
            "padding_policy": "zero-filled",
            "required_sot_policy": "nonzero-at-or-before-eof-nondecreasing",
            "section_offset_alignment": "per-section-codec-no-global-rule",
            "scope": "header-sot-count-canvas-envelope-only",
            "unused_sot_policy": "zero-or-at-or-before-eof",
        },
    }


@pytest.mark.optional_runtime
def test_configured_official_v400_golden_matches_envelope_contract() -> None:
    golden_path = os.environ.get("LIVE2D_V400_GOLDEN_PATH")
    if not golden_path:
        pytest.skip("LIVE2D_V400_GOLDEN_PATH is not configured")
    payload = Path(golden_path).read_bytes()

    envelope = parse_moc3_v400_envelope(payload)

    assert hashlib.sha256(payload).hexdigest() in _OFFICIAL_V400_GOLDEN_SHA256
    assert envelope.header.version == 3
    assert any(offset % 64 for offset in envelope.required_sot_offsets[2:])
