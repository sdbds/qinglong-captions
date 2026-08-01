from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

from module.auto_rig.export.live2d.cubism_core import exercise_moc_with_core
from module.auto_rig.export.live2d.moc3 import Moc3CanvasInfo, parse_moc3_v400_envelope
from module.auto_rig.export.live2d.moc3_codec import (
    Moc3CodecError,
    Moc3V400Document,
    decode_moc3_v400,
    empty_v400_sections,
    encode_moc3_v400,
    moc3_v400_sections_descriptor,
)
from module.auto_rig.export.live2d.moc3_sections_kernel import MOC3_V400_SECTION_SPECS


def test_moc3_v400_sections_descriptor_covers_the_typed_kernel() -> None:
    descriptor = moc3_v400_sections_descriptor()

    assert descriptor["descriptor_id"] == "moc3-v400-sections-v1"
    assert descriptor["descriptor_version"] == "moc3-v400-sections-v1"
    assert descriptor["section_index"] == 1
    payload = descriptor["payload"]
    assert isinstance(payload, dict)
    assert payload["section_count"] == len(MOC3_V400_SECTION_SPECS) == 100
    assert [section["sot_index"] for section in payload["sections"]] == list(range(2, 102))
    assert payload["sections"][-1]["name"] == "additional.quad_transforms"


def _empty_document() -> Moc3V400Document:
    return Moc3V400Document(
        counts=(0,) * 23,
        canvas=Moc3CanvasInfo(
            pixels_per_unit=1024.0,
            origin_x=512.0,
            origin_y=512.0,
            width=1024.0,
            height=1024.0,
            flag=0,
        ),
        sections=empty_v400_sections(),
    )


def _one_part_document(*, part_id: object = "PartRoot", visible: object = True) -> Moc3V400Document:
    counts = [0] * 23
    counts[0] = 1
    sections = empty_v400_sections()
    sections.update(
        {
            "part.ids": (part_id,),
            "part.keyform_binding_band_indices": (0,),
            "part.keyform_begin_indices": (0,),
            "part.keyform_counts": (0,),
            "part.visibles": (visible,),
            "part.enables": (True,),
            "part.parent_part_indices": (-1,),
        }
    )
    return Moc3V400Document(counts=tuple(counts), canvas=_empty_document().canvas, sections=sections)


def test_v400_section_registry_is_complete_and_ordered() -> None:
    assert len(MOC3_V400_SECTION_SPECS) == 100
    assert tuple(spec.sot_index for spec in MOC3_V400_SECTION_SPECS) == tuple(range(2, 102))
    assert MOC3_V400_SECTION_SPECS[0].name == "part.runtime_space"
    assert MOC3_V400_SECTION_SPECS[-1].name == "additional.quad_transforms"
    assert len({spec.name for spec in MOC3_V400_SECTION_SPECS}) == 100


def test_empty_v400_document_encodes_deterministically_and_round_trips() -> None:
    document = _empty_document()

    first = encode_moc3_v400(document)
    second = encode_moc3_v400(document)
    decoded = decode_moc3_v400(first)
    envelope = parse_moc3_v400_envelope(first)

    assert first == second
    assert decoded == document
    assert len(first) % 64 == 0
    assert envelope.counts == (0,) * 23
    assert envelope.required_sot_offsets[2:] == (2176,) * 100


@pytest.mark.parametrize(
    "document",
    [
        _one_part_document(part_id="non-ascii-\u89d2\u8272"),
        _one_part_document(part_id="x" * 64),
        _one_part_document(visible=2),
    ],
)
def test_v400_encoder_rejects_invalid_fixed_id_or_bool(document: Moc3V400Document) -> None:
    with pytest.raises(Moc3CodecError):
        encode_moc3_v400(document)


def test_v400_decoder_rejects_nonzero_runtime_space() -> None:
    payload = bytearray(encode_moc3_v400(_one_part_document()))
    first_section_offset = struct.unpack_from("<I", payload, 64 + 2 * 4)[0]
    payload[first_section_offset] = 1

    with pytest.raises(Moc3CodecError, match="runtime"):
        decode_moc3_v400(bytes(payload))


@pytest.mark.optional_runtime
def test_official_sdk_v400_goldens_decode_reencode_and_pass_core(tmp_path: Path) -> None:
    sdk_root = os.environ.get("LIVE2D_SDK_ROOT")
    core_path = os.environ.get("LIVE2D_CUBISM_CORE_PATH")
    if not sdk_root or not core_path:
        pytest.skip("LIVE2D_SDK_ROOT and LIVE2D_CUBISM_CORE_PATH are required")

    resources = Path(sdk_root) / "Samples" / "Resources"
    for model_name in ("Hiyori", "Mark", "Rice"):
        source_path = resources / model_name / f"{model_name}.moc3"
        document = decode_moc3_v400(source_path.read_bytes())
        encoded = encode_moc3_v400(document)
        output_path = tmp_path / f"{model_name}.moc3"
        output_path.write_bytes(encoded)
        result = exercise_moc_with_core(core_path, output_path, capture_model_state=True)

        assert document.counts[4] == len(document.section("art_mesh.ids"))
        assert document.counts[5] == len(document.section("parameter.ids"))
        assert result.consistency is True
        assert result.drawable_count == document.counts[4]
        assert result.parameter_count == document.counts[5]
        assert result.model_state is not None
        assert tuple(drawable.id for drawable in result.model_state.drawables) == document.section("art_mesh.ids")
