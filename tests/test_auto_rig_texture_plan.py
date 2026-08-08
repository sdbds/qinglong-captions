from __future__ import annotations

import hashlib
import struct
from dataclasses import replace
from importlib.metadata import version as distribution_version
from pathlib import Path

import pytest

from module.auto_rig.jcs import jcs_sha256
from module.auto_rig.texture_plan import (
    CANONICAL_PNG_ENCODER_VERSION,
    CANONICAL_TEXTURE_PAGE_SET_VERSION,
    TEXTURE_EXTRUSION_PX,
    TEXTURE_MAX_PAGES,
    TEXTURE_PACKER_VERSION,
    TEXTURE_PAGE_PLAN_VERSION,
    TEXTURE_PAGE_SIZE,
    TEXTURE_SAFETY_GAP_PX,
    TexturePlanError,
    TextureRegionInput,
    build_texture_page_plan,
    materialize_canonical_texture_pages,
    texture_region_input,
)
from module.auto_rig.texture_sources import LoadedTextureRegion

_SHA256_ZERO = "sha256:" + "0" * 64


def _region(part_id: str, width: int, height: int) -> TextureRegionInput:
    return TextureRegionInput(
        part_id=part_id,
        source_kind="see_through",
        variant_id=None,
        source_xyxy=(0, 0, width, height),
        width=width,
        height=height,
        rgba_sha256=_SHA256_ZERO,
        alpha_mode="straight",
        color_space="srgb_bytes",
        pixel_contract_version="texture-pixel-rgba8-straight-srgb-v1",
    )


def _loaded_region(part_id: str, width: int, height: int, rgba: bytes) -> LoadedTextureRegion:
    rgba_sha = f"sha256:{hashlib.sha256(rgba).hexdigest()}"
    return LoadedTextureRegion(
        part_id=part_id,
        source_kind="see_through",
        variant_id=None,
        xyxy=(0, 0, width, height),
        width=width,
        height=height,
        rgba_u8=rgba,
        rgba_sha256=rgba_sha,
        source_file_sha256=_SHA256_ZERO,
        alpha_mode="straight",
        color_space="srgb_bytes",
        pixel_contract_version="texture-pixel-rgba8-straight-srgb-v1",
    )


def _png_chunks(payload: bytes) -> tuple[bytes, ...]:
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    chunks = []
    offset = 8
    while offset < len(payload):
        length = struct.unpack_from(">I", payload, offset)[0]
        kind = payload[offset + 4 : offset + 8]
        chunks.append(kind)
        offset += 12 + length
    assert offset == len(payload)
    return tuple(chunks)


def test_texture_page_plan_freezes_profile_and_three_nested_rectangles() -> None:
    plan = build_texture_page_plan((_region("part/face", 10, 20),))

    assert plan.schema_version == TEXTURE_PAGE_PLAN_VERSION
    assert plan.packer_version == TEXTURE_PACKER_VERSION
    assert plan.page_width == plan.page_height == TEXTURE_PAGE_SIZE == 2048
    assert plan.max_pages == TEXTURE_MAX_PAGES == 4
    assert plan.extrusion_px == TEXTURE_EXTRUSION_PX == 2
    assert plan.safety_gap_px == TEXTURE_SAFETY_GAP_PX == 2
    assert plan.fit is True
    assert plan.failure_reason is None
    assert plan.used_page_count == 1
    assert tuple(page.index for page in plan.pages) == (0,)
    assert plan.pages[0].relative_path == "page_0.png"
    placement = plan.placements[0]
    assert placement.part_id == "part/face"
    assert placement.rotation is False
    assert placement.packed_footprint.to_tuple() == (0, 0, 18, 28)
    assert placement.extrusion_rect.to_tuple() == (2, 2, 14, 24)
    assert placement.content_rect.to_tuple() == (4, 4, 10, 20)
    assert placement.u0 == pytest.approx(4 / 2048)
    assert placement.v_top0 == pytest.approx(4 / 2048)
    assert placement.u1 == pytest.approx(14 / 2048)
    assert placement.v_top1 == pytest.approx(24 / 2048)
    assert plan.sum_padded_region_area == 18 * 28
    assert plan.budget_occupancy == pytest.approx((18 * 28) / (4 * 2048 * 2048))
    assert plan.used_page_fill == pytest.approx((18 * 28) / (2048 * 2048))
    assert plan.plan_sha256 == jcs_sha256(plan.semantic_payload())


def test_texture_page_plan_is_independent_of_input_order_and_uses_stable_id_ties() -> None:
    regions = (
        _region("part/z", 10, 20),
        _region("part/a", 10, 20),
        _region("part/m", 8, 8),
    )

    forward = build_texture_page_plan(regions)
    reversed_plan = build_texture_page_plan(reversed(regions))

    assert reversed_plan == forward
    assert forward.input_part_ids == ("part/a", "part/z", "part/m")
    assert {placement.part_id: placement.packed_footprint.to_tuple() for placement in forward.placements}[
        "part/a"
    ] == (0, 0, 18, 28)


def test_texture_page_plan_footprints_never_overlap() -> None:
    plan = build_texture_page_plan(
        tuple(_region(f"part/p{index}", 300 + index * 7, 200 + index * 11) for index in range(20))
    )

    assert plan.fit is True
    for page in plan.pages:
        placements = [item for item in plan.placements if item.page_index == page.index]
        for index, first in enumerate(placements):
            for second in placements[index + 1 :]:
                assert not first.packed_footprint.overlaps(second.packed_footprint)


def test_texture_page_plan_rejects_single_region_larger_than_usable_page() -> None:
    plan = build_texture_page_plan((_region("part/too-wide", 2041, 10),))

    assert plan.fit is False
    assert plan.failure_reason == "region_oversize"
    assert plan.unplaced_part_ids == ("part/too-wide",)
    assert plan.pages == ()
    assert plan.placements == ()


def test_texture_page_plan_uses_at_most_four_contiguous_pages() -> None:
    four = build_texture_page_plan(
        tuple(_region(f"part/p{index}", 2040, 2040) for index in range(4))
    )
    five = build_texture_page_plan(
        tuple(_region(f"part/p{index}", 2040, 2040) for index in range(5))
    )

    assert four.fit is True
    assert tuple(page.index for page in four.pages) == (0, 1, 2, 3)
    assert tuple(page.relative_path for page in four.pages) == (
        "page_0.png",
        "page_1.png",
        "page_2.png",
        "page_3.png",
    )
    assert five.fit is False
    assert five.failure_reason == "page_budget_exceeded"
    assert five.used_page_count == 4
    assert len(five.unplaced_part_ids) == 1


def test_texture_page_plan_does_not_replace_geometry_with_total_area_estimate() -> None:
    plan = build_texture_page_plan(
        tuple(_region(f"part/p{index}", 1192, 1192) for index in range(5))
    )

    assert plan.sum_padded_region_area < 4 * 2048 * 2048
    assert plan.fit is False
    assert plan.failure_reason == "page_budget_exceeded"


@pytest.mark.parametrize(
    "region",
    (
        replace(_region("part/a", 1, 1), part_id=""),
        replace(_region("part/a", 1, 1), rgba_sha256="bad"),
        replace(_region("part/a", 1, 1), alpha_mode="premultiplied"),
        replace(_region("part/a", 1, 1), source_xyxy=(0,)),
    ),
)
def test_texture_page_plan_rejects_malformed_region_contract(region: TextureRegionInput) -> None:
    with pytest.raises(TexturePlanError) as captured:
        build_texture_page_plan((region,))

    assert captured.value.code == "invalid_texture_region"


def test_texture_page_plan_rejects_duplicate_part_id() -> None:
    with pytest.raises(TexturePlanError) as captured:
        build_texture_page_plan((_region("part/a", 10, 10), _region("part/a", 20, 20)))

    assert captured.value.code == "invalid_texture_region"


def test_canonical_page_materialization_preserves_content_extrusion_and_transparent_gap(
    tmp_path: Path,
) -> None:
    rgba = bytes(
        (
            10, 20, 30, 255,
            40, 50, 60, 128,
            9, 8, 7, 0,
            70, 80, 90, 64,
        )
    )
    loaded = _loaded_region("part/face", 2, 2, rgba)
    plan = build_texture_page_plan((texture_region_input(loaded),))

    page_set = materialize_canonical_texture_pages(plan, (loaded,), item_root=tmp_path)

    assert page_set.schema_version == CANONICAL_TEXTURE_PAGE_SET_VERSION
    assert page_set.encoder.schema_version == CANONICAL_PNG_ENCODER_VERSION
    assert page_set.encoder.pillow_version == distribution_version("Pillow")
    assert len(page_set.pages) == 1
    page = page_set.pages[0]
    payload = (tmp_path / Path(*page.file.path.split("/"))).read_bytes()
    assert page.encoded_png_sha256 == f"sha256:{hashlib.sha256(payload).hexdigest()}"
    assert _png_chunks(payload) == (b"IHDR", b"IDAT", b"IEND")

    from PIL import Image, features

    assert page_set.encoder.zlib_runtime_version == features.version_codec("zlib")

    with Image.open(tmp_path / Path(*page.file.path.split("/"))) as image:
        decoded = image.convert("RGBA")
        assert decoded.size == (2048, 2048)
        assert decoded.getpixel((0, 0)) == (0, 0, 0, 0)
        assert decoded.getpixel((1, 1)) == (0, 0, 0, 0)
        assert decoded.crop((4, 4, 6, 6)).tobytes() == rgba
        assert decoded.getpixel((2, 2)) == (10, 20, 30, 255)
        assert decoded.getpixel((7, 2)) == (40, 50, 60, 128)
        assert decoded.getpixel((2, 7)) == (9, 8, 7, 0)
        assert decoded.getpixel((7, 7)) == (70, 80, 90, 64)
        raw_page = decoded.tobytes()
    assert page.rgba_sha256 == f"sha256:{hashlib.sha256(raw_page).hexdigest()}"
    assert page_set.set_sha256 == jcs_sha256(page_set.semantic_payload())


def test_canonical_page_encoding_is_byte_deterministic_and_removes_obsolete_pages(
    tmp_path: Path,
) -> None:
    rgba = bytes((10, 20, 30, 255)) * 4
    loaded = _loaded_region("part/face", 2, 2, rgba)
    plan = build_texture_page_plan((texture_region_input(loaded),))

    first = materialize_canonical_texture_pages(plan, (loaded,), item_root=tmp_path)
    first_bytes = (tmp_path / Path(*first.pages[0].file.path.split("/"))).read_bytes()
    obsolete = tmp_path / "rig" / "shared" / "textures" / "page_1.png"
    obsolete.write_bytes(b"obsolete")
    second = materialize_canonical_texture_pages(plan, (loaded,), item_root=tmp_path)
    second_bytes = (tmp_path / Path(*second.pages[0].file.path.split("/"))).read_bytes()

    assert first == second
    assert first_bytes == second_bytes
    assert not obsolete.exists()


def test_canonical_page_materialization_rejects_region_bytes_outside_plan(
    tmp_path: Path,
) -> None:
    rgba = bytes((10, 20, 30, 255)) * 4
    loaded = _loaded_region("part/face", 2, 2, rgba)
    plan = build_texture_page_plan((texture_region_input(loaded),))
    mutated_rgba = bytes((99, 88, 77, 255)) * 4
    mutated = _loaded_region("part/face", 2, 2, mutated_rgba)

    with pytest.raises(TexturePlanError) as captured:
        materialize_canonical_texture_pages(plan, (mutated,), item_root=tmp_path)

    assert captured.value.code == "invalid_texture_materialization"
