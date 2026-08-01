from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from module.auto_rig.artifacts import sha256_file
from module.auto_rig.contracts import (
    AUTO_RIG_INPUT_CONTRACT_VERSION,
    AutoRigCanvasContract,
    AutoRigContractError,
    AutoRigInputContract,
    AutoRigPartContract,
    ValidatedPartSource,
)
from module.auto_rig.jcs import jcs_sha256
from module.auto_rig.native_variants import NativeVariantCandidate, NativeVariantSet
from module.auto_rig.texture_sources import (
    TEXTURE_PIXEL_CONTRACT_VERSION,
    load_base_texture_regions,
    load_native_texture_regions,
)


def _raw_sha(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _contract(tmp_path: Path, *parts: tuple[str, tuple[int, int, int, int], bytes]):
    depth = tmp_path / "depth.bin"
    depth.write_bytes(b"depth")
    records = []
    for part_id, xyxy, rgba in parts:
        width = xyxy[2] - xyxy[0]
        height = xyxy[3] - xyxy[1]
        color = tmp_path / f"{part_id.removeprefix('part/')}.png"
        Image.frombytes("RGBA", (width, height), rgba).save(color)
        tag = part_id.removeprefix("part/")
        records.append(
            AutoRigPartContract(
                source_tag=tag,
                base_tag=tag,
                semantic_slug=tag,
                side=None,
                part_id=part_id,
                xyxy=xyxy,
                depth_median=0.5,
                source=ValidatedPartSource(
                    mode="png",
                    color_path=color,
                    depth_path=depth,
                    color_sha256=sha256_file(color),
                    depth_sha256=sha256_file(depth),
                    layer_name=None,
                ),
            )
        )
    placeholder = tmp_path / "placeholder"
    return AutoRigInputContract(
        schema_version=AUTO_RIG_INPUT_CONTRACT_VERSION,
        item_root=tmp_path,
        tag_version="v3",
        canvas=AutoRigCanvasContract(width=768, height=768, resolution=768),
        save_to_psd=False,
        tblr_split=False,
        payload_mode="png",
        source_image_path=placeholder,
        source_image_sha256="sha256:" + "0" * 64,
        layerdiff_manifest_path=placeholder,
        optimized_manifest_path=placeholder,
        optimized_info_path=placeholder,
        parts=tuple(reversed(records)),
    )


def _candidate(tmp_path: Path, rgba: bytes) -> NativeVariantCandidate:
    path = tmp_path / "blink.png"
    Image.frombytes("RGBA", (2, 2), rgba).save(path)
    alpha = Image.frombytes("RGBA", (2, 2), rgba).getchannel("A").tobytes()
    return NativeVariantCandidate(
        variant_id="blink",
        part_id="part/native.blink",
        semantic_role="eye_closed.coupled",
        composite_mode="occluding_overlay_v1",
        base_part_ids=("part/eyelash.xmax", "part/eyelash.xmin"),
        draw_anchor_part_id="part/eyelash.xmin",
        anchor_base_tag="eyelash",
        anchor_depth_median=0.5,
        xyxy=(10, 20, 12, 22),
        relative_path="blink.png",
        png_path=path,
        file_sha256=sha256_file(path),
        rgba_mode="RGBA",
        alpha_mode="straight",
        color_space="sRGB",
        alpha_mass_u8_sum=sum(alpha),
        alpha_u8=alpha,
    )


def _variant_set(candidate: NativeVariantCandidate) -> NativeVariantSet:
    provisional = NativeVariantSet(
        schema_version="native-variant-set-v1",
        present=True,
        manifest_path=None,
        entries=(candidate,),
        native_variant_set_sha256="",
    )
    return replace(
        provisional,
        native_variant_set_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def test_base_texture_regions_preserve_straight_rgba_and_sort_by_part_id(
    tmp_path: Path,
) -> None:
    face = bytes((30, 60, 90, 255, 9, 8, 7, 0, 1, 2, 3, 128, 4, 5, 6, 64))
    mouth = bytes((200, 20, 40, 255, 12, 34, 56, 0, 1, 1, 1, 255, 2, 2, 2, 255))
    contract = _contract(
        tmp_path,
        ("part/mouth", (30, 40, 32, 42), mouth),
        ("part/face", (10, 20, 12, 22), face),
    )

    regions = load_base_texture_regions(contract)

    assert tuple(region.part_id for region in regions) == ("part/face", "part/mouth")
    assert regions[0].source_kind == "see_through"
    assert regions[0].xyxy == (10, 20, 12, 22)
    assert (regions[0].width, regions[0].height) == (2, 2)
    assert regions[0].rgba_u8 == face
    assert regions[0].rgba_sha256 == _raw_sha(face)
    assert regions[0].alpha_mode == "straight"
    assert regions[0].color_space == "srgb_bytes"
    assert regions[0].pixel_contract_version == TEXTURE_PIXEL_CONTRACT_VERSION


def test_native_texture_region_reverifies_png_and_parsed_alpha(tmp_path: Path) -> None:
    rgba = bytes((5, 6, 7, 255, 8, 9, 10, 0, 11, 12, 13, 128, 14, 15, 16, 64))
    candidate = _candidate(tmp_path, rgba)

    region = load_native_texture_regions(_variant_set(candidate))[0]

    assert region.part_id == "part/native.blink"
    assert region.source_kind == "native_variant"
    assert region.variant_id == "blink"
    assert region.xyxy == candidate.xyxy
    assert region.rgba_u8 == rgba
    assert region.rgba_sha256 == _raw_sha(rgba)


def test_texture_source_rejects_payload_changed_after_snapshot(tmp_path: Path) -> None:
    rgba = bytes((10, 20, 30, 255)) * 4
    contract = _contract(tmp_path, ("part/face", (0, 0, 2, 2), rgba))
    Image.new("RGBA", (2, 2), (99, 88, 77, 255)).save(tmp_path / "face.png")

    with pytest.raises(AutoRigContractError, match="changed after validation"):
        load_base_texture_regions(contract)


def test_native_texture_source_rejects_mutation_or_wrong_mode(tmp_path: Path) -> None:
    rgba = bytes((10, 20, 30, 255)) * 4
    candidate = _candidate(tmp_path, rgba)
    Image.new("RGB", (2, 2), (10, 20, 30)).save(candidate.png_path)

    with pytest.raises(AutoRigContractError, match="changed after validation"):
        load_native_texture_regions(_variant_set(candidate))
