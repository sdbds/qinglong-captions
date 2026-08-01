from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from module.auto_rig.contracts import AutoRigContractError, load_auto_rig_input_contract
from module.auto_rig.mask_sources import load_validated_part_alphas
from module.auto_rig.tag_registry import V3_RAW_TAGS
from module.auto_rig.texture_sources import load_base_texture_regions


def _rgba_with_alpha(size: tuple[int, int], alpha: bytes) -> Image.Image:
    image = Image.new("RGBA", size, (30, 90, 160, 255))
    image.putalpha(Image.frombytes("L", size, alpha))
    return image


def _write_psd(
    root: Path,
    *,
    edge: int,
    parts: tuple[tuple[str, tuple[int, int, int, int], bytes], ...],
) -> None:
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer

    color = PSDImage.new(mode="RGBA", size=(edge, edge))
    depth = PSDImage.new(mode="L", size=(edge, edge))
    for tag, (x1, y1, x2, y2), alpha in parts:
        size = (x2 - x1, y2 - y1)
        color.append(
            PixelLayer.frompil(
                _rgba_with_alpha(size, alpha),
                color,
                name=tag,
                top=y1,
                left=x1,
            )
        )
        depth.append(
            PixelLayer.frompil(Image.new("L", size, 120), depth, name=tag, top=y1, left=x1)
        )
    color.save(root / "final.psd")
    depth.save(root / "final_depth.psd")


def _write_item(root: Path, *, mode: str) -> dict[str, bytes]:
    edge = 768
    (root / "layerdiff").mkdir(parents=True)
    (root / "optimized").mkdir()
    Image.new("RGBA", (edge, edge), (240, 240, 240, 255)).save(root / "src_img.png")
    (root / "layerdiff" / "manifest.json").write_text(
        json.dumps(
            {
                "source_path": "input.png",
                "resolution": edge,
                "tag_version": "v3",
                "parts": sorted(f"{tag}.png" for tag in V3_RAW_TAGS),
            }
        ),
        encoding="utf-8",
    )
    alpha_by_tag = {
        "front hair": bytes((0, 255, 0, 0, 128, 255, 0, 0, 0, 255, 0, 0)),
        "face": bytes((0, 0, 255, 0, 255, 255, 0, 0, 255, 0, 0, 0)),
    }
    bboxes = {
        "front hair": (20, 10, 24, 13),
        "face": (5, 30, 9, 33),
    }
    parts = {
        tag: {"tag": tag, "xyxy": list(bboxes[tag]), "depth_median": 0.4 + index * 0.1}
        for index, tag in enumerate(("front hair", "face"))
    }
    (root / "optimized" / "info.json").write_text(
        json.dumps({"parts": parts, "frame_size": [edge, edge]}),
        encoding="utf-8",
    )
    if mode == "png":
        generated = ["info.json"]
        for tag in parts:
            bbox = bboxes[tag]
            size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            _rgba_with_alpha(size, alpha_by_tag[tag]).save(root / "optimized" / f"{tag}.png")
            Image.new("L", size, 120).save(root / "optimized" / f"{tag}_depth.png")
            generated.extend((f"{tag}.png", f"{tag}_depth.png"))
        save_to_psd = False
        final_psd = None
    else:
        _write_psd(
            root,
            edge=edge,
            parts=tuple((tag, bboxes[tag], alpha_by_tag[tag]) for tag in parts),
        )
        generated = ["info.json"]
        save_to_psd = True
        final_psd = str(root / "final.psd")
    (root / "optimized" / "manifest.json").write_text(
        json.dumps(
            {
                "source_path": "input.png",
                "save_to_psd": save_to_psd,
                "tblr_split": False,
                "generated_files": sorted(generated),
                "final_psd": final_psd,
            }
        ),
        encoding="utf-8",
    )
    return alpha_by_tag


@pytest.mark.parametrize("mode", ("png", "psd"))
def test_validated_png_and_psd_sources_decode_identical_alpha(
    tmp_path: Path,
    mode: str,
) -> None:
    expected = _write_item(tmp_path, mode=mode)
    contract = load_auto_rig_input_contract(tmp_path)

    loaded = load_validated_part_alphas(contract)

    assert tuple(record.part.part_id for record in loaded) == (
        "part/face",
        "part/front-hair",
    )
    assert {record.part.source_tag: record.alpha_u8 for record in loaded} == expected
    assert all((record.width, record.height) == (4, 3) for record in loaded)


@pytest.mark.parametrize("mode", ("png", "psd"))
def test_alpha_decoder_rejects_payload_changed_after_contract_validation(
    tmp_path: Path,
    mode: str,
) -> None:
    alpha_by_tag = _write_item(tmp_path, mode=mode)
    contract = load_auto_rig_input_contract(tmp_path)
    mutated = dict(alpha_by_tag)
    mutated["face"] = bytes((255,)) + mutated["face"][1:]
    if mode == "png":
        _rgba_with_alpha((4, 3), mutated["face"]).save(tmp_path / "optimized" / "face.png")
    else:
        bboxes = {"front hair": (20, 10, 24, 13), "face": (5, 30, 9, 33)}
        _write_psd(
            tmp_path,
            edge=768,
            parts=tuple(
                (tag, bboxes[tag], mutated[tag]) for tag in ("front hair", "face")
            ),
        )

    with pytest.raises(AutoRigContractError, match="changed after validation"):
        load_validated_part_alphas(contract)


@pytest.mark.parametrize("mode", ("png", "psd"))
def test_validated_png_and_psd_sources_decode_identical_straight_rgba(
    tmp_path: Path,
    mode: str,
) -> None:
    alpha_by_tag = _write_item(tmp_path, mode=mode)
    contract = load_auto_rig_input_contract(tmp_path)

    regions = load_base_texture_regions(contract)

    by_tag = {region.part_id.removeprefix("part/"): region for region in regions}
    for tag, alpha in alpha_by_tag.items():
        expected = _rgba_with_alpha((4, 3), alpha).tobytes()
        assert by_tag[tag.replace(" ", "-")].rgba_u8 == expected
