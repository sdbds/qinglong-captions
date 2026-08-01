from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from module.auto_rig.contracts import AutoRigContractError, load_auto_rig_input_contract
from module.auto_rig.tag_registry import V3_RAW_TAGS


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _part_geometry(tags: tuple[str, ...]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for index, tag in enumerate(tags):
        x1 = 10 + index * 30
        y1 = 20 + index * 10
        result[tag] = {
            "tag": tag,
            "xyxy": [x1, y1, x1 + 20, y1 + 30],
            "depth_median": 0.2 + index * 0.01,
        }
    return result


def _rgba_part(size: tuple[int, int]) -> Image.Image:
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((3, 4, size[0] - 4, size[1] - 5), fill=(80, 120, 200, 220))
    return image


def _write_psd_payload(
    root: Path,
    edge: int,
    parts: dict[str, dict[str, object]],
) -> None:
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer

    color = PSDImage.new(mode="RGBA", size=(edge, edge))
    depth = PSDImage.new(mode="L", size=(edge, edge))
    for tag, part in parts.items():
        x1, y1, x2, y2 = part["xyxy"]
        size = (x2 - x1, y2 - y1)
        color.append(
            PixelLayer.frompil(_rgba_part(size), color, name=tag, top=y1, left=x1)
        )
        depth.append(
            PixelLayer.frompil(Image.new("L", size, 128), depth, name=tag, top=y1, left=x1)
        )
    color.save(root / "final.psd")
    depth.save(root / "final_depth.psd")


def _write_item(
    root: Path,
    *,
    edge: int = 1024,
    frame_size: tuple[int, int] | None = None,
    tags: tuple[str, ...] = ("face", "front hair"),
    tblr_split: bool = False,
    mode: str = "png",
) -> dict[str, dict[str, object]]:
    (root / "layerdiff").mkdir(parents=True)
    (root / "optimized").mkdir()
    Image.new("RGB", (edge, edge), (240, 240, 240)).save(root / "src_img.png")
    (root / "layerdiff" / "manifest.json").write_bytes(
        _json_bytes(
            {
                "source_path": "input.png",
                "resolution": edge,
                "tag_version": "v3",
                "parts": [f"{tag}.png" for tag in V3_RAW_TAGS],
            }
        )
    )
    parts = _part_geometry(tags)
    (root / "optimized" / "info.json").write_bytes(
        _json_bytes(
            {
                "frame_size": list(frame_size or (edge, edge)),
                "parts": parts,
            }
        )
    )
    if mode == "png":
        generated_files = ["info.json"]
        for tag, part in parts.items():
            x1, y1, x2, y2 = part["xyxy"]
            size = (x2 - x1, y2 - y1)
            _rgba_part(size).save(root / "optimized" / f"{tag}.png")
            Image.new("L", size, 128).save(root / "optimized" / f"{tag}_depth.png")
            generated_files.extend((f"{tag}.png", f"{tag}_depth.png"))
        save_to_psd = False
        final_psd = None
    elif mode == "psd":
        _write_psd_payload(root, edge, parts)
        generated_files = ["info.json"]
        save_to_psd = True
        final_psd = str(root / "final.psd")
    else:
        raise AssertionError("unknown test mode")
    (root / "optimized" / "manifest.json").write_bytes(
        _json_bytes(
            {
                "source_path": "input.png",
                "save_to_psd": save_to_psd,
                "tblr_split": tblr_split,
                "generated_files": sorted(generated_files),
                "final_psd": final_psd,
            }
        )
    )
    return parts


@pytest.mark.parametrize("edge", (768, 1024, 1280))
def test_load_png_contract_accepts_frozen_canvas_profiles(tmp_path: Path, edge: int) -> None:
    _write_item(tmp_path, edge=edge)

    contract = load_auto_rig_input_contract(tmp_path)

    assert contract.tag_version == "v3"
    assert contract.canvas.width == edge
    assert contract.canvas.height == edge
    assert contract.payload_mode == "png"
    assert tuple(part.part_id for part in contract.parts) == (
        "part/face",
        "part/front-hair",
    )
    assert all(part.source.layer_name is None for part in contract.parts)
    assert all(part.source.color_path.parent.name == "optimized" for part in contract.parts)


def test_load_contract_accepts_complete_lr_split_or_unsplit_family(tmp_path: Path) -> None:
    split_root = tmp_path / "split"
    unsplit_root = tmp_path / "unsplit"
    _write_item(
        split_root,
        tags=("face", "handwear-r", "handwear-l"),
        tblr_split=True,
    )
    _write_item(unsplit_root, tags=("face", "handwear"), tblr_split=True)

    split = load_auto_rig_input_contract(split_root)
    unsplit = load_auto_rig_input_contract(unsplit_root)

    assert {part.part_id for part in split.parts} == {
        "part/face",
        "part/handwear.xmin",
        "part/handwear.xmax",
    }
    assert {part.part_id for part in unsplit.parts} == {"part/face", "part/handwear"}


def test_load_psd_contract_validates_canvas_layer_names_and_stored_rectangles(tmp_path: Path) -> None:
    parts = _write_item(tmp_path, mode="psd")

    contract = load_auto_rig_input_contract(tmp_path)

    assert contract.payload_mode == "psd"
    assert {part.source.layer_name for part in contract.parts} == set(parts)
    assert all(part.source.color_path == tmp_path / "final.psd" for part in contract.parts)
    assert all(part.source.depth_path == tmp_path / "final_depth.psd" for part in contract.parts)


@pytest.mark.parametrize(
    "edge,frame_size,code",
    (
        (2048, (2048, 2048), "unsupported_auto_rig_canvas_resolution"),
        (1024, (1024, 1000), "unsupported_non_square_frame"),
    ),
)
def test_load_contract_rejects_unsupported_or_non_square_canvas(
    tmp_path: Path,
    edge: int,
    frame_size: tuple[int, int],
    code: str,
) -> None:
    _write_item(tmp_path, edge=edge, frame_size=frame_size)

    with pytest.raises(AutoRigContractError) as captured:
        load_auto_rig_input_contract(tmp_path)

    assert captured.value.code == code


def test_load_contract_never_falls_back_to_root_info(tmp_path: Path) -> None:
    _write_item(tmp_path)
    optimized_info = tmp_path / "optimized" / "info.json"
    root_info = tmp_path / "info.json"
    root_info.write_bytes(optimized_info.read_bytes())
    optimized_info.unlink()

    with pytest.raises(AutoRigContractError, match="optimized/info.json"):
        load_auto_rig_input_contract(tmp_path)


def test_load_contract_never_falls_back_to_root_part_png(tmp_path: Path) -> None:
    _write_item(tmp_path)
    root_part = tmp_path / "face.png"
    root_part.write_bytes((tmp_path / "optimized" / "face.png").read_bytes())
    (tmp_path / "optimized" / "face.png").unlink()

    with pytest.raises(AutoRigContractError, match="optimized/face.png"):
        load_auto_rig_input_contract(tmp_path)


def test_contract_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    _write_item(tmp_path)
    (tmp_path / "optimized" / "info.json").write_text(
        '{"frame_size":[1024,1024],"parts":{},"parts":{}}',
        encoding="utf-8",
    )

    with pytest.raises(AutoRigContractError, match="duplicate key: parts"):
        load_auto_rig_input_contract(tmp_path)


@pytest.mark.parametrize("mutation", ("wrong-size", "wrong-mode", "extra-file", "missing-depth"))
def test_png_contract_rejects_payload_and_inventory_mismatches(
    tmp_path: Path,
    mutation: str,
) -> None:
    _write_item(tmp_path)
    if mutation == "wrong-size":
        Image.new("RGBA", (19, 30)).save(tmp_path / "optimized" / "face.png")
    elif mutation == "wrong-mode":
        Image.new("RGB", (20, 30)).save(tmp_path / "optimized" / "face.png")
    elif mutation == "extra-file":
        (tmp_path / "optimized" / "stale.png").write_bytes(b"stale")
    else:
        (tmp_path / "optimized" / "face_depth.png").unlink()

    with pytest.raises(AutoRigContractError):
        load_auto_rig_input_contract(tmp_path)


def test_psd_contract_rejects_layer_rectangle_different_from_xyxy(tmp_path: Path) -> None:
    parts = _write_item(tmp_path, mode="psd")
    parts["face"]["xyxy"] = [11, 20, 31, 50]
    (tmp_path / "optimized" / "info.json").write_bytes(
        _json_bytes({"frame_size": [1024, 1024], "parts": parts})
    )

    with pytest.raises(AutoRigContractError) as captured:
        load_auto_rig_input_contract(tmp_path)

    assert captured.value.code == "psd_bbox_mismatch"


def test_contract_rejects_required_file_symlink(tmp_path: Path) -> None:
    item_root = tmp_path / "item"
    external = tmp_path / "external-src.png"
    _write_item(item_root)
    Image.new("RGB", (1024, 1024)).save(external)
    (item_root / "src_img.png").unlink()
    try:
        os.symlink(external, item_root / "src_img.png")
    except OSError:
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(AutoRigContractError, match="reparse|symlink"):
        load_auto_rig_input_contract(item_root)


def test_contract_rejects_unknown_tag_version_and_non_finite_depth(tmp_path: Path) -> None:
    parts = _write_item(tmp_path)
    layerdiff_manifest = json.loads((tmp_path / "layerdiff" / "manifest.json").read_text())
    layerdiff_manifest["tag_version"] = "v2"
    (tmp_path / "layerdiff" / "manifest.json").write_bytes(_json_bytes(layerdiff_manifest))

    with pytest.raises(AutoRigContractError):
        load_auto_rig_input_contract(tmp_path)

    layerdiff_manifest["tag_version"] = "v3"
    (tmp_path / "layerdiff" / "manifest.json").write_bytes(_json_bytes(layerdiff_manifest))
    parts["face"]["depth_median"] = float("nan")
    (tmp_path / "optimized" / "info.json").write_bytes(
        _json_bytes({"frame_size": [1024, 1024], "parts": parts})
    )
    with pytest.raises(AutoRigContractError):
        load_auto_rig_input_contract(tmp_path)
