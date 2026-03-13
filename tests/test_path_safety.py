import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


from module.caption_pipeline.postprocess import _assign_ocr_image_names, _rewrite_ocr_image_paths
from utils.path_safety import safe_child_path, safe_leaf_name, safe_sibling_path


def test_safe_leaf_name_strips_parent_paths_and_defaults():
    assert safe_leaf_name(r"..\nested/evil.png") == "evil.png"
    assert safe_leaf_name("../", default_name="image.png") == "image.png"
    assert safe_leaf_name("", default_name="image.png") == "image.png"


def test_safe_child_path_stays_under_base_dir(tmp_path):
    target = safe_child_path(tmp_path, r"..\nested/evil.png")
    assert target == tmp_path.resolve() / "evil.png"
    assert target.parent == tmp_path.resolve()


def test_safe_sibling_path_stays_next_to_source_file(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")

    target = safe_sibling_path(source, ".srt")

    assert target == source.with_suffix(".srt").resolve()
    assert target.parent == source.parent.resolve()


def test_assign_ocr_image_names_deduplicates_and_builds_rewrite_maps():
    images = [
        SimpleNamespace(id="../unsafe.png"),
        SimpleNamespace(id="nested/unsafe.png"),
        SimpleNamespace(id="plain.png"),
        SimpleNamespace(id="PLAIN.png"),
    ]

    names, raw_name_map, leaf_name_map = _assign_ocr_image_names(images)

    assert names == ["unsafe.png", "unsafe_2.png", "plain.png", "PLAIN_2.png"]
    assert raw_name_map["../unsafe.png"] == "unsafe.png"
    assert raw_name_map["nested/unsafe.png"] == "unsafe_2.png"
    assert leaf_name_map["unsafe.png"] == "unsafe.png"
    assert leaf_name_map["plain.png"] == "plain.png"


def test_rewrite_ocr_image_paths_uses_raw_and_leaf_maps():
    markdown = "![raw](../unsafe.png)\n![leaf](unsafe.png)\n![keep](missing.png)"
    rewritten = _rewrite_ocr_image_paths(
        markdown,
        "doc_assets",
        raw_name_map={"../unsafe.png": "unsafe.png"},
        leaf_name_map={"unsafe.png": "unsafe.png"},
    )

    assert "![raw](doc_assets/unsafe.png)" in rewritten
    assert "![leaf](doc_assets/unsafe.png)" in rewritten
    assert "![keep](missing.png)" in rewritten
