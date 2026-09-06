from __future__ import annotations

import io
import json
import os
from pathlib import Path

import lance
import pyarrow as pa
import pytest
from PIL import Image

from module.lanceImport import load_data
from module.lanceexport import extract_from_lance
from utils import caption_index as caption_index_module
from utils.caption_index import CaptionIndex, load_caption_index
from utils.lance_blob import build_lance_schema, build_lance_value_array


def _png() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(stream, format="PNG")
    return stream.getvalue()


def _dataset(root: Path, rows: list[dict]):
    schema = build_lance_schema([
        ("uris", pa.string()),
        ("mime", pa.string()),
        ("captions", pa.list_(pa.string())),
        ("blob", pa.large_binary()),
    ])
    arrays = [build_lance_value_array([row.get(field.name) for row in rows], field) for field in schema]
    return lance.write_dataset(
        pa.Table.from_arrays(arrays, schema=schema),
        str(root / "case.lance"),
        data_storage_version="2.2",
    )


def test_set_caption_replaces_only_the_same_physical_file_identity():
    assert hasattr(caption_index_module, "caption_file_identity")
    caption_file_identity = caption_index_module.caption_file_identity
    index = CaptionIndex(
        captions={"same.jpg": "same__hash.txt"},
        files={"same__hash.txt": "same.jpg", "same__history.md": "same.jpg"},
    )

    index.set_caption("same.jpg", "same__hash.TXT")

    assert index.captions == {"same.jpg": "same__hash.TXT"}
    assert index.files["same__history.md"] == "same.jpg"
    if caption_file_identity("same__hash.txt") == caption_file_identity("same__hash.TXT"):
        assert "same__hash.txt" not in index.files
    else:
        assert index.files["same__hash.txt"] == "same.jpg"
    assert index.files["same__hash.TXT"] == "same.jpg"


def test_set_caption_rejects_alias_owned_by_another_source():
    assert hasattr(caption_index_module, "caption_file_identity")
    caption_file_identity = caption_index_module.caption_file_identity
    index = CaptionIndex(captions={"a.jpg": "shared.txt"}, files={"shared.txt": "a.jpg"})

    if caption_file_identity("shared.txt") == caption_file_identity("shared.TXT"):
        with pytest.raises(ValueError, match="ownership"):
            index.set_caption("b.jpg", "shared.TXT")
    else:
        index.set_caption("b.jpg", "shared.TXT")
        assert index.files == {"shared.txt": "a.jpg", "shared.TXT": "b.jpg"}


def test_load_collapses_same_owner_alias_to_the_active_spelling(tmp_path):
    payload = {
        "kind": "caption_index",
        "schema_version": 1,
        "captions": {"a.jpg": "caption.TXT"},
        "files": {
            "caption.txt": "a.jpg",
            "caption.TXT": "a.jpg",
            "history.md": "a.jpg",
        },
    }
    (tmp_path / ".qinglong-captions.json").write_text(json.dumps(payload), encoding="utf-8")

    index = load_caption_index(tmp_path, strict=True)

    assert index is not None
    assert index.captions == {"a.jpg": "caption.TXT"}
    assert index.files["caption.TXT"] == "a.jpg"
    assert index.files["history.md"] == "a.jpg"
    if caption_index_module.caption_file_identity("caption.txt") == caption_index_module.caption_file_identity("caption.TXT"):
        assert "caption.txt" not in index.files
    else:
        assert index.files["caption.txt"] == "a.jpg"


def test_set_caption_identity_work_is_constant_after_index_construction(monkeypatch):
    count = 2000
    index = CaptionIndex(
        captions={f"source-{i}.jpg": f"caption-{i}.txt" for i in range(count)},
        files={f"caption-{i}.txt": f"source-{i}.jpg" for i in range(count)},
    )
    actual_identity = caption_index_module.caption_file_identity
    calls = 0

    def counted_identity(path):
        nonlocal calls
        calls += 1
        return actual_identity(path)

    monkeypatch.setattr(caption_index_module, "caption_file_identity", counted_identity)
    index.set_caption("source-0.jpg", "replacement.txt")

    assert calls <= 3


@pytest.mark.skipif(os.name != "nt", reason="Windows case-insensitive filesystem regression")
def test_case_only_extension_reexport_keeps_roundtrip_associations(tmp_path):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    rows = []
    expected = {}
    for extension, caption in ((".jpg", "jpg caption"), (".png", "png caption")):
        source = source_root / f"same{extension}"
        source.write_bytes(_png())
        rows.append({"uris": str(source), "mime": "image/png", "captions": [caption], "blob": None})
        expected[source] = [caption]

    dataset = _dataset(tmp_path, rows)
    extract_from_lance(dataset, str(tmp_path / "out"), caption_extension=".txt", clip_with_caption=False)
    extract_from_lance(dataset, str(tmp_path / "out"), caption_extension=".TXT", clip_with_caption=False)

    assert load_caption_index(source_root, strict=True) is not None
    imported = load_data(str(source_root))
    assert {Path(item["file_path"]): item["caption"] for item in imported
            if Path(item["file_path"]).suffix.lower() in {".jpg", ".png"}} == expected
    assert len(imported) == 2
