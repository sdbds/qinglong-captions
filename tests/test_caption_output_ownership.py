import io
import json
from pathlib import Path

import lance
import pyarrow as pa
import pytest
from PIL import Image
from rich.console import Console

from module.caption_pipeline import orchestrator
from module.lanceexport import extract_from_lance
from module.providers.base import CaptionResult
from tests.test_caption_pipeline import _process_batch_args
from utils.caption_index import CAPTION_INDEX_NAME, load_caption_index
from utils.lance_blob import build_lance_schema, build_lance_value_array
from utils.output_writer import write_caption_output


def _dataset(path, rows):
    schema = build_lance_schema([
        ("uris", pa.string()), ("mime", pa.string()),
        ("captions", pa.list_(pa.string())), ("blob", pa.large_binary()),
        ("duration", pa.int64()), ("hash", pa.string()),
    ])
    arrays = [build_lance_value_array([row.get(field.name) for row in rows], field) for field in schema]
    return lance.write_dataset(pa.Table.from_arrays(arrays, schema=schema), str(path), data_storage_version="2.2")


def _row(source, mime, caption):
    return {"uris": str(source), "mime": mime, "captions": caption, "blob": None, "duration": 0, "hash": source.name}


def _caption_batch(inputs, dataset, monkeypatch):
    monkeypatch.setattr(orchestrator, "create_scene_detector", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "postprocess_caption_content", lambda output, *args: output)

    def provider(**kwargs):
        if kwargs["mime"].startswith("image"):
            return CaptionResult.success("caption for " + Path(kwargs["uri"]).name)
        return CaptionResult.skipped("text source is not captioned")

    orchestrator.process_batch(
        _process_batch_args(inputs), {}, api_process_batch_fn=provider,
        transform2lance_fn=lambda **kwargs: dataset,
        extract_from_lance_fn=extract_from_lance,
        console_obj=Console(file=io.StringIO(), force_terminal=False),
    )


def test_caption_batch_never_overwrites_another_primary_uri(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    image = inputs / "sample.png"
    text = inputs / "sample.txt"
    Image.new("RGB", (4, 4), "red").save(image)
    text.write_text("ORIGINAL PRIMARY TEXT", encoding="utf-8")
    dataset = _dataset(tmp_path / "input.lance", [
        _row(image, "image/png", []), _row(text, "text/plain", []),
    ])
    _caption_batch(inputs, dataset, monkeypatch)
    assert text.read_text(encoding="utf-8") == "ORIGINAL PRIMARY TEXT"


def test_caption_batch_and_export_use_the_same_disambiguated_files(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    images = [inputs / "sample.png", inputs / "sample.jpg"]
    for image in images:
        Image.new("RGB", (4, 4), "red").save(image)
    dataset = _dataset(tmp_path / "input.lance", [_row(image, "image/" + image.suffix[1:], []) for image in images])
    _caption_batch(inputs, dataset, monkeypatch)
    captions = list(inputs.glob("*.txt"))
    assert len(captions) == 2
    assert {path.read_text(encoding="utf-8").strip() for path in captions} == {
        "caption for sample.png", "caption for sample.jpg",
    }
    _caption_batch(inputs, dataset, monkeypatch)
    assert sorted(inputs.glob("*.txt")) == sorted(captions)


@pytest.mark.parametrize("second_writer", ["export", "generation"])
def test_json_companion_ownership_is_checked_before_any_write(tmp_path, second_writer):
    image = tmp_path / "scene.png"
    document = tmp_path / "scene.pdf"
    Image.new("RGB", (4, 4), "blue").save(image)
    document.write_bytes(b"%PDF-example")
    first = _dataset(tmp_path / "first.lance", [
        _row(image, "image/png", [json.dumps({"description": "image caption", "provider": "image"})]),
    ])
    extract_from_lance(first, str(tmp_path / "out"), clip_with_caption=False)
    json_path = tmp_path / "scene.json"
    previous = json_path.read_bytes()
    payload = {"markdown": "document caption", "provider": "document"}
    with pytest.raises(ValueError, match="[Oo]wnership|[Uu]nowned"):
        if second_writer == "export":
            second = _dataset(tmp_path / "second.lance", [_row(document, "application/pdf", [json.dumps(payload)])])
            extract_from_lance(second, str(tmp_path / "out"), clip_with_caption=False)
        else:
            write_caption_output(document, payload, "application/pdf")
    assert json_path.read_bytes() == previous
    assert not (tmp_path / "scene.md").exists()
    assert load_caption_index(tmp_path, strict=True).files["scene.json"] == "scene.png"


def test_json_companion_remains_owned_after_caption_format_changes(tmp_path):
    source = tmp_path / "image.png"
    source.write_bytes(b"image")
    write_caption_output(source, {"description": "first"}, "image/png")
    write_caption_output(source, {"markdown": "second", "caption_extension": ".md"}, "image/png")
    index = load_caption_index(tmp_path, strict=True)
    assert index.captions == {"image.png": "image.md"}
    assert index.files == {"image.txt": "image.png", "image.md": "image.png", "image.json": "image.png"}


def test_unowned_legacy_json_companion_is_not_silently_overwritten(tmp_path):
    source = tmp_path / "image.png"
    source.write_bytes(b"image")
    (tmp_path / "image.txt").write_text("old caption", encoding="utf-8")
    companion = tmp_path / "image.json"
    companion.write_text('{"description": "unverified data"}', encoding="utf-8")
    (tmp_path / CAPTION_INDEX_NAME).write_text(json.dumps({
        "kind": "caption_index", "schema_version": 1,
        "captions": {"image.png": "image.txt"}, "files": {"image.txt": "image.png"},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="[Oo]wnership|[Uu]nowned"):
        write_caption_output(source, {"description": "new caption"}, "image/png")
    assert (tmp_path / "image.txt").read_text(encoding="utf-8") == "old caption"
    assert json.loads(companion.read_text(encoding="utf-8"))["description"] == "unverified data"
