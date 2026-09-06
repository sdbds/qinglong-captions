from __future__ import annotations

import json
from pathlib import Path

import pytest

from module import lanceexport
from module.lanceImport import load_data
from tests.test_export_integrity import _dataset, _png
from utils import output_writer
from utils.caption_index import CAPTION_INDEX_NAME, CaptionFileTransaction, load_caption_index


def _portable_dataset(root, source, description, *, with_caption=True, color="red"):
    root.mkdir()
    return _dataset(root, [{
        "uris": str(source), "mime": "image/png", "blob": _png(color),
        "captions": [json.dumps({"description": description})] if with_caption else [],
    }])


def _files(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


@pytest.mark.parametrize("separate_captions", [False, True])
@pytest.mark.parametrize("with_caption", [False, True])
def test_portable_export_rejects_different_origin_before_overwriting(tmp_path, separate_captions, with_caption):
    output = tmp_path / "out"
    captions = tmp_path / "captions" if separate_captions else None
    sources = [tmp_path / name / "missing" / "same.png" for name in ("first", "second")]
    datasets = [_portable_dataset(tmp_path / name, source, name, with_caption=with_caption)
                for name, source in zip(("first", "second"), sources)]
    options = {"clip_with_caption": False, "caption_dir": str(captions) if captions else None}
    lanceexport.extract_from_lance(datasets[0], str(output), **options)
    before = _files(output), _files(captions) if captions else {}

    with pytest.raises(ValueError, match="source|ownership|origin"):
        lanceexport.extract_from_lance(datasets[1], str(output), **options)

    assert (_files(output), _files(captions) if captions else {}) == before


def test_portable_export_retains_local_import_alias_and_same_origin_updates(tmp_path):
    output = tmp_path / "out"
    source = tmp_path / "missing" / "same.png"
    for name in ("first", "second"):
        dataset = _portable_dataset(tmp_path / name, source, name)
        lanceexport.extract_from_lance(dataset, str(output), clip_with_caption=False)
        assert [(Path(row["file_path"]).name, row["caption"]) for row in load_data(str(output))] == [
            ("same.png", [name]),
        ]

    # The restored primary can be captioned locally without claiming a new origin.
    output_writer.write_caption_output(output / "same.png", {"description": "local"}, "image/png")
    assert load_data(str(output))[0]["caption"] == ["local"]
    lanceexport.extract_from_lance(dataset, str(output), clip_with_caption=False)
    assert load_data(str(output))[0]["caption"] == ["second"]


def test_restored_source_in_output_directory_can_be_restored_again(tmp_path):
    output, captions = tmp_path / "out", tmp_path / "captions"
    source = output / "same.png"
    dataset = _portable_dataset(tmp_path / "dataset", source, "caption")
    for _ in range(2):
        lanceexport.extract_from_lance(dataset, str(output), clip_with_caption=False, caption_dir=str(captions))
        assert load_data(str(output), texts_dir=str(captions))[0]["caption"] == ["caption"]
        source.unlink()


def test_portable_export_does_not_guess_origin_of_legacy_index(tmp_path):
    output = tmp_path / "out"
    source = tmp_path / "missing" / "same.png"
    dataset = _portable_dataset(tmp_path / "dataset", source, "caption")
    lanceexport.extract_from_lance(dataset, str(output), clip_with_caption=False)
    index_path = output / CAPTION_INDEX_NAME
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload.pop("origins", None)
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    before = _files(output)

    with pytest.raises(ValueError, match="origin|source"):
        lanceexport.extract_from_lance(dataset, str(output), clip_with_caption=False)

    assert _files(output) == before


@pytest.mark.parametrize("origin", ["C:/other-host/source.png", "/other-host/source.png"])
def test_foreign_platform_origin_does_not_break_local_import(tmp_path, origin):
    output = tmp_path / "out"
    dataset = _portable_dataset(tmp_path / "dataset", tmp_path / "missing.png", "caption")
    lanceexport.extract_from_lance(dataset, str(output), clip_with_caption=False)
    index_path = output / CAPTION_INDEX_NAME
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["origins"] = {"missing.png": origin}
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_caption_index(output, strict=True) is not None
    assert load_data(str(output))[0]["caption"] == ["caption"]


def test_missing_retained_caption_cannot_be_reassigned_to_portable_primary(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    image = output / "same.png"
    image.write_bytes(_png("red"))
    output_writer.write_caption_output(image, {"description": "caption"}, "image/png")
    image.with_suffix(".txt").unlink()
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    dataset = _dataset(dataset_root, [{"uris": str(tmp_path / "missing" / "same.txt"),
                                       "mime": "text/plain", "blob": b"new primary", "captions": []}])
    before = _files(output)

    with pytest.raises(ValueError, match="source|ownership|primary"):
        lanceexport.extract_from_lance(dataset, str(output), clip_with_caption=False)

    assert _files(output) == before
    assert load_caption_index(output, strict=True) is not None


def test_missing_portable_primary_cannot_be_reassigned_to_caption(tmp_path):
    output = tmp_path / "out"
    dataset = _portable_dataset(tmp_path / "dataset", tmp_path / "missing" / "same.png", "", with_caption=False)
    lanceexport.extract_from_lance(dataset, str(output), clip_with_caption=False)
    (output / "same.png").unlink()
    image = output / "same.jpg"
    image.write_bytes(_png("red"))
    before = _files(output)

    with pytest.raises(ValueError, match="source|ownership|primary"):
        output_writer.write_caption_output(image, {"description": "caption", "caption_extension": ".png"}, "image/png")

    assert _files(output) == before
    assert load_caption_index(output, strict=True) is not None


@pytest.mark.parametrize("writer", ["generation", "export"])
@pytest.mark.parametrize("previous_success", [False, True])
def test_structured_index_failure_rolls_back_and_retry_succeeds(tmp_path, monkeypatch, writer, previous_success):
    source = tmp_path / "image.png"
    source.write_bytes(_png("red"))
    datasets = {}

    def write(description):
        if writer == "generation":
            return output_writer.write_caption_output(source, {"description": description}, "image/png")
        if description not in datasets:
            dataset_root = tmp_path / description
            dataset_root.mkdir()
            datasets[description] = _dataset(dataset_root, [{
                "uris": str(source), "mime": "image/png", "blob": None,
                "captions": [json.dumps({"description": description})],
            }])
        return lanceexport.extract_from_lance(datasets[description], str(tmp_path / "out"), clip_with_caption=False)

    if previous_success:
        write("previous")
    paths = [source.with_suffix(".txt"), source.with_suffix(".json"), tmp_path / CAPTION_INDEX_NAME]
    before = {path: path.read_bytes() for path in paths if path.exists()}

    def fail_commit(*args):
        raise OSError("simulated index commit failure")

    with monkeypatch.context() as context:
        context.setattr(output_writer if writer == "generation" else lanceexport, "write_caption_index", fail_commit)
        with pytest.raises(OSError, match="simulated index commit"):
            write("replacement")

    assert {path: path.read_bytes() for path in paths if path.exists()} == before
    write("replacement")
    assert source.with_suffix(".txt").read_text(encoding="utf-8").strip() == "replacement"
    assert json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))["description"] == "replacement"
    assert load_caption_index(tmp_path, strict=True).files["image.json"] == "image.png"


def test_portable_export_rolls_back_media_and_both_indexes_on_caption_index_failure(tmp_path, monkeypatch):
    output, captions = tmp_path / "out", tmp_path / "captions"
    source = tmp_path / "missing" / "same.png"
    dataset = _portable_dataset(tmp_path / "dataset", source, "caption")
    original_writer = lanceexport.write_caption_index

    def fail_caption_index(directory, index):
        if directory == captions.resolve():
            raise OSError("simulated caption index failure")
        original_writer(directory, index)

    with monkeypatch.context() as context:
        context.setattr(lanceexport, "write_caption_index", fail_caption_index)
        with pytest.raises(OSError, match="simulated caption index"):
            lanceexport.extract_from_lance(dataset, str(output), clip_with_caption=False, caption_dir=str(captions))

    assert not (output / "same.png").exists()
    assert not (output / CAPTION_INDEX_NAME).exists()
    assert not (captions / "same.json").exists()
    assert not (captions / CAPTION_INDEX_NAME).exists()
    lanceexport.extract_from_lance(dataset, str(output), clip_with_caption=False, caption_dir=str(captions))
    assert load_data(str(output), texts_dir=str(captions))[0]["caption"] == ["caption"]


def test_failed_caption_writer_does_not_leave_unowned_structured_files(tmp_path, monkeypatch):
    source = tmp_path / "image.png"
    source.write_bytes(_png("red"))
    dataset = _dataset(tmp_path, [{"uris": str(source), "mime": "image/png", "blob": None,
                                   "captions": [json.dumps({"description": "caption"})]}])

    def incomplete_write(*args, **kwargs):
        source.with_suffix(".json").write_text("{}", encoding="utf-8")
        source.with_suffix(".txt").write_text("partial", encoding="utf-8")
        return False

    with monkeypatch.context() as context:
        context.setattr(lanceexport, "save_caption", incomplete_write)
        lanceexport.extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=False)

    assert not source.with_suffix(".json").exists()
    assert not source.with_suffix(".txt").exists()
    lanceexport.extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=False)
    assert source.with_suffix(".txt").read_text(encoding="utf-8") == "caption"


@pytest.mark.parametrize("previous_success", [False, True])
def test_failed_caption_also_restores_portable_media(tmp_path, monkeypatch, previous_success):
    output = tmp_path / "out"
    source = tmp_path / "missing.png"
    if previous_success:
        previous = _portable_dataset(tmp_path / "previous", source, "previous")
        lanceexport.extract_from_lance(previous, str(output), clip_with_caption=False)
    replacement = _portable_dataset(tmp_path / "replacement", source, "replacement", color="blue")
    paths = [output / name for name in ("missing.png", "missing.txt", "missing.json", CAPTION_INDEX_NAME)]
    before = {path: path.read_bytes() for path in paths if path.exists()}

    def incomplete_write(*args, **kwargs):
        (output / "missing.json").write_text("{}", encoding="utf-8")
        return False

    with monkeypatch.context() as context:
        context.setattr(lanceexport, "save_caption", incomplete_write)
        lanceexport.extract_from_lance(replacement, str(output), clip_with_caption=False)

    assert {path: path.read_bytes() for path in paths if path.exists()} == before
    lanceexport.extract_from_lance(replacement, str(output), clip_with_caption=False)
    assert (output / "missing.png").read_bytes() == _png("blue")
    assert load_data(str(output))[0]["caption"] == ["replacement"]


def test_failed_index_commit_does_not_publish_document_pages(tmp_path, monkeypatch):
    source = tmp_path / "document.pdf"
    source.write_bytes(b"source")
    markdown = '<header style="background-color: #f5f5f5;"><strong> Page 1 </strong></header>\nPage body'
    dataset = _dataset(tmp_path, [{"uris": str(source), "mime": "application/pdf", "blob": None,
                                   "captions": [markdown]}])

    def fail_commit(*args):
        raise OSError("simulated index failure")

    with monkeypatch.context() as context:
        context.setattr(lanceexport, "write_caption_index", fail_commit)
        with pytest.raises(OSError, match="simulated index"):
            lanceexport.extract_from_lance(dataset, str(tmp_path / "out"))

    assert not source.with_suffix("").exists()
    lanceexport.extract_from_lance(dataset, str(tmp_path / "out"))
    assert (source.with_suffix("") / "document_1.md").is_file()


def test_mixed_json_write_failure_is_rolled_back(tmp_path, monkeypatch):
    source = tmp_path / "image.png"
    source.write_bytes(_png("red"))
    output_writer.write_caption_output(source, {"description": "previous"}, "image/png")
    paths = [source.with_suffix(".txt"), source.with_suffix(".json"), tmp_path / CAPTION_INDEX_NAME]
    before = {path: path.read_bytes() for path in paths}
    dataset = _dataset(tmp_path, [{"uris": str(source), "mime": "image/png", "blob": None,
                                   "captions": ["before", json.dumps({"description": "replacement"}), "after"]}])

    def fail_json_write(*args, **kwargs):
        raise OSError("simulated JSON write failure")

    with monkeypatch.context() as context:
        context.setattr(lanceexport.json, "dump", fail_json_write)
        lanceexport.extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=False)

    assert {path: path.read_bytes() for path in paths} == before
    lanceexport.extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=False)
    assert json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))["description"] == "replacement"
    assert "before" in source.with_suffix(".txt").read_text(encoding="utf-8")


def test_private_publication_backups_do_not_become_input_assets(tmp_path):
    source = tmp_path / "image.png"
    source.write_bytes(_png("red"))
    caption = source.with_suffix(".txt")
    caption.write_text("legacy caption", encoding="utf-8")

    with CaptionFileTransaction(tmp_path) as publication:
        publication.watch(caption)
        assert [(Path(row["file_path"]), row["caption"]) for row in load_data(str(tmp_path))] == [
            (source, ["legacy caption"]),
        ]
