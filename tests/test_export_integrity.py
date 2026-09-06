from __future__ import annotations

import io
import hashlib
import json
from pathlib import Path

import lance
import pyarrow as pa
import pytest
from PIL import Image

from module.lanceexport import extract_from_lance
from utils.lance_blob import build_lance_schema, build_lance_value_array
from utils.output_writer import write_caption_output


def _dataset(root, records):
    schema = build_lance_schema([
        ("uris", pa.string()), ("mime", pa.string()),
        ("captions", pa.list_(pa.string())), ("blob", pa.large_binary()),
    ])
    arrays = [build_lance_value_array([row.get(field.name) for row in records], field) for field in schema]
    return lance.write_dataset(pa.Table.from_arrays(arrays, schema=schema), str(root / "test.lance"), data_storage_version="2.2")


def _png(color):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_portable_export_preserves_duplicate_leaf_names_and_captions(tmp_path):
    records = [
        {"uris": str(tmp_path / folder / "same.png"), "mime": "image/png", "captions": [caption], "blob": _png(color)}
        for folder, color, caption in [("a", "red", "red caption"), ("b", "blue", "blue caption")]
    ]
    dataset = _dataset(tmp_path, records)
    output = tmp_path / "output"
    extract_from_lance(dataset, str(output), clip_with_caption=False)
    images = sorted(output.rglob("*.png"))
    assert len(images) == 2
    exported = {}
    for path in images:
        with Image.open(path) as image:
            exported[image.getpixel((0, 0))] = path.with_suffix(".txt").read_text(encoding="utf-8").strip()
    assert exported == {(255, 0, 0): "red caption", (0, 0, 255): "blue caption"}
    names = [path.relative_to(output) for path in images]
    extract_from_lance(dataset, str(output), clip_with_caption=False)
    assert [path.relative_to(output) for path in sorted(output.rglob("*.png"))] == names


def test_separate_caption_directory_does_not_flatten_distinct_inputs(tmp_path):
    sources = [tmp_path / "a" / "same.png", tmp_path / "b" / "same.png"]
    for source in sources:
        source.parent.mkdir()
        source.write_bytes(_png("red"))
    dataset = _dataset(tmp_path, [
        {"uris": str(source), "mime": "image/png", "captions": [caption], "blob": None}
        for source, caption in zip(sources, ["first", "second"])
    ])
    output = tmp_path / "captions"
    extract_from_lance(dataset, str(tmp_path / "output"), caption_dir=str(output), clip_with_caption=False)
    assert {path.read_text(encoding="utf-8").strip() for path in output.rglob("*.txt")} == {"first", "second"}


@pytest.mark.parametrize("explicit_extension", [None, ".txt"])
def test_text_caption_export_never_replaces_primary_source(tmp_path, explicit_extension):
    source = tmp_path / "original.txt"
    source.write_text("ORIGINAL SOURCE", encoding="utf-8")
    dataset = _dataset(tmp_path, [{"uris": str(source), "mime": "text/plain", "captions": ["ANNOTATION"], "blob": None}])
    extract_from_lance(dataset, str(tmp_path / "out"), caption_extension=explicit_extension, clip_with_caption=False)
    assert source.read_text(encoding="utf-8") == "ORIGINAL SOURCE"
    assert (tmp_path / "original.caption.txt").read_text(encoding="utf-8").strip() == "ANNOTATION"


def test_export_preserves_other_primary_files_that_share_caption_stem(tmp_path):
    image = tmp_path / "sample.png"
    text = tmp_path / "sample.txt"
    image.write_bytes(_png("red"))
    text.write_text("PRIMARY TEXT", encoding="utf-8")
    dataset = _dataset(tmp_path, [
        {"uris": str(image), "mime": "image/png", "captions": ["IMAGE CAPTION"], "blob": None},
        {"uris": str(text), "mime": "text/plain", "captions": [], "blob": None},
    ])
    extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=False)
    assert text.read_text(encoding="utf-8") == "PRIMARY TEXT"
    assert any(path.read_text(encoding="utf-8").strip() == "IMAGE CAPTION" for path in tmp_path.glob("*.txt") if path != text)


def test_caption_writer_preserves_source_when_payload_requests_its_extension(tmp_path):
    source = tmp_path / "image.png"
    original = _png("red")
    source.write_bytes(original)
    target, _ = write_caption_output(source, {"description": "caption", "caption_extension": ".png"}, "image/png")
    assert source.read_bytes() == original
    assert target != source
    assert target.read_text(encoding="utf-8") == "caption"


@pytest.mark.parametrize("existing_sources", [False, True])
def test_export_rejects_hash_name_colliding_with_natural_name_before_writing(tmp_path, existing_sources):
    first = tmp_path / "a" / "same.png"
    digest = hashlib.sha256(str(first).encode("utf-8")).hexdigest()[:16]
    sources = [first, tmp_path / "b" / "same.png", tmp_path / "c" / f"same__{digest}.png"]
    if existing_sources:
        for source in sources:
            source.parent.mkdir(parents=True)
            source.write_bytes(_png("red"))
    dataset = _dataset(tmp_path, [
        {"uris": str(source), "mime": "image/png", "captions": [str(index)], "blob": _png("blue")}
        for index, source in enumerate(sources)
    ])
    output = tmp_path / "out"
    kwargs = {"caption_dir": str(output)} if existing_sources else {}
    with pytest.raises(ValueError, match="[Cc]ollision|[Aa]mbiguous"):
        extract_from_lance(dataset, str(output), clip_with_caption=False, **kwargs)
    assert not any(path.is_file() for path in output.rglob("*"))
    if existing_sources:
        assert all(source.read_bytes() == _png("red") for source in sources)


def test_export_rejects_caption_avoidance_colliding_with_another_caption(tmp_path):
    text = tmp_path / "report.txt"
    image = tmp_path / "report.caption.png"
    text.write_text("primary text", encoding="utf-8")
    image.write_bytes(_png("red"))
    dataset = _dataset(tmp_path, [
        {"uris": str(text), "mime": "text/plain", "captions": ["text annotation"], "blob": None},
        {"uris": str(image), "mime": "image/png", "captions": ["image annotation"], "blob": None},
    ])
    with pytest.raises(ValueError, match="[Cc]ollision|[Aa]mbiguous"):
        extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=False)
    assert text.read_text(encoding="utf-8") == "primary text"
    assert not (tmp_path / "report.caption.txt").exists()


def test_export_preflight_includes_structured_json_companion(tmp_path):
    text = tmp_path / "report.json"
    image = tmp_path / "report.caption.caption.png"
    text.write_text("{}", encoding="utf-8")
    image.write_bytes(_png("red"))
    dataset = _dataset(tmp_path, [
        {"uris": str(text), "mime": "text/plain", "captions": ['{"description": "text annotation"}'], "blob": None},
        {"uris": str(image), "mime": "image/png", "captions": ["image annotation"], "blob": None},
    ])
    with pytest.raises(ValueError, match="[Cc]ollision|[Aa]mbiguous"):
        extract_from_lance(dataset, str(tmp_path / "out"), caption_extension=".json", clip_with_caption=False)
    assert text.read_text(encoding="utf-8") == "{}"
    assert not (tmp_path / "report.caption.json").exists()


def test_clip_uses_exported_media_and_actual_caption_directory(tmp_path, monkeypatch):
    import module.lanceexport as exporter

    source = tmp_path / "missing" / "video.mp4"
    dataset = _dataset(tmp_path, [{"uris": str(source), "mime": "video/mp4", "blob": b"video",
                                  "captions": ["1\n00:00:00,000 --> 00:00:01,000\nhello\n"]}])
    calls = []
    monkeypatch.setattr(exporter, "split_video_with_imageio_ffmpeg",
                        lambda media, subs, writer: calls.append((Path(media), subs[0].text)))
    output = tmp_path / "out"
    extract_from_lance(dataset, str(output), caption_dir=str(tmp_path / "captions"), clip_with_caption=True)
    assert calls == [(output / "video.mp4", "hello")]


def test_failed_caption_save_does_not_clip_with_stale_subtitles(tmp_path, monkeypatch):
    import module.lanceexport as exporter

    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    source.with_suffix(".srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nstale\n", encoding="utf-8")
    dataset = _dataset(tmp_path, [{"uris": str(source), "mime": "video/mp4", "blob": None,
                                  "captions": ["new captions"]}])
    monkeypatch.setattr(exporter, "save_caption", lambda *args, **kwargs: False)
    calls = []
    monkeypatch.setattr(exporter, "split_video_with_imageio_ffmpeg", lambda *args: calls.append(args))
    extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=True)
    assert calls == []


def test_clip_association_survives_portable_duplicate_basename_renaming(tmp_path, monkeypatch):
    import module.lanceexport as exporter

    dataset = _dataset(tmp_path, [
        {"uris": str(tmp_path / folder / "video.mp4"), "mime": "video/mp4", "blob": label.encode(),
         "captions": [f"1\n00:00:00,000 --> 00:00:01,000\n{label}\n"]}
        for folder, label in [("a", "first"), ("b", "second")]
    ])
    calls = []
    monkeypatch.setattr(exporter, "split_video_with_imageio_ffmpeg",
                        lambda media, subs, writer: calls.append((Path(media).read_bytes().decode(), subs[0].text)))
    extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=True)
    assert calls == [("first", "first"), ("second", "second")]


def test_export_does_not_attempt_to_clip_an_unavailable_primary(tmp_path, monkeypatch):
    import module.lanceexport as exporter

    dataset = _dataset(tmp_path, [{"uris": str(tmp_path / "missing" / "video.mp4"), "mime": "video/mp4", "blob": None,
                                  "captions": ["1\n00:00:00,000 --> 00:00:01,000\nhello\n"]}])
    calls = []
    monkeypatch.setattr(exporter, "split_video_with_imageio_ffmpeg", lambda *args: calls.append(args))
    extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=True)
    assert calls == []


def test_export_rejects_duplicate_uris_before_writing(tmp_path):
    record = {"uris": str(tmp_path / "missing.png"), "mime": "image/png", "blob": _png("red"), "captions": ["caption"]}
    dataset = _dataset(tmp_path, [record, record])
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="duplicate source URIs"):
        extract_from_lance(dataset, str(output), clip_with_caption=False)
    assert not output.exists()


def test_export_rejects_shared_clip_namespace_only_when_clipping(tmp_path, monkeypatch):
    import module.lanceexport as exporter

    sources = [tmp_path / "video.mp4", tmp_path / "video.mkv"]
    for source in sources:
        source.write_bytes(b"video")
    dataset = _dataset(tmp_path, [
        {"uris": str(source), "mime": "video/mp4", "blob": None,
         "captions": ["1\n00:00:00,000 --> 00:00:01,000\nhello\n"]}
        for source in sources
    ])
    monkeypatch.setattr(exporter, "split_video_with_imageio_ffmpeg", lambda *args: None)
    with pytest.raises(ValueError, match="[Cc]ollision|[Aa]mbiguous"):
        extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=True)
    assert not list(tmp_path.glob("*.srt"))
    extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=False)
    assert len(list(tmp_path.glob("*.srt"))) == 2


def test_export_clip_namespace_cannot_contain_dataset_primary(tmp_path, monkeypatch):
    import module.lanceexport as exporter

    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    primary = tmp_path / "video_clip" / "video_1.mp4"
    primary.parent.mkdir()
    primary.write_bytes(b"primary")
    dataset = _dataset(tmp_path, [
        {"uris": str(source), "mime": "video/mp4", "blob": None,
         "captions": ["1\n00:00:00,000 --> 00:00:01,000\nhello\n"]},
        {"uris": str(primary), "mime": "video/mp4", "blob": None, "captions": []},
    ])
    monkeypatch.setattr(exporter, "split_video_with_imageio_ffmpeg", lambda *args: None)
    with pytest.raises(ValueError, match="[Cc]ollision|[Aa]mbiguous"):
        extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=True)
    assert not source.with_suffix(".srt").exists()
    assert primary.read_bytes() == b"primary"


def test_export_page_namespace_cannot_contain_dataset_primary(tmp_path):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"document")
    primary = tmp_path / "report" / "report_1.md"
    primary.parent.mkdir()
    primary.write_text("primary", encoding="utf-8")
    page = '<header style="background-color: #f5f5f5;"><strong> Page 1 </strong></header>changed'
    dataset = _dataset(tmp_path, [
        {"uris": str(source), "mime": "application/pdf", "blob": None, "captions": [page]},
        {"uris": str(primary), "mime": "text/markdown", "blob": None, "captions": []},
    ])
    with pytest.raises(ValueError, match="[Cc]ollision|[Aa]mbiguous"):
        extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=True)
    assert not source.with_suffix(".md").exists()
    assert primary.read_text(encoding="utf-8") == "primary"


@pytest.mark.parametrize("external_captions", [False, True])
def test_exported_duplicate_names_roundtrip_through_directory_import(tmp_path, external_captions):
    from module.lanceImport import load_data

    sources = tmp_path / "sources"
    paths = [sources / "a" / "same.png", sources / "b" / "same.png"] if external_captions else [sources / "same.png", sources / "same.jpg"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_png("red"))
    dataset = _dataset(tmp_path, [
        {"uris": str(path), "mime": "image/png", "captions": [caption], "blob": None}
        for path, caption in zip(paths, ["first caption", "second caption"])
    ])
    caption_dir = tmp_path / "captions" if external_captions else None
    extract_from_lance(dataset, str(tmp_path / "out"), caption_dir=str(caption_dir) if caption_dir else None,
                       clip_with_caption=False)
    items = load_data(str(sources), texts_dir=str(caption_dir) if caption_dir else None)
    assert {Path(item["file_path"]): item["caption"] for item in items} == {
        paths[0]: ["first caption"], paths[1]: ["second caption"],
    }
    if not external_captions:
        moved = tmp_path / "moved"
        sources.rename(moved)
        items = load_data(str(moved))
        assert {Path(item["file_path"]).name: item["caption"] for item in items} == {
            "same.png": ["first caption"], "same.jpg": ["second caption"],
        }


def test_exported_text_annotation_is_not_imported_as_an_extra_document(tmp_path):
    from module.lanceImport import load_data

    sources = tmp_path / "sources"
    sources.mkdir()
    source = sources / "original.txt"
    source.write_text("source body", encoding="utf-8")
    dataset = _dataset(tmp_path, [{"uris": str(source), "mime": "text/plain", "captions": ["annotation"], "blob": None}])
    extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=False)
    items = load_data(str(sources))
    assert [(Path(item["file_path"]), item["caption"]) for item in items] == [(source, ["annotation"])]
    assert source.read_text(encoding="utf-8") == "source body"


def test_export_rejects_unowned_caption_index_before_writing(tmp_path):
    source = tmp_path / "sample.png"
    source.write_bytes(_png("red"))
    index = tmp_path / ".qinglong-captions.json"
    index.write_text("user content", encoding="utf-8")
    dataset = _dataset(tmp_path, [{"uris": str(source), "mime": "image/png", "captions": ["annotation"], "blob": None}])
    with pytest.raises(ValueError, match="[Ii]ndex"):
        extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=False)
    assert not source.with_suffix(".txt").exists()
    assert index.read_text(encoding="utf-8") == "user content"


def test_export_commits_one_caption_index_per_directory(tmp_path, monkeypatch):
    import module.lanceexport as exporter

    paths = [tmp_path / f"image{index}.png" for index in range(3)]
    for path in paths:
        path.write_bytes(_png("red"))
    dataset = _dataset(tmp_path, [
        {"uris": str(path), "mime": "image/png", "captions": ["annotation"], "blob": None}
        for path in paths
    ])
    actual_writer = exporter.write_caption_index
    commits = []

    def record_commit(directory, entries):
        commits.append(directory)
        actual_writer(directory, entries)

    monkeypatch.setattr(exporter, "write_caption_index", record_commit)
    extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=False)
    assert commits == [tmp_path]


def test_reexport_keeps_old_annotations_owned_without_deleting_them(tmp_path):
    from module.lanceImport import load_data

    sources = tmp_path / "sources"
    sources.mkdir()
    paths = [sources / "same.png", sources / "same.jpg"]
    for path in paths:
        path.write_bytes(_png("red"))
    dataset = _dataset(tmp_path, [
        {"uris": str(path), "mime": "image/png", "captions": [caption], "blob": None}
        for path, caption in zip(paths, ["first", "second"])
    ])
    extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=False)
    previous = {path: path.read_bytes() for path in sources.glob("*.txt")}
    extract_from_lance(dataset, str(tmp_path / "out"), caption_suffix="_zh", clip_with_caption=False)
    assert all(path.read_bytes() == content for path, content in previous.items())
    assert {Path(item["file_path"]): item["caption"] for item in load_data(str(sources))} == {
        paths[0]: ["first"], paths[1]: ["second"],
    }


@pytest.mark.parametrize("entries", [{"a.png": "a.png"}, {"a.png": "shared.txt", "b.png": "shared.txt"}])
def test_invalid_index_cannot_hide_sources_or_share_annotations(tmp_path, entries):
    from module.lanceImport import load_data

    for name in ["a", "b"]:
        (tmp_path / f"{name}.png").write_bytes(_png("red"))
        (tmp_path / f"{name}.txt").write_text(name, encoding="utf-8")
    (tmp_path / "shared.txt").write_text("wrong", encoding="utf-8")
    index = tmp_path / ".qinglong-captions.json"
    index.write_text(json.dumps({"kind": "caption_index", "schema_version": 1, "captions": entries}), encoding="utf-8")
    items = load_data(str(tmp_path))
    assert {Path(item["file_path"]).name: item["caption"] for item in items if Path(item["file_path"]).suffix == ".png"} == {
        "a.png": ["a"], "b.png": ["b"],
    }


def test_missing_index_target_falls_back_to_existing_legacy_caption(tmp_path):
    from module.lanceImport import load_data

    (tmp_path / "a.png").write_bytes(_png("red"))
    (tmp_path / "a.txt").write_text("legacy annotation", encoding="utf-8")
    (tmp_path / ".qinglong-captions.json").write_text(json.dumps({
        "kind": "caption_index", "schema_version": 1, "captions": {"a.png": "missing.txt"},
    }), encoding="utf-8")
    items = load_data(str(tmp_path))
    assert [(Path(item["file_path"]).name, item["caption"]) for item in items] == [("a.png", ["legacy annotation"])]


def test_caption_index_preserves_valid_leading_spaces_in_filenames(tmp_path):
    from module.lanceImport import load_data

    source = tmp_path / " leading space.png"
    source.write_bytes(_png("red"))
    dataset = _dataset(tmp_path, [{"uris": str(source), "mime": "image/png", "captions": ["annotation"], "blob": None}])
    for _ in range(2):
        extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=False)
    images = [item for item in load_data(str(tmp_path)) if Path(item["file_path"]).suffix == ".png"]
    assert [(Path(item["file_path"]), item["caption"]) for item in images] == [(source, ["annotation"])]
