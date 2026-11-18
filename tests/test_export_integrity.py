from __future__ import annotations

import io
import hashlib
import json
import sys
from pathlib import Path

import lance
import pyarrow as pa
import pytest
from PIL import Image

from module.lanceexport import extract_from_lance
from utils.lance_blob import build_lance_schema, build_lance_value_array
from utils.output_writer import write_caption_output


def _dataset(root, records):
    fields = [
        ("uris", pa.string()), ("mime", pa.string()),
        ("captions", pa.list_(pa.string())), ("blob", pa.large_binary()),
    ]
    if any("hash" in row for row in records):
        fields.append(("hash", pa.string()))
    schema = build_lance_schema(fields)
    arrays = [build_lance_value_array([row.get(field.name) for row in records], field) for field in schema]
    return lance.write_dataset(pa.Table.from_arrays(arrays, schema=schema), str(root / "test.lance"), data_storage_version="2.2")


def _png(color):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_standalone_export_uses_dedicated_default_directory(tmp_path, monkeypatch):
    from module.lanceexport import main

    source = tmp_path / "sample.png"
    source.write_bytes(_png("red"))
    dataset = _dataset(tmp_path, [{"uris": str(source), "mime": "image/png", "captions": ["exported"]}])
    dataset.tags.create("gemini", dataset.version)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["lanceexport", dataset.uri, "--not_clip_with_caption"])

    main()

    assert (tmp_path / "exports" / "sample.png").read_bytes() == _png("red")
    assert (tmp_path / "exports" / "sample.txt").read_text(encoding="utf-8").strip() == "exported"
    assert not source.with_suffix(".txt").exists()


@pytest.mark.parametrize("stored_blob", [False, True])
def test_default_export_leaves_existing_source_and_sidecars_untouched(tmp_path, stored_blob):
    originals = tmp_path / "originals"
    originals.mkdir()
    source = originals / "sample.png"
    source.write_bytes(_png("red"))
    source.with_suffix(".txt").write_text("original caption", encoding="utf-8")
    source.with_suffix(".json").write_text("original metadata", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in originals.iterdir()}
    dataset = _dataset(tmp_path, [{
        "uris": str(source), "mime": "image/png", "blob": _png("blue") if stored_blob else None,
        "captions": ['{"description": "exported caption"}'],
    }])
    output = tmp_path / "out"
    for _ in range(2):
        extract_from_lance(dataset, str(output), clip_with_caption=False)
        assert (output / "sample.png").read_bytes() == _png("blue" if stored_blob else "red")
        assert (output / "sample.txt").read_text(encoding="utf-8") == "exported caption"
        assert json.loads((output / "sample.json").read_text(encoding="utf-8")) == {
            "description": "exported caption",
        }
        assert {path.name: path.read_bytes() for path in originals.iterdir()} == before


def test_default_export_does_not_overwrite_existing_unowned_primary(tmp_path):
    source = tmp_path / "sample.png"
    source.write_bytes(_png("red"))
    output = tmp_path / "out"
    output.mkdir()
    (output / "sample.png").write_bytes(_png("blue"))
    dataset = _dataset(tmp_path, [{"uris": str(source), "mime": "image/png", "captions": ["exported"]}])

    with pytest.raises(ValueError, match="origin|ownership"):
        extract_from_lance(dataset, str(output), clip_with_caption=False)

    assert (output / "sample.png").read_bytes() == _png("blue")
    assert not (output / "sample.txt").exists()
    assert not source.with_suffix(".txt").exists()


@pytest.mark.parametrize("restore_original", [False, True])
@pytest.mark.parametrize("separate_captions", [False, True])
def test_caption_only_export_never_adopts_a_later_unowned_primary(tmp_path, restore_original, separate_captions):
    source = tmp_path / "sample.png"
    output = tmp_path / "out"
    captions = tmp_path / "captions" if separate_captions else output
    dataset = _dataset(tmp_path, [{"uris": str(source), "mime": "image/png", "captions": ["annotation"]}])
    options = {"caption_dir": str(captions), "clip_with_caption": False}
    for _ in range(2):
        extract_from_lance(dataset, str(output), **options)
    assert (captions / "sample.txt").read_text(encoding="utf-8").strip() == "annotation"
    target = output / "sample.png"
    assert not target.exists()
    target.write_bytes(_png("blue"))
    if restore_original:
        source.write_bytes(_png("red"))

    with pytest.raises(ValueError, match="origin|ownership|primary"):
        extract_from_lance(dataset, str(output), **options)

    assert target.read_bytes() == _png("blue")


@pytest.mark.parametrize("legacy_index", [False, True])
def test_export_requires_evidence_for_the_current_primary_bytes(tmp_path, legacy_index):
    from utils.caption_index import CAPTION_INDEX_NAME

    source = tmp_path / "missing.png"
    output = tmp_path / "out"
    dataset = _dataset(tmp_path, [{"uris": str(source), "mime": "image/png", "blob": _png("red"),
                                   "captions": ["annotation"]}])
    extract_from_lance(dataset, str(output), clip_with_caption=False)
    index_path = output / CAPTION_INDEX_NAME
    if legacy_index:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index_path.write_text(json.dumps({key: value for key, value in index.items()
                                         if key in {"kind", "schema_version", "captions", "files", "origins"}}),
                              encoding="utf-8")
    else:
        (output / "missing.png").write_bytes(_png("blue"))
    expected = (output / "missing.png").read_bytes()
    before_index = index_path.read_bytes()

    with pytest.raises(ValueError, match="origin|ownership|primary"):
        extract_from_lance(dataset, str(output), clip_with_caption=False)

    assert (output / "missing.png").read_bytes() == expected
    assert index_path.read_bytes() == before_index


def test_blobless_export_rejects_an_original_that_no_longer_matches_its_row(tmp_path):
    source = tmp_path / "sample.png"
    source.write_bytes(_png("red"))
    output = tmp_path / "out"
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    dataset = _dataset(tmp_path, [{"uris": str(source), "mime": "image/png", "hash": expected_hash,
                                   "captions": ["annotation"]}])
    extract_from_lance(dataset, str(output), clip_with_caption=False)
    before = {file.name: file.read_bytes() for file in output.iterdir() if file.is_file()}
    source.write_bytes(_png("blue"))

    with pytest.raises(ValueError, match="hash|digest|snapshot"):
        extract_from_lance(dataset, str(output), clip_with_caption=False)

    assert {file.name: file.read_bytes() for file in output.iterdir() if file.is_file()} == before
    assert source.read_bytes() == _png("blue")


@pytest.mark.parametrize("separate_captions", [False, True])
def test_default_export_does_not_claim_an_unowned_caption(tmp_path, separate_captions):
    source = tmp_path / "sample.png"
    source.write_bytes(_png("red"))
    output = tmp_path / "out"
    captions = tmp_path / "captions" if separate_captions else output
    options = {"clip_with_caption": False, "caption_dir": str(captions) if separate_captions else None}
    dataset = _dataset(tmp_path, [{"uris": str(source), "mime": "image/png", "captions": []}])
    extract_from_lance(dataset, str(output), **options)
    caption = captions / "sample.txt"
    caption.write_text("unrelated text", encoding="utf-8")
    new_root = tmp_path / "annotated"
    new_root.mkdir()
    annotated = _dataset(new_root, [{"uris": str(source), "mime": "image/png", "captions": ["exported"]}])

    with pytest.raises(ValueError, match="unowned|ownership|origin"):
        extract_from_lance(annotated, str(output), **options)

    assert caption.read_text(encoding="utf-8") == "unrelated text"
    assert not source.with_suffix(".txt").exists()


def test_selected_in_place_rows_keep_full_dataset_collision_protection(tmp_path):
    source = tmp_path / "sample.png"
    skipped = tmp_path / "sample.jpg"
    missing = tmp_path / "missing.png"
    source.write_bytes(_png("red"))
    skipped.write_bytes(_png("blue"))
    old_caption = skipped.with_suffix(".txt")
    old_caption.write_text("untouched", encoding="utf-8")
    dataset = _dataset(tmp_path, [
        {"uris": str(source), "mime": "image/png", "captions": ["selected"]},
        {"uris": str(skipped), "mime": "image/jpeg", "captions": ["do not publish"]},
        {"uris": str(missing), "mime": "image/png", "blob": _png("blue"), "captions": ["do not restore"]},
    ])
    output = tmp_path / "out"

    extract_from_lance(dataset, str(output), clip_with_caption=False, in_place=True, include_uris={str(source)})

    assert old_caption.read_text(encoding="utf-8") == "untouched"
    assert [path.read_text(encoding="utf-8").strip() for path in tmp_path.glob("sample__*.txt")] == ["selected"]
    assert not list(output.glob("*"))


@pytest.mark.parametrize("existing_pages", [False, True])
def test_default_export_keeps_existing_derived_pages_untouched_on_rerun(tmp_path, existing_pages):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"source document")
    output = tmp_path / "out"
    page = output / "report" / "report_1.md"
    if existing_pages:
        page.parent.mkdir(parents=True)
        page.write_text("existing document", encoding="utf-8")
    dataset = _dataset(tmp_path, [{
        "uris": str(source), "mime": "application/pdf",
        "captions": ['<header style="background-color: #f5f5f5;"><strong> Page 1 </strong></header>Page body'],
    }])

    if existing_pages:
        with pytest.raises(ValueError, match="derived|page"):
            extract_from_lance(dataset, str(output))
        assert page.read_text(encoding="utf-8") == "existing document"
        assert not (output / "report.md").exists()
    else:
        for _ in range(2):
            extract_from_lance(dataset, str(output))
        assert page.read_text(encoding="utf-8") == "Page body"
        assert (output / "report.md").is_file()
    assert not source.with_suffix(".md").exists()


@pytest.mark.parametrize("stored_hash", [False, True])
def test_blob_backed_pages_reuse_the_snapshot_after_original_changes_or_disappears(tmp_path, stored_hash):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"earlier local document")
    blob = b"selected document snapshot"
    record = {"uris": str(source), "mime": "application/pdf", "blob": blob,
              "captions": ['<header style="background-color: #f5f5f5;"><strong> Page 1 </strong></header>text']}
    if stored_hash:
        record["hash"] = hashlib.sha256(blob).hexdigest()
    dataset = _dataset(tmp_path, [record])
    output = tmp_path / "out"
    extract_from_lance(dataset, str(output))
    page = output / "report" / "report_1.md"
    first_mtime = page.stat().st_mtime_ns

    source.write_bytes(b"edited local document of a different length")
    extract_from_lance(dataset, str(output))
    source.unlink()
    extract_from_lance(dataset, str(output))

    assert (output / "report.pdf").read_bytes() == blob
    assert page.read_text(encoding="utf-8") == "text"
    assert page.stat().st_mtime_ns == first_mtime


def test_default_export_rejects_changed_derived_inputs_without_replacing_outputs(tmp_path):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"source document")
    output = tmp_path / "out"
    page = '<header style="background-color: #f5f5f5;"><strong> Page 1 </strong></header>'
    dataset = _dataset(tmp_path, [{
        "uris": str(source), "mime": "application/pdf", "captions": [page + "first"],
    }])
    extract_from_lance(dataset, str(output))
    source.write_bytes(b"changed document")

    with pytest.raises(ValueError, match="derived"):
        extract_from_lance(dataset, str(output))

    assert (output / "report.pdf").read_bytes() == b"source document"
    assert (output / "report.md").read_text(encoding="utf-8") == page + "first"
    assert (output / "report" / "report_1.md").read_text(encoding="utf-8") == "first"


def test_default_export_does_not_claim_an_edited_derived_output(tmp_path):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"source document")
    output = tmp_path / "out"
    dataset = _dataset(tmp_path, [{
        "uris": str(source), "mime": "application/pdf",
        "captions": ['<header style="background-color: #f5f5f5;"><strong> Page 1 </strong></header>Page body'],
    }])
    extract_from_lance(dataset, str(output))
    page = output / "report" / "report_1.md"
    page.write_text("local edit", encoding="utf-8")

    with pytest.raises(ValueError, match="derived"):
        extract_from_lance(dataset, str(output))

    assert page.read_text(encoding="utf-8") == "local edit"


@pytest.mark.parametrize("existing_sources", [False, True])
def test_portable_export_preserves_duplicate_leaf_names_and_captions(tmp_path, existing_sources):
    records = [
        {"uris": str(tmp_path / folder / "same.png"), "mime": "image/png", "captions": [caption], "blob": _png(color)}
        for folder, color, caption in [("a", "red", "red caption"), ("b", "blue", "blue caption")]
    ]
    if existing_sources:
        for record in records:
            primary = Path(record["uris"])
            primary.parent.mkdir()
            primary.write_bytes(record["blob"])
            record["blob"] = None
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
    extract_from_lance(dataset, str(tmp_path / "out"), caption_extension=explicit_extension, clip_with_caption=False,
                       in_place=True)
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
    extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=False, in_place=True)
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
    with pytest.raises(OSError, match=r"video\.mp4"):
        extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=True)
    assert calls == []


@pytest.mark.parametrize("failure", ["media", "caption"])
def test_export_reports_failed_row_but_commits_later_successful_rows(tmp_path, monkeypatch, failure):
    from module import lanceexport
    from utils.caption_index import load_caption_index

    output = tmp_path / "out"
    dataset = _dataset(tmp_path, [
        {"uris": str(tmp_path / name), "mime": "image/png", "blob": _png("red"), "captions": ["annotation"]}
        for name in ("bad.png", "good.png")
    ])
    if failure == "media":
        original_atomic = lanceexport.atomic_output_path

        def fail_first_media(path):
            if path == output / "bad.png":
                raise OSError("simulated media write failure")
            return original_atomic(path)

        monkeypatch.setattr(lanceexport, "atomic_output_path", fail_first_media)
        dataset.tags.create("gemini", dataset.version)
        monkeypatch.setattr(sys, "argv", ["lanceexport", dataset.uri, "--output_dir", str(output), "--not_clip_with_caption"])
        with pytest.raises(OSError, match=r"bad\.png"):
            lanceexport.main()
    else:
        def fail_first_caption(path, mode, **kwargs):
            if path == output / "bad.txt" and mode == "w":
                raise OSError("simulated caption write failure")
            return open(path, mode, **kwargs)

        monkeypatch.setattr(lanceexport, "open", fail_first_caption, raising=False)
        with pytest.raises(OSError, match=r"bad\.png"):
            extract_from_lance(dataset, str(output), clip_with_caption=False)

    assert not (output / "bad.png").exists()
    assert not (output / "bad.txt").exists()
    assert (output / "good.png").read_bytes() == _png("red")
    assert (output / "good.txt").read_text(encoding="utf-8").strip() == "annotation"
    index = load_caption_index(output, strict=True)
    assert index.captions == {"good.png": "good.txt"}
    assert set(index.primary_sha256) == {"good.png"}


def test_failed_clip_caption_is_reported_and_never_marked_complete(tmp_path, monkeypatch):
    from module import lanceexport
    from utils.lance_utils import DERIVED_EXPORT_MARKER

    output = tmp_path / "out"
    captions = ["1\n00:00:00,000 --> 00:00:01,000\nclip caption\n"]
    dataset = _dataset(tmp_path, [{"uris": str(tmp_path / "audio.wav"), "mime": "audio/wav",
                                   "blob": b"audio", "captions": captions}])
    clip_dir = output / "audio_clip"

    def write_clip(media, subs, writer):
        clip = clip_dir / "audio_1.wav"
        clip.write_bytes(media.read_bytes())
        writer(clip, [subs[0].text], "image")

    def fail_clip_caption(path, mode, **kwargs):
        if path == clip_dir / "audio_1.txt" and mode == "w":
            raise OSError("simulated clip caption write failure")
        return open(path, mode, **kwargs)

    monkeypatch.setattr(lanceexport, "split_video_with_imageio_ffmpeg", write_clip)
    monkeypatch.setattr(lanceexport, "open", fail_clip_caption, raising=False)
    with pytest.raises(OSError, match=r"audio_1"):
        extract_from_lance(dataset, str(output))

    assert (output / "audio.wav").read_bytes() == b"audio"
    assert (output / "audio.srt").read_text(encoding="utf-8") == captions[0]
    assert not (clip_dir / DERIVED_EXPORT_MARKER).exists()


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
        extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=True, in_place=True)
    assert not list(tmp_path.glob("*.srt"))
    extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=False, in_place=True)
    assert len(list(tmp_path.glob("*.srt"))) == 2


@pytest.mark.parametrize("in_place", [False, True])
def test_export_reuses_unchanged_clips_after_dataset_version_bump(tmp_path, monkeypatch, in_place):
    from module import lanceexport

    source = tmp_path / "audio.wav"
    original = b"example audio"
    source.write_bytes(original)
    captions = ["1\n00:00:00,000 --> 00:00:01,000\nunchanged\n"]
    dataset = _dataset(tmp_path, [{
        "uris": str(source), "mime": "audio/wav", "captions": captions, "blob": original,
        "hash": hashlib.sha256(original).hexdigest(),
    }])
    runs = []

    def write_clip(media, subs, writer):
        clip_dir = media.parent / f"{media.stem}_clip"
        clip_dir.mkdir(exist_ok=True)
        clip = clip_dir / f"{media.stem}_1.wav"
        clip.write_bytes(media.read_bytes())
        runs.append(clip)

    monkeypatch.setattr(lanceexport, "split_video_with_imageio_ffmpeg", write_clip)
    output = tmp_path / "out"
    extract_from_lance(dataset, str(output), in_place=in_place)
    original_version = dataset.version
    dataset.add_columns({"version_marker": "CAST(1 AS INT)"})
    newer = lance.dataset(dataset.uri)
    assert newer.version > original_version
    extract_from_lance(newer, str(output), in_place=in_place)

    assert len(runs) == 1
    assert runs[0].read_bytes() == original


def test_in_place_export_rejects_unowned_clip_dir_before_any_writes(tmp_path):
    source = tmp_path / "audio.wav"
    source.write_bytes(b"example audio")
    clip = tmp_path / "audio_clip" / "audio_1.wav"
    clip.parent.mkdir()
    clip.write_bytes(b"user clip")
    dataset = _dataset(tmp_path, [{
        "uris": str(source), "mime": "audio/wav",
        "captions": ["1\n00:00:00,000 --> 00:00:01,000\ncaption\n"],
    }])

    with pytest.raises(ValueError, match="derived"):
        extract_from_lance(dataset, str(tmp_path / "out"), in_place=True)

    assert not source.with_suffix(".srt").exists()
    assert clip.read_bytes() == b"user clip"


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
        extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=True, in_place=True)
    assert not source.with_suffix(".srt").exists()
    assert primary.read_bytes() == b"primary"


def test_in_place_export_keeps_legacy_document_page_directory_untouched(tmp_path):
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
    extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=True, in_place=True)

    assert source.with_suffix(".md").read_text(encoding="utf-8") == page
    assert primary.read_text(encoding="utf-8") == "primary"
    assert list(primary.parent.iterdir()) == [primary]


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
                       clip_with_caption=False, in_place=True)
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
    extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=False, in_place=True)
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
        extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=False, in_place=True)
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
    extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=False, in_place=True)
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
    extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=False, in_place=True)
    previous = {path: path.read_bytes() for path in sources.glob("*.txt")}
    extract_from_lance(dataset, str(tmp_path / "out"), caption_suffix="_zh", clip_with_caption=False, in_place=True)
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
        extract_from_lance(dataset, str(tmp_path / "out"), clip_with_caption=False, in_place=True)
    images = [item for item in load_data(str(tmp_path)) if Path(item["file_path"]).suffix == ".png"]
    assert [(Path(item["file_path"]), item["caption"]) for item in images] == [(source, ["annotation"])]
