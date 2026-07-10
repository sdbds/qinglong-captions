import io
import sys
from pathlib import Path

import pytest
import pyarrow as pa
from rich.console import Console


ROOT = Path(__file__).resolve().parent.parent


class _UriScanner:
    def __init__(self, uris):
        self._uris = uris

    def to_batches(self):
        return [{"uris": pa.array(self._uris, type=pa.string())}]


class _UriDataset:
    def __init__(self, uris):
        self._uris = uris

    def scanner(self, **kwargs):
        assert kwargs["columns"] == ["uris"]
        return _UriScanner(self._uris)


def test_resolve_lance_rebuild_source_accepts_directory(tmp_path):
    from utils.lance_rebuild import resolve_lance_rebuild_source

    assert resolve_lance_rebuild_source(str(tmp_path)) == tmp_path


def test_resolve_lance_rebuild_source_accepts_lance_path(tmp_path):
    from utils.lance_rebuild import resolve_lance_rebuild_source

    dataset_path = tmp_path / "dataset.lance"

    assert resolve_lance_rebuild_source(str(dataset_path)) == tmp_path


def test_resolve_lance_rebuild_source_rejects_open_dataset_object():
    from utils.lance_rebuild import resolve_lance_rebuild_source

    assert resolve_lance_rebuild_source(object()) is None


def test_rebuild_data_falls_back_to_existing_dataset_uris_when_directory_scan_empty():
    from utils.lance_rebuild import load_lance_rebuild_data

    data = load_lance_rebuild_data(
        Path("missing"),
        _UriDataset(["a.png", "b.png"]),
        load_data_fn=lambda _source: [],
        read_sidecar_caption_fn=lambda uri, extension: [f"{uri}{extension}"],
        caption_extension=".txt",
    )

    assert data == [
        {"file_path": "a.png", "caption": ["a.png.txt"], "chunk_offsets": []},
        {"file_path": "b.png", "caption": ["b.png.txt"], "chunk_offsets": []},
    ]


def test_rebuild_data_dataset_uri_fallback_detects_caption_sidecar_extensions(tmp_path):
    from utils.lance_rebuild import load_lance_rebuild_data

    image_path = tmp_path / "image.png"
    audio_path = tmp_path / "audio.wav"
    document_path = tmp_path / "document.pdf"
    image_path.write_bytes(b"image")
    audio_path.write_bytes(b"audio")
    document_path.write_bytes(b"document")
    image_path.with_suffix(".txt").write_text("tag one\ntag two", encoding="utf-8")
    audio_path.with_suffix(".srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
    document_path.with_suffix(".md").write_text("# Title\n", encoding="utf-8")

    data = load_lance_rebuild_data(
        tmp_path,
        _UriDataset([str(image_path), str(audio_path), str(document_path)]),
        load_data_fn=lambda _source: [],
    )

    assert data == [
        {"file_path": str(image_path), "caption": ["tag one", "tag two"], "chunk_offsets": []},
        {"file_path": str(audio_path), "caption": ["1\n00:00:00,000 --> 00:00:01,000\nhello\n"], "chunk_offsets": []},
        {"file_path": str(document_path), "caption": ["# Title\n"], "chunk_offsets": []},
    ]


def test_caption_rebuild_uses_load_data_sidecar_detection_for_txt_md_srt(tmp_path):
    from module.lanceImport import load_data

    image_path = tmp_path / "image.png"
    document_path = tmp_path / "document.pdf"
    audio_path = tmp_path / "audio.wav"
    image_path.write_bytes(b"image")
    document_path.write_bytes(b"document")
    audio_path.write_bytes(b"audio")
    image_path.with_suffix(".txt").write_text("line 1\nline 2", encoding="utf-8")
    document_path.with_suffix(".md").write_text("# Title\n", encoding="utf-8")
    audio_path.with_suffix(".srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")

    captions_by_name = {Path(item["file_path"]).name: item["caption"] for item in load_data(str(tmp_path))}

    assert captions_by_name["image.png"] == ["line 1", "line 2"]
    assert captions_by_name["document.pdf"] == ["# Title\n"]
    assert captions_by_name["audio.wav"] == ["1\n00:00:00,000 --> 00:00:01,000\nhello\n"]


def test_rebuild_lance_from_sidecars_calls_transform_with_loaded_data(tmp_path):
    from utils.lance_rebuild import rebuild_lance_from_sidecars

    calls = []
    loaded_data = [{"file_path": str(tmp_path / "a.png"), "caption": ["caption"], "chunk_offsets": []}]
    rebuilt_dataset = object()

    def fake_transform(*args, **kwargs):
        calls.append((args, kwargs))
        return rebuilt_dataset

    result = rebuild_lance_from_sidecars(
        tmp_path,
        output_name="dataset",
        dataset=None,
        tag="gemini",
        transform2lance_fn=fake_transform,
        load_data_fn=lambda _source: loaded_data,
        console=Console(file=sys.stdout, force_terminal=False),
    )

    assert result is rebuilt_dataset
    args, kwargs = calls[0]
    assert args == (str(tmp_path),)
    assert kwargs["output_name"] == "dataset"
    assert kwargs["save_binary"] is False
    assert kwargs["not_save_disk"] is False
    assert kwargs["tag"] == "gemini"
    assert kwargs["load_condition"]() == loaded_data


def test_lance_blob_v2_caption_update_preserves_blob_without_rebuild(tmp_path):
    lance = pytest.importorskip("lance")
    if not hasattr(lance, "blob_field") or not hasattr(lance, "blob_array"):
        pytest.skip("Lance build does not expose Blob v2 APIs")

    from config.config import DATASET_SCHEMA
    from module.caption_pipeline.dataset_sync import update_dataset_captions
    from utils.lance_blob import build_lance_schema, build_lance_value_array, take_blob_files

    source_path = tmp_path / "a.png"
    source_path.write_bytes(b"image")
    schema = build_lance_schema(DATASET_SCHEMA, data_storage_version="2.2")
    values = {
        "uris": str(source_path),
        "mime": "image/png",
        "width": 1,
        "height": 1,
        "channels": 3,
        "depth": 8,
        "hash": "hash",
        "size": 0,
        "has_audio": False,
        "duration": 0,
        "num_frames": 1,
        "frame_rate": 0.0,
        "blob": b"image",
        "captions": [],
        "chunk_offsets": [],
    }
    arrays = [build_lance_value_array([values[field.name]], field) for field in schema]
    batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
    dataset = lance.write_dataset(
        pa.RecordBatchReader.from_batches(schema, [batch]),
        str(tmp_path / "dataset.lance"),
        schema=schema,
        mode="overwrite",
        data_storage_version="2.2",
    )

    def fake_transform(*_args, **_kwargs):
        pytest.fail("schema-preserving update must not rebuild the dataset")

    version_before = dataset.version
    blob_before = take_blob_files(dataset, [0], "blob")[0].readall()

    result = update_dataset_captions(
        dataset,
        [str(source_path)],
        ["caption"],
        merge_batch_size=10,
        console=Console(file=io.StringIO(), force_terminal=False, color_system=None),
        dataset_dir=str(tmp_path),
        transform2lance_fn=fake_transform,
        load_data_fn=lambda _source: [{"file_path": str(source_path), "caption": ["caption"], "chunk_offsets": []}],
    )

    assert result is dataset
    dataset.checkout_latest()
    assert dataset.version == version_before + 1
    assert dataset.to_table(columns=["captions"]).to_pylist() == [{"captions": ["caption"]}]
    assert take_blob_files(dataset, [0], "blob")[0].readall() == blob_before
