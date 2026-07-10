import importlib
import io
import sys
import types
from pathlib import Path

import pytest
import pyarrow as pa

ROOT = Path(__file__).resolve().parent.parent


from utils.lance_blob import take_blob_files


class NewSignatureDataset:
    def take_blobs(self, blob_column, ids=None, addresses=None, indices=None):
        values = indices if indices is not None else ids if ids is not None else addresses
        return [f"{blob_column}:{row_id}" for row_id in values]


class OldSignatureDataset:
    def take_blobs(self, ids, blob_column):
        return [f"{blob_column}:{row_id}" for row_id in ids]


class NullSkippingDataset:
    def take_blobs(self, blob_column, ids=None, addresses=None, indices=None):
        return [f"{blob_column}:{row_id}" for row_id in indices if row_id == 1]

    def take(self, indices, columns=None):
        values = [{"size": 3 if row_id == 1 else 0} for row_id in indices]
        return pa.table({"blob": values})


def test_take_blob_files_supports_new_signature():
    assert take_blob_files(NewSignatureDataset(), [1, 2], "blob") == ["blob:1", "blob:2"]


def test_take_blob_files_supports_new_signature_ids():
    assert take_blob_files(NewSignatureDataset(), ids=[5, 6], blob_column="blob") == ["blob:5", "blob:6"]


def test_take_blob_files_supports_legacy_signature():
    assert take_blob_files(OldSignatureDataset(), [3, 4], "blob") == ["blob:3", "blob:4"]


def test_take_blob_files_aligns_null_skipping_results():
    assert take_blob_files(NullSkippingDataset(), [0, 1, 2], "blob") == [None, "blob:1", None]


def _load_module_without_lance_runtime_types(monkeypatch, module_name: str):
    fake_lance = types.ModuleType("lance")
    fake_lance.dataset = lambda *args, **kwargs: None
    fake_lance.write_dataset = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "lance", fake_lance)
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_lanceexport_imports_without_blobfile_runtime_symbol(monkeypatch):
    module = _load_module_without_lance_runtime_types(monkeypatch, "module.lanceexport")
    assert hasattr(module, "save_blob")


def test_lanceimport_imports_without_lancedataset_runtime_symbol(monkeypatch):
    module = _load_module_without_lance_runtime_types(monkeypatch, "module.lanceImport")
    assert hasattr(module, "transform2lance")


def test_lanceimport_cli_defaults_to_no_binary_save():
    from module.lanceImport import setup_parser

    parser = setup_parser()

    assert parser.parse_args(["dataset"]).save_binary is False
    assert parser.parse_args(["dataset", "--no_save_binary"]).save_binary is False
    assert parser.parse_args(["dataset", "--save_binary"]).save_binary is True


def test_lanceimport_process_accepts_unsized_iterable(monkeypatch, tmp_path):
    import module.lanceImport as lance_import

    source = tmp_path / "image.png"
    source.write_bytes(b"image")
    metadata = lance_import.Metadata(
        uris=str(source),
        mime="image/png",
        width=1,
        height=1,
        depth=8,
        channels=3,
        hash="hash",
        size=5,
        has_audio=False,
        duration=0,
        num_frames=1,
        frame_rate=0.0,
        blob=b"",
    )
    monkeypatch.setattr(lance_import.FileProcessor, "load_metadata", staticmethod(lambda *_args, **_kwargs: metadata))

    rows = (
        row
        for row in [
            {
                "file_path": str(source),
                "caption": ["caption"],
                "chunk_offsets": [],
            }
        ]
    )

    batches = list(lance_import.process(rows, save_binary=False))

    assert len(batches) == 1
    assert batches[0].num_rows == 1


def test_transform2lance_uses_streaming_default_loader(monkeypatch, tmp_path):
    import module.lanceImport as lance_import

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video")
    media.with_suffix(".txt").write_text("paired caption", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("standalone text", encoding="utf-8")
    captured_rows = []

    class FakeRecordBatchReader:
        @staticmethod
        def from_batches(_schema, batches):
            return ("reader", batches)

    def fake_process(data, *_args, **_kwargs):
        assert not isinstance(data, list)
        captured_rows.extend(list(data))
        return []

    def fake_write_dataset(reader, *_args, **_kwargs):
        assert reader[0] == "reader"
        return object()

    monkeypatch.setattr(lance_import.pa, "RecordBatchReader", FakeRecordBatchReader)
    monkeypatch.setattr(lance_import, "process", fake_process)
    monkeypatch.setattr(lance_import.lance, "write_dataset", fake_write_dataset)
    monkeypatch.setattr(lance_import, "update_or_create_tag", lambda *_args, **_kwargs: None)

    dataset = lance_import.transform2lance(str(tmp_path))

    assert dataset is not None
    assert [Path(row["file_path"]).name for row in captured_rows] == ["clip.mp4", "notes.txt"]
    assert captured_rows[0]["caption"] == ["paired caption"]
    assert captured_rows[1]["caption"] == []


class ReadableBlob:
    def __init__(self, payload: bytes):
        self._stream = io.BytesIO(payload)

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


def test_save_blob_supports_blob_like_reader(tmp_path):
    from module.lanceexport import save_blob

    target = tmp_path / "blob.bin"
    payload = b"blob-payload"

    assert save_blob(target, ReadableBlob(payload), {}, "application") is True
    assert target.read_bytes() == payload
