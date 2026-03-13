import importlib
import io
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


from utils.lance_blob import take_blob_files


class NewSignatureDataset:
    def take_blobs(self, blob_column, ids=None):
        return [f"{blob_column}:{row_id}" for row_id in ids]


class OldSignatureDataset:
    def take_blobs(self, ids, blob_column):
        return [f"{blob_column}:{row_id}" for row_id in ids]


def test_take_blob_files_supports_new_signature():
    assert take_blob_files(NewSignatureDataset(), [1, 2], "blob") == ["blob:1", "blob:2"]


def test_take_blob_files_supports_legacy_signature():
    assert take_blob_files(OldSignatureDataset(), [3, 4], "blob") == ["blob:3", "blob:4"]


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
