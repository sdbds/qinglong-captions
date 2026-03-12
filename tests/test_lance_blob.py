import sys
from pathlib import Path


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
