from __future__ import annotations

import pyarrow as pa
import pytest


def _write_blob_dataset(tmp_path, rows):
    lance = pytest.importorskip("lance")
    from utils.lance_blob import build_lance_schema, build_lance_value_array

    if not hasattr(lance, "blob_field") or not hasattr(lance, "blob_array"):
        pytest.skip("Lance build does not expose Blob v2 APIs")

    schema = build_lance_schema(
        [
            ("uris", pa.string()),
            ("mime", pa.string()),
            ("blob", pa.binary()),
            ("captions", pa.list_(pa.string())),
            ("count", pa.int64()),
        ],
        data_storage_version="2.2",
    )
    arrays = [build_lance_value_array([row[field.name] for row in rows], field) for field in schema]
    batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
    return lance.write_dataset(
        pa.RecordBatchReader.from_batches(schema, [batch]),
        str(tmp_path / "updates.lance"),
        schema=schema,
        mode="overwrite",
        data_storage_version="2.2",
    )


def _blob_bytes(dataset):
    from utils.lance_blob import take_blob_files

    return [blob.readall() if blob is not None else None for blob in take_blob_files(dataset, range(dataset.count_rows()), "blob")]


def test_merge_rows_preserves_blob_v2_and_unowned_columns_in_one_version(tmp_path):
    from utils.lance_updates import LanceRowUpdate, merge_rows_preserving_schema

    dataset = _write_blob_dataset(
        tmp_path,
        [
            {"uris": "a", "mime": "image/png", "blob": b"alpha", "captions": ["old-a"], "count": 1},
            {"uris": "b", "mime": "image/jpeg", "blob": b"beta", "captions": ["old-b"], "count": 2},
        ],
    )
    version_before = dataset.version
    blobs_before = {
        row["uris"]: blob
        for row, blob in zip(dataset.to_table(columns=["uris"]).to_pylist(), _blob_bytes(dataset))
    }

    merge_rows_preserving_schema(
        dataset,
        [
            LanceRowUpdate(uri="a", values={"captions": ["new-a"]}),
            LanceRowUpdate(uri="b", values={"captions": ["new-b"]}),
        ],
        batch_size=1,
    )

    dataset.checkout_latest()
    assert dataset.version == version_before + 1
    rows = sorted(dataset.to_table(columns=["uris", "mime", "captions", "count"]).to_pylist(), key=lambda row: row["uris"])
    assert rows == [
        {"uris": "a", "mime": "image/png", "captions": ["new-a"], "count": 1},
        {"uris": "b", "mime": "image/jpeg", "captions": ["new-b"], "count": 2},
    ]
    blobs_after = {
        row["uris"]: blob
        for row, blob in zip(dataset.to_table(columns=["uris"]).to_pylist(), _blob_bytes(dataset))
    }
    assert blobs_after == blobs_before


@pytest.mark.parametrize(
    "updates",
    [
        lambda LanceRowUpdate: [
            LanceRowUpdate(uri="a", values={"captions": ["one"]}),
            LanceRowUpdate(uri="a", values={"captions": ["two"]}),
        ],
        lambda LanceRowUpdate: [LanceRowUpdate(uri="missing", values={"captions": ["new"]})],
        lambda LanceRowUpdate: [LanceRowUpdate(uri="a", values={"unknown": "new"})],
    ],
)
def test_merge_rows_rejects_invalid_updates_without_creating_a_version(tmp_path, updates):
    from utils.lance_updates import LanceRowUpdate, LanceUpdateValidationError, merge_rows_preserving_schema

    dataset = _write_blob_dataset(
        tmp_path,
        [{"uris": "a", "mime": "image/png", "blob": b"alpha", "captions": ["old"], "count": 1}],
    )
    version_before = dataset.version

    with pytest.raises(LanceUpdateValidationError):
        merge_rows_preserving_schema(dataset, updates(LanceRowUpdate), batch_size=1)

    dataset.checkout_latest()
    assert dataset.version == version_before


def test_merge_rows_rejects_non_unique_target_key_without_creating_a_version(tmp_path):
    from utils.lance_updates import LanceRowUpdate, LanceUpdateValidationError, merge_rows_preserving_schema

    dataset = _write_blob_dataset(
        tmp_path,
        [
            {"uris": "a", "mime": "image/png", "blob": b"one", "captions": ["old-1"], "count": 1},
            {"uris": "a", "mime": "image/png", "blob": b"two", "captions": ["old-2"], "count": 2},
        ],
    )
    version_before = dataset.version

    with pytest.raises(LanceUpdateValidationError):
        merge_rows_preserving_schema(
            dataset,
            [LanceRowUpdate(uri="a", values={"captions": ["new"]})],
        )

    dataset.checkout_latest()
    assert dataset.version == version_before


def test_merge_rows_failure_while_consuming_second_batch_is_atomic(tmp_path, monkeypatch):
    import utils.lance_updates as lance_updates
    from utils.lance_updates import LanceRowUpdate, LanceUpdateStorageError, merge_rows_preserving_schema

    dataset = _write_blob_dataset(
        tmp_path,
        [
            {"uris": "a", "mime": "image/png", "blob": b"one", "captions": ["old-a"], "count": 1},
            {"uris": "b", "mime": "image/png", "blob": b"two", "captions": ["old-b"], "count": 2},
        ],
    )
    version_before = dataset.version
    original_read = lance_updates._read_blob_values
    read_count = 0

    def fail_second_blob_batch(*args, **kwargs):
        nonlocal read_count
        read_count += 1
        if read_count == 2:
            raise LanceUpdateStorageError("injected second-batch failure")
        return original_read(*args, **kwargs)

    monkeypatch.setattr(lance_updates, "_read_blob_values", fail_second_blob_batch)

    with pytest.raises(LanceUpdateStorageError):
        merge_rows_preserving_schema(
            dataset,
            [
                LanceRowUpdate(uri="a", values={"captions": ["new-a"]}),
                LanceRowUpdate(uri="b", values={"captions": ["new-b"]}),
            ],
            batch_size=1,
        )

    dataset.checkout_latest()
    assert dataset.version == version_before
    assert dataset.to_table(columns=["captions"]).to_pylist() == [
        {"captions": ["old-a"]},
        {"captions": ["old-b"]},
    ]
