from __future__ import annotations

import pyarrow as pa
import pytest


def _write_dataset(tmp_path):
    lance = pytest.importorskip("lance")
    return lance.write_dataset(
        pa.table({"uris": ["a", "b"], "captions": [["old-a"], ["old-b"]], "count": [1, 3]}),
        str(tmp_path / "updates.lance"),
    )


def test_stale_handle_fails_without_overwriting_newer_unowned_values(tmp_path):
    lance = pytest.importorskip("lance")
    from utils.lance_updates import LanceRowUpdate, LanceUpdateConflictError, merge_rows_preserving_schema

    dataset = _write_dataset(tmp_path)
    stale_dataset = lance.dataset(dataset.uri)
    merge_rows_preserving_schema(dataset, [LanceRowUpdate("a", {"count": 2})])
    committed_version = dataset.version

    with pytest.raises(LanceUpdateConflictError):
        merge_rows_preserving_schema(stale_dataset, [LanceRowUpdate("a", {"captions": ["new-a"]})])

    latest = lance.dataset(dataset.uri)
    assert latest.version == committed_version
    assert sorted(latest.to_table().to_pylist(), key=lambda row: row["uris"]) == [
        {"uris": "a", "captions": ["old-a"], "count": 2},
        {"uris": "b", "captions": ["old-b"], "count": 3},
    ]


def test_reopened_handle_can_update_captions_after_another_writer(tmp_path):
    lance = pytest.importorskip("lance")
    from utils.lance_updates import LanceRowUpdate, merge_rows_preserving_schema

    dataset = _write_dataset(tmp_path)
    merge_rows_preserving_schema(dataset, [LanceRowUpdate("a", {"count": 2})])
    current_dataset = lance.dataset(dataset.uri)
    version_before = current_dataset.version

    merge_rows_preserving_schema(
        current_dataset,
        [LanceRowUpdate("a", {"captions": ["new-a"]}), LanceRowUpdate("b", {"captions": ["new-b"]})],
        batch_size=1,
    )

    latest = lance.dataset(dataset.uri)
    assert latest.version == version_before + 1
    assert sorted(latest.to_table().to_pylist(), key=lambda row: row["uris"]) == [
        {"uris": "a", "captions": ["new-a"], "count": 2},
        {"uris": "b", "captions": ["new-b"], "count": 3},
    ]


def test_conflict_while_consuming_update_batches_is_atomic(tmp_path, monkeypatch):
    lance = pytest.importorskip("lance")
    import utils.lance_updates as lance_updates
    from utils.lance_updates import LanceRowUpdate, LanceUpdateConflictError, merge_rows_preserving_schema

    dataset = _write_dataset(tmp_path)
    other_writer = lance.dataset(dataset.uri)
    version_before = dataset.version
    build_batches = lance_updates._build_batches

    def batches_with_concurrent_commit(*args, **kwargs):
        for index, batch in enumerate(build_batches(*args, **kwargs)):
            if index == 1:
                other_writer.update({"count": "2"}, where="uris = 'a'")
            yield batch

    monkeypatch.setattr(lance_updates, "_build_batches", batches_with_concurrent_commit)

    with pytest.raises(LanceUpdateConflictError):
        merge_rows_preserving_schema(
            dataset,
            [LanceRowUpdate("a", {"captions": ["new-a"]}), LanceRowUpdate("b", {"captions": ["new-b"]})],
            batch_size=1,
        )

    latest = lance.dataset(dataset.uri)
    assert latest.version == version_before + 1
    assert sorted(latest.to_table().to_pylist(), key=lambda row: row["uris"]) == [
        {"uris": "a", "captions": ["old-a"], "count": 2},
        {"uris": "b", "captions": ["old-b"], "count": 3},
    ]
