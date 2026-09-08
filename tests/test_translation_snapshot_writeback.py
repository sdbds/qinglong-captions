from __future__ import annotations

import io

import lance
import pyarrow as pa
import pytest
from rich.console import Console

from module import texttranslate
from utils.lance_blob import build_lance_schema, build_lance_value_array, take_blob_files
from utils.lance_updates import LanceRowUpdate, LanceUpdateConflictError, merge_rows_preserving_schema


class RecordingTranslator:
    model_id = "offline-snapshot-test"
    backend = "direct"

    def __init__(self):
        self.calls = []

    def translate(self, text, **_kwargs):
        self.calls.append(text)
        return "translated " + text


@pytest.fixture(autouse=True)
def quiet_translation_console(monkeypatch):
    monkeypatch.setattr(texttranslate, "console", Console(file=io.StringIO(), force_terminal=False, color_system=None))


def _write_source_dataset(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("historical source", encoding="utf-8")
    schema = build_lance_schema(
        [
            ("uris", pa.string()),
            ("captions", pa.list_(pa.string())),
            ("chunk_offsets", pa.list_(pa.int32())),
            ("count", pa.int64()),
            ("blob", pa.large_binary()),
        ],
        data_storage_version="2.2",
    ).with_metadata({b"owner": b"translation-snapshot-test"})
    row = {
        "uris": str(source),
        "captions": ["historical source"],
        "chunk_offsets": [],
        "count": 1,
        "blob": b"historical blob",
    }
    arrays = [build_lance_value_array([row[field.name]], field) for field in schema]
    dataset = lance.write_dataset(
        pa.Table.from_arrays(arrays, schema=schema),
        str(tmp_path / "source.lance"),
        data_storage_version="2.2",
    )
    dataset.tags.create("selected.source", dataset.version)
    return source, dataset


def _translate(dataset_path, translator, *, export_root=None):
    texttranslate.translate_dataset(
        dataset_path=dataset_path,
        source_version="selected.source",
        translation_tag="translated",
        translator=translator,
        source_lang="en",
        target_lang="zh_cn",
        max_chars=2200,
        context_chars=0,
        glossary="",
        export_root=export_root,
        merge_batch_size=1,
    )


def test_historical_translation_input_writes_into_latest_non_caption_values(tmp_path):
    source, dataset = _write_source_dataset(tmp_path)
    original_schema = dataset.schema
    merge_rows_preserving_schema(
        dataset,
        [LanceRowUpdate(str(source), {"captions": ["newer source"], "count": 2, "blob": b"newer blob"})],
    )
    before_writeback = dataset.version
    translator = RecordingTranslator()

    _translate(tmp_path / "source.lance", translator)

    latest = lance.dataset(dataset.uri)
    assert translator.calls == ["historical source"]
    assert latest.version == before_writeback + 1
    assert latest.schema.equals(original_schema, check_metadata=True)
    assert latest.to_table(columns=["uris", "captions", "chunk_offsets", "count"]).to_pylist() == [
        {"uris": str(source), "captions": ["translated historical source\n"], "chunk_offsets": [29], "count": 2}
    ]
    assert take_blob_files(latest, [0], "blob")[0].readall() == b"newer blob"
    assert latest.tags.get_version("translated") == latest.version
    historical = lance.dataset(dataset.uri, version="selected.source")
    assert historical.to_table(columns=["captions", "count"]).to_pylist() == [
        {"captions": ["historical source"], "count": 1}
    ]


@pytest.mark.parametrize("resume", [False, True])
def test_sequential_rerun_from_historical_source_preserves_latest_unowned_values(tmp_path, resume):
    source, dataset = _write_source_dataset(tmp_path)
    translator = RecordingTranslator()
    export_root = tmp_path if resume else None
    _translate(tmp_path / "source.lance", translator, export_root=export_root)
    current = lance.dataset(dataset.uri)
    merge_rows_preserving_schema(current, [LanceRowUpdate(str(source), {"count": 2, "blob": b"newer blob"})])
    before_second_run = current.version

    _translate(tmp_path / "source.lance", translator, export_root=export_root)

    assert translator.calls == ["historical source"] * (1 if resume else 2)
    latest = lance.dataset(dataset.uri)
    assert latest.version == before_second_run + 1
    assert latest.to_table(columns=["captions", "count"]).to_pylist() == [
        {"captions": ["translated historical source\n"], "count": 2}
    ]
    assert take_blob_files(latest, [0], "blob")[0].readall() == b"newer blob"


def test_translation_writeback_fails_closed_on_race_after_snapshot_acquisition(tmp_path, monkeypatch):
    source, dataset = _write_source_dataset(tmp_path)
    before_run = dataset.version

    def commit_competing_update(target_ds, updates, **kwargs):
        competitor = lance.dataset(dataset.uri)
        merge_rows_preserving_schema(competitor, [LanceRowUpdate(str(source), {"count": 2})])
        return merge_rows_preserving_schema(target_ds, updates, **kwargs)

    monkeypatch.setattr(texttranslate, "merge_rows_preserving_schema", commit_competing_update)

    with pytest.raises(LanceUpdateConflictError):
        _translate(tmp_path / "source.lance", RecordingTranslator())

    latest = lance.dataset(dataset.uri)
    assert latest.version == before_run + 1
    assert latest.to_table(columns=["captions", "count"]).to_pylist() == [
        {"captions": ["historical source"], "count": 2}
    ]
    assert "translated" not in latest.tags.list()
