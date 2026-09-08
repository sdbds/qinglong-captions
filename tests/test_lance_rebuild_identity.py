from __future__ import annotations

import importlib
import io

import pyarrow as pa
import pytest
from PIL import Image
from rich.console import Console


def _write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 3), "red").save(path)
    return path


def _write_dataset(path, uris, *, storage_version="2.2"):
    lance = pytest.importorskip("lance")
    from utils.lance_blob import build_lance_schema, build_lance_value_array

    schema = build_lance_schema(
        [
            ("uris", pa.string()),
            ("mime", pa.string()),
            ("blob", pa.large_binary()),
            ("captions", pa.list_(pa.string())),
            ("count", pa.int64()),
            ("custom", pa.struct([("reviewed", pa.bool_()), ("scores", pa.list_(pa.float64()))])),
        ],
        data_storage_version=storage_version,
    )
    count_index = schema.get_field_index("count")
    schema = schema.set(count_index, schema.field("count").with_metadata({b"unit": b"observations"}))
    schema = schema.with_metadata({b"owner": b"rebuild-regression"})
    rows = [
        {
            "uris": str(uri),
            "mime": "image/png",
            "blob": f"saved-blob-{index}".encode(),
            "captions": [f"old-{index}"],
            "count": 20 + index,
            "custom": {"reviewed": True, "scores": [0.25, 0.75]},
        }
        for index, uri in enumerate(uris)
    ]
    arrays = [build_lance_value_array([row[field.name] for row in rows], field) for field in schema]
    batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
    return lance.write_dataset(
        pa.RecordBatchReader.from_batches(schema, [batch]),
        str(path),
        schema=schema,
        data_storage_version=storage_version,
    )


def _rebuild(source_dir, dataset, *, caption_extension=None, read_sidecar_caption_fn=None):
    from module.lanceImport import load_data, transform2lance
    from utils.lance_rebuild import rebuild_lance_from_sidecars

    return rebuild_lance_from_sidecars(
        source_dir,
        output_name="selected",
        dataset=dataset,
        tag="WDtagger",
        transform2lance_fn=transform2lance,
        load_data_fn=load_data,
        console=Console(file=io.StringIO(), force_terminal=False, color_system=None),
        caption_extension=caption_extension,
        read_sidecar_caption_fn=read_sidecar_caption_fn,
    )


def test_rebuild_keeps_external_uris_when_dataset_directory_has_unrelated_files(tmp_path):
    source = _write_image(tmp_path / "media" / "selected.png")
    source.with_suffix(".txt").write_text("new tags", encoding="utf-8")
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    (dataset_dir / "README.md").write_text("Not part of the selected dataset", encoding="utf-8")
    dataset = _write_dataset(dataset_dir / "selected.lance", [source])

    rebuilt = _rebuild(dataset_dir, dataset)

    assert rebuilt.to_table(columns=["uris", "captions"]).to_pylist() == [
        {"uris": str(source), "captions": ["new tags"]},
    ]
    assert rebuilt.uri == dataset.uri


@pytest.mark.parametrize("storage_version", ["2.0", "2.2"])
def test_rebuild_preserves_schema_metadata_non_caption_values_and_blob_representation(tmp_path, storage_version):
    from utils.lance_blob import is_blob_v2_field, take_blob_files

    source = _write_image(tmp_path / "selected.png")
    source.with_suffix(".txt").write_text("new tags", encoding="utf-8")
    dataset = _write_dataset(tmp_path / "selected.lance", [source], storage_version=storage_version)
    original_schema = dataset.schema
    version_before = dataset.version

    rebuilt = _rebuild(tmp_path, dataset)

    assert rebuilt.schema.equals(original_schema, check_metadata=True)
    assert rebuilt.version == version_before + 1
    assert rebuilt.to_table(columns=["uris", "mime", "captions", "count", "custom"]).to_pylist() == [
        {
            "uris": str(source),
            "mime": "image/png",
            "captions": ["new tags"],
            "count": 20,
            "custom": {"reviewed": True, "scores": [0.25, 0.75]},
        }
    ]
    if storage_version == "2.2":
        assert is_blob_v2_field(rebuilt.schema.field("blob"))
    else:
        assert not is_blob_v2_field(rebuilt.schema.field("blob"))
    assert take_blob_files(rebuilt, [0], "blob")[0].readall() == b"saved-blob-0"
    assert rebuilt.tags.get_version("WDtagger") == rebuilt.version


def test_rebuild_preserves_existing_captions_for_rows_without_sidecars(tmp_path):
    selected = _write_image(tmp_path / "selected.png")
    untouched = _write_image(tmp_path / "untouched.png")
    selected.with_suffix(".txt").write_text("new tags", encoding="utf-8")
    dataset = _write_dataset(tmp_path / "selected.lance", [selected, untouched])

    rebuilt = _rebuild(tmp_path, dataset)

    assert sorted(rebuilt.to_table(columns=["uris", "captions"]).to_pylist(), key=lambda row: row["uris"]) == [
        {"uris": str(selected), "captions": ["new tags"]},
        {"uris": str(untouched), "captions": ["old-1"]},
    ]


def test_rebuild_without_any_sidecars_does_not_create_a_data_version(tmp_path):
    source = _write_image(tmp_path / "selected.png")
    dataset = _write_dataset(tmp_path / "selected.lance", [source])
    version_before = dataset.version

    rebuilt = _rebuild(tmp_path, dataset)

    assert rebuilt.to_table(columns=["captions"]).to_pylist() == [{"captions": ["old-0"]}]
    assert rebuilt.version == version_before


@pytest.mark.parametrize("extension, content", [(".txt", ""), (".txt", " \n\t\n"), (".md", " \n")])
def test_rebuild_does_not_clear_existing_captions_for_empty_sidecars(tmp_path, extension, content):
    source = _write_image(tmp_path / "selected.png")
    source.with_suffix(extension).write_text(content, encoding="utf-8")
    dataset = _write_dataset(tmp_path / "selected.lance", [source])
    version_before = dataset.version

    rebuilt = _rebuild(tmp_path, dataset)

    assert rebuilt.to_table(columns=["captions"]).to_pylist() == [{"captions": ["old-0"]}]
    assert rebuilt.version == version_before


def test_rebuild_uses_supplied_caption_reader_without_assuming_a_sidecar_path(tmp_path):
    source = _write_image(tmp_path / "selected.png")
    indexed_sidecar = tmp_path / "indexed.caption"
    indexed_sidecar.write_text("indexed tags", encoding="utf-8")
    dataset = _write_dataset(tmp_path / "selected.lance", [source])

    def read_indexed_caption(uri, extension):
        if uri == str(source) and extension == ".txt":
            return indexed_sidecar.read_text(encoding="utf-8").splitlines()
        return []

    rebuilt = _rebuild(tmp_path, dataset, caption_extension=".txt", read_sidecar_caption_fn=read_indexed_caption)

    assert rebuilt.to_table(columns=["captions"]).to_pylist() == [{"captions": ["indexed tags"]}]


def test_rebuild_uses_existing_dataset_when_source_directory_is_unavailable(tmp_path):
    source = _write_image(tmp_path / "selected.png")
    source.with_suffix(".txt").write_text("new tags", encoding="utf-8")
    dataset = _write_dataset(tmp_path / "selected.lance", [source])

    rebuilt = _rebuild(None, dataset)

    assert rebuilt is dataset
    assert rebuilt.to_table(columns=["captions"]).to_pylist() == [{"captions": ["new tags"]}]


def test_rebuild_imports_source_directory_when_no_existing_dataset_is_supplied(tmp_path, monkeypatch):
    lance = pytest.importorskip("lance")
    source = _write_image(tmp_path / "selected.png")
    source.with_suffix(".txt").write_text("new tags", encoding="utf-8")

    with monkeypatch.context() as context:
        importer = importlib.import_module("module.lanceImport")
        context.setattr(importer, "lance", lance)
        rebuilt = _rebuild(tmp_path, None)

    assert rebuilt.to_table(columns=["uris", "captions"]).to_pylist() == [
        {"uris": str(source), "captions": ["new tags"]},
    ]
    assert rebuilt.tags.get_version("WDtagger") == rebuilt.version


def test_rebuild_data_uses_existing_dataset_uris_instead_of_directory_rows(tmp_path):
    from module.lanceImport import load_data
    from utils.lance_rebuild import load_lance_rebuild_data

    source = _write_image(tmp_path / "media" / "selected.png")
    source.with_suffix(".txt").write_text("new tags", encoding="utf-8")
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    (dataset_dir / "README.md").write_text("Unrelated document", encoding="utf-8")
    dataset = _write_dataset(dataset_dir / "selected.lance", [source])

    data = load_lance_rebuild_data(dataset_dir, dataset, load_data_fn=load_data)

    assert data == [{"file_path": str(source), "caption": ["new tags"], "chunk_offsets": []}]
