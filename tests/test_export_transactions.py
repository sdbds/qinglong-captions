from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from unittest.mock import patch

import lance
import pyarrow as pa
import pytest
from PIL import Image

from module import lanceexport
from module.lanceImport import load_data
from utils.lance_blob import build_lance_schema, build_lance_value_array


def _dataset(root, name, sources, captions):
    schema = build_lance_schema([
        ("uris", pa.string()), ("mime", pa.string()),
        ("captions", pa.list_(pa.string())), ("blob", pa.large_binary()),
    ])
    records = [{"uris": str(source), "mime": "image/png", "captions": [caption], "blob": None}
               for source, caption in zip(sources, captions)]
    arrays = [build_lance_value_array([row[field.name] for row in records], field) for field in schema]
    return lance.write_dataset(pa.Table.from_arrays(arrays, schema=schema), str(root / name), data_storage_version="2.2")


def _export_in_process(dataset_path, output_path, barrier, caption_dir=None):
    original_load = lanceexport.load_caption_index
    loads = 0

    def read_same_initial_snapshot(directory, *, strict=False):
        nonlocal loads
        result = original_load(directory, strict=strict)
        loads += 1
        if loads == 1:
            barrier.wait(timeout=30)
        return result

    with patch.object(lanceexport, "load_caption_index", read_same_initial_snapshot):
        lanceexport.extract_from_lance(lance.dataset(dataset_path), output_path,
                                       clip_with_caption=False, caption_dir=caption_dir)


def test_process_exports_to_shared_directory_preserve_both_indexes(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    datasets = []
    expected = {}
    for stem in ("a", "b"):
        paths = [sources / f"{stem}.png", sources / f"{stem}.jpg"]
        captions = [f"{stem} PNG caption", f"{stem} JPG caption"]
        for path in paths:
            Image.new("RGB", (4, 4), "red").save(path)
        datasets.append(_dataset(tmp_path, f"{stem}.lance", paths, captions))
        expected.update({path: [caption] for path, caption in zip(paths, captions)})
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    workers = [context.Process(target=_export_in_process,
                               args=(str(tmp_path / f"{stem}.lance"), str(tmp_path / "out"), barrier))
               for stem in ("a", "b")]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=60)
        assert all(worker.exitcode == 0 for worker in workers)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)
    assert {Path(item["file_path"]): item["caption"] for item in load_data(str(sources))} == expected


@pytest.mark.parametrize("structured", [False, True])
def test_export_preflight_path_resolution_scales_linearly(tmp_path, monkeypatch, structured):
    caption = json.dumps({"description": "caption"}) if structured else "caption"
    original_resolve = Path.resolve
    counts = []
    for count in (16, 32):
        ds = _dataset(tmp_path, f"{count}.lance",
                      [tmp_path / "missing" / f"image-{index}.png" for index in range(count)], [caption] * count)
        calls = 0

        def resolve(path, *args, **kwargs):
            nonlocal calls
            calls += 1
            return original_resolve(path, *args, **kwargs)

        with monkeypatch.context() as context:
            context.setattr(Path, "resolve", resolve)
            lanceexport._plan_export_targets(ds, tmp_path / "out", None, clip_with_caption=False)
        counts.append(calls)
    assert counts[1] <= counts[0] * 2.5 + 20, counts


def test_competing_output_ownership_is_rejected_before_overwriting_caption(tmp_path):
    sources = tmp_path / "sources"
    captions = tmp_path / "captions"
    paths = []
    for stem in ("a", "b"):
        source = sources / stem / "same.png"
        source.parent.mkdir(parents=True)
        Image.new("RGB", (4, 4), "red").save(source)
        _dataset(tmp_path, f"{stem}.lance", [source], [f"{stem} caption"])
        paths.append(source)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    workers = [context.Process(target=_export_in_process,
                               args=(str(tmp_path / f"{stem}.lance"), str(tmp_path / "out"), barrier, str(captions)))
               for stem in ("a", "b")]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=60)
        assert sorted(worker.exitcode for worker in workers if worker.exitcode is not None) == [0, 1]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)
    winner = 0 if workers[0].exitcode == 0 else 1
    expected = [f"{'a' if winner == 0 else 'b'} caption"]
    assert (captions / "same.txt").read_text(encoding="utf-8").strip() == expected[0]
    imported = {Path(item["file_path"]): item["caption"] for item in load_data(str(sources), texts_dir=str(captions))}
    assert imported[paths[winner]] == expected
    assert imported[paths[1 - winner]] == []


def test_index_commit_failure_releases_lock_for_retry(tmp_path, monkeypatch):
    from filelock import FileLock

    source = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "red").save(source)
    ds = _dataset(tmp_path, "input.lance", [source], ["caption"])

    def fail_commit(*_):
        raise OSError("simulated index write failure")

    with monkeypatch.context() as context:
        context.setattr(lanceexport, "write_caption_index", fail_commit)
        with pytest.raises(OSError, match="simulated"):
            lanceexport.extract_from_lance(ds, str(tmp_path / "out"), clip_with_caption=False)
    with FileLock(str(tmp_path / ".qinglong-captions.json.lock"), timeout=0):
        pass
    lanceexport.extract_from_lance(ds, str(tmp_path / "out"), clip_with_caption=False)
    assert load_data(str(tmp_path))[0]["caption"] == ["caption"]


def test_existing_nonempty_lock_file_is_not_truncated(tmp_path):
    source = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "red").save(source)
    ds = _dataset(tmp_path, "input.lance", [source], ["caption"])
    lock = tmp_path / ".qinglong-captions.json.lock"
    lock.write_text("user file", encoding="utf-8")
    with pytest.raises(ValueError, match="unowned.*lock"):
        lanceexport.extract_from_lance(ds, str(tmp_path / "out"), clip_with_caption=False)
    assert lock.read_text(encoding="utf-8") == "user file"
    assert not source.with_suffix(".txt").exists()


def test_lock_file_created_after_preflight_is_preserved_on_rejection(tmp_path, monkeypatch):
    source = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "red").save(source)
    ds = _dataset(tmp_path, "input.lance", [source], ["caption"])
    lock = tmp_path / ".qinglong-captions.json.lock"
    original_plan = lanceexport._plan_export_targets
    calls = 0

    def introduce_user_file(*args, **kwargs):
        nonlocal calls
        plan = original_plan(*args, **kwargs)
        calls += 1
        if calls == 1:
            lock.write_text("user file", encoding="utf-8")
        return plan

    monkeypatch.setattr(lanceexport, "_plan_export_targets", introduce_user_file)
    with pytest.raises(ValueError, match="unowned.*lock"):
        lanceexport.extract_from_lance(ds, str(tmp_path / "out"), clip_with_caption=False)
    assert lock.read_text(encoding="utf-8") == "user file"
    assert not source.with_suffix(".txt").exists()


def test_distinct_directories_have_one_lock_order_regardless_of_row_order(tmp_path, monkeypatch):
    from filelock import FileLock

    sources = tmp_path / "sources"
    paths = [sources / name / "image.png" for name in ("ss", "\u00df")]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), "red").save(path)
    if paths[0].samefile(paths[1]):
        pytest.skip("The test requires distinct directories with a shared Unicode casefold")
    acquisitions = []

    class RecordingLock(FileLock):
        def __enter__(self):
            acquisitions.append(Path(self.lock_file).parent)
            return super().__enter__()

    monkeypatch.setattr(lanceexport, "FileLock", RecordingLock)
    expected = {paths[0]: ["first"], paths[1]: ["second"]}
    for index, order in enumerate((paths, paths[::-1])):
        ds = _dataset(tmp_path, f"{index}.lance", order, [expected[path][0] for path in order])
        lanceexport.extract_from_lance(ds, str(tmp_path / "out"), clip_with_caption=False)
    assert len(acquisitions) == 4
    assert acquisitions[:2] == acquisitions[2:]
    assert {Path(item["file_path"]): item["caption"] for item in load_data(str(sources))} == expected
