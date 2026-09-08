import importlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import lance
import numpy as np
import pyarrow as pa
import pytest
from PIL import Image

from module.wdtagger import outputs
from utils.caption_index import CAPTION_INDEX_NAME, caption_index_key, load_caption_index
from utils.output_writer import write_caption_output


def _image(path, color="red"):
    Image.new("RGB", (4, 4), color).save(path)
    return path


def _write(source, text, **kwargs):
    return outputs.write_sidecar_caption(
        str(source), [text], caption_extension=".txt", caption_separator=", ", **kwargs
    )


def test_same_stem_outputs_remain_distinct_when_reimported(tmp_path):
    first = _image(tmp_path / "sample.jpg")
    second = _image(tmp_path / "sample.png", "blue")

    _write(first, "red content")
    _write(second, "blue content")

    assert outputs.read_sidecar_caption(str(first), ".txt") == ["red content"]
    assert outputs.read_sidecar_caption(str(second), ".txt") == ["blue content"]
    importer = importlib.import_module("module.lanceImport")
    rows = {Path(row["file_path"]).name: row["caption"] for row in importer.load_data(str(tmp_path))}
    assert rows == {"sample.jpg": ["red content"], "sample.png": ["blue content"]}


def test_new_same_stem_source_does_not_inherit_another_sources_caption(tmp_path):
    first = _image(tmp_path / "sample.jpg")
    _write(first, "first tags")
    second = _image(tmp_path / "sample.png", "blue")

    assert outputs.has_sidecar_caption(str(second), ".txt") is False
    assert outputs.read_sidecar_caption(str(second), ".txt") == []
    _write(second, "second tags")
    assert outputs.read_sidecar_caption(str(first), ".txt") == ["first tags"]


def test_tagger_reads_and_rewrites_the_existing_indexed_filename(tmp_path):
    source = _image(tmp_path / "sample.png")
    canonical, _ = write_caption_output(source, "old tags", "image/png", caption_base=tmp_path / "unique.png")

    assert outputs.has_sidecar_caption(str(source), ".txt") is True
    assert outputs.read_sidecar_caption(str(source), ".txt") == ["old tags"]
    _write(source, "new tags")

    assert canonical.read_text(encoding="utf-8") == "new tags"
    assert not source.with_suffix(".txt").exists()
    assert outputs.read_sidecar_caption(str(source), ".txt") == ["new tags"]


def test_candidate_scan_skips_an_existing_indexed_caption(tmp_path):
    source = _image(tmp_path / "sample.png")
    write_caption_output(source, "existing", "image/png", caption_base=tmp_path / "unique.png")
    from module.wdtagger.cli import setup_parser
    from module.wdtagger.lance_io import filter_uncaptioned_batch

    batch = pa.record_batch({"uris": [str(source)], "captions": pa.array([[]], type=pa.list_(pa.string()))})
    assert filter_uncaptioned_batch(batch, setup_parser().parse_args([str(tmp_path)])) is None


def test_ambiguous_unindexed_legacy_caption_is_not_adopted_or_overwritten(tmp_path):
    first = _image(tmp_path / "sample.jpg")
    second = _image(tmp_path / "sample.png", "blue")
    legacy = tmp_path / "sample.txt"
    legacy.write_text("unknown owner", encoding="utf-8")

    assert outputs.read_sidecar_caption(str(first), ".txt") == []
    assert outputs.read_sidecar_caption(str(second), ".txt") == []
    _write(first, "first")
    _write(second, "second")

    assert legacy.read_text(encoding="utf-8") == "unknown owner"
    assert outputs.read_sidecar_caption(str(first), ".txt") == ["first"]
    assert outputs.read_sidecar_caption(str(second), ".txt") == ["second"]


def test_unambiguous_legacy_caption_remains_compatible(tmp_path):
    source = _image(tmp_path / "sample.jpg")
    source.with_suffix(".txt").write_text("existing tags", encoding="utf-8")
    assert outputs.has_sidecar_caption(str(source), ".txt") is True
    assert outputs.read_sidecar_caption(str(source), ".txt") == ["existing tags"]


def _fail_index_commit_once(monkeypatch):
    replace = os.replace
    failed = False

    def fail_once(source, target):
        nonlocal failed
        if Path(target).name == CAPTION_INDEX_NAME and not failed:
            failed = True
            raise OSError("injected index commit failure")
        return replace(source, target)

    monkeypatch.setattr(os, "replace", fail_once)


def test_failed_index_commit_restores_caption_and_allows_retry(tmp_path, monkeypatch):
    source = _image(tmp_path / "sample.jpg")
    caption, _ = write_caption_output(source, "old tags", "image/jpeg")
    before_index = (tmp_path / CAPTION_INDEX_NAME).read_bytes()
    _fail_index_commit_once(monkeypatch)

    with pytest.raises(OSError, match="index commit failure"):
        _write(source, "replacement")

    assert caption.read_text(encoding="utf-8") == "old tags"
    assert (tmp_path / CAPTION_INDEX_NAME).read_bytes() == before_index
    _write(source, "replacement")
    assert outputs.read_sidecar_caption(str(source), ".txt") == ["replacement"]


def test_invalid_index_is_not_bypassed_for_tagging(tmp_path):
    source = _image(tmp_path / "sample.jpg")
    caption = source.with_suffix(".txt")
    caption.write_text("old tags", encoding="utf-8")
    (tmp_path / CAPTION_INDEX_NAME).write_text('{"unrelated": true}', encoding="utf-8")

    with pytest.raises(ValueError, match="caption index"):
        _write(source, "replacement")
    assert caption.read_text(encoding="utf-8") == "old tags"


def test_caption_extension_cannot_overwrite_another_primary_image(tmp_path):
    source = _image(tmp_path / "sample.jpg")
    other = _image(tmp_path / "sample.png", "blue")
    original = other.read_bytes()

    outputs.write_sidecar_caption(str(source), ["tags"], caption_extension=".png", caption_separator=", ")

    assert other.read_bytes() == original
    assert outputs.read_sidecar_caption(str(source), ".png") == ["tags"]


def test_custom_multi_part_caption_extension_is_preserved(tmp_path):
    source = _image(tmp_path / "sample.jpg")

    outputs.write_sidecar_caption(str(source), ["tags"], caption_extension=".tags.txt", caption_separator=", ")

    assert (tmp_path / "sample.tags.txt").is_file()
    assert outputs.read_sidecar_caption(str(source), ".tags.txt") == ["tags"]
    assert not (tmp_path / "sample.txt").exists()


@pytest.mark.parametrize(("extension", "legacy_name"), [
    (".tags.txt", "sample.tags.txt"),
    (".tags.notes.txt", "sample.tags.notes.txt"),
])
def test_custom_extension_cannot_adopt_another_images_legacy_caption(tmp_path, extension, legacy_name):
    source = _image(tmp_path / "sample.png")
    other = _image(tmp_path / "sample.tags.png", "blue")
    legacy = tmp_path / legacy_name
    legacy.write_text("other image caption", encoding="utf-8")

    assert outputs.read_sidecar_caption(str(source), extension) == []
    outputs.write_sidecar_caption(str(source), ["new tags"], caption_extension=extension, caption_separator=", ")

    assert legacy.read_text(encoding="utf-8") == "other image caption"
    assert outputs.read_sidecar_caption(str(source), extension) == ["new tags"]
    if extension == ".tags.txt":
        assert outputs.read_sidecar_caption(str(other), ".txt") == ["other image caption"]


def test_legacy_caption_symlink_cannot_overwrite_an_unrelated_file(tmp_path):
    source = _image(tmp_path / "sample.jpg")
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("keep me", encoding="utf-8")
    try:
        source.with_suffix(".txt").symlink_to(unrelated)
    except OSError as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symlink"):
        _write(source, "new tags")

    assert unrelated.read_text(encoding="utf-8") == "keep me"


def _tagger_run(tmp_path, monkeypatch, *, append=False, merge_batch_size=100):
    from module.wdtagger import lance_io, runner
    from module.wdtagger.cli import finalize_args, setup_parser
    from module.wdtagger.taxonomy import LabelData

    # Earlier runtime-isolation tests can leave helpers bound to a lightweight Lance module.
    monkeypatch.setattr(lance_io, "lance", lance)
    monkeypatch.setattr(runner, "resolve_dataset", lance_io.resolve_dataset)
    args = finalize_args(setup_parser().parse_args([
        str(tmp_path), "--lance_update_mode", "merge", "--merge_batch_size", str(merge_batch_size),
        *(["--append_tags"] if append else ["--overwrite"]),
    ]))
    labels = LabelData(["new_tag"], {"general": np.array([0])}, {0: "general"})

    class Session:
        def run(self, _outputs, inputs):
            return [np.full((len(inputs["pixels"]), 1), 10.0)]

    monkeypatch.setattr(runner, "load_and_preprocess_batch", lambda uris, _cl: (uris, [np.zeros((4, 4, 3)) for _ in uris]))
    monkeypatch.setattr(runner, "_print_tag_frequencies", lambda _frequencies: None)
    runner.main(args, load_model_and_tags_fn=lambda _args: (Session(), "pixels", labels, {}))


def _dataset(tmp_path, source):
    path = tmp_path / "dataset.lance"
    lance.write_dataset(pa.table({
        "uris": [str(source)], "mime": ["image/png"],
        "captions": pa.array([[]], type=pa.list_(pa.string())),
    }), str(path))
    return path


def test_runner_appends_to_indexed_caption_and_persists_the_same_tags(tmp_path, monkeypatch):
    source = _image(tmp_path / "sample.png")
    path = _dataset(tmp_path, source)
    caption, _ = write_caption_output(source, "old_tag", "image/png", caption_base=tmp_path / "unique.png")

    _tagger_run(tmp_path, monkeypatch, append=True)

    assert caption.read_text(encoding="utf-8") == "old_tag, new_tag"
    assert lance.dataset(str(path)).to_table(columns=["captions"]).to_pylist() == [{"captions": ["old_tag", "new_tag"]}]
    assert not source.with_suffix(".txt").exists()


def test_runner_does_not_merge_tags_before_their_publication_succeeds(tmp_path, monkeypatch):
    source = _image(tmp_path / "sample.png")
    path = _dataset(tmp_path, source)
    _fail_index_commit_once(monkeypatch)

    with pytest.raises(OSError, match="index commit failure"):
        _tagger_run(tmp_path, monkeypatch, merge_batch_size=1)

    assert lance.dataset(str(path)).to_table(columns=["captions"]).to_pylist() == [{"captions": []}]
    assert not source.with_suffix(".txt").exists()


def test_concurrent_appends_keep_both_updates(tmp_path):
    source = _image(tmp_path / "sample.jpg")
    _write(source, "original")
    barrier = Barrier(2)

    def append(text):
        barrier.wait(timeout=5)
        _write(source, text, append=True)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(append, ["first", "second"]))

    caption = outputs.read_sidecar_caption(str(source), ".txt")[0]
    assert sorted(caption.split(", ")) == ["first", "original", "second"]
    index = load_caption_index(tmp_path, strict=True)
    assert len(index.captions) == 1
    assert caption_index_key(source, tmp_path) in index.captions
