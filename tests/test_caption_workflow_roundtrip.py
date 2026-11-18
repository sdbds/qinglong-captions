import runpy
import sys
import wave
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import lance
import pyarrow as pa
import pysrt
import pytest
from PIL import Image
from rich.console import Console


def test_audio_job_saves_distinct_timed_cues(tmp_path):
    from module.caption_pipeline.orchestrator import CaptionJob, _process_single_caption_job
    from module.providers.base import CaptionResult

    source = tmp_path / "song.wav"
    source.touch()
    expected = [(0, 700, "First line\nstill first"), (1000, 1800, "Second")]
    raw = "1\n00:00:00,000 --> 00:00:00,700\nFirst line\nstill first\n\n2\n00:00:01,000 --> 00:00:01,800\nSecond\n"
    _process_single_caption_job(
        CaptionJob(0, str(source), "audio/wav", 2000, ""),
        SimpleNamespace(segment_time=None), {}, api_process_batch_fn=lambda **_: CaptionResult(raw=raw),
        console_obj=Console(file=StringIO()),
    )

    cues = pysrt.open(source.with_suffix(".srt"), encoding="utf-8")
    assert [(cue.start.ordinal, cue.end.ordinal, cue.text) for cue in cues] == expected


@pytest.mark.parametrize(("name", "use_parent"), [("dataset.lance", True), ("custom.lance", True), ("custom.lance", False)])
def test_caption_opens_existing_dataset_without_reimporting(tmp_path, name, use_parent):
    from module.caption_pipeline.orchestrator import _resolve_dataset
    from module.lanceImport import transform2lance

    path = tmp_path / name
    original = lance.write_dataset(pa.table({"uris": ["missing.wav"], "manual_rating": [7]}), str(path))
    args = SimpleNamespace(dataset_dir=str(tmp_path if use_parent else path), gemini_api_key="", mistral_api_key="")
    opened = _resolve_dataset(args, transform2lance)

    assert Path(opened.uri).resolve() == path.resolve()
    assert opened.version == original.version
    assert opened.to_table().to_pylist() == [{"uris": "missing.wav", "manual_rating": 7}]
    assert not (path / "dataset.lance").exists()


def test_caption_refuses_ambiguous_datasets_before_import(tmp_path):
    from module.caption_pipeline.orchestrator import _resolve_dataset
    from module.lanceImport import transform2lance

    for name in ("a.lance", "b.lance"):
        lance.write_dataset(pa.table({"uris": [name]}), str(tmp_path / name))
    args = SimpleNamespace(dataset_dir=str(tmp_path), gemini_api_key="", mistral_api_key="")
    with pytest.raises(ValueError, match="Multiple Lance datasets"):
        _resolve_dataset(args, transform2lance)
    assert not (tmp_path / "dataset.lance").exists()


def test_caption_directory_import_cannot_replace_a_concurrent_creation(tmp_path):
    from module.caption_pipeline.orchestrator import _resolve_dataset
    from module.lanceImport import transform2lance

    path = tmp_path / "dataset.lance"
    def import_after_another_writer(**kwargs):
        lance.write_dataset(pa.table({"manual_rating": [7]}), str(path))
        return transform2lance(**kwargs)

    args = SimpleNamespace(dataset_dir=str(tmp_path), gemini_api_key="", mistral_api_key="")
    with pytest.raises((ValueError, OSError, RuntimeError)):
        _resolve_dataset(args, import_after_another_writer)
    assert lance.dataset(str(path)).to_table().to_pylist() == [{"manual_rating": 7}]


def _run_cli(monkeypatch, source, provider, *options, generate_clips=False):
    from module import api_handler_v2

    monkeypatch.setattr(api_handler_v2, "api_process_batch", provider)
    clip_options = [] if generate_clips else ["--not_clip_with_caption"]
    config_dir = Path(source).parent / "empty-fixture-config"
    monkeypatch.setattr(sys, "argv", ["captioner", str(source), "--config_dir", str(config_dir), *clip_options, *options])
    pixel_limit = Image.MAX_IMAGE_PIXELS
    try:
        runpy.run_module("module.captioner", run_name="__main__")
    except SystemExit as exc:
        return exc.code
    finally:
        Image.MAX_IMAGE_PIXELS = pixel_limit
    return 0


@pytest.mark.parametrize(("states", "expected_code"), [
    (["ok"], 0), (["skip"], 0), (["fail"], 1), (["empty"], 1),
    (["ok", "fail", "skip"], 3), (["raise", "ok"], 3),
])
def test_cli_reports_batch_outcome_without_republishing_failed_items(tmp_path, monkeypatch, states, expected_code):
    from module.lanceImport import transform2lance
    from module.providers.base import CaptionResult

    for state in states:
        Image.new("RGB", (3, 4)).save(tmp_path / f"{state}.png")
        (tmp_path / f"{state}.txt").write_text("from initial import", encoding="utf-8")
    ds = transform2lance(str(tmp_path))
    version = ds.version
    for state in states:
        (tmp_path / f"{state}.txt").write_text("more recent manual text", encoding="utf-8")

    def provider(**kwargs):
        state = Path(kwargs["uri"]).stem
        if state == "raise":
            raise RuntimeError("fixture provider error")
        return {"ok": CaptionResult.success("new caption"), "skip": CaptionResult.skipped("policy"),
                "fail": CaptionResult.failed("fixture error"), "empty": CaptionResult.success("")}[state]

    assert _run_cli(monkeypatch, tmp_path, provider) == expected_code
    current = lance.dataset(ds.uri)
    assert current.version == version + int("ok" in states)
    for row in current.to_table(columns=["uris", "captions"]).to_pylist():
        path = Path(row["uris"])
        if path.stem == "ok":
            assert row["captions"] == ["new caption"]
            assert path.with_suffix(".txt").read_text(encoding="utf-8").strip() == "new caption"
        else:
            assert row["captions"] == ["from initial import"]
            assert path.with_suffix(".txt").read_text(encoding="utf-8") == "more recent manual text"


def test_missing_original_aborts_before_inference_or_writes(tmp_path, monkeypatch):
    from module.lanceImport import transform2lance
    from module.providers.base import CaptionResult

    for name in ("present", "missing"):
        Image.new("RGB", (4, 3)).save(tmp_path / f"{name}.png")
    ds = transform2lance(str(tmp_path), save_binary=True)
    (tmp_path / "missing.png").unlink()
    calls = []

    def provider(**kwargs):
        calls.append(kwargs["uri"])
        return CaptionResult.success("must not be written")

    with pytest.raises(FileNotFoundError, match="missing.png"):
        _run_cli(monkeypatch, tmp_path / "dataset.lance", provider)
    assert calls == []
    assert not (tmp_path / "present.txt").exists()
    assert not (tmp_path / "missing.png").exists()
    assert lance.dataset(ds.uri).version == ds.version


def test_changed_original_aborts_before_labeling_different_blob(tmp_path, monkeypatch):
    from module.lanceImport import transform2lance

    media = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "red").save(media)
    ds = transform2lance(str(tmp_path), save_binary=True)
    Image.new("RGB", (4, 4), "blue").save(media)
    with pytest.raises(ValueError, match="changed.*image.png"):
        _run_cli(monkeypatch, Path(ds.uri), lambda **_: pytest.fail("Inference must not run"))
    assert not media.with_suffix(".txt").exists()
    assert lance.dataset(str(ds.uri)).version == ds.version


def test_bad_merge_size_is_rejected_before_any_sidecar_write(tmp_path, monkeypatch):
    from module.lanceImport import transform2lance

    media = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(media)
    ds = transform2lance(str(tmp_path))
    with pytest.raises(ValueError, match="merge_batch_size"):
        _run_cli(monkeypatch, Path(ds.uri), lambda **_: pytest.fail("Inference must not run"), "--merge_batch_size=0")
    assert not media.with_suffix(".txt").exists()
    assert lance.dataset(str(ds.uri)).version == ds.version


def test_explicit_export_in_place_retains_legacy_writeback(tmp_path, monkeypatch):
    from module.lanceexport import main
    from module.lanceImport import transform2lance

    Image.new("RGB", (4, 4)).save(tmp_path / "photo.png")
    caption = tmp_path / "photo.txt"
    caption.write_text("saved version", encoding="utf-8")
    ds = transform2lance(str(tmp_path))
    caption.write_text("draft", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["lanceexport", str(ds.uri), "--output_dir", str(tmp_path / "out"),
                                      "--in_place", "--not_clip_with_caption"])
    main()
    assert caption.read_text(encoding="utf-8").strip() == "saved version"


@pytest.mark.parametrize("with_manual_srt", [False, True])
def test_import_generate_export_and_rerun_preserve_snapshot_and_originals(tmp_path, monkeypatch, with_manual_srt):
    from module.lanceImport import iter_data_items, transform2lance
    from module.lanceexport import extract_from_lance
    from module.providers.base import CaptionResult
    from utils.lance_blob import take_blob_files

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    audio = inputs / "sample.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        stream.writeframes(bytes(8000 * 2 * 3))
    original = audio.read_bytes()
    manual_cue = b"1\r\n00:00:02,500 --> 00:00:02,900\r\nHand-written cue\r\n"
    if with_manual_srt:
        audio.with_suffix(".srt").write_bytes(manual_cue)
    ds = transform2lance(str(inputs), save_binary=True)
    original_captions = ds.to_table(columns=["captions"]).to_pylist()
    ds.add_columns({"manual_rating": "7"})
    ds.tags.create("before-caption", ds.version)
    initial_version = ds.version
    path = inputs / "dataset.lance"
    output = tmp_path / "export"
    calls = []

    def provider(**kwargs):
        clip = Path(kwargs["uri"])
        assert clip.is_file() and clip != audio
        calls.append(clip.name)
        return CaptionResult.success("1\n00:00:00,000 --> 00:00:00,200\n" + clip.stem + "\n")

    expected_cues = None
    for attempt, entry in enumerate((inputs, path), 1):
        assert _run_cli(monkeypatch, entry, provider, "--segment_time=1", generate_clips=not with_manual_srt) == 0
        current = lance.dataset(str(path))
        row = current.to_table(columns=["uris", "captions", "manual_rating"]).to_pylist()[0]
        assert (current.count_rows(), row["manual_rating"], row["uris"]) == (1, 7, str(audio))
        assert take_blob_files(current, [0])[0].readall() == original
        assert current.version == initial_version + attempt
        assert lance.dataset(str(path), version="before-caption").to_table(columns=["captions"]).to_pylist() == original_captions

        cues = [(s.start.ordinal, s.end.ordinal, s.text) for s in pysrt.from_string(row["captions"][0])]
        assert len(cues) == 3 + int(with_manual_srt) and all(a[0] < b[0] for a, b in zip(cues, cues[1:]))
        if with_manual_srt:
            assert audio.with_suffix(".srt").read_bytes() == manual_cue
            assert cues[-1] == (2500, 2900, "Hand-written cue")
        if expected_cues is None:
            expected_cues = cues
        assert cues == expected_cues
        extract_from_lance(current, str(output))
        exported = pysrt.open(output / "sample.srt", encoding="utf-8")
        assert [(s.start.ordinal, s.end.ordinal, s.text) for s in exported] == expected_cues
        assert audio.read_bytes() == (output / "sample.wav").read_bytes() == original
        assert len(list((output / "sample_clip").glob("*.wav"))) == len(cues)
        assert [Path(item["file_path"]) for item in iter_data_items(str(inputs))] == [audio]
        assert not (path / "dataset.lance").exists()
    assert len(calls) == 6

    manual = b"manual edit after generation\r\n"
    audio.with_suffix(".srt").write_bytes(manual)
    extract_from_lance(lance.dataset(str(path)), str(output), clip_with_caption=False)
    assert audio.with_suffix(".srt").read_bytes() == manual
