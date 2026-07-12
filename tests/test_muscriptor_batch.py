from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from filelock import FileLock

from module.muscriptor_tool.options import BatchOptions, OutputFormat, TranscriptionOptions
from module.muscriptor_tool.outputs import TranscriptionResult


def write_audio_tree(root: Path, names: tuple[str, ...]) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"audio")


def successful_transcriber(calls: dict[str, object]):
    def transcribe(_loaded, source, _options, targets, **_kwargs):
        calls.setdefault("files", []).append(Path(source).name)
        outputs = {}
        for key, path in targets.requested_paths().items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"MThd") if key == "midi" else path.write_text("[]\n", encoding="utf-8")
            outputs[key] = str(path)
        return TranscriptionResult(
            note_count=0,
            event_count=0,
            chunk_count=1,
            completed_chunks=1,
            outputs=outputs,
            warnings=("EMPTY_TRANSCRIPTION",),
            midi_bytes=b"MThd" if targets.midi else None,
        )

    return transcribe


def loader_with_calls(calls: dict[str, object]):
    def load(options):
        calls["loads"] = int(calls.get("loads", 0)) + 1
        return SimpleNamespace(
            package_version="0.2.1",
            requested_device=options.device,
            resolved_device="cpu",
        )

    return load


def test_discovery_preserves_relative_paths_and_prunes_output_tree(tmp_path: Path):
    from module.muscriptor_tool.batch import discover_inputs, item_output_dir

    source = tmp_path / "album"
    output = source / "generated"
    write_audio_tree(source, ("disc1/song.wav",))
    write_audio_tree(output, ("old.mp3",))

    found = discover_inputs(source, output_dir=output, recursive=True)

    assert [item.relative_path.as_posix() for item in found] == ["disc1/song.wav"]
    assert item_output_dir(output, found[0]) == output / "disc1" / "song.wav"


def test_discovery_rejects_same_input_and_output_directory(tmp_path: Path):
    from module.muscriptor_tool.batch import discover_inputs

    with pytest.raises(ValueError, match="output directory"):
        discover_inputs(tmp_path, output_dir=tmp_path, recursive=True)


def test_non_recursive_discovery_ignores_nested_files(tmp_path: Path):
    from module.muscriptor_tool.batch import discover_inputs

    write_audio_tree(tmp_path, ("root.wav", "nested/song.mp3"))

    found = discover_inputs(tmp_path, output_dir=tmp_path.parent / "out", recursive=False)

    assert [item.relative_path.as_posix() for item in found] == ["root.wav"]


def test_batch_loads_once_and_transcribes_each_pending_file_once(tmp_path: Path):
    from module.muscriptor_tool.batch import run_batch

    inputs = tmp_path / "inputs"
    write_audio_tree(inputs, ("a.wav", "sub/b.flac"))
    calls: dict[str, object] = {"loads": 0, "files": []}

    summary = run_batch(
        inputs,
        tmp_path / "out",
        BatchOptions(),
        model_loader=loader_with_calls(calls),
        transcriber=successful_transcriber(calls),
        package_version="0.2.1",
        resolved_device="cpu",
    )

    assert calls == {"loads": 1, "files": ["a.wav", "b.flac"]}
    assert summary.processed == 2
    assert summary.failed == 0
    assert (tmp_path / "out" / "manifest.json").exists()


def test_all_complete_items_skip_without_model_or_preview_preflight(tmp_path: Path):
    from module.muscriptor_tool.batch import run_batch

    inputs = tmp_path / "inputs"
    output = tmp_path / "out"
    write_audio_tree(inputs, ("song.wav",))
    calls: dict[str, object] = {"loads": 0, "files": []}
    first = run_batch(
        inputs,
        output,
        BatchOptions(),
        model_loader=loader_with_calls(calls),
        transcriber=successful_transcriber(calls),
        package_version="0.2.1",
        resolved_device="cpu",
    )
    assert first.processed == 1

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("completed batch performed unnecessary work")

    second = run_batch(
        inputs,
        output,
        BatchOptions(),
        model_loader=fail_if_called,
        transcriber=fail_if_called,
        preview_preflight=fail_if_called,
        package_version="0.2.1",
        resolved_device="cpu",
    )

    assert second.skipped == 1
    assert second.processed == 0


def test_source_mtime_change_invalidates_completion(tmp_path: Path):
    from module.muscriptor_tool.batch import run_batch

    inputs = tmp_path / "inputs"
    output = tmp_path / "out"
    write_audio_tree(inputs, ("song.wav",))
    calls: dict[str, object] = {"loads": 0, "files": []}
    kwargs = {
        "model_loader": loader_with_calls(calls),
        "transcriber": successful_transcriber(calls),
        "package_version": "0.2.1",
        "resolved_device": "cpu",
    }
    run_batch(inputs, output, BatchOptions(), **kwargs)
    source = inputs / "song.wav"
    source.write_bytes(b"changed audio")

    summary = run_batch(inputs, output, BatchOptions(), **kwargs)

    assert summary.processed == 1
    assert calls["loads"] == 2


def test_file_failure_continues_and_writes_failed_metadata(tmp_path: Path):
    from module.muscriptor_tool.batch import run_batch

    inputs = tmp_path / "inputs"
    write_audio_tree(inputs, ("a.wav", "b.wav"))
    calls: dict[str, object] = {"loads": 0, "files": []}
    success = successful_transcriber(calls)

    def transcribe(loaded, source, options, targets, **kwargs):
        if Path(source).name == "a.wav":
            raise RuntimeError("decode failed")
        return success(loaded, source, options, targets, **kwargs)

    summary = run_batch(
        inputs,
        tmp_path / "out",
        BatchOptions(),
        model_loader=loader_with_calls(calls),
        transcriber=transcribe,
        package_version="0.2.1",
        resolved_device="cpu",
    )

    assert summary.failed == 1
    assert summary.processed == 1
    failed_metadata = json.loads((tmp_path / "out" / "a.wav" / "metadata.json").read_text(encoding="utf-8"))
    assert failed_metadata["status"] == "failed"
    assert failed_metadata["error"]["type"] == "RuntimeError"


def test_fail_fast_stops_after_first_file_failure(tmp_path: Path):
    from module.muscriptor_tool.batch import run_batch

    inputs = tmp_path / "inputs"
    write_audio_tree(inputs, ("a.wav", "b.wav"))
    calls: dict[str, object] = {"loads": 0, "files": []}

    def fail(_loaded, source, _options, _targets, **_kwargs):
        calls.setdefault("files", []).append(Path(source).name)
        raise RuntimeError("failed")

    summary = run_batch(
        inputs,
        tmp_path / "out",
        BatchOptions(fail_fast=True),
        model_loader=loader_with_calls(calls),
        transcriber=fail,
        package_version="0.2.1",
        resolved_device="cpu",
    )

    assert summary.failed == 1
    assert calls["files"] == ["a.wav"]


def test_output_lock_prevents_two_batches_using_same_directory(tmp_path: Path):
    from module.muscriptor_tool.batch import OutputDirectoryBusyError, run_batch

    inputs = tmp_path / "inputs"
    output = tmp_path / "out"
    write_audio_tree(inputs, ("song.wav",))
    output.mkdir()
    lock = FileLock(str(output / ".muscriptor.lock"))

    with lock:
        with pytest.raises(OutputDirectoryBusyError, match="already in use"):
            run_batch(
                inputs,
                output,
                BatchOptions(),
                package_version="0.2.1",
                resolved_device="cpu",
            )


def test_prune_known_outputs_never_deletes_user_files(tmp_path: Path):
    from module.muscriptor_tool.manifest import prune_known_outputs

    item_dir = tmp_path / "song.wav"
    item_dir.mkdir()
    for name in ("preview.wav", "preview.mp3", "events.json", "keep.txt"):
        (item_dir / name).write_text(name, encoding="utf-8")

    prune_known_outputs(item_dir, requested_names={"preview.mp3", "events.json"})

    assert not (item_dir / "preview.wav").exists()
    assert (item_dir / "preview.mp3").exists()
    assert (item_dir / "events.json").exists()
    assert (item_dir / "keep.txt").read_text(encoding="utf-8") == "keep.txt"


def test_run_signature_changes_with_output_selection(tmp_path: Path):
    from module.muscriptor_tool.batch import discover_inputs
    from module.muscriptor_tool.manifest import run_signature

    write_audio_tree(tmp_path / "inputs", ("song.wav",))
    item = discover_inputs(tmp_path / "inputs", output_dir=tmp_path / "out", recursive=True)[0]
    midi = BatchOptions(output_formats=(OutputFormat.MIDI,))
    json_only = BatchOptions(output_formats=(OutputFormat.JSON,))

    assert run_signature(item, midi, package_version="0.2.1", resolved_device="cpu") != run_signature(
        item,
        json_only,
        package_version="0.2.1",
        resolved_device="cpu",
    )
