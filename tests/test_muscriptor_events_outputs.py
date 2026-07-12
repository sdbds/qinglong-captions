from __future__ import annotations

import json
import warnings
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from module.muscriptor_tool.options import OutputFormat, TranscriptionOptions


class ProgressEvent:
    def __init__(self, completed: int, total: int):
        self.completed = completed
        self.total = total


def start(index: int = 0, pitch: int = 60):
    return SimpleNamespace(pitch=pitch, start_time=0.25, index=index, instrument="piano")


def end(start_event, end_time: float = 1.5):
    return SimpleNamespace(end_time=end_time, start_event=start_event, start_event_index=start_event.index)


class FakeLoadedModel:
    def __init__(self, events, *, warn: bool = False, fail: bool = False):
        self.events = list(events)
        self.warn = warn
        self.fail = fail
        self.progress_event_type = ProgressEvent
        self.transcribe_calls = 0
        self.midi_calls = 0

    def transcribe(self, _source, _options):
        self.transcribe_calls += 1
        for event in self.events:
            yield event
        if self.warn:
            warnings.warn("chunk did not emit EOS", RuntimeWarning)
        if self.fail:
            raise RuntimeError("inference failed")

    def midi_bytes(self, events):
        self.midi_calls += 1
        return b"MThd" + bytes([len(list(events))])


def test_transcribe_once_fans_one_event_stream_to_all_formats(tmp_path: Path):
    from module.muscriptor_tool.outputs import OutputTargets, transcribe_once

    note_start = start()
    loaded = FakeLoadedModel(
        [ProgressEvent(0, 1), note_start, end(note_start), ProgressEvent(1, 1)],
    )
    targets = OutputTargets.for_directory(
        tmp_path,
        (OutputFormat.MIDI, OutputFormat.JSON, OutputFormat.JSONL),
    )

    result = transcribe_once(loaded, Path("song.wav"), TranscriptionOptions(), targets)

    assert loaded.transcribe_calls == 1
    assert loaded.midi_calls == 1
    assert json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))[-1]["type"] == "end"
    assert len((tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert (tmp_path / "transcription.mid").read_bytes() == b"MThd\x02"
    assert result.note_count == 1
    assert result.event_count == 2
    assert result.chunk_count == 1


def test_jsonl_only_does_not_build_midi(tmp_path: Path):
    from module.muscriptor_tool.outputs import OutputTargets, transcribe_once

    note_start = start()
    loaded = FakeLoadedModel([note_start, end(note_start)])
    targets = OutputTargets.for_directory(tmp_path, (OutputFormat.JSONL,))

    result = transcribe_once(loaded, Path("song.wav"), TranscriptionOptions(), targets)

    assert loaded.midi_calls == 0
    assert result.midi_bytes is None
    assert (tmp_path / "events.jsonl").exists()


def test_non_strict_eos_warning_is_returned_for_metadata(tmp_path: Path):
    from module.muscriptor_tool.outputs import OutputTargets, transcribe_once

    loaded = FakeLoadedModel([], warn=True)
    targets = OutputTargets.for_directory(tmp_path, (OutputFormat.JSON,))

    result = transcribe_once(loaded, Path("song.wav"), TranscriptionOptions(), targets)

    assert "EMPTY_TRANSCRIPTION" in result.warnings
    assert any("did not emit EOS" in item for item in result.warnings)


def test_atomic_outputs_leave_no_part_files_after_failure(tmp_path: Path):
    from module.muscriptor_tool.outputs import OutputTargets, transcribe_once

    loaded = FakeLoadedModel([start()], fail=True)
    targets = OutputTargets.for_directory(tmp_path, (OutputFormat.JSONL,))

    with pytest.raises(RuntimeError, match="inference failed"):
        transcribe_once(loaded, Path("song.wav"), TranscriptionOptions(), targets)

    assert not (tmp_path / "events.jsonl").exists()
    assert list(tmp_path.glob("*.part*")) == []


def test_print_notes_writes_only_to_stderr(tmp_path: Path, capsys):
    from module.muscriptor_tool.outputs import OutputTargets, transcribe_once

    note_start = start()
    loaded = FakeLoadedModel([note_start])
    targets = OutputTargets.for_directory(tmp_path, (OutputFormat.JSON,))
    options = TranscriptionOptions(print_notes=True)

    transcribe_once(loaded, Path("song.wav"), options, targets)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "pitch=60" in captured.err


def test_preview_reuses_same_event_stream_and_midi_bytes(tmp_path: Path):
    from module.muscriptor_tool.auralization import PreviewRuntime
    from module.muscriptor_tool.options import PreviewContent, PreviewFormat, PreviewRequest
    from module.muscriptor_tool.outputs import OutputTargets, transcribe_once

    note_start = start()
    loaded = FakeLoadedModel([note_start, end(note_start)])
    targets = OutputTargets.for_directory(tmp_path, (OutputFormat.JSONL,))
    preview_runtime = PreviewRuntime(
        PreviewRequest(PreviewContent.MIDI, PreviewFormat.WAV),
        tmp_path / "MuseScore_General.sf2",
    )
    preview_target = tmp_path / "preview.wav"
    render_calls = []

    def render(runtime, *, midi_bytes, original_audio_path, output_path):
        render_calls.append((runtime, midi_bytes, Path(original_audio_path)))
        Path(output_path).write_bytes(b"preview")

    result = transcribe_once(
        loaded,
        Path("song.wav"),
        TranscriptionOptions(),
        targets,
        preview_runtime=preview_runtime,
        preview_target=preview_target,
        preview_renderer=render,
    )

    assert loaded.transcribe_calls == 1
    assert loaded.midi_calls == 1
    assert render_calls[0][1] == b"MThd\x02"
    assert result.outputs["preview"] == str(preview_target)


def test_stdout_stream_targets_do_not_require_temporary_files(tmp_path: Path):
    from module.muscriptor_tool.outputs import OutputTargets, transcribe_once

    note_start = start()
    loaded = FakeLoadedModel([note_start, end(note_start)])
    midi_stream = BytesIO()
    json_stream = StringIO()
    jsonl_stream = StringIO()

    result = transcribe_once(
        loaded,
        Path("song.wav"),
        TranscriptionOptions(),
        OutputTargets(
            midi_stream=midi_stream,
            json_stream=json_stream,
            jsonl_stream=jsonl_stream,
        ),
    )

    assert midi_stream.getvalue() == b"MThd\x02"
    assert json.loads(json_stream.getvalue())[-1]["type"] == "end"
    assert len(jsonl_stream.getvalue().splitlines()) == 2
    assert result.outputs == {"midi": "-", "json": "-", "jsonl": "-"}
    assert list(tmp_path.iterdir()) == []
