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


def start(index: int = 0, pitch: int = 60, instrument: str = "piano"):
    return SimpleNamespace(
        pitch=pitch,
        start_time=0.25,
        index=index,
        instrument=instrument,
    )


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

    def detect_beat_grid(self, source, mode, onsets):
        return None

    def export_midi(self, events, beat_grid, timing, *, quantize):
        return self.midi_bytes(events), False


def test_timed_midi_and_sheets_share_beats_and_engrave_only_the_quantized_copy(tmp_path, monkeypatch):
    from module.muscriptor_tool import outputs

    note = start()
    class Model(FakeLoadedModel):
        def detect_beat_grid(self, source, mode, onsets):
            assert source == Path("song.wav") and mode == "best-effort"
            assert onsets == [0.25]
            calls.append("beats")
            return grid

        def export_midi(self, events, beat_grid, timing, *, quantize):
            assert beat_grid is grid and len(events) == 2
            calls.append(("midi", quantize))
            return (b"snapped", True) if quantize else (b"performance", False)

    def engrave(midi_bytes, *, quantized):
        assert midi_bytes == b"snapped" and quantized
        return b"engraved"

    grid, calls = object(), []
    loaded = Model([note, end(note)])
    targets = outputs.OutputTargets.for_directory(tmp_path, ("midi", "quantized-midi", "sheets"))
    monkeypatch.setattr(outputs, "engrave_sheets", engrave)
    result = outputs.transcribe_once(loaded, Path("song.wav"), TranscriptionOptions(), targets)

    assert loaded.transcribe_calls == 1
    assert calls == ["beats", ("midi", False), ("midi", True)]
    assert {key: Path(p).read_bytes() for key, p in result.outputs.items()} == {
        "midi": b"performance", "quantized-midi": b"snapped", "sheets": b"engraved",
    }


def test_transcribe_once_fans_one_event_stream_to_all_formats(tmp_path: Path):
    from module.muscriptor_tool.outputs import OutputTargets, transcribe_once

    note_start = start()
    loaded = FakeLoadedModel(
        [ProgressEvent(0, 1), note_start, end(note_start), ProgressEvent(1, 1)],
    )
    targets = OutputTargets.for_directory(
        tmp_path,
        (OutputFormat.MIDI, OutputFormat.JSON, OutputFormat.JSONL),
        output_stem="03.With You",
    )

    result = transcribe_once(loaded, Path("song.wav"), TranscriptionOptions(), targets)

    assert loaded.transcribe_calls == 1
    assert loaded.midi_calls == 1
    assert json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))[-1]["type"] == "end"
    assert len((tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert (tmp_path / "03.With You.mid").read_bytes() == b"MThd\x02"
    assert result.note_count == 1
    assert result.event_count == 2
    assert result.chunk_count == 1
    assert result.detected_instruments == ("piano",)


def test_transcribe_once_reports_unique_instruments_in_first_seen_order(tmp_path: Path):
    from module.muscriptor_tool.outputs import OutputTargets, transcribe_once

    piano = start(index=0, instrument="acoustic_piano")
    violin = start(index=1, instrument="violin")
    second_piano = start(index=2, instrument="acoustic_piano")
    loaded = FakeLoadedModel(
        [piano, end(piano), violin, end(violin), second_piano, end(second_piano)]
    )

    result = transcribe_once(
        loaded,
        Path("song.wav"),
        TranscriptionOptions(),
        OutputTargets.for_directory(tmp_path, (OutputFormat.JSONL,)),
    )

    assert result.detected_instruments == ("acoustic_piano", "violin")


def test_jsonl_only_does_not_build_midi(tmp_path: Path):
    from module.muscriptor_tool.outputs import OutputTargets, transcribe_once

    note_start = start()
    loaded = FakeLoadedModel([note_start, end(note_start)])
    loaded.detect_beat_grid = lambda *_args: pytest.fail("JSONL should not run the beat tracker")
    targets = OutputTargets.for_directory(tmp_path, (OutputFormat.JSONL,))

    result = transcribe_once(loaded, Path("song.wav"), TranscriptionOptions(), targets)

    assert loaded.midi_calls == 0
    assert result.midi_bytes is None
    assert (tmp_path / "events.jsonl").exists()


@pytest.mark.parametrize("mode", ["best-effort", "true", "false"])
def test_unavailable_beat_checkpoint_does_not_break_offline_midi(tmp_path, mode):
    from module.muscriptor_tool.adaptive_batch import AdaptiveBatchModelProxy
    from module.muscriptor_tool.options import MidiTiming
    from module.muscriptor_tool.outputs import OutputTargets, RequestedOutputError, transcribe_once
    from module.muscriptor_tool.runtime import LoadedModel

    class Offline:
        def detect_beat_grid_for(self, *_args):
            assert mode != "false"
            raise ValueError("checkpoint unavailable")

    target = OutputTargets(midi=tmp_path / "song.mid")
    options = TranscriptionOptions(timing=MidiTiming(detect_tempo=mode))
    model = FakeLoadedModel([])
    model.detect_beat_grid = LoadedModel(
        AdaptiveBatchModelProxy(Offline(), None), "0.4.0", "cpu", "cpu", ProgressEvent
    ).detect_beat_grid
    if mode == "true":
        pytest.importorskip("muscriptor")
        with pytest.raises(RequestedOutputError, match="checkpoint unavailable"):
            transcribe_once(model, Path("song.wav"), options, target)
        assert not target.midi.exists()
    else:
        result = transcribe_once(model, Path("song.wav"), options, target)
        assert target.midi.read_bytes().startswith(b"MThd")
        assert any("checkpoint unavailable" in message for message in result.warnings) == (mode == "best-effort")


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


def test_quantized_only_preview_plays_the_exported_midi(tmp_path):
    from module.muscriptor_tool.outputs import OutputTargets, transcribe_once

    class Model(FakeLoadedModel):
        def export_midi(self, events, beat_grid, timing, *, quantize):
            self.midi_calls += 1
            return (b"snapped", True) if quantize else (b"performance", False)

    model = Model([])
    targets = OutputTargets.for_directory(tmp_path, ("quantized-midi",))
    rendered = []
    result = transcribe_once(
        model, Path("song.wav"), TranscriptionOptions(), targets,
        preview_runtime=object(), preview_target=tmp_path / "preview.wav",
        preview_renderer=lambda _runtime, **kwargs: rendered.append(kwargs["midi_bytes"]),
    )
    assert model.midi_calls == 1
    assert rendered == [targets.quantized_midi.read_bytes()] == [result.midi_bytes] == [b"snapped"]


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


def test_preview_failure_exposes_completed_symbolic_result(tmp_path: Path):
    from module.muscriptor_tool.auralization import PreviewRuntime
    from module.muscriptor_tool.options import PreviewContent, PreviewFormat, PreviewRequest
    from module.muscriptor_tool.outputs import OutputTargets, RequestedOutputError, transcribe_once

    note_start = start()
    loaded = FakeLoadedModel([note_start, end(note_start)])
    targets = OutputTargets.for_directory(tmp_path, (OutputFormat.MIDI,), output_stem="song")
    preview_runtime = PreviewRuntime(
        PreviewRequest(PreviewContent.MIDI, PreviewFormat.WAV),
        tmp_path / "MuseScore_General.sf2",
    )

    def fail_preview(*_args, **_kwargs):
        raise RuntimeError("FluidSynth failed")

    with pytest.raises(RequestedOutputError, match="preview") as raised:
        transcribe_once(
            loaded,
            Path("song.wav"),
            TranscriptionOptions(),
            targets,
            preview_runtime=preview_runtime,
            preview_target=tmp_path / "preview.wav",
            preview_renderer=fail_preview,
        )

    assert raised.value.result.note_count == 1
    assert raised.value.result.event_count == 2
    assert raised.value.result.detected_instruments == ("piano",)
    assert raised.value.result.outputs == {"midi": str(tmp_path / "song.mid")}
    assert (tmp_path / "song.mid").is_file()
