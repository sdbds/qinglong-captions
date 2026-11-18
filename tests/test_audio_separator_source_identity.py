from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import lance
import pyarrow as pa
import pytest
import torch
from filelock import FileLock
from rich.console import Console

import module.audio_separator as separator_cli
import module.muscriptor_tool.runtime as runtime
from module.audio_separator_core import read_audio_via_ffmpeg, write_audio_via_ffmpeg


@pytest.fixture
def audio_cli(monkeypatch, tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required for audio round-trip coverage")
    separated = []
    transcribed = []
    closed = []

    def read_audio(path):
        return read_audio_via_ffmpeg(path, ffmpeg_executable=ffmpeg, sample_rate=44100, channels=1)

    def write_audio(waveform, path, output_format="wav"):
        return write_audio_via_ffmpeg(
            waveform, path, ffmpeg_executable=ffmpeg, sample_rate=44100, output_format=output_format,
        )

    class Separator:
        def __init__(self, **kwargs):
            self.providers = ["stub-inference"]
            self.metadata = SimpleNamespace(stem_names=("vocals",), sample_rate=44100)
            self.model_tag = "review"

        def read_audio(self, source_path):
            return read_audio(source_path)

        def separate_file(self, source_path, **kwargs):
            separated.append(Path(source_path))
            return {"vocals": self.read_audio(source_path)}

        def write_audio(self, waveform, output_path, *, output_format):
            return write_audio(waveform, output_path, output_format)

        def close(self):
            closed.append(True)

    class Loaded:
        progress_event_type = type("Progress", (), {})

        def transcribe(self, source, options):
            transcribed.append(float(read_audio(source).mean()))
            return iter(())

        def midi_bytes(self, events):
            return b"review-midi"

    monkeypatch.setattr(separator_cli, "console", Console(file=io.StringIO(), force_terminal=False, color_system=None))
    monkeypatch.setattr(separator_cli, "AudioSeparator", Separator)
    monkeypatch.setattr(runtime, "load_model", lambda *args, **kwargs: Loaded())
    return SimpleNamespace(separated=separated, transcribed=transcribed, closed=closed, write=write_audio)


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("external_sources", [False, True])
def test_cli_isolates_sources_before_separation_and_deferred_midi(tmp_path, audio_cli, overwrite, external_sources):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    first = tmp_path / "album-a" / "song.wav" if external_sources else inputs / "song.flac"
    second = tmp_path / "album-b" / "song.wav" if external_sources else inputs / "song.wav"
    audio_cli.write(torch.full((1, 4410), 0.125), first, first.suffix[1:])
    audio_cli.write(torch.full((1, 4410), 0.5), second, second.suffix[1:])
    lance.write_dataset(pa.table({
        "uris": [str(first), str(second)], "mime": ["audio/flac", "audio/wav"],
    }), inputs / "sources.lance")
    args = [str(inputs), "--muscriptor_midi", "--muscriptor_device", "cpu"]
    assert separator_cli.main([*args, *(["--overwrite"] if overwrite else [])]) == 0
    assert audio_cli.separated == [first, second]
    assert audio_cli.transcribed == [0.125, 0.5]
    metadata = list(inputs.rglob("*.metadata.json"))
    assert len(metadata) == 2
    assert {json.loads(path.read_text(encoding="utf-8"))["source_path"] for path in metadata} == {str(first), str(second)}
    assert len(list(inputs.rglob("*.mid"))) == 2
    assert separator_cli.main(args) == 0
    assert audio_cli.separated == [first, second]
    assert audio_cli.transcribed == [0.125, 0.5]


def test_unowned_legacy_directory_is_preserved_not_reused(tmp_path, audio_cli):
    source = tmp_path / "song.wav"
    audio_cli.write(torch.full((1, 4410), 0.25), source)
    legacy = tmp_path / "song"
    legacy.mkdir()
    old_stem = legacy / "song_(vocals)_review.wav"
    audio_cli.write(torch.full((1, 4410), 0.75), old_stem)
    original = old_stem.read_bytes()
    assert separator_cli.main([str(source), "--muscriptor_midi", "--muscriptor_device", "cpu"]) == 0
    assert audio_cli.transcribed == [0.25]
    assert old_stem.read_bytes() == original


def test_owned_noncolliding_legacy_directory_can_resume(tmp_path, audio_cli):
    source = tmp_path / "song.wav"
    audio_cli.write(torch.full((1, 4410), 0.25), source)
    args = [str(source), "--muscriptor_midi", "--muscriptor_device", "cpu"]
    assert separator_cli.main(args) == 0
    created = next(tmp_path.rglob("*.metadata.json")).parent.parent
    legacy = tmp_path / "song"
    if created != legacy:
        created.rename(legacy)
    assert separator_cli.main(args) == 0
    assert audio_cli.separated == [source]
    assert (legacy / "04_stem_midi" / "song_(vocals).mid").is_file()


@pytest.mark.parametrize("nested", [False, True])
def test_batch_rejects_output_alias_collision_before_loading_model(monkeypatch, tmp_path, nested):
    sources = [tmp_path / "one.wav", tmp_path / "two.wav"]
    for source in sources:
        source.write_bytes(b"source")
    monkeypatch.setattr(separator_cli, "collect_audio_inputs", lambda _: (sources, tmp_path))
    monkeypatch.setattr(separator_cli, "build_song_output_dir", lambda source, **kwargs: (
        tmp_path / "shared" / "nested" if nested and source == sources[1] else tmp_path / "shared"
    ))
    loaded = []
    monkeypatch.setattr(separator_cli, "AudioSeparator", lambda **kwargs: loaded.append(True))
    assert separator_cli.main([str(tmp_path)]) == 1
    assert loaded == []
    assert not (tmp_path / "shared").exists()


def test_source_content_change_cannot_reuse_previous_separation(tmp_path, audio_cli):
    source = tmp_path / "song.wav"
    args = [str(source), "--muscriptor_midi", "--muscriptor_device", "cpu"]
    audio_cli.write(torch.full((1, 4410), 0.125), source)
    assert separator_cli.main(args) == 0
    audio_cli.write(torch.full((1, 4410), 0.5), source)
    assert separator_cli.main(args) == 0
    assert audio_cli.transcribed == [0.125, 0.5]


def test_sheetsage2_runs_after_close_and_reuses_only_current_primary_stems(monkeypatch, tmp_path, audio_cli):
    source = tmp_path / "Track with spaces.wav"
    audio_cli.write(torch.full((1, 4410), 0.25), source)
    child_inputs = []

    def run(command, **kwargs):
        assert len(audio_cli.closed) == len(child_inputs) + 1
        assert command[1:4] == ["run", "--no-project", "--script"]
        assert Path(command[4]).name == "sheetsage2_tool.py"
        assert "--device=cuda:1" in command and "--melody-only" in command
        manifest = Path(command[command.index("--stem-manifest") + 1])
        child_inputs.append(json.loads(manifest.read_text(encoding="utf-8"))["stems"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(separator_cli, "which", lambda name: "uv", raising=False)
    monkeypatch.setattr(separator_cli, "subprocess", SimpleNamespace(run=run), raising=False)
    args = [
        str(tmp_path), "--audio_files_only", "--sheetsage2_midi",
        "--sheetsage2_device=cuda:1", "--sheetsage2_melody_only",
    ]
    assert separator_cli.main(args) == 0
    stem_dir = tmp_path / "Track with spaces.wav.stems"
    expected = str(stem_dir / "Track with spaces_(vocals)_review.wav")
    assert Path(expected).is_file()
    audio_cli.write(torch.full((1, 4410), 0.5), stem_dir / "unrelated.wav")

    assert separator_cli.main(args) == 0
    assert child_inputs == [[expected], [expected]]
    assert audio_cli.separated == [source]
    assert audio_cli.transcribed == []
    assert not list(tmp_path.glob("*.lance"))

    # The Audio Separator tab may later index this folder, including its generated WAVs.
    lance.write_dataset(pa.table({
        "uris": [str(source), expected, str(stem_dir / "unrelated.wav")],
        "mime": ["audio/wav"] * 3,
    }), tmp_path / "sources.lance")
    assert separator_cli.main([arg for arg in args if arg != "--audio_files_only"]) == 0
    assert child_inputs == [[expected], [expected], [expected]]
    assert audio_cli.separated == [source]


def test_sheetsage2_archives_only_previous_outputs_for_a_now_silent_stem(monkeypatch, tmp_path, audio_cli):
    source = tmp_path / "song.wav"
    audio_cli.write(torch.full((1, 4410), 0.25), source)
    assert separator_cli.main([str(source)]) == 0
    stem = tmp_path / "song.wav.stems" / "song_(vocals)_review.wav"
    target = stem.parent / "04_stem_midi" / "sheetsage2" / stem.name
    target.mkdir(parents=True)
    (target / "transcription.mid").write_bytes(b"old MIDI")
    (target / "score.abc").write_text("Edited score", encoding="utf-8")
    (target / "notes.txt").write_text("My notes", encoding="utf-8")
    (target / "metadata.json").write_text(json.dumps({
        "signature": {"model": "m-a-p/SheetSage2", "source": str(stem)},
        "files": {"transcription.mid": 8, "score.abc": 12},
    }), encoding="utf-8")
    audio_cli.write(torch.zeros((1, 4410)), source)

    def unexpected_worker(*args, **kwargs):
        raise AssertionError("silent stems should not launch the SheetSage2 environment")

    monkeypatch.setattr(separator_cli, "_run_sheetsage2_stems", unexpected_worker)
    assert separator_cli.main([str(source), "--sheetsage2_midi"]) == 0
    assert not (target / "transcription.mid").exists()
    assert not (target / "score.abc").exists()
    assert [path.read_text(encoding="utf-8") for path in target.rglob("score.abc")] == ["Edited score"]
    assert (target / "notes.txt").read_text(encoding="utf-8") == "My notes"


@pytest.mark.parametrize("overwrite", [False, True])
def test_silent_retirement_failure_reports_incomplete_but_keeps_audio(tmp_path, audio_cli, overwrite):
    source = tmp_path / "song.wav"
    audio_cli.write(torch.zeros((1, 4410)), source)
    assert separator_cli.main([str(source)]) == 0
    stem = tmp_path / "song.wav.stems" / "song_(vocals)_review.wav"
    target = stem.parent / "04_stem_midi" / "sheetsage2" / stem.name
    target.mkdir(parents=True)
    (target / "transcription.mid").write_bytes(b"previous MIDI")
    (target / "metadata.json").write_text(json.dumps({
        "signature": {"model": "m-a-p/SheetSage2", "source": str(stem)}, "files": {"transcription.mid": 13},
    }), encoding="utf-8")
    args = [str(source), "--sheetsage2_midi", *(["--overwrite"] if overwrite else [])]

    with FileLock(target.parent / ".sheetsage2.lock"):
        assert separator_cli.main(args) == 1
    assert (target / "transcription.mid").read_bytes() == b"previous MIDI"
    assert stem.is_file()
    assert json.loads((stem.parent / "separation_metadata.json").read_text(encoding="utf-8"))["status"] == "ok"

    assert separator_cli.main([str(source), "--sheetsage2_midi"]) == 0
    assert not (target / "transcription.mid").exists()
