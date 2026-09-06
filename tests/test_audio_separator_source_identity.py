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
            pass

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
    return SimpleNamespace(separated=separated, transcribed=transcribed, write=write_audio)


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
