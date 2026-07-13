from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from module.muscriptor_tool.options import PreviewContent, PreviewFormat, PreviewRequest


class FakeSoundFile:
    def __init__(self):
        self.written_shape = None
        self.written_path = None

    def write(self, path, data, sample_rate):
        self.written_path = Path(path)
        self.written_shape = np.asarray(data).shape
        self.written_path.write_bytes(b"encoded")
        assert sample_rate == 44100

    def read(self, path, dtype="float32", always_2d=False):
        assert Path(path) == self.written_path
        assert dtype == "float32"
        channels = 1 if len(self.written_shape) == 1 else self.written_shape[1]
        shape = (32, channels) if always_2d else self.written_shape
        return np.zeros(shape, dtype=np.float32), 44100


def successful_run(_command, **_kwargs):
    return SimpleNamespace(returncode=0, stdout=b"FluidSynth 2", stderr=b"")


def test_preview_api_exposes_no_custom_or_system_sound_source():
    from module.muscriptor_tool.auralization import preflight_preview, render_preview

    preflight_parameters = inspect.signature(preflight_preview).parameters
    render_parameters = inspect.signature(render_preview).parameters

    assert "soundfont_path" not in preflight_parameters
    assert "soundfont_path" not in render_parameters
    assert "system_synth" not in render_parameters


def test_mp3_comparison_probe_uses_real_stereo_write_and_official_sf2(tmp_path: Path):
    from module.muscriptor_tool.auralization import preflight_preview

    sf2 = tmp_path / "MuseScore_General.sf2"
    sf2.write_bytes(b"sf2")
    fake_sf = FakeSoundFile()

    runtime = preflight_preview(
        PreviewRequest(PreviewContent.COMPARISON, PreviewFormat.MP3),
        soundfile_module=fake_sf,
        which=lambda _name: "fluidsynth",
        run=successful_run,
        resolve_default_sf2=lambda: sf2,
    )

    assert fake_sf.written_path.suffix == ".mp3"
    assert fake_sf.written_shape == (32, 2)
    assert runtime.soundfont_path == sf2
    assert runtime.renderer_id == "muscriptor-0.2.1:SF2_URL"


def test_codec_failure_short_circuits_before_fluidsynth_or_sf2():
    from module.muscriptor_tool.auralization import PreviewUnavailable, preflight_preview

    calls = []

    class FailingSoundFile(FakeSoundFile):
        def write(self, *_args, **_kwargs):
            raise RuntimeError("MP3 unavailable")

    with pytest.raises(PreviewUnavailable, match="mp3"):
        preflight_preview(
            PreviewRequest(PreviewContent.MIDI, PreviewFormat.MP3),
            soundfile_module=FailingSoundFile(),
            which=lambda name: calls.append(name),
            run=lambda *_args, **_kwargs: calls.append("run"),
            resolve_default_sf2=lambda: calls.append("sf2"),
        )

    assert calls == []


def test_missing_fluidsynth_does_not_download_default_sf2():
    from module.muscriptor_tool.auralization import PreviewUnavailable, preflight_preview

    calls = []

    with pytest.raises(PreviewUnavailable, match="FluidSynth"):
        preflight_preview(
            PreviewRequest(PreviewContent.MIDI, PreviewFormat.WAV),
            soundfile_module=FakeSoundFile(),
            which=lambda _name: None,
            resolve_default_sf2=lambda: calls.append("sf2"),
        )

    assert calls == []


def test_soundfont_resolution_error_includes_actionable_root_cause(tmp_path: Path):
    from module.muscriptor_tool.auralization import PreviewUnavailable, preflight_preview

    def fail_resolver():
        raise ImportError("SOCKS proxy requires the socksio package")

    with pytest.raises(PreviewUnavailable, match="socksio"):
        preflight_preview(
            PreviewRequest(PreviewContent.MIDI, PreviewFormat.WAV),
            soundfile_module=FakeSoundFile(),
            which=lambda _name: "fluidsynth",
            run=successful_run,
            resolve_default_sf2=fail_resolver,
        )


def test_render_preview_uses_synthesize_for_midi_mode(tmp_path: Path):
    from module.muscriptor_tool.auralization import PreviewRuntime, render_preview

    sf2 = tmp_path / "MuseScore_General.sf2"
    sf2.write_bytes(b"sf2")
    runtime = PreviewRuntime(PreviewRequest(PreviewContent.MIDI, PreviewFormat.WAV), sf2)
    target = tmp_path / "preview.wav"
    calls = []

    def synthesize(midi_path, output_path, soundfont_path=None):
        calls.append((Path(midi_path), Path(output_path), Path(soundfont_path)))
        Path(output_path).write_bytes(b"wav")

    render_preview(
        runtime,
        midi_bytes=b"MThd",
        original_audio_path=tmp_path / "source.wav",
        output_path=target,
        synthesize_func=synthesize,
        auralize_func=lambda *_args, **_kwargs: pytest.fail("auralize must not run"),
    )

    assert target.read_bytes() == b"wav"
    assert calls[0][2] == sf2
    assert list(tmp_path.glob("*.part*")) == []
    assert list(tmp_path.glob("*.mid")) == []


def test_render_preview_uses_auralize_for_comparison_mode(tmp_path: Path):
    from module.muscriptor_tool.auralization import PreviewRuntime, render_preview

    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    sf2 = tmp_path / "MuseScore_General.sf2"
    sf2.write_bytes(b"sf2")
    runtime = PreviewRuntime(PreviewRequest(PreviewContent.COMPARISON, PreviewFormat.MP3), sf2)
    target = tmp_path / "preview.mp3"
    calls = []

    def auralize(midi_path, original_audio_path, output_path, soundfont_path=None):
        calls.append((Path(midi_path), Path(original_audio_path), Path(soundfont_path)))
        Path(output_path).write_bytes(b"mp3")

    render_preview(
        runtime,
        midi_bytes=b"MThd",
        original_audio_path=source,
        output_path=target,
        synthesize_func=lambda *_args, **_kwargs: pytest.fail("synthesize must not run"),
        auralize_func=auralize,
    )

    assert target.read_bytes() == b"mp3"
    assert calls[0][1] == source
    assert calls[0][2] == sf2


def test_render_preview_rejects_extension_mismatch(tmp_path: Path):
    from module.muscriptor_tool.auralization import PreviewRuntime, render_preview

    sf2 = tmp_path / "MuseScore_General.sf2"
    sf2.write_bytes(b"sf2")
    runtime = PreviewRuntime(PreviewRequest(PreviewContent.MIDI, PreviewFormat.WAV), sf2)

    with pytest.raises(ValueError, match="extension"):
        render_preview(
            runtime,
            midi_bytes=b"MThd",
            original_audio_path=tmp_path / "source.wav",
            output_path=tmp_path / "preview.mp3",
        )
