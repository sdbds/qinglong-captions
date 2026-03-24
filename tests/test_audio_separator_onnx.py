import asyncio
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from einops import rearrange


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "gui"))
sys.path.insert(2, str(ROOT / "gui" / "wizard"))

from module.audio_separator import (
    HARMONY_OUTPUT_DIRNAME,
    build_parser,
    build_song_output_dir,
    collect_audio_inputs,
    run_audio_separator,
)
from module.audio_separator_core import (
    AudioSeparatorMetadata,
    AudioSeparator,
    DEFAULT_HARMONY_SEPARATOR_REPO_ID,
    build_stft_features,
    compute_chunk_positions,
    download_audio_separator_metadata,
    resolve_audio_separator_runtime_config,
    finalize_stem_outputs,
    load_audio_separator_metadata_file,
    mask_to_complex_tensor,
    overlap_seconds_to_step_size,
    segment_size_to_chunk_size,
)
from module.vocal_midi import (
    DEFAULT_GAME_MODEL_REPO_ID,
    DEFAULT_VOCAL_MIDI_BATCH_SIZE,
    DEFAULT_VOCAL_MIDI_EST_THRESHOLD,
    DEFAULT_VOCAL_MIDI_NSTEPS,
    DEFAULT_VOCAL_MIDI_SEG_RADIUS,
    DEFAULT_VOCAL_MIDI_SEG_THRESHOLD,
    DEFAULT_VOCAL_MIDI_T0,
    GameOnnxConfig,
    GameOnnxTranscriber,
    build_vocal_midi_output_dir,
    build_parser as build_vocal_midi_parser,
    resolve_vocal_midi_runtime_config,
    resolve_vocal_midi_input,
)
from module.onnx_runtime import OnnxRuntimeConfig
from utils.onnx_export import (
    MODEL_TYPE_MEL_BAND_ROFORMER,
    build_bs_roformer_kwargs,
    build_export_metadata,
    build_mel_band_roformer_kwargs,
    create_dummy_stft_features,
    detect_roformer_model_type,
    rewrite_checkpoint_for_latest_mel_band_roformer,
)

step6_tools = importlib.import_module("step6_tools")
ToolsStep = step6_tools.ToolsStep


def test_load_audio_separator_metadata_file_reads_contract(tmp_path):
    metadata_path = tmp_path / "model.json"
    metadata_path.write_text(
        json.dumps(
            {
                "input_name": "stft_features",
                "output_name": "mask",
                "input_layout": ["batch", "frames", "freq_channels_complex"],
                "output_layout": ["batch", "stems", "freq_channels", "frames", "complex"],
                "sample_rate": 44100,
                "chunk_size": 563200,
                "num_channels": 2,
                "num_stems": 6,
                "stem_names": ["bass", "drums", "other", "vocals", "guitar", "piano"],
                "stft": {
                    "n_fft": 2048,
                    "hop_length": 512,
                    "win_length": 2048,
                    "normalized": False,
                    "zero_dc": True,
                },
            }
        ),
        encoding="utf-8",
    )

    metadata = load_audio_separator_metadata_file(metadata_path, repo_id="bdsqlsz/BS-ROFO-SW-Fixed-ONNX")

    assert metadata.repo_id == "bdsqlsz/BS-ROFO-SW-Fixed-ONNX"
    assert metadata.freq_bins == 1025
    assert metadata.freq_channels == 2050
    assert metadata.default_segment_size == 1101
    assert metadata.stem_names[3] == "vocals"
    assert metadata.zero_dc is True


def test_download_audio_separator_metadata_logs_download_and_existing(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    logs = []
    requested = []

    payload = {
        "input_name": "stft_features",
        "output_name": "mask",
        "input_layout": ["batch", "frames", "freq_channels_complex"],
        "output_layout": ["batch", "stems", "freq_channels", "frames", "complex"],
        "sample_rate": 44100,
        "chunk_size": 563200,
        "num_channels": 2,
        "num_stems": 2,
        "stem_names": ["vocals", "instrumental"],
        "stft": {
            "n_fft": 2048,
            "hop_length": 512,
            "win_length": 2048,
            "normalized": False,
        },
    }

    def fake_download(*, repo_id, filename, local_dir=None, force_download=False):
        requested.append((repo_id, filename, local_dir, force_download))
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload), encoding="utf-8")
        return str(target)

    metadata, metadata_path = download_audio_separator_metadata(
        "demo/repo",
        local_dir=cache_dir,
        downloader=fake_download,
        logger=logs.append,
    )

    assert metadata.repo_id == "demo/repo"
    assert metadata_path == cache_dir / "model.json"
    assert any("Downloading audio separator metadata" in message for message in logs)
    assert any("Downloaded audio separator metadata" in message for message in logs)

    logs.clear()
    requested.clear()
    metadata_again, metadata_path_again = download_audio_separator_metadata(
        "demo/repo",
        local_dir=cache_dir,
        downloader=fake_download,
        logger=logs.append,
    )

    assert metadata_again.repo_id == "demo/repo"
    assert metadata_path_again == cache_dir / "model.json"
    assert requested == []
    assert logs == [f"[green]Using existing audio separator metadata[/green] {cache_dir / 'model.json'}"]


def test_audio_separator_forwards_logger_to_model_downloads(tmp_path):
    logs = []
    captured = {}

    payload = {
        "input_name": "stft_features",
        "output_name": "mask",
        "input_layout": ["batch", "frames", "freq_channels_complex"],
        "output_layout": ["batch", "stems", "freq_channels", "frames", "complex"],
        "sample_rate": 44100,
        "chunk_size": 563200,
        "num_channels": 2,
        "num_stems": 2,
        "stem_names": ["vocals", "instrumental"],
        "stft": {
            "n_fft": 2048,
            "hop_length": 512,
            "win_length": 2048,
            "normalized": False,
        },
    }

    def fake_metadata_download(*, repo_id, filename, local_dir=None, force_download=False):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload), encoding="utf-8")
        return str(target)

    def fake_artifact_loader(repo_id, onnx_filename, **kwargs):
        captured["artifact"] = (repo_id, onnx_filename, kwargs)
        target = tmp_path / "model.onnx"
        target.write_text("onnx", encoding="utf-8")
        return target

    class FakeSession:
        @staticmethod
        def get_inputs():
            return [SimpleNamespace(name="stft_features")]

        @staticmethod
        def get_outputs():
            return [SimpleNamespace(name="mask")]

    def fake_load_session_bundle(**kwargs):
        return SimpleNamespace(
            sessions={"model": FakeSession()},
            providers=("CPUExecutionProvider",),
        )

    separator = AudioSeparator(
        repo_id="demo/repo",
        model_dir=tmp_path,
        runtime_config=OnnxRuntimeConfig(execution_provider="cpu"),
        metadata_downloader=fake_metadata_download,
        artifact_loader=fake_artifact_loader,
        session_bundle_loader=fake_load_session_bundle,
        logger=logs.append,
        ffmpeg_executable="ffmpeg",
    )

    assert separator.input_name == "stft_features"
    assert callable(captured["artifact"][2]["logger"])
    assert any("Downloading audio separator metadata" in message for message in logs)
    assert any("Downloaded audio separator metadata" in message for message in logs)


def test_segment_size_to_chunk_size_matches_model_formula():
    assert segment_size_to_chunk_size(1101, 512) == 563200


def test_overlap_seconds_to_step_size_matches_reference_mdxc_logic():
    assert overlap_seconds_to_step_size(4, sample_rate=25, chunk_size=400) == 100
    assert overlap_seconds_to_step_size(8, sample_rate=100, chunk_size=400) == 400


def test_compute_chunk_positions_appends_tail_chunk():
    starts, step_size = compute_chunk_positions(total_samples=1000, chunk_size=400, step_size=100)

    assert step_size == 100
    assert starts[0] == 0
    assert starts[-1] == 600
    assert 300 in starts


def test_build_stft_features_matches_expected_feature_size():
    metadata = AudioSeparatorMetadata(
        repo_id="demo/repo",
        input_name="stft_features",
        output_name="mask",
        input_layout=("batch", "frames", "freq_channels_complex"),
        output_layout=("batch", "stems", "freq_channels", "frames", "complex"),
        sample_rate=8000,
        chunk_size=56,
        num_channels=2,
        num_stems=2,
        stem_names=("vocals", "instrumental"),
        n_fft=8,
        hop_length=4,
        win_length=8,
        normalized=False,
    )
    waveform = torch.randn(2, 56)

    features, stft = build_stft_features(waveform, metadata)

    assert stft.shape[:2] == (2, 5)
    assert features.ndim == 2
    assert features.shape[1] == metadata.feature_size
    expected = rearrange(torch.view_as_real(stft), "s f t c -> t (f s c)")
    assert torch.allclose(features, expected)


def test_mask_to_complex_tensor_reshapes_output_layout():
    metadata = AudioSeparatorMetadata(
        repo_id="demo/repo",
        input_name="stft_features",
        output_name="mask",
        input_layout=("batch", "frames", "freq_channels_complex"),
        output_layout=("batch", "stems", "freq_channels", "frames", "complex"),
        sample_rate=8000,
        chunk_size=56,
        num_channels=2,
        num_stems=2,
        stem_names=("vocals", "instrumental"),
        n_fft=8,
        hop_length=4,
        win_length=8,
        normalized=False,
    )
    mask = np.zeros((2, 10, 7, 2), dtype=np.float32)

    complex_mask = mask_to_complex_tensor(mask, metadata)

    assert complex_mask.shape == (2, 2, 5, 7)
    assert torch.is_complex(complex_mask)
    expected = torch.view_as_complex(
        rearrange(torch.from_numpy(mask), "n (f s) t c -> n s f t c", s=metadata.num_channels).contiguous()
    )
    assert torch.equal(complex_mask, expected)


def test_build_song_output_dir_keeps_relative_parent(tmp_path):
    output_root = tmp_path / "out"
    input_root = tmp_path / "input"
    source = input_root / "album_a" / "song.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"demo")

    song_output_dir = build_song_output_dir(source, output_root=output_root, input_root=input_root)

    assert song_output_dir == output_root / "album_a" / "song"


class _FakeColumn:
    def __init__(self, values):
        self._values = values

    def to_pylist(self):
        return list(self._values)


class _FakeTable:
    def __init__(self, uris):
        self._uris = uris

    def column(self, name):
        assert name == "uris"
        return _FakeColumn(self._uris)


class _FakeLanceDataset:
    def __init__(self, uris):
        self._uris = uris

    def to_table(self, *, columns, filter):
        assert columns == ["uris"]
        assert filter == "mime LIKE 'audio/%'"
        return _FakeTable(self._uris)


def test_collect_audio_inputs_reads_lance_dataset(monkeypatch, tmp_path):
    lance_dir = tmp_path / "dataset.lance"
    lance_dir.mkdir()
    source = tmp_path / "song.mp3"
    source.write_bytes(b"demo")
    dataset = _FakeLanceDataset([str(source)])

    monkeypatch.setattr("module.audio_separator._open_lance_dataset", lambda path: dataset)

    files, input_root = collect_audio_inputs(lance_dir)

    assert files == [source]
    assert input_root is None


def test_collect_audio_inputs_converts_dir_to_lance(monkeypatch, tmp_path):
    source_dir = tmp_path / "music"
    source_dir.mkdir()
    source = source_dir / "album" / "song.flac"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"demo")
    dataset = _FakeLanceDataset([str(source)])
    opened = []

    monkeypatch.setattr("module.audio_separator._transform_dir_to_lance", lambda path: opened.append(path) or dataset)

    files, input_root = collect_audio_inputs(source_dir)

    assert files == [source]
    assert input_root == source_dir
    assert opened == [source_dir]

def test_audio_separator_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["demo.wav"])

    assert args.output_format == "wav"
    assert args.segment_size == 1101
    assert args.overlap == 8
    assert args.batch_size == 1
    assert args.harmony_separation is False
    assert args.harmony_repo_id == DEFAULT_HARMONY_SEPARATOR_REPO_ID


def test_vocal_midi_parser_defaults():
    parser = build_vocal_midi_parser()
    args = parser.parse_args(["demo.wav"])

    assert args.repo_id == DEFAULT_GAME_MODEL_REPO_ID
    assert args.language is None
    assert args.output_formats == "mid"
    assert args.batch_size == 4
    assert args.seg_threshold == 0.2
    assert args.seg_radius == 0.02
    assert args.t0 == 0.0
    assert args.nsteps == 8
    assert args.est_threshold == 0.2


def test_resolve_audio_separator_runtime_config_enables_trt_layer_norm_fp32_fallback():
    runtime = resolve_audio_separator_runtime_config()

    assert runtime.provider_options["tensorrt"]["trt_layer_norm_fp32_fallback"] is True
    assert runtime.provider_options["tensorrt"]["trt_fp16_enable"] is False


def test_resolve_vocal_midi_runtime_config_enables_trt_layer_norm_fp32_fallback():
    runtime = resolve_vocal_midi_runtime_config()

    assert runtime.provider_options["tensorrt"]["trt_layer_norm_fp32_fallback"] is True
    assert runtime.provider_options["tensorrt"]["trt_fp16_enable"] is False


def test_resolve_vocal_midi_input_prefers_dry_vocal_then_vocals_then_source(tmp_path):
    source_path = tmp_path / "song.wav"
    source_path.write_bytes(b"demo")
    song_output_dir = tmp_path / "song"
    song_output_dir.mkdir()

    vocals_path = song_output_dir / "song_(vocals)_primary.wav"
    vocals_path.write_bytes(b"vocals")
    dry_vocal_dir = song_output_dir / HARMONY_OUTPUT_DIRNAME
    dry_vocal_dir.mkdir()
    dry_vocal_path = dry_vocal_dir / "song_(dry_vocal)_karaoke.wav"
    dry_vocal_path.write_bytes(b"dry")

    assert resolve_vocal_midi_input(source_path, song_output_dir) == dry_vocal_path

    dry_vocal_path.unlink()
    assert resolve_vocal_midi_input(source_path, song_output_dir) == vocals_path

    vocals_path.unlink()
    assert resolve_vocal_midi_input(source_path, song_output_dir) == source_path


def test_vocal_midi_single_known_duration_skips_dur2bd_and_uses_empty_boundaries():
    transcriber = GameOnnxTranscriber.__new__(GameOnnxTranscriber)
    transcriber.repo_id = DEFAULT_GAME_MODEL_REPO_ID
    transcriber.config = GameOnnxConfig(
        repo_id=DEFAULT_GAME_MODEL_REPO_ID,
        samplerate=4,
        timestep=0.5,
        languages={"zh": 4},
        loop=True,
        embedding_dim=256,
    )
    transcriber.bd2dur_output_names = ("durations", "maskN")

    captured = {}

    class _Dur2BdSession:
        def run(self, *_args, **_kwargs):
            raise AssertionError("dur2bd should not run when there is only one known duration")

    class _Bd2DurSession:
        def run(self, _output_names, feed):
            boundaries = feed["boundaries"]
            captured["bd2dur_boundaries"] = boundaries.copy()
            batch_size = boundaries.shape[0]
            durations = np.full((batch_size, 1), 1.0, dtype=np.float32)
            mask_n = np.ones((batch_size, 1), dtype=np.bool_)
            return durations, mask_n

    transcriber.sessions = {
        "dur2bd": _Dur2BdSession(),
        "bd2dur": _Bd2DurSession(),
    }

    def fake_run_encoder(waveform, duration):
        batch_size = waveform.shape[0]
        mask_t = np.ones((batch_size, 3), dtype=np.bool_)
        return (
            np.zeros((batch_size, 3, 2), dtype=np.float32),
            np.zeros((batch_size, 3, 2), dtype=np.float32),
            mask_t,
        )

    def fake_run_segmenter(x_seg, known_boundaries, mask_t, **_kwargs):
        captured["known_boundaries"] = known_boundaries.copy()
        captured["mask_t"] = mask_t.copy()
        return np.zeros_like(mask_t, dtype=np.bool_)

    def fake_run_estimator(x_est, boundaries, mask_t, mask_n, *, est_threshold):
        batch_size = boundaries.shape[0]
        return np.ones((batch_size, 1), dtype=np.bool_), np.full((batch_size, 1), 60.0, dtype=np.float32)

    transcriber._run_encoder = fake_run_encoder
    transcriber._run_segmenter = fake_run_segmenter
    transcriber._run_estimator = fake_run_estimator

    notes = transcriber._predict_chunk_notes(
        [{"offset": 0.0, "waveform": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)}],
        language="zh",
        seg_threshold=0.2,
        seg_radius=0.02,
        t0=0.0,
        nsteps=8,
        est_threshold=0.2,
    )

    assert captured["known_boundaries"].shape == (1, 3)
    assert not captured["known_boundaries"].any()
    assert np.array_equal(captured["known_boundaries"], captured["bd2dur_boundaries"])
    assert len(notes) == 1


def test_audio_separator_script_help_runs_from_script_path():
    result = subprocess.run(
        [sys.executable, str(ROOT / "module" / "audio_separator.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "segment_size" in result.stdout
    assert "output_format" in result.stdout


def test_tools_step_audio_separator_maps_args(monkeypatch, tmp_path):
    step = ToolsStep()
    input_dir = tmp_path / "music"
    input_dir.mkdir()

    captured = {}

    async def fake_run_job(script_key, args, name, **kwargs):
        captured["script_key"] = script_key
        captured["args"] = list(args)
        captured["name"] = name
        captured["kwargs"] = kwargs
        return SimpleNamespace(status="ok")

    notifications = []
    monkeypatch.setattr(step6_tools.ui, "notify", lambda message, **kwargs: notifications.append((message, kwargs)))

    step.audio_separator_input = SimpleNamespace(value=str(input_dir))
    step.audio_separator_output_format = SimpleNamespace(value="flac")
    step.panel = SimpleNamespace(run_job=fake_run_job)
    step.config["audio_separator_overwrite"] = True
    step.config["audio_separator_harmony_separation"] = True

    asyncio.run(step._start_audio_separator())

    assert notifications == []
    assert captured["script_key"] == "module.audio_separator"
    assert captured["name"] == "Audio Separator"
    assert captured["args"][0] == str(input_dir)
    assert "--output_format=flac" in captured["args"]
    assert "--segment_size=1101" in captured["args"]
    assert "--overlap=8" in captured["args"]
    assert "--batch_size=1" in captured["args"]
    assert "--overwrite" in captured["args"]
    assert "--harmony_separation" in captured["args"]


def test_tools_step_exposes_full_game_model_labels():
    assert step6_tools.ToolsStep.VOCAL_MIDI_MODELS["bdsqlsz/GAME-1.0-small-ONNX"] == "GAME-1.0-small-ONNX"
    assert step6_tools.ToolsStep.VOCAL_MIDI_MODELS["bdsqlsz/GAME-1.0-medium-ONNX"] == "GAME-1.0-medium-ONNX"
    assert step6_tools.ToolsStep.VOCAL_MIDI_MODELS["bdsqlsz/GAME-1.0-large-ONNX"] == "GAME-1.0-large-ONNX"


def test_tools_step_vocal_midi_defaults_follow_backend_constants():
    step = ToolsStep()

    assert step.config["audio_separator_vocal_midi_batch_size"] == DEFAULT_VOCAL_MIDI_BATCH_SIZE
    assert step.config["audio_separator_vocal_midi_seg_threshold"] == DEFAULT_VOCAL_MIDI_SEG_THRESHOLD
    assert step.config["audio_separator_vocal_midi_seg_radius"] == DEFAULT_VOCAL_MIDI_SEG_RADIUS
    assert step.config["audio_separator_vocal_midi_t0"] == DEFAULT_VOCAL_MIDI_T0
    assert step.config["audio_separator_vocal_midi_nsteps"] == DEFAULT_VOCAL_MIDI_NSTEPS
    assert step.config["audio_separator_vocal_midi_est_threshold"] == DEFAULT_VOCAL_MIDI_EST_THRESHOLD


def test_tools_step_audio_separator_maps_vocal_midi_args(monkeypatch, tmp_path):
    step = ToolsStep()
    input_dir = tmp_path / "music"
    input_dir.mkdir()

    captured = {}

    async def fake_run_job(script_key, args, name, **kwargs):
        captured["script_key"] = script_key
        captured["args"] = list(args)
        captured["name"] = name
        captured["kwargs"] = kwargs
        return SimpleNamespace(status="ok")

    notifications = []
    monkeypatch.setattr(step6_tools.ui, "notify", lambda message, **kwargs: notifications.append((message, kwargs)))

    step.audio_separator_input = SimpleNamespace(value=str(input_dir))
    step.audio_separator_output_format = SimpleNamespace(value="wav")
    step.audio_separator_vocal_midi_model = SimpleNamespace(value="bdsqlsz/GAME-1.0-large-ONNX")
    step.audio_separator_vocal_midi_language = SimpleNamespace(value="zh")
    step.audio_separator_vocal_midi_output_formats = SimpleNamespace(value=["mid", "csv"])
    step.panel = SimpleNamespace(run_job=fake_run_job)
    step.config["audio_separator_vocal_midi"] = True
    step.config["audio_separator_vocal_midi_batch_size"] = 6
    step.config["audio_separator_vocal_midi_seg_threshold"] = 0.3
    step.config["audio_separator_vocal_midi_seg_radius"] = 0.05
    step.config["audio_separator_vocal_midi_t0"] = 0.1
    step.config["audio_separator_vocal_midi_nsteps"] = 10
    step.config["audio_separator_vocal_midi_est_threshold"] = 0.4

    asyncio.run(step._start_audio_separator())

    assert notifications == []
    assert captured["script_key"] == "module.audio_separator"
    assert "--vocal_midi" in captured["args"]
    assert "--vocal_midi_repo_id=bdsqlsz/GAME-1.0-large-ONNX" in captured["args"]
    assert "--vocal_midi_language=zh" in captured["args"]
    assert "--vocal_midi_output_formats=mid,csv" in captured["args"]
    assert "--vocal_midi_batch_size=6" in captured["args"]
    assert "--vocal_midi_seg_threshold=0.3" in captured["args"]
    assert "--vocal_midi_seg_radius=0.05" in captured["args"]
    assert "--vocal_midi_t0=0.1" in captured["args"]
    assert "--vocal_midi_nsteps=10" in captured["args"]
    assert "--vocal_midi_est_threshold=0.4" in captured["args"]
    assert captured["kwargs"]["runner_kwargs"]["uv_extra_args"] == ["--extra", "vocal-midi"]


def test_tools_step_audio_separator_requires_existing_input(monkeypatch, tmp_path):
    step = ToolsStep()
    missing = tmp_path / "missing"
    notifications = []
    run_calls = []

    async def fake_run_job(*args, **kwargs):
        run_calls.append((args, kwargs))
        return SimpleNamespace(status="ok")

    monkeypatch.setattr(step6_tools.ui, "notify", lambda message, **kwargs: notifications.append((message, kwargs)))

    step.audio_separator_input = SimpleNamespace(value=str(missing))
    step.audio_separator_output_format = SimpleNamespace(value="wav")
    step.panel = SimpleNamespace(run_job=fake_run_job)

    asyncio.run(step._start_audio_separator())

    assert run_calls == []
    assert notifications
    assert notifications[-1][1]["type"] == "warning"
def test_run_audio_separator_batches_harmony_phase_after_primary(monkeypatch, tmp_path):
    input_dir = tmp_path / "music"
    input_dir.mkdir()
    (input_dir / "song_a.wav").write_bytes(b"a")
    (input_dir / "song_b.wav").write_bytes(b"b")

    calls = []
    writes = []

    class _FakeSeparator:
        def __init__(self, *, repo_id, model_dir, force_download, logger=None):
            self.repo_id = repo_id
            self.model_tag = "harmony" if repo_id == DEFAULT_HARMONY_SEPARATOR_REPO_ID else "primary"
            self.providers = ("CPUExecutionProvider",)
            self.metadata = SimpleNamespace(default_segment_size=801, sample_rate=44100)

        def separate_file(self, source_path, *, segment_size, overlap, batch_size):
            calls.append(
                ("separate", self.repo_id, Path(source_path).name, int(segment_size), float(overlap), int(batch_size))
            )
            if self.repo_id == DEFAULT_HARMONY_SEPARATOR_REPO_ID:
                return {
                    "Vocals": torch.full((2, 16000), 0.2, dtype=torch.float32),
                    "Instrumental": torch.full((2, 16000), 0.1, dtype=torch.float32),
                }

            vocal_level = 0.05 if Path(source_path).stem == "song_a" else 0.0
            return {
                "vocals": torch.full((2, 16000), vocal_level, dtype=torch.float32),
                "drums": torch.full((2, 16000), 0.02, dtype=torch.float32),
            }

        def write_audio(self, waveform, output_path, *, output_format):
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"out")
            writes.append((self.repo_id, output_path.relative_to(input_dir), output_format))
            return output_path

    monkeypatch.setattr("module.audio_separator.AudioSeparator", _FakeSeparator)
    monkeypatch.setattr("module.audio_separator.console.print", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "module.audio_separator.collect_audio_inputs",
        lambda path: ([input_dir / "song_a.wav", input_dir / "song_b.wav"], input_dir),
    )

    args = build_parser().parse_args([str(input_dir), "--harmony_separation"])

    result = run_audio_separator(args)

    assert result == 0
    primary_calls = [call for call in calls if call[1] != DEFAULT_HARMONY_SEPARATOR_REPO_ID]
    harmony_calls = [call for call in calls if call[1] == DEFAULT_HARMONY_SEPARATOR_REPO_ID]
    assert [call[2] for call in primary_calls] == ["song_a.wav", "song_b.wav"]
    assert [call[2] for call in harmony_calls] == ["song_a_(vocals)_primary.wav", "song_b_(vocals)_primary.wav"]
    assert all(call[3] == 801 for call in harmony_calls)
    assert calls.index(harmony_calls[0]) > calls.index(primary_calls[-1])

    harmony_outputs = [str(path) for repo_id, path, _ in writes if repo_id == DEFAULT_HARMONY_SEPARATOR_REPO_ID]
    assert any(HARMONY_OUTPUT_DIRNAME in path and "dry_vocal" in path for path in harmony_outputs)
    assert any(HARMONY_OUTPUT_DIRNAME in path and "harmony" in path for path in harmony_outputs)


def test_run_audio_separator_vocal_midi_uses_current_run_input_path_without_rescan(monkeypatch, tmp_path):
    input_dir = tmp_path / "music"
    input_dir.mkdir()
    source_path = input_dir / "song.wav"
    source_path.write_bytes(b"a")

    stale_harmony_dir = input_dir / "song" / HARMONY_OUTPUT_DIRNAME
    stale_harmony_dir.mkdir(parents=True)
    stale_old = stale_harmony_dir / "a_(dry_vocal)_old.wav"
    stale_old.write_bytes(b"old")

    transcribe_calls = []

    class _FakeSeparator:
        def __init__(self, *, repo_id, model_dir, force_download, logger=None):
            self.repo_id = repo_id
            self.model_tag = "harmony" if repo_id == DEFAULT_HARMONY_SEPARATOR_REPO_ID else "primary"
            self.providers = ("CPUExecutionProvider",)
            self.metadata = SimpleNamespace(default_segment_size=801, sample_rate=44100)

        def separate_file(self, source_path, *, segment_size, overlap, batch_size):
            if self.repo_id == DEFAULT_HARMONY_SEPARATOR_REPO_ID:
                return {
                    "Vocals": torch.full((2, 16000), 0.2, dtype=torch.float32),
                    "Instrumental": torch.full((2, 16000), 0.1, dtype=torch.float32),
                }
            return {
                "vocals": torch.full((2, 16000), 0.05, dtype=torch.float32),
                "drums": torch.full((2, 16000), 0.02, dtype=torch.float32),
            }

        def write_audio(self, waveform, output_path, *, output_format):
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"out")
            return output_path

    class _FakeTranscriber:
        def __init__(self, *, repo_id, model_dir, force_download, logger=None):
            self.providers = ("CPUExecutionProvider",)

        def transcribe_file(self, input_path, **kwargs):
            transcribe_calls.append(Path(input_path))
            return {"mid": Path(kwargs["output_dir"]) / "song.mid"}

    monkeypatch.setattr("module.audio_separator.AudioSeparator", _FakeSeparator)
    monkeypatch.setattr("module.audio_separator.GameOnnxTranscriber", _FakeTranscriber)
    monkeypatch.setattr("module.audio_separator.console.print", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "module.audio_separator.collect_audio_inputs",
        lambda path: ([source_path], input_dir),
    )

    args = build_parser().parse_args([str(input_dir), "--harmony_separation", "--vocal_midi", "--overwrite"])

    result = run_audio_separator(args)

    assert result == 0
    assert transcribe_calls == [input_dir / "song" / HARMONY_OUTPUT_DIRNAME / "song_(dry_vocal)_harmony.wav"]
    assert transcribe_calls[0] != stale_old


def test_build_vocal_midi_output_dir_without_model_name(tmp_path):
    song_output_dir = tmp_path / "song_output"

    assert build_vocal_midi_output_dir(song_output_dir, "bdsqlsz/GAME-1.0-small-ONNX") == song_output_dir / "03_vocal_midi"


def test_build_bs_roformer_kwargs_matches_reference_zero_dc_default():
    kwargs = build_bs_roformer_kwargs(
        {
            "audio": {"num_channels": 2},
            "model": {
                "dim": 256,
                "depth": 12,
                "num_stems": 6,
                "time_transformer_depth": 1,
                "freq_transformer_depth": 1,
                "freqs_per_bands": (2, 4, 8),
                "dim_head": 64,
                "heads": 8,
                "attn_dropout": 0.1,
                "ff_dropout": 0.1,
                "dim_freqs_in": 1025,
                "stft_n_fft": 2048,
                "stft_hop_length": 512,
                "stft_win_length": 2048,
                "mask_estimator_depth": 2,
                "multi_stft_resolution_loss_weight": 1.0,
                "multi_stft_resolutions_window_sizes": (4096, 2048),
                "multi_stft_hop_size": 147,
                "multi_stft_normalized": False,
            },
        }
    )

    assert kwargs["zero_dc"] is True
    assert kwargs["num_residual_streams"] == 1
    assert kwargs["flash_attn"] is False


def test_build_export_metadata_prefers_inference_dim_t_for_chunk_size(tmp_path):
    metadata = build_export_metadata(
        {
            "audio": {
                "chunk_size": 588800,
                "num_channels": 2,
                "sample_rate": 44100,
                "hop_length": 441,
                "n_fft": 2048,
            },
            "model": {
                "num_stems": 6,
                "stft_n_fft": 2048,
                "stft_hop_length": 512,
                "stft_win_length": 2048,
            },
            "training": {
                "instruments": ["bass", "drums", "other", "vocals", "guitar", "piano"],
            },
            "inference": {
                "dim_t": 1101,
            },
        },
        checkpoint_path=tmp_path / "model.ckpt",
        output_path=tmp_path / "model.onnx",
    )

    assert metadata["chunk_size"] == 563200


def test_create_dummy_stft_features_matches_reference_layout():
    config = {
        "audio": {
            "chunk_size": 56,
            "num_channels": 2,
        },
        "model": {
            "stft_n_fft": 8,
            "stft_hop_length": 4,
            "stft_win_length": 8,
            "stft_normalized": False,
        },
    }

    torch.manual_seed(1234)
    raw_audio = torch.randn(1, 2, 56, device="cpu")
    flat_audio = raw_audio.reshape(-1, 56)
    window = torch.hann_window(8)
    stft = torch.stft(
        flat_audio,
        n_fft=8,
        hop_length=4,
        win_length=8,
        normalized=False,
        window=window,
        return_complex=True,
    )
    expected = rearrange(torch.view_as_real(stft).reshape(1, 2, stft.shape[1], stft.shape[2], 2), "b s f t c -> b t (f s c)")

    torch.manual_seed(1234)
    features = create_dummy_stft_features(config, torch.device("cpu"))

    assert torch.allclose(features, expected)


def test_detect_roformer_model_type_identifies_mel_band_config():
    model_type = detect_roformer_model_type({"model": {"num_bands": 60}})

    assert model_type == MODEL_TYPE_MEL_BAND_ROFORMER


def test_build_mel_band_roformer_kwargs_applies_compatibility_defaults():
    kwargs = build_mel_band_roformer_kwargs(
        {
            "audio": {"num_channels": 2},
            "model": {
                "dim": 384,
                "depth": 6,
                "num_bands": 60,
                "time_transformer_depth": 1,
                "freq_transformer_depth": 1,
                "dim_head": 64,
                "heads": 8,
                "attn_dropout": 0.0,
                "ff_dropout": 0.0,
                "dim_freqs_in": 1025,
                "sample_rate": 44100,
                "stft_n_fft": 2048,
                "stft_hop_length": 441,
                "stft_win_length": 2048,
                "mask_estimator_depth": 2,
                "multi_stft_resolution_loss_weight": 1.0,
                "multi_stft_resolutions_window_sizes": (4096, 2048),
                "multi_stft_hop_size": 147,
                "multi_stft_normalized": False,
            },
            "training": {
                "instruments": ["Vocals", "Instrumental"],
                "target_instrument": "Vocals",
            },
        }
    )

    assert kwargs["num_stems"] == 1
    assert kwargs["flash_attn"] is False
    assert kwargs["linear_transformer_depth"] == 0
    assert kwargs["num_residual_streams"] == 1
    assert kwargs["add_value_residual"] is False
    assert kwargs["zero_dc"] is True


def test_rewrite_checkpoint_for_latest_mel_band_roformer_shifts_layer_slots_and_branches():
    state_dict = {
        "layers.0.0.layers.0.0.norm.gamma": torch.tensor([1.0]),
        "layers.0.1.layers.0.0.to_qkv.weight": torch.tensor([2.0]),
    }

    rewritten = rewrite_checkpoint_for_latest_mel_band_roformer(state_dict)

    assert "layers.0.1.layers.0.0.branch.norm.gamma" in rewritten
    assert "layers.0.2.layers.0.0.branch.to_qkv.weight" in rewritten


def test_build_export_metadata_single_target_keeps_primary_and_secondary_stems(tmp_path):
    metadata = build_export_metadata(
        {
            "audio": {
                "chunk_size": 352800,
                "num_channels": 2,
                "sample_rate": 44100,
                "n_fft": 2048,
            },
            "model": {
                "dim": 384,
                "depth": 6,
                "num_bands": 60,
                "num_stems": 1,
                "sample_rate": 44100,
                "stft_n_fft": 2048,
                "stft_hop_length": 441,
                "stft_win_length": 2048,
            },
            "training": {
                "instruments": ["Vocals", "Instrumental"],
                "target_instrument": "Vocals",
            },
            "inference": {
                "dim_t": 801,
            },
        },
        checkpoint_path=tmp_path / "model.ckpt",
        output_path=tmp_path / "model.onnx",
    )

    assert metadata["model_type"] == MODEL_TYPE_MEL_BAND_ROFORMER
    assert metadata["num_stems"] == 1
    assert metadata["stem_names"] == ["Vocals"]
    assert metadata["secondary_stem_name"] == "Instrumental"
    assert metadata["stft"]["zero_dc"] is True


def test_finalize_stem_outputs_adds_secondary_residual_for_single_target():
    metadata = AudioSeparatorMetadata(
        repo_id="demo/repo",
        input_name="stft_features",
        output_name="mask",
        input_layout=("batch", "frames", "freq_channels_complex"),
        output_layout=("batch", "stems", "freq_channels", "frames", "complex"),
        sample_rate=44100,
        chunk_size=16,
        num_channels=2,
        num_stems=1,
        stem_names=("Vocals",),
        n_fft=8,
        hop_length=4,
        win_length=8,
        normalized=False,
        secondary_stem_name="Instrumental",
    )
    separated = torch.full((1, 2, 4), 0.25, dtype=torch.float32)
    mix = torch.ones((2, 4), dtype=torch.float32)

    outputs = finalize_stem_outputs(separated, mix, metadata)

    assert tuple(outputs.keys()) == ("Vocals", "Instrumental")
    assert torch.allclose(outputs["Vocals"], torch.full((2, 4), 0.25))
    assert torch.allclose(outputs["Instrumental"], torch.full((2, 4), 0.75))
