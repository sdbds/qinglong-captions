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

from module.audio_separator import build_parser, build_song_output_dir, collect_audio_inputs
from module.audio_separator_core import (
    AudioSeparatorMetadata,
    build_stft_features,
    compute_chunk_positions,
    load_audio_separator_metadata_file,
    mask_to_complex_tensor,
    overlap_seconds_to_step_size,
    segment_size_to_chunk_size,
)
from utils.onnx_export import build_bs_roformer_kwargs, build_export_metadata, create_dummy_stft_features

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
