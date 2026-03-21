from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import imageio_ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from config.loader import load_config
from module.onnx_runtime import OnnxModelSpec, OnnxRuntimeConfig, load_single_model_bundle, resolve_tool_runtime_config

DEFAULT_AUDIO_SEPARATOR_REPO_ID = "bdsqlsz/BS-ROFO-SW-Fixed-ONNX"
DEFAULT_AUDIO_SEPARATOR_MODEL_DIR = "audio_separator"
DEFAULT_OUTPUT_FORMAT = "wav"
DEFAULT_SEGMENT_SIZE = 1151
DEFAULT_OVERLAP = 8
DEFAULT_BATCH_SIZE = 1
METADATA_FILENAME = "model.json"
MODEL_FILENAME = "model.onnx"
SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")
SUPPORTED_OUTPUT_FORMATS = ("wav", "flac", "mp3")
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@dataclass(frozen=True)
class AudioSeparatorMetadata:
    repo_id: str
    input_name: str
    output_name: str
    input_layout: tuple[str, ...]
    output_layout: tuple[str, ...]
    sample_rate: int
    chunk_size: int
    num_channels: int
    num_stems: int
    stem_names: tuple[str, ...]
    n_fft: int
    hop_length: int
    win_length: int
    normalized: bool

    @classmethod
    def from_dict(cls, repo_id: str, payload: dict[str, Any]) -> "AudioSeparatorMetadata":
        stft = payload.get("stft", {})
        metadata = cls(
            repo_id=repo_id,
            input_name=str(payload.get("input_name", "stft_features")),
            output_name=str(payload.get("output_name", "mask")),
            input_layout=tuple(str(item) for item in payload.get("input_layout", ()) or ()),
            output_layout=tuple(str(item) for item in payload.get("output_layout", ()) or ()),
            sample_rate=int(payload.get("sample_rate", 44100)),
            chunk_size=int(payload.get("chunk_size", 0)),
            num_channels=int(payload.get("num_channels", 2)),
            num_stems=int(payload.get("num_stems", 0)),
            stem_names=tuple(str(item) for item in payload.get("stem_names", ()) or ()),
            n_fft=int(stft.get("n_fft", 2048)),
            hop_length=int(stft.get("hop_length", 512)),
            win_length=int(stft.get("win_length", 2048)),
            normalized=bool(stft.get("normalized", False)),
        )
        metadata.validate()
        return metadata

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"Invalid chunk_size: {self.chunk_size!r}")
        if self.sample_rate <= 0:
            raise ValueError(f"Invalid sample_rate: {self.sample_rate!r}")
        if self.num_channels <= 0:
            raise ValueError(f"Invalid num_channels: {self.num_channels!r}")
        if self.num_stems <= 0:
            raise ValueError(f"Invalid num_stems: {self.num_stems!r}")
        if len(self.stem_names) != self.num_stems:
            raise ValueError(
                f"Stem name count mismatch: expected {self.num_stems}, got {len(self.stem_names)}"
            )
        if self.hop_length <= 0 or self.n_fft <= 0 or self.win_length <= 0:
            raise ValueError(
                f"Invalid STFT config: n_fft={self.n_fft}, hop_length={self.hop_length}, win_length={self.win_length}"
            )

    @property
    def freq_bins(self) -> int:
        return (self.n_fft // 2) + 1

    @property
    def freq_channels(self) -> int:
        return self.num_channels * self.freq_bins

    @property
    def feature_size(self) -> int:
        return self.freq_channels * 2

    @property
    def default_segment_size(self) -> int:
        return (self.chunk_size // self.hop_length) + 1


def derive_model_tag(repo_id: str) -> str:
    slug = repo_id.rsplit("/", 1)[-1].strip() or "model"
    if slug.lower().endswith("-onnx"):
        slug = slug[:-5]
    return slug or "model"


def build_local_model_dir(model_dir: str | Path, repo_id: str) -> Path:
    return Path(model_dir) / repo_id.replace("/", "_")


def load_audio_separator_metadata_file(path: str | Path, *, repo_id: str) -> AudioSeparatorMetadata:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return AudioSeparatorMetadata.from_dict(repo_id, payload)


def download_audio_separator_metadata(
    repo_id: str,
    *,
    local_dir: str | Path,
    force_download: bool = False,
    downloader: Callable[..., str] | None = None,
) -> tuple[AudioSeparatorMetadata, Path]:
    downloader = downloader or hf_hub_download
    metadata_path = Path(
        downloader(
            repo_id=repo_id,
            filename=METADATA_FILENAME,
            local_dir=str(Path(local_dir)),
            force_download=force_download,
        )
    )
    return load_audio_separator_metadata_file(metadata_path, repo_id=repo_id), metadata_path


def resolve_audio_separator_runtime_config(
    *,
    force_download: bool = False,
    config_dir: str | Path = CONFIG_DIR,
) -> OnnxRuntimeConfig:
    config = load_config(str(config_dir))
    return resolve_tool_runtime_config(
        config,
        tool_name="audio_separator",
        cli_override={"force_download": force_download},
    )


def segment_size_to_chunk_size(segment_size: int, hop_length: int) -> int:
    if int(segment_size) <= 1:
        raise ValueError(f"segment_size must be > 1, got {segment_size!r}")
    if int(hop_length) <= 0:
        raise ValueError(f"hop_length must be > 0, got {hop_length!r}")
    return int(hop_length) * (int(segment_size) - 1)


def compute_chunk_positions(total_samples: int, chunk_size: int, overlap: int) -> tuple[list[int], int]:
    if total_samples < 0:
        raise ValueError(f"total_samples must be >= 0, got {total_samples!r}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size!r}")
    if overlap <= 0:
        raise ValueError(f"overlap must be > 0, got {overlap!r}")

    step_size = max(1, chunk_size // overlap)
    if total_samples <= chunk_size:
        return [0], step_size

    last_start = total_samples - chunk_size
    starts = list(range(0, last_start + 1, step_size))
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return starts, step_size


def build_stft_features(
    waveform: torch.Tensor,
    metadata: AudioSeparatorMetadata,
    *,
    window: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if waveform.ndim != 2:
        raise ValueError(f"waveform must have shape [channels, samples], got {tuple(waveform.shape)}")
    if waveform.shape[0] != metadata.num_channels:
        raise ValueError(f"Expected {metadata.num_channels} channels, got {waveform.shape[0]}")

    stft_window = window if window is not None else torch.hann_window(metadata.win_length, dtype=torch.float32)
    stft = torch.stft(
        waveform,
        n_fft=metadata.n_fft,
        hop_length=metadata.hop_length,
        win_length=metadata.win_length,
        normalized=metadata.normalized,
        window=stft_window,
        return_complex=True,
    )
    stft_real = torch.view_as_real(stft).contiguous()
    _, freqs, frames, _ = stft_real.shape
    features = stft_real.reshape(metadata.num_channels * freqs, frames, 2).permute(1, 0, 2).reshape(frames, -1)
    return features.to(dtype=torch.float32).contiguous(), stft.contiguous()


def mask_to_complex_tensor(mask: np.ndarray | torch.Tensor, metadata: AudioSeparatorMetadata) -> torch.Tensor:
    mask_tensor = torch.as_tensor(mask, dtype=torch.float32)
    if mask_tensor.ndim != 4:
        raise ValueError(f"Mask output must have shape [stems, freq_channels, frames, complex], got {tuple(mask_tensor.shape)}")
    stems, freq_channels, _, complex_dim = mask_tensor.shape
    if stems != metadata.num_stems:
        raise ValueError(f"Expected {metadata.num_stems} stems, got {stems}")
    if freq_channels != metadata.freq_channels:
        raise ValueError(f"Expected {metadata.freq_channels} freq_channels, got {freq_channels}")
    if complex_dim != 2:
        raise ValueError(f"Expected complex dimension 2, got {complex_dim}")

    reshaped = mask_tensor.reshape(
        metadata.num_stems,
        metadata.num_channels,
        metadata.freq_bins,
        mask_tensor.shape[2],
        2,
    ).contiguous()
    return torch.view_as_complex(reshaped)


def reconstruct_chunk_waveforms(
    mask: np.ndarray | torch.Tensor,
    stft: torch.Tensor,
    metadata: AudioSeparatorMetadata,
    *,
    length: int,
    window: torch.Tensor | None = None,
) -> torch.Tensor:
    if stft.ndim != 3:
        raise ValueError(f"stft must have shape [channels, freq, frames], got {tuple(stft.shape)}")

    mask_complex = mask_to_complex_tensor(mask, metadata)
    stft_frames = stft.shape[-1]
    mask_frames = mask_complex.shape[-1]
    if stft_frames != mask_frames:
        frame_count = min(stft_frames, mask_frames)
        stft = stft[..., :frame_count]
        mask_complex = mask_complex[..., :frame_count]

    stft_window = window if window is not None else torch.hann_window(metadata.win_length, dtype=torch.float32)
    chunk_stft = mask_complex * stft.unsqueeze(0)
    outputs = []
    for stem_index in range(metadata.num_stems):
        outputs.append(
            torch.istft(
                chunk_stft[stem_index],
                n_fft=metadata.n_fft,
                hop_length=metadata.hop_length,
                win_length=metadata.win_length,
                normalized=metadata.normalized,
                window=stft_window,
                length=length,
            )
        )
    return torch.stack(outputs, dim=0).to(dtype=torch.float32)


def _run_ffmpeg(command: list[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def read_audio_via_ffmpeg(
    source_path: str | Path,
    *,
    ffmpeg_executable: str,
    sample_rate: int,
    channels: int,
) -> torch.Tensor:
    command = [
        ffmpeg_executable,
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(source_path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-",
    ]
    result = _run_ffmpeg(command)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip() or "ffmpeg decode failed")

    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(f"No audio samples decoded from {source_path}")
    usable = audio.size - (audio.size % channels)
    if usable <= 0:
        raise RuntimeError(f"Decoded audio shape is invalid for {source_path}")

    audio = audio[:usable].reshape(-1, channels).T.copy()
    return torch.from_numpy(audio)


def _ffmpeg_encoder_args(output_format: str) -> list[str]:
    normalized = output_format.lower()
    if normalized == "wav":
        return ["-c:a", "pcm_s16le"]
    if normalized == "flac":
        return ["-c:a", "flac"]
    if normalized == "mp3":
        return ["-c:a", "libmp3lame", "-q:a", "2"]
    raise ValueError(f"Unsupported output format: {output_format}")


def normalize_for_export(waveform: torch.Tensor) -> torch.Tensor:
    peak = float(waveform.abs().max().item()) if waveform.numel() else 0.0
    if peak > 1.0:
        return waveform / peak
    return waveform


def write_audio_via_ffmpeg(
    waveform: torch.Tensor,
    output_path: str | Path,
    *,
    ffmpeg_executable: str,
    sample_rate: int,
    output_format: str,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    waveform = normalize_for_export(waveform.detach().cpu().to(dtype=torch.float32).contiguous())
    if waveform.ndim != 2:
        raise ValueError(f"waveform must have shape [channels, samples], got {tuple(waveform.shape)}")

    interleaved = waveform.T.contiguous().numpy().astype(np.float32, copy=False)
    command = [
        ffmpeg_executable,
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        str(waveform.shape[0]),
        "-ar",
        str(sample_rate),
        "-i",
        "-",
        *_ffmpeg_encoder_args(output_format),
        str(output_path),
    ]
    result = _run_ffmpeg(command, input_bytes=interleaved.tobytes())
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip() or "ffmpeg encode failed")
    return output_path


class AudioSeparator:
    def __init__(
        self,
        *,
        repo_id: str = DEFAULT_AUDIO_SEPARATOR_REPO_ID,
        model_dir: str | Path = DEFAULT_AUDIO_SEPARATOR_MODEL_DIR,
        force_download: bool = False,
        runtime_config: OnnxRuntimeConfig | None = None,
        config_dir: str | Path = CONFIG_DIR,
        metadata_downloader: Callable[..., str] | None = None,
        artifact_loader: Callable[..., Path] | None = None,
        session_bundle_loader: Callable[..., Any] | None = None,
        ffmpeg_executable: str | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.model_dir = build_local_model_dir(model_dir, repo_id)
        self.runtime_config = runtime_config or resolve_audio_separator_runtime_config(
            force_download=force_download,
            config_dir=config_dir,
        )
        self.metadata, self.metadata_path = download_audio_separator_metadata(
            repo_id,
            local_dir=self.model_dir,
            force_download=self.runtime_config.force_download,
            downloader=metadata_downloader,
        )
        self.spec = OnnxModelSpec(
            repo_id=repo_id,
            onnx_filename=MODEL_FILENAME,
            local_dir=self.model_dir,
            bundle_key=f"audio_separator:{repo_id}",
        )
        self.bundle = load_single_model_bundle(
            spec=self.spec,
            runtime_config=self.runtime_config,
            artifact_loader=artifact_loader,
            session_bundle_loader=session_bundle_loader,
        )
        self.session = self.bundle.session
        self.providers = tuple(self.bundle.providers)
        self.model_tag = derive_model_tag(repo_id)
        self.ffmpeg_executable = ffmpeg_executable or imageio_ffmpeg.get_ffmpeg_exe()
        self.input_name = self.bundle.input_metas[0].name if self.bundle.input_metas else self.metadata.input_name
        outputs = tuple(self.session.get_outputs()) if hasattr(self.session, "get_outputs") else ()
        self.output_name = outputs[0].name if outputs else self.metadata.output_name
        self._stft_window = torch.hann_window(self.metadata.win_length, dtype=torch.float32)

    def read_audio(self, source_path: str | Path) -> torch.Tensor:
        return read_audio_via_ffmpeg(
            source_path,
            ffmpeg_executable=self.ffmpeg_executable,
            sample_rate=self.metadata.sample_rate,
            channels=self.metadata.num_channels,
        )

    def write_audio(self, waveform: torch.Tensor, output_path: str | Path, *, output_format: str) -> Path:
        return write_audio_via_ffmpeg(
            waveform,
            output_path,
            ffmpeg_executable=self.ffmpeg_executable,
            sample_rate=self.metadata.sample_rate,
            output_format=output_format,
        )

    def separate_file(
        self,
        source_path: str | Path,
        *,
        segment_size: int = DEFAULT_SEGMENT_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> dict[str, torch.Tensor]:
        waveform = self.read_audio(source_path)
        return self.separate_waveform(
            waveform,
            segment_size=segment_size,
            overlap=overlap,
            batch_size=batch_size,
        )

    def separate_waveform(
        self,
        waveform: torch.Tensor,
        *,
        segment_size: int = DEFAULT_SEGMENT_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size!r}")

        chunk_size = segment_size_to_chunk_size(int(segment_size), self.metadata.hop_length)
        starts, _ = compute_chunk_positions(int(waveform.shape[-1]), chunk_size, int(overlap))
        mix = waveform.to(dtype=torch.float32).contiguous()
        total_samples = int(mix.shape[-1])
        accumulator = torch.zeros(
            self.metadata.num_stems,
            self.metadata.num_channels,
            total_samples,
            dtype=torch.float32,
        )
        counter = torch.zeros(total_samples, dtype=torch.float32)
        chunk_window = torch.hamming_window(chunk_size, periodic=False, dtype=torch.float32)

        with torch.inference_mode():
            for index in range(0, len(starts), int(batch_size)):
                batch_starts = starts[index : index + int(batch_size)]
                batch_features: list[torch.Tensor] = []
                batch_stfts: list[torch.Tensor] = []
                batch_lengths: list[int] = []

                for start in batch_starts:
                    chunk = mix[:, start : start + chunk_size]
                    valid_length = min(chunk_size, total_samples - start)
                    if chunk.shape[-1] < chunk_size:
                        chunk = F.pad(chunk, (0, chunk_size - chunk.shape[-1]))
                    features, stft = build_stft_features(chunk, self.metadata, window=self._stft_window)
                    batch_features.append(features)
                    batch_stfts.append(stft)
                    batch_lengths.append(valid_length)

                feed = np.ascontiguousarray(torch.stack(batch_features, dim=0).cpu().numpy())
                outputs = self.session.run([self.output_name], {self.input_name: feed})[0]

                for batch_item, start in enumerate(batch_starts):
                    stem_chunk = reconstruct_chunk_waveforms(
                        outputs[batch_item],
                        batch_stfts[batch_item],
                        self.metadata,
                        length=chunk_size,
                        window=self._stft_window,
                    )
                    valid_length = batch_lengths[batch_item]
                    weights = chunk_window[:valid_length]
                    accumulator[..., start : start + valid_length] += stem_chunk[..., :valid_length] * weights
                    counter[start : start + valid_length] += weights

        separated = accumulator / counter.clamp_min(1e-8).view(1, 1, -1)
        return {
            stem_name: separated[index].clone()
            for index, stem_name in enumerate(self.metadata.stem_names)
        }
