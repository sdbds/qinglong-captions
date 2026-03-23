from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import imageio_ffmpeg
import numpy as np
from rich.console import Console
from rich.pretty import Pretty
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.loader import load_config
from module.onnx_runtime import (
    OnnxMultiModelSpec,
    OnnxRuntimeConfig,
    build_local_model_dir,
    load_multi_model_bundle,
    resolve_tool_runtime_config,
)
from utils.console_util import print_exception
from utils.path_safety import safe_child_path, safe_leaf_name

console = Console(color_system="truecolor", force_terminal=True)

DEFAULT_GAME_MODEL_REPO_ID = "bdsqlsz/GAME-1.0-large-ONNX"
DEFAULT_VOCAL_MIDI_MODEL_DIR = "vocal_midi"
DEFAULT_VOCAL_MIDI_BATCH_SIZE = 4
DEFAULT_VOCAL_MIDI_SEG_THRESHOLD = 0.2
DEFAULT_VOCAL_MIDI_SEG_RADIUS = 0.02
DEFAULT_VOCAL_MIDI_T0 = 0.0
DEFAULT_VOCAL_MIDI_NSTEPS = 8
DEFAULT_VOCAL_MIDI_EST_THRESHOLD = 0.2
DEFAULT_VOCAL_MIDI_OUTPUT_FORMATS = "mid"
VOCAL_MIDI_OUTPUT_DIRNAME = "03_vocal_midi"
CONFIG_FILENAME = "config.json"
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac")
SUPPORTED_OUTPUT_FORMATS = ("mid", "txt", "csv")
HARMONY_OUTPUT_DIRNAME = "02_harmony_split"

GAME_ONNX_MODEL_LABELS: dict[str, str] = {
    "bdsqlsz/GAME-1.0-small-ONNX": "GAME-1.0-small-ONNX",
    "bdsqlsz/GAME-1.0-medium-ONNX": "GAME-1.0-medium-ONNX",
    "bdsqlsz/GAME-1.0-large-ONNX": "GAME-1.0-large-ONNX",
}

GAME_ONNX_ARTIFACTS = {
    "encoder": "encoder.onnx",
    "segmenter": "segmenter.onnx",
    "estimator": "estimator.onnx",
    "dur2bd": "dur2bd.onnx",
    "bd2dur": "bd2dur.onnx",
}


def derive_game_model_label(repo_id: str) -> str:
    return GAME_ONNX_MODEL_LABELS.get(repo_id, repo_id.rsplit("/", 1)[-1].strip() or "GAME-ONNX")


@dataclass(frozen=True)
class GameOnnxConfig:
    repo_id: str
    samplerate: int
    timestep: float
    languages: Mapping[str, int]
    loop: bool
    embedding_dim: int

    @classmethod
    def from_dict(cls, repo_id: str, payload: Mapping[str, Any]) -> "GameOnnxConfig":
        languages = payload.get("languages", {}) or {}
        return cls(
            repo_id=repo_id,
            samplerate=int(payload.get("samplerate", 44100)),
            timestep=float(payload.get("timestep", 0.01)),
            languages={str(key): int(value) for key, value in languages.items()},
            loop=bool(payload.get("loop", True)),
            embedding_dim=int(payload.get("embedding_dim", 256)),
        )


@dataclass(frozen=True)
class NoteInfo:
    onset: float
    offset: float
    pitch: float


def load_game_config_file(path: str | Path, *, repo_id: str) -> GameOnnxConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return GameOnnxConfig.from_dict(repo_id, payload)


def resolve_vocal_midi_runtime_config(
    *,
    force_download: bool = False,
    config_dir: str | Path = CONFIG_DIR,
) -> OnnxRuntimeConfig:
    config = load_config(str(config_dir))
    return resolve_tool_runtime_config(
        config,
        tool_name="vocal_midi",
        cli_override={"force_download": force_download},
    )


def parse_output_formats(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ("mid",)
    raw_values = [item.strip().lower() for item in value.split(",")] if isinstance(value, str) else [str(item).strip().lower() for item in value]
    parsed: list[str] = []
    for item in raw_values:
        if not item:
            continue
        if item not in SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(f"Unsupported output format: {item!r}. Supported: {', '.join(SUPPORTED_OUTPUT_FORMATS)}")
        if item not in parsed:
            parsed.append(item)
    return tuple(parsed or ("mid",))


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
    channels: int = 1,
) -> np.ndarray:
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
    audio = audio[:usable].reshape(-1, channels)
    if channels == 1:
        return audio[:, 0].copy()
    return audio.T.copy()


def get_rms(
    y: np.ndarray,
    *,
    frame_length: int = 2048,
    hop_length: int = 512,
    pad_mode: str = "constant",
) -> np.ndarray:
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)
    out_strides = y.strides + (y.strides[-1],)
    out_shape = (y.shape[0] - frame_length + 1, frame_length)
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
    framed = xw[::hop_length]
    power = np.mean(np.abs(framed) ** 2, axis=-1, keepdims=True)
    return np.sqrt(power).T


class Slicer:
    def __init__(
        self,
        sr: int,
        threshold: float = -40.0,
        min_length: int = 1000,
        min_interval: int = 200,
        hop_size: int = 20,
        max_sil_kept: int = 100,
    ) -> None:
        if not min_length >= min_interval >= hop_size:
            raise ValueError("min_length >= min_interval >= hop_size must hold")
        if not max_sil_kept >= hop_size:
            raise ValueError("max_sil_kept >= hop_size must hold")
        min_interval_samples = sr * min_interval / 1000
        self.sr = sr
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval_samples), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval_samples / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform: np.ndarray, begin: int, end: int) -> dict[str, Any]:
        start_sample = begin * self.hop_size
        end_sample = min(waveform.shape[-1], end * self.hop_size)
        return {
            "offset": start_sample / self.sr,
            "waveform": waveform[start_sample:end_sample],
        }

    def slice(self, waveform: np.ndarray) -> list[dict[str, Any]]:
        if (waveform.shape[0] + self.hop_size - 1) // self.hop_size <= self.min_length:
            return [{"offset": 0.0, "waveform": waveform}]
        rms_list = get_rms(y=waveform, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
        sil_tags: list[tuple[int, int]] = []
        silence_start: int | None = None
        clip_start = 0
        for index, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = index
                continue
            if silence_start is None:
                continue
            is_leading_silence = silence_start == 0 and index > self.max_sil_kept
            need_slice_middle = index - silence_start >= self.min_interval and index - clip_start >= self.min_length
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            if index - silence_start <= self.max_sil_kept:
                pos = int(rms_list[silence_start : index + 1].argmin()) + silence_start
                sil_tags.append((0, pos) if silence_start == 0 else (pos, pos))
                clip_start = pos
            else:
                pos_l = int(rms_list[silence_start : silence_start + self.max_sil_kept + 1].argmin()) + silence_start
                pos_r = int(rms_list[index - self.max_sil_kept : index + 1].argmin()) + index - self.max_sil_kept
                sil_tags.append((0, pos_r) if silence_start == 0 else (pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = int(rms_list[silence_start : silence_end + 1].argmin()) + silence_start
            sil_tags.append((pos, total_frames + 1))
        if not sil_tags:
            return [{"offset": 0.0, "waveform": waveform}]
        chunks: list[dict[str, Any]] = []
        if sil_tags[0][0] > 0:
            chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))
        for index in range(len(sil_tags) - 1):
            chunks.append(self._apply_slice(waveform, sil_tags[index][1], sil_tags[index + 1][0]))
        if sil_tags[-1][1] < total_frames:
            chunks.append(self._apply_slice(waveform, sil_tags[-1][1], total_frames))
        return chunks


def _resolve_language_id(language: str | None, lang_map: Mapping[str, int], repo_id: str) -> int:
    if language in (None, "", "auto"):
        return 0
    if language not in lang_map:
        raise ValueError(
            f"Language '{language}' not supported by {repo_id}. Supported languages: {', '.join(sorted(lang_map.keys()))}"
        )
    return int(lang_map[language])


def _sampling_schedule(t0: float, nsteps: int) -> list[float]:
    step = (1.0 - float(t0)) / int(nsteps)
    return [float(t0) + index * step for index in range(int(nsteps))]


def _radius_to_frames(seg_radius: float, timestep: float) -> np.ndarray:
    return np.array(max(1, round(float(seg_radius) / float(timestep))), dtype=np.int64)


def _scalar_float32(value: float) -> np.ndarray:
    return np.array(value, dtype=np.float32)


def _sanitize_notes(notes: list[NoteInfo]) -> list[NoteInfo]:
    sorted_notes = sorted(notes, key=lambda item: (item.onset, item.offset, item.pitch))
    sanitized: list[NoteInfo] = []
    last_time = 0.0
    for note in sorted_notes:
        onset = max(note.onset, last_time)
        offset = max(note.offset, onset)
        if offset <= onset:
            continue
        sanitized.append(NoteInfo(onset=onset, offset=offset, pitch=note.pitch))
        last_time = offset
    return sanitized


def _midi_to_note_name(pitch: float) -> str:
    midi_pitch = int(round(float(pitch)))
    cents = int(round((float(pitch) - midi_pitch) * 100))
    note_names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    note_name = note_names[midi_pitch % 12]
    octave = (midi_pitch // 12) - 1
    base = f"{note_name}{octave}"
    if cents == 0:
        return base
    sign = "+" if cents > 0 else ""
    return f"{base}{sign}{cents}"


def build_vocal_midi_output_dir(song_output_dir: Path, repo_id: str) -> Path:
    root = safe_child_path(song_output_dir, VOCAL_MIDI_OUTPUT_DIRNAME, default_name=VOCAL_MIDI_OUTPUT_DIRNAME)
    return root


def resolve_vocal_midi_input(source_path: Path, song_output_dir: Path) -> Path:
    song_output_dir = Path(song_output_dir)
    harmony_dir = song_output_dir / HARMONY_OUTPUT_DIRNAME
    if harmony_dir.is_dir():
        dry_vocals = sorted(
            path
            for path in harmony_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
            and "dry_vocal" in path.stem.lower()
        )
        if dry_vocals:
            return dry_vocals[0]

    vocal_stems = sorted(
        path
        for path in song_output_dir.glob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        and "vocals" in path.stem.lower()
    )
    if vocal_stems:
        return vocal_stems[0]
    return Path(source_path)


class GameOnnxTranscriber:
    def __init__(
        self,
        *,
        repo_id: str = DEFAULT_GAME_MODEL_REPO_ID,
        model_dir: str | Path = DEFAULT_VOCAL_MIDI_MODEL_DIR,
        force_download: bool = False,
        runtime_config: OnnxRuntimeConfig | None = None,
        config_dir: str | Path = CONFIG_DIR,
        config_downloader: Callable[..., str] | None = None,
        artifact_loader: Callable[..., dict[str, Path]] | None = None,
        support_file_loader: Callable[..., dict[str, Path]] | None = None,
        session_bundle_loader: Callable[..., Any] | None = None,
        ffmpeg_executable: str | None = None,
        logger: Callable[..., Any] | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.model_dir = build_local_model_dir(model_dir, repo_id)
        self.logger = logger
        self.runtime_config = runtime_config or resolve_vocal_midi_runtime_config(
            force_download=force_download,
            config_dir=config_dir,
        )
        if support_file_loader is None and config_downloader is not None:
            def _config_support_loader(
                inner_repo_id: str,
                files: Mapping[str, str],
                *,
                local_dir: str | Path | None = None,
                force_download: bool = False,
                logger: Callable[..., Any] | None = None,
            ) -> dict[str, Path]:
                del logger
                return {
                    name: Path(
                        config_downloader(
                            repo_id=inner_repo_id,
                            filename=filename,
                            local_dir=str(Path(local_dir)) if local_dir is not None else None,
                            force_download=force_download,
                        )
                    )
                    for name, filename in files.items()
                }

            support_file_loader = _config_support_loader

        self.bundle = load_multi_model_bundle(
            spec=OnnxMultiModelSpec(
                repo_id=repo_id,
                artifacts=GAME_ONNX_ARTIFACTS,
                support_files={"config": CONFIG_FILENAME},
                local_dir=self.model_dir,
                bundle_key=f"vocal_midi:{repo_id}",
            ),
            runtime_config=self.runtime_config,
            artifact_loader=artifact_loader,
            support_file_loader=support_file_loader,
            session_bundle_loader=session_bundle_loader,
            logger=self.logger,
        )
        self.artifacts = self.bundle.artifact_paths
        self.config_path = self.bundle.support_paths["config"]
        self.config = load_game_config_file(self.config_path, repo_id=repo_id)
        self.sessions = self.bundle.sessions
        self.providers = tuple(self.bundle.providers)
        self.model_label = derive_game_model_label(repo_id)
        self.ffmpeg_executable = ffmpeg_executable or imageio_ffmpeg.get_ffmpeg_exe()
        self.slicer = Slicer(sr=self.config.samplerate, threshold=-40.0, min_length=1000, min_interval=200, max_sil_kept=100)

        self.encoder_output_names = tuple(output.name for output in self.sessions["encoder"].get_outputs())
        self.segmenter_input_names = {meta.name for meta in self.sessions["segmenter"].get_inputs()}
        self.segmenter_output_names = tuple(output.name for output in self.sessions["segmenter"].get_outputs())
        self.bd2dur_output_names = tuple(output.name for output in self.sessions["bd2dur"].get_outputs())
        self.estimator_output_names = tuple(output.name for output in self.sessions["estimator"].get_outputs())

    def read_audio(self, source_path: str | Path) -> np.ndarray:
        return read_audio_via_ffmpeg(
            source_path,
            ffmpeg_executable=self.ffmpeg_executable,
            sample_rate=self.config.samplerate,
            channels=1,
        )

    def _run_encoder(self, waveform: np.ndarray, duration: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        outputs = self.sessions["encoder"].run(
            list(self.encoder_output_names),
            {
                "waveform": np.ascontiguousarray(waveform.astype(np.float32, copy=False)),
                "duration": np.ascontiguousarray(duration.astype(np.float32, copy=False)),
            },
        )
        mapped = dict(zip(self.encoder_output_names, outputs))
        return mapped["x_seg"], mapped["x_est"], mapped["maskT"].astype(np.bool_)

    def _run_segmenter(
        self,
        x_seg: np.ndarray,
        known_boundaries: np.ndarray,
        mask_t: np.ndarray,
        *,
        language_id: int,
        seg_threshold: float,
        seg_radius: float,
        t0: float,
        nsteps: int,
    ) -> np.ndarray:
        threshold = _scalar_float32(seg_threshold)
        radius = _radius_to_frames(seg_radius, self.config.timestep)
        boundaries = known_boundaries.astype(np.bool_)
        if "prev_boundaries" in self.segmenter_input_names and "t" in self.segmenter_input_names:
            for sample_t in _sampling_schedule(t0, nsteps):
                feed = {
                    "x_seg": np.ascontiguousarray(x_seg.astype(np.float32, copy=False)),
                    "known_boundaries": known_boundaries.astype(np.bool_),
                    "prev_boundaries": boundaries.astype(np.bool_),
                    "t": np.full((x_seg.shape[0],), sample_t, dtype=np.float32),
                    "maskT": mask_t.astype(np.bool_),
                    "threshold": threshold,
                    "radius": radius,
                }
                if "language" in self.segmenter_input_names:
                    feed["language"] = np.full((x_seg.shape[0],), language_id, dtype=np.int64)
                boundaries = self.sessions["segmenter"].run(list(self.segmenter_output_names), feed)[0].astype(np.bool_)
            return boundaries

        feed = {
            "x_seg": np.ascontiguousarray(x_seg.astype(np.float32, copy=False)),
            "known_boundaries": known_boundaries.astype(np.bool_),
            "maskT": mask_t.astype(np.bool_),
            "threshold": threshold,
            "radius": radius,
        }
        if "language" in self.segmenter_input_names:
            feed["language"] = np.full((x_seg.shape[0],), language_id, dtype=np.int64)
        return self.sessions["segmenter"].run(list(self.segmenter_output_names), feed)[0].astype(np.bool_)

    def _run_estimator(
        self,
        x_est: np.ndarray,
        boundaries: np.ndarray,
        mask_t: np.ndarray,
        mask_n: np.ndarray,
        *,
        est_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        outputs = self.sessions["estimator"].run(
            list(self.estimator_output_names),
            {
                "x_est": np.ascontiguousarray(x_est.astype(np.float32, copy=False)),
                "boundaries": boundaries.astype(np.bool_),
                "maskT": mask_t.astype(np.bool_),
                "maskN": mask_n.astype(np.bool_),
                "threshold": _scalar_float32(est_threshold),
            },
        )
        mapped = dict(zip(self.estimator_output_names, outputs))
        return mapped["presence"].astype(np.bool_), mapped["scores"].astype(np.float32)

    def _predict_chunk_notes(
        self,
        chunks: list[dict[str, Any]],
        *,
        language: str | None,
        seg_threshold: float,
        seg_radius: float,
        t0: float,
        nsteps: int,
        est_threshold: float,
    ) -> list[NoteInfo]:
        language_id = _resolve_language_id(language, self.config.languages, self.repo_id)
        durations_sec = np.array([chunk["waveform"].shape[0] / self.config.samplerate for chunk in chunks], dtype=np.float32)
        max_len = max(chunk["waveform"].shape[0] for chunk in chunks)
        batch_waveform = np.zeros((len(chunks), max_len), dtype=np.float32)
        for index, chunk in enumerate(chunks):
            waveform = np.asarray(chunk["waveform"], dtype=np.float32)
            batch_waveform[index, : waveform.shape[0]] = waveform
        known_durations = durations_sec.reshape(len(chunks), 1)

        x_seg, x_est, mask_t = self._run_encoder(batch_waveform, durations_sec)
        if known_durations.shape[1] <= 1:
            known_boundaries = np.zeros_like(mask_t, dtype=np.bool_)
        else:
            known_boundaries = self.sessions["dur2bd"].run(
                None,
                {
                    "durations": known_durations,
                    "maskT": mask_t.astype(np.bool_),
                },
            )[0].astype(np.bool_)
        boundaries = self._run_segmenter(
            x_seg,
            known_boundaries,
            mask_t,
            language_id=language_id,
            seg_threshold=seg_threshold,
            seg_radius=seg_radius,
            t0=t0,
            nsteps=nsteps,
        )
        durations, mask_n = self.sessions["bd2dur"].run(
            list(self.bd2dur_output_names),
            {
                "boundaries": boundaries.astype(np.bool_),
                "maskT": mask_t.astype(np.bool_),
            },
        )
        presence, scores = self._run_estimator(
            x_est,
            boundaries,
            mask_t,
            mask_n,
            est_threshold=est_threshold,
        )

        notes: list[NoteInfo] = []
        for index, chunk in enumerate(chunks):
            durations_1d = np.asarray(durations[index], dtype=np.float32)
            presence_1d = np.asarray(presence[index], dtype=np.bool_)
            scores_1d = np.asarray(scores[index], dtype=np.float32)
            chunk_length = float(durations_sec[index])
            chunk_offset = float(chunk["offset"])
            onset = np.pad(durations_1d[:-1], (1, 0), mode="constant", constant_values=0.0).cumsum()
            offset_values = durations_1d.cumsum()
            onset = np.clip(onset, 0.0, chunk_length) + chunk_offset
            offset_values = np.clip(offset_values, 0.0, chunk_length) + chunk_offset
            for note_onset, note_offset, score, valid in zip(onset.tolist(), offset_values.tolist(), scores_1d.tolist(), presence_1d.tolist()):
                if note_offset - note_onset <= 0 or not valid:
                    continue
                notes.append(NoteInfo(onset=note_onset, offset=note_offset, pitch=float(score)))
        return notes

    def transcribe_waveform(
        self,
        waveform: np.ndarray,
        *,
        language: str | None = None,
        batch_size: int = DEFAULT_VOCAL_MIDI_BATCH_SIZE,
        seg_threshold: float = DEFAULT_VOCAL_MIDI_SEG_THRESHOLD,
        seg_radius: float = DEFAULT_VOCAL_MIDI_SEG_RADIUS,
        t0: float = DEFAULT_VOCAL_MIDI_T0,
        nsteps: int = DEFAULT_VOCAL_MIDI_NSTEPS,
        est_threshold: float = DEFAULT_VOCAL_MIDI_EST_THRESHOLD,
    ) -> list[NoteInfo]:
        notes: list[NoteInfo] = []
        pending: list[dict[str, Any]] = []
        for chunk in self.slicer.slice(waveform):
            chunk_waveform = np.asarray(chunk["waveform"], dtype=np.float32)
            if chunk_waveform.size == 0:
                continue
            pending.append({"offset": float(chunk["offset"]), "waveform": chunk_waveform})
            if len(pending) >= int(batch_size):
                notes.extend(
                    self._predict_chunk_notes(
                        pending,
                        language=language,
                        seg_threshold=seg_threshold,
                        seg_radius=seg_radius,
                        t0=t0,
                        nsteps=nsteps,
                        est_threshold=est_threshold,
                    )
                )
                pending = []
        if pending:
            notes.extend(
                self._predict_chunk_notes(
                    pending,
                    language=language,
                    seg_threshold=seg_threshold,
                    seg_radius=seg_radius,
                    t0=t0,
                    nsteps=nsteps,
                    est_threshold=est_threshold,
                )
            )
        return _sanitize_notes(notes)

    def _save_midi_file(self, output_path: Path, notes: Sequence[NoteInfo]) -> Path:
        import mido

        track = mido.MidiTrack()
        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
        last_time = 0
        for note in notes:
            onset_ticks = round(note.onset * 120 * 8)
            offset_ticks = round(note.offset * 120 * 8)
            midi_pitch = round(note.pitch)
            if offset_ticks <= onset_ticks:
                continue
            track.append(mido.Message("note_on", note=midi_pitch, time=onset_ticks - last_time))
            track.append(mido.Message("note_off", note=midi_pitch, time=offset_ticks - onset_ticks))
            last_time = offset_ticks

        midi_file = mido.MidiFile(charset="utf8")
        midi_file.tracks.append(track)
        midi_file.save(output_path)
        return output_path

    def _save_text_file(self, output_path: Path, notes: Sequence[NoteInfo], *, as_csv: bool) -> Path:
        rows = [
            {
                "onset": f"{note.onset:.3f}",
                "offset": f"{note.offset:.3f}",
                "pitch": _midi_to_note_name(note.pitch),
            }
            for note in notes
        ]
        if as_csv:
            with output_path.open(encoding="utf-8", mode="w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["onset", "offset", "pitch"])
                writer.writeheader()
                writer.writerows(rows)
            return output_path

        with output_path.open(encoding="utf-8", mode="w") as handle:
            for row in rows:
                handle.write(f"{row['onset']}\t{row['offset']}\t{row['pitch']}\n")
        return output_path

    def transcribe_file(
        self,
        input_path: str | Path,
        *,
        output_dir: str | Path | None = None,
        output_stem: str | None = None,
        output_formats: str | Sequence[str] = DEFAULT_VOCAL_MIDI_OUTPUT_FORMATS,
        language: str | None = None,
        batch_size: int = DEFAULT_VOCAL_MIDI_BATCH_SIZE,
        seg_threshold: float = DEFAULT_VOCAL_MIDI_SEG_THRESHOLD,
        seg_radius: float = DEFAULT_VOCAL_MIDI_SEG_RADIUS,
        t0: float = DEFAULT_VOCAL_MIDI_T0,
        nsteps: int = DEFAULT_VOCAL_MIDI_NSTEPS,
        est_threshold: float = DEFAULT_VOCAL_MIDI_EST_THRESHOLD,
    ) -> dict[str, Path]:
        input_path = Path(input_path).expanduser()
        save_dir = Path(output_dir).expanduser() if output_dir is not None else input_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = safe_leaf_name(output_stem or input_path.stem, default_name="song")
        formats = parse_output_formats(output_formats)
        waveform = self.read_audio(input_path)
        notes = self.transcribe_waveform(
            waveform,
            language=language,
            batch_size=batch_size,
            seg_threshold=seg_threshold,
            seg_radius=seg_radius,
            t0=t0,
            nsteps=nsteps,
            est_threshold=est_threshold,
        )

        saved: dict[str, Path] = {}
        for file_format in formats:
            output_path = safe_child_path(save_dir, f"{stem}.{file_format}", default_name=f"song.{file_format}")
            if file_format == "mid":
                saved[file_format] = self._save_midi_file(output_path, notes)
            elif file_format == "txt":
                saved[file_format] = self._save_text_file(output_path, notes, as_csv=False)
            else:
                saved[file_format] = self._save_text_file(output_path, notes, as_csv=True)
            if self.logger is not None:
                self.logger(f"[green]Saved vocal MIDI output[/green] {saved[file_format]}")
        return saved


def collect_audio_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio file: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract singing voice melody to MIDI with GAME ONNX.")
    parser.add_argument("input_path", help="Input audio file or directory")
    parser.add_argument("--repo_id", default=DEFAULT_GAME_MODEL_REPO_ID, choices=tuple(GAME_ONNX_MODEL_LABELS.keys()))
    parser.add_argument("--model_dir", default=DEFAULT_VOCAL_MIDI_MODEL_DIR)
    parser.add_argument("--language", default=None)
    parser.add_argument("--output_formats", default=DEFAULT_VOCAL_MIDI_OUTPUT_FORMATS)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_VOCAL_MIDI_BATCH_SIZE)
    parser.add_argument("--seg_threshold", type=float, default=DEFAULT_VOCAL_MIDI_SEG_THRESHOLD)
    parser.add_argument("--seg_radius", type=float, default=DEFAULT_VOCAL_MIDI_SEG_RADIUS)
    parser.add_argument("--t0", type=float, default=DEFAULT_VOCAL_MIDI_T0)
    parser.add_argument("--nsteps", type=int, default=DEFAULT_VOCAL_MIDI_NSTEPS)
    parser.add_argument("--est_threshold", type=float, default=DEFAULT_VOCAL_MIDI_EST_THRESHOLD)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force_download", action="store_true")
    return parser


def run_vocal_midi(args: argparse.Namespace) -> int:
    input_path = Path(args.input_path).expanduser()
    if not input_path.exists():
        console.print(f"[red]Input path does not exist:[/red] {input_path}")
        return 1

    audio_files = collect_audio_inputs(input_path)
    output_formats = parse_output_formats(args.output_formats)
    if not audio_files:
        console.print(f"[yellow]No supported audio files found under:[/yellow] {input_path}")
        return 1

    transcriber = GameOnnxTranscriber(
        repo_id=args.repo_id,
        model_dir=args.model_dir,
        force_download=bool(args.force_download),
        logger=console.print,
    )
    console.print("[cyan]Providers:[/cyan]")
    console.print(Pretty(transcriber.providers, indent_guides=True, expand_all=True))

    failures = 0
    skipped = 0
    processed = 0
    with Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(spinner_name="dots"),
        MofNCompleteColumn(separator="/"),
        BarColumn(bar_width=40, complete_style="green", finished_style="bold green"),
        TaskProgressColumn(),
        TextColumn("|"),
        TimeElapsedColumn(),
        TextColumn("|"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("[bold cyan]Extracting vocal MIDI...", total=len(audio_files))
        for source_path in audio_files:
            save_dir = Path(args.output_dir).expanduser() if args.output_dir else source_path.parent
            stem = safe_leaf_name(source_path.stem, default_name="song")
            if not args.overwrite and all((save_dir / f"{stem}.{fmt}").exists() for fmt in output_formats):
                skipped += 1
                progress.console.print(f"[yellow]Skipping existing outputs:[/yellow] {source_path}")
                progress.update(task, advance=1)
                continue
            try:
                transcriber.transcribe_file(
                    source_path,
                    output_dir=save_dir,
                    output_stem=stem,
                    output_formats=output_formats,
                    language=args.language,
                    batch_size=args.batch_size,
                    seg_threshold=args.seg_threshold,
                    seg_radius=args.seg_radius,
                    t0=args.t0,
                    nsteps=args.nsteps,
                    est_threshold=args.est_threshold,
                )
                processed += 1
            except Exception as exc:  # pragma: no cover
                failures += 1
                print_exception(progress.console, exc, prefix=f"Failed vocal MIDI extraction for {source_path}")
            finally:
                progress.update(task, advance=1)

    console.print(f"[bold]Finished.[/bold] processed={processed} skipped={skipped} failed={failures}")
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_vocal_midi(args)


if __name__ == "__main__":
    sys.exit(main())
