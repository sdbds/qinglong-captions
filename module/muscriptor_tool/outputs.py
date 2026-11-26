from __future__ import annotations

import json
import sys
import warnings as warnings_module
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Iterator, TextIO

from module.music_export import atomic_output_path

from .events import EventStats, event_to_dict, is_progress_event
from .options import OutputFormat, TempoDetection, TranscriptionOptions
from .runtime import LoadedModel


@contextmanager
def _optional_jsonl_writer(
    target: Path | None,
    direct_stream: TextIO | None = None,
) -> Iterator[TextIO | None]:
    if direct_stream is not None:
        yield direct_stream
        return
    if target is None:
        yield None
        return
    with atomic_output_path(target) as temporary:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            yield stream


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    with atomic_output_path(target) as temporary:
        temporary.write_bytes(payload)


def _atomic_write_text(target: Path, payload: str) -> None:
    with atomic_output_path(target) as temporary:
        temporary.write_text(payload, encoding="utf-8", newline="\n")


def preflight_sheets() -> None:
    from muscriptor.utils.sheets import find_musescore

    find_musescore()


def engrave_sheets(midi_bytes: bytes, *, quantized: bool) -> bytes:
    from muscriptor.server import engrave_to_zip

    return engrave_to_zip(midi_bytes, quantized=quantized)


@dataclass(frozen=True)
class OutputTargets:
    midi: Path | None = None
    json: Path | None = None
    jsonl: Path | None = None
    midi_stream: BinaryIO | None = None
    json_stream: TextIO | None = None
    jsonl_stream: TextIO | None = None
    quantized_midi: Path | None = None
    sheets: Path | None = None
    quantized_midi_stream: BinaryIO | None = None
    sheets_stream: BinaryIO | None = None

    @classmethod
    def for_directory(
        cls,
        output_dir: Path,
        formats: Iterable[OutputFormat],
        *,
        output_stem: str = "transcription",
    ) -> "OutputTargets":
        output_dir = Path(output_dir)
        selected = {OutputFormat(item) for item in formats}
        return cls(
            midi=output_dir / f"{output_stem}.mid" if OutputFormat.MIDI in selected else None,
            quantized_midi=output_dir / f"{output_stem}_quantized.mid" if OutputFormat.QUANTIZED_MIDI in selected else None,
            sheets=output_dir / f"{output_stem}_sheets.zip" if OutputFormat.SHEETS in selected else None,
            json=output_dir / "events.json" if OutputFormat.JSON in selected else None,
            jsonl=output_dir / "events.jsonl" if OutputFormat.JSONL in selected else None,
        )

    def requested_paths(self) -> dict[str, Path]:
        return {
            key: value
            for key, value in (
                ("midi", self.midi), ("quantized-midi", self.quantized_midi), ("sheets", self.sheets),
                ("json", self.json), ("jsonl", self.jsonl),
            )
            if value is not None
        }

    def stream_outputs(self) -> dict[str, str]:
        return {
            key: "-"
            for key, value in (
                ("midi", self.midi_stream),
                ("quantized-midi", self.quantized_midi_stream),
                ("sheets", self.sheets_stream),
                ("json", self.json_stream),
                ("jsonl", self.jsonl_stream),
            )
            if value is not None
        }


@dataclass(frozen=True)
class TranscriptionResult:
    note_count: int
    event_count: int
    chunk_count: int
    completed_chunks: int
    outputs: dict[str, str]
    warnings: tuple[str, ...]
    midi_bytes: bytes | None = None
    detected_instruments: tuple[str, ...] = ()


class RequestedOutputError(RuntimeError):
    """A requested output failed after transcription produced usable results."""

    def __init__(self, message: str, *, result: TranscriptionResult):
        super().__init__(message)
        self.result = result


def _warning_messages(captured: Iterable[warnings_module.WarningMessage]) -> list[str]:
    return [f"{item.category.__name__}: {item.message}" for item in captured]


def transcribe_once(
    loaded: LoadedModel | Any,
    source: Path,
    options: TranscriptionOptions,
    targets: OutputTargets,
    *,
    require_midi_bytes: bool = False,
    stderr: TextIO | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    preview_runtime: Any | None = None,
    preview_target: Path | None = None,
    preview_renderer: Callable[..., None] | None = None,
) -> TranscriptionResult:
    stderr = stderr or sys.stderr
    stats = EventStats()
    original_events: list[Any] = []
    json_events: list[dict[str, Any]] = []
    if preview_runtime is not None and preview_target is None:
        raise ValueError("preview_target is required when preview_runtime is enabled")
    need_quantized = any(
        target is not None
        for target in (targets.quantized_midi, targets.quantized_midi_stream, targets.sheets, targets.sheets_stream)
    )
    need_performance = (
        targets.midi is not None
        or targets.midi_stream is not None
        or require_midi_bytes
        or (preview_runtime is not None and not need_quantized)
    )
    collect_originals = need_performance or need_quantized
    collect_json = targets.json is not None or targets.json_stream is not None

    with warnings_module.catch_warnings(record=True) as captured_warnings:
        warnings_module.simplefilter("always")
        with _optional_jsonl_writer(targets.jsonl, targets.jsonl_stream) as jsonl_stream:
            for event in loaded.transcribe(Path(source), options):
                if is_progress_event(event, loaded.progress_event_type):
                    completed = int(event.completed)
                    total = int(event.total)
                    stats.observe_progress(completed=completed, total=total)
                    if progress_callback is not None:
                        progress_callback(completed, total)
                    continue

                payload = event_to_dict(event)
                stats.observe_event(payload)
                if jsonl_stream is not None:
                    jsonl_stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                    jsonl_stream.flush()
                if collect_originals:
                    original_events.append(event)
                if collect_json:
                    json_events.append(payload)
                if options.print_notes:
                    print(event, file=stderr)

    warning_messages = _warning_messages(captured_warnings)
    if stats.event_count == 0:
        warning_messages.append("EMPTY_TRANSCRIPTION")

    midi_payload: bytes | None = None

    if targets.json is not None or targets.json_stream is not None:
        json_payload = json.dumps(json_events, ensure_ascii=False, indent=2) + "\n"
        if targets.json is not None:
            _atomic_write_text(targets.json, json_payload)
        if targets.json_stream is not None:
            targets.json_stream.write(json_payload)
            targets.json_stream.flush()

    outputs = {
        **{key: str(path) for key, path in targets.requested_paths().items() if key in ("json", "jsonl")},
        **{key: "-" for key in targets.stream_outputs() if key in ("json", "jsonl")},
    }

    def write_binary(key: str, path: Path | None, stream: BinaryIO | None, data: bytes) -> None:
        if path is not None:
            _atomic_write_bytes(path, data)
            outputs[key] = str(path)
        if stream is not None:
            stream.write(data)
            stream.flush()
            outputs[key] = "-"

    def result() -> TranscriptionResult:
        return TranscriptionResult(
            note_count=stats.note_count,
            event_count=stats.event_count,
            chunk_count=stats.chunk_count,
            completed_chunks=stats.completed_chunks,
            outputs=dict(outputs),
            warnings=tuple(warning_messages),
            midi_bytes=midi_payload,
            detected_instruments=tuple(stats.detected_instruments),
        )

    if not collect_originals:
        return result()

    stage = "MIDI"
    try:
        timing = options.timing
        grid = None
        if timing.detect_tempo is not TempoDetection.OFF:
            onsets = [event.start_time for event in original_events if hasattr(event, "start_time")]
            with warnings_module.catch_warnings(record=True) as beat_warnings:
                warnings_module.simplefilter("always")
                grid = loaded.detect_beat_grid(Path(source), timing.detect_tempo.upstream_value, onsets)
            warning_messages.extend(_warning_messages(beat_warnings))
            if grid is None and timing.bpm is None:
                warning_messages.append("No beat grid detected; using 120 BPM without a time signature.")
        if need_performance:
            if timing.detect_tempo is TempoDetection.OFF and timing.bpm is None:
                # The stem pipeline requests the original, audio-aligned MIDI contract.
                midi_payload = loaded.midi_bytes(original_events)
            else:
                midi_payload, _ = loaded.export_midi(original_events, grid, timing, quantize=False)
            write_binary("midi", targets.midi, targets.midi_stream, midi_payload)
        if need_quantized:
            stage = "quantized MIDI"
            snapped_midi, snapped = loaded.export_midi(original_events, grid, timing, quantize=True)
            if not snapped:
                warning_messages.append("No usable beat subdivision; notes were not quantized.")
            if not need_performance:
                midi_payload = snapped_midi
            write_binary("quantized-midi", targets.quantized_midi, targets.quantized_midi_stream, snapped_midi)
            if targets.sheets is not None or targets.sheets_stream is not None:
                stage = "sheets"
                archive = engrave_sheets(snapped_midi, quantized=snapped)
                write_binary("sheets", targets.sheets, targets.sheets_stream, archive)

        if preview_runtime is not None:
            stage = "preview"
            if preview_renderer is None:
                from .auralization import render_preview

                preview_renderer = render_preview
            preview_renderer(
                preview_runtime,
                midi_bytes=midi_payload,
                original_audio_path=Path(source),
                output_path=Path(preview_target),
            )
            outputs["preview"] = str(preview_target)
    except Exception as exc:
        raise RequestedOutputError(
            f"Requested {stage} output failed: {type(exc).__name__}: {exc}", result=result()
        ) from exc
    return result()
