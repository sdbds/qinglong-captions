from __future__ import annotations

import json
import os
import sys
import uuid
import warnings as warnings_module
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Iterator, TextIO

from .events import EventStats, event_to_dict, is_progress_event
from .options import OutputFormat, TranscriptionOptions
from .runtime import LoadedModel


@contextmanager
def atomic_output_path(target: Path) -> Iterator[Path]:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    temporary = target.with_name(f"{target.stem}.{os.getpid()}.{nonce}.part{target.suffix}")
    try:
        yield temporary
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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


@dataclass(frozen=True)
class OutputTargets:
    midi: Path | None = None
    json: Path | None = None
    jsonl: Path | None = None
    midi_stream: BinaryIO | None = None
    json_stream: TextIO | None = None
    jsonl_stream: TextIO | None = None

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
            json=output_dir / "events.json" if OutputFormat.JSON in selected else None,
            jsonl=output_dir / "events.jsonl" if OutputFormat.JSONL in selected else None,
        )

    @property
    def needs_event_collection(self) -> bool:
        return any((self.midi is not None, self.json is not None, self.midi_stream, self.json_stream))

    def requested_paths(self) -> dict[str, Path]:
        return {
            key: value
            for key, value in (("midi", self.midi), ("json", self.json), ("jsonl", self.jsonl))
            if value is not None
        }

    def stream_outputs(self) -> dict[str, str]:
        return {
            key: "-"
            for key, value in (
                ("midi", self.midi_stream),
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
    collect_originals = (
        targets.midi is not None
        or targets.midi_stream is not None
        or require_midi_bytes
        or preview_runtime is not None
    )
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
    if collect_originals:
        midi_payload = loaded.midi_bytes(original_events)
        if targets.midi is not None:
            _atomic_write_bytes(targets.midi, midi_payload)
        if targets.midi_stream is not None:
            targets.midi_stream.write(midi_payload)
            targets.midi_stream.flush()

    if targets.json is not None or targets.json_stream is not None:
        json_payload = json.dumps(json_events, ensure_ascii=False, indent=2) + "\n"
        if targets.json is not None:
            _atomic_write_text(targets.json, json_payload)
        if targets.json_stream is not None:
            targets.json_stream.write(json_payload)
            targets.json_stream.flush()

    outputs = {
        **{key: str(path) for key, path in targets.requested_paths().items() if path.exists()},
        **targets.stream_outputs(),
    }
    if preview_runtime is not None:
        if preview_target is None:
            raise ValueError("preview_target is required when preview_runtime is enabled")
        if midi_payload is None:
            raise RuntimeError("Preview rendering requires MIDI bytes")
        if preview_renderer is None:
            from .auralization import render_preview

            preview_renderer = render_preview
        try:
            preview_renderer(
                preview_runtime,
                midi_bytes=midi_payload,
                original_audio_path=Path(source),
                output_path=Path(preview_target),
            )
        except Exception as exc:
            partial_result = TranscriptionResult(
                note_count=stats.note_count,
                event_count=stats.event_count,
                chunk_count=stats.chunk_count,
                completed_chunks=stats.completed_chunks,
                outputs=dict(outputs),
                warnings=tuple(warning_messages),
                midi_bytes=midi_payload,
                detected_instruments=tuple(stats.detected_instruments),
            )
            raise RequestedOutputError(
                f"Requested preview output failed: {type(exc).__name__}: {exc}",
                result=partial_result,
            ) from exc
        outputs["preview"] = str(preview_target)
    return TranscriptionResult(
        note_count=stats.note_count,
        event_count=stats.event_count,
        chunk_count=stats.chunk_count,
        completed_chunks=stats.completed_chunks,
        outputs=outputs,
        warnings=tuple(warning_messages),
        midi_bytes=midi_payload,
        detected_instruments=tuple(stats.detected_instruments),
    )
