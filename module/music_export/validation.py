from __future__ import annotations

from pathlib import Path
from typing import Any


class MusicExportValidationError(ValueError):
    """Raised when a generated symbolic music file cannot be read back."""


def validate_nonempty_file(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise MusicExportValidationError(f"Export is missing or empty: {path}")
    return path


def validate_midi_file(
    path: str | Path,
    *,
    require_note_events: bool = False,
) -> Any:
    path = validate_nonempty_file(path)
    import mido

    parsed = mido.MidiFile(path)
    if not parsed.tracks:
        raise MusicExportValidationError(f"MIDI contains no tracks: {path}")
    if require_note_events:
        has_note_on = any(
            message.type == "note_on" and getattr(message, "velocity", 0) > 0
            for track in parsed.tracks
            for message in track
        )
        if not has_note_on:
            raise MusicExportValidationError(f"MIDI contains no note events: {path}")
    return parsed


def validate_musicxml_file(path: str | Path) -> Any:
    path = validate_nonempty_file(path)
    from music21 import converter

    parsed = converter.parse(path)
    parts = tuple(getattr(parsed, "parts", ()))
    if not parts:
        raise MusicExportValidationError(f"MusicXML contains no parts: {path}")
    if not any(tuple(part.getElementsByClass("Measure")) for part in parts):
        raise MusicExportValidationError(f"MusicXML contains no measures: {path}")
    if not any(True for _ in parsed.recurse().notesAndRests):
        raise MusicExportValidationError(
            f"MusicXML contains no notes or rests: {path}"
        )
    return parsed
