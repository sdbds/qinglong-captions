from __future__ import annotations

from pathlib import Path

import pytest


def _minimal_score():
    from music21 import meter, note, stream

    score = stream.Score()
    part = stream.Part()
    measure = stream.Measure(number=1)
    measure.append(meter.TimeSignature("4/4"))
    measure.append(note.Note("C4", quarterLength=1))
    measure.append(note.Rest(quarterLength=3))
    part.append(measure)
    score.insert(0, part)
    return score


def test_common_music21_writers_round_trip_without_domain_models(tmp_path: Path):
    from module.music_export.music21_writers import write_midi_score, write_musicxml_score
    from module.music_export.validation import validate_midi_file, validate_musicxml_file

    score = _minimal_score()
    musicxml_path = tmp_path / "score.musicxml"
    midi_path = tmp_path / "score.mid"

    write_musicxml_score(score, musicxml_path)
    write_midi_score(score, midi_path)

    parsed = validate_musicxml_file(musicxml_path)
    validate_midi_file(midi_path)
    assert len(parsed.parts) == 1
    assert list(parsed.recurse().notes)


def test_musicxml_writer_can_preserve_an_already_measured_score(
    monkeypatch,
    tmp_path: Path,
):
    from module.music_export.music21_writers import write_musicxml_score

    score = _minimal_score()
    target = tmp_path / "score.musicxml"
    calls = []

    def fake_write(format_name, *, fp, **kwargs):
        calls.append((format_name, fp, kwargs))
        Path(fp).write_text("<score-partwise/>", encoding="utf-8")

    monkeypatch.setattr(score, "write", fake_write)

    write_musicxml_score(score, target, make_notation=False)

    assert calls == [
        (
            "musicxml",
            str(target),
            {"makeNotation": False},
        )
    ]


def test_midi_writer_falls_back_to_written_order_for_malformed_repeats(
    tmp_path: Path,
):
    from music21 import bar, note, stream

    from module.music_export.music21_writers import (
        MIDI_REPEAT_FALLBACK_WARNING,
        write_midi_score,
    )
    from module.music_export.validation import validate_midi_file

    score = stream.Score()
    part = stream.Part()
    first = stream.Measure(number=1)
    first.append(note.Note("C4"))
    first.rightBarline = bar.Repeat(direction="end")
    second = stream.Measure(number=2)
    second.leftBarline = bar.Repeat(direction="start")
    second.append(note.Note("D4"))
    part.append((first, second))
    score.append(part)
    target = tmp_path / "malformed-repeat.mid"

    warning = write_midi_score(score, target)

    validate_midi_file(target, require_note_events=True)
    assert warning == MIDI_REPEAT_FALLBACK_WARNING
    assert isinstance(first.rightBarline, bar.Repeat)
    assert isinstance(second.leftBarline, bar.Repeat)


@pytest.mark.parametrize(
    ("name", "payload", "validator_name"),
    (
        ("empty.musicxml", b"", "validate_musicxml_file"),
        ("broken.musicxml", b"not xml", "validate_musicxml_file"),
        ("empty.mid", b"", "validate_midi_file"),
        ("broken.mid", b"not midi", "validate_midi_file"),
    ),
)
def test_common_validators_reject_empty_or_malformed_files(
    tmp_path: Path,
    name: str,
    payload: bytes,
    validator_name: str,
):
    from module.music_export import validation

    path = tmp_path / name
    path.write_bytes(payload)

    with pytest.raises(Exception):
        getattr(validation, validator_name)(path)


def test_midi_validator_can_require_note_events(tmp_path: Path):
    import mido

    from module.music_export.validation import validate_midi_file

    midi = mido.MidiFile()
    midi.tracks.append(mido.MidiTrack())
    target = tmp_path / "rest-only.mid"
    midi.save(target)

    validate_midi_file(target, require_note_events=False)
    with pytest.raises(Exception, match="note events"):
        validate_midi_file(target, require_note_events=True)
