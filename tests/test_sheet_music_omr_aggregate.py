from __future__ import annotations

import warnings
from fractions import Fraction
from pathlib import Path

import pytest


def _page_score(
    pitches_by_part: dict[str, tuple[tuple[str, float], ...]],
):
    from music21 import note, stream

    score = stream.Score()
    for part_id, measures in pitches_by_part.items():
        part = stream.Part(id=part_id)
        for number, (pitch, duration) in enumerate(measures, start=1):
            measure = stream.Measure(number=number)
            measure.append(note.Note(pitch, quarterLength=duration))
            part.append(measure)
        score.insert(0, part)
    return score


def _kern_page(score, *, measure_count: int | None = None):
    from music21 import stream

    from module.sheet_music_omr.aggregate import KernPageScore

    if measure_count is None:
        measure_count = max(len(tuple(part.getElementsByClass(stream.Measure))) for part in score.parts)
    rows = ["**kern\t**kern", "=\t="]
    for _ in range(measure_count):
        rows.extend(("4C\t4c", "=\t="))
    rows.append("*-\t*-")
    return KernPageScore(
        score=score,
        kern_text="\n".join(rows) + "\n",
    )


def _notation_kern() -> str:
    return (
        "**kern\t**kern\n"
        "*clefF4\t*clefG2\n"
        "*k[b-]\t*k[b-]\n"
        "*M4/4\t*M4/4\n"
        "=1\t=1\n"
        "4B-\t16ccL\n"
        ".\t16dd\n"
        ".\t16ee\n"
        ".\t16ffJ\n"
        "4B-\t4gg\n"
        "4Bn\t4aa\n"
        "4B-\t4bb\n"
        "=2\t=2\n"
        "*-\t*-\n"
    )


def _score_note_signature(score) -> tuple:
    signature = []
    for part in score.parts:
        events = []
        for event in part.recurse().notes:
            pitches = tuple(pitch.midi for pitch in (event.pitches if hasattr(event, "pitches") else (event.pitch,)))
            events.append(
                (
                    event.getOffsetInHierarchy(part),
                    event.quarterLength,
                    pitches,
                )
            )
        signature.append((str(part.id), tuple(events)))
    return tuple(signature)


def test_omr_normalization_reconstructs_one_piano_grand_staff(
    tmp_path: Path,
):
    from music21 import layout, stream

    from module.music_export.music21_writers import write_midi_score
    from module.sheet_music_omr.aggregate import (
        KernPageScore,
        normalize_page_score,
    )
    from module.sheet_music_omr.decode import parse_kern_score

    kern_text = _notation_kern()
    parsed = parse_kern_score(kern_text)

    normalized = normalize_page_score(KernPageScore(score=parsed, kern_text=kern_text)).score

    assert all(isinstance(part, stream.PartStaff) for part in normalized.parts)
    groups = tuple(normalized.recurse().getElementsByClass(layout.StaffGroup))
    assert len(groups) == 1
    assert groups[0].symbol == "brace"
    assert groups[0].barTogether is True
    assert [str(part.id) for part in groups[0].getSpannedElements()] == ["spine_1", "spine_0"]
    assert _score_note_signature(normalized) == _score_note_signature(parsed)
    parsed_midi = tmp_path / "parsed.mid"
    normalized_midi = tmp_path / "normalized.mid"
    write_midi_score(parsed, parsed_midi)
    write_midi_score(normalized, normalized_midi)
    assert normalized_midi.read_bytes() == parsed_midi.read_bytes()


def test_omr_normalization_recalculates_display_accidentals():
    from module.sheet_music_omr.aggregate import (
        KernPageScore,
        normalize_page_score,
    )
    from module.sheet_music_omr.decode import parse_kern_score

    kern_text = _notation_kern()
    normalized = normalize_page_score(
        KernPageScore(
            score=parse_kern_score(kern_text),
            kern_text=kern_text,
        )
    ).score
    lower_staff = next(part for part in normalized.parts if str(part.id) == "spine_0")

    assert [(event.pitch.accidental.name, event.pitch.accidental.displayStatus) for event in lower_staff.recurse().notes] == [
        ("flat", False),
        ("flat", False),
        ("natural", True),
        ("flat", True),
    ]


def test_omr_normalization_completes_lazy_secondary_beams():
    from module.sheet_music_omr.aggregate import (
        KernPageScore,
        normalize_page_score,
    )
    from module.sheet_music_omr.decode import parse_kern_score

    kern_text = _notation_kern()
    normalized = normalize_page_score(
        KernPageScore(
            score=parse_kern_score(kern_text),
            kern_text=kern_text,
        )
    ).score
    upper_staff = next(part for part in normalized.parts if str(part.id) == "spine_1")
    sixteenths = tuple(upper_staff.recurse().notes)[:4]

    assert [tuple((beam.type, beam.direction) for beam in event.beams if beam.number == 2) for event in sixteenths] == [
        (("start", None),),
        (("continue", None),),
        (("continue", None),),
        (("stop", None),),
    ]


def test_omr_normalization_corrects_left_partial_beam_direction():
    from module.sheet_music_omr.aggregate import (
        KernPageScore,
        normalize_page_score,
    )
    from module.sheet_music_omr.decode import parse_kern_score

    kern_text = (
        "**kern\t**kern\n*clefF4\t*clefG2\n*M4/4\t*M4/4\n=1\t=1\n4C\t8.ccL\n.\t16ddJk\n4D\t4ee\n4E\t4ff\n4F\t4gg\n=2\t=2\n*-\t*-\n"
    )
    normalized = normalize_page_score(
        KernPageScore(
            score=parse_kern_score(kern_text),
            kern_text=kern_text,
        )
    ).score
    upper_staff = next(part for part in normalized.parts if str(part.id) == "spine_1")
    terminal = tuple(upper_staff.recurse().notes)[1]
    partial = terminal.beams.getByNumber(2)

    assert partial.type == "partial"
    assert partial.direction == "left"


def test_pdf_aggregate_appends_pages_and_renumbers_measures():
    from module.sheet_music_omr.aggregate import combine_page_scores

    first = _page_score(
        {
            "spine_1": (("C4", 1),),
            "spine_0": (("C3", 1),),
        }
    )
    second = _page_score(
        {
            "spine_1": (("D4", 1),),
            "spine_0": (("D3", 1),),
        }
    )

    aggregate = combine_page_scores((_kern_page(first), _kern_page(second)))

    assert [part.id for part in aggregate.score.parts] == [
        "spine_1",
        "spine_0",
    ]
    assert [measure.number for measure in aggregate.score.parts[0].getElementsByClass("Measure")] == [1, 2]
    assert [item.pitch.nameWithOctave for item in aggregate.score.parts[0].recurse().notes] == ["C4", "D4"]
    assert aggregate.padding_count == 0
    assert aggregate.padding_quarter_length == 0


def test_pdf_aggregate_uses_hidden_silent_padding_to_align_parts():
    from module.sheet_music_omr.aggregate import combine_page_scores

    page = _page_score(
        {
            "spine_1": (("C4", 1),),
            "spine_0": (("C3", 2),),
        }
    )

    aggregate = combine_page_scores((_kern_page(page),))

    assert [part.highestTime for part in aggregate.score.parts] == [2.0, 2.0]
    assert aggregate.padding_count == 1
    assert aggregate.padding_quarter_length == Fraction(1, 1)
    hidden_rests = [item for item in aggregate.score.parts[0].recurse().getElementsByClass("Rest") if item.style.hideObjectOnPrint]
    assert len(hidden_rests) == 1
    assert hidden_rests[0].quarterLength == 1


def test_pdf_aggregate_splits_complex_padding_for_musicxml(
    tmp_path: Path,
):
    import xml.etree.ElementTree as ET

    from music21 import note

    from module.music_export.music21_writers import write_musicxml_score
    from module.music_export.validation import validate_musicxml_file
    from module.sheet_music_omr.aggregate import combine_page_scores

    page = _page_score(
        {
            "spine_1": (("C4", 1),),
            "spine_0": (("C3", 1.5),),
        }
    )
    left_measure = page.parts[1].getElementsByClass("Measure").first()
    left_measure.append(note.Note("E3", quarterLength=0.125))

    aggregate = combine_page_scores((_kern_page(page),))
    hidden_rests = [item for item in aggregate.score.parts[0].recurse().getElementsByClass("Rest") if item.style.hideObjectOnPrint]

    assert aggregate.padding_count == 1
    assert aggregate.padding_quarter_length == Fraction(5, 8)
    assert [rest.quarterLength for rest in hidden_rests] == [0.5, 0.125]
    assert all(not rest.duration.isComplex for rest in hidden_rests)
    output = tmp_path / "aggregate.musicxml"
    write_musicxml_score(
        aggregate.score,
        output,
        make_notation=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_musicxml_file(output)
    root = ET.parse(output).getroot()
    assert len(root.findall("./part")) == 1
    assert [element.text for element in root.findall(".//staves")] == ["2"]


def test_pdf_aggregate_rejects_missing_part_measure_instead_of_synthesizing_it():
    from module.sheet_music_omr.aggregate import (
        PdfAggregationError,
        combine_page_scores,
    )

    page = _page_score(
        {
            "spine_1": (("C4", 1), ("D4", 1)),
            "spine_0": (("C3", 1),),
        }
    )

    with pytest.raises(PdfAggregationError, match="measure count"):
        combine_page_scores((_kern_page(page, measure_count=2),))


def test_pdf_aggregate_rejects_part_identity_drift_between_pages():
    from module.sheet_music_omr.aggregate import (
        PdfAggregationError,
        combine_page_scores,
    )

    first = _page_score(
        {
            "spine_1": (("C4", 1),),
            "spine_0": (("C3", 1),),
        }
    )
    second = _page_score(
        {
            "spine_1": (("D4", 1),),
            "other": (("D3", 1),),
        }
    )

    with pytest.raises(PdfAggregationError, match="part ids"):
        combine_page_scores((_kern_page(first), _kern_page(second)))


def test_pdf_aggregate_aligns_music21_omitted_null_only_measure_slots():
    from music21 import stream

    from module.sheet_music_omr.aggregate import (
        KernPageScore,
        combine_page_scores,
    )
    from module.sheet_music_omr.decode import parse_kern_score

    kern_text = (
        "**kern\t**kern\n"
        "*clefF4\t*clefG2\n"
        "*M4/4\t*M4/4\n"
        ".\t8cc\n"
        "=\t=\n"
        "1C\t1ee\n"
        "=\t=\n"
        "*^\t*\n"
        "1D\t1F\t.\n"
        "*v\t*v\t*\n"
        "=\t=\n"
        "1E\t1gg\n"
        "=\t=\n"
        "*-\t*-\n"
    )
    page_score = parse_kern_score(kern_text)
    parsed_counts = {str(part.id): len(tuple(part.getElementsByClass(stream.Measure))) for part in page_score.parts}
    assert parsed_counts == {"spine_1": 4, "spine_0": 3}

    aggregate = combine_page_scores((KernPageScore(score=page_score, kern_text=kern_text),))

    aggregate_counts = {str(part.id): len(tuple(part.getElementsByClass(stream.Measure))) for part in aggregate.score.parts}
    assert aggregate_counts == {"spine_1": 4, "spine_0": 4}
    assert [part.highestTime for part in aggregate.score.parts] == [12.5, 12.5]
    assert aggregate.padding_count == 2
    assert aggregate.padding_quarter_length == Fraction(9, 2)


def test_pdf_aggregate_reanchors_context_and_deduplicates_page_headers():
    from music21 import clef, key, meter, stream

    from module.sheet_music_omr.aggregate import (
        KernPageScore,
        combine_page_scores,
    )
    from module.sheet_music_omr.decode import parse_kern_score

    first_kern = (
        "**kern\t**kern\n"
        "*clefG2\t*clefG2\n"
        "*k[b-]\t*k[b-]\n"
        "*M4/4\t*M4/4\n"
        "*met(c)\t*met(c)\n"
        "=\t=\n"
        "1C\t1e\n"
        "*clefF4\t*\n"
        "=\t=\n"
        "1D\t1f\n"
        "=\t=\n"
        "*-\t*-\n"
    )
    second_kern = "**kern\t**kern\n*clefF4\t*clefG2\n*k[b-]\t*k[b-]\n*M4/4\t*M4/4\n*met(c)\t*met(c)\n=\t=\n1E\t1g\n=\t=\n*-\t*-\n"
    pages = tuple(
        KernPageScore(
            score=parse_kern_score(kern_text),
            kern_text=kern_text,
        )
        for kern_text in (first_kern, second_kern)
    )

    aggregate = combine_page_scores(pages)

    upper, lower = aggregate.score.parts
    upper_measures = tuple(upper.getElementsByClass(stream.Measure))
    lower_measures = tuple(lower.getElementsByClass(stream.Measure))
    assert [
        (index, type(item).__name__)
        for index, measure in enumerate(upper_measures, start=1)
        for item in measure.getElementsByClass(clef.Clef)
    ] == [(1, "TrebleClef")]
    assert [
        (index, type(item).__name__)
        for index, measure in enumerate(lower_measures, start=1)
        for item in measure.getElementsByClass(clef.Clef)
    ] == [(1, "TrebleClef"), (2, "BassClef")]
    for part in (upper, lower):
        measures = tuple(part.getElementsByClass(stream.Measure))
        assert [
            index for index, measure in enumerate(measures, start=1) if tuple(measure.getElementsByClass(key.KeySignature))
        ] == [1]
        assert [
            index for index, measure in enumerate(measures, start=1) if tuple(measure.getElementsByClass(meter.TimeSignature))
        ] == [1]
        first_meter = measures[0].getElementsByClass(meter.TimeSignature).first()
        assert first_meter.symbol == "common"


def test_pdf_aggregate_places_mid_measure_and_boundary_clef_changes():
    from music21 import clef, stream

    from module.sheet_music_omr.aggregate import (
        KernPageScore,
        combine_page_scores,
    )
    from module.sheet_music_omr.decode import parse_kern_score

    kern_text = (
        "**kern\t**kern\n"
        "*clefG2\t*clefG2\n"
        "=1\t=1\n"
        "8C\t8c\n"
        "*clefF4\t*\n"
        "8D\t8d\n"
        "*clefG2\t*\n"
        "=2\t=2\n"
        "4E\t4e\n"
        "=3\t=3\n"
        "*-\t*-\n"
    )
    parsed = parse_kern_score(kern_text)

    aggregate = combine_page_scores(
        (KernPageScore(score=parsed, kern_text=kern_text),)
    )

    lower = next(
        part for part in aggregate.score.parts if str(part.id) == "spine_0"
    )
    measures = tuple(lower.getElementsByClass(stream.Measure))
    clef_positions = [
        (
            measure.number,
            type(item).__name__,
            item.getOffsetInHierarchy(measure),
        )
        for measure in measures
        for item in measure.getElementsByClass(clef.Clef)
    ]
    assert clef_positions == [
        (1, "TrebleClef", 0.0),
        (1, "BassClef", 0.5),
        (2, "TrebleClef", 0.0),
    ]


def test_pdf_aggregate_restores_grace_notes_left_at_part_level_by_music21():
    from music21 import stream

    from module.sheet_music_omr.aggregate import (
        KernPageScore,
        combine_page_scores,
    )
    from module.sheet_music_omr.decode import parse_kern_score

    kern_text = (
        "**kern\t**kern\n"
        "*clefG2\t*clefG2\n"
        "*M3/4\t*M3/4\n"
        "=\t=\n"
        "4C\t4c\n"
        "2C\t2c\n"
        "*\t*clefF4\n"
        "=\t=\n"
        ".\t32qqG\n"
        ".\t32qqA\n"
        "4D\t4B\n"
        "2D\t2B\n"
        "*\t*clefG2\n"
        "=\t=\n"
        "4E\t4e\n"
        "2E\t2e\n"
        "=\t=\n"
        "*-\t*-\n"
    )
    parsed = parse_kern_score(kern_text)
    upper = next(part for part in parsed.parts if str(part.id) == "spine_1")
    assert [event.priority for event in upper.notesAndRests] == [9, 10]

    aggregate = combine_page_scores(
        (KernPageScore(score=parsed, kern_text=kern_text),)
    )

    upper = next(
        part for part in aggregate.score.parts if str(part.id) == "spine_1"
    )
    assert tuple(upper.notesAndRests) == ()
    second_measure = tuple(upper.getElementsByClass(stream.Measure))[1]
    grace_notes = [
        event
        for event in second_measure.recurse().notes
        if event.duration.isGrace
    ]
    assert [event.priority for event in grace_notes] == [9, 10]
    assert [
        event.getOffsetInHierarchy(second_measure) for event in grace_notes
    ] == [0.0, 0.0]
