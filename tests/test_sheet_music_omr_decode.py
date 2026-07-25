import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def test_bekern_special_tokens_reconstruct_exact_text():
    from module.sheet_music_omr.decode import decode_bekern_tokens

    assert decode_bekern_tokens(("4", "c", "<s>", "4", "e", "<t>", "4", "g", "<b>", "*-", "<t>")) == "4c 4e\t4g\n*-\t"


def test_complete_pinned_polish_scores_fixture_reconstructs_exact_kern():
    from module.sheet_music_omr.decode import reconstruct_kern

    fixture = json.loads((FIXTURES / "musvit_polish_scores_val0.json").read_text(encoding="utf-8"))

    assert reconstruct_kern(fixture["tokens"]) == fixture["expected_kern"]


def test_headerless_body_receives_exact_two_spine_envelope():
    from module.sheet_music_omr.decode import reconstruct_kern_envelope

    body = "*clefG2\t*clefG2\n=M\t=M\n4c\t4e\n*-\t\n"

    assert reconstruct_kern_envelope(body) == ("**kern\t**kern\n*clefG2\t*clefG2\n=M\t=M\n4c\t4e\n*-\t*-\n")


def test_existing_exact_envelope_is_not_duplicated():
    from module.sheet_music_omr.decode import reconstruct_kern_envelope

    kern = "**kern\t**kern\n4c\t4e\n*-\t*-\n"

    assert reconstruct_kern_envelope(kern) == kern


def test_existing_three_spine_terminator_is_preserved_after_split():
    from module.sheet_music_omr.decode import reconstruct_kern

    tokens = (
        "*",
        "<t>",
        "*^",
        "<b>",
        "4",
        "c",
        "<t>",
        "4",
        "e",
        "<t>",
        "4",
        "g",
        "<b>",
        "*-",
        "<t>",
        "*-",
        "<t>",
        "*-",
        "<b>",
    )

    assert reconstruct_kern(tokens).endswith("*-\t*-\t*-\n")


def test_single_remaining_spine_can_terminate_after_other_spine_ends():
    from module.sheet_music_omr.decode import reconstruct_kern_envelope

    kern = "**kern\t**kern\n*-\t*\n4c\n*-\n"

    assert reconstruct_kern_envelope(kern) == kern


@pytest.mark.parametrize(
    "body",
    [
        "**ekern\t**ekern\n4c\t4e\n*-\t",
        "**kern\n4c\n*-",
        "**foo\t**foo\n4c\t4e\n*-\t",
        "4c\t4e\n**kern\t**kern\n*-\t",
    ],
)
def test_unsupported_exclusive_interpretations_fail(body: str):
    from module.sheet_music_omr.decode import KernStructureError, reconstruct_kern_envelope

    with pytest.raises(KernStructureError, match="exclusive interpretation"):
        reconstruct_kern_envelope(body)


@pytest.mark.parametrize(
    "terminator",
    ["*-", "*-\t*", "*\t*-", "4c\t4e", ""],
)
def test_only_known_stripped_terminal_envelope_is_repaired(terminator: str):
    from module.sheet_music_omr.decode import KernStructureError, reconstruct_kern_envelope

    body = f"4c\t4e\n{terminator}" if terminator else "4c\t4e"
    with pytest.raises(KernStructureError, match="terminator"):
        reconstruct_kern_envelope(body)


def test_spine_validator_rejects_malformed_inner_rows():
    from module.sheet_music_omr.decode import KernStructureError, validate_kern_spines

    kern = "**kern\t**kern\n4c\n*-\t*-\n"

    with pytest.raises(KernStructureError, match="spine count"):
        validate_kern_spines(kern)


def test_spine_validator_tracks_split_and_join_operations():
    from module.sheet_music_omr.decode import validate_kern_spines

    kern = "**kern\t**kern\n*^\t*\n4c\t4e\t4g\n*v\t*v\t*\n4c\t4g\n*-\t*-\n"

    validate_kern_spines(kern)


def test_spine_validator_accepts_complete_dynamic_three_spine_terminator():
    from module.sheet_music_omr.decode import validate_kern_spines

    kern = "**kern\t**kern\n*\t*^\n4c\t4e\t4g\n*-\t*-\t*-\n"

    validate_kern_spines(kern)


def test_spine_validator_rejects_terminator_width_that_misses_active_spine():
    from module.sheet_music_omr.decode import KernStructureError, validate_kern_spines

    kern = "**kern\t**kern\n*\t*^\n4c\t4e\t4g\n*-\t*-\n"

    with pytest.raises(KernStructureError, match="spine count"):
        validate_kern_spines(kern)


def test_reconstruction_error_retains_exact_diagnostic_candidate():
    from module.sheet_music_omr.decode import KernStructureError, reconstruct_kern

    tokens = ("4", "c", "<t>", "4", "e")

    with pytest.raises(KernStructureError) as captured:
        reconstruct_kern(tokens)

    assert captured.value.candidate == "4c\t4e\n"


def test_minimal_valid_kern_parses_as_music21_score():
    from module.sheet_music_omr.decode import parse_kern_score

    kern = "**kern\t**kern\n*clefG2\t*clefG2\n*M4/4\t*M4/4\n=1\t=1\n4c\t4e\n*-\t*-\n"

    score = parse_kern_score(kern)

    assert len(score.parts) == 2
    assert len(tuple(score.recurse().notes)) == 2


@pytest.mark.parametrize(
    ("marker", "expected_direction"),
    (("k", "left"), ("K", "right")),
)
def test_kern_parser_preserves_partial_beam_direction(
    marker: str,
    expected_direction: str,
):
    from module.sheet_music_omr.decode import parse_kern_score

    kern = f"**kern\t**kern\n*M4/4\t*M4/4\n=1\t=1\n4C\t8.ccL\n.\t16ddJ{marker}\n4D\t4ee\n4E\t4ff\n4F\t4gg\n=2\t=2\n*-\t*-\n"

    score = parse_kern_score(kern)
    upper_staff = next(part for part in score.parts if str(part.id) == "spine_1")
    terminal = tuple(upper_staff.recurse().notes)[1]

    assert terminal.beams.getByNumber(2).direction == expected_direction


def test_partial_beam_restore_supports_legacy_humdrum_positions():
    from types import SimpleNamespace

    from music21 import note, stream

    from module.sheet_music_omr.decode import (
        _restore_partial_beam_directions,
    )

    parsed_note = note.Note("C4", type="16th")
    parsed_note.beams.fill(2)
    parsed_note.beams.setByNumber(1, "stop")
    parsed_note.beams.setByNumber(2, "partial", "right")
    parsed_note.priority = 4
    parsed_stream = stream.Stream((parsed_note,))
    source_event = SimpleNamespace(contents="16cJk", position=4)
    spine = SimpleNamespace(
        stream=parsed_stream,
        eventList=(source_event,),
    )
    data_collection = SimpleNamespace(spineCollection=SimpleNamespace(spines=(spine,)))

    _restore_partial_beam_directions(data_collection)

    assert parsed_note.beams.getByNumber(2).direction == "left"


def test_kern_parser_supports_legacy_collection_parse_return(
    monkeypatch,
):
    from music21 import note, stream
    from music21.humdrum import spineParser

    from module.sheet_music_omr.decode import parse_kern_score

    legacy_score = stream.Score()
    for part_id, pitch in (("spine_1", "C4"), ("spine_0", "C3")):
        part = stream.Part(id=part_id)
        part.append(note.Note(pitch))
        legacy_score.insert(0, part)

    class LegacyCollection:
        def __init__(self, kern_text):
            self.stream = legacy_score
            self.spineCollection = None

        def parse(self):
            return None

    monkeypatch.setattr(
        spineParser,
        "HumdrumDataCollection",
        LegacyCollection,
    )
    kern = "**kern\t**kern\n*M4/4\t*M4/4\n4c\t4C\n*-\t*-\n"

    assert parse_kern_score(kern) is legacy_score


def test_reciprocal_only_event_fails_before_music21_can_warn(capsys):
    from module.sheet_music_omr.decode import KernStructureError, parse_kern_score

    kern = "**kern\t**kern\n*M4/4\t*M4/4\n16\t4e\n*-\t*-\n"

    with pytest.raises(KernStructureError, match=r"line 3.*16"):
        parse_kern_score(kern)

    assert "humdrum.spineParser: WARNING" not in capsys.readouterr().err
