from __future__ import annotations

from pathlib import Path
from typing import Any

MIDI_REPEAT_FALLBACK_WARNING = (
    "repeat expansion failed; exported written-order MIDI with repeat "
    "directives removed"
)


class Music21WriteError(ValueError):
    """Raised when music21 cannot serialize the supplied notation score."""


def _require_well_formed_score(score: Any) -> None:
    checker = getattr(score, "isWellFormedNotation", None)
    if checker is not None and checker() is False:
        raise Music21WriteError("music21 rejected the generated notation")


def _write_score(
    score: Any,
    format_name: str,
    target: str | Path,
    **write_kwargs: Any,
) -> None:
    target = Path(target)
    _require_well_formed_score(score)
    score.write(format_name, fp=str(target), **write_kwargs)
    if not target.is_file():
        raise Music21WriteError(
            f"music21 did not create the requested {format_name} file: {target}"
        )


def write_musicxml_score(
    score: Any,
    target: str | Path,
    *,
    make_notation: bool | None = None,
) -> None:
    write_kwargs = (
        {}
        if make_notation is None
        else {"makeNotation": bool(make_notation)}
    )
    _write_score(score, "musicxml", target, **write_kwargs)


def _without_repeat_directives(score: Any) -> Any:
    import copy

    from music21 import bar, repeat, spanner, stream

    linear_score = copy.deepcopy(score)
    for measure in linear_score.recurse().getElementsByClass(stream.Measure):
        if isinstance(measure.leftBarline, bar.Repeat):
            measure.leftBarline = None
        if isinstance(measure.rightBarline, bar.Repeat):
            measure.rightBarline = bar.Barline("regular")

    repeat_elements = tuple(
        element
        for element in linear_score.recurse()
        if isinstance(
            element,
            (repeat.RepeatExpression, spanner.RepeatBracket),
        )
    )
    if repeat_elements:
        linear_score.remove(repeat_elements, recurse=True)
    return linear_score


def write_midi_score(score: Any, target: str | Path) -> str | None:
    from music21 import repeat

    target = Path(target)
    try:
        _write_score(score, "midi", target)
    except repeat.ExpanderException:
        target.unlink(missing_ok=True)
        _write_score(_without_repeat_directives(score), "midi", target)
        return MIDI_REPEAT_FALLBACK_WARNING
    return None
