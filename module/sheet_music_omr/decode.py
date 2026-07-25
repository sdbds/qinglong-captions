"""BeKern token reconstruction and strict two-spine Kern validation."""

from __future__ import annotations

from typing import Any, Iterable

_SEMANTIC_TOKENS = {
    "<s>": " ",
    "<t>": "\t",
    "<b>": "\n",
}
_SPINE_MANIPULATORS = frozenset({"*^", "*v", "*x", "*+", "*-"})


class KernStructureError(ValueError):
    """Raised when decoded tokens cannot form the pinned two-spine document."""

    def __init__(self, message: str, *, candidate: str | None = None) -> None:
        super().__init__(message)
        self.candidate = candidate


def decode_bekern_tokens(tokens: Iterable[str]) -> str:
    decoded: list[str] = []
    for token in tokens:
        if not isinstance(token, str):
            raise TypeError(f"BeKern token must be a string, got {type(token).__name__}")
        decoded.append(_SEMANTIC_TOKENS.get(token, token))
    return "".join(decoded)


def _normalized_candidate(decoded_text: str) -> str:
    text = decoded_text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    return text + "\n"


def diagnostic_kern_candidate(tokens: Iterable[str]) -> str:
    return _normalized_candidate(decode_bekern_tokens(tokens))


def reconstruct_kern_envelope(decoded_text: str) -> str:
    text = decoded_text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if not text:
        raise KernStructureError("decoded Kern body is empty")

    lines = text.split("\n")
    if lines[0] == "**kern\t**kern":
        pass
    elif lines[0].startswith("**"):
        raise KernStructureError("unsupported exclusive interpretation")
    else:
        lines.insert(0, "**kern\t**kern")

    if any(line.startswith("**") for line in lines[1:]):
        raise KernStructureError("exclusive interpretation outside the first line")

    terminal_fields = lines[-1].split("\t")
    if terminal_fields and all(field == "*-" for field in terminal_fields):
        pass
    elif lines[-1] == "*-\t":
        lines[-1] = "*-\t*-"
    else:
        raise KernStructureError("missing a complete Kern spine terminator")

    kern_text = "\n".join(lines) + "\n"
    validate_kern_spines(kern_text)
    return kern_text


def next_kern_spine_count(fields: list[str], *, line_number: int) -> int:
    if not any(field in _SPINE_MANIPULATORS for field in fields):
        return len(fields)
    if not all(field.startswith("*") for field in fields):
        raise KernStructureError(f"spine operation row {line_number} mixes interpretations and data")

    exchange_count = sum(field == "*x" for field in fields)
    if exchange_count not in {0, 2}:
        raise KernStructureError(f"spine exchange row {line_number} must contain exactly two *x tokens")
    if any(field == "*+" for field in fields):
        raise KernStructureError(f"unsupported spine addition operation on row {line_number}")

    next_count = 0
    index = 0
    while index < len(fields):
        field = fields[index]
        if field == "*v":
            run_end = index
            while run_end < len(fields) and fields[run_end] == "*v":
                run_end += 1
            if run_end - index < 2:
                raise KernStructureError(f"spine join row {line_number} requires adjacent *v tokens")
            next_count += 1
            index = run_end
            continue
        if field == "*^":
            next_count += 2
        elif field == "*-":
            pass
        else:
            next_count += 1
        index += 1
    return next_count


def validate_kern_spines(kern_text: str) -> None:
    if not kern_text.endswith("\n"):
        raise KernStructureError("Kern document must end with a newline")
    lines = kern_text[:-1].split("\n")
    if not lines or lines[0] != "**kern\t**kern":
        raise KernStructureError("Kern document must start with two **kern spines")

    spine_count = 2
    for line_number, line in enumerate(lines[1:], start=2):
        if not line:
            raise KernStructureError(f"empty Kern row at line {line_number}")
        fields = line.split("\t")
        is_final = line_number == len(lines)
        if len(fields) != spine_count:
            label = (
                "Kern terminator spine count mismatch"
                if is_final and all(field == "*-" for field in fields)
                else "Kern spine count mismatch"
            )
            raise KernStructureError(f"{label} on line {line_number}: expected {spine_count}, got {len(fields)}")
        next_count = next_kern_spine_count(
            fields,
            line_number=line_number,
        )
        if next_count == 0 and not is_final:
            raise KernStructureError(f"all Kern spines terminate before the final row at line {line_number}")
        if is_final and next_count != 0:
            raise KernStructureError("final Kern row does not terminate every spine")
        spine_count = next_count


def validate_kern_events(kern_text: str) -> None:
    from music21 import note
    from music21.humdrum.spineParser import hdStringToNote

    lines = kern_text[:-1].split("\n")
    for line_number, line in enumerate(lines[1:], start=2):
        for field in line.split("\t"):
            if field == "." or field.startswith("*") or field.startswith("=") or field.startswith("!"):
                continue
            try:
                parsed = tuple(hdStringToNote(component) for component in field.split())
                if " " in field and not any(isinstance(item, note.Note) for item in parsed):
                    raise ValueError("chord contains no pitched notes")
            except Exception as exc:
                raise KernStructureError(f"invalid Kern event at line {line_number}: {field!r}: {exc}") from exc


def reconstruct_kern(tokens: Iterable[str]) -> str:
    candidate = diagnostic_kern_candidate(tokens)
    try:
        kern_text = reconstruct_kern_envelope(candidate)
        validate_kern_spines(kern_text)
    except KernStructureError as exc:
        if exc.candidate is None:
            exc.candidate = candidate
        raise
    return kern_text


def _restore_partial_beam_directions(data_collection: Any) -> None:
    spine_collection = data_collection.spineCollection
    if spine_collection is None:
        return

    for spine in spine_collection.spines:
        objects_by_position: dict[int, list[Any]] = {}
        for event in spine.stream.recurse().notes:
            position = getattr(event, "priority", None)
            if position is not None:
                objects_by_position.setdefault(position, []).append(event)

        for source_event in spine.eventList:
            if source_event is None:
                continue
            contents = source_event.contents or ""
            directions = ("left",) * contents.count("k") + ("right",) * contents.count("K")
            if not directions:
                continue
            source_position = getattr(source_event, "lineNumber", None)
            if source_position is None:
                source_position = getattr(source_event, "position", None)
            if source_position is None:
                continue
            candidates = []
            for event in objects_by_position.get(source_position, ()):
                partial_beams = tuple(beam for beam in event.beams if beam.type == "partial")
                if len(partial_beams) == len(directions):
                    candidates.append((event, partial_beams))
            if len(candidates) != 1:
                continue
            _, partial_beams = candidates[0]
            for beam, direction in zip(
                partial_beams,
                directions,
                strict=True,
            ):
                beam.direction = direction


def parse_kern_score(kern_text: str) -> Any:
    validate_kern_spines(kern_text)
    validate_kern_events(kern_text)
    from music21.humdrum.spineParser import HumdrumDataCollection

    data_collection = HumdrumDataCollection(kern_text)
    parsed = data_collection.parse()
    score = parsed if parsed is not None else data_collection.stream
    _restore_partial_beam_directions(data_collection)
    parts = tuple(getattr(score, "parts", ()))
    if not parts:
        raise KernStructureError("music21 parsed no score parts from Kern")
    if not any(True for _ in score.recurse().notesAndRests):
        raise KernStructureError("music21 parsed no notes or rests from Kern")
    return score
