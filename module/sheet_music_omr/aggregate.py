"""Deterministic page-score aggregation for one PDF document."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Sequence

from .decode import validate_kern_spines

_SPINE_MANIPULATORS = frozenset({"*^", "*v", "*x", "*+", "*-"})
_CONTEXT_AT_START = "start"
_CONTEXT_BEFORE_NEXT_DATA = "before_next_data"
_CONTEXT_AT_END = "end"


class PdfAggregationError(ValueError):
    """Raised when independently decoded pages cannot form one score."""


@dataclass(frozen=True)
class KernPageScore:
    score: Any
    kern_text: str


@dataclass(frozen=True)
class _KernContextPlacement:
    root_id: int
    measure_index: int
    kind: str
    token: str
    line_number: int
    position: str = _CONTEXT_AT_START


@dataclass(frozen=True)
class _KernEventPlacement:
    root_id: int
    line_number: int
    measure_index: int


@dataclass(frozen=True)
class _KernPageStructure:
    measure_slots: tuple[tuple[bool, bool], ...]
    contexts: tuple[_KernContextPlacement, ...]
    events: tuple[_KernEventPlacement, ...]


@dataclass(frozen=True)
class ScoreAggregationResult:
    score: Any
    padding_count: int
    padding_quarter_length: Any


def _ordered_parts(score: Any, *, page_number: int) -> tuple[Any, ...]:
    parts = tuple(score.parts)
    if not parts:
        raise PdfAggregationError(f"PDF page {page_number} contains no score parts")
    part_ids = tuple(str(part.id) for part in parts)
    if any(not part_id for part_id in part_ids):
        raise PdfAggregationError(f"PDF page {page_number} contains an empty part id")
    if len(set(part_ids)) != len(part_ids):
        raise PdfAggregationError(f"PDF page {page_number} contains duplicate part ids")
    return parts


def _next_root_ids(
    fields: list[str],
    root_ids: list[frozenset[int]],
    *,
    page_number: int,
    line_number: int,
) -> list[frozenset[int]]:
    exchange_indices = [index for index, field in enumerate(fields) if field == "*x"]
    if exchange_indices:
        if len(exchange_indices) != 2:
            raise PdfAggregationError(f"PDF page {page_number} Kern row {line_number} has an invalid spine exchange")
        exchanged = list(root_ids)
        left, right = exchange_indices
        exchanged[left], exchanged[right] = (
            exchanged[right],
            exchanged[left],
        )
        return exchanged

    next_ids: list[frozenset[int]] = []
    index = 0
    while index < len(fields):
        field = fields[index]
        root_id = root_ids[index]
        if field == "*v":
            run_end = index
            while run_end < len(fields) and fields[run_end] == "*v":
                run_end += 1
            joined_roots = root_ids[index:run_end]
            next_ids.append(frozenset().union(*joined_roots))
            index = run_end
            continue
        if field == "*^":
            next_ids.extend((root_id, root_id))
        elif field == "*+":
            raise PdfAggregationError(f"PDF page {page_number} Kern row {line_number} uses unsupported spine addition")
        elif field != "*-":
            next_ids.append(root_id)
        index += 1
    return next_ids


def _context_kind(field: str) -> str | None:
    if field.startswith("*clef"):
        return "clef"
    if field.startswith("*k[") and field.endswith("]"):
        return "key"
    if field.startswith("*M") and "/" in field:
        return "meter"
    if field in {"*met(c)", "*met(C)", "*met(c|)", "*met(C|)"}:
        return "meter_symbol"
    return None


def _kern_page_structure(
    kern_text: str,
    *,
    page_number: int,
) -> _KernPageStructure:
    try:
        validate_kern_spines(kern_text)
    except Exception as exc:
        raise PdfAggregationError(f"PDF page {page_number} has invalid Kern spine structure") from exc

    root_ids = [frozenset({0}), frozenset({1})]
    occupied = [False, False]
    measure_slots: list[tuple[bool, bool]] = []
    contexts: list[_KernContextPlacement] = []
    context_tokens: dict[tuple[int, int, str], str] = {}
    event_measure_indices: dict[tuple[int, int], int] = {}
    pending_contexts: dict[int, list[int]] = {0: [], 1: []}
    lines = kern_text[:-1].split("\n")
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        barline_fields = [field.startswith("=") for field in fields]
        if any(barline_fields):
            if not all(barline_fields):
                raise PdfAggregationError(f"PDF page {page_number} Kern row {line_number} has a partial barline")
            if not any(occupied):
                if not measure_slots:
                    continue
                raise PdfAggregationError(
                    f"PDF page {page_number} Kern measure slot {len(measure_slots) + 1} has no recognized events"
                )
            measure_slots.append((occupied[0], occupied[1]))
            occupied = [False, False]
            for root_id, context_indices in pending_contexts.items():
                for context_index in context_indices:
                    contexts[context_index] = replace(
                        contexts[context_index],
                        measure_index=len(measure_slots),
                    )
                pending_contexts[root_id] = []
            continue

        if any(field in _SPINE_MANIPULATORS for field in fields):
            root_ids = _next_root_ids(
                fields,
                root_ids,
                page_number=page_number,
                line_number=line_number,
            )
            continue

        context_fields = [_context_kind(field) for field in fields]
        if any(kind is not None for kind in context_fields):
            if not all(field.startswith("*") for field in fields):
                raise PdfAggregationError(f"PDF page {page_number} Kern row {line_number} mixes context interpretations and data")
            for roots, field, kind in zip(
                root_ids,
                fields,
                context_fields,
                strict=True,
            ):
                if kind is None:
                    continue
                if len(roots) != 1:
                    raise PdfAggregationError(
                        f"PDF page {page_number} Kern row {line_number} places context on a cross-part joined spine"
                    )
                root_id = next(iter(roots))
                measure_index = len(measure_slots)
                context_key = (root_id, line_number, kind)
                previous_token = context_tokens.get(context_key)
                if previous_token is not None:
                    if previous_token != field:
                        raise PdfAggregationError(
                            f"PDF page {page_number} Kern row {line_number} "
                            f"has conflicting {kind} interpretations for "
                            f"spine_{root_id}"
                    )
                    continue
                context_tokens[context_key] = field
                context_index = len(contexts)
                contexts.append(
                    _KernContextPlacement(
                        root_id=root_id,
                        measure_index=measure_index,
                        kind=kind,
                        token=field,
                        line_number=line_number,
                    )
                )
                if occupied[root_id]:
                    pending_contexts[root_id].append(context_index)
            continue

        for roots, field in zip(root_ids, fields, strict=True):
            if field != "." and not field.startswith("*") and not field.startswith("!"):
                if len(roots) != 1:
                    raise PdfAggregationError(
                        f"PDF page {page_number} Kern row {line_number} places data on a cross-part joined spine"
                    )
                root_id = next(iter(roots))
                for context_index in pending_contexts[root_id]:
                    contexts[context_index] = replace(
                        contexts[context_index],
                        position=_CONTEXT_BEFORE_NEXT_DATA,
                    )
                pending_contexts[root_id] = []
                event_key = (root_id, line_number)
                measure_index = len(measure_slots)
                previous_measure_index = event_measure_indices.get(event_key)
                if (
                    previous_measure_index is not None
                    and previous_measure_index != measure_index
                ):
                    raise PdfAggregationError(
                        f"PDF page {page_number} Kern row {line_number} maps "
                        f"spine_{root_id} data to multiple measures"
                    )
                event_measure_indices[event_key] = measure_index
                occupied[root_id] = True

    for root_id, context_indices in pending_contexts.items():
        for context_index in context_indices:
            contexts[context_index] = replace(
                contexts[context_index],
                position=_CONTEXT_AT_END,
            )
        pending_contexts[root_id] = []
    if any(occupied):
        measure_slots.append((occupied[0], occupied[1]))
    if not measure_slots:
        raise PdfAggregationError(f"PDF page {page_number} Kern contains no measure slots")
    for context_index, context in enumerate(contexts):
        if context.measure_index == len(measure_slots):
            contexts[context_index] = replace(
                context,
                measure_index=len(measure_slots) - 1,
                position=_CONTEXT_AT_END,
            )
            context = contexts[context_index]
        if context.measure_index >= len(measure_slots):
            raise PdfAggregationError(
                f"PDF page {page_number} Kern row {context.line_number} places {context.kind} context after the final measure"
            )
    return _KernPageStructure(
        measure_slots=tuple(measure_slots),
        contexts=tuple(contexts),
        events=tuple(
            _KernEventPlacement(
                root_id=root_id,
                line_number=line_number,
                measure_index=measure_index,
            )
            for (root_id, line_number), measure_index in event_measure_indices.items()
        ),
    )


def _part_root_id(part_id: str, *, page_number: int) -> int:
    expected_ids = {"spine_0": 0, "spine_1": 1}
    try:
        return expected_ids[part_id]
    except KeyError as exc:
        raise PdfAggregationError(
            f"PDF page {page_number} contains unsupported part id {part_id!r}; expected music21 Humdrum spine ids"
        ) from exc


def _is_empty_transport_measure(measure: Any) -> bool:
    return measure.quarterLength == 0 and not any(True for _ in measure.recurse().notesAndRests)


def _map_part_measures(
    part: Any,
    measure_slots: tuple[tuple[bool, bool], ...],
    *,
    page_number: int,
) -> tuple[Any | None, ...]:
    from music21 import stream

    part_id = str(part.id)
    root_id = _part_root_id(part_id, page_number=page_number)
    parsed_measures = tuple(part.getElementsByClass(stream.Measure))
    mapped: list[Any | None] = []
    parsed_index = 0
    for slot_index, occupancy in enumerate(measure_slots, start=1):
        candidate = parsed_measures[parsed_index] if parsed_index < len(parsed_measures) else None
        if occupancy[root_id]:
            if candidate is None or _is_empty_transport_measure(candidate):
                raise PdfAggregationError(
                    f"PDF page {page_number} part {part_id!r} measure count/content mismatch at Kern slot {slot_index}"
                )
            mapped.append(candidate)
            parsed_index += 1
        elif candidate is not None and _is_empty_transport_measure(candidate):
            mapped.append(candidate)
            parsed_index += 1
        else:
            mapped.append(None)

    remaining = len(parsed_measures) - parsed_index
    if remaining:
        raise PdfAggregationError(
            f"PDF page {page_number} part {part_id!r} measure count mismatch: {remaining} parsed measure(s) do not map to Kern"
        )
    return tuple(mapped)


def _context_classes() -> dict[str, type[Any]]:
    from music21 import clef, key, meter

    return {
        "clef": clef.Clef,
        "key": key.KeySignature,
        "meter": meter.TimeSignature,
    }


def _remove_measure_context(measure: Any) -> None:
    context_types = tuple(_context_classes().values())
    for element in tuple(measure.recurse().getElementsByClass(context_types)):
        active_site = element.activeSite
        if active_site is not None:
            active_site.remove(element)


def _context_insert_offset(
    placement: _KernContextPlacement,
    source_measure: Any | None,
    *,
    page_number: int,
    part_id: str,
) -> Any:
    if placement.position == _CONTEXT_AT_START:
        return 0
    if source_measure is None:
        raise PdfAggregationError(
            f"PDF page {page_number} part {part_id!r} cannot place "
            f"{placement.kind} context from Kern row {placement.line_number} "
            "inside an omitted measure"
        )
    if placement.position == _CONTEXT_AT_END:
        return source_measure.quarterLength
    if placement.position != _CONTEXT_BEFORE_NEXT_DATA:
        raise PdfAggregationError(
            f"PDF page {page_number} Kern row {placement.line_number} has "
            f"unknown context position {placement.position!r}"
        )

    offset = _next_data_offset(
        source_measure,
        after_line=placement.line_number,
        page_number=page_number,
        part_id=part_id,
        label=f"{placement.kind} context",
    )
    if offset is None:
        raise PdfAggregationError(
            f"PDF page {page_number} part {part_id!r} cannot locate data "
            f"after {placement.kind} context on Kern row "
            f"{placement.line_number}"
        )
    return offset


def _next_data_offset(
    source_measure: Any,
    *,
    after_line: int,
    page_number: int,
    part_id: str,
    label: str,
) -> Any | None:
    candidates = [
        event
        for event in source_measure.recurse().notesAndRests
        if isinstance(getattr(event, "priority", None), int)
        and event.priority > after_line
    ]
    if not candidates:
        return None
    next_line = min(event.priority for event in candidates)
    offsets = [
        event.getOffsetInHierarchy(source_measure)
        for event in candidates
        if event.priority == next_line
    ]
    offset = offsets[0]
    if any(candidate_offset != offset for candidate_offset in offsets[1:]):
        raise PdfAggregationError(
            f"PDF page {page_number} part {part_id!r} has ambiguous timing "
            f"after {label} on Kern row {after_line}"
        )
    return offset


def _restore_part_level_events(
    source_parts: tuple[Any, ...],
    mapped_by_part: dict[str, tuple[Any | None, ...]],
    prepared: dict[str, list[Any]],
    structure: _KernPageStructure,
    *,
    page_number: int,
) -> None:
    event_measure_indices = {
        (placement.root_id, placement.line_number): placement.measure_index
        for placement in structure.events
    }
    for part in source_parts:
        part_id = str(part.id)
        root_id = _part_root_id(part_id, page_number=page_number)
        for event in part.notesAndRests:
            line_number = getattr(event, "priority", None)
            if not isinstance(line_number, int):
                raise PdfAggregationError(
                    f"PDF page {page_number} part {part_id!r} has a "
                    "part-level event without Kern line metadata"
                )
            measure_index = event_measure_indices.get((root_id, line_number))
            if measure_index is None:
                raise PdfAggregationError(
                    f"PDF page {page_number} part {part_id!r} cannot map "
                    f"part-level event from Kern row {line_number}"
                )
            source_measure = mapped_by_part[part_id][measure_index]
            if source_measure is None:
                raise PdfAggregationError(
                    f"PDF page {page_number} part {part_id!r} cannot place "
                    f"part-level event from Kern row {line_number} inside "
                    "an omitted measure"
                )
            offset = _next_data_offset(
                source_measure,
                after_line=line_number,
                page_number=page_number,
                part_id=part_id,
                label="part-level event",
            )
            if offset is None:
                offset = source_measure.quarterLength
            prepared[part_id][measure_index].insert(
                offset,
                copy.deepcopy(event),
            )


def _prepare_page_measures(
    source_parts: tuple[Any, ...],
    mapped_by_part: dict[str, tuple[Any | None, ...]],
    structure: _KernPageStructure,
    *,
    page_number: int,
) -> dict[str, tuple[Any, ...]]:
    context_classes = _context_classes()
    source_context: dict[str, dict[str, tuple[Any, ...]]] = {}
    prepared: dict[str, list[Any]] = {}
    for part in source_parts:
        part_id = str(part.id)
        source_context[part_id] = {
            kind: tuple(part.recurse().getElementsByClass(context_class)) for kind, context_class in context_classes.items()
        }
        prepared[part_id] = []
        for measure_index, source_measure in enumerate(mapped_by_part[part_id]):
            measure = (
                _silent_measure(
                    mapped_by_part,
                    part_id=part_id,
                    measure_index=measure_index,
                )
                if source_measure is None
                else copy.deepcopy(source_measure)
            )
            _remove_measure_context(measure)
            prepared[part_id].append(measure)

    _restore_part_level_events(
        source_parts,
        mapped_by_part,
        prepared,
        structure,
        page_number=page_number,
    )

    placements_by_root_kind: dict[tuple[int, str], list[_KernContextPlacement]] = {}
    for placement in structure.contexts:
        placements_by_root_kind.setdefault(
            (placement.root_id, placement.kind),
            [],
        ).append(placement)

    for part in source_parts:
        part_id = str(part.id)
        root_id = _part_root_id(part_id, page_number=page_number)
        for kind in context_classes:
            placements = placements_by_root_kind.get((root_id, kind), [])
            elements = source_context[part_id][kind]
            elements_by_line: dict[int, Any] = {}
            for element in elements:
                line_number = getattr(element, "priority", None)
                if not isinstance(line_number, int) or line_number in elements_by_line:
                    raise PdfAggregationError(
                        f"PDF page {page_number} part {part_id!r} has "
                        f"ambiguous parsed {kind} context line metadata"
                    )
                elements_by_line[line_number] = element
            expected_lines = {placement.line_number for placement in placements}
            if set(elements_by_line) != expected_lines:
                raise PdfAggregationError(
                    f"PDF page {page_number} part {part_id!r} parsed "
                    f"{sorted(elements_by_line)} {kind} context line(s), but "
                    f"Kern declares {sorted(expected_lines)}"
                )
            for placement in placements:
                element = elements_by_line[placement.line_number]
                source_measure = mapped_by_part[part_id][
                    placement.measure_index
                ]
                offset = _context_insert_offset(
                    placement,
                    source_measure,
                    page_number=page_number,
                    part_id=part_id,
                )
                prepared[part_id][placement.measure_index].insert(
                    offset,
                    copy.deepcopy(element),
                )

    meter_class = context_classes["meter"]
    for placement in structure.contexts:
        if placement.kind != "meter_symbol":
            continue
        part_id = f"spine_{placement.root_id}"
        measure = prepared[part_id][placement.measure_index]
        time_signatures = tuple(measure.getElementsByClass(meter_class))
        if len(time_signatures) != 1:
            raise PdfAggregationError(
                f"PDF page {page_number} Kern row {placement.line_number} "
                "declares a meter symbol without one time signature in "
                f"spine_{placement.root_id}"
            )
        time_signatures[0].symbol = "cut" if "|" in placement.token else "common"
    return {part_id: tuple(measures) for part_id, measures in prepared.items()}


def _context_signature(element: Any) -> tuple[Any, ...] | None:
    from music21 import clef, key, meter

    if isinstance(element, clef.Clef):
        return (
            "clef",
            type(element).__name__,
            element.sign,
            element.line,
            element.octaveChange,
        )
    if isinstance(element, key.KeySignature):
        return ("key", element.sharps)
    if isinstance(element, meter.TimeSignature):
        return ("meter", element.ratioString, element.symbol)
    return None


def _deduplicate_context(
    measure: Any,
    state: dict[str, tuple[Any, ...]],
) -> None:
    for element in tuple(measure):
        signature = _context_signature(element)
        if signature is None:
            continue
        kind = str(signature[0])
        if state.get(kind) == signature:
            measure.remove(element)
        else:
            state[kind] = signature


def _silent_measure(
    mapped_by_part: dict[str, tuple[Any | None, ...]],
    *,
    part_id: str,
    measure_index: int,
) -> Any:
    from music21 import stream

    measure = stream.Measure()
    barline_template = next(
        (mapped[measure_index] for mapped in mapped_by_part.values() if mapped[measure_index] is not None),
        None,
    )
    if barline_template is not None:
        if barline_template.leftBarline is not None:
            measure.leftBarline = copy.deepcopy(barline_template.leftBarline)
        if barline_template.rightBarline is not None:
            measure.rightBarline = copy.deepcopy(barline_template.rightBarline)

    return measure


def _next_voice_id(measure: Any) -> int:
    from music21 import stream

    used_ids = {int(str(voice.id)) for voice in measure.getElementsByClass(stream.Voice) if str(voice.id).isdigit()}
    voice_id = 1
    while voice_id in used_ids:
        voice_id += 1
    return voice_id


def _set_beam(
    event: Any,
    number: int,
    beam_type: str,
    direction: str | None = None,
) -> None:
    existing = {beam.number: (beam.type, beam.direction) for beam in event.beams}
    if number not in existing:
        event.beams.fill(max((number, *existing)))
        for existing_number, (
            existing_type,
            existing_direction,
        ) in existing.items():
            if existing_type is None:
                continue
            event.beams.setByNumber(
                existing_number,
                existing_type,
                existing_direction,
            )
    event.beams.setByNumber(number, beam_type, direction)


def _required_beam_count(event: Any) -> int:
    from music21 import beam

    naive = beam.Beams.naiveBeams((event,))[0]
    return 0 if naive is None else len(naive)


def _primary_beam_groups(container: Any) -> tuple[tuple[Any, ...], ...]:
    groups: list[tuple[Any, ...]] = []
    pending: list[Any] = []
    for event in container.notes:
        try:
            primary = event.beams.getByNumber(1)
        except IndexError:
            pending = []
            continue
        if primary.type == "start":
            pending = [event]
        elif primary.type == "continue" and pending:
            pending.append(event)
        elif primary.type == "stop" and pending:
            pending.append(event)
            groups.append(tuple(pending))
            pending = []
        else:
            pending = []
    return tuple(groups)


def _beam_runs(
    group: tuple[Any, ...],
    level: int,
) -> tuple[tuple[int, tuple[Any, ...]], ...]:
    runs: list[tuple[int, tuple[Any, ...]]] = []
    run_start = 0
    run: list[Any] = []
    for index, event in enumerate(group):
        if _required_beam_count(event) >= level:
            if not run:
                run_start = index
            run.append(event)
        elif run:
            runs.append((run_start, tuple(run)))
            run = []
    if run:
        runs.append((run_start, tuple(run)))
    return tuple(runs)


def _complete_lazy_beams(part: Any) -> None:
    from music21 import stream

    for measure in part.getElementsByClass(stream.Measure):
        voices = tuple(measure.getElementsByClass(stream.Voice))
        containers = voices
        if tuple(measure.notes) or not containers:
            containers += (measure,)
        for container in containers:
            for group in _primary_beam_groups(container):
                max_level = max(
                    (_required_beam_count(event) for event in group),
                    default=1,
                )
                for level in range(2, max_level + 1):
                    for run_start, run in _beam_runs(group, level):
                        existing = tuple(beam for event in run for beam in event.beams if beam.number == level)
                        if existing:
                            continue
                        if len(run) == 1:
                            direction = "right" if run_start == 0 else "left"
                            _set_beam(
                                run[0],
                                level,
                                "partial",
                                direction,
                            )
                            continue
                        for index, event in enumerate(run):
                            beam_type = "continue"
                            if index == 0:
                                beam_type = "start"
                            elif index == len(run) - 1:
                                beam_type = "stop"
                            _set_beam(event, level, beam_type)


def _reconstruct_display_notation(parts: Sequence[Any]) -> None:
    for part in parts:
        part.makeAccidentals(
            inPlace=True,
            overrideStatus=True,
            cautionaryPitchClass=False,
            cautionaryNotImmediateRepeat=False,
        )
        _complete_lazy_beams(part)


def combine_page_scores(
    page_scores: Sequence[KernPageScore],
) -> ScoreAggregationResult:
    from music21 import layout, note, stream

    pages = tuple(page_scores)
    if not pages:
        raise PdfAggregationError("PDF aggregation requires at least one page")

    first_parts = _ordered_parts(pages[0].score, page_number=1)
    expected_part_ids = tuple(str(part.id) for part in first_parts)
    destination_parts = {part_id: stream.PartStaff(id=part_id) for part_id in expected_part_ids}
    combined = stream.Score(id="musvit_pdf_aggregate")
    for part_id in expected_part_ids:
        combined.insert(0, destination_parts[part_id])
    combined.insert(
        0,
        layout.StaffGroup(
            [destination_parts[part_id] for part_id in expected_part_ids],
            symbol="brace",
            barTogether=True,
        ),
    )

    measure_number = 1
    padding_count = 0
    padding_quarter_length: Any = 0
    context_state: dict[str, dict[str, tuple[Any, ...]]] = {part_id: {} for part_id in expected_part_ids}
    for page_number, page in enumerate(pages, start=1):
        source_parts = _ordered_parts(
            page.score,
            page_number=page_number,
        )
        actual_part_ids = tuple(str(part.id) for part in source_parts)
        if actual_part_ids != expected_part_ids:
            raise PdfAggregationError(f"PDF page {page_number} part ids {actual_part_ids!r} do not match {expected_part_ids!r}")

        structure = _kern_page_structure(
            page.kern_text,
            page_number=page_number,
        )
        mapped_by_part = {
            str(part.id): _map_part_measures(
                part,
                structure.measure_slots,
                page_number=page_number,
            )
            for part in source_parts
        }
        prepared_by_part = _prepare_page_measures(
            source_parts,
            mapped_by_part,
            structure,
            page_number=page_number,
        )
        for page_measure_index in range(len(structure.measure_slots)):
            alignment_duration = max(prepared_by_part[part_id][page_measure_index].quarterLength for part_id in expected_part_ids)
            for part_id in expected_part_ids:
                measure = prepared_by_part[part_id][page_measure_index]
                gap = alignment_duration - measure.quarterLength
                if gap > 0:
                    padding_rest = note.Rest(quarterLength=gap)
                    padding_offset = measure.highestTime
                    voices = tuple(measure.getElementsByClass(stream.Voice))
                    if voices:
                        padding_container = stream.Voice(id=_next_voice_id(measure))
                        measure.insert(0, padding_container)
                    else:
                        padding_container = measure
                    for rest_component in padding_rest.splitAtDurations():
                        rest_component.style.hideObjectOnPrint = True
                        padding_container.insert(
                            padding_offset,
                            rest_component,
                        )
                        padding_offset += rest_component.quarterLength
                    padding_count += 1
                    padding_quarter_length += gap
                _deduplicate_context(
                    measure,
                    context_state[part_id],
                )
                measure.number = measure_number
                destination_parts[part_id].append(measure)
            measure_number += 1

    _reconstruct_display_notation(tuple(destination_parts[part_id] for part_id in expected_part_ids))
    if combined.isWellFormedNotation() is False:
        raise PdfAggregationError("music21 rejected the aggregated PDF score structure")
    return ScoreAggregationResult(
        score=combined,
        padding_count=padding_count,
        padding_quarter_length=padding_quarter_length,
    )


def normalize_page_score(
    page_score: KernPageScore,
) -> ScoreAggregationResult:
    """Normalize one Kern page onto its shared measure/context grid."""
    return combine_page_scores((page_score,))
