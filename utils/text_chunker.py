from __future__ import annotations

import re
from typing import List, Sequence, Tuple

_SENTENCE_RE = re.compile(r"[。！？.!?]+[\"')\]]*(?:\s+|$)")
_PROTECTED_SPAN_RE = re.compile(
    r"https?://[^\s)>]+|"
    r"(?:[A-Za-z]:\\|\\\\|(?:\./|\.\./|/))[^\s)>]+"
)
_LIST_ITEM_RE = re.compile(r"(?:[-*+]\s+|\d+[.)]\s+)")


def compute_chunk_offsets(markdown: str, max_chars: int = 2000) -> List[int]:
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")
    if not markdown:
        return []

    chunks: list[tuple[int, int]] = []
    chunk_start = 0
    chunk_end = 0

    for start, end, kind in _segment_blocks(markdown):
        block_len = end - start
        if block_len <= 0:
            continue

        if block_len > max_chars and kind not in {"code", "table"}:
            if chunk_end > chunk_start:
                chunks.append((chunk_start, chunk_end))
                chunk_start = 0
                chunk_end = 0
            chunks.extend(_split_large_span(markdown, start, end, max_chars))
            continue

        if chunk_end <= chunk_start:
            chunk_start = start
            chunk_end = end
            continue

        if end - chunk_start > max_chars:
            chunks.append((chunk_start, chunk_end))
            chunk_start = start
        chunk_end = end

    if chunk_end > chunk_start:
        chunks.append((chunk_start, chunk_end))

    if not chunks:
        chunks = _split_large_span(markdown, 0, len(markdown), max_chars)

    offsets: list[int] = []
    for _, end in chunks:
        if not offsets or offsets[-1] != end:
            offsets.append(end)
    if not offsets:
        return [len(markdown)]
    if offsets[-1] != len(markdown):
        offsets[-1] = len(markdown)
    return offsets


def slice_by_offsets(markdown: str, offsets: Sequence[int]) -> List[str]:
    chunks: list[str] = []
    start = 0
    for end in offsets:
        chunks.append(markdown[start:end])
        start = end
    return chunks


def _segment_blocks(text: str) -> List[Tuple[int, int, str]]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return []

    starts = _line_starts(lines)
    blocks: list[tuple[int, int, str]] = []
    index = 0

    while index < len(lines):
        stripped = lines[index].lstrip()
        if not stripped:
            index += 1
            continue

        start = starts[index]

        if _is_fence_start(stripped):
            end_index = _consume_fence(lines, index)
            blocks.append((start, starts[end_index], "code"))
            index = end_index
            continue

        if _looks_like_table_header(lines, index):
            end_index = _consume_table(lines, index)
            blocks.append((start, starts[end_index], "table"))
            index = end_index
            continue

        if _looks_like_heading(stripped):
            blocks.append((start, starts[index + 1], "heading"))
            index += 1
            continue

        if _looks_like_blockquote(stripped):
            end_index = _consume_blockquote(lines, index)
            blocks.append((start, starts[end_index], "blockquote"))
            index = end_index
            continue

        if _looks_like_list_item(stripped):
            end_index = _consume_list(lines, index)
            blocks.append((start, starts[end_index], "list"))
            index = end_index
            continue

        end_index = _consume_paragraph(lines, index)
        blocks.append((start, starts[end_index], "paragraph"))
        index = end_index

    return blocks


def _line_starts(lines: Sequence[str]) -> List[int]:
    starts = [0]
    cursor = 0
    for line in lines:
        cursor += len(line)
        starts.append(cursor)
    return starts


def _is_fence_start(stripped: str) -> bool:
    return stripped.startswith("```") or stripped.startswith("~~~")


def _consume_fence(lines: Sequence[str], start_index: int) -> int:
    opener = lines[start_index].lstrip()[:3]
    index = start_index + 1
    while index < len(lines):
        if lines[index].lstrip().startswith(opener):
            return index + 1
        index += 1
    return len(lines)


def _looks_like_heading(stripped: str) -> bool:
    return bool(re.match(r"#{1,6}\s", stripped))


def _looks_like_table_header(lines: Sequence[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    return _looks_like_table_row(lines[index]) and _is_table_separator(lines[index + 1])


def _looks_like_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.count("|") >= 2


def _is_table_separator(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    return set(stripped.replace("|", "").strip()) <= {"-", ":", " "}


def _consume_table(lines: Sequence[str], start_index: int) -> int:
    index = start_index + 2
    while index < len(lines) and _looks_like_table_row(lines[index]):
        index += 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    return index


def _looks_like_blockquote(stripped: str) -> bool:
    return stripped.startswith(">")


def _consume_blockquote(lines: Sequence[str], start_index: int) -> int:
    index = start_index
    while index < len(lines):
        stripped = lines[index].lstrip()
        if not stripped:
            index += 1
            continue
        if not stripped.startswith(">"):
            break
        index += 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    return index


def _looks_like_list_item(stripped: str) -> bool:
    return bool(_LIST_ITEM_RE.match(stripped))


def _looks_like_list_continuation(line: str) -> bool:
    stripped = line.lstrip()
    if not stripped:
        return True
    if _looks_like_list_item(stripped):
        return True
    indent = len(line) - len(line.lstrip(" "))
    return indent >= 2


def _consume_list(lines: Sequence[str], start_index: int) -> int:
    index = start_index
    while index < len(lines):
        if not _looks_like_list_continuation(lines[index]):
            break
        index += 1
        if index < len(lines) and not lines[index - 1].strip():
            break
    while index < len(lines) and not lines[index].strip():
        index += 1
    return index


def _consume_paragraph(lines: Sequence[str], start_index: int) -> int:
    index = start_index
    while index < len(lines):
        stripped = lines[index].lstrip()
        if not stripped:
            break
        if index != start_index and (
            _is_fence_start(stripped)
            or _looks_like_heading(stripped)
            or _looks_like_table_header(lines, index)
            or _looks_like_blockquote(stripped)
            or _looks_like_list_item(stripped)
        ):
            break
        index += 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    return index


def _split_large_span(text: str, start: int, end: int, max_chars: int) -> List[Tuple[int, int]]:
    segment = text[start:end]
    protected_spans = [(start + match.start(), start + match.end()) for match in _PROTECTED_SPAN_RE.finditer(segment)]
    sentence_ends = [
        start + match.end()
        for match in _SENTENCE_RE.finditer(segment)
        if not _is_inside_protected(start + match.start(), protected_spans)
    ]

    chunks: list[tuple[int, int]] = []
    current_start = start

    for boundary in sentence_ends:
        if boundary - current_start > max_chars:
            chunks.extend(_hard_split(text, current_start, boundary, max_chars))
            current_start = chunks[-1][1]
        if boundary > current_start and boundary - current_start <= max_chars:
            chunks.append((current_start, boundary))
            current_start = boundary

    if current_start < end:
        remaining = end - current_start
        if remaining > max_chars:
            chunks.extend(_hard_split(text, current_start, end, max_chars))
        else:
            chunks.append((current_start, end))

    return chunks or [(start, end)]


def _is_inside_protected(position: int, spans: Sequence[Tuple[int, int]]) -> bool:
    for start, end in spans:
        if start <= position < end:
            return True
    return False


def _hard_split(text: str, start: int, end: int, max_chars: int) -> List[Tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        split_at = min(end, cursor + max_chars)
        newline_at = text.rfind("\n", cursor, split_at)
        if newline_at > cursor + max_chars // 2:
            split_at = newline_at + 1
        elif split_at < end:
            space_at = text.rfind(" ", cursor, split_at)
            if space_at > cursor + max_chars // 2:
                split_at = space_at + 1
        chunks.append((cursor, split_at))
        cursor = split_at
    return chunks
