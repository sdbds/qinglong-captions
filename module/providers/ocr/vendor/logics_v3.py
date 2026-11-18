"""Pure prompt/state/DSL helpers adapted from alibaba/Logics-Parsing (Apache-2.0).

Pinned source: d75d42dbd92c518a4def19ac5895792828138d35/inference_v3.py.
The model runtime/CLI are omitted; fragment boundary whitespace is preserved.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

WINDOW_SIZE = 2
MIN_PIXELS = 65536
MAX_PIXELS = 2048 * 2048
RESIZE_FACTOR = 32
PDF_RENDER_DPI = 200
MAX_NEW_TOKENS = 16384
POSTPROCESS_VERSION = "adjacent_paired_continuation_v2_no_trim"
STATE_MAX_CLOSED = 32
STATE_TAIL_CHARS = 200

SYSTEM_MESSAGE = "You are a helpful assistant"

LONG_DOCUMENT_PROMPT = """Task mode: longdoc_parse.
The attached page inputs are consecutive pages from a long document. Parse only these pages and output structured document labels.
Use the previous-window state below only as context for continuing the hierarchy or unfinished content. Do not copy the state into the answer.

Output rules:
- Output labels only, one item per line. Do not output markdown, JSON, explanations, or extra text.
- Follow the natural reading order across the current pages.
- Every label includes an absolute 0-based page id. Figure, chart and table locations, plus caption references to them, also include bounding-box coordinates normalized from 0 to 1000.
- Title: <|title|><|level|>N<|text|>TITLE_TEXT<|loc|>page, where N is 1-7.
- Text: <|text|>TEXT<|loc|>page.
- Table: <|table|>TABLE_HTML_OR_TEXT<|loc|>page,x1,y1,x2,y2.
- Figure/chart/formula: <|figure|><|loc|>page,x1,y1,x2,y2, <|chart|><|loc|>page,x1,y1,x2,y2, or <|formula|>FORMULA_TEXT<|loc|>page.
- Caption: <|caption|>CAPTION_TEXT<|loc|>page<|ref|>target_page,x1,y1,x2,y2.
- Use <|truncated|> when a text/table block continues later, and <|continued|> when it continues from earlier pages.
- For cross-column or merged text/table blocks, insert <break> at each content discontinuity and reduce each <|loc|> marker to its page id.
- Finish after the final complete label. Do not add a separate terminator line to the DSL.

Window metadata:
- Window index: {window_idx} (0-based)
- Window number: {window_number} / {total_windows}
- Total document pages: {total_pages}
- Is final window: {is_final_window}
Current page ids: {pages}

Previous-window state:
{state}
"""

SINGLE_PAGE_PROMPT = """Task mode: single_page_parse.
The attached page input is one page from a document. Parse only this page and output structured document labels.

Output rules:
- Output labels only, one item per line. Do not output markdown, JSON, explanations, or extra text.
- Follow the natural reading order on the current page.
- Only figure, chart and table labels carry a location: <|loc|>page,x1,y1,x2,y2, using the 0-based page id shown below and coordinates normalized as integers from 0 to 1000. All other labels carry no location.
- Title: <|title|>TITLE_TEXT.
- Text: <|text|>TEXT.
- Table: <|table|>TABLE_HTML_OR_TEXT<|loc|>page,x1,y1,x2,y2.
- Figure/chart: <|figure|><|loc|>page,x1,y1,x2,y2, or <|chart|><|loc|>page,x1,y1,x2,y2.
- Formula: <|formula|>FORMULA_TEXT.
- Caption: <|caption|>CAPTION_TEXT.
- Do not output title levels, hierarchy open/close tokens, continuation/truncation markers, document state, or markdown.
- Finish after the final complete label. Do not add a separate terminator line to the DSL.

Current page id: 0
"""

TITLE_STATE_RE = re.compile(
    r"<\|title\|><\|level\|>(?P<level>\d+)<\|text\|>(?P<text>.*?)(?=<\|loc\|>|$)"
)
# Keep page-only title locations, matching the training state.
STATE_LOC_RE = re.compile(r"<\|loc\|>(?P<loc>\d+(?:,\d+,\d+,\d+,\d+)?)")
DSL_TAG_RE = re.compile(r"<\|[^>]+?\|>")


@dataclass
class OpenTitle:
    level: int
    text: str
    loc: str = ""

    def to_state_line(self) -> str:
        line = f"<|title|><|level|>{self.level}<|text|>{self.text}"
        return line + (f"<|loc|>{self.loc}" if self.loc else "")


@dataclass
class WindowState:
    closed_sections: list[str] = field(default_factory=list)
    open_titles: list[OpenTitle] = field(default_factory=list)
    tail: Optional[str] = None

    def format(self) -> str:
        lines = ["<|state|>"]
        for section in self.closed_sections[-STATE_MAX_CLOSED:]:
            lines.append(f"<|section|>{section}<|closed|>")
        if self.open_titles:
            lines.append("<|open_tree|>")
            lines.extend(title.to_state_line() for title in self.open_titles)
            lines.append("<|/open_tree|>")
        if self.tail:
            lines.append(f"<|tail|>{self.tail}<|truncated|>")
        lines.append("<|/state|>")
        return "\n".join(lines)

    def update(self, output: str) -> None:
        last_content = None
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            title_match = TITLE_STATE_RE.search(line)
            if title_match:
                loc_match = STATE_LOC_RE.search(line)
                self._push_title(
                    OpenTitle(
                        level=int(title_match.group("level")),
                        text=" ".join(title_match.group("text").split()),
                        loc=loc_match.group("loc") if loc_match else "",
                    )
                )
            if line.startswith(
                (
                    "<|text|>",
                    "<|table|>",
                    "<|image|>",
                    "<|figure|>",
                    "<|chart|>",
                    "<|formula|>",
                    "<|caption|>",
                    "<|continued|>",
                )
            ):
                text = " ".join(DSL_TAG_RE.sub(" ", line).split())
                last_content = (
                    "..." + text[-STATE_TAIL_CHARS:]
                    if len(text) > STATE_TAIL_CHARS
                    else text
                )
                if "<|truncated|>" in line:
                    self.tail = last_content
        if last_content and not output.rstrip().endswith("<|truncated|>"):
            self.tail = last_content

    def _push_title(self, title: OpenTitle) -> None:
        while self.open_titles and self.open_titles[-1].level >= title.level:
            closed = self.open_titles.pop()
            if closed.text:
                self.closed_sections.append(closed.text)
        self.open_titles.append(title)



LABEL_START_RE = re.compile(
    r"^(<\|continued\|>)?(<\|title\|>|<\|text\|>|<\|table\|>|<\|image\|>|"
    r"<\|figure\|>|<\|chart\|>|<\|formula\|>|<\|caption\|>|"
    r"<\|code\|>|<\|pseudocode\|>|<\|chemistry\|>|<\|music\|>|<\|end\|>)"
)
LABEL_START_ANY_RE = re.compile(
    r"(?=(?:<\|continued\|>)?(?:<\|title\|>|<\|text\|>|<\|table\|>|<\|image\|>|"
    r"<\|figure\|>|<\|chart\|>|<\|formula\|>|<\|caption\|>|"
    r"<\|code\|>|<\|pseudocode\|>|<\|chemistry\|>|<\|music\|>|<\|end\|>))"
)
TAG_RE = re.compile(r"^<\|([^|]+)\|>")
NUMBER = r"-?(?:\d+(?:\.\d*)?|\.\d+)"
BBOX_RE = re.compile(
    rf"<\|loc\|>\s*({NUMBER})\s*,\s*({NUMBER})\s*,\s*({NUMBER})\s*,\s*"
    rf"({NUMBER})\s*,\s*({NUMBER})"
)
PAGE_RE = re.compile(rf"<\|loc\|>\s*({NUMBER})")
METADATA_RE = re.compile(rf"<\|(?:loc|ref)\|>\s*{NUMBER}(?:\s*,\s*{NUMBER}){{0,4}}")
TITLE_RE = re.compile(
    r"^<\|title\|><\|level\|>(?P<level>-?\d+)<\|text\|>"
    r"(?P<text>.*?)(?=<\|loc\|>|<\|ref\|>|<\|truncated\|>|$)",
    re.DOTALL,
)
CONTROL_RE = re.compile(r"<\|(end|continued|truncated)\|>")


def split_logical_lines(text: str) -> list[str]:
    logical_lines = []
    current_parts = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        starts = [match.start() for match in LABEL_START_ANY_RE.finditer(stripped)]
        title_payload = re.match(r"^<\|title\|><\|level\|>-?\d+(<\|text\|>)", stripped)
        if title_payload:
            starts = [start for start in starts if start != title_payload.start(1)]
        starts = [
            start
            for start in sorted(set(starts))
            if stripped[max(0, start - len("<|continued|>")) : start] != "<|continued|>"
        ]
        pieces = [stripped]
        if starts and starts[0] == 0:
            boundaries = starts + [len(stripped)]
            pieces = [
                stripped[boundaries[index] : boundaries[index + 1]].strip()
                for index in range(len(boundaries) - 1)
                if stripped[boundaries[index] : boundaries[index + 1]].strip()
            ]
        if len(pieces) > 1:
            if current_parts:
                logical_lines.append("\n".join(current_parts))
                current_parts = []
            logical_lines.extend(pieces)
        elif LABEL_START_RE.match(pieces[0]):
            if current_parts:
                logical_lines.append("\n".join(current_parts))
            current_parts = [pieces[0]]
        elif current_parts:
            current_parts.append(raw_line)
    if current_parts:
        logical_lines.append("\n".join(current_parts))
    return logical_lines


def clean_content(text: str) -> str:
    text = CONTROL_RE.sub("", text)
    text = METADATA_RE.sub("", text)
    text = re.sub(r"<\|[^|]+?\|>", "", text)
    return html.unescape(text).strip("\r\n")


def parse_dsl(raw_output: str) -> list[dict]:
    events = []
    for raw_line in split_logical_lines(raw_output):
        line = raw_line.strip()
        continued = line.startswith("<|continued|>")
        if continued:
            line = line[len("<|continued|>") :]
        if line == "<|end|>":
            continue
        locs = [
            tuple(int(float(match.group(index))) for index in range(1, 6))
            for match in BBOX_RE.finditer(line)
        ]
        page_ids = [int(float(match.group(1))) for match in PAGE_RE.finditer(line)]
        title_match = TITLE_RE.match(line)
        if title_match:
            events.append(
                {
                    "type": "title",
                    "level": int(title_match.group("level")),
                    "text": clean_content(title_match.group("text")),
                    "locs": locs,
                    "page_ids": page_ids,
                    "continued": continued,
                    "truncated": "<|truncated|>" in line,
                }
            )
            continue
        tag_match = TAG_RE.match(line)
        if not tag_match:
            continue
        kind = tag_match.group(1)
        if kind == "title":
            level = 1
        elif kind not in {
            "text",
            "table",
            "image",
            "figure",
            "chart",
            "formula",
            "caption",
            "code",
            "pseudocode",
            "chemistry",
            "music",
        }:
            continue
        else:
            level = None
        events.append(
            {
                "type": "figure" if kind == "image" else kind,
                "level": level,
                "text": clean_content(line[len(f"<|{kind}|>") :]),
                "locs": locs,
                "page_ids": page_ids,
                "continued": continued,
                "truncated": "<|truncated|>" in line,
            }
        )
    return events


def merge_tables(parts: list[str]) -> str:
    parts = [
        segment.strip()
        for part in parts
        for segment in re.split(r"\s*<break>\s*", part or "")
        if segment.strip()
    ]
    if len(parts) <= 1:
        return parts[0] if parts else ""
    rows = []
    for part in parts:
        table_rows = re.findall(r"<tr\b.*?</tr>", part, flags=re.IGNORECASE | re.DOTALL)
        rows.extend(table_rows or [f"<tr><td>{part}</td></tr>"])
    return "<table>" + "".join(rows) + "</table>"


def merge_continuations(events: list[dict]) -> list[dict]:
    merged = []
    previous = None
    mergeable = {"text", "table", "caption", "formula"}
    for source in events:
        event = dict(source)
        for key in ("locs", "page_ids", "refs"):
            if key in event:
                event[key] = list(event[key])
        kind = event["type"]
        if kind == "end":
            previous = None
            continue
        # Only the immediately preceding block can supply a continuation.
        if (
            previous is not None
            and kind in mergeable
            and previous["type"] == kind
            and previous.get("truncated")
            and event.get("continued")
        ):
            previous["parts"].append(event.get("text", ""))
            for key in ("locs", "page_ids", "refs"):
                if key in event:
                    previous.setdefault(key, []).extend(event[key])
            previous["text"] = (
                merge_tables(previous["parts"])
                if kind == "table" else "".join(previous["parts"])
            )
            previous["truncated"] = bool(event.get("truncated"))
            previous["continued_chain"] = True
        else:
            event["parts"] = [event.get("text", "")]
            merged.append(event)
            previous = event
    return merged


def strip_formula_delimiters(text: str) -> str:
    text = str(text or "").strip()
    pairs = (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$"))
    changed = True
    while text and changed:
        changed = False
        for left, right in pairs:
            if text.startswith(left) and text.endswith(right) and len(text) >= len(left) + len(right):
                text = text[len(left):-len(right)].strip()
                changed = True
                break
    for left, _ in pairs:
        if text.startswith(left) and len(text) > len(left):
            return text[len(left):].strip()
    return text


def _trim_inline_math_spaces(markdown: str) -> str:
    # Consume complete formulas so adjacent delimiters cannot pair across prose.
    pattern = re.compile(
        r"(?<!\\)\$\$[\s\S]*?\$\$"
        r"|(?<![\\$])\$(?!\$)(?P<math>(?:\\[^\r\n]|[^$\\\r\n])+)\$(?!\$)"
    )
    def replace(match):
        math = match.group("math")
        padded = re.fullmatch(
            r"[^\S\r\n]+((?:\\[^\r\n]|[^$\\\r\n])*?)[^\S\r\n]+", math or ""
        )
        return "$" + padded[1] + "$" if padded and padded[1].strip() else match.group(0)
    return pattern.sub(replace, markdown)


def to_markdown(
    raw_output: str,
    visual_renderer: Optional[Callable[[dict, int], Optional[str]]] = None,
) -> str:
    blocks = []
    visual_index = 0
    for event in merge_continuations(parse_dsl(raw_output)):
        kind = event["type"]
        content = event["text"]
        block = None
        if kind == "title" and content:
            level = max(1, min(int(event.get("level") or 1), 6))
            block = f"{'#' * level} {content}"
        elif kind in {"text", "caption", "code", "pseudocode", "chemistry", "music"}:
            block = content or None
        elif kind == "table":
            block = merge_tables([content]) or None
        elif kind == "formula":
            formula = strip_formula_delimiters(content)
            block = f"$${formula}$$" if formula else None
        elif kind in {"figure", "chart"}:
            block = visual_renderer(event, visual_index) if visual_renderer else None
            visual_index += 1
        if block:
            blocks.append(block.strip())
    return _trim_inline_math_spaces(re.sub(r"\n{3,}", "\n\n", "\n\n".join(blocks)).strip())


def to_hierarchy_markdown(raw_output: str) -> str:
    titles = [
        event
        for event in parse_dsl(raw_output)
        if event["type"] == "title" and event["text"]
    ]
    lines = ["## 文档目录", ""]
    if not titles:
        return "\n".join(lines + ["_未识别到标题层级_"])
    level_stack = []
    for event in titles:
        level = max(1, min(int(event.get("level") or 1), 7))
        while level_stack and level_stack[-1] >= level:
            level_stack.pop()
        title = re.sub(
            r"([\\`*_[\]<>])", r"\\\1", event["text"].replace("\n", " ").strip()
        )
        page_ids = event.get("page_ids") or []
        page = f" ····· {page_ids[0] + 1}" if page_ids else ""
        label = f"**{title}**" if not level_stack else title
        lines.append(f"{'  ' * len(level_stack)}- {label}{page}")
        level_stack.append(level)
    return "\n".join(lines)


def postprocess(
    raw_output: str, pages: list[Path], output_dir: Path, mode: str
) -> tuple[str, str]:
    image_dir = output_dir / "images"

    def render_visual(event: dict, visual_index: int) -> Optional[str]:
        rendered = []
        for location_index, location in enumerate(event["locs"]):
            page_index, x1, y1, x2, y2 = location
            if not 0 <= page_index < len(pages):
                continue
            with Image.open(pages[page_index]) as source:
                width, height = source.size
                box = (
                    max(0, min(int(x1 / 1000 * width), width)),
                    max(0, min(int(y1 / 1000 * height), height)),
                    max(0, min(int(x2 / 1000 * width), width)),
                    max(0, min(int(y2 / 1000 * height), height)),
                )
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                image_dir.mkdir(parents=True, exist_ok=True)
                filename = f"page_{page_index + 1:04d}_{event['type']}_{visual_index}_{location_index}.png"
                source.crop(box).save(image_dir / filename)
                rendered.append(f"![{event['type']}](images/{filename})")
        return "\n\n".join(rendered) or None

    markdown = to_markdown(raw_output, render_visual)
    hierarchy = to_hierarchy_markdown(raw_output) if mode == "long_document" else ""
    return markdown, hierarchy
