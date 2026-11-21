"""Source ranges of rendered image links, for editing Markdown without reserializing it."""

import re
from bisect import bisect_right
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Iterator

from markdown_it import MarkdownIt
from markdown_it.rules_inline import html_inline, image

_HTML_ATTRIBUTE = re.compile(r"""([^\s/=>]+)(?:\s*=\s*("[^"]*"|'[^']*'|[^\s>]*))?""")


@dataclass(frozen=True)
class ImageLink:
    start: int
    end: int
    url_start: int
    url_end: int
    href: str
    is_markdown: bool


class _HTMLImageParser(HTMLParser):
    CDATA_CONTENT_ELEMENTS = (*HTMLParser.CDATA_CONTENT_ELEMENTS, "textarea", "title")

    def __init__(self, text: str, offset: int):
        super().__init__()
        self.line_starts = [offset, *(offset + m.end() for m in re.finditer("\n", text))]
        self.links = []
        self.feed(text)
        self.close()

    def parse_marked_section(self, i, report=1):
        try:
            return super().parse_marked_section(i, report)
        except AssertionError:
            # HTML treats unrecognized declarations as bogus comments, not a fatal parse error.
            return self.parse_bogus_comment(i, report)

    def handle_starttag(self, tag, attrs):
        if tag != "img":
            return
        raw = self.get_starttag_text()
        line, column = self.getpos()
        start = self.line_starts[line - 1] + column
        # HTMLParser handles comments, quoting and character references, but doesn't
        # expose attribute offsets. Locate the parsed attribute in this exact start tag.
        for (name, value), match in zip(attrs, _HTML_ATTRIBUTE.finditer(raw, len("<img"))):
            if match[1].lower() != name:
                return
            if name == "src":
                if value and match[2]:
                    quote = int(match[2][0] in "\"'")
                    self.links.append(
                        ImageLink(
                            start, start + len(raw), start + match.start(2) + quote, start + match.end(2) - quote, value, False
                        )
                    )
                return


def iter_image_links(markdown: str) -> Iterator[ImageLink]:
    """Find inline Markdown destinations and HTML img sources outside code examples."""
    if "![" not in markdown and "<img" not in markdown.lower():
        return
    # The block/inline APIs expect the normalized input normally supplied by CommonMark's core.
    # Only CRLF shortens the text; record its end boundary in normalized coordinates.
    crlf_ends = [match.end() - count for count, match in enumerate(re.finditer("\r\n", markdown), 1)]
    markdown = markdown.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "\ufffd")
    parser = MarkdownIt("commonmark")
    links = []

    def capture_image(state, silent):
        start = state.pos
        matched = image(state, silent)
        if not matched or silent or state.src is not raw:
            return matched
        label_end = parser.helpers.parseLinkLabel(state, start + 1, False)
        if raw[label_end + 1 : label_end + 2] != "(":
            return True  # A reference definition, not an inline destination we can edit.
        url_start = label_end + 2
        while url_start < state.pos and raw[url_start] in " \t\n\r":
            url_start += 1
        destination = parser.helpers.parseLinkDestination(raw, url_start, state.pos)
        if destination.ok:
            bracket = int(raw[url_start] == "<")
            links.append(
                ImageLink(
                    base + start,
                    base + state.pos,
                    base + url_start + bracket,
                    base + destination.pos - bracket,
                    state.tokens[-1].attrs["src"],
                    True,
                )
            )
        return True

    def capture_inline_html(state, silent):
        start = state.pos
        matched = html_inline(state, silent)
        if matched and not silent and state.src is raw and raw[start : start + 4].lower() == "<img":
            links.extend(_HTMLImageParser(raw[start : state.pos], base + start).links)
        return matched

    parser.inline.ruler.at("image", capture_image)
    parser.inline.ruler.at("html_inline", capture_inline_html)
    # Block maps exclude fenced/indented code. Tokenize the original slices so offsets
    # address the source, not CommonMark's trimmed paragraph/list contents.
    starts = [0, *(match.end() for match in re.finditer("\n", markdown)), len(markdown)]
    blocks, env = [], {}
    parser.block.parse(markdown, parser, env, blocks)
    for block in blocks:
        if block.type not in {"inline", "html_block"} or not block.map:
            continue
        base, end = (starts[line] for line in block.map)
        raw = markdown[base:end]
        if block.type == "inline":
            parser.inline.parse(raw, parser, env, [])
        elif "<img" in raw.lower() and not raw.lstrip().lower().startswith(("<pre", "<code")):
            links.extend(_HTMLImageParser(raw, base).links)
    if not crlf_ends:
        yield from links
        return
    for link in links:
        positions = (link.start, link.end, link.url_start, link.url_end)
        yield ImageLink(*(pos + bisect_right(crlf_ends, pos) for pos in positions), link.href, link.is_markdown)
