"""Stateful Logics-Parsing-V3 windows independent of inference transport."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from utils.output_writer import write_markdown_output

from .document import page_metadata, write_layout_json
from .vendor import logics_v3 as upstream

_REF_RE = re.compile(upstream.BBOX_RE.pattern.replace("loc", "ref"))


def parse_blocks(raw: str, *, page_ids: list[int], total_pages: int, single_page: bool) -> list[dict]:
    blocks = []
    for line in upstream.split_logical_lines(raw):
        for block in upstream.parse_dsl(line):
            block["refs"] = [[int(float(match[i])) for i in range(1, 6)] for match in _REF_RE.finditer(line)]
            if single_page and not block["page_ids"]:
                block["page_ids"] = [0]
            if not single_page and not block["page_ids"]:
                raise ValueError("Logics V3 long-document labels must include page locations")
            if block["type"] == "title" and not 1 <= int(block.get("level") or 1) <= 7:
                raise ValueError("Logics V3 title levels must be between 1 and 7")
            if any(page not in page_ids for page in block["page_ids"]):
                raise ValueError("Logics V3 emitted a location outside the current page window")
            for page, x1, y1, x2, y2 in [*block["locs"], *block["refs"]]:
                if not 0 <= page < total_pages or not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                    raise ValueError("Logics V3 emitted an invalid bbox or caption reference")
            blocks.append(block)
    if raw.strip() and not blocks:
        raise ValueError("Logics V3 returned no valid document labels")
    return blocks


def run_document(
    pages: list[Path],
    infer: Callable[[list[Path], str], str],
    output_dir: Path,
    *,
    model_id: str,
    window_size: int = 2,
) -> str:
    if window_size < 1:
        raise ValueError("Logics V3 window_size must be >= 1")
    if not pages:
        raise ValueError("Logics V3 document has no pages")
    single_page = len(pages) == 1
    state = upstream.WindowState()
    windows, outputs, blocks = [], [], []
    total_windows = (len(pages) + window_size - 1) // window_size
    for index, start in enumerate(range(0, len(pages), window_size)):
        window = pages[start : start + window_size]
        page_ids = list(range(start, start + len(window)))
        prompt = (
            upstream.SINGLE_PAGE_PROMPT
            if single_page
            else upstream.LONG_DOCUMENT_PROMPT.format(
                window_idx=index,
                window_number=index + 1,
                total_windows=total_windows,
                total_pages=len(pages),
                is_final_window=str(index == total_windows - 1).lower(),
                pages=", ".join(map(str, page_ids)),
                state=state.format(),
            )
        )
        raw = re.sub(r"<?\|end\|>", "", infer(window, prompt)).strip("\r\n")
        window_blocks = parse_blocks(raw, page_ids=page_ids, total_pages=len(pages), single_page=single_page)
        blocks.extend(window_blocks)
        state.update(raw)
        # The formatted state is bounded; bound the retained history as well.
        state.closed_sections = state.closed_sections[-upstream.STATE_MAX_CLOSED :]
        outputs.append(raw)
        windows.append({"pages": page_ids, "raw_output": raw})
    raw_output = "\n".join(outputs)
    mode = "single_page" if single_page else "long_document"
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown, hierarchy = upstream.postprocess(raw_output, pages, output_dir, mode)
    merged = upstream.merge_continuations(blocks)
    # References belong to captions; merging must retain references from every fragment.
    for block in merged:
        block["page_ids"] = list(dict.fromkeys(block["page_ids"]))
    (output_dir / "raw_output.dsl").write_text(raw_output, encoding="utf-8")
    write_markdown_output(output_dir, hierarchy, filename="hierarchy.md")
    write_layout_json(
        output_dir,
        {
            "schema_version": 1,
            "model_id": model_id,
            "coordinate_system": "normalized_0_1000",
            "mode": mode,
            "page_count": len(pages),
            "pages": page_metadata(pages, output_dir),
            "window_size": window_size if not single_page else 1,
            "windows": windows,
            "blocks": merged,
            "state": state.format(),
            "raw_output": raw_output,
            "markdown": markdown,
            "complete_generation": True,
        },
    )
    return markdown
