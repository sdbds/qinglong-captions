# -*- coding: utf-8 -*-
"""
Parsing and display helpers for caption workflows.
All logs and comments are in English per project convention.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Any

from rich.console import Console
from rich.text import Text

from utils.console_util import (
    CaptionAndRateLayout,
    CaptionPairImageLayout,
    MarkdownLayout,
    CaptionLayout,
)


def extract_code_block_content(response_text: str, code_type: Optional[str] = None, console: Optional[Console] = None) -> str:
    """Extract the content between the last pair of triple backticks.

    - If code_type is provided and the content starts with it, strip the type prefix.
    - Preserve the original logging semantics.
    """
    if not response_text:
        return ""

    markers: List[int] = []
    start = 0
    while True:
        pos = response_text.find("```", start)
        if pos == -1:
            break
        markers.append(pos)
        start = pos + 3

    if len(markers) >= 2:
        first_marker = markers[-2]
        second_marker = markers[-1]
        content = response_text[first_marker + 3 : second_marker].strip()
        if code_type and content.startswith(code_type):
            content = content[len(code_type) :].strip()
        if console:
            console.print(f"[blue]Extracted content length:[/blue] {len(content)}")
            console.print(f"[blue]Found {len(markers)} ``` markers[/blue]")
        return content
    else:
        if console:
            console.print(f"[red]Not enough ``` markers: found {len(markers)}[/red]")
        return ""


def display_caption_and_rate(
    *,
    title: str,
    tag_description: str,
    long_description: str,
    pixels: Any,
    rating: List[Any],
    average_score: float,
    panel_height: int,
    console: Console,
) -> None:
    """Display a caption card with rating list and average score."""
    layout = CaptionAndRateLayout(
        tag_description=tag_description,
        rating=rating,
        average_score=average_score,
        long_description=long_description,
        pixels=pixels,
        panel_height=panel_height,
        console=console,
    )
    layout.print(title=title)


def display_pair_image_description(
    *,
    title: str,
    description: str,
    pixels: Any,
    pair_pixels: Any,
    panel_height: int,
    console: Console,
) -> None:
    """Display a two-image layout with a long text description."""
    layout = CaptionPairImageLayout(
        description=description,
        pixels=pixels,
        pair_pixels=pair_pixels,
        panel_height=panel_height,
        console=console,
    )
    layout.print(title=title)


def display_markdown(
    *,
    title: str,
    markdown_content: str,
    pixels: Any,
    panel_height: int,
    console: Console,
) -> None:
    """Display markdown content, optionally with pixels on the side."""
    layout = MarkdownLayout(
        pixels=pixels,
        markdown_content=markdown_content,
        panel_height=panel_height,
        console=console,
    )
    layout.print(title=title)


def process_llm_response(result: str) -> Tuple[str, str]:
    """Extract short/long description split by '###' with cleanup.

    Returns (short_description, long_description)
    """
    if result and "###" in result:
        short_description, long_description = result.split("###")[-2:]
        short_description = " ".join(short_description.split(":", 1)[-1].split())
        long_description = " ".join(long_description.split(":", 1)[-1].split())
    else:
        short_description = ""
        long_description = ""
    return short_description, long_description


def display_caption_layout(
    *,
    title: str,
    tag_description: str,
    short_description: str,
    long_description: str,
    pixels: Any,
    short_highlight_rate: Any,
    long_highlight_rate: Any,
    panel_height: int,
    console: Console,
) -> None:
    """Display a caption layout with short/long sections and highlight rates."""
    layout = CaptionLayout(
        tag_description=tag_description,
        short_description=short_description,
        long_description=long_description,
        pixels=pixels,
        short_highlight_rate=short_highlight_rate,
        long_highlight_rate=long_highlight_rate,
        panel_height=panel_height,
        console=console,
    )
    layout.print(title=title)
