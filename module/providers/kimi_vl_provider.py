# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional, Tuple

from rich.console import Console
from rich.progress import Progress
from rich.text import Text
from rich_pixels import Pixels

from utils.parse_display import (
    display_caption_and_rate,
    display_caption_layout,
    display_pair_image_description,
    extract_code_block_content,
    process_llm_response,
)
from utils.stream_util import format_description


def _collect_stream_kimi(completion: Any, console: Console) -> str:
    chunks: list[str] = []
    for chunk in completion:
        try:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content is not None:
                chunks.append(delta.content)
                console.print(".", end="", style="blue")
        except Exception:
            pass
    console.print("\n")
    return "".join(chunks)


def _parse_kimi_response(response_text: str, mode: str = "all") -> Tuple[str, str, str, Any, float]:
    """Extract tag description, short/long text, rating list/dict, and average score.

    Returns (tag_description, short_description, long_description, rating, average_score)
    with graceful fallbacks so UI can render even on unstructured output.
    """

    tag_description = ""
    short_description = ""
    long_description = ""
    rating: Any = []
    average_score: float = 0.0

    # Try JSON first
    try:
        data = json.loads(response_text)
        if isinstance(data, dict):
            # Tags can be list or comma-separated string
            tags_value = data.get("tags")
            if isinstance(tags_value, list):
                tag_description = ", ".join(str(t) for t in tags_value)
            elif isinstance(tags_value, str):
                tag_description = tags_value

            rating = data.get("rating") or data.get("scores") or []
            try:
                average_score = float(data.get("average_score", 0) or data.get("avg_score", 0) or 0)
            except Exception:
                average_score = 0.0

            short_description = data.get("short_description", "") or data.get("short", "")
            long_description = data.get("long_description", "") or data.get("description", "")
    except Exception:
        pass

    # If still empty, try ### split
    if not short_description and not long_description and "###" in response_text:
        short_description, long_description = process_llm_response(response_text)

    # Fallbacks
    if mode != "short":
        if not long_description:
            long_description = response_text.strip()
    if mode != "long":
        if not short_description and "\n" in long_description:
            short_description = long_description.split("\n", 1)[0].strip()

    return tag_description, short_description, long_description, rating, average_score


def _load_tags_from_json(uri: str) -> list[str]:
    """Load pre-generated tags from datasets/tags.json keyed by absolute file path."""
    tags_json_path = Path(__file__).resolve().parents[2] / "datasets" / "tags.json"
    if not tags_json_path.exists():
        return []
    try:
        data = json.loads(tags_json_path.read_text(encoding="utf-8"))
        entry = data.get(str(Path(uri).resolve()))
        if not isinstance(entry, dict):
            return []
        tags = []
        for v in entry.values():
            if isinstance(v, list):
                tags.extend([str(i) for i in v])
        return tags
    except Exception as e:
        console.print(f"[red]Error loading or parsing {tags_json_path}: {e}[/red]")
        return []


def _inject_tags_into_messages(messages: list[dict[str, Any]], tags: list[str]) -> list[dict[str, Any]]:
    """Append tags hint into the text part of user message so model can see them."""
    if not tags:
        return messages
    tag_str = ", ".join(tags)
    for msg in messages:
        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if isinstance(part, dict) and part.get("type") == "text":
                    existing = part.get("text", "")
                    part["text"] = f"{existing}\nExisting tags: {tag_str}".strip()
                    return messages
    return messages


def attempt_kimi_vl(
    *,
    client: Any,
    model_path: str,
    messages: list[dict[str, Any]],
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    uri: str,
    image_pixels: Optional[Pixels] = None,
    pair_pixels: Optional[Pixels] = None,
    thinking: str = "enabled",
    mode: str = "all",
) -> str:
    start_time = time.time()

    # Try to load existing tags from sidecar .txt (align with pixtral behavior)
    captions: list[str] = []
    captions_path = Path(uri).with_suffix(".txt")
    if captions_path.exists():
        try:
            with open(captions_path, "r", encoding="utf-8") as f:
                captions = [line.strip() for line in f.readlines() if line.strip()]
        except Exception:
            pass

    # Try to load tags from datasets/tags.json
    tags_from_json = _load_tags_from_json(uri)

    # Inject tags hint into prompt if available (sidecar or json)
    merged_tags = tags_from_json if tags_from_json else captions
    messages = _inject_tags_into_messages(messages, merged_tags)

    extra_body = {"thinking": {"type": thinking}} if thinking in ("enabled", "disabled") else None
    temperature = 0.6 if thinking == "disabled" else 1.0

    completion = client.chat.completions.create(
        model=model_path,
        messages=messages,
        temperature=temperature,
        top_p=0.95,
        max_tokens=8192,
        response_format={"type": "json_object"},
        stream=False,
        extra_body=extra_body,
    )

    if progress and task_id is not None:
        progress.update(task_id, description="Generating captions")

    # In JSON mode, content is a JSON string
    response_text = completion.choices[0].message.content or ""

    try:
        parsed = json.loads(response_text)
        if isinstance(parsed, dict):
            if mode == "short":
                parsed.pop("long", None)
                parsed.pop("long_description", None)
            elif mode == "long":
                parsed.pop("short", None)
                parsed.pop("short_description", None)
            response_text = json.dumps(parsed, ensure_ascii=False)
    except Exception:
        pass

    elapsed_time = time.time() - start_time
    console.print(f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds")

    try:
        console.print(response_text)
    except Exception:
        console.print(Text(response_text))

    response_text = response_text.replace("[green]", "<font color='green'>").replace("[/green]", "</font>")

    tag_description, short_desc, long_desc, rating, average_score = _parse_kimi_response(response_text, mode=mode)

    # If model didn't return tags but we have sidecar/json tags, use them as tags source
    if not tag_description and merged_tags:
        tag_description = ", ".join(merged_tags)

    if pair_pixels is not None and image_pixels is not None:
        display_pair_image_description(
            title=Path(uri).name,
            description=long_desc,
            pixels=image_pixels,
            pair_pixels=pair_pixels,
            panel_height=32,
            console=console,
        )
        return long_desc

    if image_pixels is not None:
        # Align with pixtral: use CaptionLayout + wdtagger highlight when tags exist
        short_highlight_rate = 0
        long_highlight_rate = 0
        if tag_description:
            short_desc, short_highlight_rate = format_description(short_desc, tag_description)
            long_desc, long_highlight_rate = format_description(long_desc, tag_description)

        display_caption_layout(
            title=Path(uri).name,
            tag_description=tag_description,
            short_description=short_desc,
            long_description=long_desc,
            pixels=image_pixels,
            short_highlight_rate=short_highlight_rate,
            long_highlight_rate=long_highlight_rate,
            panel_height=32,
            console=console,
        )

    return response_text
