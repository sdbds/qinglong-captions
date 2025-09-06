# -*- coding: utf-8 -*-
"""
GLM provider attempt logic extracted for Phase 5.
Keeps behavior and logging identical.
"""
from __future__ import annotations

import time
from typing import Any, Iterable, Optional

from rich.console import Console
from rich.text import Text
from rich.progress import Progress

from utils.parse_display import extract_code_block_content


def _collect_stream_glm(responses: Iterable[Any], console: Console) -> str:
    """Collect streamed text from GLM responses.

    Preserve original behavior: print raw chunk, print the whole aggregated text each step.
    """
    chunks = ""
    for chunk in responses:
        print(chunk)
        if (
            hasattr(chunk.choices[0].delta, "content")
            and chunk.choices[0].delta.content is not None
        ):
            chunks += chunk.choices[0].delta.content
        try:
            console.print(chunks, end="", overflow="ellipsis")
        except Exception:
            console.print(Text(chunks), end="", overflow="ellipsis")
        finally:
            console.file.flush()
    console.print("\n")
    return chunks


def attempt_glm(
    *,
    client: Any,
    model_path: str,
    messages: list[dict[str, Any]],
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
) -> str:
    """Single-attempt GLM request.

    Returns SRT content, raises on retryable conditions.
    """
    start_time = time.time()

    responses = client.chat.completions.create(
        model=model_path,
        messages=messages,
        stream=True,
    )

    if progress and task_id is not None:
        progress.update(task_id, description="Generating captions")
    response_text = _collect_stream_glm(responses, console)

    elapsed_time = time.time() - start_time
    console.print(f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds")

    try:
        console.print(response_text)
    except Exception:
        console.print(Text(response_text))

    response_text = response_text.replace("[green]", "<font color='green'>").replace("[/green]", "</font>")

    content = extract_code_block_content(response_text, "srt", console)
    if not content:
        raise Exception("RETRY_EMPTY_CONTENT")

    if progress and task_id is not None:
        progress.update(task_id, description="Processing media...")
    return content
