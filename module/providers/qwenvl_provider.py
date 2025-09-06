# -*- coding: utf-8 -*-
"""
Qwen-VL provider attempt logic extracted for Phase 5.
Keeps behavior and logging identical.
"""
from __future__ import annotations

import time
from typing import Any, Iterable, Optional

from rich.console import Console
from rich.text import Text
from rich.progress import Progress

from utils.parse_display import extract_code_block_content


def _collect_stream_qwen(responses: Iterable[Any], console: Console) -> str:
    """Collect streamed text from QwenVL responses.

    Preserve original behavior: print raw chunk, print the whole aggregated text each step.
    """
    chunks = ""
    for chunk in responses:
        print(chunk)
        try:
            # Original code assumes first element exists
            chunks += chunk.output.choices[0].message.content[0]["text"]
        except Exception:
            # Fallback: try generic text fields if shape differs
            try:
                chunks += getattr(chunk, "text", "") or ""
            except Exception:
                pass
        try:
            console.print(chunks, end="", overflow="ellipsis")
        except Exception:
            console.print(Text(chunks), end="", overflow="ellipsis")
        finally:
            console.file.flush()
    console.print("\n")
    return chunks


def attempt_qwenvl(
    *,
    model_path: str,
    api_key: str,
    messages: list[dict[str, Any]],
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
) -> str:
    """Single-attempt Qwen-VL request.

    Returns SRT content, raises on retryable conditions.
    """
    # Import dashscope lazily to avoid hard dependency at module import
    import dashscope  # type: ignore

    start_time = time.time()

    responses = dashscope.MultiModalConversation.call(
        model=model_path,
        messages=messages,
        api_key=api_key,
        stream=True,
        incremental_output=True,
    )

    if progress and task_id is not None:
        progress.update(task_id, description="Generating captions")
    response_text = _collect_stream_qwen(responses, console)

    elapsed_time = time.time() - start_time
    console.print(f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds")

    try:
        console.print(response_text)
    except Exception:
        console.print(Text(response_text))

    response_text = response_text.replace("[green]", "<font color='green'>").replace("[/green]", "</font>")

    content = extract_code_block_content(response_text, "srt", console)
    if not content:
        # Trigger retry when content is empty
        raise Exception("RETRY_EMPTY_CONTENT")

    if progress and task_id is not None:
        progress.update(task_id, description="Processing media...")
    return content
