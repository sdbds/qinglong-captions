# -*- coding: utf-8 -*-
"""
Ark (Volcano Engine) provider attempt logic.
Follows the same structure and logging style as other providers (GLM/Qwen/StepFun).
English logs/comments by convention.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Optional

from rich.console import Console
from rich.progress import Progress
from rich.text import Text

from utils.parse_display import extract_code_block_content


def _collect_stream_ark(responses: Iterable[Any], console: Console) -> str:
    """Collect streamed text from Ark responses.

    Tries OpenAI-like delta.content first, falls back to other possible fields.
    """
    chunks: list[str] = []
    for chunk in responses:
        text_piece = ""
        try:
            # OpenAI-like streaming
            if (
                hasattr(chunk, "choices")
                and chunk.choices
                and hasattr(chunk.choices[0], "delta")
                and getattr(chunk.choices[0].delta, "content", None) is not None
            ):
                text_piece = chunk.choices[0].delta.content
            # Sometimes streaming message content can appear fully in message
            elif (
                hasattr(chunk, "choices")
                and chunk.choices
                and hasattr(chunk.choices[0], "message")
                and getattr(chunk.choices[0].message, "content", None)
            ):
                text_piece = chunk.choices[0].message.content  # type: ignore
            else:
                # Generic fallback
                text_piece = getattr(chunk, "text", "") or ""
        except Exception:
            pass

        if text_piece:
            chunks.append(text_piece)
            try:
                console.print(text_piece, end="", overflow="ellipsis")
            except Exception:
                console.print(Text(text_piece), end="", overflow="ellipsis")
            finally:
                console.file.flush()

    console.print("\n")
    return "".join(chunks)


def attempt_ark(
    *,
    client: Any,
    model_path: str,
    messages: list[dict[str, Any]],
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
) -> str:
    """Single-attempt Ark request.

    Returns SRT content. Raises on retryable conditions (RETRY_EMPTY_CONTENT).
    """
    # Lazy import to avoid hard dependency at import-time
    # from volcenginesdkarkruntime import Ark  # client provided by caller

    start_time = time.time()

    completion = client.chat.completions.create(
        model=model_path.replace(".", "-"),
        messages=messages,
        stream=True,
    )

    if progress and task_id is not None:
        progress.update(task_id, description="Generating captions")
    response_text = _collect_stream_ark(completion, console)

    elapsed_time = time.time() - start_time
    console.print(f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds")

    try:
        console.print(response_text)
    except Exception:
        console.print(Text(response_text))

    # Normalize inline color tags to HTML-like for consistency
    response_text = response_text.replace("[green]", "<font color='green'>").replace("[/green]", "</font>")

    content = extract_code_block_content(response_text, "srt", console)
    if not content:
        # Trigger retry when content is empty
        raise Exception("RETRY_EMPTY_CONTENT")

    if progress and task_id is not None:
        progress.update(task_id, description="Processing media...")
    return content
