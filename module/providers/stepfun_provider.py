# -*- coding: utf-8 -*-
"""
StepFun provider attempt logic extracted from api_handler for Phase 5.
Keeps behavior and logging identical. English logs/comments by convention.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI
from rich.console import Console
from rich.progress import Progress
from rich.text import Text
from rich_pixels import Pixels

from utils.parse_display import (
    display_caption_and_rate,
    display_pair_image_description,
    extract_code_block_content,
)


def _collect_stream_stepfun(completion: Any, console: Console) -> str:
    """Collect streamed text from StepFun(OpenAI-compatible) responses."""
    chunks: list[str] = []
    for chunk in completion:
        if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content is not None:
            chunks.append(chunk.choices[0].delta.content)
            console.print(".", end="", style="blue")
    console.print("\n")
    return "".join(chunks)


def attempt_stepfun(
    *,
    client: OpenAI,
    model_path: str,
    mime: str,
    system_prompt: str,
    prompt: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    uri: str,
    image_blob: Optional[str] = None,
    image_pixels: Optional[Pixels] = None,
    has_pair: bool = False,
    pair_blob: Optional[str] = None,
    pair_pixels: Optional[Pixels] = None,
    video_file_id: Optional[str] = None,
) -> str:
    """Single-attempt StepFun request.

    Returns the SRT content for video, otherwise the response text for image.
    May raise exceptions (e.g., RETRY_EMPTY_CONTENT) to trigger with_retry.
    """
    start_time = time.time()

    if mime.startswith("video"):
        if not video_file_id:
            raise RuntimeError("Missing video_file_id for StepFun video request")
        completion = client.chat.completions.create(
            model=model_path,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": "stepfile://" + video_file_id},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                },
            ],
            temperature=0.7,
            top_p=0.95,
            max_tokens=8192,
            stream=True,
        )
    elif mime.startswith("image"):
        if has_pair:
            completion = client.chat.completions.create(
                model=model_path,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": image_blob,
                            },
                            {
                                "type": "image_url",
                                "image_url": pair_blob,
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    },
                ],
                temperature=0.7,
                top_p=0.95,
                max_tokens=8192,
                stream=True,
            )
        else:
            completion = client.chat.completions.create(
                model=model_path,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": image_blob,
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    },
                ],
                temperature=0.7,
                top_p=0.95,
                max_tokens=8192,
                stream=True,
            )
    else:
        raise RuntimeError(f"Unsupported mime for StepFun: {mime}")

    if progress and task_id is not None:
        progress.update(task_id, description="Generating captions")
    response_text = _collect_stream_stepfun(completion, console)

    elapsed_time = time.time() - start_time
    console.print(f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds")

    try:
        console.print(response_text)
    except Exception:
        console.print(Text(response_text))

    response_text = response_text.replace("[green]", "<font color='green'>").replace("[/green]", "</font>")

    if mime.startswith("video"):
        content = extract_code_block_content(response_text, "srt", console)
        if not content:
            raise Exception("RETRY_EMPTY_CONTENT")
        if progress and task_id is not None:
            progress.update(task_id, description="Processing media...")
        return content

    # image branch
    if has_pair and pair_pixels is not None and image_pixels is not None:
        display_pair_image_description(
            title=Path(uri).name,
            description=response_text,
            pixels=image_pixels,
            pair_pixels=pair_pixels,
            panel_height=32,
            console=console,
        )
        return response_text
    else:
        display_caption_and_rate(
            title=Path(uri).name,
            tag_description="",
            long_description=response_text,
            pixels=image_pixels,
            rating=[],
            average_score=0,
            panel_height=32,
            console=console,
        )
        return response_text
