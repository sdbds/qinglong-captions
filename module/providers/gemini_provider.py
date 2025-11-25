# -*- coding: utf-8 -*-
"""
Gemini provider attempt logic extracted for Phase 5.
Keeps behavior and logging identical.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional

from google.genai import types
from rich.console import Console
from rich.progress import Progress
from rich.text import Text
from rich_pixels import Pixels

from utils.parse_display import (
    display_caption_and_rate,
    display_pair_image_description,
    extract_code_block_content,
)


def _save_binary_file(file_name: Any, data: bytes) -> None:
    f = open(file_name, "wb")
    try:
        f.write(data)
    finally:
        f.close()


def _collect_stream_gemini(response: Iterable[Any], uri: str, console: Console) -> str:
    """Collect streamed text and inline_data from Gemini responses.

    - Accumulate chunk.text into final response_text (same as original)
    - For inline_data, save paired text buffer and image/file to disk
    - Preserve printing/flush behaviors
    """
    chunks: List[str] = []
    part_index = 0
    text_buffer: List[str] = []
    for chunk in response:
        if not getattr(chunk, "candidates", None) or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
            continue
        if getattr(chunk, "text", None):
            chunks.append(chunk.text)
            console.print("")
            try:
                console.print(chunk.text, end="", overflow="ellipsis")
            except Exception:
                console.print(Text(chunk.text), end="", overflow="ellipsis")
            finally:
                console.file.flush()
        for part in chunk.candidates[0].content.parts:
            if getattr(part, "text", None):
                text_content = str(part.text)
                if text_content:
                    try:
                        console.print(text_content)
                    except Exception:
                        console.print(Text(text_content), markup=False)
                    finally:
                        console.file.flush()
                    text_buffer.append(text_content)
            if getattr(part, "inline_data", None):
                part_index += 1
                clean_text = "".join(text_buffer).strip()
                if clean_text:
                    text_path = Path(uri).with_name(f"{Path(uri).stem}_{part_index}.txt")
                    _save_binary_file(text_path, clean_text.encode("utf-8"))
                    console.print(f"[blue]Text part saved to: {text_path.name}[/blue]")
                image_path = Path(uri).with_stem(f"{Path(uri).stem}_{part_index}")
                _save_binary_file(image_path, part.inline_data.data)
                console.print(f"[blue]File of mime type {part.inline_data.mime_type} saved to: {image_path.name}[/blue]")
                text_buffer.clear()
    console.print("\n")
    return "".join(chunks)


def attempt_gemini(
    *,
    client: Any,
    model_path: str,
    mime: str,
    prompt: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    uri: str,
    genai_config: Any,
    # One of the following media inputs depending on mime:
    files: Optional[List[Any]] = None,
    audio_bytes: Optional[bytes] = None,
    image_blob: Optional[str] = None,
    pixels: Optional[Pixels] = None,
    # Pair image support
    pair_blob: Optional[str] = None,
    pair_pixels: Optional[Pixels] = None,
    pair_blob_list: Optional[List[str]] = None,
    gemini_task: str = "",
) -> str:
    """Single-attempt Gemini request.

    Returns SRT content for video/audio; for image returns response_text or relevant field.
    May raise RETRY_* exceptions to trigger with_retry.
    """
    console.print("[blue]Generating captions...[/blue]")
    start_time = time.time()

    # Build contents based on mime
    if mime.startswith("video") or (mime.startswith("audio") and files and len(files) > 0):
        response = client.models.generate_content_stream(
            model=model_path,
            contents=[
                types.Part.from_uri(file_uri=files[0].uri, mime_type=mime),
                types.Part.from_text(text=prompt),
            ],
            config=genai_config,
        )
    elif mime.startswith("audio"):
        # allow fallback to reading bytes
        audio_blob = audio_bytes or Path(uri).read_bytes()
        response = client.models.generate_content_stream(
            model=model_path,
            contents=[
                types.Part.from_bytes(data=audio_blob, mime_type=mime),
                types.Part.from_text(text=prompt),
            ],
            config=genai_config,
        )
    elif mime.startswith("image"):
        if pair_blob:
            image_parts: List[Any] = [types.Part.from_bytes(data=pair_blob, mime_type="image/jpeg")]
            if pair_blob_list:
                image_parts.extend([types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in pair_blob_list])
            if image_blob:
                image_parts.append(types.Part.from_bytes(data=image_blob, mime_type="image/jpeg"))
            image_parts.append(types.Part.from_text(text=prompt))
            response = client.models.generate_content_stream(
                model=model_path,
                contents=image_parts,
                config=genai_config,
            )
        else:
            response = client.models.generate_content_stream(
                model=model_path,
                contents=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=image_blob, mime_type="image/jpeg"),
                ],
                config=genai_config,
            )
    else:
        raise Exception("RETRY_UNSUPPORTED_MIME")

    if progress and task_id is not None:
        progress.update(task_id, description="Generating captions")
    response_text = _collect_stream_gemini(response, uri, console)
    if mime.startswith("image"):
        response_text = response_text.replace("*", "").strip()

    elapsed_time = time.time() - start_time
    console.print(f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds")

    try:
        console.print(response_text)
    except Exception:
        console.print(Text(response_text))

    if mime.startswith("image"):
        if isinstance(response_text, str) and not gemini_task:
            try:
                captions = json.loads(response_text)
            except json.JSONDecodeError as e:
                console.print(f"[red]Error decoding JSON: {e}[/red]")
                if "Expecting value: line 1 column 1 (char 0)" in str(e):
                    console.print("[red]Image was filtered, skipping[/red]")
                    return ""
                else:
                    raise e
        else:
            captions = response_text

        if gemini_task:
            display_caption_and_rate(
                title=Path(uri).name,
                tag_description="",
                long_description=response_text,
                pixels=pixels,
                rating=[],
                average_score=0.0,
                panel_height=32,
                console=console,
            )
            return response_text
        elif pair_pixels is not None:
            description = captions.get("prompt", "") if isinstance(captions, dict) else ""
            display_pair_image_description(
                title=Path(uri).name,
                description=description,
                pixels=pixels,
                pair_pixels=pair_pixels,
                panel_height=32,
                console=console,
            )
            return description
        else:
            description = captions.get("description", "") if isinstance(captions, dict) else ""
            scores = captions.get("scores", []) if isinstance(captions, dict) else []
            average_score = captions.get("average_score", 0.0) if isinstance(captions, dict) else 0.0
            display_caption_and_rate(
                title=Path(uri).name,
                tag_description="",
                long_description=description,
                pixels=pixels,
                rating=scores,
                average_score=average_score,
                panel_height=32,
                console=console,
            )
            return response_text

    # Video/Audio: parse SRT
    response_text = response_text.replace("[green]", "<font color='green'>").replace("[/green]", "</font>")
    if "```markdown" in response_text and "```srt" not in response_text:
        response_text = response_text.replace("```markdown", "```srt")
    content = extract_code_block_content(response_text, "srt", console)
    if not content:
        raise Exception("RETRY_EMPTY_CONTENT")
    if progress and task_id is not None:
        progress.update(task_id, description="Processing media...")
    return content
