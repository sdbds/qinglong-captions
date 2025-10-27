# -*- coding: utf-8 -*-
"""
Local Moondream VLM provider.
- Mirrors pixtral image flows for caption display and OCR markdown display.
- Keeps logs and UI consistent with existing providers.
All logs and comments are in English.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from rich.console import Console
from rich.text import Text
from rich.progress import Progress

from utils.parse_display import (
    display_markdown,
    display_caption_layout,
)
from utils.stream_util import format_description
from utils.transformer_loader import transformerLoader, resolve_device_dtype


_TRANS_LOADER: Optional[transformerLoader] = None


def _ensure_loader(device_map: Dict[str, str]) -> transformerLoader:
    global _TRANS_LOADER
    if _TRANS_LOADER is None:
        _TRANS_LOADER = transformerLoader(attn_kw=None, device_map=device_map)
    return _TRANS_LOADER


def attempt_moondream(
    *,
    model_id: str,
    mime: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    # Common
    uri: str,
    # Image inputs
    pixels: Any = None,
    image: Optional[Image.Image] = None,
    captions: Optional[List[str]] = None,
    character_name: str = "",
    tags_highlightrate: float = 0.0,
    # General query prompt
    prompt_text: Optional[str] = None,
    reasoning: bool = False,
    # Image OCR
    ocr: bool = False,
    base64_image: Optional[str] = None,
    task: str = "caption",
) -> Any:
    """Single-attempt Moondream request.

    Args:
        task: Task combinations (comma-separated) - "caption", "query", "point", "detect", or "all"
              - caption: Generate image descriptions
              - query: Ask open-ended questions about images
              - point: Identify specific points (x, y coordinates) for objects
              - detect: Provide bounding boxes for objects
              - all: Execute all available tasks

    Returns:
      - Combined results from all requested tasks
      - For OCR: str markdown content (highest priority)
    """
    start_time = time.time()

    if mime.startswith("application"):
        console.print(
            Text("Moondream does not support PDF/document OCR.", style="yellow")
        )
        return ""

    # Expect preprocessed PIL image from api_handler
    if image is None:
        console.print(
            Text("Moondream requires an image object from api_handler", style="red")
        )
        return ""

    device, dtype, _ = resolve_device_dtype()
    device_map = {"": device}
    loader = _ensure_loader(device_map)

    if console:
        console.print(f"[blue]Loading Moondream model:[/blue] {model_id}")
    model = loader.get_or_load_model(
        model_id,
        AutoModelForCausalLM,
        dtype=dtype,
        trust_remote_code=True,
        device_map=device_map,
        console=console,
    )
    try:
        model.compile()
        if console:
            console.print("[blue]Moondream compile() finished[/blue]")
    except Exception as e:
        if console:
            console.print(Text(f"Moondream compile() failed: {e}", style="yellow"))

    # Parse task combinations
    task_list = []
    if task == "all":
        task_list = ["caption", "query", "point", "detect"]
    else:
        task_list = [t.strip() for t in task.split(",") if t.strip()]

    # OCR has highest priority and overrides other tasks
    if ocr:
        question = (
            "Convert all visible text in the image to clean Markdown. "
            "Preserve headings, lists, tables, and structure where possible."
        )
        try:
            result = model.query(image=image, question=question, reasoning=False)
            content = result.get("answer", "")
        except Exception as e:
            console.print(Text(f"Moondream OCR failed: {e}", style="yellow"))
            raise Exception("RETRY_MOONDREAM_OCR")

        display_markdown(
            title=Path(uri).name,
            markdown_content=content,
            pixels=pixels,
            panel_height=32,
            console=console,
        )
        elapsed_time = time.time() - start_time
        console.print(f"[blue]OCR processing took:[/blue] {elapsed_time:.2f} seconds")
        return content

    # Execute requested tasks and combine results
    results = []

    # Caption task
    if "caption" in task_list:
        try:
            short = model.caption(image, length="short")
            long = model.caption(image, length="long")
            short_description = short.get("caption", "")
            long_description = long.get("caption", "")

            captions = captions or []
            tag_description = captions[0] if len(captions) > 0 else ""

            if len(captions) > 0:
                short_description, short_highlight_rate = format_description(
                    short_description, tag_description
                )
                long_description, long_highlight_rate = format_description(
                    long_description, tag_description
                )
            else:
                short_highlight_rate = 0
                long_highlight_rate = 0

            caption_result = f"### Short Description\n{short_description}\n\n### Long Description\n{long_description}"
            # Store highlight rates as metadata in the result
            caption_metadata = {
                "short_highlight_rate": short_highlight_rate,
                "long_highlight_rate": long_highlight_rate,
            }
            results.append(("Caption", caption_result, caption_metadata))

        except Exception as e:
            console.print(Text(f"Moondream caption failed: {e}", style="yellow"))
            results.append(("Caption", f"Error: {e}", {}))

    # Query task
    if "query" in task_list:
        if (prompt_text is not None) and (str(prompt_text).strip() != ""):
            try:
                result = model.query(
                    image=image, question=str(prompt_text), reasoning=bool(reasoning)
                )
                query_result = result.get("answer", "")
                results.append(("Query", query_result, {}))
            except Exception as e:
                console.print(Text(f"Moondream query failed: {e}", style="yellow"))
                results.append(("Query", f"Error: {e}", {}))
        else:
            results.append(("Query", "No prompt provided for query task", {}))

    # Point task
    if "point" in task_list:
        if (prompt_text is not None) and (str(prompt_text).strip() != ""):
            try:
                result = model.point(image, str(prompt_text))
                points = result.get("points", [])
                point_result = f"Found {len(points)} point(s) for '{prompt_text}':\n"
                for i, point in enumerate(points):
                    point_result += (
                        f"Point {i+1}: x={point['x']:.3f}, y={point['y']:.3f}\n"
                    )
                results.append(("Point", point_result, {}))
            except Exception as e:
                console.print(Text(f"Moondream point failed: {e}", style="yellow"))
                results.append(("Point", f"Error: {e}", {}))
        else:
            results.append(("Point", "No prompt provided for point task", {}))

    # Detect task
    if "detect" in task_list:
        if (prompt_text is not None) and (str(prompt_text).strip() != ""):
            try:
                result = model.detect(image, str(prompt_text))
                objects = result.get("objects", [])
                detect_result = f"Found {len(objects)} object(s) for '{prompt_text}':\n"
                for i, obj in enumerate(objects):
                    detect_result += (
                        f"Object {i+1}: "
                        f"x_min={obj['x_min']:.3f}, y_min={obj['y_min']:.3f}, "
                        f"x_max={obj['x_max']:.3f}, y_max={obj['y_max']:.3f}\n"
                    )
                results.append(("Detect", detect_result, {}))
            except Exception as e:
                console.print(Text(f"Moondream detect failed: {e}", style="yellow"))
                results.append(("Detect", f"Error: {e}", {}))
        else:
            results.append(("Detect", "No prompt provided for detect task", {}))

    # Combine all results and display appropriately
    if results:
        # Check if caption task is included
        has_caption = any(task_name == "Caption" for task_name, _, _ in results)

        if has_caption and len(results) == 1:
            # Only caption task - use dedicated caption display
            caption_item = next(item for item in results if item[0] == "Caption")
            caption_result = caption_item[1]
            caption_metadata = caption_item[2]

            # Parse caption result to extract short and long descriptions
            lines = caption_result.split("\n")
            short_desc = ""
            long_desc = ""

            current_section = None
            for line in lines:
                if line.startswith("### Short Description"):
                    current_section = "short"
                elif line.startswith("### Long Description"):
                    current_section = "long"
                elif line.startswith("###") or line.strip() == "":
                    continue
                elif current_section == "short":
                    short_desc += line + "\n"
                elif current_section == "long":
                    long_desc += line + "\n"

            display_caption_layout(
                title=Path(uri).name,
                short_description=short_desc.strip(),
                long_description=long_desc.strip(),
                short_highlight_rate=caption_metadata.get("short_highlight_rate", 0),
                long_highlight_rate=caption_metadata.get("long_highlight_rate", 0),
                pixels=pixels,
                console=console,
            )

            elapsed_time = time.time() - start_time
            console.print(
                f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds"
            )
            return caption_result

        elif has_caption:
            # Caption + other tasks - show caption with dedicated layout, then other results in markdown
            caption_item = next(item for item in results if item[0] == "Caption")
            caption_result = caption_item[1]
            caption_metadata = caption_item[2]
            other_results = [
                (task_name, result, metadata)
                for task_name, result, metadata in results
                if task_name != "Caption"
            ]

            # Parse and display caption
            lines = caption_result.split("\n")
            short_desc = ""
            long_desc = ""

            current_section = None
            for line in lines:
                if line.startswith("### Short Description"):
                    current_section = "short"
                elif line.startswith("### Long Description"):
                    current_section = "long"
                elif line.startswith("###") or line.strip() == "":
                    continue
                elif current_section == "short":
                    short_desc += line + "\n"
                elif current_section == "long":
                    long_desc += line + "\n"

            display_caption_layout(
                title=Path(uri).name,
                short_description=short_desc.strip(),
                long_description=long_desc.strip(),
                short_highlight_rate=caption_metadata.get("short_highlight_rate", 0),
                long_highlight_rate=caption_metadata.get("long_highlight_rate", 0),
                pixels=pixels,
                console=console,
            )

            # Display other results in markdown
            if other_results:
                other_content = ""
                for task_name, result, _ in other_results:
                    other_content += f"## {task_name} Result\n{result}\n\n"

                console.print("\n")  # Add spacing before other results
                display_markdown(
                    title=f"{Path(uri).name} - Additional Results",
                    markdown_content=other_content.strip(),
                    pixels=pixels,
                    panel_height=32,
                    console=console,
                )

            elapsed_time = time.time() - start_time
            console.print(
                f"[blue]Combined tasks processing took:[/blue] {elapsed_time:.2f} seconds"
            )

            # Return combined content
            combined_content = (
                caption_result + "\n\n" + other_content.strip()
                if other_results
                else caption_result
            )
            return combined_content.strip()

        else:
            # No caption task - use regular markdown display for all results
            combined_content = ""
            for task_name, result, _ in results:
                combined_content += f"## {task_name} Result\n{result}\n\n"

            display_markdown(
                title=Path(uri).name,
                markdown_content=combined_content.strip(),
                pixels=pixels,
                panel_height=48,  # Larger panel for combined results
                console=console,
            )
            elapsed_time = time.time() - start_time
            console.print(
                f"[blue]Combined tasks processing took:[/blue] {elapsed_time:.2f} seconds"
            )
            return combined_content.strip()
    else:
        return "No valid tasks to execute"
