# -*- coding: utf-8 -*-
"""
Pixtral (Mistral) provider attempt logic extracted for Phase 5.
Keeps behavior and logging identical to the original branch.
"""

from __future__ import annotations

import base64
import io
import re
import time
from pathlib import Path
from typing import Any, List, Optional

from PIL import Image
from rich.console import Console
from rich.progress import Progress
from rich.text import Text
from rich_pixels import Pixels

from utils.parse_display import (
    display_caption_layout,
    display_markdown,
    process_llm_response,
)
from utils.stream_util import format_description


def attempt_pixtral(
    *,
    client: Any,
    model_path: str,
    mime: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    # Common
    uri: str,
    # Image chat path
    messages: Optional[List[dict]] = None,
    pixels: Optional[Pixels] = None,
    captions: Optional[List[str]] = None,
    prompt_text: Optional[str] = None,
    character_name: str = "",
    tags_highlightrate: float = 0.0,
    # Image OCR path
    ocr: bool = False,
    base64_image: Optional[str] = None,
    # Document (application) OCR path
    document_image: bool = False,
    signed_url_url: Optional[str] = None,
) -> Any:
    """Single-attempt Pixtral request.

    Returns:
      - For image chat: str content (model response)
      - For image OCR: str markdown
      - For application OCR: list of pages (each with markdown/optional image), and side-effect displays
    May raise exceptions like RETRY_PIXTRAL_* to trigger with_retry.
    """
    start_time = time.time()

    if mime.startswith("application"):
        # PDF OCR path
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url_url,
            },
            include_image_base64=document_image,
        )
        pages = ocr_response.pages
        console.print(f"[bold cyan]PDF共有 {len(pages)} 页[/bold cyan]")

        for page in pages:
            # Try preview first image on the page
            ocr_pixels = None
            if page.images and len(page.images) > 0:
                first_image = page.images[0]
                if hasattr(first_image, "image_base64") and first_image.image_base64:
                    try:
                        base64_str = first_image.image_base64
                        if base64_str.startswith("data:"):
                            base64_content = base64_str.split(",", 1)[1]
                            image_data = base64.b64decode(base64_content)
                        else:
                            image_data = base64.b64decode(base64_str)
                        ocr_image = Image.open(io.BytesIO(image_data))
                        ocr_pixels = Pixels.from_image(
                            ocr_image,
                            resize=(ocr_image.width // 18, ocr_image.height // 18),
                        )
                    except Exception as e:
                        console.print(f"[yellow]Error loading image: {e}[/yellow]")
                        ocr_pixels = None
                else:
                    console.print("[yellow]Image found but no base64 data available[/yellow]")
            display_markdown(
                title=f"{Path(uri).name} - Page {page.index + 1}",
                markdown_content=page.markdown,
                pixels=ocr_pixels,
                panel_height=32,
                console=console,
            )

        elapsed_time = time.time() - start_time
        if progress and task_id is not None:
            progress.update(task_id, description="Generating captions")
        if elapsed_time < 0:
            elapsed_time = 0
        console.print(f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds")
        # Preserve original behavior: side-effect display, return empty string
        return ""

    if ocr:
        # Image OCR
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_image}",
            },
        )
        content = ocr_response.pages[0].markdown
        display_markdown(
            title=Path(uri).name,
            markdown_content=content,
            pixels=pixels,
            panel_height=32,
            console=console,
        )
    else:
        # Image chat
        chat_response = client.chat.complete(model=model_path, messages=messages)
        content = chat_response.choices[0].message.content

        short_description, long_description = process_llm_response(content)
        captions = captions or []
        if len(captions) > 0:
            tag_description = (
                (
                    prompt_text.rsplit("<s>[INST]", 1)[-1]
                    .rsplit("<|end_of_focus|>", 1)[-1]  # fallback safe split no-op
                    .rsplit(">.", 1)[-1]
                    .rsplit(").", 1)[-1]
                    .replace(" from", ",")
                )
                .rsplit("[IMG][/INST]", 1)[0]
                .strip()
            )
            short_description, short_highlight_rate = format_description(short_description, tag_description)
            long_description, long_highlight_rate = format_description(long_description, tag_description)
        else:
            tag_description = ""
            short_highlight_rate = 0
            long_highlight_rate = 0

        # Display
        display_caption_layout(
            title=Path(uri).name,
            tag_description=tag_description,
            short_description=short_description,
            long_description=long_description,
            pixels=pixels,
            short_highlight_rate=short_highlight_rate,
            long_highlight_rate=long_highlight_rate,
            panel_height=32,
            console=console,
        )

        # Validations (non OCR only)
        if character_name:
            clean_char_name = character_name.split(",")[0].split(" from ")[0].strip("<>")
            if clean_char_name not in content:
                # Check case-insensitive match and fix if found
                pattern = re.compile(re.escape(clean_char_name), re.IGNORECASE)
                if pattern.search(content):
                    content = pattern.sub(clean_char_name, content)
                    console.print(f"[yellow]Fixed character name case: [green]{clean_char_name}[/green][/yellow]")
                else:
                    console.print()
                    console.print(Text(content))
                    console.print(f"Character name [green]{clean_char_name}[/green] not found")
                    raise Exception("RETRY_PIXTRAL_CHAR")

        if "###" not in content:
            console.print(Text(content))
            console.print(Text("No ###, retrying...", style="yellow"))
            raise Exception("RETRY_PIXTRAL_NO_MARK")

        # Tags highlight-rate threshold
        if (
            any(f"{i}women" in tag_description for i in range(2, 5))
            or ("1man" in tag_description and "1woman" in tag_description)
            or "multiple girls" in tag_description
            or "multiple boys" in tag_description
        ):
            threshold = tags_highlightrate * 100 / 2
        else:
            threshold = tags_highlightrate * 100
        if int(re.search(r"\d+", str(long_highlight_rate)).group()) < threshold and len(captions) > 0:
            console.print(f"[red]long_description highlight rate is too low: {long_highlight_rate}%, retrying...[/red]")
            raise Exception("RETRY_PIXTRAL_RATE")

        if isinstance(content, str) and "502" in content:
            console.print("[yellow]Received 502 error[/yellow]")
            raise Exception("RETRY_PIXTRAL_502")

    elapsed_time = time.time() - start_time
    console.print(f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds")
    return content
