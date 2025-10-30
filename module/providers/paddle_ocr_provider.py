from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from paddleocr import PaddleOCRVL
from rich.console import Console
from rich.progress import Progress
from rich_pixels import Pixels

from utils.parse_display import display_markdown


def _run_pipeline(pipeline, image_path: str, out_dir: Path) -> None:
    output = pipeline.predict(image_path)
    for res in output:
        try:
            res.print()
        except Exception:
            pass
        try:
            res.save_to_json(save_path=str(out_dir))
        except Exception:
            pass
        try:
            res.save_to_markdown(save_path=str(out_dir))
        except Exception:
            pass


def attempt_paddle_ocr(
    *,
    uri: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    pixels: Optional[Pixels] = None,
    output_dir: Optional[str] = None,
) -> str:
    start = time.time()

    image_path = str(Path(uri))
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    pipeline = PaddleOCRVL()
    out_dir = Path(output_dir) if output_dir else Path(uri).with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_pipeline(pipeline, image_path, out_dir)

    md_files = sorted(out_dir.glob("*.md"))
    content_parts = []
    for md in md_files:
        try:
            content_parts.append(md.read_text(encoding="utf-8"))
        except Exception:
            continue
    content = "\n\n".join(content_parts).strip()

    try:
        display_markdown(
            title=Path(uri).name,
            markdown_content=content,
            pixels=pixels,
            panel_height=32,
            console=console,
        )
    except Exception:
        pass

    elapsed = time.time() - start
    if progress and task_id is not None:
        progress.update(task_id, description="Generating captions")
    console.print(f"[blue]Caption generation took:[/blue] {elapsed:.2f} seconds")

    if content and output_dir:
        try:
            (Path(output_dir) / "result.md").write_text(content, encoding="utf-8")
        except Exception:
            pass

    return content
