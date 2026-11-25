"""Moondream Provider"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from contextlib import closing
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

from PIL import Image
from rich.console import Console
from rich.progress import Progress
from rich.text import Text

from module.providers.base import CaptionResult, MediaContext, PromptContext
from module.providers.local_vlm_base import LocalVLMProvider
from module.providers.registry import register_provider
from utils.console_util import print_exception
from utils.ocr_model_config import MOONDREAM_PHOTON_MODEL, MOONDREAM_PHOTON_REPO, is_moondream_photon, moondream_needs_spatial_tasks
from utils.parse_display import (
    display_caption_layout,
    display_markdown,
)
from utils.stream_util import format_description

_OCR_PROMPT = (
    "Transcribe only the text visible in this image, in reading order, without translating it. "
    "Use Markdown to preserve headings, lists, and tables where present. "
    "Return only the transcription. Do not describe the image or add commentary."
)


def _load_photon_model():
    try:
        from moondream import photon
    except ImportError as exc:
        raise ImportError(
            "Moondream 3.1 requires the Photon SDK. Use the caption launcher or install "
            "`--extra moondream` in the caption runtime."
        ) from exc
    if sys.platform != "win32":
        return photon(MOONDREAM_PHOTON_MODEL)

    import torch

    if not (torch.version.cuda or "").startswith("12."):
        raise RuntimeError(
            "Moondream Photon on Windows requires a CUDA 12 build of PyTorch. "
            "Use 4.captioner.ps1 or the GUI to launch the isolated cu128 runtime."
        )

    from kestrel.models.moondream import weights

    # Mapping the 10.5 GB checkpoint can crash inside PyTorch on Windows. Kestrel
    # 0.9.1 has no reader option; scope pread to its own synchronous startup.
    reader = weights.safetensors
    weights.safetensors = SimpleNamespace(safe_open=partial(reader.safe_open, backend="pread"))
    try:
        return photon(MOONDREAM_PHOTON_MODEL)
    finally:
        weights.safetensors = reader


def attempt_moondream(
    *,
    get_model: Callable[[], Any],
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
        console.print(Text("Moondream does not support PDF/document OCR.", style="yellow"))
        return ""

    # Expect preprocessed PIL image from api_handler
    if image is None:
        console.print(Text("Moondream requires an image object from api_handler", style="red"))
        return ""

    model = get_model()

    # Parse task combinations
    task_list = []
    if task == "all":
        task_list = ["caption", "query", "point", "detect"]
    else:
        task_list = [t.strip() for t in task.split(",") if t.strip()]

    # OCR has highest priority and overrides other tasks
    if ocr:
        try:
            result = model.query(image=image, question=_OCR_PROMPT, reasoning=False)
            content = result.get("answer", "")
        except Exception as e:
            print_exception(console, e, prefix="Moondream OCR failed", summary_style="yellow")
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
    errors = []

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
                short_description, short_highlight_rate = format_description(short_description, tag_description)
                long_description, long_highlight_rate = format_description(long_description, tag_description)
            else:
                short_highlight_rate = 0
                long_highlight_rate = 0

            caption_result = f"### Short Description\n{short_description}\n\n### Long Description\n{long_description}"
            # Store highlight rates and tag_description as metadata in the result
            caption_metadata = {
                "tag_description": tag_description,
                "short_highlight_rate": short_highlight_rate,
                "long_highlight_rate": long_highlight_rate,
            }
            results.append(("Caption", caption_result, caption_metadata))

        except Exception as e:
            print_exception(console, e, prefix="Moondream caption failed", summary_style="yellow")
            errors.append(f"Caption: {e}")

    # Query task
    if "query" in task_list:
        if (prompt_text is not None) and (str(prompt_text).strip() != ""):
            try:
                result = model.query(image=image, question=str(prompt_text), reasoning=bool(reasoning))
                query_result = result.get("answer", "")
                results.append(("Query", query_result, {}))
            except Exception as e:
                print_exception(console, e, prefix="Moondream query failed", summary_style="yellow")
                errors.append(f"Query: {e}")
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
                    point_result += f"Point {i + 1}: x={point['x']:.3f}, y={point['y']:.3f}\n"
                results.append(("Point", point_result, {}))
            except Exception as e:
                print_exception(console, e, prefix="Moondream point failed", summary_style="yellow")
                errors.append(f"Point: {e}")
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
                        f"Object {i + 1}: "
                        f"x_min={obj['x_min']:.3f}, y_min={obj['y_min']:.3f}, "
                        f"x_max={obj['x_max']:.3f}, y_max={obj['y_max']:.3f}\n"
                    )
                results.append(("Detect", detect_result, {}))
            except Exception as e:
                print_exception(console, e, prefix="Moondream detect failed", summary_style="yellow")
                errors.append(f"Detect: {e}")
        else:
            results.append(("Detect", "No prompt provided for detect task", {}))

    if not results and errors:
        raise RuntimeError("Moondream task(s) failed: " + "; ".join(errors))

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
                tag_description=caption_metadata.get("tag_description", ""),
                short_description=short_desc.strip(),
                long_description=long_desc.strip(),
                short_highlight_rate=caption_metadata.get("short_highlight_rate", 0),
                long_highlight_rate=caption_metadata.get("long_highlight_rate", 0),
                pixels=pixels,
                panel_height=32,
                console=console,
            )

            elapsed_time = time.time() - start_time
            console.print(f"[blue]Caption generation took:[/blue] {elapsed_time:.2f} seconds")
            return caption_result

        elif has_caption:
            # Caption + other tasks - show caption with dedicated layout, then other results in markdown
            caption_item = next(item for item in results if item[0] == "Caption")
            caption_result = caption_item[1]
            caption_metadata = caption_item[2]
            other_results = [(task_name, result, metadata) for task_name, result, metadata in results if task_name != "Caption"]

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
                tag_description=caption_metadata.get("tag_description", ""),
                short_description=short_desc.strip(),
                long_description=long_desc.strip(),
                short_highlight_rate=caption_metadata.get("short_highlight_rate", 0),
                long_highlight_rate=caption_metadata.get("long_highlight_rate", 0),
                pixels=pixels,
                panel_height=32,
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
            console.print(f"[blue]Combined tasks processing took:[/blue] {elapsed_time:.2f} seconds")

            # Return combined content
            combined_content = caption_result + "\n\n" + other_content.strip() if other_results else caption_result
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
            console.print(f"[blue]Combined tasks processing took:[/blue] {elapsed_time:.2f} seconds")
            return combined_content.strip()
    else:
        return "No valid tasks to execute"


# ---------------------------------------------------------------------------
# V2 Provider class
# ---------------------------------------------------------------------------

@register_provider("moondream")
class MoondreamProvider(LocalVLMProvider):
    """Moondream Local VLM Provider"""

    default_model_id = MOONDREAM_PHOTON_REPO
    _attn_implementation = "eager"

    def _load_model(self):
        model_id = self.model_id
        if is_moondream_photon(model_id):
            self.log(f"Loading Moondream with local Photon: {model_id}", "blue")
            return _load_photon_model()

        from transformers import AutoModelForCausalLM

        from utils.transformer_loader import resolve_device_dtype, transformerLoader

        device, dtype, _ = resolve_device_dtype()
        loader = transformerLoader(attn_kw=None, device_map={"": device})
        model = loader.load_model(model_id, AutoModelForCausalLM, dtype=dtype, console=self.ctx.console)
        try:
            model.compile()
        except Exception as exc:
            print_exception(self.ctx.console, exc, prefix="Moondream compile() failed", summary_style="yellow")
        return model

    @classmethod
    def can_handle(cls, args: Any, mime: str) -> bool:
        vlm_model = getattr(args, "vlm_image_model", "")
        ocr_model = getattr(args, "ocr_model", "")
        # Moondream 可以作为 VLM 或 OCR 使用
        return (vlm_model == "moondream" or ocr_model == "moondream") and mime.startswith("image")

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        # 读取配置
        vlm_section = self.model_config
        reasoning = vlm_section.get("reasoning")
        ocr_mode = getattr(self.ctx.args, "ocr_model", "") == "moondream"
        tasks = vlm_section.get("tasks", "caption")

        runtime = self.get_runtime_backend()
        if runtime.is_openai:
            if moondream_needs_spatial_tasks(str(tasks), image_ocr=ocr_mode):
                self.log("Moondream point/detect tasks require direct backend; falling back to direct runtime.", "yellow")
            else:
                user_prompt = _OCR_PROMPT if ocr_mode else prompts.user
                return self.attempt_via_openai_backend(media, prompts, user_prompt=user_prompt)

        # 读取 captions
        captions: List[str] = []
        captions_path = Path(media.uri).with_suffix(".txt")
        if captions_path.exists():
            with open(captions_path, "r", encoding="utf-8") as f:
                captions = [line.strip() for line in f.readlines()]

        with Image.open(media.uri) as source_image, closing(source_image.convert("RGB")) as image:
            result = attempt_moondream(
                get_model=self._get_or_load_model,
                mime=media.mime,
                console=self.ctx.console,
                progress=self.ctx.progress,
                task_id=self.ctx.task_id,
                uri=media.uri,
                pixels=media.pixels,
                image=image,
                captions=captions,
                tags_highlightrate=getattr(self.ctx.args, "tags_highlightrate", 0.0),
                prompt_text=prompts.user,
                reasoning=bool(reasoning) if reasoning is not None else False,
                ocr=ocr_mode,
                task=tasks,
            )

        return CaptionResult(raw=result, metadata={"provider": self.name})

    def get_retry_config(self):
        cfg = super().get_retry_config()

        def classify(e):
            msg = str(e)
            if "RETRY_MOONDREAM_" in msg:
                return cfg.base_wait
            return None

        cfg.classify_error = classify
        return cfg
