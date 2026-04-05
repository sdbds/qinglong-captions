"""Logics Parsing v2 OCR Provider."""

from __future__ import annotations

import html
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from PIL import Image
from rich.console import Console
from rich.progress import Progress
from rich_pixels import Pixels

from module.providers.base import CaptionResult, MediaContext, PromptContext
from module.providers.ocr_base import OCRProvider
from module.providers.registry import register_provider
from utils.parse_display import display_markdown
from utils.output_writer import write_markdown_output
from utils.stream_util import pdf_to_images_high_quality
from utils.transformer_loader import resolve_device_dtype, transformerLoader

_TRANS_LOADER: Optional[transformerLoader] = None
_INLINE_CODE_WRAPPER_RE = re.compile(r"</?(?:pre|code)[^>]*>", re.IGNORECASE)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


def _strip_code_wrappers(content: str) -> str:
    cleaned = content.replace("```", "")
    cleaned = _INLINE_CODE_WRAPPER_RE.sub("", cleaned)
    cleaned = html.unescape(cleaned)
    return cleaned.strip()


def _render_fenced_block(content: str, language: str) -> str:
    body = _strip_code_wrappers(content)
    if not body:
        return ""
    return f"```{language}\n{body}\n```" if language else f"```\n{body}\n```"


def _remove_music_metadata_lines(text: str) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Z:"):
            continue
        kept.append(line.rstrip())
    return "\n".join(kept).strip()


def _render_chart_block(content: str) -> str:
    body = _strip_code_wrappers(content)
    if not body:
        return ""

    body = re.sub(r"^\s*(click|style|linkStyle|stroke|classDef|class)\b.*$", "", body, flags=re.MULTILINE | re.IGNORECASE)
    body = re.sub(r"^\s*(?:%%|::icon).*$", "", body, flags=re.MULTILINE)
    body = body.strip()
    if not body:
        return ""
    if body.startswith("```mermaid"):
        return body
    if body.startswith("mermaid"):
        return f"```{body}\n```"
    return f"```mermaid\n{body}\n```"


def _render_music_block(content: str) -> str:
    body = _remove_music_metadata_lines(_strip_code_wrappers(content))
    if not body:
        return ""
    if body.startswith("```abc"):
        return body
    if body.startswith("abc"):
        return f"```{body}\n```"
    return f"```abc\n{body}\n```"


def _replace_div_block(text: str, class_name: str, renderer: Callable[[str], str]) -> str:
    pattern = re.compile(
        rf'<div[^>]*class="(?:[^"]*\s)?{re.escape(class_name)}(?:\s[^"]*)?"[^>]*>(.*?)</div>',
        re.DOTALL | re.IGNORECASE,
    )

    def replace(match: re.Match[str]) -> str:
        rendered = renderer(match.group(1))
        if not rendered:
            return ""
        return f"\n\n{rendered}\n\n"

    return pattern.sub(replace, text)


def _logics_html_to_markdown(raw_text: str) -> str:
    if not raw_text:
        return ""

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    if "<" not in text or ">" not in text:
        return text.strip()

    text = re.sub(r"<img\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = _replace_div_block(text, "code", lambda content: _render_fenced_block(content, "code"))
    text = _replace_div_block(text, "pseudocode", lambda content: _render_fenced_block(content, ""))
    text = _replace_div_block(text, "chart", _render_chart_block)
    text = _replace_div_block(text, "music", _render_music_block)

    for class_name in ("image", "chemistry", "table", "formula", "image caption", "table caption"):
        text = _replace_div_block(text, class_name, _strip_code_wrappers)

    text = _BR_RE.sub("\n", text)
    text = re.sub(r"</?(?:span|div|p|section|article|main|body|html)[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).replace("\xa0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@torch.inference_mode()
def attempt_logics_ocr(
    *,
    uri: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    model_id: str = "Logics-MLLM/Logics-Parsing-v2",
    prompt_text: str = "QwenVL HTML",
    pixels: Optional[Pixels] = None,
    output_dir: Optional[str] = None,
    max_new_tokens: int = 16384,
    min_pixels: int = 3136,
    max_pixels: int = 7200 * 32 * 32,
) -> str:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    start_time = time.time()
    source_path = Path(uri)
    if not output_dir:
        output_dir = str(source_path.with_suffix(""))

    device, dtype, attn_impl = resolve_device_dtype()
    global _TRANS_LOADER
    if _TRANS_LOADER is None:
        _TRANS_LOADER = transformerLoader(
            attn_kw="attn_implementation",
            device_map="auto",
            supports_flex_attn=bool(getattr(self, "_supports_flex_attn", False)),
        )

    processor = _TRANS_LOADER.get_or_load_processor(model_id, AutoProcessor, console=console)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        if hasattr(image_processor, "min_pixels"):
            image_processor.min_pixels = int(min_pixels)
        if hasattr(image_processor, "max_pixels"):
            image_processor.max_pixels = int(max_pixels)

    model = _TRANS_LOADER.get_or_load_model(
        model_id,
        AutoModelForImageTextToText,
        dtype=dtype,
        attn_impl=attn_impl,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_safetensors=True,
        console=console,
    )
    console.print(f"[blue]Using Logics OCR model:[/blue] {model_id} (device={device}, dtype={dtype})")

    def infer_single_image(image_path: Path, image: Image.Image) -> tuple[str, str]:
        messages = OCRProvider.build_ocr_messages(str(image_path), prompt_text)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )
        try:
            trimmed_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        except Exception:
            trimmed_ids = [generated_ids[0]] if hasattr(generated_ids, "__getitem__") else []

        output_texts = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        raw_output = output_texts[0] if output_texts else ""
        return raw_output, _logics_html_to_markdown(raw_output)

    def persist_outputs(target_dir: Path, stem: str, raw_output: str, markdown_output: str) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / f"{stem}_raw.html").write_text(raw_output, encoding="utf-8")
        write_markdown_output(target_dir, markdown_output, filename=f"{stem}.md")

    if source_path.suffix.lower() == ".pdf":
        page_outputs: list[str] = []
        images = pdf_to_images_high_quality(str(source_path))
        for page_index, pil_img in enumerate(images, start=1):
            page_dir = Path(output_dir) / f"page_{page_index:04d}"
            page_img_path = page_dir / f"page_{page_index:04d}.png"
            page_dir.mkdir(parents=True, exist_ok=True)
            try:
                pil_img.save(page_img_path)
            except Exception:
                try:
                    pil_img.convert("RGB").save(page_img_path)
                except Exception:
                    continue

            raw_page, markdown_page = infer_single_image(page_img_path, pil_img)
            persist_outputs(page_dir, page_img_path.stem, raw_page, markdown_page)
            if markdown_page.strip():
                page_outputs.append(markdown_page.strip())

        content = "\n<--- Page Split --->\n".join(page_outputs).strip()
        if content:
            final_dir = Path(output_dir)
            final_dir.mkdir(parents=True, exist_ok=True)
            write_markdown_output(final_dir, content, filename=f"{source_path.stem}.md")
    else:
        with Image.open(str(source_path)) as opened:
            pil_img = opened.convert("RGB")
        raw_output, content = infer_single_image(source_path, pil_img)
        persist_outputs(Path(output_dir), source_path.stem, raw_output, content)

    if content:
        try:
            display_markdown(
                title=source_path.name,
                markdown_content=content,
                pixels=pixels,
                panel_height=32,
                console=console,
            )
        except Exception:
            pass

    elapsed = time.time() - start_time
    if progress and task_id is not None:
        progress.update(task_id, description="Generating captions")
    console.print(f"[blue]Caption generation took:[/blue] {elapsed:.2f} seconds")
    return content


@register_provider("logics_ocr")
class LogicsOCRProvider(OCRProvider):
    """Local OCR provider for Logics Parsing v2."""

    default_model_id = "Logics-MLLM/Logics-Parsing-v2"
    default_prompt = "QwenVL HTML"

    def get_prompts(self, mime: str):
        prompts = self.ctx.config.get("prompts", {})
        prompt = str(prompts.get("logics_ocr_prompt", "") or "").strip() or self.default_prompt
        return "", prompt

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        if self.get_runtime_backend().is_openai:
            return self.attempt_via_openai_backend(media, prompts)

        output_dir = media.extras.get("output_dir")
        result = attempt_logics_ocr(
            uri=media.uri,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
            model_id=self._get_model_config("model_id", self.default_model_id),
            prompt_text=prompts.user,
            pixels=media.pixels,
            output_dir=str(output_dir) if output_dir else None,
            max_new_tokens=self._get_model_config("max_new_tokens", 16384),
            min_pixels=self._get_model_config("min_pixels", 3136),
            max_pixels=self._get_model_config("max_pixels", 7200 * 32 * 32),
        )
        return CaptionResult(
            raw=result if isinstance(result, str) else str(result),
            metadata={"provider": self.name, "output_dir": str(output_dir)},
        )
