"""Logics Parsing V3 stateful document OCR, with the legacy V2 HTML adapter."""

from __future__ import annotations

import base64
import html
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from PIL import Image
from rich.console import Console
from rich.progress import Progress
from rich_pixels import Pixels

from module.providers.backends import OpenAIChatRuntime, RuntimeBackendConfig, find_model_config_section
from module.providers.base import CaptionResult, MediaContext, PromptContext
from module.providers.ocr_base import OCRProvider
from module.providers.registry import register_provider
from utils.ocr_model_config import ocr_model_version
from utils.output_writer import write_markdown_output
from utils.parse_display import display_markdown
from utils.stream_util import iter_pdf_pages_high_quality
from utils.transformer_loader import resolve_device_dtype, transformerLoader

from .document import ensure_complete_generation, generation_eos_ids, prepare_document_pages, rebase_generated_images

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


def _attempt_logics_v3(
    *,
    uri: str,
    console: Console,
    model_id: str,
    output_dir: Optional[str],
    prompt_text: str,
    max_new_tokens: int,
    min_pixels: int,
    max_pixels: int,
    window_size: int,
    supports_flex_attn: bool,
    runtime: Optional[RuntimeBackendConfig],
    pixels: Optional[Pixels],
    progress: Optional[Progress],
    task_id: Optional[Any],
) -> str:
    from .logics_v3 import run_document
    from .vendor.logics_v3 import SYSTEM_MESSAGE

    if prompt_text.strip() not in {"", "QwenVL HTML"}:
        raise ValueError("Logics V3 uses fixed stateful document prompts; clear logics_ocr_prompt for V3")
    if max_new_tokens < 1 or window_size < 1:
        raise ValueError("Logics V3 max_new_tokens and window_size must be positive")
    start_time = time.time()
    source = Path(uri)
    destination = Path(output_dir) if output_dir else source.with_suffix("")
    minimum, maximum = int(min_pixels) or 65536, int(max_pixels) or 4194304
    if not 0 < minimum <= maximum:
        raise ValueError("Logics V3 requires 0 < min_pixels <= max_pixels")
    pages = prepare_document_pages(source, destination)
    remote = bool(runtime and runtime.is_openai)
    if remote:
        backend = OpenAIChatRuntime(runtime)
    else:
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        _, dtype, attn_impl = resolve_device_dtype()
        global _TRANS_LOADER
        if _TRANS_LOADER is None:
            _TRANS_LOADER = transformerLoader(
                attn_kw="attn_implementation",
                device_map="auto",
                supports_flex_attn=supports_flex_attn,
            )
        processor = _TRANS_LOADER.get_or_load_processor(model_id, AutoProcessor, console=console)
        model = _TRANS_LOADER.get_or_load_model(
            model_id,
            Qwen3_5ForConditionalGeneration,
            dtype=dtype,
            attn_impl=attn_impl,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_safetensors=True,
            console=console,
        )
        eos_ids = generation_eos_ids(model, processor)

    def infer(window: list[Path], prompt: str) -> str:
        if remote:
            images = [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + base64.b64encode(page.read_bytes()).decode("ascii")},
                }
                for page in window
            ]
        else:
            images = [{"type": "image", "image": str(page)} for page in window]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
            {"role": "user", "content": [*images, {"type": "text", "text": prompt}]},
        ]
        image_kwargs = {"min_pixels": minimum, "max_pixels": maximum}
        if remote:
            return backend.complete(
                messages,
                require_complete=True,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "mm_processor_kwargs": {"images_kwargs": image_kwargs},
                    "skip_special_tokens": False,
                },
            )
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
            images_kwargs=image_kwargs,
        ).to(model.device)
        generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
        if eos_ids:
            generation_kwargs["eos_token_id"] = eos_ids
        output = model.generate(**inputs, **generation_kwargs)
        tokens = output[0, inputs["input_ids"].shape[1] :]
        ensure_complete_generation(tokens, max_new_tokens=max_new_tokens, eos_token_ids=eos_ids)
        if len(tokens) and int(tokens[-1]) in eos_ids:
            tokens = tokens[:-1]
        return processor.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    content = run_document(pages, infer, destination, model_id=model_id, window_size=window_size)
    write_markdown_output(destination, content)
    write_markdown_output(destination, content, filename=f"{source.stem}.md")
    display_markdown(title=source.name, markdown_content=content, pixels=pixels, panel_height=32, console=console)
    if progress and task_id is not None:
        progress.update(task_id, description="Generating captions")
    console.print(f"[blue]Caption generation took:[/blue] {time.time() - start_time:.2f} seconds")
    return content


@torch.inference_mode()
def attempt_logics_ocr(
    *,
    uri: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    model_id: str = "Logics-MLLM/Logics-Parsing-V3",
    model_version: str = "auto",
    prompt_text: str = "QwenVL HTML",
    pixels: Optional[Pixels] = None,
    output_dir: Optional[str] = None,
    supports_flex_attn: bool = False,
    max_new_tokens: int = 16384,
    min_pixels: int = 0,
    max_pixels: int = 0,
    window_size: int = 2,
    runtime: Optional[RuntimeBackendConfig] = None,
) -> str:
    if ocr_model_version("logics", model_id, model_version) == 3:
        return _attempt_logics_v3(
            uri=uri,
            console=console,
            model_id=model_id,
            output_dir=output_dir,
            prompt_text=prompt_text,
            max_new_tokens=max_new_tokens,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            window_size=window_size,
            supports_flex_attn=supports_flex_attn,
            runtime=runtime,
            pixels=pixels,
            progress=progress,
            task_id=task_id,
        )
    from transformers import AutoModelForImageTextToText, AutoProcessor

    min_pixels = int(min_pixels) or 3136
    max_pixels = int(max_pixels) or 7372800
    prompt_text = prompt_text or "QwenVL HTML"

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
            supports_flex_attn=supports_flex_attn,
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
        for rendered_page in iter_pdf_pages_high_quality(str(source_path)):
            page_index = rendered_page.page_number
            pil_img = rendered_page.image
            try:
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
            finally:
                pil_img.close()

        content = "\n<--- Page Split --->\n".join(page_outputs).strip()
        if content:
            final_dir = Path(output_dir)
            final_dir.mkdir(parents=True, exist_ok=True)
            write_markdown_output(final_dir, content, filename=f"{source_path.stem}.md")
    else:
        with Image.open(str(source_path)) as opened:
            pil_img = opened.convert("RGB")
        try:
            raw_output, content = infer_single_image(source_path, pil_img)
            persist_outputs(Path(output_dir), source_path.stem, raw_output, content)
        finally:
            pil_img.close()

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
    """Local and OpenAI-compatible Logics Parsing provider."""

    default_model_id = "Logics-MLLM/Logics-Parsing-V3"
    default_prompt = "QwenVL HTML"

    def get_runtime_backend(self):
        runtime = super().get_runtime_backend()
        section = self.ctx.config.get(self.name, {})
        model_id = section.get("model_id", self.default_model_id)
        if runtime.is_openai and "logics-parsing" in runtime.model_id.lower():
            model_id = runtime.model_id
        version = ocr_model_version("logics", model_id, section.get("model_version", "auto"))
        selected = find_model_config_section(self.ctx.config, runtime.model_id, preferred_sections=(self.name,))
        if (
            version == 3
            and not {"runtime_max_tokens", "max_new_tokens"}.intersection(section)
            and not {"runtime_max_tokens", "max_new_tokens"}.intersection(selected)
            and getattr(self.ctx.args, "local_runtime_max_tokens", None) is None
        ):
            runtime = replace(runtime, max_tokens=16384)
        return runtime

    def get_prompts(self, mime: str):
        prompts = self.ctx.config.get("prompts", {})
        model_id = self._get_model_config("model_id", self.default_model_id)
        runtime = self.get_runtime_backend()
        if runtime.is_openai and "logics-parsing" in runtime.model_id.lower():
            model_id = runtime.model_id
        version = ocr_model_version("logics", model_id, self._get_model_config("model_version", "auto"))
        prompt = str(prompts.get("logics_ocr_prompt", "") or "").strip()
        if version == 3 and prompt in {"", self.default_prompt}:
            return "", ""
        prompt = prompt or self.default_prompt
        return "", prompt

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        runtime = self.get_runtime_backend()
        model_id = self._get_model_config("model_id", self.default_model_id)
        if runtime.is_openai and "logics-parsing" in runtime.model_id.lower():
            model_id = runtime.model_id
        version = ocr_model_version("logics", model_id, self._get_model_config("model_version", "auto"))
        if runtime.is_openai and version == 2:
            return self.attempt_via_openai_backend(media, prompts)

        output_dir = self.artifact_output_dir(media)
        result = attempt_logics_ocr(
            uri=media.uri,
            console=self.ctx.console,
            progress=self.ctx.progress,
            task_id=self.ctx.task_id,
            model_id=model_id,
            model_version=str(version),
            prompt_text=prompts.user,
            pixels=media.pixels,
            output_dir=str(output_dir) if output_dir else None,
            supports_flex_attn=bool(getattr(self, "_supports_flex_attn", False)),
            max_new_tokens=self._get_model_config("max_new_tokens", 16384),
            min_pixels=self._get_model_config("min_pixels", 0),
            max_pixels=self._get_model_config("max_pixels", 0),
            window_size=self._get_model_config("window_size", 2),
            runtime=runtime,
        )
        if version == 3:
            result = rebase_generated_images(result, Path(output_dir), Path(media.uri).parent)
        return CaptionResult(
            raw=result if isinstance(result, str) else str(result),
            metadata={
                "provider": self.name,
                "output_dir": str(output_dir),
                "model_id": model_id,
                "runtime_backend": runtime.mode,
                "runtime_model_id": runtime.model_id,
            },
        )
