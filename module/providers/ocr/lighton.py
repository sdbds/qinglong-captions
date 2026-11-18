"""LightOn OCR Provider"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from rich.console import Console
from rich.progress import Progress
from rich_pixels import Pixels

from module.providers.backends import OpenAIChatRuntime, RuntimeBackendConfig, find_model_config_section
from module.providers.base import CaptionResult, MediaContext, PromptContext, Provider
from module.providers.ocr_base import OCRProvider
from module.providers.registry import register_provider
from utils.ocr_model_config import ocr_model_version
from utils.output_writer import write_markdown_output
from utils.parse_display import display_markdown
from utils.transformer_loader import resolve_device_dtype, transformerLoader

from .document import ensure_complete_generation, generation_eos_ids, page_metadata, prepare_document_pages, write_layout_json

_TRANS_LOADER: Optional[transformerLoader] = None

_GROUNDING_RE = re.compile(r"!\[([a-zA-Z_][\w-]*?\+?)\]\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


def parse_grounding(raw: str) -> tuple[str, list[dict]]:
    for marker in re.finditer(r"!\[[A-Za-z_][\w+-]*\]\(([^)\n]*,[^)\n]*)\)", raw):
        if marker[1].count(",") >= 2 and not _GROUNDING_RE.fullmatch(marker[0]):
            raise ValueError(f"Malformed LightOn grounding marker: {marker[0]}")
    matches = list(_GROUNDING_RE.finditer(raw))
    if not matches:
        if not raw.strip():
            return "", []
        raise ValueError("LightOn grounding output contains no valid layout markers")
    if raw[: matches[0].start()].strip():
        raise ValueError("Unexpected text before LightOn grounding layout markers")
    blocks = []
    for index, match in enumerate(matches):
        bbox = [int(match[group]) for group in range(2, 6)]
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise ValueError(f"Invalid LightOn grounding bbox: {bbox}")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        blocks.append(
            {
                "label": match[1].removesuffix("+"),
                "bbox": bbox,
                "text": raw[match.end() : end].strip(),
                "continued": match[1].endswith("+"),
            }
        )
    parts = []
    for block in blocks:
        if block["continued"] and parts:
            parts[-1] += " " + block["text"]
        elif block["text"]:
            parts.append(block["text"])
    return "\n\n".join(parts), blocks


def resolve_lighton_mode(version: int, mode: str, prompt_text: Optional[str]) -> str:
    if mode not in {"auto", "plain", "grounding"}:
        raise ValueError(f"Unsupported LightOn OCR mode: {mode}")
    if mode == "plain" and str(prompt_text or "").strip() == "grounding":
        raise ValueError("LightOn plain mode conflicts with the grounding prompt; clear the prompt or change mode")
    if mode == "auto":
        mode = "grounding" if version == 3 else "plain"
    if str(prompt_text or "").strip() == "grounding":
        mode = "grounding"
    if mode == "grounding" and version != 3:
        raise ValueError("Grounding requires LightOnOCR-3; select plain mode for legacy models")
    return mode


def _build_messages(image_path: str, prompt_text: Optional[str]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [Provider.build_image_part(image_path, field_name="url")]
    if prompt_text and prompt_text.strip():
        content.append(Provider.build_text_part(prompt_text))
    return [Provider.build_message("user", content)]


def _move_inputs_to_device(inputs: Any, *, device: torch.device, dtype: torch.dtype) -> Any:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue
        if torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def _uses_qwen35(model_id: str, architecture: str) -> bool:
    if architecture not in {"auto", "qwen3_5", "lighton_ocr"}:
        raise ValueError(f"Unsupported LightOn architecture: {architecture}")
    if architecture != "auto":
        return architecture == "qwen3_5"
    if re.search(r"lightonocr-3-(?:0\.8|4)b", model_id, re.IGNORECASE):
        return True
    config_path = Path(model_id) / "config.json"
    if config_path.is_file():
        return json.loads(config_path.read_text(encoding="utf-8")).get("model_type") == "qwen3_5"
    return False


@torch.inference_mode()
def attempt_lighton_ocr(
    *,
    uri: str,
    console: Console,
    progress: Optional[Progress],
    task_id: Optional[Any],
    model_id: str = "lightonai/LightOnOCR-3-1B",
    model_version: str = "auto",
    prompt_text: Optional[str] = None,
    pixels: Optional[Pixels] = None,
    output_dir: Optional[str] = None,
    supports_flex_attn: bool = False,
    max_new_tokens: int = 8192,
    mode: str = "auto",
    architecture: str = "auto",
    longest_edge: int = 0,
    temperature: Optional[float] = None,
    runtime: Optional[RuntimeBackendConfig] = None,
) -> str:
    """Run version-aware plain/grounded OCR; Markdown never contains layout markers."""
    start_time = time.time()
    version = ocr_model_version("lighton", model_id, model_version)
    mode = resolve_lighton_mode(version, mode, prompt_text)
    if version == 3 and str(prompt_text or "").strip() not in {"", "grounding"}:
        raise ValueError("LightOnOCR-3 supports only an empty prompt or 'grounding'")
    prompt = "grounding" if mode == "grounding" else (prompt_text or "")
    qwen = _uses_qwen35(model_id, architecture)
    longest_edge = int(longest_edge) or (2048 if qwen else 1540)
    if longest_edge < 1 or max_new_tokens < 1:
        raise ValueError("LightOn longest_edge and max_new_tokens must be positive")
    if temperature is None:
        temperature = 0.1 if version == 3 else 0.0
    if temperature < 0:
        raise ValueError("LightOn temperature must be nonnegative")
    source = Path(uri)
    destination = Path(output_dir) if output_dir else source.with_suffix("")
    pages = prepare_document_pages(source, destination, longest_edge=longest_edge, dpi=200)
    remote = bool(runtime and runtime.is_openai)
    if remote:
        backend = OpenAIChatRuntime(runtime)
    else:
        from transformers import AutoProcessor

        if qwen:
            from transformers import Qwen3_5ForConditionalGeneration as model_class
        else:
            from transformers import LightOnOcrForConditionalGeneration as model_class

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
            model_class,
            dtype=dtype,
            attn_impl=attn_impl,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_safetensors=True,
            console=console,
        )
        eos_ids = generation_eos_ids(model, processor)

    def infer(page: Path) -> str:
        if remote:
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + base64.b64encode(page.read_bytes()).decode("ascii")},
                }
            ]
            if prompt:
                content.append({"type": "text", "text": prompt})
            return backend.complete(
                [{"role": "user", "content": content}],
                require_complete=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}} if qwen else None,
            )
        template_kwargs = {"enable_thinking": False} if qwen else {}
        inputs = processor.apply_chat_template(
            _build_messages(str(page), prompt),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        )
        prepared = _move_inputs_to_device(inputs, device=model.device, dtype=dtype)
        generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            generation_kwargs.update(temperature=temperature, top_p=1.0)
        if eos_ids:
            generation_kwargs["eos_token_id"] = eos_ids
        output_ids = model.generate(**prepared, **generation_kwargs)
        generated = output_ids[0, prepared["input_ids"].shape[1] :]
        ensure_complete_generation(generated, max_new_tokens=max_new_tokens, eos_token_ids=eos_ids)
        return processor.decode(generated, skip_special_tokens=True)

    page_results = page_metadata(pages, destination)
    all_contents = []
    for page, metadata in zip(pages, page_results):
        raw = infer(page)
        content, blocks = parse_grounding(raw) if mode == "grounding" else (raw, [])
        metadata.update(blocks=blocks, raw_output=raw, markdown=content)
        all_contents.append(content.strip())
    # Publish only after every page completed and parsed successfully.
    for page, metadata in zip(pages, page_results):
        write_markdown_output(page.parent, metadata["markdown"])
        if version == 3:
            (page.parent / "result_raw.txt").write_text(metadata["raw_output"], encoding="utf-8")
            write_layout_json(
                page.parent,
                {
                    "schema_version": 1,
                    "model_id": model_id,
                    "mode": mode,
                    "coordinate_system": "normalized_0_1000",
                    **metadata,
                    "image": page.name,
                },
            )
    content = "\n<--- Page Split --->\n".join(all_contents)
    if version == 3:
        write_layout_json(
            destination,
            {
                "schema_version": 1,
                "model_id": model_id,
                "mode": mode,
                "coordinate_system": "normalized_0_1000",
                "page_count": len(pages),
                "pages": page_results,
                "markdown": content,
                "complete_generation": True,
            },
        )
    write_markdown_output(destination, content)
    try:
        display_markdown(title=source.name, markdown_content=content, pixels=pixels, panel_height=32, console=console)
    except Exception:
        pass
    if progress and task_id is not None:
        progress.update(task_id, description="Generating captions")
    console.print(f"[blue]Caption generation took:[/blue] {time.time() - start_time:.2f} seconds")
    return content


@register_provider("lighton_ocr")
class LightOnOCRProvider(OCRProvider):
    """LightOnOCR 1/2 plain OCR and LightOnOCR 3 layout-aware OCR."""

    default_model_id = "lightonai/LightOnOCR-3-1B"
    default_prompt = ""

    def get_runtime_backend(self):
        runtime = super().get_runtime_backend()
        section = self.ctx.config.get(self.name, {})
        model_id = section.get("model_id", self.default_model_id)
        if "lightonocr" in runtime.model_id.lower():
            model_id = runtime.model_id
        version = ocr_model_version("lighton", model_id, section.get("model_version", "auto"))
        selected = find_model_config_section(self.ctx.config, runtime.model_id, preferred_sections=(self.name,))
        if (
            version == 3
            and "temperature" not in section
            and "runtime_temperature" not in section
            and "temperature" not in selected
            and getattr(self.ctx.args, "local_runtime_temperature", None) is None
        ):
            runtime = replace(runtime, temperature=0.1)
        if (
            version == 3
            and not {"runtime_max_tokens", "max_new_tokens"}.intersection(section)
            and not {"runtime_max_tokens", "max_new_tokens"}.intersection(selected)
            and getattr(self.ctx.args, "local_runtime_max_tokens", None) is None
        ):
            runtime = replace(runtime, max_tokens=8192)
        return runtime

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        runtime = self.get_runtime_backend()
        model_id = self._get_model_config("model_id", self.default_model_id)
        if runtime.is_openai and "lightonocr" in runtime.model_id.lower():
            model_id = runtime.model_id
        version = ocr_model_version("lighton", model_id, self._get_model_config("model_version", "auto"))
        mode = self._get_model_config("mode", "auto")
        resolve_lighton_mode(version, mode, prompts.user)
        if runtime.is_openai and version < 3:
            return self.attempt_via_openai_backend(media, prompts)
        output_dir = self.artifact_output_dir(media)
        result = attempt_lighton_ocr(
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
            max_new_tokens=self._get_model_config("max_new_tokens", 8192 if version == 3 else 4096),
            mode=mode,
            architecture=self._get_model_config("architecture", "auto"),
            longest_edge=self._get_model_config("longest_edge", 0),
            temperature=self._get_model_config("temperature", None),
            runtime=runtime,
        )
        return CaptionResult(
            raw=result,
            metadata={
                "provider": self.name,
                "output_dir": str(output_dir),
                "model_id": model_id,
                "runtime_backend": runtime.mode,
                "runtime_model_id": runtime.model_id,
            },
        )
