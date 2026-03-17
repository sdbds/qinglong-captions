"""Dots OCR provider."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

from providers.backends import OpenAIChatRuntime, find_model_config_section, resolve_runtime_backend
from providers.base import CaptionResult, MediaContext, PromptContext
from providers.ocr_base import OCRProvider
from providers.registry import register_provider
from providers.utils import build_vision_messages, encode_image_to_blob
from utils.output_writer import write_markdown_output
from utils.parse_display import display_markdown, extract_code_block_content
from utils.stream_util import pdf_to_images_high_quality
from utils.transformer_loader import resolve_device_dtype, transformerLoader

DEFAULT_PROMPT_MODE = "prompt_layout_all_en"
DEFAULT_SVG_MODEL_ID = "davanstrien/dots.ocr-1.5-svg"
_TRANS_LOADER: Optional[transformerLoader] = None

try:
    import importlib.metadata as importlib_metadata
except ImportError:  # pragma: no cover - Python < 3.8 fallback
    import importlib_metadata  # type: ignore[no-redef]


def _find_upstream_prompts_file() -> Path | None:
    """Locate the upstream prompts.py file without importing the broken package."""
    for dist_name in ("dots_ocr", "dots-ocr"):
        try:
            distribution = importlib_metadata.distribution(dist_name)
        except importlib_metadata.PackageNotFoundError:
            continue
        candidate = Path(distribution.locate_file("dots_ocr/utils/prompts.py"))
        if candidate.is_file():
            return candidate

    for entry in sys.path:
        candidate = Path(entry) / "dots_ocr" / "utils" / "prompts.py"
        if candidate.is_file():
            return candidate
    return None


def _load_prompt_mapping_from_file(prompts_path: Path) -> dict[str, str]:
    """Load dict_promptmode_to_prompt from the upstream prompts.py source file."""
    spec = importlib.util.spec_from_file_location("_dots_ocr_prompts", str(prompts_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load prompts spec from {prompts_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompt_map = getattr(module, "dict_promptmode_to_prompt", None)
    if not isinstance(prompt_map, dict) or not prompt_map:
        raise ValueError(f"dict_promptmode_to_prompt missing from {prompts_path}")
    return {str(key): str(value) for key, value in prompt_map.items()}


def _load_upstream_prompt_mapping() -> dict[str, str]:
    """Load the official dots OCR prompt-mode mapping from the upstream package."""
    prompts_path = _find_upstream_prompts_file()
    if prompts_path is None:
        raise ImportError(
            "dots_ocr prompt mapping not available. Install the dots-ocr extra."
        )
    return _load_prompt_mapping_from_file(prompts_path)


def _download_model_snapshot(repo_id: str) -> str:
    """Resolve an HF repo id to a local snapshot path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "dots_ocr model download support is unavailable. Install the dots-ocr extra."
        ) from exc
    return str(snapshot_download(repo_id=repo_id))


@lru_cache(maxsize=8)
def _resolve_model_source(model_id: str) -> str:
    """Use a local snapshot path to avoid HF dynamic-module import issues on dotted repo names."""
    candidate = Path(model_id).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return str(Path(_download_model_snapshot(model_id)).resolve())


def _save_pdf_page_image(pil_image, page_path: Path) -> str:
    """Persist a rendered PDF page image and return its file path."""
    page_path = Path(page_path)
    page_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pil_image.save(page_path)
    except Exception:
        pil_image.convert("RGB").save(page_path)
    return str(page_path)


@register_provider("dots_ocr")
class DotsOCRProvider(OCRProvider):
    """OCR provider backed by davanstrien/dots.ocr-1.5."""

    default_model_id = "davanstrien/dots.ocr-1.5"
    default_prompt = ""
    default_svg_model_id = DEFAULT_SVG_MODEL_ID

    def resolve_prompts(self, uri: str, mime: str) -> PromptContext:
        _, prompt_text = self._resolve_prompt_mode_and_prompt()
        char_name, char_prompt = self._get_character_prompt(uri)
        return PromptContext(
            system="",
            user=f"{char_prompt}{prompt_text}" if char_prompt else prompt_text,
            character_name=char_name,
            character_prompt=char_prompt,
        )

    def _resolve_prompt_mode_and_prompt(self) -> tuple[str, str]:
        prompt_map = _load_upstream_prompt_mapping()
        prompt_mode = str(
            self._get_model_config("prompt_mode", DEFAULT_PROMPT_MODE)
            or DEFAULT_PROMPT_MODE
        ).strip()
        if prompt_mode not in prompt_map:
            raise ValueError(f"Unsupported dots_ocr prompt_mode: {prompt_mode}")

        prompt_text = prompt_map[prompt_mode]
        provider_prompt = str(self._get_model_config("prompt", "") or "").strip()
        global_prompt = str(
            self.ctx.config.get("prompts", {}).get("dots_ocr_prompt", "") or ""
        ).strip()
        if provider_prompt:
            prompt_text = provider_prompt
        elif global_prompt:
            prompt_text = global_prompt
        return prompt_mode, prompt_text

    def _select_model_id(self, prompt_mode: str) -> str:
        if prompt_mode == "prompt_image_to_svg":
            return str(
                self._get_model_config("svg_model_id", self.default_svg_model_id)
                or self.default_svg_model_id
            )
        return str(
            self._get_model_config("model_id", self.default_model_id)
            or self.default_model_id
        )

    def _write_text_result(self, output_dir: Path, content: str) -> Path:
        return write_markdown_output(Path(output_dir), str(content), filename="result.md")

    def _write_svg_result(self, output_dir: Path, content: str) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "result.svg"
        result_path.write_text(str(content), encoding="utf-8")
        return result_path

    def _build_svg_pdf_markdown(self, page_dirs: list[Path]) -> str:
        blocks: list[str] = []
        for page_number, page_dir in enumerate(page_dirs, start=1):
            blocks.append(f"## Page {page_number}\n![]({page_dir.name}/result.svg)")
        return "\n\n".join(blocks)

    def _run_direct_generation(
        self,
        *,
        image_path: str,
        prompt_mode: str,
        prompt_text: str,
        model_id: str,
        max_new_tokens: int,
    ) -> str:
        try:
            from qwen_vl_utils import process_vision_info
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "dots_ocr direct generation dependencies are unavailable. Install the dots-ocr extra."
            ) from exc

        _, dtype, attn_impl = resolve_device_dtype()
        global _TRANS_LOADER
        if _TRANS_LOADER is None:
            _TRANS_LOADER = transformerLoader(attn_kw="attn_implementation", device_map="auto")

        load_source = _resolve_model_source(model_id)
        processor = _TRANS_LOADER.get_or_load_processor(
            load_source,
            AutoProcessor,
            console=self.ctx.console,
            trust_remote_code=True,
        )
        model = _TRANS_LOADER.get_or_load_model(
            load_source,
            AutoModelForCausalLM,
            dtype=dtype,
            attn_impl=attn_impl,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_safetensors=True,
            console=self.ctx.console,
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        try:
            model_device = model.device
        except Exception:
            model_device = next(model.parameters()).device
        inputs = inputs.to(model_device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        content = output_texts[0] if output_texts else ""
        if prompt_mode == "prompt_image_to_svg":
            extracted = extract_code_block_content(content, code_type="svg", console=self.ctx.console)
            return extracted or content
        return content

    def _get_runtime_backend_for_prompt_mode(self, prompt_mode: str):
        provider_section = self.ctx.config.get(self.name, {})
        selected_model_id = self._select_model_id(prompt_mode)
        model_section = find_model_config_section(
            self.ctx.config,
            selected_model_id,
            preferred_sections=(self.name,),
        )
        default_temperature = float(
            model_section.get(
                "temperature",
                provider_section.get("runtime_temperature", provider_section.get("temperature", 0.0)),
            )
        )
        default_max_tokens = int(
            model_section.get(
                "runtime_max_tokens",
                model_section.get(
                    "max_new_tokens",
                    provider_section.get("runtime_max_tokens", provider_section.get("max_new_tokens", 4096)),
                ),
            )
        )
        return resolve_runtime_backend(
            self.ctx.args,
            provider_section,
            arg_prefix="local_runtime",
            shared_prefix="openai",
            default_model_id=selected_model_id,
            default_temperature=default_temperature,
            default_max_tokens=default_max_tokens,
        )

    def _complete_via_openai_runtime(
        self,
        *,
        image_path: str,
        prompt_mode: str,
        prompt_text: str,
    ) -> str:
        runtime = self._get_runtime_backend_for_prompt_mode(prompt_mode)
        backend = OpenAIChatRuntime(runtime)
        blob, _ = encode_image_to_blob(image_path, to_rgb=True)
        if not blob:
            raise RuntimeError(f"DOTS_OCR_IMAGE_ENCODE_FAILED: {image_path}")

        messages = build_vision_messages(
            "",
            prompt_text,
            blob,
            text_first=False,
        )
        content = backend.complete(messages)
        if prompt_mode == "prompt_image_to_svg":
            extracted = extract_code_block_content(content, code_type="svg", console=self.ctx.console)
            return extracted or content
        return content

    def _attempt_via_openai_backend(
        self,
        media: MediaContext,
        prompt_mode: str,
        prompt_text: str,
    ) -> CaptionResult:
        output_dir = Path(media.extras.get("output_dir") or Path(media.uri).with_suffix(""))
        output_dir.mkdir(parents=True, exist_ok=True)
        runtime = self._get_runtime_backend_for_prompt_mode(prompt_mode)

        if media.mime.startswith("application/pdf"):
            images = pdf_to_images_high_quality(media.uri)
            if prompt_mode == "prompt_image_to_svg":
                page_dirs: list[Path] = []
                for index, pil_image in enumerate(images, start=1):
                    page_dir = output_dir / f"page_{index:04d}"
                    page_image_path = page_dir / f"page_{index:04d}.png"
                    try:
                        saved_image_path = _save_pdf_page_image(pil_image, page_image_path)
                    except Exception:
                        continue
                    svg_content = self._complete_via_openai_runtime(
                        image_path=str(saved_image_path),
                        prompt_mode=prompt_mode,
                        prompt_text=prompt_text,
                    )
                    self._write_svg_result(page_dir, svg_content)
                    page_dirs.append(page_dir)

                root_markdown = self._build_svg_pdf_markdown(page_dirs)
                self._write_text_result(output_dir, root_markdown)
                return CaptionResult(
                    raw=root_markdown,
                    metadata={
                        "provider": self.name,
                        "output_dir": str(output_dir),
                        "runtime_backend": runtime.mode,
                        "runtime_model_id": runtime.model_id,
                        "prompt_mode": prompt_mode,
                    },
                )

            page_contents: list[str] = []
            for index, pil_image in enumerate(images, start=1):
                page_dir = output_dir / f"page_{index:04d}"
                page_image_path = page_dir / f"page_{index:04d}.png"
                try:
                    saved_image_path = _save_pdf_page_image(pil_image, page_image_path)
                except Exception:
                    continue
                page_content = self._complete_via_openai_runtime(
                    image_path=str(saved_image_path),
                    prompt_mode=prompt_mode,
                    prompt_text=prompt_text,
                )
                self._write_text_result(page_dir, page_content)
                page_contents.append(str(page_content).strip())

            content = "\n<--- Page Split --->\n".join(page_contents)
            self._write_text_result(output_dir, content)
            self._display_text_result(media, content)
            return CaptionResult(
                raw=content,
                metadata={
                    "provider": self.name,
                    "output_dir": str(output_dir),
                    "runtime_backend": runtime.mode,
                    "runtime_model_id": runtime.model_id,
                    "prompt_mode": prompt_mode,
                },
            )

        content = self._complete_via_openai_runtime(
            image_path=media.uri,
            prompt_mode=prompt_mode,
            prompt_text=prompt_text,
        )
        if prompt_mode == "prompt_image_to_svg":
            self._write_svg_result(output_dir, content)
        else:
            self._write_text_result(output_dir, content)
            self._display_text_result(media, content)
        return CaptionResult(
            raw=str(content),
            metadata={
                "provider": self.name,
                "output_dir": str(output_dir),
                "runtime_backend": runtime.mode,
                "runtime_model_id": runtime.model_id,
                "prompt_mode": prompt_mode,
            },
        )

    def _display_text_result(self, media: MediaContext, content: str) -> None:
        try:
            display_markdown(
                title=Path(media.uri).name,
                markdown_content=content,
                pixels=media.pixels,
                panel_height=32,
                console=self.ctx.console,
            )
        except Exception:
            pass

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        prompt_mode, resolved_prompt = self._resolve_prompt_mode_and_prompt()
        prompt_text = prompts.user or resolved_prompt
        model_id = self._select_model_id(prompt_mode)
        output_dir = Path(media.extras.get("output_dir") or Path(media.uri).with_suffix(""))
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.get_runtime_backend().is_openai:
            return self._attempt_via_openai_backend(media, prompt_mode, prompt_text)

        max_new_tokens = int(self._get_model_config("max_new_tokens", 8192) or 8192)

        if media.mime.startswith("application/pdf"):
            images = pdf_to_images_high_quality(media.uri)
            if prompt_mode == "prompt_image_to_svg":
                page_dirs: list[Path] = []
                for index, pil_image in enumerate(images, start=1):
                    page_dir = output_dir / f"page_{index:04d}"
                    page_image_path = page_dir / f"page_{index:04d}.png"
                    try:
                        saved_image_path = _save_pdf_page_image(pil_image, page_image_path)
                    except Exception:
                        continue
                    svg_content = self._run_direct_generation(
                        image_path=str(saved_image_path),
                        prompt_mode=prompt_mode,
                        prompt_text=prompt_text,
                        model_id=model_id,
                        max_new_tokens=max_new_tokens,
                    )
                    self._write_svg_result(page_dir, svg_content)
                    page_dirs.append(page_dir)

                root_markdown = self._build_svg_pdf_markdown(page_dirs)
                self._write_text_result(output_dir, root_markdown)
                return CaptionResult(
                    raw=root_markdown,
                    metadata={
                        "provider": self.name,
                        "output_dir": str(output_dir),
                        "prompt_mode": prompt_mode,
                        "model_id": model_id,
                    },
                )

            page_contents: list[str] = []
            for index, pil_image in enumerate(images, start=1):
                page_dir = output_dir / f"page_{index:04d}"
                page_image_path = page_dir / f"page_{index:04d}.png"
                try:
                    saved_image_path = _save_pdf_page_image(pil_image, page_image_path)
                except Exception:
                    continue
                page_content = self._run_direct_generation(
                    image_path=str(saved_image_path),
                    prompt_mode=prompt_mode,
                    prompt_text=prompt_text,
                    model_id=model_id,
                    max_new_tokens=max_new_tokens,
                )
                self._write_text_result(page_dir, page_content)
                page_contents.append(str(page_content).strip())

            content = "\n<--- Page Split --->\n".join(page_contents)
            self._write_text_result(output_dir, content)
            self._display_text_result(media, content)
            return CaptionResult(
                raw=content,
                metadata={
                    "provider": self.name,
                    "output_dir": str(output_dir),
                    "prompt_mode": prompt_mode,
                    "model_id": model_id,
                },
            )

        content = self._run_direct_generation(
            image_path=media.uri,
            prompt_mode=prompt_mode,
            prompt_text=prompt_text,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
        )
        if prompt_mode == "prompt_image_to_svg":
            self._write_svg_result(output_dir, content)
        else:
            self._write_text_result(output_dir, content)
            self._display_text_result(media, content)
        return CaptionResult(
            raw=str(content),
            metadata={
                "provider": self.name,
                "output_dir": str(output_dir),
                "prompt_mode": prompt_mode,
                "model_id": model_id,
            },
        )
