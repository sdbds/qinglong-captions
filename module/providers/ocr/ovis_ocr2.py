"""OvisOCR2 provider with Direct Transformers and OpenAI-compatible runtimes."""

from __future__ import annotations

import base64
import io
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

from module.providers.backends import OpenAIChatRuntime, RuntimeBackendConfig
from module.providers.base import CaptionResult, MediaContext, MediaModality, PromptContext
from module.providers.ocr_base import OCRProvider
from module.providers.registry import register_provider
from utils.output_writer import write_markdown_output
from utils.parse_display import display_markdown
from utils.stream_util import iter_pdf_pages_high_quality
from utils.transformer_loader import resolve_device_dtype, transformerLoader

_DEFAULT_MIN_PIXELS = 448 * 448
_DEFAULT_MAX_PIXELS = 2880 * 2880
_BBOX_IMAGE_PATTERN = re.compile(r"<img src=" + r'"images/bbox_(\d+)_(\d+)_(\d+)_(\d+)\.jpg" />')
_TRANS_LOADER: Optional[transformerLoader] = None


def _clean_truncated_repeats(
    text: str,
    min_text_len: int = 8000,
    max_period: int = 200,
    min_period: int = 1,
    min_repeat_chars: int = 100,
    min_repeat_times: int = 5,
) -> str:
    """Remove a repeated truncated tail using the algorithm from the model card."""

    n = len(text)
    if n < min_text_len:
        return text

    max_period = min(max_period, n - 1)
    for unit_len in range(min_period, max_period + 1):
        if text[n - 1] != text[n - 1 - unit_len]:
            continue

        match_len = 1
        idx = n - 2
        while idx >= unit_len and text[idx] == text[idx - unit_len]:
            match_len += 1
            idx -= 1

        total_len = match_len + unit_len
        repeat_times = total_len // unit_len
        tail_len = total_len % unit_len
        if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
            return text[: n - total_len + unit_len] + text[n - tail_len :]

    return text


def _process_visual_regions(
    markdown: str,
    page_image: Image.Image,
    output_dir: Path,
    *,
    mode: str,
    warn: Callable[[str], None],
) -> str:
    if mode not in {"crop", "drop"}:
        raise ValueError(f"Unsupported ovis_ocr2 visual_region_mode: {mode}")
    if not markdown or not _BBOX_IMAGE_PATTERN.search(markdown):
        return markdown
    if mode == "drop":
        return _BBOX_IMAGE_PATTERN.sub("", markdown)

    width, height = page_image.size
    images_dir = Path(output_dir) / "images"
    saved_assets: dict[str, bool] = {}

    def replace_tag(match: re.Match[str]) -> str:
        left, top, right, bottom = match.groups()
        asset_name = f"bbox_{left}_{top}_{right}_{bottom}.jpg"
        cached = saved_assets.get(asset_name)
        if cached is not None:
            return match.group(0) if cached else ""

        x1 = max(0, min(width, round(int(left) * width / 1000)))
        y1 = max(0, min(height, round(int(top) * height / 1000)))
        x2 = max(0, min(width, round(int(right) * width / 1000)))
        y2 = max(0, min(height, round(int(bottom) * height / 1000)))
        if x2 <= x1 or y2 <= y1:
            saved_assets[asset_name] = False
            warn(f"Dropped invalid visual-region bbox: {asset_name}")
            return ""

        asset_path = images_dir / asset_name
        try:
            images_dir.mkdir(parents=True, exist_ok=True)
            crop = page_image.crop((x1, y1, x2, y2)).convert("RGB")
            try:
                crop.save(asset_path)
            finally:
                crop.close()
        except Exception as exc:
            saved_assets[asset_name] = False
            try:
                asset_path.unlink(missing_ok=True)
            except Exception:
                pass
            warn(f"Dropped visual-region tag after crop save failed for {asset_name}: {exc}")
            return ""

        saved_assets[asset_name] = True
        return match.group(0)

    return _BBOX_IMAGE_PATTERN.sub(replace_tag, markdown)


def _prefix_visual_region_paths(markdown: str, page_dir_name: str) -> str:
    if not markdown:
        return markdown

    def prefix_tag(match: re.Match[str]) -> str:
        return match.group(0).replace('src="images/', f'src="{page_dir_name}/images/', 1)

    return _BBOX_IMAGE_PATTERN.sub(prefix_tag, markdown)


def _save_page_snapshot(image: Image.Image, path: Path, warn: Callable[[str], None]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG")
        return True
    except Exception as exc:
        warn(f"Could not save debug page snapshot {path.name}: {exc}")
        return False


class _DirectPageInferencer:
    def __init__(
        self,
        *,
        model_id: str,
        max_new_tokens: int,
        min_pixels: int,
        max_pixels: int,
        console: Any,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = int(max_new_tokens)
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.console = console

    def _load_components(self) -> tuple[Any, Any, str]:
        try:
            from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "OvisOCR2 Direct inference requires transformers[serving]>=5.7.0. Install the ovis-ocr2 extra."
            ) from exc

        device, dtype, attn_impl = resolve_device_dtype()
        global _TRANS_LOADER
        if _TRANS_LOADER is None:
            _TRANS_LOADER = transformerLoader(device_map="auto")

        processor = _TRANS_LOADER.get_or_load_processor(
            self.model_id,
            AutoProcessor,
            console=self.console,
            trust_remote_code=False,
        )
        model = _TRANS_LOADER.get_or_load_model(
            self.model_id,
            Qwen3_5ForConditionalGeneration,
            dtype=dtype,
            attn_impl=attn_impl,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_safetensors=True,
            console=self.console,
        )
        return processor, model, device

    def infer_page(self, image: Image.Image, prompt: str) -> str:
        import torch

        processor, model, fallback_device = self._load_components()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
            images_kwargs={"min_pixels": self.min_pixels, "max_pixels": self.max_pixels},
        )
        inputs = inputs.to(getattr(model, "device", fallback_device))
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        input_ids = inputs["input_ids"]
        trimmed_ids = [output_ids[len(source_ids) :] for source_ids, output_ids in zip(input_ids, generated_ids)]
        decoded = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0] if decoded else ""


class _OpenAIPageInferencer:
    def __init__(self, *, runtime: RuntimeBackendConfig, min_pixels: int, max_pixels: int) -> None:
        self.runtime = OpenAIChatRuntime(runtime)
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)

    @staticmethod
    def _png_data_url(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{payload}"

    def infer_page(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": self._png_data_url(image)},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self.runtime.complete(
            messages,
            extra_body={
                "mm_processor_kwargs": {
                    "images_kwargs": {
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    }
                },
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )


@register_provider("ovis_ocr2")
class OvisOCR2Provider(OCRProvider):
    default_model_id = "ATH-MaaS/OvisOCR2"
    default_prompt = (
        "\nExtract all readable content from the image in natural human reading order and output the result as a single "
        "Markdown document. For charts or images, represent them using an HTML image tag: "
        '<img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, where left, top, right, bottom are bounding box '
        "coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as HTML: <table>...</table>. "
        "Transcribe all other text as standard Markdown. Preserve the original text without translation or paraphrasing."
    )

    @classmethod
    def can_handle(cls, args: Any, mime: str) -> bool:
        if getattr(args, "ocr_model", "") != cls.name:
            return False
        if mime == "application/pdf":
            return True
        return mime.startswith("image/") and bool(getattr(args, "document_image", False))

    def prepare_media(self, uri: str, mime: str, args: Any) -> MediaContext:
        return MediaContext(
            uri=uri,
            mime=mime,
            sha256hash="",
            modality=MediaModality.DOCUMENT if mime == "application/pdf" else MediaModality.IMAGE,
            extras={"output_dir": Path(uri).with_suffix("")},
        )

    def _provider_config(self) -> dict[str, Any]:
        section = self.ctx.config.get(self.name, {})
        return section if isinstance(section, dict) else {}

    def get_prompts(self, mime: str):
        provider_prompt = str(self._provider_config().get("prompt", "") or "").strip()
        if provider_prompt:
            return "", provider_prompt

        prompts = self.ctx.config.get("prompts", {})
        legacy_prompt = str(prompts.get("ovis_ocr2_prompt", "") or "").strip() if isinstance(prompts, dict) else ""
        return "", legacy_prompt or self.default_prompt

    def _get_visual_region_mode(self) -> str:
        mode = str(self._provider_config().get("visual_region_mode", "crop") or "crop").strip().lower()
        if mode not in {"crop", "drop"}:
            raise ValueError(f"Unsupported ovis_ocr2 visual_region_mode: {mode}")
        return mode

    def get_runtime_backend(self) -> RuntimeBackendConfig:
        runtime = super().get_runtime_backend()
        section = self._provider_config()
        args_top_p = getattr(self.ctx.args, "local_runtime_top_p", None)
        if args_top_p not in (None, "") or section.get("runtime_top_p") not in (None, ""):
            return runtime
        configured_top_p = section.get("top_p")
        if configured_top_p in (None, ""):
            return runtime
        return replace(runtime, top_p=float(configured_top_p))

    def _create_inferencer(self, runtime: RuntimeBackendConfig):
        min_pixels = int(self._get_model_config("min_pixels", _DEFAULT_MIN_PIXELS))
        max_pixels = int(self._get_model_config("max_pixels", _DEFAULT_MAX_PIXELS))
        if min_pixels <= 0 or max_pixels < min_pixels:
            raise ValueError("ovis_ocr2 pixel bounds must satisfy 0 < min_pixels <= max_pixels")
        if runtime.is_openai:
            return _OpenAIPageInferencer(
                runtime=runtime,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        return _DirectPageInferencer(
            model_id=runtime.model_id or self.default_model_id,
            max_new_tokens=runtime.max_tokens,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            console=self.ctx.console,
        )

    def _warn(self, message: str) -> None:
        self.ctx.console.print(f"[yellow]OvisOCR2 warning:[/yellow] {message}")

    def _process_page(
        self,
        *,
        image: Image.Image,
        output_dir: Path,
        snapshot_name: str,
        inferencer: Any,
        prompt: str,
        visual_region_mode: str,
    ) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_page_snapshot(image, output_dir / snapshot_name, self._warn)
        raw_text = inferencer.infer_page(image, prompt)
        if raw_text is None or not str(raw_text).strip():
            raise ValueError("OvisOCR2 returned empty output for page")

        cleaned = _clean_truncated_repeats(str(raw_text).strip())
        rendered = _process_visual_regions(
            cleaned,
            image,
            output_dir,
            mode=visual_region_mode,
            warn=self._warn,
        )
        write_markdown_output(output_dir, rendered)
        return rendered

    def _attempt_pdf(
        self,
        media: MediaContext,
        *,
        output_dir: Path,
        inferencer: Any,
        prompt: str,
        visual_region_mode: str,
    ) -> tuple[str, list[int]]:
        page_contents: list[str] = []
        failed_pages: list[int] = []
        attempted_pages = 0
        last_error: Optional[BaseException] = None

        for rendered_page in iter_pdf_pages_high_quality(media.uri):
            attempted_pages += 1
            page_number = int(rendered_page.page_number)
            page_dir = output_dir / f"page_{page_number:04d}"
            page_image: Optional[Image.Image] = None
            try:
                page_image = rendered_page.image.convert("RGB")
                page_content = self._process_page(
                    image=page_image,
                    output_dir=page_dir,
                    snapshot_name=f"page_{page_number:04d}.png",
                    inferencer=inferencer,
                    prompt=prompt,
                    visual_region_mode=visual_region_mode,
                )
            except Exception as exc:
                failed_pages.append(page_number)
                last_error = exc
                self._warn(f"Page {page_number} failed and was skipped: {exc}")
            else:
                page_contents.append(_prefix_visual_region_paths(page_content, page_dir.name))
            finally:
                if page_image is not None:
                    page_image.close()
                try:
                    rendered_page.image.close()
                except Exception:
                    pass

        if not page_contents:
            raise RuntimeError(f"OvisOCR2 failed to process all {attempted_pages} PDF pages") from last_error

        content = "\n<--- Page Split --->\n".join(page_contents)
        write_markdown_output(output_dir, content)
        return content, failed_pages

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        start_time = time.time()
        output_dir = Path(media.extras.get("output_dir") or Path(media.uri).with_suffix(""))
        visual_region_mode = self._get_visual_region_mode()
        runtime = self.get_runtime_backend()
        inferencer = self._create_inferencer(runtime)

        if media.mime == "application/pdf":
            content, failed_pages = self._attempt_pdf(
                media,
                output_dir=output_dir,
                inferencer=inferencer,
                prompt=prompts.user,
                visual_region_mode=visual_region_mode,
            )
        else:
            with Image.open(media.uri) as source:
                page_image = source.convert("RGB")
            try:
                content = self._process_page(
                    image=page_image,
                    output_dir=output_dir,
                    snapshot_name="page.png",
                    inferencer=inferencer,
                    prompt=prompts.user,
                    visual_region_mode=visual_region_mode,
                )
            finally:
                page_image.close()
            failed_pages = []

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

        self.ctx.console.print(f"[blue]Caption generation took:[/blue] {time.time() - start_time:.2f} seconds")
        return CaptionResult(
            raw=content,
            metadata={
                "provider": self.name,
                "output_dir": str(output_dir),
                "runtime_backend": runtime.mode,
                "runtime_model_id": runtime.model_id,
                "failed_pages": failed_pages,
                "visual_region_mode": visual_region_mode,
            },
        )
