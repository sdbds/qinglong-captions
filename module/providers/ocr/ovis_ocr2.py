"""OvisOCR2 provider with Direct Transformers and OpenAI-compatible runtimes."""

from __future__ import annotations

import base64
import io
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

from module.providers.backends import OpenAIChatRuntime, RuntimeBackendConfig
from module.providers.base import CaptionResult, MediaContext, MediaModality, PromptContext
from module.providers.ocr.ovis_ocr2_contract import OVIS_OCR2_DEFAULT_PROMPT
from module.providers.ocr_base import OCRProvider
from module.providers.registry import register_provider
from utils.output_writer import write_markdown_output
from utils.parse_display import display_markdown
from utils.stream_util import iter_pdf_pages_high_quality
from utils.transformer_loader import resolve_device_dtype, transformerLoader

_DEFAULT_MIN_PIXELS = 448 * 448
_DEFAULT_MAX_PIXELS = 2880 * 2880
_EARLY_REPEAT_START_TOKENS = 128
_EARLY_REPEAT_CHECK_INTERVAL = 32
_EARLY_REPEAT_TAIL_TOKENS = 768
_EARLY_REPEAT_MIN_CHARS = 200
_EARLY_REPEAT_MIN_TIMES = 8
_REPEAT_FINGERPRINT_MIN_CHARS = 64
_BBOX_IMAGE_PATTERN = re.compile(r"<img src=" + r'"images/bbox_(\d+)_(\d+)_(\d+)_(\d+)\.jpg" />')
_EMPTY_THINKING_BLOCK_PATTERN = re.compile(r"(?m)^[ \t]*<think>[ \t]*(?:\r?\n[ \t]*)*</think>[ \t]*(?=\r?$)")
_TRANS_LOADER: Optional[transformerLoader] = None


@dataclass(frozen=True)
class _RepeatedTailMatch:
    period_len: int
    matched_chars: int
    repeat_times: int
    trailing_chars: int
    suffix_fingerprint: str


def _find_repeated_tail(
    text: str,
    *,
    min_text_len: int,
    max_period: int,
    min_period: int,
    min_repeat_chars: int,
    min_repeat_times: int,
    expected_period_len: Optional[int] = None,
) -> Optional[_RepeatedTailMatch]:
    n = len(text)
    if n < min_text_len or n < 2:
        return None

    lower_period = max(1, min_period)
    upper_period = min(max_period, n - 1)
    periods = (expected_period_len,) if expected_period_len is not None else range(lower_period, upper_period + 1)
    for period_len in periods:
        if period_len < lower_period or period_len > upper_period:
            continue
        if text[n - 1] != text[n - 1 - period_len]:
            continue

        match_len = 1
        index = n - 2
        while index >= period_len and text[index] == text[index - period_len]:
            match_len += 1
            index -= 1

        matched_chars = match_len + period_len
        repeat_times = matched_chars // period_len
        trailing_chars = matched_chars % period_len
        if repeat_times < min_repeat_times or matched_chars < min_repeat_chars:
            continue

        fingerprint_len = min(
            matched_chars,
            max(_REPEAT_FINGERPRINT_MIN_CHARS, period_len * 2),
        )
        return _RepeatedTailMatch(
            period_len=period_len,
            matched_chars=matched_chars,
            repeat_times=repeat_times,
            trailing_chars=trailing_chars,
            suffix_fingerprint=text[-fingerprint_len:],
        )
    return None


def _collapse_repeated_tail(text: str, match: _RepeatedTailMatch) -> str:
    suffix_start = len(text) - match.matched_chars + match.period_len
    trailing = text[-match.trailing_chars :] if match.trailing_chars else ""
    return text[:suffix_start] + trailing


def _clean_truncated_repeats(
    text: str,
    min_text_len: int = 8000,
    max_period: int = 200,
    min_period: int = 1,
    min_repeat_chars: int = 100,
    min_repeat_times: int = 5,
) -> str:
    """Remove a repeated truncated tail using the algorithm from the model card."""

    match = _find_repeated_tail(
        text,
        min_text_len=min_text_len,
        max_period=max_period,
        min_period=min_period,
        min_repeat_chars=min_repeat_chars,
        min_repeat_times=min_repeat_times,
    )
    return _collapse_repeated_tail(text, match) if match is not None else text


def _truncate_at_thinking_marker(text: str) -> str:
    marker = _EMPTY_THINKING_BLOCK_PATTERN.search(text)
    return text[: marker.start()].rstrip() if marker is not None else text


def _normalize_triggered_repeat(
    text: str,
    trigger: _RepeatedTailMatch,
) -> Optional[str]:
    if not text.endswith(trigger.suffix_fingerprint):
        return None

    full_match = _find_repeated_tail(
        text,
        min_text_len=0,
        max_period=trigger.period_len,
        min_period=trigger.period_len,
        min_repeat_chars=_EARLY_REPEAT_MIN_CHARS,
        min_repeat_times=_EARLY_REPEAT_MIN_TIMES,
        expected_period_len=trigger.period_len,
    )
    if full_match is None:
        return None
    return _collapse_repeated_tail(text, full_match)


class _RepeatedTailStoppingCriteria:
    def __init__(self, processor: Any, prompt_length: int) -> None:
        self.processor = processor
        self.prompt_length = int(prompt_length)
        self.triggered_match: Optional[_RepeatedTailMatch] = None
        self.triggered_at_tokens: Optional[int] = None
        self.stop_reason: Optional[str] = None
        self._next_check = _EARLY_REPEAT_START_TOKENS

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
        import torch

        if int(input_ids.shape[0]) != 1:
            raise ValueError("OvisOCR2 repeated-tail stopping requires batch size 1")

        generated_tokens = max(0, int(input_ids.shape[1]) - self.prompt_length)
        if generated_tokens < self._next_check:
            return torch.zeros((1,), device=input_ids.device, dtype=torch.bool)

        self._next_check = generated_tokens + _EARLY_REPEAT_CHECK_INTERVAL
        tail_start = max(
            self.prompt_length,
            int(input_ids.shape[1]) - _EARLY_REPEAT_TAIL_TOKENS,
        )
        decoded = self.processor.batch_decode(
            input_ids[:, tail_start:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        tail_text = decoded[0] if decoded else ""
        if _EMPTY_THINKING_BLOCK_PATTERN.search(tail_text) is not None:
            self.stop_reason = "thinking_marker"
            self.triggered_at_tokens = generated_tokens
            return torch.ones((1,), device=input_ids.device, dtype=torch.bool)

        match = _find_repeated_tail(
            tail_text,
            min_text_len=0,
            max_period=200,
            min_period=1,
            min_repeat_chars=_EARLY_REPEAT_MIN_CHARS,
            min_repeat_times=_EARLY_REPEAT_MIN_TIMES,
        )
        if match is not None:
            self.triggered_match = match
            self.triggered_at_tokens = generated_tokens
            self.stop_reason = "repeated_tail"
        return torch.full((1,), match is not None, device=input_ids.device, dtype=torch.bool)


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
            processor_kwargs={
                "images_kwargs": {"min_pixels": self.min_pixels, "max_pixels": self.max_pixels},
            },
        )
        inputs = inputs.to(getattr(model, "device", fallback_device))
        stopping_criterion = _RepeatedTailStoppingCriteria(
            processor,
            prompt_length=int(inputs["input_ids"].shape[1]),
        )
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                stopping_criteria=[stopping_criterion],
            )
        input_ids = inputs["input_ids"]
        trimmed_ids = [output_ids[len(source_ids) :] for source_ids, output_ids in zip(input_ids, generated_ids)]
        decoded = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        text = decoded[0] if decoded else ""
        if stopping_criterion.stop_reason == "thinking_marker":
            self.console.print(
                "[yellow]OvisOCR2 stopped at generated thinking marker:[/yellow] "
                f"generated_tokens={stopping_criterion.triggered_at_tokens}"
            )
        trigger = stopping_criterion.triggered_match
        if trigger is not None:
            normalized = _normalize_triggered_repeat(text, trigger)
            if normalized is None:
                self.console.print(
                    "[yellow]OvisOCR2 warning:[/yellow] could not revalidate repeated tail; preserving decoded output"
                )
            else:
                text = normalized
                self.console.print(
                    "[yellow]OvisOCR2 stopped repeated output:[/yellow] "
                    f"generated_tokens={stopping_criterion.triggered_at_tokens}, "
                    f"period_chars={trigger.period_len}, repeat_times={trigger.repeat_times}"
                )
        return text


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
    default_prompt = OVIS_OCR2_DEFAULT_PROMPT

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

        cleaned = _truncate_at_thinking_marker(str(raw_text).strip())
        if not cleaned:
            raise ValueError("OvisOCR2 returned empty output for page")
        cleaned = _clean_truncated_repeats(cleaned)
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
