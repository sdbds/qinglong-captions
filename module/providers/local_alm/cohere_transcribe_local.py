"""Cohere Transcribe local audio-language-model provider."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from module.caption_pipeline.postprocess import strip_reasoning_sections
from module.providers.base import CaptionResult, MediaContext, PromptContext
from module.providers.local_alm_base import ALMTaskContract, LocalALMProvider
from module.providers.registry import register_provider
from utils.parse_display import extract_code_block_content


def _is_cuda_device(device: Any) -> bool:
    return str(device or "").startswith("cuda")


def _resolve_pretrained_device_map(device: Any, device_map: Any) -> Any:
    if device_map != "auto":
        return device_map
    if not _is_cuda_device(device):
        return device_map
    text = str(device)
    if text == "cuda":
        return "auto"
    return {"": text}


@register_provider("cohere_transcribe_local")
class CohereTranscribeLocalProvider(LocalALMProvider):
    default_model_id = "CohereLabs/cohere-transcribe-03-2026"
    task_contract = ALMTaskContract(
        task_kind="transcribe",
        consumes_prompts=False,
        requires_language=True,
        default_caption_extension=".txt",
    )
    transcribe_config_keys = (
        "language",
        "punctuation",
        "batch_size",
        "compile",
        "pipeline_detokenization",
    )
    default_transcribe_kwargs = {
        "punctuation": True,
        "compile": False,
        "pipeline_detokenization": False,
    }

    @classmethod
    def can_handle(cls, args: Any, mime: str) -> bool:
        return getattr(args, "alm_model", "") == "cohere_transcribe_local" and mime.startswith("audio")

    def _load_model(self):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        from utils.transformer_loader import load_pretrained_component, move_pretrained_component, resolve_device_dtype

        model_id = self.model_id
        try:
            device, dtype, attn_impl = resolve_device_dtype(
                supports_flex_attn=bool(getattr(self, "_supports_flex_attn", False))
            )
        except TypeError:
            device, dtype, attn_impl = resolve_device_dtype()
        self.log(f"Loading Cohere Transcribe model: {model_id} (device={device}, dtype={dtype})", "blue")

        processor = load_pretrained_component(
            AutoProcessor,
            model_id,
            console=self.ctx.console,
            component_name="processor",
            trust_remote_code=True,
        )

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl
        if _is_cuda_device(device):
            load_kwargs["device_map"] = _resolve_pretrained_device_map(device, "auto")

        model = load_pretrained_component(
            AutoModelForSpeechSeq2Seq,
            model_id,
            console=self.ctx.console,
            component_name="model",
            **load_kwargs,
        )
        try:
            model = model.eval()
        except Exception:
            pass
        if not _is_cuda_device(device):
            model = move_pretrained_component(model, device=device)

        return {"model": model, "processor": processor}

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        del prompts  # Cohere Transcribe exposes a dedicated ASR API and ignores chat-style prompts.

        cached = self._get_or_load_model()
        model = cached["model"]
        processor = cached["processor"]

        if not hasattr(model, "transcribe"):
            raise RuntimeError(
                "Loaded Cohere Transcribe model does not expose model.transcribe(); "
                "check the installed transformers version and trust_remote_code support."
            )

        audio_path = str(Path(media.uri).resolve())
        texts = model.transcribe(
            processor=processor,
            audio_files=[audio_path],
            **self._resolve_transcribe_kwargs(),
        )

        if isinstance(texts, (list, tuple)):
            raw = texts[0] if texts else ""
        else:
            raw = texts

        return CaptionResult(raw="" if raw is None else str(raw), metadata={"provider": self.name})

    def _resolve_transcribe_kwargs(self) -> dict[str, Any]:
        transcribe_kwargs = dict(self.default_transcribe_kwargs)
        for key in self.transcribe_config_keys:
            if key in self.model_config:
                transcribe_kwargs[key] = self.model_config[key]

        runtime_language = getattr(self.ctx.args, "alm_language", None)
        language_value = runtime_language if runtime_language not in (None, "") else transcribe_kwargs.get("language", "")
        language = str(language_value).strip().lower()
        if not language:
            raise RuntimeError(
                "Cohere Transcribe requires an ISO 639-1 language code via "
                "--alm_language or [cohere_transcribe_local].language (for example: zh, en, ja)."
            )

        resolved = {
            "language": language,
            "punctuation": bool(transcribe_kwargs.get("punctuation", True)),
            "compile": bool(transcribe_kwargs.get("compile", False)),
            "pipeline_detokenization": bool(transcribe_kwargs.get("pipeline_detokenization", False)),
        }

        batch_size = transcribe_kwargs.get("batch_size")
        if batch_size not in (None, ""):
            resolved["batch_size"] = int(batch_size)

        return resolved

    def _normalize_transcript_text(self, output: str) -> str:
        cleaned = strip_reasoning_sections(output)
        if not cleaned:
            return ""

        if "```" in cleaned:
            extracted = extract_code_block_content(cleaned, console=self.ctx.console)
            if extracted:
                cleaned = extracted

        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    def post_validate(self, result: CaptionResult, media: MediaContext, args) -> CaptionResult:
        try:
            result.raw = self._normalize_transcript_text(result.raw)
            if not result.raw:
                raise ValueError("EMPTY_TRANSCRIPT_OUTPUT")
            result.parsed = {
                "task_kind": self.task_contract.task_kind,
                "transcript": result.raw,
                "caption_extension": self.task_contract.default_caption_extension,
                "provider": self.name,
            }
        except Exception as exc:
            raise Exception(f"RETRY_INVALID_TRANSCRIPT: {exc}") from exc
        return result
