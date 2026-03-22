"""ACE-Step Transcriber local audio-language-model provider."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from module.caption_pipeline.postprocess import strip_reasoning_sections
from module.providers.base import CaptionResult, MediaContext, PromptContext, build_chat_text_message
from module.providers.local_alm_base import LocalALMProvider
from module.providers.registry import register_provider
from utils.parse_display import extract_code_block_content


@register_provider("acestep_transcriber_local")
class AceStepTranscriberLocalProvider(LocalALMProvider):
    default_model_id = "ACE-Step/acestep-transcriber"
    generate_config_keys = ("max_new_tokens", "do_sample", "temperature", "top_p", "top_k")
    default_generate_kwargs = {
        "max_new_tokens": 1024,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 0.0,
        "top_k": 0,
    }

    @classmethod
    def can_handle(cls, args: Any, mime: str) -> bool:
        return getattr(args, "alm_model", "") == "acestep_transcriber_local" and mime.startswith("audio")

    def _load_model(self):
        from transformers import AutoModelForCausalLM, AutoProcessor

        from utils.transformer_loader import load_pretrained_component, resolve_device_dtype

        model_id = self.model_id
        device, dtype, attn_impl = resolve_device_dtype()
        self.log(f"Loading ACE-Step Transcriber model: {model_id} (device={device}, dtype={dtype})", "blue")

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
        if device == "cuda":
            load_kwargs["device_map"] = "auto"

        model = load_pretrained_component(
            AutoModelForCausalLM,
            model_id,
            console=self.ctx.console,
            component_name="model",
            **load_kwargs,
        )
        try:
            model = model.eval()
        except Exception:
            pass
        if device != "cuda" and hasattr(model, "to"):
            model = model.to(device)

        return {"model": model, "processor": processor}

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        cached = self._get_or_load_model()
        model = cached["model"]
        processor = cached["processor"]

        conversation = []
        if prompts.system:
            conversation.append(build_chat_text_message("system", prompts.system))
        conversation.append(
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": str(Path(media.uri).resolve())}},
                    {"type": "text", "text": prompts.user},
                ],
            }
        )

        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            model_dtype = self._resolve_model_input_dtype(model)
            if model_dtype is not None:
                try:
                    inputs = inputs.to(model.device, model_dtype)
                except TypeError:
                    inputs = inputs.to(model.device)
            else:
                inputs = inputs.to(model.device)
        inputs = self._move_inputs_to_model(inputs, model)

        generate_kwargs = self._apply_generate_length_cap(inputs, model, self._resolve_generate_kwargs())
        output_ids = model.generate(**inputs, **generate_kwargs)

        input_ids = inputs["input_ids"]
        generated_ids = self._extract_generated_ids(output_ids, input_ids)
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        return CaptionResult(raw=decoded[0] if decoded else "", metadata={"provider": self.name})

    def _resolve_generate_kwargs(self) -> dict[str, Any]:
        generate_kwargs = dict(self.default_generate_kwargs)
        for key in self.generate_config_keys:
            if key in self.model_config:
                generate_kwargs[key] = self.model_config[key]

        config_generate_kwargs = self.model_config.get("generate_kwargs", {})
        if isinstance(config_generate_kwargs, dict):
            for key in self.generate_config_keys:
                if key not in self.model_config and key in config_generate_kwargs:
                    generate_kwargs[key] = config_generate_kwargs[key]

        if "max_new_tokens" in generate_kwargs:
            generate_kwargs["max_new_tokens"] = int(generate_kwargs["max_new_tokens"])
        if "top_k" in generate_kwargs:
            generate_kwargs["top_k"] = int(generate_kwargs["top_k"])

        return generate_kwargs

    def _apply_generate_length_cap(
        self,
        inputs: Mapping[str, Any] | dict[str, Any],
        model: Any,
        generate_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        input_length = self._sequence_length(input_ids)
        model_max_length = self._resolve_model_max_length(model)
        if input_length is None or model_max_length is None:
            return generate_kwargs

        requested_max_new_tokens = generate_kwargs.get("max_new_tokens")
        if requested_max_new_tokens is None:
            return generate_kwargs

        available_new_tokens = model_max_length - input_length
        if available_new_tokens < 1:
            available_new_tokens = 1

        if int(requested_max_new_tokens) <= available_new_tokens:
            return generate_kwargs

        capped_kwargs = dict(generate_kwargs)
        capped_kwargs["max_new_tokens"] = available_new_tokens
        self.log(
            f"Capping max_new_tokens from {requested_max_new_tokens} to {available_new_tokens} to fit the model context window ({model_max_length}) with input length {input_length}.",
            "yellow",
        )
        return capped_kwargs

    @staticmethod
    def _sequence_length(input_ids: Any) -> int | None:
        shape = getattr(input_ids, "shape", None)
        if not shape:
            return None
        try:
            return int(shape[-1])
        except (TypeError, ValueError, IndexError):
            return None

    @staticmethod
    def _resolve_model_max_length(model: Any) -> int | None:
        candidate_objects = [getattr(model, "config", None), getattr(getattr(model, "language_model", None), "config", None)]
        for obj in candidate_objects:
            if obj is None:
                continue
            for value in (
                getattr(obj, "max_position_embeddings", None),
                getattr(getattr(obj, "text_config", None), "max_position_embeddings", None),
                getattr(obj, "max_length", None),
                getattr(getattr(obj, "text_config", None), "max_length", None),
            ):
                if value is None:
                    continue
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _extract_generated_ids(output_ids: Any, input_ids: Any) -> Any:
        input_length = AceStepTranscriberLocalProvider._sequence_length(input_ids)
        output_length = AceStepTranscriberLocalProvider._sequence_length(output_ids)
        if input_length is None or output_length is None:
            return output_ids
        if output_length <= input_length:
            return output_ids
        return output_ids[:, input_length:]

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
                "description": result.raw,
                "caption_extension": ".txt",
                "provider": self.name,
            }
        except Exception as exc:
            raise Exception(f"RETRY_INVALID_TRANSCRIPT: {exc}") from exc
        return result
