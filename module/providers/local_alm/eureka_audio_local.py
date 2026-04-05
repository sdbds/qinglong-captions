"""Eureka Audio local audio-language-model provider."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from module.caption_pipeline.postprocess import strip_reasoning_sections
from module.providers.base import CaptionResult, MediaContext, PromptContext
from module.providers.local_alm_base import LocalALMProvider
from module.providers.registry import register_provider
from utils.parse_display import extract_code_block_content


@register_provider("eureka_audio_local")
class EurekaAudioLocalProvider(LocalALMProvider):
    default_model_id = "cslys1999/Eureka-Audio-Instruct"
    generate_config_keys = ("max_new_tokens", "do_sample", "temperature", "top_p", "top_k", "repetition_penalty")
    default_generate_kwargs = {
        "max_new_tokens": 512,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 0.0,
        "top_k": 0,
    }

    @classmethod
    def can_handle(cls, args: Any, mime: str) -> bool:
        return getattr(args, "alm_model", "") == "eureka_audio_local" and mime.startswith("audio")

    def _load_model(self):
        try:
            from eureka_infer.api import EurekaAudio
        except ImportError as exc:
            raise RuntimeError(
                "Eureka-Audio is not installed. Run `uv sync --extra eureka-audio-local` to install the upstream wrapper."
            ) from exc

        from utils.transformer_loader import resolve_device_dtype

        resolved_device, _, _ = resolve_device_dtype()
        device = self._resolve_runtime_device(resolved_device)
        model_path = self.model_id
        self.log(f"Loading Eureka Audio model: {model_path} (device={device})", "blue")
        return {"model": EurekaAudio(model_path=model_path, device=device)}

    def _resolve_runtime_device(self, resolved_device: str) -> str:
        configured = str(self.model_config.get("device", "")).strip()
        if configured:
            return configured
        if resolved_device.startswith("cuda"):
            return resolved_device
        return "cpu"

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        cached = self._get_or_load_model()
        model = cached["model"]
        messages = self._build_messages(media, prompts)
        response = model.generate(messages, **self._resolve_generate_kwargs())
        return CaptionResult(raw="" if response is None else str(response), metadata={"provider": self.name})

    def _build_messages(self, media: MediaContext, prompts: PromptContext) -> list[dict[str, Any]]:
        audio_path = str(Path(media.uri).resolve())
        messages: list[dict[str, Any]] = []
        if prompts.system:
            messages.append(self.build_message("system", [self.build_text_part(prompts.system)]))
        messages.append(
            self.build_message(
                "user",
                [
                    self._build_audio_url_part(audio_path),
                    self.build_text_part(prompts.user),
                ],
            )
        )
        return messages

    @staticmethod
    def _build_audio_url_part(audio_path: str) -> dict[str, Any]:
        return {
            "type": "audio_url",
            "audio_url": {
                "url": audio_path,
            },
        }

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
        if not bool(generate_kwargs.get("do_sample")):
            generate_kwargs.pop("temperature", None)
            generate_kwargs.pop("top_p", None)
            generate_kwargs.pop("top_k", None)

        return generate_kwargs

    def _normalize_summary_text(self, output: str) -> str:
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
            result.raw = self._normalize_summary_text(result.raw)
            if not result.raw:
                raise ValueError("EMPTY_SUMMARY_OUTPUT")
            result.parsed = {
                "description": result.raw,
                "caption_extension": ".txt",
                "provider": self.name,
            }
        except Exception as exc:
            raise Exception(f"RETRY_INVALID_SUMMARY: {exc}") from exc
        return result
