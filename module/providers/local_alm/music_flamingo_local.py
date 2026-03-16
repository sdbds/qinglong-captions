"""Music Flamingo local audio-language-model provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from providers.base import CaptionResult, MediaContext, PromptContext
from module.caption_pipeline.postprocess import normalize_and_validate_subtitle_text
from providers.local_alm_base import LocalALMProvider
from providers.registry import register_provider


@register_provider("music_flamingo_local")
class MusicFlamingoLocalProvider(LocalALMProvider):
    default_model_id = "nvidia/music-flamingo-think-2601-hf"
    generate_config_keys = ("max_new_tokens", "do_sample", "temperature", "top_p")
    default_generate_kwargs = {
        "max_new_tokens": 2048,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    @classmethod
    def can_handle(cls, args: Any, mime: str) -> bool:
        return getattr(args, "alm_model", "") == "music_flamingo_local" and mime.startswith("audio")

    def _load_model(self):
        from transformers import AutoProcessor, MusicFlamingoForConditionalGeneration

        from utils.transformer_loader import resolve_device_dtype

        model_id = self.model_id
        device, dtype, _attn_impl = resolve_device_dtype()
        self.log(f"Loading Music Flamingo model: {model_id} (device={device}, dtype={dtype})", "blue")

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        if device == "cuda":
            load_kwargs["device_map"] = "auto"

        model = MusicFlamingoForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
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
            conversation.append({"role": "system", "content": prompts.system})
        conversation.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompts.user},
                    {"type": "audio", "path": str(Path(media.uri).resolve())},
                ],
            }
        )

        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device)
        inputs = self._move_inputs_to_model(inputs, model)

        output_ids = model.generate(**inputs, **self._resolve_generate_kwargs())

        input_ids = inputs["input_ids"]
        generated_ids = output_ids[:, input_ids.shape[1] :]
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

        return generate_kwargs

    def post_validate(self, result: CaptionResult, media: MediaContext, args) -> CaptionResult:
        try:
            result.raw = normalize_and_validate_subtitle_text(result.raw, self.ctx.console)
        except Exception as exc:
            raise Exception(f"RETRY_INVALID_SRT: {exc}") from exc
        return result
