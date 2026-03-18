"""Reka Edge Local Provider.

Supports direct Transformers inference and OpenAI-compatible local servers
such as vLLM with the reka plugin.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from module.providers.base import CaptionResult, MediaContext, PromptContext
from module.providers.capabilities import ProviderCapabilities
from module.providers.local_vlm_base import LocalVLMProvider
from module.providers.registry import register_provider

from utils.parse_display import extract_code_block_content


@register_provider("reka_edge_local")
class RekaEdgeLocalProvider(LocalVLMProvider):
    """Local provider for RekaAI/reka-edge-2603."""

    default_model_id = "RekaAI/reka-edge-2603"
    capabilities = ProviderCapabilities(
        supports_streaming=False,
        supports_images=True,
        supports_video=True,
    )

    _runtime_stop_sequences = ["\n\n<sep>"]

    @classmethod
    def can_handle(cls, args: Any, mime: str) -> bool:
        return getattr(args, "vlm_image_model", "") == "reka_edge_local" and mime.startswith(("image", "video"))

    def _load_model(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from utils.transformer_loader import resolve_device_dtype

        model_id = self.model_id
        device, dtype, _ = resolve_device_dtype()
        dtype = self._select_dtype(device=device, dtype=dtype, torch_module=torch)

        self.log(f"Loading Reka Edge model: {model_id} (device={device}, dtype={dtype})", "blue")

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        if device == "cuda":
            load_kwargs["device_map"] = "auto"

        model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs).eval()
        if device != "cuda":
            model = model.to(device)

        return {
            "model": model,
            "processor": processor,
            "device": device,
            "dtype": dtype,
            "torch": torch,
        }

    def _select_dtype(self, *, device: str, dtype: Any, torch_module: Any):
        requested_dtype = getattr(self.ctx.args, "dtype", None) or self.model_config.get("dtype")
        if isinstance(requested_dtype, str):
            normalized = requested_dtype.strip().lower()
            dtype_aliases = {
                "float16": torch_module.float16,
                "fp16": torch_module.float16,
                "half": torch_module.float16,
                "bfloat16": torch_module.bfloat16,
                "bf16": torch_module.bfloat16,
                "float32": torch_module.float32,
                "fp32": torch_module.float32,
            }
            if normalized in dtype_aliases:
                return dtype_aliases[normalized]

        if device == "cuda":
            return torch_module.float16
        return dtype

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        start_time = time.time()

        if self.get_runtime_backend().is_openai:
            result = self.attempt_via_openai_backend(
                media,
                prompts,
                stop=self._runtime_stop_sequences,
            )
            normalized = self._normalize_output(media, result.raw)
            result.raw = normalized
            return result

        cached = self._get_or_load_model()
        model = cached["model"]
        processor = cached["processor"]
        torch = cached["torch"]
        device = cached["device"]
        dtype = cached["dtype"]

        inputs = processor.apply_chat_template(
            self._build_messages(media, prompts),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = self._move_inputs_to_device(inputs, device=device, dtype=dtype)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.model_config.get("max_new_tokens", self.model_config.get("out_seq_length", 1024))),
            "do_sample": bool(self.model_config.get("do_sample", False)),
        }

        if generation_kwargs["do_sample"]:
            generation_kwargs["temperature"] = float(self.model_config.get("temperature", 0.2))
            generation_kwargs["top_p"] = float(self.model_config.get("top_p", 0.95))
            generation_kwargs["top_k"] = int(self.model_config.get("top_k", 0))

        repetition_penalty = self.model_config.get("repetition_penalty")
        if repetition_penalty not in (None, ""):
            generation_kwargs["repetition_penalty"] = float(repetition_penalty)

        sep_token_id = processor.tokenizer.convert_tokens_to_ids("<sep>")
        eos_token_ids = [processor.tokenizer.eos_token_id]
        if isinstance(sep_token_id, int) and sep_token_id >= 0:
            eos_token_ids.append(sep_token_id)
        generation_kwargs["eos_token_id"] = eos_token_ids

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)

        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0, input_len:]
        response_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        response_text = self._normalize_output(media, response_text)

        elapsed_time = time.time() - start_time
        self.log(f"Caption generation took: {elapsed_time:.2f} seconds", "blue")

        try:
            self.ctx.console.print(response_text)
        except Exception:
            from rich.text import Text

            self.ctx.console.print(Text(response_text))

        if self.ctx.progress and self.ctx.task_id is not None:
            self.ctx.progress.update(self.ctx.task_id, description="Processing media...")

        return CaptionResult(raw=response_text, metadata={"provider": self.name})

    def _build_messages(self, media: MediaContext, prompts: PromptContext) -> list[dict[str, Any]]:
        user_content: list[dict[str, Any]] = []

        if media.mime.startswith("video"):
            user_content.append({"type": "video", "video": str(Path(media.uri).resolve())})
        else:
            user_content.append({"type": "image", "image": str(Path(media.uri).resolve())})
            pair_uri = media.extras.get("pair_uri")
            if pair_uri:
                user_content.append({"type": "image", "image": str(Path(pair_uri).resolve())})

        user_content.append({"type": "text", "text": prompts.user})

        messages = [{"role": "user", "content": user_content}]
        if prompts.system:
            messages.insert(0, {"role": "system", "content": prompts.system})
        return messages

    def _move_inputs_to_device(self, inputs: Any, *, device: str, dtype: Any):
        try:
            moved = {}
            for key, value in inputs.items():
                if not hasattr(value, "to"):
                    moved[key] = value
                    continue
                if getattr(value, "is_floating_point", lambda: False)():
                    moved[key] = value.to(device=device, dtype=dtype)
                else:
                    moved[key] = value.to(device=device)
            return moved
        except Exception:
            return inputs

    def _normalize_output(self, media: MediaContext, response_text: str) -> str:
        cleaned = (response_text or "").replace("<sep>", "").strip()
        if media.mime.startswith("video"):
            extracted = extract_code_block_content(cleaned, "srt", self.ctx.console)
            if extracted:
                return extracted
        return cleaned

    def get_retry_config(self):
        cfg = super().get_retry_config()

        def classify(e):
            msg = str(e)
            if "CUDA out of memory" in msg or "OutOfMemoryError" in msg:
                return None
            return cfg.base_wait

        cfg.classify_error = classify
        return cfg
