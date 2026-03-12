"""Penguin-VL Local Provider

Tencent Penguin-VL-8B: A compact VLM with LLM-based vision encoder.
Uses AutoModelForCausalLM + AutoProcessor with trust_remote_code=True.
"""

import time
from pathlib import Path
from typing import Any

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.local_vlm_base import LocalVLMProvider
from providers.registry import register_provider


@register_provider("penguin_vl_local")
class PenguinVLLocalProvider(LocalVLMProvider):
    """Tencent Penguin-VL Local Provider"""

    default_model_id = "tencent/Penguin-VL-8B"
    _attn_implementation = "eager"

    @classmethod
    def can_handle(cls, args: Any, mime: str) -> bool:
        return getattr(args, "vlm_image_model", "") == "penguin_vl_local" and mime.startswith("image")

    def _load_model(self):
        """Load Penguin-VL model and processor."""
        from transformers import AutoModelForCausalLM, AutoProcessor

        from utils.transformer_loader import resolve_device_dtype, transformerLoader

        device, dtype, attn_impl = resolve_device_dtype()
        model_id = self.model_id

        self.log(f"Loading Penguin-VL model: {model_id} (device={device}, dtype={dtype})", "blue")

        loader = transformerLoader(attn_kw="attn_implementation", device_map="auto")

        processor = loader.get_or_load_processor(
            model_id,
            AutoProcessor,
            console=self.ctx.console,
            trust_remote_code=True,
        )

        model = loader.get_or_load_model(
            model_id,
            AutoModelForCausalLM,
            dtype=dtype,
            attn_impl=attn_impl,
            trust_remote_code=True,
            device_map="auto",
            console=self.ctx.console,
        )

        return {"model": model, "processor": processor, "device": device}

    def attempt(self, media: MediaContext, prompts: PromptContext) -> CaptionResult:
        if self.get_runtime_backend().is_openai:
            return self.attempt_via_openai_backend(media, prompts)

        import torch

        start_time = time.time()

        cached = self._get_or_load_model()
        model = cached["model"]
        processor = cached["processor"]

        # Build conversation in Penguin-VL format
        user_content = []

        # Handle pair images
        if media.extras.get("pair_uri"):
            pair_path = Path(media.extras["pair_uri"])
            self.log(f"Pair image: {pair_path}", "yellow")
            user_content.append({"type": "image", "image": {"image_path": str(Path(media.uri).resolve())}})
            user_content.append({"type": "image", "image": {"image_path": str(pair_path)}})
        else:
            user_content.append({"type": "image", "image": {"image_path": str(Path(media.uri).resolve())}})

        user_content.append({"type": "text", "text": prompts.user})

        conversation = [
            {"role": "system", "content": prompts.system},
            {"role": "user", "content": user_content},
        ]

        # Read generation config from config.toml
        config = self.model_config
        gen_kwargs = {
            "max_new_tokens": int(config.get("out_seq_length", 4096)),
            "do_sample": not bool(config.get("greedy", False)),
            "temperature": float(config.get("temperature", 0.7)),
            "top_p": float(config.get("top_p", 0.8)),
            "top_k": int(config.get("top_k", 20)),
            "repetition_penalty": float(config.get("repetition_penalty", 1.0)),
        }

        # Use processor to prepare inputs
        inputs = processor(conversation=conversation, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        # Generate
        output_ids = model.generate(**inputs, **gen_kwargs)

        # Decode - trim input tokens
        try:
            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, input_len:]
        except Exception:
            generated_ids = output_ids

        response_text = processor.decode(generated_ids[0], skip_special_tokens=True)

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

    def get_retry_config(self):
        cfg = super().get_retry_config()

        def classify(e):
            msg = str(e)
            if "CUDA out of memory" in msg or "OutOfMemoryError" in msg:
                return None  # Don't retry OOM
            return cfg.base_wait

        cfg.classify_error = classify
        return cfg
