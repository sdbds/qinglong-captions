"""Penguin-VL Local Provider

Tencent Penguin-VL-8B: A compact VLM with LLM-based vision encoder.
Uses AutoModelForCausalLM + AutoProcessor with trust_remote_code=True.
"""

import sys
import time
from pathlib import Path
from typing import Any

from providers.base import CaptionResult, MediaContext, PromptContext
from providers.local_vlm_base import LocalVLMProvider
from providers.registry import register_provider


def _penguin_vision_attention_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    past_key_value=None,
    cache_position=None,
    cu_seqlens=None,
    **kwargs,
):
    import torch
    import torch.nn.functional as F
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv

    module = sys.modules[self.__class__.__module__]
    apply_multimodal_rotary_pos_emb = getattr(module, "apply_multimodal_rotary_pos_emb")

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    if query_states.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in self.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    dropout_p = 0.0 if not self.training else self.attention_dropout
    if cu_seqlens is None:
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=self.is_causal,
        )
    else:
        outputs = []
        cu_seqlens = cu_seqlens.tolist()
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            outputs.append(
                F.scaled_dot_product_attention(
                    query_states[:, :, start:end, :],
                    key_states[:, :, start:end, :],
                    value_states[:, :, start:end, :],
                    attn_mask=None,
                    dropout_p=dropout_p,
                    is_causal=self.is_causal,
                )
            )
        attn_output = torch.cat(outputs, dim=2)

    attn_output = attn_output.transpose(1, 2).contiguous().view(*input_shape, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def _patch_penguin_vision_attention(model) -> bool:
    try:
        vision_encoder = model.get_model().get_vision_encoder()
    except Exception:
        return False
    layers = getattr(getattr(vision_encoder, "encoder", None), "layers", None)
    if not layers:
        return False

    attn_cls = type(layers[0].self_attn)
    module = sys.modules.get(attn_cls.__module__)
    has_flash_kernel = bool(module and hasattr(module, "flash_attn_varlen_func"))
    attn_impl = getattr(getattr(model, "config", None), "_attn_implementation", None)

    if has_flash_kernel and attn_impl != "eager":
        return False
    if getattr(attn_cls, "_qinglong_sdpa_patched", False):
        return False

    attn_cls.forward = _penguin_vision_attention_forward
    attn_cls._qinglong_sdpa_patched = True
    return True


def _resolve_model_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except Exception:
        return None


def _resolve_model_dtype(model):
    dtype = getattr(model, "dtype", None)
    if dtype is not None:
        return dtype
    try:
        return next(model.parameters()).dtype
    except Exception:
        return None


def _move_processor_inputs_to_model(inputs, model):
    device = _resolve_model_device(model)
    dtype = _resolve_model_dtype(model)
    moved = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue

        tensor = value if device is None else value.to(device)
        if dtype is not None and getattr(tensor, "is_floating_point", lambda: False)():
            tensor = tensor.to(dtype=dtype)
        moved[key] = tensor
    return moved


@register_provider("penguin_vl_local")
class PenguinVLLocalProvider(LocalVLMProvider):
    """Tencent Penguin-VL Local Provider"""

    default_model_id = "tencent/Penguin-VL-8B"
    _attn_implementation = ""

    @classmethod
    def can_handle(cls, args: Any, mime: str) -> bool:
        return getattr(args, "vlm_image_model", "") == "penguin_vl_local" and mime.startswith("image")

    def _load_model(self):
        """Load Penguin-VL model and processor."""
        from transformers import AutoModelForCausalLM, AutoProcessor
        from utils.transformer_loader import resolve_device_dtype, transformerLoader

        device, dtype, attn_impl = resolve_device_dtype()
        if self._attn_implementation:
            attn_impl = self._attn_implementation
        model_id = self.model_id

        self.log(f"Loading Penguin-VL model: {model_id} (device={device}, dtype={dtype}, attn={attn_impl})", "blue")

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

        if _patch_penguin_vision_attention(model):
            self.log("Patched Penguin vision encoder to use PyTorch SDPA fallback", "yellow")

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
        inputs = _move_processor_inputs_to_model(inputs, model)

        # Generate
        output_ids = model.generate(**inputs, **gen_kwargs)

        # Decode - trim input tokens
        try:
            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, input_len:] if output_ids.shape[1] > input_len else output_ids
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
