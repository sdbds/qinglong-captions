from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


class FlashAttentionBackendError(RuntimeError):
    """Raised only for failures attributable to the optional FA2 path."""


class SDPoseFlashAttnProcessor:
    def __init__(
        self,
        *,
        flash_func: Callable[..., torch.Tensor] | None = None,
    ) -> None:
        if flash_func is None:
            try:
                from flash_attn import flash_attn_func
            except Exception as exc:
                raise FlashAttentionBackendError("flash_attn could not be imported") from exc
            flash_func = flash_attn_func
            self._requires_cuda_half = True
        else:
            self._requires_cuda_half = False
        self._flash_func = flash_func
        self.saw_self_attention = False
        self.saw_cross_attention = False

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *_: object,
        **__: object,
    ) -> torch.Tensor:
        if attention_mask is not None:
            raise FlashAttentionBackendError("SDPose FA2 does not support an attention mask")
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        elif input_ndim != 3:
            raise FlashAttentionBackendError("SDPose FA2 expects rank-3 or rank-4 hidden states")

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        is_cross = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        batch_size = query.shape[0]
        inner_dim = key.shape[-1]
        if inner_dim % attn.heads:
            raise FlashAttentionBackendError("SDPose FA2 attention width is not divisible by its head count")
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query.transpose(1, 2)).transpose(1, 2)
        if attn.norm_k is not None:
            key = attn.norm_k(key.transpose(1, 2)).transpose(1, 2)
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        if self._requires_cuda_half and (not query.is_cuda or query.dtype not in {torch.float16, torch.bfloat16}):
            raise FlashAttentionBackendError("SDPose FA2 requires CUDA FP16 or BF16 query/key/value tensors")

        try:
            output = self._flash_func(
                query,
                key,
                value,
                dropout_p=0.0,
                softmax_scale=attn.scale,
                causal=False,
            )
        except Exception as exc:
            raise FlashAttentionBackendError(f"SDPose FA2 kernel failed: {exc}") from exc
        if output.shape != query.shape or not bool(torch.isfinite(output).all()):
            raise FlashAttentionBackendError("SDPose FA2 returned a wrong-shaped or non-finite tensor")
        self.saw_cross_attention |= is_cross
        self.saw_self_attention |= not is_cross

        hidden_states = output.reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size,
                channel,
                height,
                width,
            )
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


__all__ = ["FlashAttentionBackendError", "SDPoseFlashAttnProcessor"]
