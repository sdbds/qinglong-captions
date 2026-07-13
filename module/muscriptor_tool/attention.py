from __future__ import annotations

from types import MethodType
from typing import Any, Callable

# Mirrors muscriptor/muscriptor#7 for the pinned 0.2.1 runtime. Remove this
# instance patch once the pinned upstream release contains the same strategy.


def attention_mask_strategy(query_length: int, key_length: int) -> str:
    if query_length == 1:
        return "unmasked"
    if query_length == key_length:
        return "causal"
    return "bottom_right"


def _build_optimized_forward() -> Callable[..., Any]:
    import torch
    from torch.nn import functional as F

    def forward(self: Any, query: Any, model_state: Any | None = None):
        state = self.get_state(model_state)
        projected = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        batch, query_length, _ = projected.shape
        packed = projected.view(
            batch,
            query_length,
            3,
            self.num_heads,
            self.dim_per_head,
        )
        q, k, v = packed.unbind(dim=2)
        k, v = self._complete_kv(k, v, state)

        dtype = q.dtype
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        key_length = k.shape[2]
        strategy = attention_mask_strategy(query_length, key_length)

        if strategy == "causal":
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=True,
            )
        elif strategy == "unmasked":
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            mask = torch.ones(
                query_length,
                key_length,
                dtype=torch.bool,
                device=query.device,
            ).tril(key_length - query_length)
            attn_bias = torch.zeros(
                query_length,
                key_length,
                dtype=q.dtype,
                device=query.device,
            )
            attn_bias.masked_fill_(~mask, float("-inf"))
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=0.0,
            )

        x = x.transpose(1, 2).to(dtype).contiguous()
        x = x.view(batch, query_length, self.embed_dim)
        return self.out_proj(x)

    return forward


def optimize_muscriptor_sdpa(model: Any) -> int:
    transformer = getattr(getattr(model, "_model", None), "transformer", None)
    layers = getattr(transformer, "layers", ())
    attentions = [getattr(layer, "self_attn", None) for layer in layers]
    attentions = [attention for attention in attentions if attention is not None]
    if not attentions:
        return 0

    required = (
        "get_state",
        "_complete_kv",
        "in_proj_weight",
        "in_proj_bias",
        "num_heads",
        "dim_per_head",
        "embed_dim",
        "out_proj",
    )
    if any(not all(hasattr(attention, name) for name in required) for attention in attentions):
        return 0

    forward = _build_optimized_forward()
    optimized = 0
    for attention in attentions:
        if getattr(attention, "_qinglong_optimized_sdpa", False):
            continue
        attention.forward = MethodType(forward, attention)
        attention._qinglong_optimized_sdpa = True
        optimized += 1
    return optimized
