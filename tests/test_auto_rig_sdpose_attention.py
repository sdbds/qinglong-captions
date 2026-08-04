from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

from module.auto_rig.pose.attention import (
    FlashAttentionBackendError,
    SDPoseFlashAttnProcessor,
)


def _sdpa_in_flash_layout(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    **_: object,
) -> torch.Tensor:
    output = functional.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    )
    return output.transpose(1, 2)


@pytest.mark.parametrize("cross_attention", [False, True])
def test_flash_processor_matches_diffusers_sdpa_for_self_and_cross_attention(
    cross_attention: bool,
) -> None:
    torch.manual_seed(7)
    attention = Attention(
        query_dim=32,
        cross_attention_dim=24 if cross_attention else None,
        heads=4,
        dim_head=8,
        dropout=0.0,
    ).eval()
    hidden = torch.randn((2, 11, 32))
    encoder = torch.randn((2, 3, 24)) if cross_attention else None

    expected = AttnProcessor2_0()(attention, hidden, encoder)
    processor = SDPoseFlashAttnProcessor(flash_func=_sdpa_in_flash_layout)
    actual = processor(attention, hidden, encoder)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert processor.saw_cross_attention is cross_attention
    assert processor.saw_self_attention is (not cross_attention)


def test_flash_processor_rejects_attention_masks_instead_of_ignoring_them() -> None:
    attention = Attention(query_dim=32, heads=4, dim_head=8).eval()
    processor = SDPoseFlashAttnProcessor(flash_func=_sdpa_in_flash_layout)

    with pytest.raises(FlashAttentionBackendError, match="mask"):
        processor(
            attention,
            torch.randn((1, 5, 32)),
            attention_mask=torch.ones((1, 5)),
        )


@pytest.mark.gpu
@pytest.mark.optional_runtime
def test_installed_flash_attention_kernel_runs_on_the_current_cuda_stack() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("flash_attn", exc_type=ImportError)
    torch.manual_seed(11)
    attention = Attention(query_dim=64, heads=4, dim_head=16, dropout=0.0).to(
        device="cuda",
        dtype=torch.float16,
    )
    hidden = torch.randn((1, 32, 64), device="cuda", dtype=torch.float16)
    expected = AttnProcessor2_0()(attention, hidden)
    processor = SDPoseFlashAttnProcessor()

    actual = processor(attention, hidden)

    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-3, rtol=2e-3)
    assert processor.saw_self_attention is True
