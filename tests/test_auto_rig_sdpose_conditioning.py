from __future__ import annotations

import hashlib

import torch

from module.auto_rig.pose.conditioning import (
    SDPOSE_EMPTY_PROMPT_SHA256,
    sdpose_empty_prompt,
    sdpose_empty_prompt_provenance,
)


def test_frozen_empty_prompt_has_the_body_unet_cross_attention_contract() -> None:
    prompt = sdpose_empty_prompt()

    assert prompt.shape == (1, 2, 1024)
    assert prompt.dtype == torch.float16
    assert torch.isfinite(prompt).all()
    assert hashlib.sha256(prompt.numpy().tobytes()).hexdigest() == (
        "168b6113df66f4f1006e36bcd143aa46215a28ff60d0e769cc8a412c16c0b951"
    )
    assert SDPOSE_EMPTY_PROMPT_SHA256 == ("168b6113df66f4f1006e36bcd143aa46215a28ff60d0e769cc8a412c16c0b951")


def test_empty_prompt_returns_independent_tensors_and_pins_its_source() -> None:
    first = sdpose_empty_prompt()
    first.zero_()

    assert torch.count_nonzero(sdpose_empty_prompt()) > 0
    assert sdpose_empty_prompt_provenance() == {
        "source_commit": "14b05228cef127ce529bc0c08660770d4af3e9a8",
        "source_path": "comfy_extras/nodes_lotus.py",
        "source_repo": "https://github.com/Comfy-Org/ComfyUI",
        "tensor_dtype": "float16-le",
        "tensor_sha256": "168b6113df66f4f1006e36bcd143aa46215a28ff60d0e769cc8a412c16c0b951",
        "tensor_shape": [1, 2, 1024],
    }
