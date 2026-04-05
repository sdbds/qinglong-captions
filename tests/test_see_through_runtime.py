import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import module.see_through.runtime as runtime_module
from module.see_through.runtime import resolve_attention_backend, resolve_dtype


def test_resolve_attention_backend_prefers_force_eager_escape_hatch():
    context = resolve_attention_backend(force_eager_attention=True)

    assert context.attention_backend == "eager"
    assert context.reason


def test_resolve_attention_backend_uses_flash_attn_on_windows_when_available(monkeypatch):
    monkeypatch.setattr(runtime_module.sys, "platform", "win32")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True, raising=False)
    monkeypatch.setattr(runtime_module, "_FLASH_ATTN_PROBE_CACHE", None)
    monkeypatch.setattr(runtime_module, "probe_flash_attn_runtime", lambda: (True, "flash-attn import probe succeeded"))

    context = resolve_attention_backend()

    assert context.attention_backend == "flash_attn"
    assert context.reason == "flash-attn import probe succeeded"


def test_resolve_attention_backend_falls_back_to_sdpa_when_flash_attn_probe_fails_on_windows(monkeypatch):
    monkeypatch.setattr(runtime_module.sys, "platform", "win32")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True, raising=False)
    monkeypatch.setattr(runtime_module, "_FLASH_ATTN_PROBE_CACHE", None)
    monkeypatch.setattr(
        runtime_module,
        "probe_flash_attn_runtime",
        lambda: (False, "flash-attn import probe failed: returncode=3221225785"),
    )

    context = resolve_attention_backend()

    assert context.attention_backend == "sdpa"
    assert "flash-attn import probe failed" in context.reason


def test_resolve_dtype_treats_cuda_indexed_device_as_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True, raising=False)

    assert resolve_dtype("bfloat16", device="cuda:1") == torch.bfloat16
    assert resolve_dtype("float16", device="cuda:1") == torch.float16
