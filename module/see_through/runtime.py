from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass

import torch


_FLASH_ATTN_PROBE_CACHE: tuple[bool, str] | None = None


@dataclass(frozen=True)
class RuntimeContext:
    device: str
    dtype: torch.dtype
    attention_backend: str
    reason: str


def resolve_dtype(dtype_name: str | None, *, device: str | None = None) -> torch.dtype:
    normalized = str(dtype_name or "").strip().lower()
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if normalized in {"bf16", "bfloat16"}:
        if target_device == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16 if target_device == "cuda" else torch.float32
    if normalized in {"fp32", "float32"}:
        return torch.float32
    return torch.bfloat16 if target_device == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported() else torch.float32


def probe_flash_attn_runtime() -> tuple[bool, str]:
    global _FLASH_ATTN_PROBE_CACHE
    if _FLASH_ATTN_PROBE_CACHE is not None:
        return _FLASH_ATTN_PROBE_CACHE

    if importlib.util.find_spec("flash_attn") is None:
        _FLASH_ATTN_PROBE_CACHE = (False, "flash-attn package not installed")
        return _FLASH_ATTN_PROBE_CACHE

    try:
        result = subprocess.run(
            [sys.executable, "-c", "import flash_attn"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        _FLASH_ATTN_PROBE_CACHE = (False, "flash-attn import probe timed out")
        return _FLASH_ATTN_PROBE_CACHE
    except OSError as exc:
        _FLASH_ATTN_PROBE_CACHE = (False, f"flash-attn import probe failed to start: {type(exc).__name__}: {exc}")
        return _FLASH_ATTN_PROBE_CACHE

    if result.returncode == 0:
        _FLASH_ATTN_PROBE_CACHE = (True, "flash-attn import probe succeeded")
        return _FLASH_ATTN_PROBE_CACHE

    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    detail = stderr or stdout or f"returncode={result.returncode}"
    _FLASH_ATTN_PROBE_CACHE = (False, f"flash-attn import probe failed: {detail}")
    return _FLASH_ATTN_PROBE_CACHE


def resolve_attention_backend(*, force_eager_attention: bool = False, dtype_name: str | None = None) -> RuntimeContext:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = resolve_dtype(dtype_name, device=device)

    if force_eager_attention:
        return RuntimeContext(device=device, dtype=dtype, attention_backend="eager", reason="force_eager_attention enabled")

    if device != "cuda":
        return RuntimeContext(device=device, dtype=dtype, attention_backend="eager", reason="CUDA unavailable")

    flash_attn_ok, flash_attn_reason = probe_flash_attn_runtime()
    if flash_attn_ok:
        return RuntimeContext(
            device=device,
            dtype=dtype,
            attention_backend="flash_attn",
            reason=flash_attn_reason,
        )

    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        reason = "PyTorch SDPA available"
        if sys.platform.startswith("win"):
            reason = f"{flash_attn_reason}; using PyTorch SDPA"
        return RuntimeContext(device=device, dtype=dtype, attention_backend="sdpa", reason=reason)

    return RuntimeContext(device=device, dtype=dtype, attention_backend="eager", reason="No accelerated attention backend detected")
