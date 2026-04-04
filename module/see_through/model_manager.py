from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable


Factory = Callable[..., Any]

if TYPE_CHECKING:
    from .runtime import RuntimeContext


def _get_torch() -> Any | None:
    try:
        import torch
    except Exception:
        return None
    return torch


class SeeThroughModelManager:
    def __init__(
        self,
        *,
        offload_policy: str = "delete",
        runtime_context: "RuntimeContext | None" = None,
        layerdiff_factory: Factory | None = None,
        marigold_factory: Factory | None = None,
    ) -> None:
        if offload_policy not in {"delete", "cpu"}:
            raise ValueError(f"Unsupported offload_policy: {offload_policy}")

        self.offload_policy = offload_policy
        if runtime_context is None:
            try:
                from .runtime import resolve_attention_backend

                runtime_context = resolve_attention_backend()
            except Exception:
                runtime_context = SimpleNamespace(
                    device="cpu",
                    dtype="float32",
                    attention_backend="eager",
                    reason="runtime context unavailable",
                )
        self.runtime_context = runtime_context
        self._layerdiff_factory = layerdiff_factory
        self._marigold_factory = marigold_factory
        self._layerdiff: Any = None
        self._marigold: Any = None

    def get_layerdiff_pipeline(self, **kwargs: Any) -> Any:
        if self._layerdiff is None:
            factory = self._layerdiff_factory or self._default_layerdiff_factory
            self._layerdiff = factory(runtime_context=self.runtime_context, **kwargs)
        return self._layerdiff

    def get_marigold_pipeline(self, **kwargs: Any) -> Any:
        if self._marigold is None:
            factory = self._marigold_factory or self._default_marigold_factory
            self._marigold = factory(runtime_context=self.runtime_context, **kwargs)
        return self._marigold

    def release_layerdiff(self) -> None:
        self._layerdiff = self._release_value(self._layerdiff)

    def release_marigold(self) -> None:
        self._marigold = self._release_value(self._marigold)

    def release_all(self) -> None:
        self.release_layerdiff()
        self.release_marigold()

    def log_vram(self, stage_name: str) -> dict[str, float | str]:
        torch = _get_torch()
        if torch is None or not torch.cuda.is_available():
            return {"stage": stage_name, "device": "cpu"}

        return {
            "stage": stage_name,
            "device": "cuda",
            "allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 2),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2),
        }

    @staticmethod
    def _default_layerdiff_factory(**kwargs: Any) -> Any:
        from .pipelines.layerdiff import load_layerdiff_pipeline

        return load_layerdiff_pipeline(**kwargs)

    @staticmethod
    def _default_marigold_factory(**kwargs: Any) -> Any:
        from .pipelines.marigold import load_marigold_pipeline

        return load_marigold_pipeline(**kwargs)

    def _release_value(self, value: Any) -> Any:
        if value is None:
            return None

        if self.offload_policy == "cpu":
            move = getattr(value, "to", None)
            if callable(move):
                try:
                    move("cpu")
                except TypeError:
                    move(device="cpu")
            self._cleanup_cuda_cache()
            return value

        close_fn = getattr(value, "close", None)
        if callable(close_fn):
            close_fn()
        self._cleanup_cuda_cache()
        return None

    @staticmethod
    def _cleanup_cuda_cache() -> None:
        torch = _get_torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
