"""Configuration objects for shared ONNX runtime helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class OnnxRuntimeConfig:
    execution_provider: str = "auto"
    model_cache_dir: str = ""
    session_cache_dir: str = ""
    force_download: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> "OnnxRuntimeConfig":
        data = data or {}
        return cls(
            execution_provider=str(data.get("execution_provider", "auto") or "auto"),
            model_cache_dir=str(data.get("model_cache_dir", "") or ""),
            session_cache_dir=str(data.get("session_cache_dir", "") or ""),
            force_download=bool(data.get("force_download", False)),
        )

    def resolve_model_cache_dir(self, model_id: str) -> Path:
        if self.model_cache_dir:
            return Path(self.model_cache_dir)
        return Path("huggingface") / model_id.replace("/", "_")
