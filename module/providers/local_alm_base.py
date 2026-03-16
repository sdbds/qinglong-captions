"""
LocalALMProvider - 本地音频语言模型 Provider 基类

首期用于承载本地 Hugging Face 音频理解/描述模型。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, ClassVar, Dict

from .base import MediaContext, MediaModality, Provider, ProviderType
from .capabilities import ProviderCapabilities
from .catalog import provider_config_sections

_global_model_cache: Dict[str, Any] = {}
_global_cache_lock = threading.Lock()


class LocalALMProvider(Provider):
    provider_type = ProviderType.LOCAL_ALM
    capabilities = ProviderCapabilities(
        supports_streaming=False,
        supports_audio=True,
    )

    default_model_id: ClassVar[str] = ""

    @property
    def model_id(self) -> str:
        for section_name in provider_config_sections(self.name):
            section = self.ctx.config.get(section_name, {})
            if "model_id" in section:
                return section["model_id"]
        return self.default_model_id

    @property
    def model_config(self) -> Dict[str, Any]:
        for section_name in provider_config_sections(self.name):
            section = self.ctx.config.get(section_name, {})
            if section:
                return section
        return {}

    def prepare_media(self, uri: str, mime: str, args: Any) -> MediaContext:
        file_path = Path(uri)
        file_size = file_path.stat().st_size if file_path.exists() else 0
        duration_ms = 0
        if hasattr(args, "effective_segment_time"):
            duration_ms = int(getattr(args, "effective_segment_time") or 0) * 1000

        return MediaContext(
            uri=uri,
            mime=mime,
            sha256hash="",
            modality=MediaModality.AUDIO if mime.startswith("audio") else MediaModality.UNKNOWN,
            file_size=file_size,
            duration_ms=duration_ms,
        )

    def _get_or_load_model(self):
        model_key = f"{self.name}:{self.model_id}"
        with _global_cache_lock:
            if model_key in _global_model_cache:
                return _global_model_cache[model_key]

            loaded = self._load_model()
            _global_model_cache[model_key] = loaded
            return loaded

    def _load_model(self):
        raise NotImplementedError(f"{self.name} must implement _load_model()")

    def _move_inputs_to_model(self, inputs: Any, model: Any) -> Any:
        device = getattr(model, "device", None)
        dtype = getattr(model, "dtype", None)
        if device is None or not isinstance(inputs, dict):
            return inputs

        moved = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                moved[key] = value
                continue
            tensor = value.to(device)
            if key == "input_features" and dtype is not None:
                tensor = tensor.to(dtype=dtype)
            moved[key] = tensor
        return moved
