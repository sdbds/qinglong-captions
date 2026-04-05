"""
LocalALMProvider - 本地音频语言模型 Provider 基类

首期用于承载本地 Hugging Face 音频理解/描述模型。
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict

from .base import MediaContext, MediaModality, Provider, ProviderType
from .capabilities import ProviderCapabilities
from .catalog import provider_config_sections

_global_model_cache: Dict[str, Any] = {}
_global_cache_lock = threading.Lock()


@dataclass(frozen=True)
class ALMTaskContract:
    task_kind: str = "caption"
    consumes_prompts: bool = True
    requires_language: bool = False
    default_caption_extension: str = ".md"


class LocalALMProvider(Provider):
    provider_type = ProviderType.LOCAL_ALM
    capabilities = ProviderCapabilities(
        supports_streaming=False,
        supports_audio=True,
    )

    default_model_id: ClassVar[str] = ""
    _supports_flex_attn: ClassVar[bool] = False
    task_contract: ClassVar[ALMTaskContract] = ALMTaskContract()

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

    def _resolve_device_dtype(self):
        try:
            from utils.transformer_loader import resolve_device_dtype

            try:
                return resolve_device_dtype(supports_flex_attn=bool(getattr(self, "_supports_flex_attn", False)))
            except TypeError:
                return resolve_device_dtype()
        except ImportError:
            import torch

            if torch.cuda.is_available():
                return "cuda", torch.float16, "eager"
            return "cpu", torch.float32, "eager"

    @staticmethod
    def _coerce_dtype(value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        normalized = value.strip().removeprefix("torch.")
        torch_module = sys.modules.get("torch")
        if torch_module is None:
            return normalized
        return getattr(torch_module, normalized, normalized)

    def _resolve_model_input_dtype(self, model: Any) -> Any:
        for attr in ("audio_tower", "audio_encoder"):
            module = getattr(model, attr, None)
            if module is None:
                continue
            try:
                first_param = next(module.parameters())
            except Exception:
                first_param = None
            if first_param is None:
                continue
            dtype = self._coerce_dtype(getattr(first_param, "dtype", None))
            if dtype is not None:
                return dtype

        config = getattr(model, "config", None)
        if config is not None:
            for attr in ("torch_dtype", "dtype"):
                dtype = self._coerce_dtype(getattr(config, attr, None))
                if dtype is not None:
                    return dtype

        return self._coerce_dtype(getattr(model, "dtype", None))

    def _move_inputs_to_model(self, inputs: Any, model: Any) -> Any:
        device = getattr(model, "device", None)
        dtype = self._resolve_model_input_dtype(model)
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
