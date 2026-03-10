"""
LocalLLMProvider - 本地文本生成 Provider 基类

和 LocalVLM 不同，这里只处理纯文本 prompt -> 纯文本输出。
不要把媒体上下文和 PromptResolver 那套东西硬塞进来。
"""

from __future__ import annotations

import gc
import threading
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Optional, Tuple

from rich.console import Console

_global_model_cache: Dict[str, Tuple[Any, Any]] = {}
_global_cache_lock = threading.Lock()


class LocalLLMProvider(ABC):
    """本地 LLM Provider 基类。"""

    default_model_id: ClassVar[str] = ""

    def __init__(
        self,
        *,
        model_id: str = "",
        console: Optional[Console] = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        trust_remote_code: bool = True,
    ) -> None:
        self.console = console
        self.model_id = model_id or self.default_model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.trust_remote_code = trust_remote_code
        self._model_key = f"{type(self).__name__}:{self.model_id}"

    def _resolve_device_dtype(self) -> tuple[str, Any, str]:
        from utils.transformer_loader import resolve_device_dtype

        return resolve_device_dtype()

    def _load_components(self) -> Tuple[Any, Any]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device, dtype, attn_impl = self._resolve_device_dtype()
        if self.console:
            self.console.print(f"[green]Loading text model:[/green] {self.model_id} ({device}, {dtype})")

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
            "torch_dtype": dtype,
        }
        if attn_impl != "eager":
            model_kwargs["attn_implementation"] = attn_impl
        model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        try:
            model = model.eval()
        except Exception:
            pass
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer, model

    def _get_or_load_components(self) -> Tuple[Any, Any]:
        with _global_cache_lock:
            cached = _global_model_cache.get(self._model_key)
            if cached is not None:
                return cached
            components = self._load_components()
            _global_model_cache[self._model_key] = components
            return components

    def generate_text(self, prompt: str) -> str:
        tokenizer, model = self._get_or_load_components()
        inputs = tokenizer(prompt, return_tensors="pt")
        model_device = next(model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
        output_ids = model.generate(**inputs, **generation_kwargs)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    @classmethod
    def unload_model(cls, model_id: Optional[str] = None) -> None:
        with _global_cache_lock:
            if model_id:
                _global_model_cache.pop(f"{cls.__name__}:{model_id}", None)
            else:
                keys_to_remove = [key for key in _global_model_cache if key.startswith(f"{cls.__name__}:")]
                for key in keys_to_remove:
                    del _global_model_cache[key]
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @abstractmethod
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        *,
        context: str = "",
        glossary: str = "",
    ) -> str:
        """把一段文本翻译成目标语言。"""

