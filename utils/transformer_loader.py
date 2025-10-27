from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import AutoProcessor, AutoModel


def resolve_device_dtype() -> tuple[str, torch.dtype, str]:
    if torch.cuda.is_available():
        return "cuda", getattr(torch, "bfloat16", torch.float16), "flash_attention_2"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16, "eager"
    return "cpu", torch.float32, "eager"


class transformerLoader:
    def __init__(self, attn_kw: Optional[str] = "attn_implementation", device_map: Any = "auto") -> None:
        self.attn_kw = attn_kw
        self.device_map = device_map
        self._processor_cache: dict[str, Any] = {}
        self._model_cache: dict[tuple[type[Any], str], Any] = {}

    def load_processor(
        self,
        processor_id: str,
        processor_cls: Any = AutoProcessor,
        console: Optional[Any] = None,
    ) -> Any:
        key = f"{processor_cls.__name__}:{processor_id}"
        if key in self._processor_cache:
            return self._processor_cache[key]
        if console:
            console.print(f"[green]Loading processor:[/green] {processor_id}")
        processor = processor_cls.from_pretrained(processor_id)
        self._processor_cache[key] = processor
        return processor

    def is_processor_cached(self, processor_id: str, processor_cls: Any = AutoProcessor) -> bool:
        key = f"{processor_cls.__name__}:{processor_id}"
        return key in self._processor_cache

    def get_cached_processor(self, processor_id: str, processor_cls: Any = AutoProcessor) -> Optional[Any]:
        key = f"{processor_cls.__name__}:{processor_id}"
        return self._processor_cache.get(key)

    def load_model(
        self,
        model_id: str,
        model_cls: Any = AutoModel,
        *,
        dtype: torch.dtype,
        attn_impl: Optional[str] = None,
        trust_remote_code: bool = True,
        low_cpu_mem_usage: bool = True,
        device_map: Any = None,
        use_safetensors: Optional[bool] = None,
        console: Optional[Any] = None,
        extra_kwargs: Optional[dict[str, Any]] = None,
    ) -> Any:
        key = (model_cls, model_id)
        if key in self._model_cache:
            return self._model_cache[key]
        if console:
            console.print(f"[green]Loading model:[/green] {model_id} (dtype={dtype})")
        kwargs: dict[str, Any] = {}
        kwargs["trust_remote_code"] = trust_remote_code
        kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage
        if use_safetensors is not None:
            kwargs["use_safetensors"] = use_safetensors
        kwargs["device_map"] = self.device_map if device_map is None else device_map
        kwargs["torch_dtype"] = dtype
        if attn_impl and self.attn_kw:
            kwargs[self.attn_kw] = attn_impl
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        model = model_cls.from_pretrained(model_id, **kwargs)
        try:
            model = model.eval()
        except Exception:
            pass
        self._model_cache[key] = model
        return model

    def is_model_cached(self, model_id: str, model_cls: Any = AutoModel) -> bool:
        key = (model_cls, model_id)
        return key in self._model_cache

    def get_cached_model(self, model_id: str, model_cls: Any = AutoModel) -> Optional[Any]:
        key = (model_cls, model_id)
        return self._model_cache.get(key)

    def get_or_load_processor(
        self,
        processor_id: str,
        processor_cls: Any = AutoProcessor,
        console: Optional[Any] = None,
    ) -> Any:
        key = f"{processor_cls.__name__}:{processor_id}"
        if key in self._processor_cache:
            if console:
                console.print(f"[yellow]Using cached processor:[/yellow] {processor_id}")
            return self._processor_cache[key]
        return self.load_processor(processor_id, processor_cls, console=console)

    def get_or_load_model(
        self,
        model_id: str,
        model_cls: Any = AutoModel,
        *,
        dtype: torch.dtype,
        attn_impl: Optional[str] = None,
        trust_remote_code: bool = True,
        low_cpu_mem_usage: bool = True,
        device_map: Any = None,
        use_safetensors: Optional[bool] = None,
        console: Optional[Any] = None,
        extra_kwargs: Optional[dict[str, Any]] = None,
    ) -> Any:
        key = (model_cls, model_id)
        if key in self._model_cache:
            if console:
                console.print(f"[yellow]Using cached model:[/yellow] {model_id} (dtype={dtype})")
            return self._model_cache[key]
        return self.load_model(
            model_id,
            model_cls,
            dtype=dtype,
            attn_impl=attn_impl,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
            use_safetensors=use_safetensors,
            console=console,
            extra_kwargs=extra_kwargs,
        )

    def prepare_image_inputs(
        self,
        processor: Any,
        messages: list[dict[str, Any]],
        *,
        base64_image: Optional[str] = None,
        pil_image: Optional[Any] = None,
        device: Any = "cpu",
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        padding: bool = True,
        return_tensors: str = "pt",
        return_dict: bool = False,
    ) -> Any:
        if tokenize:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_dict=True,
                return_tensors=return_tensors,
            )
        else:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            img = pil_image
            if img is None and base64_image:
                try:
                    import base64, io
                    from PIL import Image  # type: ignore
                    img = Image.open(io.BytesIO(base64.b64decode(base64_image))).convert("RGB")
                except Exception:
                    img = None
            kwargs: dict[str, Any] = {"text": [text]}
            if img is not None:
                kwargs["images"] = [img]
            inputs = processor(
                padding=padding,
                return_tensors=return_tensors,
                **kwargs,
            )
        try:
            inputs = inputs.to(device)
        except Exception:
            try:
                inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}  # type: ignore
            except Exception:
                pass
        return inputs

    def evict_model(self, model_id: str, model_cls: Any = AutoModel) -> bool:
        key = (model_cls, model_id)
        return self._model_cache.pop(key, None) is not None

    def clear_model_cache(self) -> None:
        self._model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
