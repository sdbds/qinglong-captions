from __future__ import annotations

import sys
import time
from typing import Any, Optional

import torch
from transformers import AutoModel, AutoProcessor


class BufferedTextStreamer:
    """Custom text streamer that buffers tokens and outputs them in batches.

    Outputs text when encountering sentence endings (. or 。) or when buffer exceeds min_chars.
    """

    def __init__(
        self,
        tokenizer: Any,
        skip_prompt: bool = True,
        skip_special_tokens: bool = True,
        min_chars: int = 0,
    ):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.min_chars = min_chars
        self.buffer: list[str] = []
        self.token_count = 0
        self.is_prompt = True

    def put(self, value: Any) -> None:
        """Process a new token batch."""
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("BufferedTextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.is_prompt:
            self.is_prompt = False
            return

        text = self.tokenizer.decode(value, skip_special_tokens=self.skip_special_tokens)
        if not text:
            return

        self.buffer.append(text)
        self.token_count += 1

        # Flush if buffer contains sentence ending (. or 。) or exceeds min_chars
        current_text = "".join(self.buffer)
        has_sentence_ending = "." in current_text or "。" in current_text
        exceeds_min_chars = self.min_chars > 0 and len(current_text) >= self.min_chars

        if has_sentence_ending or exceeds_min_chars:
            self._flush()

    def _flush(self) -> None:
        """Output buffered text and reset buffer."""
        if not self.buffer:
            return

        text = "".join(self.buffer)
        print(text)  # Print with newline
        sys.stdout.flush()
        self.buffer.clear()

    def end(self) -> None:
        """Flush any remaining buffered text at the end of generation."""
        self._flush()


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
        use_fast: Optional[bool] = None,
        trust_remote_code: bool = True,
    ) -> Any:
        # Include use_fast in cache key to differentiate configurations
        key = f"{processor_cls.__name__}:{processor_id}:fast={use_fast}"
        if key in self._processor_cache:
            return self._processor_cache[key]
        if console:
            console.print(f"[green]Loading processor:[/green] {processor_id}")
        kwargs: dict[str, Any] = {}
        if use_fast is not None:
            kwargs["use_fast"] = use_fast
        kwargs["trust_remote_code"] = trust_remote_code
        processor = processor_cls.from_pretrained(processor_id, **kwargs)
        self._processor_cache[key] = processor
        return processor

    def is_processor_cached(self, processor_id: str, processor_cls: Any = AutoProcessor, use_fast: Optional[bool] = None) -> bool:
        key = f"{processor_cls.__name__}:{processor_id}:fast={use_fast}"
        return key in self._processor_cache

    def get_cached_processor(
        self, processor_id: str, processor_cls: Any = AutoProcessor, use_fast: Optional[bool] = None
    ) -> Optional[Any]:
        key = f"{processor_cls.__name__}:{processor_id}:fast={use_fast}"
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
        use_fast: Optional[bool] = None,
        trust_remote_code: bool = True,
    ) -> Any:
        key = f"{processor_cls.__name__}:{processor_id}:fast={use_fast}"
        if key in self._processor_cache:
            if console:
                console.print(f"[yellow]Using cached processor:[/yellow] {processor_id}")
            return self._processor_cache[key]
        return self.load_processor(
            processor_id, processor_cls, console=console, use_fast=use_fast, trust_remote_code=trust_remote_code
        )

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
                    import base64
                    import io

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

    def get_text_streamer(
        self,
        processor: Any,
        skip_prompt: bool = True,
        skip_special_tokens: bool = True,
        buffered: bool = True,
        min_chars: int = 0,
    ) -> Any:
        """Create a TextStreamer for real-time token generation output.

        Args:
            processor: The processor containing the tokenizer
            skip_prompt: Whether to skip the prompt tokens in output
            skip_special_tokens: Whether to skip special tokens in output
            buffered: Use BufferedTextStreamer (outputs in batches) vs TextStreamer (per token)
            min_chars: Minimum characters to buffer before output (0 = only flush on sentence endings)

        Returns:
            BufferedTextStreamer or TextStreamer instance for use with model.generate()
        """
        if buffered:
            return BufferedTextStreamer(
                processor.tokenizer,
                skip_prompt=skip_prompt,
                skip_special_tokens=skip_special_tokens,
                min_chars=min_chars,
            )
        else:
            from transformers import TextStreamer

            return TextStreamer(
                processor.tokenizer,
                skip_prompt=skip_prompt,
                skip_special_tokens=skip_special_tokens,
            )
