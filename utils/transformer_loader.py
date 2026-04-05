from __future__ import annotations

import importlib
import importlib.metadata
from contextlib import contextmanager, nullcontext
import importlib.util
import sys
import threading
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional

import torch
from rich.console import Console
from rich.progress import Progress

from utils.rich_progress import create_download_progress, resolve_rich_console

if TYPE_CHECKING:
    from transformers import AutoModel, AutoProcessor


def _default_auto_model():
    from transformers import AutoModel

    return AutoModel


def _default_auto_processor():
    from transformers import AutoProcessor

    return AutoProcessor


@lru_cache(maxsize=1)
def _default_console() -> Console:
    return resolve_rich_console()


def _resolve_console(console: Optional[Any]) -> Console:
    if isinstance(console, Console):
        return console
    return _default_console()


class _RichHFDownloadProgress:
    def __init__(
        self,
        progress: Progress,
        *,
        desc: str,
        total: Optional[int] = None,
        initial: int = 0,
    ) -> None:
        self._progress = progress
        self._desc = desc or "download"
        self._total = total
        self._initial = initial
        self._task_id: Optional[int] = None

    def __enter__(self) -> "_RichHFDownloadProgress":
        self._task_id = self._progress.add_task(
            f"[cyan]{self._desc}[/cyan]",
            total=self._total,
            completed=self._initial,
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._task_id is not None:
            if exc_type is None and self._total is not None:
                self._progress.update(self._task_id, completed=self._total)
            self._progress.remove_task(self._task_id)
        return False

    def update(self, advance: float) -> None:
        if self._task_id is None or not advance:
            return
        self._progress.update(self._task_id, advance=advance)


_HF_PROGRESS_PATCH_LOCK = threading.RLock()
_HF_PROGRESS_PATCH_DEPTH = 0
_HF_PROGRESS_PATCH_ORIGINAL: Any = None
_HF_PROGRESS_PATCH_PROGRESS: Optional[Progress] = None


@contextmanager
def suppress_library_progress_bars():
    """Temporarily disable noisy tqdm-based progress bars from model-loading libraries."""

    restorers: list[tuple[Any | None, bool | None]] = []
    for module_name in ("diffusers.utils.logging", "transformers.utils.logging"):
        try:
            logging_module = importlib.import_module(module_name)
        except Exception:
            continue

        disable = getattr(logging_module, "disable_progress_bar", None)
        enable = getattr(logging_module, "enable_progress_bar", None)
        is_enabled = getattr(logging_module, "is_progress_bar_enabled", None)

        if not callable(disable):
            continue

        previous_state: bool | None = None
        if callable(is_enabled):
            try:
                previous_state = bool(is_enabled())
            except Exception:
                previous_state = None

        disable()
        restorers.append((enable if callable(enable) else None, previous_state))

    try:
        yield
    finally:
        for enable, previous_state in reversed(restorers):
            if previous_state is False:
                continue
            if callable(enable):
                enable()


@contextmanager
def hf_download_reporting(console: Optional[Any] = None):
    """Render Hugging Face download progress with Rich while loading artifacts."""

    global _HF_PROGRESS_PATCH_DEPTH, _HF_PROGRESS_PATCH_ORIGINAL, _HF_PROGRESS_PATCH_PROGRESS

    try:
        from huggingface_hub import file_download
        from huggingface_hub.utils import enable_progress_bars
    except Exception:
        yield
        return

    resolved_console = _resolve_console(console)

    with _HF_PROGRESS_PATCH_LOCK:
        enable_progress_bars()
        if _HF_PROGRESS_PATCH_DEPTH == 0:
            _HF_PROGRESS_PATCH_ORIGINAL = file_download._get_progress_bar_context
            _HF_PROGRESS_PATCH_PROGRESS = create_download_progress(
                resolved_console,
                transient=False,
                expand=True,
            )
            _HF_PROGRESS_PATCH_PROGRESS.start()

            def _rich_progress_context(**kwargs: Any):
                existing_bar = kwargs.get("_tqdm_bar")
                if existing_bar is not None:
                    return nullcontext(existing_bar)

                progress = _HF_PROGRESS_PATCH_PROGRESS
                if progress is None:
                    return _HF_PROGRESS_PATCH_ORIGINAL(**kwargs)

                return _RichHFDownloadProgress(
                    progress,
                    desc=str(kwargs.get("desc") or "download"),
                    total=kwargs.get("total"),
                    initial=int(kwargs.get("initial") or 0),
                )

            file_download._get_progress_bar_context = _rich_progress_context
        _HF_PROGRESS_PATCH_DEPTH += 1

    try:
        yield
    finally:
        with _HF_PROGRESS_PATCH_LOCK:
            _HF_PROGRESS_PATCH_DEPTH -= 1
            if _HF_PROGRESS_PATCH_DEPTH == 0:
                file_download._get_progress_bar_context = _HF_PROGRESS_PATCH_ORIGINAL
                if _HF_PROGRESS_PATCH_PROGRESS is not None:
                    _HF_PROGRESS_PATCH_PROGRESS.stop()
                _HF_PROGRESS_PATCH_ORIGINAL = None
                _HF_PROGRESS_PATCH_PROGRESS = None


def load_pretrained_component(
    component_cls: Any,
    repo_id: str,
    *,
    console: Optional[Any] = None,
    component_name: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Call `from_pretrained` with Rich logging and Hugging Face download progress."""

    resolved_console = _resolve_console(console)
    label = component_name or getattr(component_cls, "__name__", "component")

    resolved_console.print(f"[cyan]Resolving Hugging Face {label}:[/cyan] {repo_id}")

    try:
        with suppress_library_progress_bars(), hf_download_reporting(resolved_console):
            loaded = component_cls.from_pretrained(repo_id, **kwargs)
    except importlib.metadata.PackageNotFoundError as exc:
        missing_pkg = str(getattr(exc, "name", "") or exc)
        if missing_pkg != "bitsandbytes":
            resolved_console.print(f"[red]Failed to load Hugging Face {label}:[/red] {repo_id}")
            raise
        resolved_console.print(f"[red]Failed to load Hugging Face {label}:[/red] {repo_id}")
        raise RuntimeError(
            "Missing dependency 'bitsandbytes'. Quantized Hugging Face checkpoints require it. "
            "If you are using see-through NF4 repos, rerun `uv sync --extra see-through`."
        ) from exc
    except Exception:
        resolved_console.print(f"[red]Failed to load Hugging Face {label}:[/red] {repo_id}")
        raise

    resolved_console.print(f"[green]Hugging Face {label} ready:[/green] {repo_id}")
    return loaded


def is_quantized_pretrained_component(component: Any) -> bool:
    return bool(
        getattr(component, "is_quantized", False)
        or getattr(component, "quantization_method", None)
        or getattr(component, "is_loaded_in_4bit", False)
        or getattr(component, "is_loaded_in_8bit", False)
    )


def _is_unsupported_quantized_move_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "bitsandbytes" in message
        or "4-bit" in message
        or "8-bit" in message
        or ("quantized" in message and ".to" in message)
        or "please use the model as it is" in message
    )


def move_pretrained_component(component: Any, *, device: Any | None = None, dtype: Any | None = None) -> Any:
    if component is None:
        return component
    move = getattr(component, "to", None)
    if not callable(move):
        return component

    quantized = is_quantized_pretrained_component(component)
    kwargs: dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None and not quantized:
        kwargs["dtype"] = dtype
    if not kwargs:
        return component

    def _resolve(result: Any) -> Any:
        return component if result is None else result

    try:
        return _resolve(move(**kwargs))
    except TypeError:
        try:
            if "device" in kwargs and "dtype" in kwargs:
                return _resolve(move(kwargs["device"], kwargs["dtype"]))
            if "device" in kwargs:
                return _resolve(move(kwargs["device"]))
            return _resolve(move(kwargs["dtype"]))
        except ValueError as exc:
            if quantized and _is_unsupported_quantized_move_error(exc):
                return component
            raise
    except ValueError as exc:
        if quantized and _is_unsupported_quantized_move_error(exc):
            return component
        raise


def _move_model_inputs_to_device(inputs: Any, *, device: Any = "cpu", dtype: Any | None = None) -> Any:
    move = getattr(inputs, "to", None)
    if callable(move):
        try:
            if dtype is not None:
                return inputs.to(device, dtype)
            return inputs.to(device)
        except TypeError:
            try:
                return inputs.to(device)
            except Exception:
                pass
        except Exception:
            pass

    if isinstance(inputs, dict):
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                moved[key] = value
                continue
            if dtype is not None and getattr(value, "is_floating_point", lambda: False)():
                try:
                    moved[key] = value.to(device=device, dtype=dtype)
                    continue
                except TypeError:
                    pass
            moved[key] = value.to(device)
        return moved

    return inputs


def prepare_multimodal_inputs(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    device: Any = "cpu",
    dtype: Any | None = None,
    add_generation_prompt: bool = True,
    return_tensors: str = "pt",
    return_dict: bool = True,
    chat_template_kwargs: Optional[dict[str, Any]] = None,
) -> Any:
    apply_kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "return_tensors": return_tensors,
        "return_dict": return_dict,
    }
    optional_kwargs = dict(chat_template_kwargs or {})
    apply_kwargs.update(optional_kwargs)

    try:
        inputs = processor.apply_chat_template(messages, **apply_kwargs)
    except TypeError:
        if not optional_kwargs:
            raise
        for key in optional_kwargs:
            apply_kwargs.pop(key, None)
        inputs = processor.apply_chat_template(messages, **apply_kwargs)

    return _move_model_inputs_to_device(inputs, device=device, dtype=dtype)


def snapshot_download_with_reporting(
    repo_id: str,
    *,
    console: Optional[Any] = None,
    **kwargs: Any,
) -> str:
    """Call `snapshot_download` with Rich logging and Hugging Face download progress."""

    from huggingface_hub import snapshot_download

    resolved_console = _resolve_console(console)
    resolved_console.print(f"[cyan]Resolving Hugging Face snapshot:[/cyan] {repo_id}")

    try:
        with hf_download_reporting(resolved_console):
            snapshot_path = snapshot_download(repo_id=repo_id, **kwargs)
    except Exception:
        resolved_console.print(f"[red]Failed to download Hugging Face snapshot:[/red] {repo_id}")
        raise

    resolved_console.print(f"[green]Hugging Face snapshot ready:[/green] {repo_id}")
    return str(snapshot_path)


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


@lru_cache(maxsize=1)
def _has_flash_attn() -> bool:
    if importlib.util.find_spec("flash_attn") is None:
        return False
    try:
        importlib.import_module("flash_attn")
    except Exception:
        return False
    return True


@lru_cache(maxsize=1)
def _has_flex_attention() -> bool:
    attention = getattr(getattr(torch, "nn", None), "attention", None)
    return hasattr(attention, "flex_attention")


@lru_cache(maxsize=1)
def _has_sdpa() -> bool:
    functional = getattr(getattr(torch, "nn", None), "functional", None)
    return hasattr(functional, "scaled_dot_product_attention")


def _is_missing_attention_backend_error(exc: BaseException, attn_impl: Optional[str]) -> bool:
    message = str(exc)
    lowered = message.lower()
    if attn_impl == "flash_attention_2":
        return "flash_attn" in message or "FlashAttention2 has been toggled on" in message
    if attn_impl == "flex_attention":
        return "flex_attention" in lowered or "torch.nn.attention.flex_attention" in lowered
    return False


def _default_cuda_attention_impl(*, supports_flex_attn: bool = False) -> str:
    if supports_flex_attn and _has_flex_attention():
        return "flex_attention"
    if _has_flash_attn():
        return "flash_attention_2"
    if _has_sdpa():
        return "sdpa"
    return "eager"


def _next_attention_fallback(attn_impl: Optional[str]) -> Optional[str]:
    if attn_impl == "flex_attention":
        if _has_flash_attn():
            return "flash_attention_2"
        return "sdpa" if _has_sdpa() else "eager"
    if attn_impl == "flash_attention_2":
        return "sdpa" if _has_sdpa() else "eager"
    if attn_impl == "sdpa":
        return "eager"
    return None


def _prefer_flex_attention(attn_impl: Optional[str], *, supports_flex_attn: bool = False) -> Optional[str]:
    if not supports_flex_attn or not _has_flex_attention():
        return attn_impl
    if attn_impl in (None, "flash_attention_2", "sdpa", "eager"):
        return "flex_attention"
    return attn_impl


def resolve_device_dtype(*, supports_flex_attn: bool = False) -> tuple[str, torch.dtype, str]:
    if torch.cuda.is_available():
        attn_impl = _default_cuda_attention_impl(supports_flex_attn=supports_flex_attn)
        return "cuda", getattr(torch, "bfloat16", torch.float16), attn_impl
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16, "eager"
    return "cpu", torch.float32, "eager"


class transformerLoader:
    def __init__(
        self,
        attn_kw: Optional[str] = "attn_implementation",
        device_map: Any = "auto",
        *,
        supports_flex_attn: bool = False,
    ) -> None:
        self.attn_kw = attn_kw
        self.device_map = device_map
        self.supports_flex_attn = supports_flex_attn
        self._processor_cache: dict[str, Any] = {}
        self._model_cache: dict[tuple[type[Any], str], Any] = {}

    def load_processor(
        self,
        processor_id: str,
        processor_cls: Any = None,
        console: Optional[Any] = None,
        use_fast: Optional[bool] = None,
        trust_remote_code: bool = True,
    ) -> Any:
        if processor_cls is None:
            processor_cls = _default_auto_processor()
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
        processor = load_pretrained_component(
            processor_cls,
            processor_id,
            console=console,
            component_name="processor",
            **kwargs,
        )
        self._processor_cache[key] = processor
        return processor

    def is_processor_cached(self, processor_id: str, processor_cls: Any = None, use_fast: Optional[bool] = None) -> bool:
        if processor_cls is None:
            processor_cls = _default_auto_processor()
        key = f"{processor_cls.__name__}:{processor_id}:fast={use_fast}"
        return key in self._processor_cache

    def get_cached_processor(
        self, processor_id: str, processor_cls: Any = None, use_fast: Optional[bool] = None
    ) -> Optional[Any]:
        if processor_cls is None:
            processor_cls = _default_auto_processor()
        key = f"{processor_cls.__name__}:{processor_id}:fast={use_fast}"
        return self._processor_cache.get(key)

    def load_model(
        self,
        model_id: str,
        model_cls: Any = None,
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
        if model_cls is None:
            model_cls = _default_auto_model()
        key = (model_cls, model_id)
        if key in self._model_cache:
            return self._model_cache[key]
        attn_impl = _prefer_flex_attention(attn_impl, supports_flex_attn=self.supports_flex_attn)
        if console:
            console.print(f"[green]Loading model:[/green] {model_id} (dtype={dtype}, attn={attn_impl or 'default'})")
        kwargs: dict[str, Any] = {}
        kwargs["trust_remote_code"] = trust_remote_code
        kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage
        if use_safetensors is not None:
            kwargs["use_safetensors"] = use_safetensors
        kwargs["device_map"] = self.device_map if device_map is None else device_map
        kwargs["torch_dtype"] = str(dtype).split(".")[-1]
        if attn_impl and self.attn_kw:
            kwargs[self.attn_kw] = attn_impl
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        attempt_kwargs = dict(kwargs)
        attempt_attn_impl = attn_impl
        while True:
            try:
                model = load_pretrained_component(
                    model_cls,
                    model_id,
                    console=console,
                    component_name="model",
                    **attempt_kwargs,
                )
                break
            except ImportError as exc:
                if not (
                    self.attn_kw
                    and attempt_attn_impl
                    and attempt_attn_impl != "eager"
                    and _is_missing_attention_backend_error(exc, attempt_attn_impl)
                ):
                    raise

                fallback_attn_impl = _next_attention_fallback(attempt_attn_impl)
                if not fallback_attn_impl or fallback_attn_impl == attempt_attn_impl:
                    raise

                attempt_kwargs = dict(kwargs)
                attempt_kwargs[self.attn_kw] = fallback_attn_impl
                if console:
                    console.print(
                        f"[yellow]flash_attn 不可用，回退到 {fallback_attn_impl} attention 继续加载[/yellow]"
                    )
                attempt_attn_impl = fallback_attn_impl
        try:
            model = model.eval()
        except Exception:
            pass
        self._model_cache[key] = model
        return model

    def is_model_cached(self, model_id: str, model_cls: Any = None) -> bool:
        if model_cls is None:
            model_cls = _default_auto_model()
        key = (model_cls, model_id)
        return key in self._model_cache

    def get_cached_model(self, model_id: str, model_cls: Any = None) -> Optional[Any]:
        if model_cls is None:
            model_cls = _default_auto_model()
        key = (model_cls, model_id)
        return self._model_cache.get(key)

    def get_or_load_processor(
        self,
        processor_id: str,
        processor_cls: Any = None,
        console: Optional[Any] = None,
        use_fast: Optional[bool] = None,
        trust_remote_code: bool = True,
    ) -> Any:
        if processor_cls is None:
            processor_cls = _default_auto_processor()
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
        model_cls: Any = None,
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
        if model_cls is None:
            model_cls = _default_auto_model()
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
        return _move_model_inputs_to_device(inputs, device=device)

    def evict_model(self, model_id: str, model_cls: Any = None) -> bool:
        if model_cls is None:
            model_cls = _default_auto_model()
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
