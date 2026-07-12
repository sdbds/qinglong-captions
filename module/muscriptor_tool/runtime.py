from __future__ import annotations

import importlib.metadata
import io
import sys
from contextlib import nullcontext, redirect_stderr
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .options import ModelVariant, TranscriptionOptions


class DeviceUnavailableError(RuntimeError):
    """Raised when an explicitly requested execution device is unavailable."""


class ModelAccessError(RuntimeError):
    """Raised when official gated weights cannot be accessed."""


@dataclass(frozen=True)
class UpstreamBindings:
    torch: Any
    model_cls: type
    progress_event_type: type
    instrument_names: tuple[str, ...]
    instrument_resolver: Callable[[Iterable[str]], list[str]]
    version: str


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    package_version: str
    requested_device: str
    resolved_device: str
    progress_event_type: type

    def transcribe(self, source: Path, options: TranscriptionOptions):
        events = iter(self.model.transcribe(audio=source, **options.upstream_kwargs()))
        try:
            while True:
                captured = io.StringIO()
                try:
                    with redirect_stderr(captured):
                        event = next(events)
                except StopIteration:
                    _relay_upstream_diagnostics(captured.getvalue(), sys.stderr)
                    return
                except BaseException:
                    _relay_upstream_diagnostics(captured.getvalue(), sys.stderr)
                    raise
                _relay_upstream_diagnostics(captured.getvalue(), sys.stderr)
                yield event
        finally:
            close = getattr(events, "close", None)
            if callable(close):
                captured = io.StringIO()
                try:
                    with redirect_stderr(captured):
                        close()
                finally:
                    _relay_upstream_diagnostics(captured.getvalue(), sys.stderr)

    def midi_bytes(self, events: Iterable[Any]) -> bytes:
        return self.model.events_to_midi_bytes(iter(events))


def _relay_upstream_diagnostics(output: str, target: Any) -> None:
    retained = [
        line
        for line in output.splitlines(keepends=True)
        if not line.lstrip().startswith("[muscriptor]")
    ]
    if not retained:
        return
    target.write("".join(retained))
    target.flush()


def _import_upstream() -> UpstreamBindings:
    import torch
    from muscriptor.events import ProgressEvent
    from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES, resolve_instrument_names
    from muscriptor.transcription_model import TranscriptionModel

    try:
        version = importlib.metadata.version("muscriptor")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return UpstreamBindings(
        torch=torch,
        model_cls=TranscriptionModel,
        progress_event_type=ProgressEvent,
        instrument_names=tuple(MT3_FULL_PLUS_GROUP_NAMES.keys()),
        instrument_resolver=resolve_instrument_names,
        version=version,
    )


def resolve_device(requested: str, torch_module: Any) -> str:
    normalized = str(requested).strip().lower()
    cuda = torch_module.cuda
    if normalized == "auto":
        if cuda.is_available() and cuda.device_count() > 0:
            return "cuda:0"
        return "cpu"
    if normalized == "cpu":
        return "cpu"
    if not cuda.is_available():
        raise DeviceUnavailableError(f"CUDA was explicitly requested but is unavailable: {normalized}")
    index = 0 if normalized == "cuda" else int(normalized.split(":", 1)[1])
    count = int(cuda.device_count())
    if index >= count:
        raise DeviceUnavailableError(f"CUDA device index {index} is invalid; available device count is {count}")
    return f"cuda:{index}"


def _http_status(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _is_model_access_error(exc: BaseException) -> bool:
    class_name = type(exc).__name__.lower()
    message = str(exc).lower()
    return (
        "gatedrepo" in class_name
        or _http_status(exc) in {401, 403}
        or "401" in message
        or "403" in message
        or "gated repo" in message
        or "gated repository" in message
    )


def _hf_download_progress(console: Any | None):
    if console is None:
        return nullcontext()

    from utils.transformer_loader import hf_download_reporting

    return hf_download_reporting(console)


def load_model(
    options: TranscriptionOptions,
    *,
    upstream: UpstreamBindings | None = None,
    console: Any | None = None,
) -> LoadedModel:
    bindings = upstream or _import_upstream()
    resolved_device = resolve_device(options.device, bindings.torch)
    repo_id = options.model.repo_id
    if console is not None:
        console.print(f"[cyan]Resolving Hugging Face model:[/cyan] {repo_id}")
    try:
        with _hf_download_progress(console):
            model = bindings.model_cls.load_model(
                weights_path=options.model.value,
                device=resolved_device,
            )
    except Exception as exc:
        if console is not None:
            console.print(f"[red]Failed to load Hugging Face model:[/red] {repo_id}")
        if not _is_model_access_error(exc):
            raise
        raise ModelAccessError(
            f"Cannot access official model https://huggingface.co/{repo_id}. "
            "Accept its Hugging Face terms, then run `hf auth login`."
        ) from exc
    if console is not None:
        console.print(f"[green]Hugging Face model ready:[/green] {repo_id}")

    actual_device = str(getattr(model, "_device", resolved_device))
    return LoadedModel(
        model=model,
        package_version=bindings.version,
        requested_device=options.device,
        resolved_device=actual_device,
        progress_event_type=bindings.progress_event_type,
    )


def list_instruments(*, upstream: UpstreamBindings | None = None) -> tuple[str, ...]:
    bindings = upstream or _import_upstream()
    return tuple(bindings.instrument_names)


def muscriptor_version(*, upstream: UpstreamBindings | None = None) -> str:
    bindings = upstream or _import_upstream()
    return str(bindings.version)


def resolve_instruments(
    values: Iterable[str],
    *,
    upstream: UpstreamBindings | None = None,
) -> tuple[str, ...]:
    bindings = upstream or _import_upstream()
    return tuple(bindings.instrument_resolver(values))
