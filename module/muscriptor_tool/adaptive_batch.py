from __future__ import annotations

import gc
import warnings
from copy import copy
from typing import Any, Callable, Iterator

from . import MUSCRIPTOR_VERSION
from .batch_profiles import reserve_bytes_for_total
from .gpu_memory import SharedMemoryMonitor


def _gib(value: int) -> str:
    return f"{value / (1024**3):.2f} GiB"


class AdaptiveBatchModelProxy:
    """Add profile-seeded CUDA batching without replacing the official WebUI."""

    def __init__(
        self,
        model: Any,
        batch_size: int | None,
        *,
        console: Any | None = None,
        package_version: str = MUSCRIPTOR_VERSION,
        initial_auto_batch_size: int | None = None,
        shared_memory_monitor: SharedMemoryMonitor | None = None,
    ):
        if initial_auto_batch_size is not None and int(initial_auto_batch_size) <= 0:
            raise ValueError("initial auto batch size must be positive")
        self._model = model
        self._batch_size = batch_size
        self._console = console
        self._package_version = package_version
        initial_batch_size = batch_size if batch_size is not None else initial_auto_batch_size
        self._resolved_auto_batch_size = (
            int(initial_batch_size) if initial_batch_size is not None else None
        )
        self._allocator_limit_configured = False
        self._shared_memory_monitor = shared_memory_monitor or SharedMemoryMonitor()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def _uses_cuda(self) -> bool:
        device = getattr(self._model, "_device", None)
        return getattr(device, "type", str(device).split(":", 1)[0]) == "cuda"

    def transcribe(self, *args: Any, **kwargs: Any):
        if not self._uses_cuda():
            kwargs["batch_size"] = self._batch_size or 1
            kwargs["prelude_forcing"] = kwargs["batch_size"] == 1
            return self._model.transcribe(*args, **kwargs)

        if self._resolved_auto_batch_size is None:
            raise RuntimeError(
                "CUDA auto batch size must come from the recorded model profile"
            )

        size = self._resolved_auto_batch_size
        kwargs["prelude_forcing"] = size == 1
        if size == 1 or self._package_version != MUSCRIPTOR_VERSION or not hasattr(
            self._model, "_generate_token_stream"
        ):
            kwargs["batch_size"] = size
            if size == 1:
                self._configure_allocator_limit()
            events = self._model.transcribe(*args, **kwargs)
        else:
            events = self._transcribe_cuda_auto(*args, **kwargs)
        return self._with_cuda_cleanup(events)

    def transcribe_and_postprocess(self, *args: Any, **kwargs: Any):
        # The blocking WebUI endpoint must call this proxy's transcribe too.
        return type(self._model).transcribe_and_postprocess(self, *args, **kwargs)

    def detect_beat_grid_for(self, audio: Any, mode: bool | str = "best-effort"):
        if mode is False:
            return None
        reader = self._model
        if self._uses_cuda():
            import torch

            # Upstream's audio loader uses _device, but Beat This! runs on CPU.
            # A shallow wrapper shares the weights without changing their device.
            reader = copy(self._model)
            reader._device = torch.device("cpu")
            if isinstance(audio, tuple):
                waveform, sample_rate = audio
                audio = (waveform.cpu(), sample_rate)
        try:
            return reader.detect_beat_grid_for(audio, mode)
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            # Beat This! reports an unavailable checkpoint as ValueError, not
            # the BeatDetectionError the upstream CLI and HTTP API handle.
            if mode == "best-effort":
                warnings.warn(f"Beat detection unavailable: {exc}", RuntimeWarning, stacklevel=2)
                return None
            from muscriptor.utils.beats import BeatDetectionError

            if isinstance(exc, BeatDetectionError):
                raise
            raise BeatDetectionError(str(exc)) from exc

    def _with_cuda_cleanup(self, events: Iterator[Any]) -> Iterator[Any]:
        try:
            yield from events
        finally:
            close = getattr(events, "close", None)
            try:
                if callable(close):
                    close()
            finally:
                self._release_cuda_cache()

    def _release_cuda_cache(self) -> None:
        import torch

        device = self._model._device
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        gc.collect()
        try:
            torch.cuda.empty_cache()
            allocated = int(torch.cuda.memory_allocated(device))
            reserved = int(torch.cuda.memory_reserved(device))
        except Exception:
            return
        if self._console is not None:
            self._console.print(
                "[green]MuScriptor CUDA cache released:[/green] "
                f"allocated {_gib(allocated)}, reserved {_gib(reserved)} "
                "(model remains loaded)"
            )

    def _transcribe_cuda_auto(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        original = self._model._generate_token_stream

        def adaptive_stream(*stream_args: Any, **stream_kwargs: Any):
            yield from self._adaptive_token_stream(original, *stream_args, **stream_kwargs)

        self._model._generate_token_stream = adaptive_stream
        kwargs["batch_size"] = 1  # The adaptive stream ignores this placeholder.
        try:
            yield from self._model.transcribe(*args, **kwargs)
        finally:
            self._model._generate_token_stream = original

    def _adaptive_token_stream(
        self,
        generate: Callable[..., Iterator[Any]],
        all_conditions: list[Any],
        seek_times: list[float],
        batch_size: int,
        *generation_args: Any,
    ) -> Iterator[Any]:
        if not seek_times:
            return

        self._configure_allocator_limit()

        batch_start = 0

        while batch_start < len(seek_times):
            requested = min(
                self._resolved_auto_batch_size or 1,
                len(seek_times) - batch_start,
            )
            items, actual_size, had_oom, used_shared_memory = self._run_with_oom_fallback(
                generate,
                all_conditions,
                seek_times,
                batch_start,
                requested,
                generation_args,
            )
            if had_oom:
                self._resolved_auto_batch_size = min(
                    self._resolved_auto_batch_size or actual_size,
                    actual_size,
                )
            if used_shared_memory:
                current_size = self._resolved_auto_batch_size or actual_size
                reduced = max(1, current_size - 2)
                self._resolved_auto_batch_size = min(
                    self._resolved_auto_batch_size or reduced,
                    reduced,
                )
                if self._console is not None:
                    self._console.print(
                        "[yellow]MuScriptor shared GPU memory detected:[/yellow] "
                        f"reducing batch size {current_size} to {reduced}"
                    )
                gc.collect()
                import torch

                torch.cuda.empty_cache()
            yield from items
            batch_start += actual_size

    def _configure_allocator_limit(self) -> None:
        if self._allocator_limit_configured:
            return

        import torch

        device = self._model._device
        free_memory, total_memory = torch.cuda.mem_get_info(device)
        process_reserved = int(torch.cuda.memory_reserved(device))
        reserve_bytes = reserve_bytes_for_total(int(total_memory))
        external_usage = max(
            0,
            int(total_memory) - int(free_memory) - process_reserved,
        )
        process_limit = max(
            process_reserved,
            int(total_memory) - reserve_bytes - external_usage,
        )
        torch.cuda.set_per_process_memory_fraction(
            process_limit / int(total_memory),
            device,
        )
        if self._console is not None:
            self._console.print(
                "[cyan]MuScriptor VRAM budget:[/cyan] "
                f"total {_gib(int(total_memory))}, reserve {_gib(reserve_bytes)}, "
                f"external {_gib(external_usage)}, allocator limit {_gib(process_limit)}"
            )
        self._allocator_limit_configured = True

    def _run_with_oom_fallback(
        self,
        generate: Callable[..., Iterator[Any]],
        all_conditions: list[Any],
        seek_times: list[float],
        batch_start: int,
        requested_size: int,
        generation_args: tuple[Any, ...],
    ) -> tuple[list[Any], int, bool, bool]:
        import torch

        size = requested_size
        had_oom = False
        while True:
            shared_before = self._shared_memory_monitor.sample()
            try:
                items = self._generate_batch_items(
                    generate,
                    all_conditions,
                    seek_times,
                    batch_start,
                    size,
                    *generation_args,
                )
                torch.cuda.synchronize(self._model._device)
                shared_after = self._shared_memory_monitor.sample()
                used_shared_memory = self._shared_memory_monitor.grew_into_shared_memory(
                    shared_before,
                    shared_after,
                )
                return items, size, had_oom, used_shared_memory
            except torch.cuda.OutOfMemoryError:
                if size <= 1:
                    raise
                had_oom = True
                smaller = max(1, size - 2)
                if self._console is not None:
                    self._console.print(
                        "[yellow]MuScriptor CUDA batch OOM:[/yellow] "
                        f"retrying batch size {size} as {smaller}"
                    )
                gc.collect()
                torch.cuda.empty_cache()
                size = smaller

    @staticmethod
    def _generate_batch_items(
        generate: Callable[..., Iterator[Any]],
        all_conditions: list[Any],
        seek_times: list[float],
        batch_start: int,
        batch_size: int,
        *generation_args: Any,
    ) -> list[Any]:
        from muscriptor.events import ChunkBoundary, ProgressEvent

        end = batch_start + batch_size
        items = []
        for item in generate(
            all_conditions[batch_start:end], seek_times[batch_start:end], batch_size, *generation_args
        ):
            # Upstream sees one slice. Restore whole-song progress and the boundary
            # that keeps notes from leaking into the next slice.
            if isinstance(item, ProgressEvent):
                item = ProgressEvent(batch_start + item.completed, len(seek_times))
            elif isinstance(item, ChunkBoundary) and item.next_seek_time is None and end < len(seek_times):
                item = ChunkBoundary(item.seek_time, seek_times[end])
            items.append(item)
        return items
