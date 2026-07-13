from __future__ import annotations

import gc
import warnings
from types import MethodType
from typing import Any, Iterator

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
        package_version: str = "0.2.1",
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
            return self._model.transcribe(*args, **kwargs)

        if self._resolved_auto_batch_size is None:
            raise RuntimeError(
                "CUDA auto batch size must come from the recorded model profile"
            )

        if self._package_version != "0.2.1" or not hasattr(
            self._model, "_generate_token_stream"
        ):
            kwargs["batch_size"] = self._batch_size
            events = self._model.transcribe(*args, **kwargs)
        else:
            events = self._transcribe_cuda_auto(*args, **kwargs)
        return self._with_cuda_cleanup(events)

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

        def adaptive_stream(model: Any, *stream_args: Any, **stream_kwargs: Any):
            yield from self._adaptive_token_stream(model, *stream_args, **stream_kwargs)

        self._model._generate_token_stream = MethodType(adaptive_stream, self._model)
        kwargs["batch_size"] = 1  # The adaptive stream ignores this placeholder.
        try:
            yield from self._model.transcribe(*args, **kwargs)
        finally:
            self._model._generate_token_stream = original

    def _adaptive_token_stream(
        self,
        model: Any,
        all_conditions: list[Any],
        seek_times: list[float],
        _batch_size: int,
        max_gen_len: int,
        use_sampling: bool,
        temperature: float,
        cfg_coef: float,
        no_eos_is_ok: bool,
        beam_size: int = 1,
    ) -> Iterator[Any]:
        if not seek_times:
            return

        self._configure_allocator_limit()

        generation_args = (
            max_gen_len,
            use_sampling,
            temperature,
            cfg_coef,
            no_eos_is_ok,
            beam_size,
        )
        batch_start = 0

        while batch_start < len(seek_times):
            requested = min(
                self._resolved_auto_batch_size or 1,
                len(seek_times) - batch_start,
            )
            items, actual_size, had_oom, used_shared_memory = self._run_with_oom_fallback(
                model,
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
        model: Any,
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
                    model,
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
        model: Any,
        all_conditions: list[Any],
        seek_times: list[float],
        batch_start: int,
        batch_size: int,
        max_gen_len: int,
        use_sampling: bool,
        temperature: float,
        cfg_coef: float,
        no_eos_is_ok: bool,
        beam_size: int,
    ) -> list[Any]:
        from muscriptor.events import ChunkBoundary, ProgressEvent

        eos_id = model._tokenizer.eos_id
        num_chunks = len(seek_times)

        def boundary(chunk_index: int) -> Any:
            next_seek_time = (
                seek_times[chunk_index + 1]
                if chunk_index + 1 < num_chunks
                else None
            )
            return ChunkBoundary(seek_times[chunk_index], next_seek_time)

        batch_conditions = all_conditions[batch_start : batch_start + batch_size]
        size = len(batch_conditions)
        buffers: list[list[int]] = [[] for _ in range(size)]
        done = [False] * size
        active = 0
        items: list[Any] = [boundary(batch_start)]

        for step in model._model.generate(
            conditions=batch_conditions,
            max_gen_len=max_gen_len,
            use_sampling=use_sampling,
            temp=temperature,
            top_k=0,
            top_p=0.0,
            cfg_coef=cfg_coef,
            early_stop_on_token=eos_id,
            beam_size=beam_size,
        ):
            row = step.tolist()
            for index in range(size):
                if done[index]:
                    continue
                token = row[index]
                if token == eos_id:
                    done[index] = True
                elif index == active:
                    items.append(token)
                else:
                    buffers[index].append(token)
            while active < size and done[active]:
                active += 1
                if active < size:
                    items.append(boundary(batch_start + active))
                    items.extend(buffers[active])
                    buffers[active] = []

        for index in range(active, size):
            if not done[index]:
                chunk_index = batch_start + index
                message = (
                    f"chunk {chunk_index} (seek={seek_times[chunk_index]:.1f}s) "
                    f"did not emit EOS within {max_gen_len} tokens"
                )
                if no_eos_is_ok:
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
                else:
                    raise RuntimeError(
                        message + " (this is only raised under --strict-eos)"
                    )
            if index != active:
                items.append(boundary(batch_start + index))
                items.extend(buffers[index])

        items.append(
            ProgressEvent(completed=batch_start + size, total=num_chunks)
        )
        return items
