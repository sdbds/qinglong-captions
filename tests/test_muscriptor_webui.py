from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


def test_server_batch_size_zero_selects_upstream_auto():
    from module.muscriptor_tool.webui import resolve_server_batch_size

    assert resolve_server_batch_size(0) is None
    assert resolve_server_batch_size(1) == 1
    assert resolve_server_batch_size(8) == 8
    with pytest.raises(ValueError, match="batch size"):
        resolve_server_batch_size(-1)


def test_batch_size_proxy_overrides_official_server_default():
    from module.muscriptor_tool.adaptive_batch import AdaptiveBatchModelProxy

    calls: list[dict[str, object]] = []

    class Model:
        marker = "official"

        def transcribe(self, *_args, **kwargs):
            calls.append(kwargs)
            return iter(())

    cpu_model = Model()
    cpu_model._device = SimpleNamespace(type="cpu")
    automatic = AdaptiveBatchModelProxy(cpu_model, None)
    explicit = AdaptiveBatchModelProxy(Model(), 6)

    assert list(automatic.transcribe("audio", batch_size=1)) == []
    assert list(explicit.transcribe("audio", batch_size=1)) == []
    assert calls == [{"batch_size": 1}, {"batch_size": 6}]
    assert automatic.marker == "official"


def test_webui_auto_batch_uses_recorded_profile_for_current_gpu():
    from module.muscriptor_tool.webui import resolve_profile_auto_batch_size

    total_memory = 24_146_083_840
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(
            get_device_properties=lambda _device: SimpleNamespace(
                total_memory=total_memory
            )
        )
    )

    assert (
        resolve_profile_auto_batch_size(
            "large",
            "cuda:0",
            torch_module=torch_module,
        )
        == 8
    )
    assert resolve_profile_auto_batch_size("large", "cpu", torch_module=torch_module) == 1


def test_profile_seed_skips_bs1_bs2_calibration_and_keeps_oom_fallback():
    from module.muscriptor_tool.adaptive_batch import AdaptiveBatchModelProxy

    proxy = AdaptiveBatchModelProxy(
        SimpleNamespace(),
        None,
        initial_auto_batch_size=6,
    )
    calls = []
    proxy._configure_allocator_limit = lambda: None

    def fake_run(
        _model,
        _conditions,
        _seek_times,
        batch_start,
        requested_size,
        _generation_args,
    ):
        calls.append((batch_start, requested_size))
        return [], requested_size, False, False

    proxy._run_with_oom_fallback = fake_run

    assert list(
        proxy._adaptive_token_stream(
            SimpleNamespace(),
            [object()] * 10,
            [float(index * 5) for index in range(10)],
            1,
            2_000,
            False,
            1.0,
            1.0,
            True,
            1,
        )
    ) == []
    assert calls == [(0, 6), (6, 4)]


def test_shared_gpu_memory_reduces_following_batches_by_two(monkeypatch):
    from module.muscriptor_tool.adaptive_batch import AdaptiveBatchModelProxy

    proxy = AdaptiveBatchModelProxy(
        SimpleNamespace(),
        None,
        initial_auto_batch_size=8,
    )
    proxy._configure_allocator_limit = lambda: None
    calls = []

    def fake_run(
        _model,
        _conditions,
        _seek_times,
        batch_start,
        requested_size,
        _generation_args,
    ):
        calls.append((batch_start, requested_size))
        return [], requested_size, False, batch_start == 0

    proxy._run_with_oom_fallback = fake_run
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
    )

    assert list(
        proxy._adaptive_token_stream(
            SimpleNamespace(),
            [object()] * 20,
            [float(index * 5) for index in range(20)],
            1,
            2_000,
            False,
            1.0,
            1.0,
            True,
            1,
        )
    ) == []
    assert calls == [(0, 8), (8, 6), (14, 6)]
    assert proxy._resolved_auto_batch_size == 6


def test_shared_memory_monitor_ignores_noise_and_detects_growth():
    from module.muscriptor_tool.gpu_memory import SharedMemoryMonitor

    monitor = SharedMemoryMonitor(detection_threshold_bytes=64 * 1024**2)

    assert monitor.grew_into_shared_memory(0, 63 * 1024**2) is False
    assert monitor.grew_into_shared_memory(128 * 1024**2, 160 * 1024**2) is False
    assert monitor.grew_into_shared_memory(128 * 1024**2, 256 * 1024**2) is True


def test_allocator_budget_subtracts_reserve_and_external_gpu_usage(monkeypatch):
    from module.muscriptor_tool.adaptive_batch import AdaptiveBatchModelProxy

    total = 24 * 1024**3
    process_reserved = 10 * 1024**3
    external = 2 * 1024**3
    free = total - process_reserved - external
    fractions = []
    cuda = SimpleNamespace(
        mem_get_info=lambda _device: (free, total),
        memory_reserved=lambda _device: process_reserved,
        set_per_process_memory_fraction=lambda fraction, device: fractions.append(
            (fraction, device)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    proxy = AdaptiveBatchModelProxy(
        SimpleNamespace(_device="cuda:0"),
        None,
        initial_auto_batch_size=8,
    )

    proxy._configure_allocator_limit()

    assert fractions == [((20 * 1024**3) / total, "cuda:0")]


def test_cuda_oom_steps_even_batch_down_by_two(monkeypatch):
    from module.muscriptor_tool.adaptive_batch import AdaptiveBatchModelProxy

    class FakeOOM(Exception):
        pass

    attempted = []
    cuda = SimpleNamespace(
        OutOfMemoryError=FakeOOM,
        synchronize=lambda _device: None,
        empty_cache=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    proxy = AdaptiveBatchModelProxy(
        SimpleNamespace(_device="cuda:0"),
        None,
        initial_auto_batch_size=8,
    )

    def generate(*args):
        size = args[4]
        attempted.append(size)
        if size == 8:
            raise FakeOOM
        return []

    proxy._generate_batch_items = generate

    assert proxy._run_with_oom_fallback(
        SimpleNamespace(),
        [object()] * 8,
        [float(index * 5) for index in range(8)],
        0,
        8,
        (2_000, False, 1.0, 1.0, True, 1),
    ) == ([], 6, True, False)
    assert attempted == [8, 6]


def test_cuda_auto_batch_rejects_missing_recorded_profile_seed():
    from module.muscriptor_tool.adaptive_batch import AdaptiveBatchModelProxy

    model = SimpleNamespace(
        _device=SimpleNamespace(type="cuda"),
        transcribe=lambda *_args, **_kwargs: iter(()),
    )

    with pytest.raises(RuntimeError, match="recorded model profile"):
        AdaptiveBatchModelProxy(model, None).transcribe("audio")


def test_cuda_cache_is_released_when_webui_stream_closes():
    from module.muscriptor_tool.adaptive_batch import AdaptiveBatchModelProxy

    class Model:
        _device = SimpleNamespace(type="cuda")

        def transcribe(self, *_args, **_kwargs):
            yield "first"
            yield "second"

    proxy = AdaptiveBatchModelProxy(Model(), 4)
    releases = []
    proxy._release_cuda_cache = lambda: releases.append("released")

    events = proxy.transcribe("audio")
    assert next(events) == "first"
    events.close()

    assert releases == ["released"]


def test_cuda_cache_cleanup_keeps_model_and_reports_resident_memory(monkeypatch):
    from module.muscriptor_tool.adaptive_batch import AdaptiveBatchModelProxy

    calls = []
    cuda = SimpleNamespace(
        synchronize=lambda device: calls.append(("synchronize", device)),
        empty_cache=lambda: calls.append(("empty_cache",)),
        memory_allocated=lambda device: 5 * 1024**3,
        memory_reserved=lambda device: 6 * 1024**3,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    messages = []
    model = SimpleNamespace(_device="cuda:0")
    proxy = AdaptiveBatchModelProxy(
        model,
        4,
        console=SimpleNamespace(print=lambda message: messages.append(str(message))),
    )

    proxy._release_cuda_cache()

    assert proxy._model is model
    assert calls == [("synchronize", "cuda:0"), ("empty_cache",)]
    assert "allocated 5.00 GiB, reserved 6.00 GiB" in messages[0]
    assert "model remains loaded" in messages[0]


def test_adaptive_batch_preserves_upstream_token_stream_order():
    torch = pytest.importorskip("torch")
    pytest.importorskip("muscriptor")
    from muscriptor.transcription_model import TranscriptionModel

    from module.muscriptor_tool.adaptive_batch import AdaptiveBatchModelProxy

    class Generator:
        def generate(self, *, conditions, **_kwargs):
            end_steps = [2 + int(value) for value in conditions]
            for step in range(max(end_steps)):
                yield torch.tensor(
                    [
                        99 if step + 1 >= end else 10 + int(value)
                        for value, end in zip(conditions, end_steps)
                    ]
                )

    fake = SimpleNamespace(
        _tokenizer=SimpleNamespace(eos_id=99),
        _model=Generator(),
    )
    conditions = [0, 1, 2, 3]
    seek_times = [0.0, 5.0, 10.0, 15.0]
    generation_args = (2_000, False, 1.0, 1.0, True, 1)

    expected = list(
        TranscriptionModel._generate_token_stream(
            fake,
            conditions,
            seek_times,
            4,
            *generation_args,
        )
    )
    actual = AdaptiveBatchModelProxy._generate_batch_items(
        fake,
        conditions,
        seek_times,
        0,
        4,
        *generation_args,
    )

    assert actual == expected


def test_webui_parser_defaults_match_project_runtime():
    from module.muscriptor_tool.webui import build_parser

    args = build_parser().parse_args([])

    assert vars(args) == {
        "host": "127.0.0.1",
        "port": 8222,
        "model": "large",
        "device": "auto",
        "batch_size": 0,
    }
