from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from module.muscriptor_tool.options import ModelVariant, TranscriptionOptions


def test_importing_runtime_does_not_import_torch_or_muscriptor():
    sys.modules.pop("module.muscriptor_tool.runtime", None)
    before = set(sys.modules)

    importlib.import_module("module.muscriptor_tool.runtime")

    added = set(sys.modules) - before
    assert "torch" not in added
    assert "muscriptor" not in added


@dataclass
class FakeCuda:
    available: bool
    count: int = 0
    total_memory: int | None = None

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.count

    def get_device_properties(self, _device: str):
        if self.total_memory is None:
            raise AttributeError("total memory is unavailable")
        return SimpleNamespace(total_memory=self.total_memory)


@dataclass
class FakeTorch:
    cuda: FakeCuda


class FakeModelClass:
    calls: list[dict[str, object]] = []

    @classmethod
    def load_model(cls, **kwargs):
        cls.calls.append(kwargs)
        return SimpleNamespace(_device=kwargs["device"])


def fake_bindings(*, cuda: bool = False, count: int = 0, total_memory: int | None = None):
    from module.muscriptor_tool.runtime import UpstreamBindings

    FakeModelClass.calls = []
    return UpstreamBindings(
        torch=FakeTorch(cuda=FakeCuda(cuda, count, total_memory)),
        model_cls=FakeModelClass,
        progress_event_type=type("ProgressEvent", (), {}),
        instrument_names=("piano", "drums"),
        instrument_resolver=lambda values: [str(value).lower() for value in values],
        version="0.2.1",
    )


def test_load_model_passes_only_official_variant():
    from module.muscriptor_tool.runtime import load_model

    loaded = load_model(
        TranscriptionOptions(model=ModelVariant.SMALL, device="cpu"),
        upstream=fake_bindings(),
    )

    assert FakeModelClass.calls == [{"weights_path": "small", "device": "cpu"}]
    assert loaded.requested_device == "cpu"
    assert loaded.resolved_device == "cpu"
    assert loaded.package_version == "0.2.1"


def test_load_model_uses_project_rich_hugging_face_reporting(monkeypatch):
    from module.muscriptor_tool import runtime

    messages: list[str] = []
    reporting = []

    class FakeConsole:
        def print(self, message):
            messages.append(str(message))

    @contextmanager
    def fake_download_progress(console):
        reporting.append(("enter", console))
        yield
        reporting.append(("exit", console))

    console = FakeConsole()
    monkeypatch.setattr(runtime, "_hf_download_progress", fake_download_progress)
    monkeypatch.setattr(runtime, "_optimize_attention", lambda *_args: 0)

    runtime.load_model(
        TranscriptionOptions(model=ModelVariant.MEDIUM, device="cpu"),
        upstream=fake_bindings(),
        console=console,
    )

    assert reporting == [("enter", console), ("exit", console)]
    assert messages == [
        "[cyan]Resolving Hugging Face model:[/cyan] MuScriptor/muscriptor-medium",
        "[green]Hugging Face model ready:[/green] MuScriptor/muscriptor-medium",
    ]


def test_load_model_reports_optimized_sdpa_layers(monkeypatch):
    from module.muscriptor_tool import runtime

    messages: list[str] = []
    console = SimpleNamespace(print=lambda message: messages.append(str(message)))
    monkeypatch.setattr(runtime, "_optimize_attention", lambda *_args: 48)

    runtime.load_model(
        TranscriptionOptions(model=ModelVariant.LARGE, device="cpu"),
        upstream=fake_bindings(),
        console=console,
    )

    assert messages[-1] == (
        "[green]MuScriptor optimized SDPA ready:[/green] 48 attention layers"
    )


def test_attention_mask_strategy_removes_only_mathematically_redundant_masks():
    from module.muscriptor_tool.attention import attention_mask_strategy

    assert attention_mask_strategy(1, 1) == "unmasked"
    assert attention_mask_strategy(512, 512) == "causal"
    assert attention_mask_strategy(1, 2_000) == "unmasked"
    assert attention_mask_strategy(4, 512) == "bottom_right"


def test_optimized_sdpa_matches_upstream_prefill_and_kv_decode():
    import copy

    torch = pytest.importorskip("torch")
    pytest.importorskip("muscriptor")
    from muscriptor.modules.streaming import increment_steps, init_states
    from muscriptor.modules.transformer import StreamingMultiheadAttention

    from module.muscriptor_tool.attention import optimize_muscriptor_sdpa

    torch.manual_seed(7)
    original = StreamingMultiheadAttention(32, 4, device="cpu", dtype=torch.float32)
    optimized = copy.deepcopy(original)
    wrapped = SimpleNamespace(
        _model=SimpleNamespace(
            transformer=SimpleNamespace(
                layers=[SimpleNamespace(self_attn=optimized)],
            )
        )
    )
    assert optimize_muscriptor_sdpa(wrapped) == 1
    assert optimize_muscriptor_sdpa(wrapped) == 0

    original_state = init_states(original, batch_size=2, sequence_length=16)
    optimized_state = init_states(optimized, batch_size=2, sequence_length=16)
    for query_length in (4, 1, 2):
        query = torch.randn(2, query_length, 32)
        expected = original(query, model_state=original_state)
        actual = optimized(query, model_state=optimized_state)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        increment_steps(original, original_state, increment=query_length)
        increment_steps(optimized, optimized_state, increment=query_length)


def test_auto_device_resolves_cuda_zero_when_available():
    from module.muscriptor_tool.runtime import resolve_device

    assert resolve_device("auto", FakeTorch(FakeCuda(True, 2))) == "cuda:0"
    assert resolve_device("auto", FakeTorch(FakeCuda(False, 0))) == "cpu"


def test_load_model_auto_falls_back_before_large_model_oom():
    from module.muscriptor_tool.runtime import load_model

    loaded = load_model(
        TranscriptionOptions(model="large", device="auto"),
        upstream=fake_bindings(cuda=True, count=1, total_memory=8 * 1024**3),
    )

    assert FakeModelClass.calls == [{"weights_path": "large", "device": "cpu"}]
    assert loaded.requested_device == "auto"
    assert loaded.resolved_device == "cpu"


def test_load_model_explicit_cuda_rejects_model_below_minimum():
    from module.muscriptor_tool.batch_profiles import InsufficientVRAMError
    from module.muscriptor_tool.runtime import load_model

    with pytest.raises(InsufficientVRAMError, match="10.28 GiB"):
        load_model(
            TranscriptionOptions(model="large", device="cuda:0"),
            upstream=fake_bindings(cuda=True, count=1, total_memory=8 * 1024**3),
        )

    assert FakeModelClass.calls == []


def test_load_model_auto_keeps_cuda_when_selected_model_fits():
    from module.muscriptor_tool.adaptive_batch import AdaptiveBatchModelProxy
    from module.muscriptor_tool.runtime import load_model

    loaded = load_model(
        TranscriptionOptions(model="small", device="auto"),
        upstream=fake_bindings(cuda=True, count=1, total_memory=8 * 1024**3),
    )

    assert FakeModelClass.calls == [{"weights_path": "small", "device": "cuda:0"}]
    assert loaded.resolved_device == "cuda:0"
    assert isinstance(loaded.model, AdaptiveBatchModelProxy)
    assert loaded.model._resolved_auto_batch_size == 14


def test_explicit_cuda_never_falls_back_to_cpu():
    from module.muscriptor_tool.runtime import DeviceUnavailableError, resolve_device

    with pytest.raises(DeviceUnavailableError, match="CUDA"):
        resolve_device("cuda", FakeTorch(FakeCuda(False, 0)))
    with pytest.raises(DeviceUnavailableError, match="index 2"):
        resolve_device("cuda:2", FakeTorch(FakeCuda(True, 2)))


def test_instrument_names_and_resolution_share_upstream_bindings():
    from module.muscriptor_tool.runtime import list_instruments, resolve_instruments

    bindings = fake_bindings()

    assert list_instruments(upstream=bindings) == ("piano", "drums")
    assert resolve_instruments(("PIANO", "DRUMS"), upstream=bindings) == ("piano", "drums")


def test_loaded_model_forwards_normalized_transcription_options():
    from module.muscriptor_tool.runtime import LoadedModel

    calls = []

    class Model:
        def transcribe(self, **kwargs):
            calls.append(kwargs)
            return iter(())

        def events_to_midi_bytes(self, events):
            return b"MThd" + bytes([len(list(events))])

    loaded = LoadedModel(
        model=Model(),
        package_version="0.2.1",
        requested_device="cpu",
        resolved_device="cpu",
        progress_event_type=type("ProgressEvent", (), {}),
    )
    options = TranscriptionOptions.from_single_cli(instruments=("piano",), cfg_coef=1.25)

    assert list(loaded.transcribe(SimpleNamespace(), options)) == []
    assert calls[0]["instruments"] == ["piano"]
    assert calls[0]["cfg_coef"] == 1.25
    assert loaded.midi_bytes(()) == b"MThd\x00"


def test_loaded_model_filters_native_timing_noise_but_preserves_other_stderr(capsys):
    from module.muscriptor_tool.runtime import LoadedModel

    class Model:
        def transcribe(self, **_kwargs):
            print("[muscriptor] load audio: 1.09s", file=sys.stderr)
            print("upstream diagnostic", file=sys.stderr)
            yield SimpleNamespace(value="event")

    loaded = LoadedModel(
        model=Model(),
        package_version="0.2.1",
        requested_device="cpu",
        resolved_device="cpu",
        progress_event_type=type("ProgressEvent", (), {}),
    )

    assert len(list(loaded.transcribe(SimpleNamespace(), TranscriptionOptions()))) == 1
    captured = capsys.readouterr()
    assert "[muscriptor]" not in captured.err
    assert "upstream diagnostic" in captured.err


def test_loaded_model_closes_upstream_generator_when_consumer_stops_early(capsys):
    from module.muscriptor_tool.runtime import LoadedModel

    state = {"closed": False}

    class Model:
        def transcribe(self, **_kwargs):
            try:
                yield SimpleNamespace(value="first")
                yield SimpleNamespace(value="second")
            finally:
                state["closed"] = True
                print("[muscriptor] close timing", file=sys.stderr)

    loaded = LoadedModel(
        model=Model(),
        package_version="0.2.1",
        requested_device="cpu",
        resolved_device="cpu",
        progress_event_type=type("ProgressEvent", (), {}),
    )
    events = loaded.transcribe(SimpleNamespace(), TranscriptionOptions())

    assert next(events).value == "first"
    events.close()

    assert state["closed"] is True
    assert "[muscriptor]" not in capsys.readouterr().err


def test_gated_download_error_becomes_actionable_model_access_error():
    from module.muscriptor_tool.runtime import ModelAccessError, load_model

    class GatedRepoError(Exception):
        pass

    class FailingModel:
        @classmethod
        def load_model(cls, **_kwargs):
            raise GatedRepoError("401 gated repository")

    bindings = fake_bindings()
    bindings = bindings.__class__(
        torch=bindings.torch,
        model_cls=FailingModel,
        progress_event_type=bindings.progress_event_type,
        instrument_names=bindings.instrument_names,
        instrument_resolver=bindings.instrument_resolver,
        version=bindings.version,
    )

    with pytest.raises(ModelAccessError, match=r"MuScriptor/muscriptor-large.*hf auth login"):
        load_model(TranscriptionOptions(), upstream=bindings)
