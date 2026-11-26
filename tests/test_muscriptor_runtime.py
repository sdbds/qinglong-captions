from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from module.muscriptor_tool.options import ModelVariant, TranscriptionOptions


def test_importing_runtime_does_not_import_torch_or_muscriptor():
    # A cold import must not replace the runtime object used by other transcription tests.
    code = (
        "import sys; before = set(sys.modules); import module.muscriptor_tool.runtime; "
        "unexpected = (set(sys.modules) - before) & {'torch', 'muscriptor'}; assert not unexpected, unexpected"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


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
        version="0.4.0",
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
    assert loaded.package_version == "0.4.0"


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

    loaded = LoadedModel(
        model=Model(),
        package_version="0.4.0",
        requested_device="cpu",
        resolved_device="cpu",
        progress_event_type=type("ProgressEvent", (), {}),
    )
    options = TranscriptionOptions.from_single_cli(instruments=("piano",), cfg_coef=1.25)

    assert list(loaded.transcribe(SimpleNamespace(), options)) == []
    assert calls[0]["instruments"] == ["piano"]
    assert calls[0]["cfg_coef"] == 1.25


def test_loaded_model_filters_native_timing_noise_but_preserves_other_stderr(capsys):
    from module.muscriptor_tool.runtime import LoadedModel

    class Model:
        def transcribe(self, **_kwargs):
            print("[muscriptor] load audio: 1.09s", file=sys.stderr)
            print("upstream diagnostic", file=sys.stderr)
            yield SimpleNamespace(value="event")

    loaded = LoadedModel(
        model=Model(),
        package_version="0.4.0",
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
        package_version="0.4.0",
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
