from __future__ import annotations

import importlib
import sys
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

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.count


@dataclass
class FakeTorch:
    cuda: FakeCuda


class FakeModelClass:
    calls: list[dict[str, object]] = []

    @classmethod
    def load_model(cls, **kwargs):
        cls.calls.append(kwargs)
        return SimpleNamespace(_device=kwargs["device"])


def fake_bindings(*, cuda: bool = False, count: int = 0):
    from module.muscriptor_tool.runtime import UpstreamBindings

    FakeModelClass.calls = []
    return UpstreamBindings(
        torch=FakeTorch(cuda=FakeCuda(cuda, count)),
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


def test_auto_device_resolves_cuda_zero_when_available():
    from module.muscriptor_tool.runtime import resolve_device

    assert resolve_device("auto", FakeTorch(FakeCuda(True, 2))) == "cuda:0"
    assert resolve_device("auto", FakeTorch(FakeCuda(False, 0))) == "cpu"


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
