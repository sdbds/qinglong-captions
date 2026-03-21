import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


def test_waterdetect_load_model_uses_single_model_bundle(monkeypatch, tmp_path):
    sys.modules.pop("module.waterdetect", None)

    fake_torch = types.ModuleType("torch")
    fake_lance = types.ModuleType("lance")
    fake_ort = types.ModuleType("onnxruntime")
    fake_hf = types.ModuleType("huggingface_hub")
    fake_transformers = types.ModuleType("transformers")
    fake_lance_import = types.ModuleType("module.lanceImport")

    class StubAutoImageProcessor:
        @staticmethod
        def from_pretrained(repo_id, use_fast=True):
            return None

    fake_hf.hf_hub_download = lambda **kwargs: str(tmp_path / kwargs["filename"])
    fake_transformers.AutoImageProcessor = StubAutoImageProcessor
    fake_lance_import.transform2lance = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "lance", fake_lance)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "module.lanceImport", fake_lance_import)

    waterdetect = importlib.import_module("module.waterdetect")

    captured = {}
    fake_session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="pixel_values")])

    class FakeProcessor:
        pass

    def fake_from_pretrained(repo_id, use_fast=True):
        captured["processor"] = (repo_id, use_fast)
        return FakeProcessor()

    def fake_loader(*, spec, runtime_config):
        captured["spec"] = spec
        captured["runtime"] = runtime_config
        return SimpleNamespace(
            session=fake_session,
            providers=("CPUExecutionProvider",),
            input_metas=(SimpleNamespace(name="pixel_values"),),
        )

    monkeypatch.setattr(waterdetect.AutoImageProcessor, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(waterdetect, "load_single_model_bundle", fake_loader, raising=False)

    args = SimpleNamespace(
        repo_id="bdsqlsz/joycaption-watermark-detection-onnx",
        model_dir=str(tmp_path),
        force_download=False,
    )

    session, input_name = waterdetect.load_model(args)

    assert session is fake_session
    assert input_name == "pixel_values"
    assert captured["processor"] == ("bdsqlsz/joycaption-watermark-detection-onnx", True)
    assert captured["spec"].bundle_key == "waterdetect:bdsqlsz/joycaption-watermark-detection-onnx"
