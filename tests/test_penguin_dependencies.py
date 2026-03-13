import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))
sys.path.insert(0, str(ROOT / "gui"))


def test_pyproject_declares_penguin_vl_local_extra():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional_deps = pyproject["project"]["optional-dependencies"]

    assert "penguin-vl-local" in optional_deps
    penguin_deps = optional_deps["penguin-vl-local"]
    assert "transformers[serving]==4.51.3" in penguin_deps
    assert any(dep.startswith("decord") for dep in penguin_deps)
    assert "ffmpeg-python==0.2.0" in penguin_deps
    assert any(dep.startswith("einops") for dep in penguin_deps)


def test_caption_step_includes_penguin_extra():
    module_path = ROOT / "gui" / "wizard" / "step4_caption.py"
    spec = importlib.util.spec_from_file_location("test_step4_caption", module_path)
    step4_caption = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(step4_caption)
    CaptionStep = step4_caption.CaptionStep

    step = CaptionStep()
    step.ocr_model = SimpleNamespace(value="")
    step.vlm_image_model = SimpleNamespace(value="penguin_vl_local")

    assert step._build_local_extra_args() == ["--extra", "penguin-vl-local"]


def test_has_flash_attn_returns_false_when_module_import_fails(monkeypatch):
    import importlib

    from utils import transformer_loader

    transformer_loader._has_flash_attn.cache_clear()
    monkeypatch.setattr(transformer_loader.importlib.util, "find_spec", lambda name: object() if name == "flash_attn" else None)

    def fake_import_module(name):
        if name == "flash_attn":
            raise ImportError("DLL load failed")
        return importlib.import_module(name)

    monkeypatch.setattr(transformer_loader.importlib, "import_module", fake_import_module)

    assert transformer_loader._has_flash_attn() is False


def test_penguin_provider_registers_ffmpeg_compat_before_loading(monkeypatch):
    import io
    import types

    from rich.console import Console

    from providers.base import ProviderContext
    from providers.local_vlm.penguin_vl_local import PenguinVLLocalProvider

    captured = {}

    class FakeLoader:
        def get_or_load_processor(self, *args, **kwargs):
            captured["processor_cls"] = args[1]
            return "processor"

        def get_or_load_model(self, *args, **kwargs):
            captured["attn_impl"] = kwargs["attn_impl"]
            return "model"

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = object
    fake_transformers.AutoModelForCausalLM = object

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr("utils.transformer_loader.resolve_device_dtype", lambda: ("cpu", "float32", "eager"))
    monkeypatch.setattr("utils.transformer_loader.transformerLoader", lambda **kwargs: FakeLoader())

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={"penguin_vl_local": {"model_id": "tencent/Penguin-VL-8B"}},
        args=SimpleNamespace(vlm_image_model="penguin_vl_local"),
    )

    cached = PenguinVLLocalProvider(ctx)._load_model()

    assert captured["processor_cls"] is fake_transformers.AutoProcessor
    assert captured["attn_impl"] == "eager"
    assert cached["processor"] == "processor"
    assert cached["model"] == "model"
