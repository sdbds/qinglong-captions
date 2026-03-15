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


def test_pyproject_declares_lfm_vl_local_extra():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional_deps = pyproject["project"]["optional-dependencies"]

    assert "lfm-vl-local" in optional_deps
    lfm_deps = optional_deps["lfm-vl-local"]
    assert any(dep.startswith("onnxruntime-gpu") for dep in lfm_deps)
    assert any(dep.startswith("transformers") for dep in lfm_deps)
    assert any(dep.startswith("huggingface_hub") for dep in lfm_deps)


def test_pyproject_declares_lighton_ocr_extra():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional_deps = pyproject["project"]["optional-dependencies"]

    assert "lighton-ocr" in optional_deps
    lighton_deps = optional_deps["lighton-ocr"]
    assert any(dep.startswith("torch==2.8.0") for dep in lighton_deps)
    assert any(dep.startswith("transformers[serving]>=5.0.0") for dep in lighton_deps)
    assert any(dep.startswith("huggingface_hub") for dep in lighton_deps)


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


def test_caption_step_includes_lfm_extra():
    module_path = ROOT / "gui" / "wizard" / "step4_caption.py"
    spec = importlib.util.spec_from_file_location("test_step4_caption_lfm", module_path)
    step4_caption = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(step4_caption)
    CaptionStep = step4_caption.CaptionStep

    step = CaptionStep()
    step.ocr_model = SimpleNamespace(value="")
    step.vlm_image_model = SimpleNamespace(value="lfm_vl_local")

    assert step._build_local_extra_args() == ["--extra", "lfm-vl-local"]


def test_caption_step_includes_lighton_extra():
    module_path = ROOT / "gui" / "wizard" / "step4_caption.py"
    spec = importlib.util.spec_from_file_location("test_step4_caption_lighton", module_path)
    step4_caption = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(step4_caption)
    CaptionStep = step4_caption.CaptionStep

    step = CaptionStep()
    step.ocr_model = SimpleNamespace(value="lighton_ocr")
    step.vlm_image_model = SimpleNamespace(value="")

    assert step._build_local_extra_args() == ["--extra", "lighton-ocr"]


def test_run_ps1_mentions_lfm_vl_local_extra():
    content = (ROOT / "4、run.ps1").read_text(encoding="utf-8")

    assert '"lfm_vl_local"' in content
    assert 'Add-UvExtra "lfm-vl-local"' in content


def test_run_ps1_mentions_lighton_ocr_extra():
    content = (ROOT / "4、run.ps1").read_text(encoding="utf-8")

    assert '"lighton_ocr"' in content
    assert 'Add-UvExtra "lighton-ocr"' in content


def test_run_ps1_locks_with_python_3_11_only():
    content = (ROOT / "4、run.ps1").read_text(encoding="utf-8")

    assert 'uv lock --python 3.11 --index-strategy $IndexStrategy' in content


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


def test_penguin_provider_uses_resolved_attention_backend(monkeypatch):
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
    monkeypatch.setattr(
        "utils.transformer_loader.resolve_device_dtype",
        lambda: ("cuda", "bfloat16", "flash_attention_2"),
    )
    monkeypatch.setattr("utils.transformer_loader.transformerLoader", lambda **kwargs: FakeLoader())

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={"penguin_vl_local": {"model_id": "tencent/Penguin-VL-8B"}},
        args=SimpleNamespace(vlm_image_model="penguin_vl_local"),
    )

    cached = PenguinVLLocalProvider(ctx)._load_model()

    assert captured["processor_cls"] is fake_transformers.AutoProcessor
    assert captured["attn_impl"] == "flash_attention_2"
    assert cached["processor"] == "processor"
    assert cached["model"] == "model"


def test_patch_penguin_vision_attention_uses_sdpa_fallback():
    import types

    import torch

    from providers.local_vlm.penguin_vl_local import _patch_penguin_vision_attention

    fake_module = types.ModuleType("test_penguin_encoder_module")
    fake_module.apply_multimodal_rotary_pos_emb = lambda q, k, cos, sin: (q, k)
    sys.modules[fake_module.__name__] = fake_module

    class FakePenguinAttention(torch.nn.Module):
        __module__ = "test_penguin_encoder_module"

        def __init__(self):
            super().__init__()
            self.head_dim = 2
            self.num_key_value_groups = 2
            self.layer_idx = 0
            self.q_proj = torch.nn.Linear(4, 4, bias=False)
            self.k_proj = torch.nn.Linear(4, 2, bias=False)
            self.v_proj = torch.nn.Linear(4, 2, bias=False)
            self.o_proj = torch.nn.Linear(4, 4, bias=False)
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            self.attention_dropout = 0.0
            self.is_causal = False
            self.config = SimpleNamespace()

    class FakeLayer:
        def __init__(self):
            self.self_attn = FakePenguinAttention()

    class FakeEncoder:
        def __init__(self):
            self.layers = [FakeLayer()]

    class FakeVisionEncoder:
        def __init__(self):
            self.encoder = FakeEncoder()

    class FakeInnerModel:
        def __init__(self):
            self._vision_encoder = FakeVisionEncoder()

        def get_vision_encoder(self):
            return self._vision_encoder

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(_attn_implementation="eager")
            self._model = FakeInnerModel()

        def get_model(self):
            return self._model

    model = FakeModel()
    assert _patch_penguin_vision_attention(model) is True

    attention = model.get_model().get_vision_encoder().encoder.layers[0].self_attn
    hidden_states = torch.randn(1, 4, 4)
    output, weights = attention(
        hidden_states=hidden_states,
        position_embeddings=(torch.empty(0), torch.empty(0)),
        cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32),
    )

    assert output.shape == hidden_states.shape
    assert weights is None


def test_penguin_sdpa_fallback_does_not_import_qwen3_modeling(monkeypatch):
    import builtins
    import types

    import torch

    from providers.local_vlm.penguin_vl_local import _patch_penguin_vision_attention

    fake_module = types.ModuleType("test_penguin_encoder_module_no_qwen3")
    fake_module.apply_multimodal_rotary_pos_emb = lambda q, k, cos, sin: (q, k)
    sys.modules[fake_module.__name__] = fake_module

    class FakePenguinAttention(torch.nn.Module):
        __module__ = "test_penguin_encoder_module_no_qwen3"

        def __init__(self):
            super().__init__()
            self.head_dim = 2
            self.num_key_value_groups = 2
            self.layer_idx = 0
            self.q_proj = torch.nn.Linear(4, 4, bias=False)
            self.k_proj = torch.nn.Linear(4, 2, bias=False)
            self.v_proj = torch.nn.Linear(4, 2, bias=False)
            self.o_proj = torch.nn.Linear(4, 4, bias=False)
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            self.attention_dropout = 0.0
            self.is_causal = False
            self.config = SimpleNamespace()

    class FakeLayer:
        def __init__(self):
            self.self_attn = FakePenguinAttention()

    class FakeEncoder:
        def __init__(self):
            self.layers = [FakeLayer()]

    class FakeVisionEncoder:
        def __init__(self):
            self.encoder = FakeEncoder()

    class FakeInnerModel:
        def __init__(self):
            self._vision_encoder = FakeVisionEncoder()

        def get_vision_encoder(self):
            return self._vision_encoder

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(_attn_implementation="eager")
            self._model = FakeInnerModel()

        def get_model(self):
            return self._model

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "transformers.models.qwen3.modeling_qwen3":
            raise AssertionError("unexpected qwen3 modeling import")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    model = FakeModel()
    assert _patch_penguin_vision_attention(model) is True

    attention = model.get_model().get_vision_encoder().encoder.layers[0].self_attn
    hidden_states = torch.randn(1, 4, 4)
    output, weights = attention(
        hidden_states=hidden_states,
        position_embeddings=(torch.empty(0), torch.empty(0)),
        cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32),
    )

    assert output.shape == hidden_states.shape
    assert weights is None


def test_penguin_attempt_preserves_non_tensor_outputs_and_casts_float_inputs(monkeypatch, tmp_path):
    import io

    import torch
    from rich.console import Console

    from providers.base import ProviderContext, PromptContext
    from providers.local_vlm.penguin_vl_local import PenguinVLLocalProvider

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake")

    captured = {}

    class FakeModel:
        device = torch.device("cpu")
        dtype = torch.bfloat16

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return torch.tensor([[10, 11, 12]])

    class FakeProcessor:
        def __call__(self, **kwargs):
            captured["processor_kwargs"] = kwargs
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
                "pixel_values": torch.randn(2, 8, dtype=torch.float32),
                "grid_sizes": torch.tensor([[1, 1, 1]], dtype=torch.int64),
                "modals": ["image"],
            }

        def decode(self, *_args, **_kwargs):
            return "ok"

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={"penguin_vl_local": {"model_id": "tencent/Penguin-VL-8B"}},
        args=SimpleNamespace(vlm_image_model="penguin_vl_local", pair_dir=""),
    )
    provider = PenguinVLLocalProvider(ctx)
    monkeypatch.setattr(provider, "_get_or_load_model", lambda: {"model": FakeModel(), "processor": FakeProcessor(), "device": "cpu"})

    media = provider.prepare_media(str(image_path), "image/png", ctx.args)
    result = provider.attempt(media, PromptContext(system="system", user="user"))

    assert captured["generate_kwargs"]["modals"] == ["image"]
    assert captured["generate_kwargs"]["input_ids"].device.type == "cpu"
    assert captured["generate_kwargs"]["input_ids"].dtype == torch.int64
    assert captured["generate_kwargs"]["pixel_values"].dtype == torch.bfloat16
    assert captured["generate_kwargs"]["grid_sizes"].dtype == torch.int64
    assert result.raw == "ok"


def test_penguin_attempt_keeps_generated_tokens_when_model_returns_new_tokens_only(monkeypatch, tmp_path):
    import io

    import torch
    from rich.console import Console

    from providers.base import ProviderContext, PromptContext
    from providers.local_vlm.penguin_vl_local import PenguinVLLocalProvider

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake")

    captured = {}

    class FakeModel:
        device = torch.device("cpu")
        dtype = torch.float32

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return torch.tensor([[101, 102, 103]])

    class FakeProcessor:
        def __call__(self, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
                "modals": ["image"],
            }

        def decode(self, token_ids, *_args, **_kwargs):
            captured["decoded_ids"] = token_ids.tolist()
            return "decoded-new-tokens"

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={"penguin_vl_local": {"model_id": "tencent/Penguin-VL-8B"}},
        args=SimpleNamespace(vlm_image_model="penguin_vl_local", pair_dir=""),
    )
    provider = PenguinVLLocalProvider(ctx)
    monkeypatch.setattr(provider, "_get_or_load_model", lambda: {"model": FakeModel(), "processor": FakeProcessor(), "device": "cpu"})

    media = provider.prepare_media(str(image_path), "image/png", ctx.args)
    result = provider.attempt(media, PromptContext(system="system", user="user"))

    assert captured["decoded_ids"] == [101, 102, 103]
    assert result.raw == "decoded-new-tokens"
