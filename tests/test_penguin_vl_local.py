from __future__ import annotations

import builtins
import importlib
import io
import sys
import types
from types import SimpleNamespace

import torch
from rich.console import Console


def _make_fake_penguin_model(module_name: str):
    fake_module = types.ModuleType(module_name)
    fake_module.apply_multimodal_rotary_pos_emb = lambda q, k, cos, sin: (q, k)
    sys.modules[module_name] = fake_module

    class FakePenguinAttention(torch.nn.Module):
        __module__ = module_name

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

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(_attn_implementation="eager")
            layer = SimpleNamespace(self_attn=FakePenguinAttention())
            vision = SimpleNamespace(encoder=SimpleNamespace(layers=[layer]))
            self._model = SimpleNamespace(get_vision_encoder=lambda: vision)

        def get_model(self):
            return self._model

    return FakeModel()


def test_has_flash_attn_returns_false_when_module_import_fails(monkeypatch):
    from utils import transformer_loader

    transformer_loader._has_flash_attn.cache_clear()
    monkeypatch.setattr(
        transformer_loader.importlib.util,
        "find_spec",
        lambda name: object() if name == "flash_attn" else None,
    )

    def fake_import_module(name):
        if name == "flash_attn":
            raise ImportError("DLL load failed")
        return importlib.import_module(name)

    monkeypatch.setattr(transformer_loader.importlib, "import_module", fake_import_module)
    assert transformer_loader._has_flash_attn() is False


def test_penguin_provider_uses_resolved_attention_backend(monkeypatch):
    from module.providers.base import ProviderContext
    from module.providers.local_vlm.penguin_vl_local import PenguinVLLocalProvider

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


def test_patch_penguin_vision_attention_uses_sdpa_fallback_without_qwen3_import(monkeypatch):
    from module.providers.local_vlm.penguin_vl_local import _patch_penguin_vision_attention

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "transformers.models.qwen3.modeling_qwen3":
            raise AssertionError("unexpected qwen3 modeling import")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    model = _make_fake_penguin_model("test_penguin_encoder_module_sdpa")
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
    from module.providers.base import PromptContext, ProviderContext
    from module.providers.local_vlm.penguin_vl_local import PenguinVLLocalProvider

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
    monkeypatch.setattr(
        provider,
        "_get_or_load_model",
        lambda: {"model": FakeModel(), "processor": FakeProcessor(), "device": "cpu"},
    )
    media = provider.prepare_media(str(image_path), "image/png", ctx.args)
    result = provider.attempt(media, PromptContext(system="system", user="user"))

    assert captured["generate_kwargs"]["modals"] == ["image"]
    assert captured["generate_kwargs"]["input_ids"].dtype == torch.int64
    assert captured["generate_kwargs"]["pixel_values"].dtype == torch.bfloat16
    assert captured["generate_kwargs"]["grid_sizes"].dtype == torch.int64
    assert result.raw == "ok"


def test_penguin_attempt_keeps_generated_tokens_when_model_returns_new_tokens_only(monkeypatch, tmp_path):
    from module.providers.base import PromptContext, ProviderContext
    from module.providers.local_vlm.penguin_vl_local import PenguinVLLocalProvider

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake")
    captured = {}

    class FakeModel:
        device = torch.device("cpu")
        dtype = torch.float32

        def generate(self, **kwargs):
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
    monkeypatch.setattr(
        provider,
        "_get_or_load_model",
        lambda: {"model": FakeModel(), "processor": FakeProcessor(), "device": "cpu"},
    )
    media = provider.prepare_media(str(image_path), "image/png", ctx.args)
    result = provider.attempt(media, PromptContext(system="system", user="user"))

    assert captured["decoded_ids"] == [101, 102, 103]
    assert result.raw == "decoded-new-tokens"
