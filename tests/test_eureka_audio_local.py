# -*- coding: utf-8 -*-

import io
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


def test_eureka_audio_defaults_to_official_model_id():
    from providers.local_alm.eureka_audio_local import EurekaAudioLocalProvider

    model_config = tomllib.loads((ROOT / "config" / "model.toml").read_text(encoding="utf-8"))

    assert EurekaAudioLocalProvider.default_model_id == "cslys1999/Eureka-Audio-Instruct"
    assert model_config["eureka_audio_local"]["model_id"] == "cslys1999/Eureka-Audio-Instruct"


def test_shipped_prompt_files_declare_eureka_audio_defaults():
    prompt_config = tomllib.loads((ROOT / "config" / "prompts.toml").read_text(encoding="utf-8"))
    legacy_config = tomllib.loads((ROOT / "config" / "config.toml").read_text(encoding="utf-8"))

    assert prompt_config["prompts"]["eureka_audio_system_prompt"] == "Descript The audio."
    assert prompt_config["prompts"]["eureka_audio_prompt"] == "Descript The audio."
    assert legacy_config["prompts"]["eureka_audio_system_prompt"] == "Descript The audio."
    assert legacy_config["prompts"]["eureka_audio_prompt"] == "Descript The audio."


def test_eureka_audio_can_handle_audio_only():
    from providers.local_alm.eureka_audio_local import EurekaAudioLocalProvider

    args = SimpleNamespace(alm_model="eureka_audio_local")

    assert EurekaAudioLocalProvider.can_handle(args, "audio/wav") is True
    assert EurekaAudioLocalProvider.can_handle(args, "image/jpeg") is False


def test_prompt_resolver_prefers_eureka_audio_prompts():
    from providers.resolver import PromptResolver

    config = {
        "prompts": {
            "audio_system_prompt": "base-system",
            "audio_prompt": "base-user",
            "eureka_audio_system_prompt": "eureka-system",
            "eureka_audio_prompt": "eureka-user",
        }
    }

    prompts = PromptResolver(config, "eureka_audio_local").resolve(
        "audio/wav",
        SimpleNamespace(pair_dir=""),
    )

    assert prompts.system == "eureka-system"
    assert prompts.user == "eureka-user"


def test_eureka_audio_post_validate_returns_structured_summary_payload(tmp_path):
    from providers.base import CaptionResult, ProviderContext
    from providers.local_alm.eureka_audio_local import EurekaAudioLocalProvider

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={},
        args=SimpleNamespace(alm_model="eureka_audio_local", wait_time=1, max_retries=3),
    )
    provider = EurekaAudioLocalProvider(ctx)
    media = provider.prepare_media(str(audio_path), "audio/wav", ctx.args)
    result = provider.post_validate(
        CaptionResult(
            raw="""<think>private reasoning</think>
```
Soft rain ambience sits under a calm piano melody.
```"""
        ),
        media,
        ctx.args,
    )

    assert result.raw == "Soft rain ambience sits under a calm piano melody."
    assert result.parsed == {
        "description": "Soft rain ambience sits under a calm piano melody.",
        "caption_extension": ".txt",
        "provider": "eureka_audio_local",
    }


def test_eureka_audio_attempt_builds_audio_url_conversation_and_decodes_new_tokens(monkeypatch, tmp_path):
    import numpy as np

    from providers.base import PromptContext, ProviderContext
    from providers.local_alm.eureka_audio_local import EurekaAudioLocalProvider

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    captured = {}

    class FakeFeatureTensor:
        def __init__(self, dtype="float32"):
            self.dtype = dtype

        def to(self, _device=None, dtype=None):
            if dtype is None:
                return self
            return FakeFeatureTensor(dtype=str(dtype))

    class FakeBatchFeature(Mapping):
        def __init__(self, data):
            self._data = dict(data)

        def to(self, device=None, dtype=None):
            captured["batch_to_args"] = (device, dtype)
            converted = dict(self._data)
            if dtype is not None and "input_features" in converted:
                converted["input_features"] = converted["input_features"].to(dtype=dtype)
            return FakeBatchFeature(converted)

        def __getitem__(self, key):
            return self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

    class FakeProcessor:
        def apply_chat_template(self, conversation, **kwargs):
            captured["conversation"] = conversation
            captured["template_kwargs"] = kwargs
            return FakeBatchFeature(
                {
                    "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
                    "input_features": FakeFeatureTensor(),
                }
            )

        def batch_decode(self, token_ids, **kwargs):
            captured["decoded_ids"] = token_ids.tolist()
            captured["decode_kwargs"] = kwargs
            return ["eureka decoded text"]

    class FakeModel:
        device = "cpu"
        dtype = "bfloat16"

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return np.array([[1, 2, 3, 7, 8]], dtype=np.int64)

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={
            "eureka_audio_local": {
                "max_new_tokens": 321,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 0.0,
                "top_k": 0,
            }
        },
        args=SimpleNamespace(alm_model="eureka_audio_local", wait_time=1, max_retries=3),
    )
    provider = EurekaAudioLocalProvider(ctx)
    monkeypatch.setattr(provider, "_get_or_load_model", lambda: {"model": FakeModel(), "processor": FakeProcessor()})

    media = provider.prepare_media(str(audio_path), "audio/wav", ctx.args)
    result = provider.attempt(
        media,
        PromptContext(system="system prompt", user="describe this audio"),
    )

    assert captured["conversation"][0] == {
        "role": "system",
        "content": [{"type": "text", "text": "system prompt"}],
    }
    assert captured["conversation"][1]["content"][0] == {
        "type": "audio_url",
        "audio_url": {"url": str(audio_path.resolve())},
    }
    assert captured["conversation"][1]["content"][1] == {"type": "text", "text": "describe this audio"}
    assert captured["template_kwargs"]["return_tensors"] == "pt"
    assert captured["batch_to_args"] == ("cpu", "bfloat16")
    assert captured["generate_kwargs"]["input_features"].dtype == "bfloat16"
    assert captured["generate_kwargs"]["max_new_tokens"] == 321
    assert captured["generate_kwargs"]["do_sample"] is False
    assert captured["generate_kwargs"]["temperature"] == 0.0
    assert captured["generate_kwargs"]["top_p"] == 0.0
    assert captured["generate_kwargs"]["top_k"] == 0
    assert captured["decoded_ids"] == [[7, 8]]
    assert result.raw == "eureka decoded text"


def test_eureka_audio_attempt_accepts_generate_returning_generated_tokens_only(monkeypatch, tmp_path):
    import numpy as np

    from providers.base import PromptContext, ProviderContext
    from providers.local_alm.eureka_audio_local import EurekaAudioLocalProvider

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    captured = {}

    class FakeFeatureTensor:
        def __init__(self, dtype="float32"):
            self.dtype = dtype

        def to(self, _device=None, dtype=None):
            if dtype is None:
                return self
            return FakeFeatureTensor(dtype=str(dtype))

    class FakeBatchFeature(Mapping):
        def __init__(self, data):
            self._data = dict(data)

        def to(self, device=None, dtype=None):
            captured["batch_to_args"] = (device, dtype)
            converted = dict(self._data)
            if dtype is not None and "input_features" in converted:
                converted["input_features"] = converted["input_features"].to(dtype=dtype)
            return FakeBatchFeature(converted)

        def __getitem__(self, key):
            return self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

    class FakeProcessor:
        def apply_chat_template(self, conversation, **kwargs):
            captured["conversation"] = conversation
            captured["template_kwargs"] = kwargs
            return FakeBatchFeature(
                {
                    "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
                    "input_features": FakeFeatureTensor(),
                }
            )

        def batch_decode(self, token_ids, **kwargs):
            captured["decoded_ids"] = token_ids.tolist()
            captured["decode_kwargs"] = kwargs
            return ["generated-only text"]

    class FakeModel:
        device = "cpu"
        dtype = "bfloat16"

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return np.array([[7, 8]], dtype=np.int64)

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={"eureka_audio_local": {"max_new_tokens": 321}},
        args=SimpleNamespace(alm_model="eureka_audio_local", wait_time=1, max_retries=3),
    )
    provider = EurekaAudioLocalProvider(ctx)
    monkeypatch.setattr(provider, "_get_or_load_model", lambda: {"model": FakeModel(), "processor": FakeProcessor()})

    media = provider.prepare_media(str(audio_path), "audio/wav", ctx.args)
    result = provider.attempt(
        media,
        PromptContext(system="system prompt", user="describe this audio"),
    )

    assert captured["decoded_ids"] == [[7, 8]]
    assert result.raw == "generated-only text"


def test_eureka_audio_load_uses_automodel_and_processor(monkeypatch):
    from providers.base import ProviderContext
    from providers.local_alm.eureka_audio_local import EurekaAudioLocalProvider

    captured = {}

    class FakeModel:
        def eval(self):
            return self

    def fake_load_pretrained_component(component_cls, model_id, **kwargs):
        component_name = kwargs.get("component_name")
        captured[f"{component_name}_cls"] = component_cls
        captured[f"{component_name}_model_id"] = model_id
        captured[f"{component_name}_kwargs"] = kwargs
        if component_name == "processor":
            return "processor"
        return FakeModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = object
    fake_transformers.AutoModelForCausalLM = object
    fake_transformer_loader = types.ModuleType("utils.transformer_loader")
    fake_transformer_loader.load_pretrained_component = fake_load_pretrained_component
    fake_transformer_loader.resolve_device_dtype = lambda: ("cuda", "bfloat16", "flash_attention_2")

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "utils.transformer_loader", fake_transformer_loader)

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={},
        args=SimpleNamespace(alm_model="eureka_audio_local"),
    )
    provider = EurekaAudioLocalProvider(ctx)

    cached = provider._load_model()

    assert cached["processor"] == "processor"
    assert isinstance(cached["model"], FakeModel)
    assert captured["model_model_id"] == "cslys1999/Eureka-Audio-Instruct"
    assert captured["model_kwargs"]["trust_remote_code"] is True
    assert captured["model_kwargs"]["device_map"] == "auto"
    assert captured["model_kwargs"]["attn_implementation"] == "flash_attention_2"
