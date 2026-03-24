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


def test_eureka_audio_attempt_builds_upstream_messages_and_calls_generate(monkeypatch, tmp_path):
    from providers.base import PromptContext, ProviderContext
    from providers.local_alm.eureka_audio_local import EurekaAudioLocalProvider

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    captured = {}
    class FakeEurekaAudio:
        def generate(self, messages, **kwargs):
            captured["messages"] = messages
            captured["generate_kwargs"] = kwargs
            return "eureka decoded text"

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
    monkeypatch.setattr(provider, "_get_or_load_model", lambda: {"model": FakeEurekaAudio()})

    media = provider.prepare_media(str(audio_path), "audio/wav", ctx.args)
    result = provider.attempt(
        media,
        PromptContext(system="system prompt", user="describe this audio"),
    )

    assert captured["messages"][0] == {
        "role": "system",
        "content": [{"type": "text", "text": "system prompt"}],
    }
    assert captured["messages"][1]["content"][0] == {
        "type": "audio_url",
        "audio_url": {"url": str(audio_path.resolve())},
    }
    assert captured["messages"][1]["content"][1] == {"type": "text", "text": "describe this audio"}
    assert captured["generate_kwargs"]["max_new_tokens"] == 321
    assert captured["generate_kwargs"]["do_sample"] is False
    assert "temperature" not in captured["generate_kwargs"]
    assert "top_p" not in captured["generate_kwargs"]
    assert "top_k" not in captured["generate_kwargs"]
    assert result.raw == "eureka decoded text"


def test_eureka_audio_attempt_preserves_sampling_controls_when_sampling_enabled(monkeypatch, tmp_path):
    from providers.base import PromptContext, ProviderContext
    from providers.local_alm.eureka_audio_local import EurekaAudioLocalProvider

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    captured = {}
    class FakeEurekaAudio:
        def generate(self, messages, **kwargs):
            captured["messages"] = messages
            captured["generate_kwargs"] = kwargs
            return "sampled text"

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={
            "eureka_audio_local": {
                "max_new_tokens": 321,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.7,
                "top_k": 12,
            }
        },
        args=SimpleNamespace(alm_model="eureka_audio_local", wait_time=1, max_retries=3),
    )
    provider = EurekaAudioLocalProvider(ctx)
    monkeypatch.setattr(provider, "_get_or_load_model", lambda: {"model": FakeEurekaAudio()})

    media = provider.prepare_media(str(audio_path), "audio/wav", ctx.args)
    result = provider.attempt(
        media,
        PromptContext(system="", user="describe this audio"),
    )

    assert captured["generate_kwargs"]["do_sample"] is True
    assert captured["generate_kwargs"]["temperature"] == 0.6
    assert captured["generate_kwargs"]["top_p"] == 0.7
    assert captured["generate_kwargs"]["top_k"] == 12
    assert result.raw == "sampled text"


def test_eureka_audio_load_uses_official_wrapper(monkeypatch):
    from providers.base import ProviderContext
    from providers.local_alm.eureka_audio_local import EurekaAudioLocalProvider

    captured = {}

    class FakeEurekaAudio:
        def __init__(self, model_path, device):
            captured["model_path"] = model_path
            captured["device"] = device

    fake_api = types.ModuleType("eureka_infer.api")
    fake_api.EurekaAudio = FakeEurekaAudio
    fake_transformer_loader = types.ModuleType("utils.transformer_loader")
    fake_transformer_loader.resolve_device_dtype = lambda: ("cuda", "bfloat16", "eager")

    monkeypatch.setitem(sys.modules, "eureka_infer.api", fake_api)
    monkeypatch.setitem(sys.modules, "utils.transformer_loader", fake_transformer_loader)

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={},
        args=SimpleNamespace(alm_model="eureka_audio_local"),
    )
    provider = EurekaAudioLocalProvider(ctx)

    cached = provider._load_model()

    assert isinstance(cached["model"], FakeEurekaAudio)
    assert captured["model_path"] == "cslys1999/Eureka-Audio-Instruct"
    assert captured["device"] == "cuda:0"
