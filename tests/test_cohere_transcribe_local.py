# -*- coding: utf-8 -*-

import io
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


def test_cohere_transcribe_defaults_to_official_model_id():
    from providers.local_alm.cohere_transcribe_local import CohereTranscribeLocalProvider

    model_config = tomllib.loads((ROOT / "config" / "model.toml").read_text(encoding="utf-8"))

    assert CohereTranscribeLocalProvider.default_model_id == "CohereLabs/cohere-transcribe-03-2026"
    assert model_config["cohere_transcribe_local"]["model_id"] == "CohereLabs/cohere-transcribe-03-2026"
    assert model_config["cohere_transcribe_local"]["language"] == "zh"


def test_cohere_transcribe_can_handle_audio_only():
    from providers.local_alm.cohere_transcribe_local import CohereTranscribeLocalProvider

    args = SimpleNamespace(alm_model="cohere_transcribe_local")

    assert CohereTranscribeLocalProvider.can_handle(args, "audio/wav") is True
    assert CohereTranscribeLocalProvider.can_handle(args, "image/jpeg") is False


def test_cohere_transcribe_post_validate_returns_structured_transcript_payload(tmp_path):
    from providers.base import CaptionResult, ProviderContext
    from providers.local_alm.cohere_transcribe_local import CohereTranscribeLocalProvider

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={},
        args=SimpleNamespace(alm_model="cohere_transcribe_local", wait_time=1, max_retries=3),
    )
    provider = CohereTranscribeLocalProvider(ctx)
    media = provider.prepare_media(str(audio_path), "audio/wav", ctx.args)
    result = provider.post_validate(
        CaptionResult(
            raw="""<think>private reasoning</think>
```
第一行转写

第二行转写
```"""
        ),
        media,
        ctx.args,
    )

    assert result.raw == "第一行转写\n\n第二行转写"
    assert result.parsed == {
        "task_kind": "transcribe",
        "transcript": "第一行转写\n\n第二行转写",
        "caption_extension": ".txt",
        "provider": "cohere_transcribe_local",
    }


def test_cohere_transcribe_declares_transcribe_task_contract():
    from providers.local_alm.cohere_transcribe_local import CohereTranscribeLocalProvider

    assert CohereTranscribeLocalProvider.task_contract.task_kind == "transcribe"
    assert CohereTranscribeLocalProvider.task_contract.consumes_prompts is False
    assert CohereTranscribeLocalProvider.task_contract.requires_language is True


def test_cohere_transcribe_attempt_calls_model_transcribe_with_configured_kwargs(monkeypatch, tmp_path):
    from providers.base import PromptContext, ProviderContext
    from providers.local_alm.cohere_transcribe_local import CohereTranscribeLocalProvider

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    captured = {}

    class FakeModel:
        def transcribe(self, **kwargs):
            captured["transcribe_kwargs"] = kwargs
            return ["hello from cohere"]

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={
            "cohere_transcribe_local": {
                "language": "ja",
                "punctuation": False,
                "compile": True,
                "pipeline_detokenization": True,
                "batch_size": 8,
            }
        },
        args=SimpleNamespace(alm_model="cohere_transcribe_local", wait_time=1, max_retries=3),
    )
    provider = CohereTranscribeLocalProvider(ctx)
    fake_processor = object()
    monkeypatch.setattr(provider, "_get_or_load_model", lambda: {"model": FakeModel(), "processor": fake_processor})

    media = provider.prepare_media(str(audio_path), "audio/wav", ctx.args)
    result = provider.attempt(
        media,
        PromptContext(system="ignored system prompt", user="ignored user prompt"),
    )

    assert captured["transcribe_kwargs"] == {
        "processor": fake_processor,
        "audio_files": [str(audio_path.resolve())],
        "language": "ja",
        "punctuation": False,
        "compile": True,
        "pipeline_detokenization": True,
        "batch_size": 8,
    }
    assert result.raw == "hello from cohere"


def test_cohere_transcribe_prefers_runtime_alm_language_over_model_config(monkeypatch, tmp_path):
    from providers.base import PromptContext, ProviderContext
    from providers.local_alm.cohere_transcribe_local import CohereTranscribeLocalProvider

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    captured = {}

    class FakeModel:
        def transcribe(self, **kwargs):
            captured["transcribe_kwargs"] = kwargs
            return ["bonjour"]

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={"cohere_transcribe_local": {"language": "ja"}},
        args=SimpleNamespace(alm_model="cohere_transcribe_local", alm_language="fr", wait_time=1, max_retries=3),
    )
    provider = CohereTranscribeLocalProvider(ctx)
    monkeypatch.setattr(provider, "_get_or_load_model", lambda: {"model": FakeModel(), "processor": object()})

    media = provider.prepare_media(str(audio_path), "audio/wav", ctx.args)
    provider.attempt(media, PromptContext(system="", user=""))

    assert captured["transcribe_kwargs"]["language"] == "fr"


def test_cohere_transcribe_requires_explicit_language_when_no_runtime_or_config_value(tmp_path):
    from providers.base import ProviderContext
    from providers.local_alm.cohere_transcribe_local import CohereTranscribeLocalProvider

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={"cohere_transcribe_local": {"language": ""}},
        args=SimpleNamespace(alm_model="cohere_transcribe_local", alm_language=None),
    )
    provider = CohereTranscribeLocalProvider(ctx)

    with pytest.raises(RuntimeError, match="language code"):
        provider._resolve_transcribe_kwargs()


def test_cohere_transcribe_load_uses_speech_seq2seq_and_processor(monkeypatch):
    from providers.base import ProviderContext
    from providers.local_alm.cohere_transcribe_local import CohereTranscribeLocalProvider

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
    fake_transformers.AutoModelForSpeechSeq2Seq = object
    fake_transformer_loader = types.ModuleType("utils.transformer_loader")
    fake_transformer_loader.load_pretrained_component = fake_load_pretrained_component
    fake_transformer_loader.resolve_device_dtype = lambda: ("cuda", "bfloat16", "flash_attention_2")

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "utils.transformer_loader", fake_transformer_loader)

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={},
        args=SimpleNamespace(alm_model="cohere_transcribe_local"),
    )
    provider = CohereTranscribeLocalProvider(ctx)

    cached = provider._load_model()

    assert cached["processor"] == "processor"
    assert isinstance(cached["model"], FakeModel)
    assert captured["model_model_id"] == "CohereLabs/cohere-transcribe-03-2026"
    assert captured["model_kwargs"]["trust_remote_code"] is True
    assert captured["model_kwargs"]["device_map"] == "auto"
    assert captured["model_kwargs"]["attn_implementation"] == "flash_attention_2"
