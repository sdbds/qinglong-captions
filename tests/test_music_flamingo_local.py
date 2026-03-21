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


def test_music_flamingo_defaults_to_fp8_model_id():
    from providers.local_alm.music_flamingo_local import MusicFlamingoLocalProvider

    model_config = tomllib.loads((ROOT / "config" / "model.toml").read_text(encoding="utf-8"))

    assert MusicFlamingoLocalProvider.default_model_id == "henry1477/music-flamingo-2601-hf-fp8"
    assert model_config["music_flamingo_local"]["model_id"] == "henry1477/music-flamingo-2601-hf-fp8"


def test_prompt_resolver_prefers_music_flamingo_audio_prompts():
    from providers.resolver import PromptResolver

    config = {
        "prompts": {
            "audio_system_prompt": "base-system",
            "audio_prompt": "base-user",
            "music_flamingo_audio_system_prompt": "mf-system",
            "music_flamingo_audio_prompt": "mf-user",
        }
    }

    prompts = PromptResolver(config, "music_flamingo_local").resolve(
        "audio/wav",
        SimpleNamespace(pair_dir=""),
    )

    assert prompts.system == "mf-system"
    assert prompts.user == "mf-user"


def test_prompt_resolver_allows_empty_music_flamingo_system_prompt():
    from providers.resolver import PromptResolver

    config = {
        "prompts": {
            "audio_system_prompt": "base-system",
            "audio_prompt": "base-user",
            "music_flamingo_audio_system_prompt": "",
            "music_flamingo_audio_prompt": "Describe this song from both a technical and artistic lens: mention tempo, harmony, and instrumentation, but also mood, lyrical themes, and structure.",
        }
    }

    prompts = PromptResolver(config, "music_flamingo_local").resolve(
        "audio/wav",
        SimpleNamespace(pair_dir=""),
    )

    assert prompts.system == ""
    assert prompts.user == "Describe this song from both a technical and artistic lens: mention tempo, harmony, and instrumentation, but also mood, lyrical themes, and structure."


def test_postprocess_audio_extracts_srt_code_block(tmp_path):
    from module.caption_pipeline.postprocess import postprocess_caption_content

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    console = Console(file=io.StringIO(), force_terminal=False)
    args = SimpleNamespace(mode="all", ocr_model="")
    output = """<think>hidden</think>
```srt
1
00:00,000 --> 00:01,000
gentle piano intro
```"""

    processed = postprocess_caption_content(output, audio_path, args, console)

    assert processed == "1\n00:00:00,000 --> 00:00:01,000\ngentle piano intro"


def test_postprocess_audio_recovers_unclosed_srt_code_block(tmp_path):
    from module.caption_pipeline.postprocess import postprocess_caption_content

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    console = Console(file=io.StringIO(), force_terminal=False)
    args = SimpleNamespace(mode="all", ocr_model="")
    output = """```srt
1
00:00,000 --> 00:01,000
gentle piano intro"""

    processed = postprocess_caption_content(output, audio_path, args, console)

    assert processed == "1\n00:00:00,000 --> 00:00:01,000\ngentle piano intro"


def test_music_flamingo_post_validate_returns_structured_summary_payload(tmp_path):
    from providers.base import CaptionResult, ProviderContext
    from providers.local_alm.music_flamingo_local import MusicFlamingoLocalProvider

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={},
        args=SimpleNamespace(alm_model="music_flamingo_local", wait_time=1, max_retries=3),
    )
    provider = MusicFlamingoLocalProvider(ctx)
    media = provider.prepare_media(str(audio_path), "audio/wav", ctx.args)
    result = provider.post_validate(
        CaptionResult(
            raw="""<think>private reasoning</think>
```
Gentle piano chords open the song before a brighter synth hook enters.
```"""
        ),
        media,
        ctx.args,
    )

    assert result.raw == "Gentle piano chords open the song before a brighter synth hook enters."
    assert result.parsed == {
        "description": "Gentle piano chords open the song before a brighter synth hook enters.",
        "caption_extension": ".txt",
        "provider": "music_flamingo_local",
    }


def test_music_flamingo_post_validate_raises_retry_on_empty_summary(tmp_path):
    from providers.base import CaptionResult, ProviderContext
    from providers.local_alm.music_flamingo_local import MusicFlamingoLocalProvider

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={},
        args=SimpleNamespace(alm_model="music_flamingo_local", wait_time=1, max_retries=3),
    )
    provider = MusicFlamingoLocalProvider(ctx)
    media = provider.prepare_media(str(audio_path), "audio/wav", ctx.args)

    try:
        provider.post_validate(CaptionResult(raw="  <think>private reasoning</think>  "), media, ctx.args)
    except Exception as exc:
        assert "RETRY_INVALID_SUMMARY" in str(exc)
    else:
        raise AssertionError("expected post_validate to raise retryable summary validation error")


def test_music_flamingo_attempt_builds_audio_conversation_and_decodes_new_tokens(monkeypatch, tmp_path):
    import numpy as np

    from providers.base import ProviderContext, PromptContext
    from providers.local_alm.music_flamingo_local import MusicFlamingoLocalProvider

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
            assert conversation[0] == {
                "role": "system",
                "content": [{"type": "text", "text": "system prompt"}],
            }
            assert isinstance(conversation[1]["content"], list)
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
            return ["decoded subtitle text"]

    class FakeModel:
        device = "cpu"
        dtype = "bfloat16"

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return np.array([[1, 2, 3, 7, 8]], dtype=np.int64)

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={
            "music_flamingo_local": {
                "max_new_tokens": 123,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.8,
            }
        },
        args=SimpleNamespace(alm_model="music_flamingo_local", wait_time=1, max_retries=3),
    )
    provider = MusicFlamingoLocalProvider(ctx)
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
    assert captured["conversation"][1]["content"][0]["type"] == "text"
    assert captured["conversation"][1]["content"][1] == {"type": "audio", "path": str(audio_path.resolve())}
    assert captured["batch_to_args"] == ("cpu", "bfloat16")
    assert captured["generate_kwargs"]["input_features"].dtype == "bfloat16"
    assert captured["generate_kwargs"]["max_new_tokens"] == 123
    assert captured["generate_kwargs"]["do_sample"] is True
    assert captured["generate_kwargs"]["temperature"] == 0.6
    assert captured["generate_kwargs"]["top_p"] == 0.8
    assert captured["decoded_ids"] == [[7, 8]]
    assert result.raw == "decoded subtitle text"


def test_music_flamingo_fp8_load_uses_auto_dtype_on_cuda(monkeypatch):
    from providers.base import ProviderContext
    from providers.local_alm.music_flamingo_local import MusicFlamingoLocalProvider

    captured = {}

    class FakeModel:
        def eval(self):
            return self

    def fake_load_pretrained_component(component_cls, model_id, **kwargs):
        component_name = kwargs.get("component_name")
        captured[f"{component_name}_model_id"] = model_id
        captured[f"{component_name}_kwargs"] = kwargs
        if component_name == "processor":
            return "processor"
        return FakeModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = object
    fake_transformers.MusicFlamingoForConditionalGeneration = object
    fake_transformer_loader = types.ModuleType("utils.transformer_loader")
    fake_transformer_loader.load_pretrained_component = fake_load_pretrained_component
    fake_transformer_loader.resolve_device_dtype = lambda: ("cuda", "bfloat16", "eager")

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "utils.transformer_loader", fake_transformer_loader)

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={},
        args=SimpleNamespace(alm_model="music_flamingo_local"),
    )
    provider = MusicFlamingoLocalProvider(ctx)

    cached = provider._load_model()

    assert cached["processor"] == "processor"
    assert isinstance(cached["model"], FakeModel)
    assert captured["model_model_id"] == "henry1477/music-flamingo-2601-hf-fp8"
    assert captured["model_kwargs"]["torch_dtype"] == "auto"
    assert captured["model_kwargs"]["device_map"] == "auto"


def test_music_flamingo_fp8_does_not_warn_on_compute_capability_89(monkeypatch):
    from providers.base import ProviderContext
    from providers.local_alm.music_flamingo_local import MusicFlamingoLocalProvider

    console_buffer = io.StringIO()

    class FakeModel:
        def eval(self):
            return self

    def fake_load_pretrained_component(component_cls, model_id, **kwargs):
        if kwargs.get("component_name") == "processor":
            return "processor"
        return FakeModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = object
    fake_transformers.MusicFlamingoForConditionalGeneration = object
    fake_transformer_loader = types.ModuleType("utils.transformer_loader")
    fake_transformer_loader.load_pretrained_component = fake_load_pretrained_component
    fake_transformer_loader.resolve_device_dtype = lambda: ("cuda", "bfloat16", "eager")

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_capability():
            return (8, 9)

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = FakeCuda()

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "utils.transformer_loader", fake_transformer_loader)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    ctx = ProviderContext(
        console=Console(file=console_buffer, force_terminal=False),
        config={},
        args=SimpleNamespace(alm_model="music_flamingo_local"),
    )

    MusicFlamingoLocalProvider(ctx)._load_model()

    output = console_buffer.getvalue()
    assert "Detected FP8 Music Flamingo weights" in output
    assert ">= 8.9" not in output


def test_music_flamingo_attempt_prefers_audio_tower_dtype_for_fp8_inputs(monkeypatch, tmp_path):
    import numpy as np

    from providers.base import ProviderContext, PromptContext
    from providers.local_alm.music_flamingo_local import MusicFlamingoLocalProvider

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    captured = {}

    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeFeatureTensor:
        def __init__(self, dtype="float32"):
            self.dtype = dtype

        def to(self, _device=None, dtype=None):
            if dtype is None:
                return self
            return FakeFeatureTensor(dtype=dtype)

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
        def apply_chat_template(self, _conversation, **_kwargs):
            return FakeBatchFeature(
                {
                    "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
                    "input_features": FakeFeatureTensor(),
                }
            )

        @staticmethod
        def batch_decode(_token_ids, **_kwargs):
            return ["decoded subtitle text"]

    class FakeAudioTower:
        @staticmethod
        def parameters():
            yield SimpleNamespace(dtype="bfloat16")

    class FakeModel:
        device = "cpu"
        dtype = "float32"
        config = SimpleNamespace(dtype="bfloat16")
        audio_tower = FakeAudioTower()

        @staticmethod
        def generate(**kwargs):
            captured["generate_kwargs"] = kwargs
            return np.array([[1, 2, 3, 7, 8]], dtype=np.int64)

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={},
        args=SimpleNamespace(alm_model="music_flamingo_local", wait_time=1, max_retries=3),
    )
    provider = MusicFlamingoLocalProvider(ctx)
    monkeypatch.setattr(provider, "_get_or_load_model", lambda: {"model": FakeModel(), "processor": FakeProcessor()})

    media = provider.prepare_media(str(audio_path), "audio/wav", ctx.args)
    result = provider.attempt(
        media,
        PromptContext(system="system prompt", user="describe this audio"),
    )

    assert captured["batch_to_args"] == ("cpu", "bfloat16")
    assert captured["generate_kwargs"]["input_features"].dtype == "bfloat16"
    assert result.raw == "decoded subtitle text"


def test_music_flamingo_attempt_caps_max_new_tokens_to_context_budget(monkeypatch, tmp_path):
    import numpy as np

    from providers.base import ProviderContext, PromptContext
    from providers.local_alm.music_flamingo_local import MusicFlamingoLocalProvider

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

        def get(self, key, default=None):
            return self._data.get(key, default)

    class FakeProcessor:
        @staticmethod
        def apply_chat_template(_conversation, **_kwargs):
            return FakeBatchFeature(
                {
                    "input_ids": np.arange(300, dtype=np.int64).reshape(1, 300),
                    "input_features": FakeFeatureTensor("bfloat16"),
                }
            )

        @staticmethod
        def batch_decode(_token_ids, **_kwargs):
            return ["decoded subtitle text"]

    class FakeModel:
        device = "cpu"
        dtype = "bfloat16"
        config = SimpleNamespace(max_position_embeddings=1200)

        @staticmethod
        def generate(**kwargs):
            captured["generate_kwargs"] = kwargs
            return np.array([[1, 2, 3, 7, 8]], dtype=np.int64)

    console_buffer = io.StringIO()
    ctx = ProviderContext(
        console=Console(file=console_buffer, force_terminal=False),
        config={"music_flamingo_local": {"max_new_tokens": 1500}},
        args=SimpleNamespace(alm_model="music_flamingo_local", wait_time=1, max_retries=3),
    )
    provider = MusicFlamingoLocalProvider(ctx)
    monkeypatch.setattr(provider, "_get_or_load_model", lambda: {"model": FakeModel(), "processor": FakeProcessor()})

    media = provider.prepare_media(str(audio_path), "audio/wav", ctx.args)
    provider.attempt(media, PromptContext(system="system prompt", user="describe this audio"))

    assert captured["generate_kwargs"]["max_new_tokens"] == 900
    assert "Capping max_new_tokens from 1500 to 900" in console_buffer.getvalue()
