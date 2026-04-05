from __future__ import annotations

import json
import sys
import types
from io import StringIO
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from rich.console import Console
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))

from module.providers.base import PromptContext, ProviderContext
from module.providers.local_vlm.gemma4_local import Gemma4LocalProvider
from tests.provider_v2_helpers import make_provider_args


def _make_provider(*, args=None, config=None) -> Gemma4LocalProvider:
    context = ProviderContext(
        console=Console(file=StringIO(), force_terminal=False),
        config=config or {},
        args=args or make_provider_args(),
    )
    return Gemma4LocalProvider(context)


def test_gemma4_defaults_to_e2b_model_id():
    provider = _make_provider(config={"gemma4_local": {}})

    assert provider.model_id == "google/gemma-4-E2B-it"


def test_gemma4_explicit_model_id_overrides_config_model_id():
    args = make_provider_args(gemma4_model_id="custom/gemma4-local")
    provider = _make_provider(args=args, config={"gemma4_local": {"model_id": "google/gemma-4-E4B-it"}})

    assert provider.model_id == "custom/gemma4-local"


def test_gemma4_uses_configured_model_id():
    provider = _make_provider(
        config={"gemma4_local": {"model_id": "protoLabsAI/gemma-4-E4B-it-FP8"}},
    )

    assert provider.model_id == "protoLabsAI/gemma-4-E4B-it-FP8"


def test_gemma4_accepts_unknown_explicit_model_id():
    provider = _make_provider(
        args=make_provider_args(gemma4_model_id="custom/gemma4-local"),
        config={"gemma4_local": {"model_id": ""}},
    )

    assert provider.model_id == "custom/gemma4-local"


def test_gemma4_backfills_missing_chat_template_from_official_repo(tmp_path, monkeypatch):
    provider = _make_provider(
        config={"gemma4_local": {"model_id": "livadies/gemma-4-31B-Ghetto-NF4"}},
    )
    template_path = tmp_path / "chat_template.jinja"
    template_path.write_text("{{ messages }}", encoding="utf-8")
    captured = {}

    class FakeTokenizer:
        chat_template = None

    class FakeProcessor:
        chat_template = None
        tokenizer = FakeTokenizer()

    def fake_hf_hub_download(repo_id, filename):
        captured["repo_id"] = repo_id
        captured["filename"] = filename
        return str(template_path)

    monkeypatch.setattr("module.providers.local_vlm.gemma4_local.hf_hub_download", fake_hf_hub_download)

    processor = FakeProcessor()
    provider._ensure_processor_chat_template(processor, provider.model_id)

    assert captured == {
        "repo_id": "google/gemma-4-31B-it",
        "filename": "chat_template.jinja",
    }
    assert processor.chat_template == "{{ messages }}"
    assert processor.tokenizer.chat_template == "{{ messages }}"


def test_gemma4_keeps_existing_chat_template_without_download(monkeypatch):
    provider = _make_provider(
        config={"gemma4_local": {"model_id": "google/gemma-4-31B-it"}},
    )

    class FakeTokenizer:
        chat_template = "existing-template"

    class FakeProcessor:
        chat_template = "existing-template"
        tokenizer = FakeTokenizer()

    def fail_hf_hub_download(*_args, **_kwargs):
        raise AssertionError("hf_hub_download should not be called when chat template already exists")

    monkeypatch.setattr("module.providers.local_vlm.gemma4_local.hf_hub_download", fail_hf_hub_download)

    processor = FakeProcessor()
    provider._ensure_processor_chat_template(processor, provider.model_id)

    assert processor.chat_template == "existing-template"
    assert processor.tokenizer.chat_template == "existing-template"


def test_gemma4_model_toml_lists_large_variants():
    model_config = tomllib.loads((ROOT / "config" / "model.toml").read_text(encoding="utf-8"))
    model_list = model_config["gemma4_local"]["model_list"]

    assert model_list["Gemma 4 26B A4B it"]["model_id"] == "google/gemma-4-26B-A4B-it"
    assert model_list["Gemma 4 26B A4B it"]["meta"]["min_vram_gb"] == 48
    assert model_list["Gemma 4 31B it"]["model_id"] == "google/gemma-4-31B-it"
    assert model_list["Gemma 4 31B it"]["meta"]["min_vram_gb"] == 64
    assert (
        model_list["Gemma 4 31B it FP8 Block"]["model_id"]
        == "RedHatAI/gemma-4-31B-it-FP8-block"
    )
    assert model_list["Gemma 4 31B it FP8 Block"]["meta"]["min_vram_gb"] == 32


def test_gemma4_attempt_delegates_input_preparation_to_common_helper(tmp_path, monkeypatch):
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"png")
    provider = _make_provider(args=make_provider_args(vlm_image_model="gemma4_local"))
    monkeypatch.setattr("module.providers.local_vlm.gemma4_local.encode_image_to_blob", lambda *_args, **_kwargs: ("blob", "pixels"))
    media = provider.prepare_media(str(image_path), "image/png", provider.ctx.args)
    captured = {}

    class FakeTokenizer:
        def decode(self, new_tokens, skip_special_tokens=True):
            return "ok"

    class FakeProcessor:
        tokenizer = FakeTokenizer()

    class FakeModel:
        def generate(self, **kwargs):
            return np.array([[1, 2, 3]])

    def _fake_prepare_multimodal_inputs(processor, messages, **kwargs):
        captured["processor"] = processor
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return {"input_ids": np.array([[1, 2]])}

    fake_torch = SimpleNamespace(inference_mode=lambda: nullcontext())
    monkeypatch.setattr("module.providers.local_vlm.gemma4_local.prepare_multimodal_inputs", _fake_prepare_multimodal_inputs)
    monkeypatch.setattr(
        provider,
        "_get_or_load_model",
        lambda: {
            "model": FakeModel(),
            "processor": FakeProcessor(),
            "torch": fake_torch,
            "device": "cuda",
            "dtype": "bfloat16",
            "model_loader": "FakeLoader",
        },
    )

    result = provider.attempt(media, PromptContext(system="system", user="caption"))

    payload = json.loads(result.raw)
    assert payload["description"] == "ok"
    assert payload["scores"] == {}
    assert payload["average_score"] == 0.0
    assert payload["caption_extension"] == ".txt"
    assert result.parsed == payload
    assert result.metadata["structured"] is True
    assert captured["kwargs"]["device"] == "cuda"
    assert captured["kwargs"]["dtype"] == "bfloat16"
    assert captured["kwargs"]["chat_template_kwargs"] == {}


def test_gemma4_openai_runtime_wraps_single_image_as_structured_payload(tmp_path, monkeypatch):
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"png")
    provider = _make_provider(args=make_provider_args(vlm_image_model="gemma4_local"))
    monkeypatch.setattr("module.providers.local_vlm.gemma4_local.encode_image_to_blob", lambda *_args, **_kwargs: ("blob", "pixels"))
    media = provider.prepare_media(str(image_path), "image/png", provider.ctx.args)

    class FakeBackend:
        def __init__(self, runtime):
            self.runtime = runtime

        def complete(self, messages):
            assert any(part.get("type") == "image_url" for part in messages[1]["content"])
            return "openai runtime caption"

    monkeypatch.setattr(
        provider,
        "get_runtime_backend",
        lambda: SimpleNamespace(is_openai=True, mode="openai", model_id="runtime/gemma4"),
    )
    monkeypatch.setattr("module.providers.local_vlm.gemma4_local.OpenAIChatRuntime", FakeBackend)

    result = provider.attempt(media, PromptContext(system="system", user="caption"))

    payload = json.loads(result.raw)
    assert payload["description"] == "openai runtime caption"
    assert payload["scores"] == {}
    assert payload["average_score"] == 0.0
    assert result.parsed == payload
    assert result.metadata["structured"] is True


def test_gemma4_openai_runtime_preserves_structured_json_image_payload(tmp_path, monkeypatch):
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"png")
    provider = _make_provider(args=make_provider_args(vlm_image_model="gemma4_local"))
    monkeypatch.setattr("module.providers.local_vlm.gemma4_local.encode_image_to_blob", lambda *_args, **_kwargs: ("blob", "pixels"))

    structured_response = {
        "description": "json desc",
        "scores": {
            "Costume & Makeup & Prop Presentation/Accuracy (in the Photo)": 9,
            "Setting & Environment Integration": 8,
            "Storytelling & Concept": 8,
        },
        "average_score": 8.4,
        "character_name": "Yae Miko",
        "series": "Genshin Impact",
    }

    class FakeBackend:
        def __init__(self, runtime):
            self.runtime = runtime

        def complete(self, messages):
            assert any(part.get("type") == "image_url" for part in messages[1]["content"])
            return json.dumps(structured_response)

    monkeypatch.setattr(
        provider,
        "get_runtime_backend",
        lambda: SimpleNamespace(is_openai=True, mode="openai", model_id="runtime/gemma4"),
    )
    monkeypatch.setattr("module.providers.local_vlm.gemma4_local.OpenAIChatRuntime", FakeBackend)

    result = provider.attempt(provider.prepare_media(str(image_path), "image/png", provider.ctx.args), PromptContext(system="system", user="caption"))

    payload = json.loads(result.raw)
    assert payload["description"] == "json desc"
    assert payload["scores"] == {
        "Costume & Makeup & Prop Presentation/Accuracy": 9,
        "Setting & Environment Integration": 5,
        "Storytelling & Concept": 5,
    }
    assert payload["average_score"] == pytest.approx(8.4)
    assert payload["character_name"] == "Yae Miko"
    assert payload["series"] == "Genshin Impact"
    assert payload["caption_extension"] == ".txt"
    assert payload["provider"] == "gemma4_local"
    assert result.parsed == payload
    assert result.metadata["structured"] is True


def test_gemma4_openai_runtime_extracts_scores_from_freeform_image_caption(tmp_path, monkeypatch):
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"png")
    provider = _make_provider(args=make_provider_args(vlm_image_model="gemma4_local"))
    monkeypatch.setattr("module.providers.local_vlm.gemma4_local.encode_image_to_blob", lambda *_args, **_kwargs: ("blob", "pixels"))
    media = provider.prepare_media(str(image_path), "image/png", provider.ctx.args)

    freeform_caption = """**Detailed Visual Description:**

The image shows a futuristic cosplayer under neon lights.

**Scores:**

1. Costume & Makeup & Prop Presentation/Accuracy (in the Photo): 9
2. Character Portrayal & Posing (Captured by the Photographer): 9
3. Setting & Environment Integration: 8
4. Lighting & Mood: 10
5. Composition & Framing (Serving the Cosplay): 9
6. Storytelling & Concept: 8
7. Level of S**y: 8
8. Figure: 8
9. Overall Impact & Uniqueness: 9

**Total Score:** 78
**Average Score:** 8.67
"""

    class FakeBackend:
        def __init__(self, runtime):
            self.runtime = runtime

        def complete(self, messages):
            assert any(part.get("type") == "image_url" for part in messages[1]["content"])
            return freeform_caption

    monkeypatch.setattr(
        provider,
        "get_runtime_backend",
        lambda: SimpleNamespace(is_openai=True, mode="openai", model_id="runtime/gemma4"),
    )
    monkeypatch.setattr("module.providers.local_vlm.gemma4_local.OpenAIChatRuntime", FakeBackend)

    result = provider.attempt(media, PromptContext(system="system", user="caption"))

    payload = json.loads(result.raw)
    assert payload["description"] == "The image shows a futuristic cosplayer under neon lights."
    assert payload["scores"] == {
        "Costume & Makeup & Prop Presentation/Accuracy": 9,
        "Character Portrayal & Posing": 9,
        "Setting & Environment Integration": 5,
        "Lighting & Mood": 10,
        "Composition & Framing": 9,
        "Storytelling & Concept": 5,
        "Level of S*e*x*y": 8,
        "Figure": 8,
        "Overall Impact & Uniqueness": 9,
    }
    assert payload["average_score"] == pytest.approx(8.67)
    assert result.parsed == payload
    assert result.metadata["structured"] is True


def test_gemma4_execute_displays_extracted_rating_for_freeform_image_caption(tmp_path, monkeypatch):
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"png")
    console = Console(file=StringIO(), force_terminal=False)
    provider = Gemma4LocalProvider(
        ProviderContext(
            console=console,
            config={"prompts": {}},
            args=make_provider_args(vlm_image_model="gemma4_local"),
        )
    )
    monkeypatch.setattr("module.providers.local_vlm.gemma4_local.encode_image_to_blob", lambda *_args, **_kwargs: ("blob", "pixels"))

    freeform_caption = """**Detailed Visual Description:**

The image shows a futuristic cosplayer under neon lights.

**Scores:**

1. Costume & Makeup & Prop Presentation/Accuracy (in the Photo): 9
2. Character Portrayal & Posing (Captured by the Photographer): 9
3. Setting & Environment Integration: 8
4. Lighting & Mood: 10
5. Composition & Framing (Serving the Cosplay): 9
6. Storytelling & Concept: 8
7. Level of S**y: 8
8. Figure: 8
9. Overall Impact & Uniqueness: 9

**Total Score:** 78
**Average Score:** 8.67
"""

    class FakeBackend:
        def __init__(self, runtime):
            self.runtime = runtime

        def complete(self, messages):
            assert any(part.get("type") == "image_url" for part in messages[1]["content"])
            return freeform_caption

    monkeypatch.setattr(
        provider,
        "get_runtime_backend",
        lambda: SimpleNamespace(is_openai=True, mode="openai", model_id="runtime/gemma4"),
    )
    monkeypatch.setattr("module.providers.local_vlm.gemma4_local.OpenAIChatRuntime", FakeBackend)

    with patch("module.providers.base.display_caption_and_rate") as mocked_display:
        result = provider.execute(str(image_path), "image/png", "hash-123")

    mocked_display.assert_called_once()
    kwargs = mocked_display.call_args.kwargs
    assert kwargs["title"] == "frame.png"
    assert kwargs["long_description"] == "The image shows a futuristic cosplayer under neon lights."
    assert kwargs["rating"] == {
        "Costume & Makeup & Prop Presentation/Accuracy": 9,
        "Character Portrayal & Posing": 9,
        "Setting & Environment Integration": 5,
        "Lighting & Mood": 10,
        "Composition & Framing": 9,
        "Storytelling & Concept": 5,
        "Level of S*e*x*y": 8,
        "Figure": 8,
        "Overall Impact & Uniqueness": 9,
    }
    assert kwargs["average_score"] == pytest.approx(8.67)
    assert result.parsed["scores"]["Lighting & Mood"] == 10


def test_gemma4_load_model_prefers_sdpa_when_cuda_resolves_flash_attention(monkeypatch):
    from module.providers.local_vlm import gemma4_local as gemma4_module

    captured = {}

    class FakeModel:
        def eval(self):
            return self

    class FakeProcessor:
        chat_template = "template"
        tokenizer = SimpleNamespace(chat_template="template")

    def fake_load_pretrained_component(component_cls, model_id, **kwargs):
        component_name = kwargs.get("component_name")
        captured[f"{component_name}_cls"] = component_cls
        captured[f"{component_name}_model_id"] = model_id
        captured[f"{component_name}_kwargs"] = kwargs
        if component_name == "processor":
            return FakeProcessor()
        return FakeModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = object
    fake_transformers.AutoModelForMultimodalLM = object
    fake_transformers.AutoModelForImageTextToText = object
    fake_transformers.AutoModelForVision2Seq = object
    fake_transformers.AutoModelForCausalLM = object

    fake_torch = types.ModuleType("torch")
    fake_torch.nn = SimpleNamespace(functional=SimpleNamespace(scaled_dot_product_attention=object()))

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(gemma4_module, "load_pretrained_component", fake_load_pretrained_component)
    monkeypatch.setattr(gemma4_module, "resolve_device_dtype", lambda *args, **kwargs: ("cuda", "bfloat16", "flash_attention_2"))

    cached = _make_provider()._load_model()

    assert isinstance(cached["processor"], FakeProcessor)
    assert isinstance(cached["model"], FakeModel)
    assert captured["model via object_kwargs"]["device_map"] == "auto"
    assert captured["model via object_kwargs"]["attn_implementation"] == "sdpa"


def test_gemma4_audio_prompt_uses_task_specific_keys(tmp_path):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"audio")
    provider = _make_provider(
        args=make_provider_args(alm_model="gemma4_local", audio_task="ast"),
        config={
            "prompts": {
                "audio_system_prompt": "generic-system",
                "audio_prompt": "generic-user",
                "gemma4_audio_ast_system_prompt": "ast-system",
                "gemma4_audio_ast_prompt": "ast-user",
            },
            "gemma4_local": {"audio_task": "asr"},
        },
    )

    prompts = provider.resolve_prompts(str(audio_path), "audio/wav")

    assert prompts.system == "ast-system"
    assert prompts.user == "ast-user"


def test_gemma4_prepare_media_rejects_audio_over_limit(tmp_path, monkeypatch):
    audio_path = tmp_path / "long.wav"
    audio_path.write_bytes(b"audio")
    provider = _make_provider(
        args=make_provider_args(alm_model="gemma4_local", audio_task="asr"),
        config={"gemma4_local": {"audio_max_seconds": 30}},
    )
    monkeypatch.setattr("module.providers.local_vlm.gemma4_local.get_video_duration", lambda _: 31_000)

    with pytest.raises(RuntimeError, match="GEMMA4_AUDIO_TOO_LONG"):
        provider.prepare_media(str(audio_path), "audio/wav", provider.ctx.args)


@pytest.mark.parametrize(
    "model_id",
    [
        "google/gemma-4-26B-A4B-it",
        "google/gemma-4-31B-it",
        "livadies/gemma-4-31B-Ghetto-NF4",
    ],
)
def test_gemma4_prepare_media_rejects_audio_for_non_audio_variants(tmp_path, model_id):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"audio")
    provider = _make_provider(
        args=make_provider_args(alm_model="gemma4_local", audio_task="asr"),
        config={"gemma4_local": {"model_id": model_id}},
    )

    with pytest.raises(RuntimeError, match="GEMMA4_AUDIO_UNSUPPORTED_MODEL"):
        provider.prepare_media(str(audio_path), "audio/wav", provider.ctx.args)


def test_gemma4_prepare_media_rejects_video_over_limit(tmp_path, monkeypatch):
    video_path = tmp_path / "long.mp4"
    video_path.write_bytes(b"video")
    provider = _make_provider(
        args=make_provider_args(vlm_image_model="gemma4_local"),
        config={"gemma4_local": {"video_max_seconds": 60}},
    )
    monkeypatch.setattr("module.providers.local_vlm.gemma4_local.get_video_duration", lambda _: 61_000)

    with pytest.raises(RuntimeError, match="GEMMA4_VIDEO_TOO_LONG"):
        provider.prepare_media(str(video_path), "video/mp4", provider.ctx.args)


def test_gemma4_prepare_media_ignores_audio_task_on_visual_inputs(tmp_path):
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A0000000D4948445200000001000000010802000000907753DE"
            "0000000C49444154789C6360000000020001E221BC330000000049454E44AE426082"
        )
    )
    provider = _make_provider(
        args=make_provider_args(vlm_image_model="gemma4_local", audio_task="ast"),
        config={"gemma4_local": {"audio_task": "asr"}},
    )

    media = provider.prepare_media(str(image_path), "image/png", provider.ctx.args)

    assert media.modality.name == "IMAGE"


def test_gemma4_direct_runtime_normalizes_ast_result(tmp_path, monkeypatch):
    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"audio")
    provider = _make_provider(
        args=make_provider_args(alm_model="gemma4_local", audio_task="ast"),
        config={"gemma4_local": {"runtime_backend": "direct", "audio_task": "ast"}},
    )
    monkeypatch.setattr("module.providers.local_vlm.gemma4_local.get_video_duration", lambda _: 10_000)
    media = provider.prepare_media(str(audio_path), "audio/wav", provider.ctx.args)

    captured_messages: list[dict] = []

    class FakeTokenizer:
        def decode(self, new_tokens, skip_special_tokens=True):
            return "```srt\n1\n00:00:00,000 --> 00:00:01,000\n你好\n```"

    class FakeProcessor:
        tokenizer = FakeTokenizer()

        def apply_chat_template(self, messages, **kwargs):
            captured_messages.extend(messages)
            return {"input_ids": np.array([[1, 2]])}

    class FakeModel:
        def generate(self, **kwargs):
            return np.array([[1, 2, 3, 4]])

    fake_torch = SimpleNamespace(inference_mode=lambda: nullcontext())
    monkeypatch.setattr(
        provider,
        "_get_or_load_model",
        lambda: {
            "model": FakeModel(),
            "processor": FakeProcessor(),
            "torch": fake_torch,
            "device": "cpu",
            "dtype": None,
            "model_loader": "FakeLoader",
        },
    )

    result = provider.attempt(media, PromptContext(system="system", user="transcribe"))

    assert captured_messages[0]["role"] == "system"
    assert captured_messages[1]["content"][0]["type"] == "audio"
    assert result.raw == "1\n00:00:00,000 --> 00:00:01,000\n你好"
    assert result.parsed["task_kind"] == "ast"
    assert result.parsed["translation_srt"] == "1\n00:00:00,000 --> 00:00:01,000\n你好"
