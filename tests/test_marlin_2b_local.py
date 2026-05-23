from __future__ import annotations

import io
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))

from module.providers.base import CaptionResult, MediaContext, MediaModality, PromptContext, ProviderContext
from module.providers.local_vlm.marlin_2b_local import Marlin2BLocalProvider
from tests.provider_v2_helpers import make_provider_args


def _make_provider(*, config=None, args=None) -> Marlin2BLocalProvider:
    return Marlin2BLocalProvider(
        ProviderContext(
            console=Console(file=io.StringIO(), force_terminal=False),
            config=config or {},
            args=args or make_provider_args(vlm_image_model="marlin_2b_local"),
        )
    )


def _video_media(video_path: Path) -> MediaContext:
    return MediaContext(
        uri=str(video_path),
        mime="video/mp4",
        sha256hash="",
        modality=MediaModality.VIDEO,
    )


def _image_media(image_path: Path, *, mime: str = "image/avif") -> MediaContext:
    return MediaContext(
        uri=str(image_path),
        mime=mime,
        sha256hash="",
        modality=MediaModality.IMAGE,
        blob="image-blob",
    )


def test_marlin_2b_model_toml_defaults():
    model_config = tomllib.loads((ROOT / "config" / "model.toml").read_text(encoding="utf-8"))
    marlin = model_config["marlin_2b_local"]

    assert marlin["model_id"] == "NemoStation/Marlin-2B"
    assert marlin["task"] == "caption"
    assert marlin["video_max_seconds"] == 120
    assert marlin["model_list"]["Marlin 2B"]["meta"]["min_vram_gb"] == 6


def test_marlin_2b_can_handle_video_and_image():
    args = make_provider_args(vlm_image_model="marlin_2b_local")

    assert Marlin2BLocalProvider.can_handle(args, "video/mp4") is True
    assert Marlin2BLocalProvider.can_handle(args, "image/jpeg") is True
    assert Marlin2BLocalProvider.can_handle(args, "image/avif") is True


def test_marlin_2b_prepare_media_rejects_over_limit(tmp_path, monkeypatch):
    video_path = tmp_path / "long.mp4"
    video_path.write_bytes(b"video")
    provider = _make_provider(config={"marlin_2b_local": {"video_max_seconds": 120}})

    monkeypatch.setattr("module.providers.local_vlm.marlin_2b_local.get_video_duration", lambda _: 121_000)

    try:
        provider.prepare_media(str(video_path), "video/mp4", provider.ctx.args)
    except RuntimeError as exc:
        assert "MARLIN2B_VIDEO_TOO_LONG" in str(exc)
    else:
        raise AssertionError("Expected long Marlin videos to be rejected before model inference")


def test_marlin_2b_prepare_media_registers_image_decoders(tmp_path, monkeypatch):
    image_path = tmp_path / "sample.avif"
    image_path.write_bytes(b"image")
    provider = _make_provider(args=make_provider_args(vlm_image_model="marlin_2b_local"))
    called = []

    monkeypatch.setattr(provider, "_register_optional_image_decoders", lambda: called.append(True))
    monkeypatch.setattr("module.providers.local_vlm_base.encode_image_to_blob", lambda *args, **kwargs: ("blob", "pixels"))

    media = provider.prepare_media(str(image_path), "image/avif", provider.ctx.args)

    assert called == [True]
    assert media.blob == "blob"
    assert media.pixels == "pixels"


def test_marlin_2b_caption_uses_canonical_prompt_by_default(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    provider = _make_provider(args=make_provider_args(vlm_image_model="marlin_2b_local"))
    captured = {}

    class FakeModel:
        def caption(self, path, **kwargs):
            captured["path"] = path
            captured["kwargs"] = kwargs
            return {
                "caption": "Scene: room\nEvents:\n<0.0 - 1.0> someone enters",
                "scene": "room",
                "events": [{"start": "0.0", "end": "1.0", "description": "someone enters"}],
            }

    fake_torch = SimpleNamespace(inference_mode=lambda: nullcontext())
    monkeypatch.setattr(
        provider,
        "_get_or_load_model",
        lambda: {"model": FakeModel(), "torch": fake_torch, "model_loader": "FakeLoader"},
    )

    result = provider.attempt(_video_media(video_path), PromptContext(system="system", user="generic video prompt"))

    assert captured["path"] == str(video_path.resolve())
    assert "prompt" not in captured["kwargs"]
    assert captured["kwargs"]["max_new_tokens"] == 2048
    assert result.parsed["caption_extension"] == ".txt"
    assert result.parsed["provider"] == "marlin_2b_local"
    assert result.parsed["events"][0]["start"] == 0.0
    assert result.raw.startswith("Scene: room")


def test_marlin_2b_caption_wraps_image_as_temporary_video(tmp_path, monkeypatch):
    image_path = tmp_path / "sample.avif"
    image_path.write_bytes(b"image")
    provider = _make_provider(args=make_provider_args(vlm_image_model="marlin_2b_local"))
    captured = {}

    def fake_write_image_as_video(image_uri, output_path):
        captured["image_uri"] = image_uri
        Path(output_path).write_bytes(b"video")

    class FakeModel:
        def caption(self, path, **kwargs):
            captured["path"] = path
            captured["exists_during_caption"] = Path(path).exists()
            return {"caption": "single frame caption", "scene": "", "events": []}

    fake_torch = SimpleNamespace(inference_mode=lambda: nullcontext())
    monkeypatch.setattr(provider, "_write_image_as_video", fake_write_image_as_video)
    monkeypatch.setattr(
        provider,
        "_get_or_load_model",
        lambda: {"model": FakeModel(), "torch": fake_torch, "model_loader": "FakeLoader"},
    )

    result = provider.attempt(_image_media(image_path), PromptContext(system="system", user="caption this image"))

    assert captured["image_uri"] == str(image_path)
    assert captured["path"].endswith(".mp4")
    assert captured["exists_during_caption"] is True
    assert result.raw == "single frame caption"


def test_marlin_2b_find_rejects_image_media(tmp_path, monkeypatch):
    image_path = tmp_path / "sample.avif"
    image_path.write_bytes(b"image")
    provider = _make_provider(
        config={"marlin_2b_local": {"task": "find", "find_event": "person enters"}},
        args=make_provider_args(vlm_image_model="marlin_2b_local"),
    )
    monkeypatch.setattr(
        provider,
        "_get_or_load_model",
        lambda: (_ for _ in ()).throw(AssertionError("image find should reject before model load")),
    )

    try:
        provider.attempt(_image_media(image_path), PromptContext(system="", user="ignored"))
    except RuntimeError as exc:
        assert "MARLIN2B_FIND_REQUIRES_VIDEO" in str(exc)
    else:
        raise AssertionError("Expected Marlin temporal grounding to reject image media")


def test_marlin_2b_find_rejects_image_media_before_openai_runtime(tmp_path, monkeypatch):
    image_path = tmp_path / "sample.avif"
    image_path.write_bytes(b"image")
    provider = _make_provider(
        config={"marlin_2b_local": {"task": "find", "find_event": "person enters"}},
        args=make_provider_args(vlm_image_model="marlin_2b_local"),
    )
    monkeypatch.setattr(
        provider,
        "get_runtime_backend",
        lambda: SimpleNamespace(is_openai=True, mode="openai", model_id="runtime/marlin"),
    )
    monkeypatch.setattr(
        provider,
        "attempt_via_openai_backend",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("image find should reject before OpenAI runtime")
        ),
    )

    try:
        provider.attempt(_image_media(image_path), PromptContext(system="", user="ignored"))
    except RuntimeError as exc:
        assert "MARLIN2B_FIND_REQUIRES_VIDEO" in str(exc)
    else:
        raise AssertionError("Expected Marlin temporal grounding to reject image media")


def test_marlin_2b_caption_allows_configured_prompt_override(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    provider = _make_provider(
        config={"marlin_2b_local": {"prompt": "custom dense caption prompt", "max_new_tokens": 128}},
        args=make_provider_args(vlm_image_model="marlin_2b_local"),
    )
    captured = {}

    class FakeModel:
        def caption(self, path, **kwargs):
            captured["kwargs"] = kwargs
            return {"caption": "custom caption", "scene": "", "events": []}

    fake_torch = SimpleNamespace(inference_mode=lambda: nullcontext())
    monkeypatch.setattr(
        provider,
        "_get_or_load_model",
        lambda: {"model": FakeModel(), "torch": fake_torch, "model_loader": "FakeLoader"},
    )

    provider.attempt(_video_media(video_path), PromptContext(system="", user="ignored"))

    assert captured["kwargs"]["prompt"] == "custom dense caption prompt"
    assert captured["kwargs"]["max_new_tokens"] == 128


def test_marlin_2b_find_mode_returns_temporal_grounding_payload(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    provider = _make_provider(
        config={"marlin_2b_local": {"task": "find", "find_event": "a person enters"}},
        args=make_provider_args(vlm_image_model="marlin_2b_local"),
    )
    captured = {}

    class FakeModel:
        def find(self, path, *, event, **kwargs):
            captured["path"] = path
            captured["event"] = event
            captured["kwargs"] = kwargs
            return {"raw": "From 1.5 to 2.5.", "span": (1.5, 2.5), "format_ok": True}

    fake_torch = SimpleNamespace(inference_mode=lambda: nullcontext())
    monkeypatch.setattr(
        provider,
        "_get_or_load_model",
        lambda: {"model": FakeModel(), "torch": fake_torch, "model_loader": "FakeLoader"},
    )

    result = provider.attempt(_video_media(video_path), PromptContext(system="", user="ignored"))

    assert captured["path"] == str(video_path.resolve())
    assert captured["event"] == "a person enters"
    assert result.parsed["task_kind"] == "temporal_grounding"
    assert result.parsed["span"] == [1.5, 2.5]
    assert result.parsed["caption_extension"] == ".txt"


def test_marlin_2b_openai_runtime_returns_structured_text_payload(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    provider = _make_provider(args=make_provider_args(vlm_image_model="marlin_2b_local"))

    monkeypatch.setattr(
        provider,
        "get_runtime_backend",
        lambda: SimpleNamespace(is_openai=True, mode="openai", model_id="runtime/marlin"),
    )
    monkeypatch.setattr(
        provider,
        "attempt_via_openai_backend",
        lambda *_args, **_kwargs: CaptionResult(
            raw="<think>scratch</think>\nruntime video caption",
            metadata={"runtime_backend": "openai", "runtime_model_id": "runtime/marlin"},
        ),
    )

    result = provider.attempt(_video_media(video_path), PromptContext(system="", user="caption"))

    assert result.raw == "runtime video caption"
    assert result.parsed == {
        "caption": "runtime video caption",
        "scene": "",
        "events": [],
        "description": "runtime video caption",
        "task_kind": "caption",
        "caption_extension": ".txt",
        "provider": "marlin_2b_local",
    }
    assert result.metadata["structured"] is True
    assert result.metadata["runtime_model_id"] == "runtime/marlin"


def test_marlin_2b_load_model_uses_hf_remote_code_and_bfloat16_on_cuda():
    import importlib

    captured = {}

    class FakeModel:
        def eval(self):
            return self

    class FakeLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return FakeModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = FakeLoader
    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float32 = "float32"

    provider = _make_provider(config={"marlin_2b_local": {"model_id": "NemoStation/Marlin-2B"}})

    with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}):
        transformer_loader_module = importlib.import_module("utils.transformer_loader")
        original_resolver = transformer_loader_module.resolve_device_dtype
        transformer_loader_module.resolve_device_dtype = lambda: ("cuda", fake_torch.float32, "eager")
        try:
            cached = provider._load_model()
        finally:
            transformer_loader_module.resolve_device_dtype = original_resolver

    assert cached["device"] == "cuda"
    assert isinstance(cached["model"], FakeModel)
    assert captured["args"] == ("NemoStation/Marlin-2B",)
    assert captured["kwargs"]["trust_remote_code"] is True
    assert captured["kwargs"]["dtype"] == fake_torch.bfloat16
    assert captured["kwargs"]["device_map"] == {"": "cuda"}
