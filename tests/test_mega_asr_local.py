# -*- coding: utf-8 -*-

import io
import sys
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


def _write_mega_asr_checkpoint_layout(ckpt_dir: Path) -> None:
    (ckpt_dir / "Qwen3-ASR-1.7B").mkdir(parents=True)
    (ckpt_dir / "Qwen3-ASR-1.7B" / "config.json").write_text("{}", encoding="utf-8")
    (ckpt_dir / "mega-asr-merged").mkdir()
    (ckpt_dir / "mega-asr-merged" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (ckpt_dir / "mega-asr-merged" / "adapter_model.safetensors").write_bytes(b"fake")
    (ckpt_dir / "audio_quality_router").mkdir()
    (ckpt_dir / "audio_quality_router" / "best_acc_model.safetensors").write_bytes(b"fake")


def test_mega_asr_defaults_to_official_weights_repo():
    from module.providers.local_alm.mega_asr_local import MegaASRLocalProvider

    model_config = tomllib.loads((ROOT / "config" / "model.toml").read_text(encoding="utf-8"))

    assert MegaASRLocalProvider.default_model_id == "zhifeixie/Mega-ASR"
    assert model_config["mega_asr_local"]["model_id"] == "zhifeixie/Mega-ASR"
    assert "source_repo_url" not in model_config["mega_asr_local"]
    assert "source_dir" not in model_config["mega_asr_local"]
    assert "auto_clone_source" not in model_config["mega_asr_local"]


def test_mega_asr_can_handle_audio_only():
    from module.providers.local_alm.mega_asr_local import MegaASRLocalProvider

    args = SimpleNamespace(alm_model="mega_asr_local")

    assert MegaASRLocalProvider.can_handle(args, "audio/wav") is True
    assert MegaASRLocalProvider.can_handle(args, "image/jpeg") is False


def test_mega_asr_load_uses_direct_model_paths(monkeypatch, tmp_path):
    from module.providers.base import ProviderContext
    from module.providers.local_alm import mega_asr_local
    from module.providers.local_alm.mega_asr_local import MegaASRLocalProvider

    ckpt_dir = tmp_path / "ckpt" / "Mega-ASR"
    _write_mega_asr_checkpoint_layout(ckpt_dir)

    captured = {}

    class FakeMegaASRDirectModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={
            "mega_asr_local": {
                "ckpt_dir": str(ckpt_dir),
                "routing": False,
                "threshold": 0.7,
                "device_map": "cpu",
                "quality_device": "cpu",
                "max_inference_batch_size": 4,
                "max_new_tokens": 128,
                "keep_delta_on_gpu": False,
                "dtype": "float32",
            }
        },
        args=SimpleNamespace(alm_model="mega_asr_local"),
    )
    provider = MegaASRLocalProvider(ctx)
    monkeypatch.setattr(mega_asr_local, "_MegaASRDirectModel", FakeMegaASRDirectModel)

    cached = provider._load_model()

    assert isinstance(cached["model"], FakeMegaASRDirectModel)
    assert captured["model_path"] == ckpt_dir / "Qwen3-ASR-1.7B"
    assert captured["lora_dir"] == ckpt_dir / "mega-asr-merged"
    assert captured["router_checkpoint"] == ckpt_dir / "audio_quality_router" / "best_acc_model.safetensors"
    assert captured["routing_enabled"] is False
    assert captured["quality_threshold"] == 0.7
    assert captured["device_map"] == "cpu"
    assert captured["quality_device"] == "cpu"
    assert captured["max_inference_batch_size"] == 4
    assert captured["max_new_tokens"] == 128
    assert captured["keep_delta_on_gpu"] is False
    assert captured["dtype"] == "float32"


def test_mega_asr_load_reports_missing_direct_dependency(monkeypatch, tmp_path):
    from module.providers.base import ProviderContext
    from module.providers.local_alm import mega_asr_local
    from module.providers.local_alm.mega_asr_local import MegaASRLocalProvider

    ckpt_dir = tmp_path / "ckpt" / "Mega-ASR"
    _write_mega_asr_checkpoint_layout(ckpt_dir)

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={
            "mega_asr_local": {
                "ckpt_dir": str(ckpt_dir),
            }
        },
        args=SimpleNamespace(alm_model="mega_asr_local"),
    )
    provider = MegaASRLocalProvider(ctx)

    def fake_direct_model(**kwargs):
        raise ModuleNotFoundError("No module named 'qwen_asr'", name="qwen_asr")

    monkeypatch.setattr(mega_asr_local, "_MegaASRDirectModel", fake_direct_model)

    with pytest.raises(RuntimeError, match="dependency import failed: qwen_asr"):
        provider._load_model()


def test_mega_asr_attempt_returns_transcript_with_route_metadata(monkeypatch, tmp_path):
    from module.providers.base import PromptContext, ProviderContext
    from module.providers.local_alm.mega_asr_local import MegaASRLocalProvider

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")
    captured = {}

    class FakeMegaASR:
        def infer(self, audio, **kwargs):
            captured["audio"] = audio
            captured["kwargs"] = kwargs
            return {
                "text": "hello from mega asr",
                "use_lora": True,
                "degraded_prob": 0.91,
                "route_source": "router",
            }

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={"mega_asr_local": {"language": "zh"}},
        args=SimpleNamespace(alm_model="mega_asr_local", alm_language="en", wait_time=1, max_retries=3),
    )
    provider = MegaASRLocalProvider(ctx)
    monkeypatch.setattr(provider, "_get_or_load_model", lambda: {"model": FakeMegaASR()})

    media = provider.prepare_media(str(audio_path), "audio/wav", ctx.args)
    result = provider.attempt(media, PromptContext(system="ignored", user="ignored"))

    assert captured["audio"] == str(audio_path.resolve())
    assert captured["kwargs"] == {"language": "en", "return_route": True}
    assert result.raw == "hello from mega asr"
    assert result.metadata == {
        "provider": "mega_asr_local",
        "use_lora": True,
        "degraded_prob": 0.91,
        "route_source": "router",
    }


def test_mega_asr_extracts_single_qwen_asr_result_text():
    from module.providers.local_alm.mega_asr_local import _extract_transcribe_text

    result = _extract_transcribe_text([SimpleNamespace(text="hello from qwen asr")])

    assert result == "hello from qwen asr"


def test_mega_asr_post_validate_returns_structured_transcript_payload(tmp_path):
    from module.providers.base import CaptionResult, ProviderContext
    from module.providers.local_alm.mega_asr_local import MegaASRLocalProvider

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={},
        args=SimpleNamespace(alm_model="mega_asr_local", wait_time=1, max_retries=3),
    )
    provider = MegaASRLocalProvider(ctx)
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
        "provider": "mega_asr_local",
    }


def test_mega_asr_declares_transcribe_task_contract():
    from module.providers.local_alm.mega_asr_local import MegaASRLocalProvider

    assert MegaASRLocalProvider.task_contract.task_kind == "transcribe"
    assert MegaASRLocalProvider.task_contract.consumes_prompts is False
