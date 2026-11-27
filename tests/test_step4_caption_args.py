from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def caption_step_cls(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "gui"))
    return importlib.import_module("gui.wizard.step4_caption").CaptionStep


def _new_step(caption_step_cls, *, ocr="", vlm="", alm="", mode="all"):
    step = caption_step_cls()
    step.api_keys = {}
    step.ocr_model = SimpleNamespace(value=ocr)
    step.vlm_image_model = SimpleNamespace(value=vlm)
    step.alm_model = SimpleNamespace(value=alm)
    step.mode = SimpleNamespace(value=mode)
    step.pair_dir = SimpleNamespace(value="")
    step.scene_detector = SimpleNamespace(value="AdaptiveDetector")
    return step


@pytest.mark.parametrize(
    ("ocr", "vlm", "alm", "expected"),
    [
        ("", "penguin_vl_local", "", ["--extra", "penguin-vl-local"]),
        ("", "lfm_vl_local", "", ["--extra", "lfm-vl-local"]),
        ("", "marlin_2b_local", "", ["--extra", "marlin-2b-local"]),
        ("lighton_ocr", "", "", ["--extra", "lighton-ocr"]),
        ("dots_ocr", "", "", ["--extra", "dots-ocr"]),
        ("qianfan_ocr", "", "", ["--extra", "qianfan-ocr"]),
        ("infinity_parser2_ocr", "", "", ["--extra", "infinity-parser2-ocr"]),
        ("logics_ocr", "", "", ["--extra", "logics-ocr"]),
        ("", "", "music_flamingo_local", ["--extra", "music-flamingo-local"]),
        ("", "", "eureka_audio_local", ["--extra", "eureka-audio-local"]),
        ("", "", "acestep_transcriber_local", ["--extra", "acestep-transcriber-local"]),
        ("", "", "cohere_transcribe_local", ["--extra", "cohere-transcribe-local"]),
        ("", "", "mega_asr_local", ["--extra", "mega-asr-local"]),
        ("", "gemma4_local", "gemma4_local", ["--extra", "gemma4-local"]),
    ],
)
def test_caption_step_local_extra_routing(caption_step_cls, ocr, vlm, alm, expected):
    step = _new_step(caption_step_cls, ocr=ocr, vlm=vlm, alm=alm)
    assert step._build_local_extra_args() == expected
    assert step._has_local_route_config() is True


def test_caption_step_quantized_repo_detection_and_extra(caption_step_cls):
    assert caption_step_cls._repo_id_requires_quantized_runtime("livadies/gemma-4-31B-Ghetto-NF4") is True
    assert caption_step_cls._repo_id_requires_quantized_runtime("custom/model-bnb-4bit") is True
    assert caption_step_cls._repo_id_requires_quantized_runtime("RedHatAI/gemma-4-31B-it-FP8-block") is True
    assert caption_step_cls._repo_id_requires_quantized_runtime("prithivMLmods/gemma-4-E4B-it-FP8") is False
    assert caption_step_cls._repo_id_requires_quantized_runtime("google/gemma-4-31B-it") is False

    step = _new_step(caption_step_cls, vlm="gemma4_local")
    step._selected_local_route_model_ids = lambda: ("livadies/gemma-4-31B-Ghetto-NF4",)
    assert step._build_local_extra_args() == ["--extra", "gemma4-local", "--extra", "quantized-runtime"]


def test_caption_step_unlimited_ocr_extra_follows_model_config(caption_step_cls, tmp_path, monkeypatch):
    from gui.utils import toml_helpers

    model_toml = tmp_path / "model.toml"
    model_toml.write_text('[unlimited_ocr]\nmodel_id = "baidu/Unlimited-OCR-2"\n', encoding="utf-8")
    monkeypatch.setattr(toml_helpers, "_MODEL_CONFIG_PATH", model_toml)

    step = _new_step(caption_step_cls, ocr="unlimited_ocr")
    assert step._build_local_extra_args() == ["--extra", "unlimited-ocr2"]

    model_toml.write_text('[unlimited_ocr]\nmodel_id = "baidu/Unlimited-OCR"\n', encoding="utf-8")
    assert step._build_local_extra_args() == ["--extra", "unlimited-ocr"]

    model_toml.write_text('[unlimited_ocr]\nmodel_id = "local/renamed"\nmodel_version = "2"\n', encoding="utf-8")
    assert step._build_local_extra_args() == ["--extra", "unlimited-ocr2"]


def test_caption_step_codex_and_grok_subscription_routing_and_args(caption_step_cls):
    codex = _new_step(caption_step_cls, ocr="dots_ocr", vlm="gemma4_local", alm="music_flamingo_local", mode="long")
    codex.config["codex_subscription"] = True
    codex.config["codex_fast"] = True
    codex.config["codex_reasoning_effort"] = "minimal"
    codex.api_keys = {"openai_api_key": SimpleNamespace(value="sk-should-not-pass")}

    assert codex._has_caption_provider_config() is True
    assert codex._build_local_extra_args() == ["--extra", "codex-subscription"]
    codex_args = codex._build_caption_args("demo-dataset")
    assert "--codex_subscription" in codex_args
    assert "--codex_fast" in codex_args
    assert "--codex_reasoning_effort=minimal" in codex_args
    assert not any(arg.startswith("--openai_api_key=") for arg in codex_args)

    codex.config["codex_auth_mode"] = "api_key"
    codex.config["codex_api_key"] = "sk-explicit"
    assert "--codex_api_key=sk-explicit" in codex._build_caption_args("demo-dataset")

    grok = _new_step(caption_step_cls, ocr="dots_ocr", vlm="gemma4_local", alm="music_flamingo_local", mode="long")
    grok.config["grok_build_subscription"] = True
    grok.config["grok_build_disable_web_search"] = False
    grok.config["grok_build_reasoning_effort"] = "none"
    grok.config["grok_build_model_name"] = "grok-build-custom"
    grok.api_keys = {"openai_api_key": SimpleNamespace(value="sk-should-not-pass")}

    assert grok._has_caption_provider_config() is True
    assert grok._build_local_extra_args() == []
    grok_args = grok._build_caption_args("demo-dataset")
    assert "--grok_build_subscription" in grok_args
    assert "--grok_build_model_name=grok-build-custom" in grok_args
    assert "--grok_build_reasoning_effort=none" in grok_args
    assert "--no-grok_build_disable_web_search" in grok_args
    assert not any(arg.startswith("--openai_api_key=") for arg in grok_args)
    assert not any(arg.startswith("--ocr_model=") for arg in grok_args)


def test_caption_step_grok_45_drops_xhigh_effort(caption_step_cls):
    class FakeSelect:
        def __init__(self, value):
            self.value = value
            self.options = {}

        def set_options(self, options, *, value=None):
            self.options = dict(options)
            self.value = value

    step = caption_step_cls()
    step.grok_build_reasoning_effort = FakeSelect("xhigh")
    step._handle_grok_build_model_change("grok-4.5")
    assert "xhigh" not in step.grok_build_reasoning_effort.options
    assert step.grok_build_reasoning_effort.value == "high"

    step._handle_grok_build_model_change("grok-4.6")
    assert step.grok_build_reasoning_effort.options["xhigh"] == "xhigh"


def test_caption_step_alm_controls_segment_time_and_log_redaction(caption_step_cls):
    class FakeSelect:
        def __init__(self):
            self.value = None
            self.options = {}

        def set_options(self, options, *, value=None):
            self.options = dict(options)
            self.value = value

    class FakeSlider:
        def __init__(self):
            self.value = None

        def set_value(self, value):
            self.value = value

    step = _new_step(caption_step_cls, alm="cohere_transcribe_local")
    step.alm_language = SimpleNamespace(value="ja")
    args = step._build_caption_args("demo-dataset")
    assert "--alm_model=cohere_transcribe_local" in args
    assert "--alm_language=ja" in args
    assert not any(arg.startswith("--segment_time=") for arg in args)

    step.config["segment_time"] = 90
    step.config["segment_time_explicit"] = True
    assert "--segment_time=90" in step._build_caption_args("demo-dataset")

    step.alm_model = SimpleNamespace(value="gemma4_local")
    step.alm_language = SimpleNamespace(value=None)
    step.alm_audio_task = SimpleNamespace(value="ast")
    assert "--audio_task=ast" in step._build_caption_args("demo-dataset")

    step.config["segment_time_explicit"] = False
    step.alm_language = FakeSelect()
    step.alm_audio_task = FakeSelect()
    step.segment_time_slider = FakeSlider()
    step._handle_alm_model_change("music_flamingo_local")
    assert step.alm_language.options == {}
    assert step.segment_time_slider.value == 1200

    formatted = caption_step_cls._format_args_for_log(
        ["demo-dataset", "--openai_api_key=super-secret", "--segment_time=90"]
    )
    assert "--openai_api_key=***" in formatted
    assert "super-secret" not in formatted
