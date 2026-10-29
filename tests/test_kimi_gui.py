"""Exercise rendered Kimi controls and their persisted CLI configuration."""

import importlib
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import toml
from nicegui import ui

from config.loader import load_config
from gui.utils import kimi_reasoning_config
from module.captioner import setup_parser

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def kimi_panel(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "gui"))
    module = importlib.import_module("gui.wizard.step4_caption")
    paths = [tmp_path / name for name in ("model.toml", "config.toml")]
    for path in paths:
        path.write_text(
            '[kimi_vl]\nreasoning_effort = "high"\nthinking = "enabled"\n'
            '[kimi_code]\nreasoning_effort = "high"\nthinking = "enabled"\n',
            encoding="utf-8",
        )
    for name in ("load_kimi_reasoning_effort", "load_kimi_thinking"):
        monkeypatch.setattr(module, name, partial(getattr(kimi_reasoning_config, name), config_path=paths[0]))
    for name in ("save_kimi_reasoning_effort", "save_kimi_thinking"):
        monkeypatch.setattr(
            module,
            name,
            partial(getattr(kimi_reasoning_config, name), model_config_path=paths[0], legacy_config_path=paths[1]),
        )

    step = module.CaptionStep()
    step.API_CONFIGS = {"Kimi-Code": step.API_CONFIGS["Kimi-Code"]}
    monkeypatch.setattr(step, "_render_codex_settings", lambda: None)
    monkeypatch.setattr(step, "_render_grok_build_settings", lambda: None)
    with ui.column() as container:
        step._render_api_settings()
    yield step, paths
    container.delete()


def test_kimi_model_labels_keep_wire_ids_and_switch_controls(kimi_panel):
    step, _ = kimi_panel
    model = step.kimi_code_api_key_model
    assert model.options["kimi-for-coding"] == "kimi-for-coding (K2.8 Preview)"
    assert model.options["kimi-for-coding-highspeed"] == "kimi-for-coding-highspeed (K2.7 Code Highspeed)"
    assert model.value == "k3-256k"

    model.set_value("kimi-for-coding")
    assert model.value == "kimi-for-coding"
    assert step._kimi_reasoning_effort_containers["Kimi-Code"].visible
    assert not step._kimi_thinking_containers["Kimi-Code"].visible
    assert step.kimi_code_reasoning_effort.options == {"low": "low", "high": "high", "max": "max"}

    step.kimi_code_reasoning_effort.set_value("max")
    model.set_value("kimi-for-coding-highspeed")
    assert not step._kimi_reasoning_effort_containers["Kimi-Code"].visible
    assert step._kimi_thinking_containers["Kimi-Code"].visible
    step.kimi_code_thinking.set_value("disabled")

    model.set_value("kimi-for-coding")
    assert step._kimi_reasoning_effort_containers["Kimi-Code"].visible
    assert not step._kimi_thinking_containers["Kimi-Code"].visible
    assert step.kimi_code_reasoning_effort.value == "max"
    assert step.config["kimi_code_reasoning_effort"] == "max"


@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_rendered_kimi_effort_persists_for_captioner_cli(kimi_panel, effort):
    step, paths = kimi_panel
    step.kimi_code_api_key_model.set_value("kimi-for-coding")
    # Exercise a value-change event even for the shipped high default.
    step.kimi_code_reasoning_effort.set_value("low")
    step.kimi_code_reasoning_effort.set_value(effort)
    for path in paths:
        assert toml.load(path)["kimi_code"]["reasoning_effort"] == effort
        assert toml.load(path)["kimi_vl"]["reasoning_effort"] == "high"

    step.api_keys["kimi_code_api_key"].set_value("test-key")
    step.ocr_model = SimpleNamespace(value="")
    step.vlm_image_model = SimpleNamespace(value="")
    step.alm_model = SimpleNamespace(value="")
    step.mode = SimpleNamespace(value="long")
    step.pair_dir = SimpleNamespace(value="")
    step.config["dir_name"] = True
    argv = step._build_caption_args("test-dataset")
    parsed = setup_parser().parse_args(argv)
    assert parsed.kimi_code_model_path == "kimi-for-coding"
    assert parsed.mode == "long"
    assert parsed.dir_name is True
    assert not any("reasoning_effort" in arg for arg in argv)
    assert load_config(str(paths[0].parent))["kimi_code"]["reasoning_effort"] == effort
