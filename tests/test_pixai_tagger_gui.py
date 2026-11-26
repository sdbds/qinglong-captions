import asyncio
import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_tagger_page_renders_and_switches_models_without_inference_dependencies():
    code = textwrap.dedent("""\
        import sys
        from pathlib import Path

        for name in ("numpy", "torch", "torchvision", "onnxruntime", "transformers"):
            sys.modules[name] = None

        from gui.path_setup import configure_sys_path
        configure_sys_path(Path.cwd())
        from nicegui import ui
        from wizard.step3_tagger import TaggerStep

        step = TaggerStep()
        with ui.column() as container:
            step.render()
        try:
            for repo_id in ("bdsqlsz/pixai-tagger-v1.0-ONNX", "pixai-labs/pixai-tagger-v1.0"):
                step.repo_id.set_value(repo_id)
                assert step.repo_id.value == repo_id, (repo_id, step.repo_id.value)
                assert step.config["batch_size"] == 1
                assert step.config["general_threshold"] == 0.17
                assert not step.general_threshold_row.visible
            step.repo_id.set_value("KaraKaraWitch/haruka-tagger-large")
            assert step.config["batch_size"] == 1
            assert step.config["artist_threshold"] == 0.25
            assert not step.general_threshold_row.visible
            step.repo_id.set_value("cella110n/cl_tagger_v2")
            assert step.config["general_threshold"] == 0.55
            assert step.general_threshold_row.visible
            step.repo_id.set_value("SmilingWolf/wd-vit-tagger-v3")
            assert step.config["general_threshold"] == 0.35
            assert "module.wdtagger.pixai" not in sys.modules
            assert "module.wdtagger.haruka" not in sys.modules
            assert "utils.wdtagger_siglip2" not in sys.modules
        finally:
            container.delete()
        """)
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def load_step(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "gui"))
    spec = importlib.util.spec_from_file_location("pixai_tagger_gui_test", ROOT / "gui/wizard/step3_tagger.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TaggerStep


def test_selecting_pixai_applies_category_defaults_and_small_batch(monkeypatch):
    step = load_step(monkeypatch)()
    assert hasattr(step, "_on_model_change"), "Model selection must update tagger defaults"
    assert "bdsqlsz/pixai-tagger-v1.0-ONNX" in step.DEFAULT_MODELS
    step._on_model_change("bdsqlsz/pixai-tagger-v1.0-ONNX")
    assert step.config["batch_size"] == 1
    assert step.config["general_threshold"] == 0.17
    assert step.config["character_threshold"] == 0.27
    assert step.config["style_threshold"] == 0.15
    step._on_model_change("cella110n/cl_tagger_v2")
    assert step.config["general_threshold"] == 0.55
    assert step.config["character_threshold"] == 0.55


@pytest.mark.parametrize("use_model_defaults", [True, False])
@pytest.mark.parametrize("repo,category", [
    ("bdsqlsz/pixai-tagger-v1.0-ONNX", "style"),
    ("pixai-labs/pixai-tagger-v1.0", "style"),
    ("KaraKaraWitch/haruka-tagger-large", "year"),
])
def test_gui_launch_preserves_model_specific_thresholds(monkeypatch, tmp_path, use_model_defaults, repo, category):
    step = load_step(monkeypatch)()
    assert hasattr(step, "_on_model_change"), "Model selection must update tagger defaults"
    assert repo in step.DEFAULT_MODELS
    step._on_model_change(repo)
    step.config[f"{category}_threshold"] = 0.0
    step.config["use_model_thresholds"] = use_model_defaults
    for key, value in {
        "train_data_dir": str(tmp_path),
        "repo_id": repo,
        "cl_tagger_v2_version": "v2_01a",
        "model_dir": "wd14_tagger_model",
        "undesired_tags": "",
        "always_first_tags": "",
        "tag_replacement": "",
    }.items():
        setattr(step, key, SimpleNamespace(value=value))
    calls = []

    async def run_job(script, args, **kwargs):
        calls.append((script, args))

    step.panel = SimpleNamespace(run_job=run_job)
    asyncio.run(step._start_tagging())
    assert calls[0][0] == "utils.wdtagger"
    assert f"--repo_id={repo}" in calls[0][1]
    assert "--batch_size=1" in calls[0][1]
    if use_model_defaults:
        assert not any("_threshold=" in arg for arg in calls[0][1])
    else:
        assert f"--{category}_threshold=0.0" in calls[0][1]
        if category == "year":
            assert "--artist_threshold=0.25" in calls[0][1]
            assert not any(arg.startswith("--style_threshold=") for arg in calls[0][1])
        else:
            assert "--character_threshold=0.27" in calls[0][1]
    assert not any(arg.startswith("--thresh=") for arg in calls[0][1])


def test_model_defaults_hide_manual_threshold_controls(monkeypatch):
    step = load_step(monkeypatch)()
    visibility = {}
    step.repo_id = SimpleNamespace(value="bdsqlsz/pixai-tagger-v1.0-ONNX")
    step.general_threshold_row = SimpleNamespace(set_visibility=lambda value: visibility.update(general=value))
    step.category_threshold_row = SimpleNamespace(set_visibility=lambda value: visibility.update(categories=value))
    step._update_threshold_visibility()
    assert visibility == {"general": False, "categories": False}
    step.config["use_model_thresholds"] = False
    step._update_threshold_visibility()
    assert visibility == {"general": True, "categories": True}


def test_rendered_model_select_dispatches_repository_value(monkeypatch):
    from nicegui import ui

    step = load_step(monkeypatch)()
    with ui.column() as container:
        step.render()
    try:
        step.repo_id.set_value("bdsqlsz/pixai-tagger-v1.0-ONNX")
        assert step.config["batch_size"] == 1
        assert step.general_threshold_row.visible is False
        assert step.category_threshold_row.visible is False
        step.config["use_model_thresholds"] = False
        step.repo_id.set_value("KaraKaraWitch/haruka-tagger-large")
        assert step.category_threshold_row.visible
        assert step.category_controls["artist"].visible and step.category_controls["year"].visible
        assert not step.category_controls["style"].visible and not step.category_controls["meta"].visible
    finally:
        container.delete()


@pytest.mark.parametrize("repo,profile", [
    ("bdsqlsz/pixai-tagger-v1.0-ONNX", "wdtagger-pixai"),
    ("pixai-labs/pixai-tagger-v1.0", "wdtagger-pixai"),
    ("KaraKaraWitch/haruka-tagger-large", "onnx-base"),
])
def test_runtime_selects_onnx_profile_without_legacy_opencv(repo, profile):
    from gui.utils.process_runner import ProcessRunner

    extra = ProcessRunner._resolve_wdtagger_extra(
        "utils.wdtagger", ["images", f"--repo_id={repo}"], "wdtagger", None
    )
    assert extra == profile
    assert ProcessRunner._profile_uses_torch([extra], [])
    assert ProcessRunner._profile_uses_torchvision([extra], [])
    assert ProcessRunner._windows_opencv_override_profile([extra]) is None
