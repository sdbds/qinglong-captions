from __future__ import annotations

from pathlib import Path

import toml

from gui.utils.process_runner import _UV_TORCH_EXTRAS, SCRIPT_REGISTRY

ROOT = Path(__file__).resolve().parents[1]


def _project() -> dict:
    return toml.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_muscriptor_extra_is_pinned_and_uses_shared_torch_profile():
    dependencies = _project()["project"]["optional-dependencies"]["muscriptor-local"]

    assert "qinglong-captions[torch-base]" in dependencies
    assert "muscriptor==0.2.1" in dependencies
    assert any(item.startswith("filelock") for item in dependencies)


def test_muscriptor_extra_conflicts_with_native_paddle_stack():
    conflicts = _project()["tool"]["uv"]["conflicts"]
    conflicting_extras = [
        {entry["extra"] for entry in group if "extra" in entry}
        for group in conflicts
    ]

    assert {"paddleocr-native", "muscriptor-local"} in conflicting_extras


def test_registry_runs_cli_in_module_mode_with_torch_patch_profile():
    assert SCRIPT_REGISTRY["module.muscriptor_tool.cli"] == (
        "-m:module.muscriptor_tool.cli",
        "muscriptor-local",
    )
    assert "muscriptor-local" in _UV_TORCH_EXTRAS


def test_config_has_closed_official_model_and_no_soundfont_source():
    config = toml.loads((ROOT / "config/model.toml").read_text(encoding="utf-8"))["muscriptor"]

    assert config["model"] == "large"
    assert config["device"] == "auto"
    assert config["batch_size"] == 0
    assert config["output_formats"] == ["midi"]
    assert config["preview_mode"] == "none"
    assert config["preview_format"] == "mp3"
    assert config["output_dir"] == ""
    assert "overwrite" not in config
    assert "model_source" not in config
    assert "weights_path" not in config
    assert "soundfont_path" not in config


def test_powershell_wrapper_uses_incremental_extra_install_and_forwards_arguments():
    wrapper = (ROOT / "2.7.music_transcription.ps1").read_text(encoding="utf-8")

    assert "muscriptor-local" in wrapper
    assert '"pip"' in wrapper
    assert '"install"' in wrapper
    assert '"-r"' in wrapper
    assert '"pyproject.toml"' in wrapper
    assert "& $PythonExe -m module.muscriptor_tool.cli batch" in wrapper
    assert "@Arguments" in wrapper
    assert "uv run" not in wrapper
    assert "Read-Host" not in wrapper
