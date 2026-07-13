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
    assert any(item.startswith("socksio") for item in dependencies)


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


def test_readmes_document_official_models_and_preview_boundary():
    readme_guides = {
        "README.md": "docs/tools/muscriptor.en.md",
        "README.zh-CN.md": "docs/tools/muscriptor.md",
    }

    for readme_path, guide_path in readme_guides.items():
        readme = (ROOT / readme_path).read_text(encoding="utf-8")
        guide = (ROOT / guide_path).read_text(encoding="utf-8")
        combined = f"{readme}\n{guide}"

        assert "MuScriptor" in readme, readme_path
        assert "FluidSynth" in readme, readme_path
        assert "fluidsynth --version" in readme, readme_path
        assert all(model in combined for model in ("small", "medium", "large")), guide_path
        assert "MIDI" in combined and "JSONL" in combined, guide_path
        assert "SoundFont" in combined, guide_path
        assert "muscriptor-local" in combined, guide_path
        assert "rouard2026muscriptoropenmodelmultiinstrument" in readme, readme_path
        assert "2607.08168" in readme, readme_path


def test_official_webui_launcher_reuses_project_venv():
    launcher = (ROOT / "2.7.1.muscriptor_webui.ps1").read_text(encoding="utf-8")

    assert '".venv\\Scripts\\python.exe"' in launcher
    assert '"muscriptor-local"' in launcher
    assert '"module.muscriptor_tool.webui"' in launcher
    assert "uvx" not in launcher.lower()
    assert '[string]$Model = "large"' in launcher
    assert '[Alias("ModelSize")]' in launcher
    assert '[int]$BatchSize = 0' in launcher
    assert '"--batch-size"' in launcher
    assert '[switch]$NoBrowser' in launcher
    assert '"$WebUrl/health"' in launcher
    assert "Start-Process $WebUrl" in launcher
    assert "Stop-Process -Id $ServerProcess.Id" in launcher
