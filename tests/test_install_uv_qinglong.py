import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_install_uv_qinglong_bootstraps_base_without_lock_generation():
    content = (ROOT / "1.install-uv-qinglong.ps1").read_text(encoding="utf-8")

    assert 'Write-Output "uv pip install dependency profile: base-only"' in content
    assert 'Write-Output "基础安装直接使用 uv pip install -r pyproject.toml，不启用任何 extra"' in content
    assert "Export-BaseRequirementsFromPyproject" not in content
    assert "Ensure-UvLockFile" not in content
    assert "uv lock --index-strategy $IndexStrategy" not in content
    assert "uv export --frozen" not in content


@pytest.mark.parametrize(
    "script_name",
    [
        "1.install-uv-qinglong.ps1",
        "2.0.video_spliter.ps1",
        "2.1.image_watermark_detect.ps1",
        "2.2.preprocess_images.ps1",
        "2.3.image_reward_model.ps1",
        "2.4.psdexport.ps1",
        "2.5.audio_separator.ps1",
        "2.6.image2psd.ps1",
        "3.tagger.ps1",
        "4.captioner.ps1",
        "5.translate.ps1",
    ],
)
def test_power_shell_entrypoints_have_a_non_windows_uv_cache_fallback(script_name):
    content = (ROOT / script_name).read_text(encoding="utf-8")

    assert '$Env:UV_CACHE_DIR = "${env:LOCALAPPDATA}/uv/cache"' not in content
    assert "elseif ($env:HOME)" in content
    assert "$($env:HOME)/.cache/uv" in content


def test_psd_pipeline_module_help_works_from_repository_root():
    result = subprocess.run(
        [sys.executable, "-m", "utils.psd_dataset_pipeline", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Process PSD folder" in result.stdout


def test_linux_pwsh_helper_uses_portable_shell_syntax():
    content = (ROOT / "0.install pwsh.sh").read_text(encoding="utf-8")

    assert content.startswith("#!/usr/bin/env bash")
    assert "command -v pwsh >/dev/null 2>&1" in content
    assert "&> /dev/null" not in content
