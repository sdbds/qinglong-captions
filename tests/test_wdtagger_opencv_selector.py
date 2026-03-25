import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.wdtagger_opencv import (  # noqa: E402
    CUDA_OPENCV_REQUIREMENTS,
    DEFAULT_OPENCV_REQUIREMENT,
    parse_nvcc_cuda_version_tag,
    resolve_wdtagger_windows_opencv_requirement,
)


def test_parse_nvcc_cuda_version_tag_reads_supported_versions():
    assert parse_nvcc_cuda_version_tag("Cuda compilation tools, release 12.8, V12.8.93") == "cu128"
    assert parse_nvcc_cuda_version_tag("Cuda compilation tools, release 12.9, V12.9.41") == "cu129"
    assert parse_nvcc_cuda_version_tag("Cuda compilation tools, release 13.0, V13.0.12") == "cu130"


def test_resolve_wdtagger_windows_opencv_requirement_falls_back_when_nvcc_missing(monkeypatch):
    monkeypatch.setattr("utils.wdtagger_opencv.shutil.which", lambda _name, path=None: None)

    selection = resolve_wdtagger_windows_opencv_requirement(platform="win32")

    assert selection.cuda_tag is None
    assert selection.package_spec == DEFAULT_OPENCV_REQUIREMENT
    assert selection.source == "default"


def test_resolve_wdtagger_windows_opencv_requirement_uses_cuda_specific_wheel(monkeypatch):
    monkeypatch.setattr("utils.wdtagger_opencv.shutil.which", lambda _name, path=None: "C:/CUDA/bin/nvcc.exe")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="Cuda compilation tools, release 12.9, V12.9.41",
            stderr="",
        )

    monkeypatch.setattr("utils.wdtagger_opencv.subprocess.run", fake_run)

    selection = resolve_wdtagger_windows_opencv_requirement(platform="win32")

    assert selection.cuda_tag == "cu129"
    assert selection.package_spec == CUDA_OPENCV_REQUIREMENTS["cu129"]
    assert selection.source == "cuda-wheel"


def test_pyproject_declares_default_windows_opencv_contrib_for_wdtagger():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    optional_deps = pyproject["project"]["optional-dependencies"]["wdtagger"]

    assert "opencv-contrib-python; sys_platform == 'win32'" in optional_deps
    assert not any(dep.startswith("transformers") for dep in optional_deps)


def test_tagger_powershell_wrapper_installs_selected_wdtagger_opencv():
    content = (ROOT / "3.tagger.ps1").read_text(encoding="utf-8")

    assert 'Install-UvExtraPatch @("wdtagger")' in content
    assert "resolve_wdtagger_windows_opencv_requirement" not in content
    assert "function Get-WdtaggerOpenCvRequirement" not in content
    assert "function Install-WdtaggerOpenCvOverride" not in content
    assert "nvcc -V" in content
    assert "4.12.0.86-cp37-abi3-win_amd64.whl" in content
    assert "4.12.0.88-cp37-abi3-win_amd64.whl" in content
    assert "4.13.0.20250812-cp37-abi3-win_amd64.whl" in content
    assert "wdtagger OpenCV package spec" in content


def test_audio_separator_powershell_wrapper_uses_vocal_midi_without_wdtagger_flow():
    content = (ROOT / "2.5.audio_separator.ps1").read_text(encoding="utf-8")

    assert 'Install-UvExtraPatch @("vocal-midi")' in content
    assert 'Install-UvExtraPatch @("wdtagger")' not in content
    assert "resolve_wdtagger_windows_opencv_requirement" not in content
    assert "wdtagger OpenCV package spec" not in content
    assert 'Write-Output "runtime dependency profile: extra:vocal-midi"' in content
