import builtins
import subprocess
import sys
import types
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
    OPENCV_DISTRIBUTION_PACKAGES,
    WdtaggerOpenCvInstallPlan,
    WdtaggerOpenCvSelection,
    build_wdtagger_opencv_install_plan,
    main,
    parse_nvcc_cuda_version_tag,
    probe_cv2_runtime,
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
    assert selection.package_name == DEFAULT_OPENCV_REQUIREMENT
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
    assert selection.package_name == "opencv-contrib-python"
    assert selection.package_spec == CUDA_OPENCV_REQUIREMENTS["cu129"]
    assert selection.source == "cuda-wheel"


def test_resolve_wdtagger_windows_opencv_requirement_uses_rolling_distribution_name_for_cu130(monkeypatch):
    monkeypatch.setattr("utils.wdtagger_opencv.shutil.which", lambda _name, path=None: "C:/CUDA/bin/nvcc.exe")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="Cuda compilation tools, release 13.0, V13.0.12",
            stderr="",
        )

    monkeypatch.setattr("utils.wdtagger_opencv.subprocess.run", fake_run)

    selection = resolve_wdtagger_windows_opencv_requirement(platform="win32")

    assert selection.cuda_tag == "cu130"
    assert selection.package_name == "opencv-contrib-python-rolling"
    assert selection.package_spec == CUDA_OPENCV_REQUIREMENTS["cu130"]
    assert selection.source == "cuda-wheel"


def test_pyproject_declares_default_windows_opencv_contrib_for_wdtagger():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    optional_deps = pyproject["project"]["optional-dependencies"]["wdtagger"]

    assert "opencv-contrib-python; sys_platform == 'win32'" in optional_deps
    assert not any(dep.startswith("transformers") for dep in optional_deps)


def test_opencv_distribution_cleanup_list_covers_conflicting_cv2_wheels():
    assert "opencv-python" in OPENCV_DISTRIBUTION_PACKAGES
    assert "opencv-contrib-python" in OPENCV_DISTRIBUTION_PACKAGES
    assert "opencv-contrib-python-rolling" in OPENCV_DISTRIBUTION_PACKAGES


def test_build_wdtagger_opencv_install_plan_adds_cpu_fallback_for_gpu(monkeypatch):
    monkeypatch.setattr(
        "utils.wdtagger_opencv.resolve_wdtagger_windows_opencv_requirement",
        lambda env=None, platform=None: WdtaggerOpenCvSelection(
            package_name="opencv-contrib-python-rolling",
            package_spec="opencv-contrib-python-rolling @ https://example.invalid/opencv-cu130.whl",
            cuda_tag="cu130",
            source="cuda-wheel",
            detail="detected CUDA toolkit cu130",
        ),
    )

    plan = build_wdtagger_opencv_install_plan(platform="win32")

    assert plan.cleanup_packages == OPENCV_DISTRIBUTION_PACKAGES
    assert len(plan.attempts) == 2
    assert plan.attempts[0].source == "cuda-wheel"
    assert plan.attempts[1].source == "cpu-fallback"
    assert plan.attempts[1].package_spec == DEFAULT_OPENCV_REQUIREMENT


def test_main_plan_install_emits_json(capsys, monkeypatch):
    monkeypatch.setattr(
        "utils.wdtagger_opencv.build_wdtagger_opencv_install_plan",
        lambda env=None, platform=None: WdtaggerOpenCvInstallPlan(
            cleanup_packages=("opencv-python",),
            attempts=(
                WdtaggerOpenCvSelection(
                    package_name="opencv-contrib-python",
                    package_spec="opencv-contrib-python",
                    cuda_tag=None,
                    source="default",
                    detail="fallback",
                ),
            ),
        ),
    )

    exit_code = main(["--plan-install", "--platform", "win32"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"cleanup_packages": ["opencv-python"]' in captured.out
    assert '"package_name": "opencv-contrib-python"' in captured.out


def test_probe_cv2_runtime_reports_success(monkeypatch):
    sys.modules.pop("cv2", None)
    fake_cv2 = types.SimpleNamespace(
        __version__="4.13.0",
        __file__="cv2.pyd",
        cuda=types.SimpleNamespace(getCudaEnabledDeviceCount=lambda: 1),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    payload, exit_code = probe_cv2_runtime()

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["version"] == "4.13.0"
    assert payload["file"] == "cv2.pyd"
    assert payload["cuda_count"] == 1


def test_probe_cv2_runtime_reports_import_error(monkeypatch):
    sys.modules.pop("cv2", None)
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cv2":
            raise ImportError("missing cv2")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    payload, exit_code = probe_cv2_runtime()

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "ImportError: missing cv2"


def test_tagger_powershell_wrapper_installs_selected_wdtagger_opencv():
    content = (ROOT / "3.tagger.ps1").read_text(encoding="utf-8")

    assert 'Install-UvExtraPatch @("wdtagger")' in content
    assert "resolve_wdtagger_windows_opencv_requirement" not in content
    assert "function Get-WdtaggerOpenCvRequirement" not in content
    assert "function Install-WdtaggerOpenCvOverride" not in content
    assert "function Get-WdtaggerOpenCvInstallPlan" in content
    assert "--plan-install --platform win32" in content
    assert "nvcc -V" not in content
    assert "4.12.0.86-cp37-abi3-win_amd64.whl" not in content
    assert "4.12.0.88-cp37-abi3-win_amd64.whl" not in content
    assert "4.13.0.20250812-cp37-abi3-win_amd64.whl" not in content
    assert "uv pip uninstall" in content
    assert "wdtagger OpenCV cleanup: removing conflicting cv2 wheels" in content
    assert "wdtagger OpenCV import probe succeeded" in content
    assert "wdtagger OpenCV GPU import probe failed" in content
    assert "wdtagger_opencv.py" in content
    assert "wdtagger_opencv_probe.py" not in content
    assert "--probe-cv2" in content
    assert 'Write-Output "uv pip install target package: $($Attempt.package_name)"' in content
    assert "wdtagger OpenCV package spec" in content


def test_audio_separator_powershell_wrapper_uses_vocal_midi_without_wdtagger_flow():
    content = (ROOT / "2.5.audio_separator.ps1").read_text(encoding="utf-8")

    assert 'Install-UvExtraPatch @("vocal-midi")' in content
    assert 'Install-UvExtraPatch @("wdtagger")' not in content
    assert "resolve_wdtagger_windows_opencv_requirement" not in content
    assert "wdtagger OpenCV package spec" not in content
    assert 'Write-Output "runtime dependency profile: extra:vocal-midi"' in content
