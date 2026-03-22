from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent.parent


def test_video_split_pyproject_extra_pins_numpy_below_2():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    optional_deps = pyproject["project"]["optional-dependencies"]

    assert "video-split" in optional_deps
    assert "numpy<2" in optional_deps["video-split"]


def test_video_split_script_installs_video_split_extra_before_running_python():
    content = (ROOT / "2.0.video_spliter.ps1").read_text(encoding="utf-8")

    assert 'Install-UvExtraPatch @("video-split")' in content
    assert 'Write-Output "runtime dependency profile: extra:video-split"' in content
    assert 'python "./module/videospilter.py"' in content
