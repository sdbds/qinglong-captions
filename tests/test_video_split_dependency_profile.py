from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent.parent


def test_pyproject_pins_scenedetect_to_v071_api_series():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "scenedetect>=0.7.1,<0.8" in pyproject["project"]["dependencies"]


def test_caption_scene_detection_remains_disabled_by_default():
    model_config = tomllib.loads(
        (ROOT / "config" / "model.toml").read_text(encoding="utf-8")
    )

    assert model_config["scene_detection"]["threshold"] == 0.0


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


def test_video_split_script_uses_v071_scene_options():
    content = (ROOT / "2.0.video_spliter.ps1").read_text(encoding="utf-8")

    assert 'backend                  = "pyav"' in content
    assert "min_scene_len_seconds    = 0.6" in content
    assert "adaptive_window_width    = 3" in content
    assert "hash_size                = 8" in content
    assert "histogram_bins           = 128" in content
    assert '"--backend=$($Config.backend)"' in content
    assert '"--min_scene_len_seconds=$($Config.min_scene_len_seconds)"' in content
    assert '"--adaptive_window_width=$($Config.adaptive_window_width)"' in content
    assert '"--hash_size=$($Config.hash_size)"' in content
    assert '"--histogram_bins=$($Config.histogram_bins)"' in content
    assert '"--min_scene_len=$($Config.min_scene_len)"' not in content
