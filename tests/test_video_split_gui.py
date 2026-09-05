import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_video_split_step(module_name: str):
    module_path = ROOT / "gui" / "wizard" / "step2_video_split.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    gui_path = str(ROOT / "gui")
    original_sys_path = list(sys.path)
    if gui_path not in sys.path:
        sys.path.insert(0, gui_path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_sys_path
    return module.VideoSplitStep


@pytest.mark.parametrize(
    ("detector", "threshold", "advanced_arg"),
    [
        ("AdaptiveDetector", 3.5, "--adaptive_window_width=3"),
        ("ContentDetector", 31.0, None),
        ("HashDetector", 0.35, "--hash_size=8"),
        ("HistogramDetector", 0.2, "--histogram_bins=128"),
        ("ThresholdDetector", 12.0, None),
    ],
)
def test_video_split_gui_builds_v071_args(detector, threshold, advanced_arg):
    VideoSplitStep = _load_video_split_step(f"test_video_split_{detector}")
    step = VideoSplitStep()
    step.detector = SimpleNamespace(value=detector)
    step.config["threshold"] = threshold

    args = step._build_args("videos", "output")

    assert args[0] == "videos"
    assert "--output_dir=output" in args
    assert "--backend=pyav" in args
    assert "--min_scene_len_seconds=0.6" in args
    assert f"--threshold={threshold}" in args
    if detector == "AdaptiveDetector":
        assert not any(arg.startswith("--detector=") for arg in args)
    else:
        assert f"--detector={detector}" in args

    advanced_args = [
        arg
        for arg in args
        if arg.startswith(
            ("--adaptive_window_width=", "--hash_size=", "--histogram_bins=")
        )
    ]
    assert advanced_args == ([advanced_arg] if advanced_arg else [])


def test_video_split_gui_defaults_match_project_profiles():
    VideoSplitStep = _load_video_split_step("test_video_split_defaults")
    step = VideoSplitStep()

    assert step.DEFAULT_THRESHOLDS == {
        "AdaptiveDetector": 3.5,
        "ContentDetector": 31.0,
        "HashDetector": 0.35,
        "HistogramDetector": 0.2,
        "ThresholdDetector": 12.0,
    }
    assert step.config["backend"] == "pyav"
    assert step.config["min_scene_len_seconds"] == 0.6
    assert step.config["adaptive_window_width"] == 3
    assert step.config["hash_size"] == 8
    assert step.config["histogram_bins"] == 128


def test_video_split_gui_switches_detector_defaults_and_advanced_visibility():
    VideoSplitStep = _load_video_split_step("test_video_split_visibility")
    step = VideoSplitStep()

    class FakeRow:
        def __init__(self):
            self.visible = None

        def set_visibility(self, visible):
            self.visible = visible

    step.advanced_rows = {
        "AdaptiveDetector": FakeRow(),
        "HashDetector": FakeRow(),
        "HistogramDetector": FakeRow(),
    }

    step._on_detector_change("HashDetector")

    assert step.config["threshold"] == 0.35
    assert step.advanced_rows["AdaptiveDetector"].visible is False
    assert step.advanced_rows["HashDetector"].visible is True
    assert step.advanced_rows["HistogramDetector"].visible is False


def test_video_split_v071_labels_exist_in_all_supported_languages():
    from gui.utils.i18n import TRANSLATIONS

    keys = {
        "video_backend",
        "min_scene_len_seconds",
        "adaptive_window_width",
        "hash_size",
        "histogram_bins",
        "log_video_backend",
        "log_min_scene_len_seconds",
    }
    for language in ("en", "zh", "ja", "ko"):
        assert keys <= TRANSLATIONS[language].keys()
