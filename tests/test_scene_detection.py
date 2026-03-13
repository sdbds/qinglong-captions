# -*- coding: utf-8 -*-

import importlib
import io
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


def _quiet_console():
    return Console(file=io.StringIO(), force_terminal=False, color_system=None)


def _load_videospilter(monkeypatch):
    fake_scenedetect = types.ModuleType("scenedetect")

    class FakeDetector:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    fake_scenedetect.AdaptiveDetector = FakeDetector
    fake_scenedetect.ContentDetector = FakeDetector
    fake_scenedetect.HashDetector = FakeDetector
    fake_scenedetect.HistogramDetector = FakeDetector
    fake_scenedetect.ThresholdDetector = FakeDetector
    fake_scenedetect.detect = lambda *args, **kwargs: []
    fake_scenedetect.open_video = lambda *args, **kwargs: None
    fake_scenedetect.split_video_ffmpeg = lambda *args, **kwargs: []

    fake_scene_manager = types.ModuleType("scenedetect.scene_manager")
    fake_scene_manager.save_images = lambda *args, **kwargs: {}
    fake_scene_manager.write_scene_list_html = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "scenedetect", fake_scenedetect)
    monkeypatch.setitem(sys.modules, "scenedetect.scene_manager", fake_scene_manager)
    sys.modules.pop("module.videospilter", None)
    return importlib.import_module("module.videospilter")


def test_run_async_in_thread_propagates_exceptions(monkeypatch):
    videospilter = _load_videospilter(monkeypatch)
    run_async_in_thread = videospilter.run_async_in_thread

    async def boom():
        raise RuntimeError("scene boom")

    future = run_async_in_thread(boom())

    try:
        future.result(timeout=1)
        raise AssertionError("expected scene detection future to raise")
    except RuntimeError as exc:
        assert "scene boom" in str(exc)


def test_scene_detector_wait_for_detection_returns_async_result(monkeypatch):
    SceneDetector = _load_videospilter(monkeypatch).SceneDetector

    expected = ["scene-a", "scene-b"]

    async def fake_detect(self, video_path):
        assert video_path == "video.mp4"
        return expected

    monkeypatch.setattr(SceneDetector, "detect_scenes_async", fake_detect)

    detector = SceneDetector(console=_quiet_console())
    detector.start_async_detection("video.mp4")

    assert detector.wait_for_detection(timeout=1) == expected
    assert detector.is_detection_complete()
    assert detector.get_scene_list() == expected


def test_scene_detector_wait_for_detection_handles_async_failure(monkeypatch):
    SceneDetector = _load_videospilter(monkeypatch).SceneDetector

    async def fake_detect(self, video_path):
        raise RuntimeError(f"bad scene detection for {video_path}")

    monkeypatch.setattr(SceneDetector, "detect_scenes_async", fake_detect)

    detector = SceneDetector(console=_quiet_console())
    detector.start_async_detection("broken.mp4")

    assert detector.wait_for_detection(timeout=1) == []
    assert detector.is_detection_complete()
    assert detector.get_scene_list() == []


def test_captioner_waits_for_scene_detection_before_aligning(monkeypatch, tmp_path):
    from module import captioner
    videospilter = _load_videospilter(monkeypatch)

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")

    batch = pa.record_batch(
        [
            pa.array([str(video_path)]),
            pa.array([None], type=pa.large_binary()),
            pa.array(["video/mp4"]),
            pa.array([[]], type=pa.list_(pa.string())),
            pa.array([1000], type=pa.int32()),
            pa.array(["hash-1"]),
        ],
        names=["uris", "blob", "mime", "captions", "duration", "hash"],
    )

    class FakeScanner:
        def to_batches(self):
            return [batch]

    class FakeMergeInsert:
        def when_matched_update_all(self):
            return self

        def execute(self, table):
            self.table = table

    class FakeTags:
        def create(self, name, value):
            self.created = (name, value)

        def update(self, name, value):
            self.updated = (name, value)

    class FakeDataset:
        def __init__(self):
            self.tags = FakeTags()
            self.merge = FakeMergeInsert()

        def scanner(self, **kwargs):
            return FakeScanner()

        def count_rows(self):
            return 1

        def merge_insert(self, on):
            assert on == "uris"
            return self.merge

    fake_dataset = FakeDataset()
    detector_instances = []

    class FakeSceneDetector:
        def __init__(self, **kwargs):
            self.started = False
            self.waited = False
            self.aligned = False
            self.scene_list = ["scene-1"]
            self.received_scene_list = None
            detector_instances.append(self)

        def start_async_detection(self, video_path):
            self.started = True
            self.video_path = video_path

        def wait_for_detection(self, video_path=None, timeout=None):
            assert self.started
            self.waited = True
            return list(self.scene_list)

        def align_subtitle(self, subs, scene_list, console=None, segment_time=None):
            assert self.waited
            self.aligned = True
            self.received_scene_list = scene_list
            return subs

    monkeypatch.setattr(captioner, "transform2lance", lambda **kwargs: fake_dataset)
    monkeypatch.setattr(captioner, "extract_from_lance", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        captioner,
        "api_process_batch",
        lambda **kwargs: "1\n00:00:00,000 --> 00:00:01,000\nhello\n",
    )
    monkeypatch.setattr(captioner, "console", _quiet_console())
    monkeypatch.setattr(videospilter, "SceneDetector", FakeSceneDetector)

    args = SimpleNamespace(
        dataset_dir=str(tmp_path),
        gemini_api_key="",
        mistral_api_key="",
        scene_threshold=1.0,
        scene_min_len=1,
        scene_detector="AdaptiveDetector",
        scene_luma_only=False,
        segment_time=5,
        document_image=False,
        not_clip_with_caption=True,
        merge_batch_size=100,
    )

    captioner.process_batch(args, config={"prompts": {}})

    detector = detector_instances[0]
    assert detector.started
    assert detector.waited
    assert detector.aligned
    assert detector.received_scene_list == ["scene-1"]
    assert video_path.with_suffix(".srt").exists()
