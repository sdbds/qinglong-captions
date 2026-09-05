# -*- coding: utf-8 -*-

import asyncio
import importlib
import io
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest
from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent


def _quiet_console():
    return Console(file=io.StringIO(), force_terminal=False, color_system=None)


def _load_videospilter(monkeypatch):
    fake_scenedetect = types.ModuleType("scenedetect")
    fake_torch = types.ModuleType("torch")

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
    fake_scenedetect.split_video_ffmpeg = lambda *args, **kwargs: 0

    fake_output = types.ModuleType("scenedetect.output")
    fake_output.save_images = lambda *args, **kwargs: {}
    fake_output.split_video_ffmpeg = lambda *args, **kwargs: 0
    fake_output.write_scene_list_html = lambda *args, **kwargs: None

    fake_video_output = types.ModuleType("scenedetect.output.video")
    fake_video_output.is_ffmpeg_available = lambda: True

    monkeypatch.setitem(sys.modules, "scenedetect", fake_scenedetect)
    monkeypatch.setitem(sys.modules, "scenedetect.output", fake_output)
    monkeypatch.setitem(sys.modules, "scenedetect.output.video", fake_video_output)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    sys.modules.pop("module.videospilter", None)
    return importlib.import_module("module.videospilter")


@pytest.mark.parametrize(
    ("detector_name", "expected_kwargs"),
    [
        (
            "AdaptiveDetector",
            {"adaptive_threshold": 3.5, "min_scene_len": 0.6, "window_width": 3},
        ),
        ("ContentDetector", {"threshold": 31.0, "min_scene_len": 0.6}),
        ("HashDetector", {"threshold": 0.35, "min_scene_len": 0.6, "size": 8}),
        (
            "HistogramDetector",
            {"threshold": 0.2, "min_scene_len": 0.6, "bins": 128},
        ),
        ("ThresholdDetector", {"threshold": 12.0, "min_scene_len": 0.6}),
    ],
)
def test_scene_detector_uses_project_profiles_and_v071_detect_options(
    monkeypatch, detector_name, expected_kwargs
):
    videospilter = _load_videospilter(monkeypatch)
    calls = []

    def fake_detect(video_path, detector, **kwargs):
        calls.append((video_path, detector, kwargs))
        return ["scene"]

    monkeypatch.setattr(videospilter, "detect", fake_detect)
    detector = videospilter.SceneDetector(detector=detector_name, console=_quiet_console())

    assert detector.detect_scenes("first.mp4") == ["scene"]
    assert detector.detect_scenes("second.mp4") == ["scene"]

    assert calls[0][0] == "first.mp4"
    assert calls[0][1].kwargs == expected_kwargs
    assert calls[0][2] == {
        "show_progress": True,
        "start_in_scene": True,
        "backend": "pyav",
    }
    assert calls[0][1] is not calls[1][1]


def test_scene_detector_preserves_legacy_frame_length(monkeypatch):
    videospilter = _load_videospilter(monkeypatch)
    created = []

    def fake_detect(_video_path, detector, **_kwargs):
        created.append(detector)
        return []

    monkeypatch.setattr(videospilter, "detect", fake_detect)
    detector = videospilter.SceneDetector(min_scene_len=124, console=_quiet_console())

    detector.detect_scenes("legacy.mp4")

    assert created[0].kwargs["min_scene_len"] == 124


def test_scene_detector_seconds_override_legacy_frames(monkeypatch):
    videospilter = _load_videospilter(monkeypatch)
    created = []
    monkeypatch.setattr(
        videospilter,
        "detect",
        lambda _path, detector, **_kwargs: created.append(detector) or [],
    )
    detector = videospilter.SceneDetector(
        min_scene_len=124,
        min_scene_len_seconds=0.8,
        console=_quiet_console(),
    )

    detector.detect_scenes("seconds.mp4")

    assert created[0].kwargs["min_scene_len"] == 0.8


def test_scene_detector_propagates_direct_detection_errors(monkeypatch):
    videospilter = _load_videospilter(monkeypatch)
    monkeypatch.setattr(
        videospilter,
        "detect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("decode failed")),
    )
    detector = videospilter.SceneDetector(console=_quiet_console())

    with pytest.raises(RuntimeError, match="decode failed"):
        detector.detect_scenes("broken.mp4")


def test_scene_detector_concurrent_results_match_sequential_results(monkeypatch):
    videospilter = _load_videospilter(monkeypatch)
    detector = videospilter.SceneDetector(console=_quiet_console())

    def fake_detect(video_path, detector, **_kwargs):
        assert detector.kwargs == {
            "adaptive_threshold": 3.5,
            "min_scene_len": 0.6,
            "window_width": 3,
        }
        return [(f"{video_path}:start", f"{video_path}:end")]

    monkeypatch.setattr(videospilter, "detect", fake_detect)
    paths = ["first.mp4", "second.mp4"]
    sequential = [detector.detect_scenes(path) for path in paths]

    with ThreadPoolExecutor(max_workers=2) as executor:
        concurrent = list(executor.map(detector.detect_scenes, paths))

    assert concurrent == sequential


def test_cli_marks_failed_video_and_continues_batch(monkeypatch, tmp_path):
    videospilter = _load_videospilter(monkeypatch)
    good_video = tmp_path / "good.mp4"
    bad_video = tmp_path / "bad.mp4"
    attempted = []

    class FakeSceneDetector:
        async def detect_scenes_async(self, video_path):
            attempted.append(Path(video_path).name)
            if video_path.endswith("bad.mp4"):
                raise RuntimeError("decode failed")
            return [("start", "end")]

    class FakeProgress:
        def __init__(self):
            self.updates = []
            self.task_count = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def add_task(self, description, **kwargs):
            task_id = self.task_count
            self.task_count += 1
            self.updates.append((task_id, {"description": description, **kwargs}))
            return task_id

        def update(self, task_id, **kwargs):
            self.updates.append((task_id, kwargs))

    progress = FakeProgress()
    args = SimpleNamespace(
        output_dir=None,
        save_html=False,
        video2images_min_number=0,
    )

    failures = asyncio.run(
        videospilter.process_videos(
            [bad_video, good_video],
            FakeSceneDetector(),
            args,
            progress,
            _quiet_console(),
        )
    )

    assert sorted(attempted) == ["bad.mp4", "good.mp4"]
    assert [path for path, _error in failures] == [bad_video]
    assert any(
        update.get("description") == "Failed: bad.mp4"
        for _task_id, update in progress.updates
    )


def test_scene_detector_timestamps_use_seconds_property(monkeypatch):
    videospilter = _load_videospilter(monkeypatch)
    detector = videospilter.SceneDetector(console=_quiet_console())

    timestamps = detector.get_timestamps(
        [(SimpleNamespace(seconds=1.25), SimpleNamespace(seconds=2.0))]
    )

    assert timestamps == [1.25]


@pytest.mark.parametrize("failure", ["exit_code", "exception", "unavailable"])
def test_cli_reports_split_failure_instead_of_success(monkeypatch, tmp_path, failure):
    videospilter = _load_videospilter(monkeypatch)
    detector = videospilter.SceneDetector(console=_quiet_console())
    video_path = tmp_path / "broken.mp4"
    monkeypatch.setattr(videospilter, "detect", lambda *_args, **_kwargs: [(0, 1), (1, 2)])

    def split(*_args, **_kwargs):
        if failure == "exception":
            raise OSError("encoder failed")
        return 1 if failure == "exit_code" else 0

    monkeypatch.setattr(videospilter, "split_video_ffmpeg", split)
    monkeypatch.setattr(
        sys.modules["scenedetect.output.video"],
        "is_ffmpeg_available",
        lambda: failure != "unavailable",
    )
    monkeypatch.setattr(videospilter, "is_ffmpeg_available", lambda: failure != "unavailable")
    updates = []
    progress = SimpleNamespace(
        add_task=lambda *_args, **_kwargs: 0,
        update=lambda _task, **kwargs: updates.append(kwargs),
    )
    args = SimpleNamespace(output_dir=None, save_html=False, video2images_min_number=0)

    result = asyncio.run(videospilter.process_video(video_path, detector, args, progress))

    assert result is not None
    assert result[0] == video_path
    assert isinstance(result[1], (RuntimeError, OSError))
    assert updates[-1]["description"] == "Failed: broken.mp4"


def test_cli_scene_detector_warns_for_legacy_frame_option(monkeypatch):
    videospilter = _load_videospilter(monkeypatch)
    args = videospilter.setup_parser().parse_args(["videos", "--min_scene_len=124"])

    with pytest.warns(FutureWarning, match="min_scene_len_seconds"):
        detector = videospilter.create_scene_detector_from_args(args, _quiet_console())

    assert detector.min_scene_len == 124


def test_cli_seconds_option_wins_over_legacy_frame_option(monkeypatch):
    videospilter = _load_videospilter(monkeypatch)
    args = videospilter.setup_parser().parse_args(
        ["videos", "--min_scene_len=124", "--min_scene_len_seconds=0.8"]
    )

    with pytest.warns(FutureWarning, match="min_scene_len_seconds"):
        detector = videospilter.create_scene_detector_from_args(args, _quiet_console())

    assert detector.min_scene_len == 0.8


@pytest.mark.parametrize("backend", ["pyav", "opencv"])
def test_cli_supports_documented_video_backends(monkeypatch, backend):
    videospilter = _load_videospilter(monkeypatch)
    args = videospilter.setup_parser().parse_args(["videos", f"--backend={backend}"])

    detector = videospilter.create_scene_detector_from_args(args, _quiet_console())

    assert detector.backend == backend


def test_cli_uses_seconds_and_advanced_defaults(monkeypatch):
    videospilter = _load_videospilter(monkeypatch)
    args = videospilter.setup_parser().parse_args(["videos"])

    detector = videospilter.create_scene_detector_from_args(args, _quiet_console())

    assert detector.min_scene_len == 0.6
    assert detector.adaptive_window_width == 3
    assert detector.hash_size == 8
    assert detector.histogram_bins == 128


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"backend": "missing"}, "backend"),
        ({"min_scene_len_seconds": -0.1}, "min_scene_len_seconds"),
        ({"adaptive_window_width": 0}, "adaptive_window_width"),
        ({"hash_size": 0}, "hash_size"),
        ({"histogram_bins": 0}, "histogram_bins"),
    ],
)
def test_scene_detector_validates_migration_options(monkeypatch, kwargs, message):
    videospilter = _load_videospilter(monkeypatch)

    with pytest.raises(ValueError, match=message):
        videospilter.SceneDetector(console=_quiet_console(), **kwargs)


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


def test_split_video_sanitizes_default_output_dir_and_creates_it(monkeypatch, tmp_path):
    videospilter = _load_videospilter(monkeypatch)
    detector = videospilter.SceneDetector(backend="pyav", console=_quiet_console())

    video_path = tmp_path / "SAM Audio, the first unified model .mp4"
    video_path.write_bytes(b"video")

    captured = {}

    def fake_open_video(path, *, backend):
        captured["open_video"] = (path, backend)
        return "video-stream"

    def fake_save_images(*, scene_list, video, output_dir, num_images):
        images_dir = Path(output_dir)
        captured["images_dir"] = images_dir
        assert images_dir.parent.name == "SAM Audio, the first unified model"
        assert images_dir.parent.exists()
        return {}

    def fake_write_scene_list_html(output_html_filename, scene_list, image_filenames=None, image_height=None, image_width=None):
        html_output = Path(output_html_filename)
        captured["html_output"] = html_output
        assert html_output.parent.exists()

    monkeypatch.setattr(videospilter, "save_images", fake_save_images)
    monkeypatch.setattr(videospilter, "write_scene_list_html", fake_write_scene_list_html)
    monkeypatch.setattr(videospilter, "open_video", fake_open_video)

    detector.split_video(
        str(video_path),
        scene_list=[("start", "end")],
        output_dir=None,
        save_html=True,
        video2images_min_number=1,
    )

    expected_base_dir = tmp_path / "SAM Audio, the first unified model"
    assert expected_base_dir.exists()
    assert captured["images_dir"] == expected_base_dir / "images"
    assert captured["html_output"] == expected_base_dir / "SAM Audio, the first unified model.html"
    assert captured["open_video"] == (str(video_path), "pyav")


def test_pyav_representative_image_stream_decodes_forward_between_frames(monkeypatch):
    videospilter = _load_videospilter(monkeypatch)

    class FakeVideoStream:
        def __init__(self):
            self.frame_number = 0
            self.seeks = []
            self.reads = []

        def reset(self):
            self.frame_number = 0

        def seek(self, target):
            self.seeks.append(target.frame_num)
            self.frame_number = target.frame_num

        def read(self, decode=True):
            self.reads.append(decode)
            self.frame_number += 1
            return "frame" if decode else True

    raw_stream = FakeVideoStream()
    stream = videospilter._ForwardReadingVideoStream(raw_stream)
    stream.reset()

    for target_frame in (10, 20, 30):
        stream.seek(SimpleNamespace(frame_num=target_frame))
        assert stream.read() == "frame"

    assert raw_stream.seeks == [10]
    assert raw_stream.reads.count(True) == 3
    assert raw_stream.reads.count(False) == 18


@pytest.mark.parametrize("backend", ["pyav", "opencv"])
def test_selected_backend_is_shared_by_detection_and_images(
    monkeypatch, tmp_path, backend
):
    videospilter = _load_videospilter(monkeypatch)
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")
    captured = {}
    stream = object()

    def fake_detect(_path, **kwargs):
        assert kwargs["detector"] is not None
        captured["detect_backend"] = kwargs["backend"]
        return [("start", "end")]

    def fake_open_video(_path, *, backend):
        captured["image_backend"] = backend
        return stream

    def fake_save_images(*, video, **_kwargs):
        captured["image_stream"] = video
        return {}

    monkeypatch.setattr(videospilter, "detect", fake_detect)
    monkeypatch.setattr(videospilter, "open_video", fake_open_video)
    monkeypatch.setattr(videospilter, "save_images", fake_save_images)
    detector = videospilter.SceneDetector(backend=backend, console=_quiet_console())

    scenes = detector.detect_scenes(str(video_path))
    detector.split_video(
        str(video_path),
        scenes,
        output_dir=tmp_path / "output",
        video2images_min_number=1,
    )

    assert captured["detect_backend"] == backend
    assert captured["image_backend"] == backend
    if backend == "pyav":
        assert captured["image_stream"]._video is stream
    else:
        assert captured["image_stream"] is stream


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
        "module.caption_pipeline.orchestrator.update_dataset_captions",
        lambda dataset, *_args, **_kwargs: dataset,
    )
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
