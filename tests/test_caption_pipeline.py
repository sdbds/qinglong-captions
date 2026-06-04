import io
import json
import sys
from pathlib import Path

import pyarrow as pa
import pysrt
import pytest
from rich.console import Console


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _quiet_console():
    return Console(file=io.StringIO(), force_terminal=False, color_system=None)


class _FakeScanner:
    def __init__(self, rows):
        self._rows = rows

    def to_batches(self):
        return [
            {
                "uris": pa.array([row["uri"] for row in self._rows]),
                "mime": pa.array([row["mime"] for row in self._rows]),
                "duration": pa.array([row.get("duration", 1000) for row in self._rows]),
                "hash": pa.array([row.get("hash", f"hash-{index}") for index, row in enumerate(self._rows)]),
            }
        ]


class _FakeDataset:
    def __init__(self, rows):
        self._rows = rows

    def scanner(self, **_kwargs):
        return _FakeScanner(self._rows)

    def count_rows(self):
        return len(self._rows)


def _process_batch_args(tmp_path, **overrides):
    from types import SimpleNamespace

    defaults = {
        "dataset_dir": str(tmp_path),
        "gemini_api_key": "",
        "mistral_api_key": "",
        "not_clip_with_caption": True,
        "merge_batch_size": 1000,
        "segment_time": None,
        "scene_detection_timeout": None,
        "cloud_max_concurrency": 1,
        "codex_max_concurrency": 1,
        "vlm_image_model": "",
        "alm_model": "",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_rows(tmp_path, names, mime="image/png"):
    rows = []
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"fake")
        rows.append({"uri": str(path), "mime": mime, "duration": 1000})
    return rows


def _patch_process_batch_io(monkeypatch, rows, update_calls=None, sidecar_calls=None):
    update_calls = [] if update_calls is None else update_calls
    sidecar_calls = [] if sidecar_calls is None else sidecar_calls

    monkeypatch.setattr("module.caption_pipeline.orchestrator._resolve_dataset", lambda *_args, **_kwargs: _FakeDataset(rows))
    monkeypatch.setattr("module.caption_pipeline.orchestrator.create_scene_detector", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "module.caption_pipeline.orchestrator.postprocess_caption_content",
        lambda output, *_args, **_kwargs: output,
    )

    def fake_write(path, output, mime):
        sidecar_calls.append((str(path), output, mime))
        return path.with_suffix(".txt"), None

    def fake_update(dataset, processed_filepaths, results, **kwargs):
        update_calls.append((list(processed_filepaths), list(results), kwargs))

    monkeypatch.setattr("module.caption_pipeline.orchestrator.write_caption_output", fake_write)
    monkeypatch.setattr("module.caption_pipeline.orchestrator.update_dataset_captions", fake_update)
    return update_calls, sidecar_calls


def _fake_registry(monkeypatch, provider_class):
    class FakeRegistry:
        def find_provider(self, _args, _mime):
            return provider_class

    monkeypatch.setattr("module.providers.get_registry", lambda: FakeRegistry())


class _FakeCloudProvider:
    name = "fake_cloud"
    capabilities = type("Capabilities", (), {"supports_cloud_concurrency": True})()


class _FakeLocalProvider:
    name = "fake_local"
    capabilities = type("Capabilities", (), {"supports_cloud_concurrency": False})()


def test_update_dataset_captions_normalizes_list_and_dict():
    from module.caption_pipeline.dataset_sync import update_dataset_captions

    executed_tables = []

    class MergeBuilder:
        def when_matched_update_all(self):
            return self

        def execute(self, table):
            executed_tables.append(table)

    class FakeTags:
        def create(self, name, value):
            self.created = (name, value)

        def update(self, name, value):
            self.updated = (name, value)

    class FakeDataset:
        def __init__(self):
            self.tags = FakeTags()

        def merge_insert(self, on):
            assert on == "uris"
            return MergeBuilder()

    dataset = FakeDataset()
    result = update_dataset_captions(
        dataset,
        ["a.jpg", "b.jpg"],
        [["short", "long"], {"description": "desc"}],
        merge_batch_size=10,
        console=_quiet_console(),
    )

    assert result is dataset
    rows = executed_tables[0].to_pylist()
    assert rows[0]["captions"] == ["short\nlong"]
    assert json.loads(rows[1]["captions"][0])["description"] == "desc"


def test_update_dataset_captions_rebuilds_from_sidecars_when_merge_fails(monkeypatch, tmp_path):
    from module.caption_pipeline.dataset_sync import update_dataset_captions

    merge_error = OSError("Blob struct missing `data` field")
    rebuilt_dataset = object()
    rebuild_calls = []

    class MergeBuilder:
        def when_matched_update_all(self):
            return self

        def execute(self, _table):
            raise merge_error

    class FakeDataset:
        def merge_insert(self, on):
            assert on == "uris"
            return MergeBuilder()

    def fake_rebuild(source_dir, **kwargs):
        rebuild_calls.append((source_dir, kwargs))
        return rebuilt_dataset

    monkeypatch.setattr("module.caption_pipeline.dataset_sync.rebuild_lance_from_sidecars", fake_rebuild)

    result = update_dataset_captions(
        FakeDataset(),
        ["a.jpg"],
        ["caption"],
        merge_batch_size=10,
        console=_quiet_console(),
        dataset_dir=str(tmp_path),
        transform2lance_fn=lambda **_kwargs: None,
        load_data_fn=lambda *_args, **_kwargs: [],
    )

    assert result is rebuilt_dataset
    source_dir, kwargs = rebuild_calls[0]
    assert source_dir == tmp_path
    assert kwargs["output_name"] == "dataset"
    assert kwargs["tag"] == "gemini"
    assert kwargs["caption_extension"] is None


def test_update_dataset_captions_raises_from_merge_error_when_rebuild_unavailable():
    from module.caption_pipeline.dataset_sync import update_dataset_captions

    merge_error = OSError("Blob struct missing `data` field")

    class MergeBuilder:
        def when_matched_update_all(self):
            return self

        def execute(self, _table):
            raise merge_error

    class FakeDataset:
        def merge_insert(self, on):
            assert on == "uris"
            return MergeBuilder()

    with pytest.raises(RuntimeError) as exc_info:
        update_dataset_captions(
            FakeDataset(),
            ["a.jpg"],
            ["caption"],
            merge_batch_size=10,
            console=_quiet_console(),
            dataset_dir=object(),
            transform2lance_fn=lambda **_kwargs: None,
            load_data_fn=lambda *_args, **_kwargs: [],
        )

    assert "fallback rebuild is unavailable" in str(exc_info.value)
    assert exc_info.value.__cause__ is merge_error


def test_update_dataset_captions_rebuild_failure_preserves_merge_context(monkeypatch, tmp_path):
    from module.caption_pipeline.dataset_sync import update_dataset_captions

    merge_error = OSError("Blob struct missing `data` field")

    class MergeBuilder:
        def when_matched_update_all(self):
            return self

        def execute(self, _table):
            raise merge_error

    class FakeDataset:
        def merge_insert(self, on):
            assert on == "uris"
            return MergeBuilder()

    def fake_rebuild(*_args, **_kwargs):
        raise ValueError("rebuild boom")

    monkeypatch.setattr("module.caption_pipeline.dataset_sync.rebuild_lance_from_sidecars", fake_rebuild)

    with pytest.raises(RuntimeError) as exc_info:
        update_dataset_captions(
            FakeDataset(),
            ["a.jpg"],
            ["caption"],
            merge_batch_size=10,
            console=_quiet_console(),
            dataset_dir=str(tmp_path),
            transform2lance_fn=lambda **_kwargs: None,
            load_data_fn=lambda *_args, **_kwargs: [],
        )

    assert "fallback rebuild also failed" in str(exc_info.value)
    assert "rebuild boom" in str(exc_info.value)
    assert exc_info.value.__cause__ is merge_error


def test_deferred_provider_timing_prints_after_visual_and_save(monkeypatch, tmp_path):
    from module.caption_pipeline.orchestrator import process_batch
    from module.providers.base import CaptionResult

    rows = _make_rows(tmp_path, ["a.png"])
    _patch_process_batch_io(monkeypatch, rows)
    _fake_registry(monkeypatch, _FakeLocalProvider)

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None)

    def fake_api_process_batch_fn(**kwargs):
        request_console = kwargs["progress"].console
        request_console.print("Using provider: codex_subscription")
        request_console.print("visual result")
        return CaptionResult(
            raw="caption",
            metadata={
                "provider": "codex_subscription",
                "duration_seconds": 1.234,
                "duration_log_label": "Codex caption completed: a.png",
                "duration_log_style": "green",
            },
        )

    process_batch(
        _process_batch_args(tmp_path),
        {},
        api_process_batch_fn=fake_api_process_batch_fn,
        transform2lance_fn=lambda **_kwargs: None,
        extract_from_lance_fn=lambda *_args, **_kwargs: None,
        console_obj=console,
    )

    output = buf.getvalue()
    using_index = output.index("Using provider: codex_subscription")
    visual_index = output.index("visual result")
    saved_index = output.index("Saved captions to")
    timing_index = output.index("Codex caption completed: a.png in 1.2s")
    assert using_index < visual_index < saved_index < timing_index


def test_process_batch_uses_rebuilt_dataset_for_extract(monkeypatch, tmp_path):
    from module.caption_pipeline.orchestrator import process_batch

    rows = _make_rows(tmp_path, ["a.png"])
    original_dataset = _FakeDataset(rows)
    rebuilt_dataset = object()
    extract_calls = []
    _fake_registry(monkeypatch, _FakeLocalProvider)

    monkeypatch.setattr("module.caption_pipeline.orchestrator._resolve_dataset", lambda *_args, **_kwargs: original_dataset)
    monkeypatch.setattr("module.caption_pipeline.orchestrator.create_scene_detector", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "module.caption_pipeline.orchestrator.postprocess_caption_content",
        lambda output, *_args, **_kwargs: output,
    )
    monkeypatch.setattr(
        "module.caption_pipeline.orchestrator.write_caption_output",
        lambda path, output, mime: (path.with_suffix(".txt"), None),
    )
    monkeypatch.setattr(
        "module.caption_pipeline.orchestrator.update_dataset_captions",
        lambda *_args, **_kwargs: rebuilt_dataset,
    )

    process_batch(
        _process_batch_args(tmp_path),
        {},
        api_process_batch_fn=lambda **_kwargs: "caption",
        transform2lance_fn=lambda **_kwargs: None,
        extract_from_lance_fn=lambda *call_args, **call_kwargs: extract_calls.append((call_args, call_kwargs)),
        console_obj=_quiet_console(),
    )

    assert extract_calls[0][0][0] is rebuilt_dataset


def test_concurrent_log_replay_does_not_rehighlight_plain_text(monkeypatch):
    from types import SimpleNamespace

    from module.caption_pipeline import orchestrator

    jobs = [
        orchestrator.CaptionJob(index=0, filepath="a.avif", mime="image/avif", duration=1000, sha256hash="hash-a"),
        orchestrator.CaptionJob(index=1, filepath="b.avif", mime="image/avif", duration=1000, sha256hash="hash-b"),
    ]
    log_text = "Saved captions to D:\\CPL2\\atomic heart\\left (atomic heart), right (atomic heart)\\4227770.txt\n"

    def fake_process_single_caption_job_buffered(job, *_args, **_kwargs):
        return orchestrator.CaptionJobResult(index=job.index, filepath=job.filepath, mime=job.mime, output="caption", log_text=log_text)

    class DummyProgress:
        def update(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(orchestrator, "_process_single_caption_job_buffered", fake_process_single_caption_job_buffered)
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=True, color_system="truecolor")

    orchestrator._run_caption_jobs_concurrently(
        jobs,
        SimpleNamespace(),
        {},
        api_process_batch_fn=lambda **_kwargs: "caption",
        console_obj=console,
        progress=DummyProgress(),
        task_id="task",
        max_workers=2,
        provider_class=type("Provider", (), {"name": "fake_cloud"})(),
    )

    output = buffer.getvalue()
    assert "\x1b[" not in output
    assert "(atomic heart)" in output
    assert "4227770.txt" in output


def test_align_subtitles_with_scenes_falls_back_on_timeout():
    from module.caption_pipeline.scene_alignment import align_subtitles_with_scenes

    subs = pysrt.from_string("1\n00:00:00,000 --> 00:00:01,000\nhello\n")

    class TimeoutSceneDetector:
        def wait_for_detection(self, filepath, timeout=None):
            raise TimeoutError("scene detection timed out")

        def align_subtitle(self, subs, scene_list, console=None, segment_time=None):
            raise AssertionError("should not align when wait fails")

    aligned = align_subtitles_with_scenes(
        subs,
        TimeoutSceneDetector(),
        filepath="video.mp4",
        segment_time=5,
        console=_quiet_console(),
        timeout=0.1,
    )

    assert aligned is subs


def test_process_segmented_media_only_merges_sidecar_once(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import patch

    from module.caption_pipeline.orchestrator import _process_segmented_media

    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    (tmp_path / "clip.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nsidecar\n",
        encoding="utf-8",
    )

    clip_dir = tmp_path / "clip_clip"
    clip_dir.mkdir()
    for index in range(2):
        (clip_dir / f"clip_{index}.mp4").write_bytes(b"chunk")

    class DummyProgress:
        def add_task(self, *_args, **_kwargs):
            return "task"

        def update(self, *_args, **_kwargs):
            return None

    def fake_api_process_batch_fn(**kwargs):
        uri = Path(kwargs["uri"])
        if uri.name == "clip_0.mp4":
            return "1\n00:00:00,000 --> 00:00:01,000\nchunk0\n"
        return "1\n00:00:00,000 --> 00:00:01,000\nchunk1\n"

    args = SimpleNamespace(segment_time=1)

    with (
        patch("module.caption_pipeline.orchestrator.split_video_with_imageio_ffmpeg", lambda *a, **k: None),
        patch("module.caption_pipeline.orchestrator.get_video_duration", lambda *_a, **_k: 1000),
    ):
        merged = _process_segmented_media(
            str(video_path),
            "video/mp4",
            2000,
            "hash",
            args,
            {},
            DummyProgress(),
            "task",
            fake_api_process_batch_fn,
            _quiet_console(),
        )

    from module.providers.base import CaptionResult

    assert isinstance(merged, CaptionResult)
    assert merged.raw.count("sidecar") == 1
    assert "chunk0" in merged.raw
    assert "chunk1" in merged.raw


def test_process_segmented_media_passes_original_path_as_directory_name_source(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import patch

    from module.caption_pipeline.orchestrator import _process_segmented_media

    source_dir = tmp_path / "Alice (Wonderland)"
    video_path = source_dir / "movie.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"video")

    clip_dir = source_dir / "movie_clip"
    clip_dir.mkdir()
    for index in range(2):
        (clip_dir / f"movie_{index}.mp4").write_bytes(b"chunk")

    class DummyProgress:
        def add_task(self, *_args, **_kwargs):
            return "task"

        def update(self, *_args, **_kwargs):
            return None

    seen_sources = []
    seen_uris = []

    def fake_api_process_batch_fn(**kwargs):
        seen_uris.append(Path(kwargs["uri"]))
        seen_sources.append(getattr(kwargs["args"], "directory_name_source_uri", ""))
        return "1\n00:00:00,000 --> 00:00:01,000\nchunk\n"

    args = SimpleNamespace(segment_time=1, dir_name=True)

    with (
        patch("module.caption_pipeline.orchestrator.split_video_with_imageio_ffmpeg", lambda *a, **k: None),
        patch("module.caption_pipeline.orchestrator.get_video_duration", lambda *_a, **_k: 1000),
    ):
        _process_segmented_media(
            str(video_path),
            "video/mp4",
            2000,
            "hash",
            args,
            {},
            DummyProgress(),
            "task",
            fake_api_process_batch_fn,
            _quiet_console(),
        )

    assert seen_uris
    assert all(uri.parent.name == "movie_clip" for uri in seen_uris)
    assert seen_sources == [str(video_path), str(video_path)]
    assert not hasattr(args, "directory_name_source_uri")


def test_process_segmented_media_merges_structured_summaries_into_txt_payload(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import patch

    from module.caption_pipeline.orchestrator import _process_segmented_media

    audio_path = tmp_path / "track.wav"
    audio_path.write_bytes(b"audio")
    (tmp_path / "track.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nsidecar\n",
        encoding="utf-8",
    )

    clip_dir = tmp_path / "track_clip"
    clip_dir.mkdir()
    for index in range(2):
        (clip_dir / f"track_{index}.wav").write_bytes(b"chunk")

    class DummyProgress:
        def add_task(self, *_args, **_kwargs):
            return "task"

        def update(self, *_args, **_kwargs):
            return None

    def fake_api_process_batch_fn(**kwargs):
        uri = Path(kwargs["uri"])
        if uri.name == "track_0.wav":
            return {"description": "Slow piano intro", "caption_extension": ".txt", "provider": "music_flamingo_local"}
        return {"description": "Full drums and vocal hook", "caption_extension": ".txt", "provider": "music_flamingo_local"}

    args = SimpleNamespace(segment_time=30)

    with patch("module.caption_pipeline.orchestrator.split_video_with_imageio_ffmpeg", lambda *a, **k: None):
        merged = _process_segmented_media(
            str(audio_path),
            "audio/wav",
            60000,
            "hash",
            args,
            {},
            DummyProgress(),
            "task",
            fake_api_process_batch_fn,
            _quiet_console(),
        )

    from module.providers.base import CaptionResult

    assert isinstance(merged, CaptionResult)
    assert merged.caption_extension == ".txt"
    assert merged.parsed["provider"] == "music_flamingo_local"
    assert len(merged.parsed["segments"]) == 2
    assert "Segment 1 [00:00:00 - 00:00:30]" in merged.text
    assert "Slow piano intro" in merged.text
    assert "Full drums and vocal hook" in merged.text
    assert "sidecar" not in merged.text


def test_process_segmented_media_merges_transcripts_without_segment_headers(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import patch

    from module.caption_pipeline.orchestrator import _process_segmented_media

    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"audio")

    clip_dir = tmp_path / "speech_clip"
    clip_dir.mkdir()
    for index in range(2):
        (clip_dir / f"speech_{index}.wav").write_bytes(b"chunk")

    class DummyProgress:
        def add_task(self, *_args, **_kwargs):
            return "task"

        def update(self, *_args, **_kwargs):
            return None

    def fake_api_process_batch_fn(**kwargs):
        uri = Path(kwargs["uri"])
        if uri.name == "speech_0.wav":
            return {"task_kind": "transcribe", "transcript": "hello", "caption_extension": ".txt", "provider": "cohere_transcribe_local"}
        return {"task_kind": "transcribe", "transcript": "world", "caption_extension": ".txt", "provider": "cohere_transcribe_local"}

    args = SimpleNamespace(segment_time=30)

    with patch("module.caption_pipeline.orchestrator.split_video_with_imageio_ffmpeg", lambda *a, **k: None):
        merged = _process_segmented_media(
            str(audio_path),
            "audio/wav",
            60000,
            "hash",
            args,
            {},
            DummyProgress(),
            "task",
            fake_api_process_batch_fn,
            _quiet_console(),
        )

    from module.providers.base import CaptionResult

    assert isinstance(merged, CaptionResult)
    assert merged.parsed["task_kind"] == "transcribe"
    assert merged.text == "hello\n\nworld"
    assert merged.parsed["provider"] == "cohere_transcribe_local"
    assert "Segment 1" not in merged.text


def test_process_segmented_media_merges_ast_chunks_into_srt_payload(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import patch

    from module.caption_pipeline.orchestrator import _process_segmented_media

    audio_path = tmp_path / "ast.wav"
    audio_path.write_bytes(b"audio")

    clip_dir = tmp_path / "ast_clip"
    clip_dir.mkdir()
    for index in range(2):
        (clip_dir / f"ast_{index}.wav").write_bytes(b"chunk")

    class DummyProgress:
        def add_task(self, *_args, **_kwargs):
            return "task"

        def update(self, *_args, **_kwargs):
            return None

    def fake_api_process_batch_fn(**kwargs):
        uri = Path(kwargs["uri"])
        if uri.name == "ast_0.wav":
            return {
                "task_kind": "ast",
                "translation_srt": "1\n00:00:00,000 --> 00:00:10,000\n你好\n",
                "caption_extension": ".srt",
                "subtitle_format": "srt",
                "provider": "gemma4_local",
            }
        return {
            "task_kind": "ast",
            "translation_srt": "1\n00:00:10,000 --> 00:00:20,000\n世界\n",
            "caption_extension": ".srt",
            "subtitle_format": "srt",
            "provider": "gemma4_local",
        }

    args = SimpleNamespace(segment_time=10)

    with patch("module.caption_pipeline.orchestrator.split_video_with_imageio_ffmpeg", lambda *a, **k: None):
        merged = _process_segmented_media(
            str(audio_path),
            "audio/wav",
            20000,
            "hash",
            args,
            {},
            DummyProgress(),
            "task",
            fake_api_process_batch_fn,
            _quiet_console(),
        )

    from module.providers.base import CaptionResult

    assert isinstance(merged, CaptionResult)
    assert merged.parsed["task_kind"] == "ast"
    assert merged.caption_extension == ".srt"
    assert merged.parsed["subtitle_format"] == "srt"
    assert merged.parsed["provider"] == "gemma4_local"
    assert merged.text == (
        "1\n00:00:00,000 --> 00:00:10,000\n你好\n\n"
        "1\n00:00:10,000 --> 00:00:20,000\n世界"
    )


def test_process_batch_skips_segmentation_when_segment_time_is_none(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import patch

    from module.caption_pipeline.orchestrator import process_batch

    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"audio")

    class FakeScanner:
        def to_batches(self):
            return [
                {
                    "uris": pa.array([str(audio_path)]),
                    "mime": pa.array(["audio/wav"]),
                    "duration": pa.array([3_600_000]),
                    "hash": pa.array(["hash"]),
                }
            ]

    class FakeDataset:
        def scanner(self, **_kwargs):
            return FakeScanner()

        def count_rows(self):
            return 1

    api_calls = []

    def fake_api_process_batch_fn(**kwargs):
        api_calls.append(kwargs["uri"])
        return {
            "task_kind": "transcribe",
            "transcript": "single pass transcript",
            "caption_extension": ".txt",
            "provider": "cohere_transcribe_local",
        }

    extract_calls = []
    args = SimpleNamespace(
        dataset_dir="ignored",
        gemini_api_key="",
        mistral_api_key="",
        not_clip_with_caption=True,
        merge_batch_size=1000,
        segment_time=None,
        scene_detection_timeout=None,
    )

    with (
        patch("module.caption_pipeline.orchestrator._resolve_dataset", return_value=FakeDataset()),
        patch("module.caption_pipeline.orchestrator.create_scene_detector", return_value=None),
        patch("module.caption_pipeline.orchestrator.postprocess_caption_content", side_effect=lambda output, *_args, **_kwargs: output),
        patch("module.caption_pipeline.orchestrator._process_segmented_media", side_effect=AssertionError("should not segment when segment_time is None")),
        patch("module.caption_pipeline.orchestrator.write_caption_output", return_value=(audio_path.with_suffix(".txt"), None)),
        patch("module.caption_pipeline.orchestrator.update_dataset_captions") as update_mock,
    ):
        process_batch(
            args,
            {},
            api_process_batch_fn=fake_api_process_batch_fn,
            transform2lance_fn=lambda **_kwargs: None,
            extract_from_lance_fn=lambda *call_args, **call_kwargs: extract_calls.append((call_args, call_kwargs)),
            console_obj=_quiet_console(),
        )

    assert api_calls == [str(audio_path)]
    update_mock.assert_called_once()
    assert update_mock.call_args.args[2] == [
        {
            "task_kind": "transcribe",
            "transcript": "single pass transcript",
            "caption_extension": ".txt",
            "provider": "cohere_transcribe_local",
        }
    ]
    assert extract_calls and extract_calls[0][0][1] == "ignored"


def test_process_batch_bypasses_segmentation_for_gemma4_audio(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import patch

    from module.caption_pipeline.orchestrator import process_batch

    audio_path = tmp_path / "speech.wav"
    audio_path.write_bytes(b"audio")

    class FakeScanner:
        def to_batches(self):
            return [
                {
                    "uris": pa.array([str(audio_path)]),
                    "mime": pa.array(["audio/wav"]),
                    "duration": pa.array([120_000]),
                    "hash": pa.array(["hash"]),
                }
            ]

    class FakeDataset:
        def scanner(self, **_kwargs):
            return FakeScanner()

        def count_rows(self):
            return 1

    api_calls = []

    def fake_api_process_batch_fn(**kwargs):
        api_calls.append(kwargs["uri"])
        return {
            "task_kind": "transcribe",
            "transcript": "single pass gemma4 transcript",
            "caption_extension": ".txt",
            "provider": "gemma4_local",
        }

    args = SimpleNamespace(
        dataset_dir="ignored",
        gemini_api_key="",
        mistral_api_key="",
        not_clip_with_caption=True,
        merge_batch_size=1000,
        segment_time=10,
        scene_detection_timeout=None,
        alm_model="gemma4_local",
        vlm_image_model="",
    )

    with (
        patch("module.caption_pipeline.orchestrator._resolve_dataset", return_value=FakeDataset()),
        patch("module.caption_pipeline.orchestrator.create_scene_detector", return_value=None),
        patch("module.caption_pipeline.orchestrator.postprocess_caption_content", side_effect=lambda output, *_args, **_kwargs: output),
        patch("module.caption_pipeline.orchestrator._process_segmented_media", side_effect=AssertionError("should bypass segmentation for gemma4_local")),
        patch("module.caption_pipeline.orchestrator.write_caption_output", return_value=(audio_path.with_suffix(".txt"), None)),
        patch("module.caption_pipeline.orchestrator.update_dataset_captions") as update_mock,
    ):
        process_batch(
            args,
            {},
            api_process_batch_fn=fake_api_process_batch_fn,
            transform2lance_fn=lambda **_kwargs: None,
            extract_from_lance_fn=lambda *_args, **_kwargs: None,
            console_obj=_quiet_console(),
        )

    assert api_calls == [str(audio_path)]
    update_mock.assert_called_once()


def test_resolve_media_segment_time_applies_marlin_default_only_to_video():
    from types import SimpleNamespace

    from module.caption_pipeline.orchestrator import _resolve_media_segment_time

    args = SimpleNamespace(
        segment_time=None,
        segment_time_explicit=False,
        vlm_image_model="marlin_2b_local",
    )
    config = {"marlin_2b_local": {"video_max_seconds": 120}}

    assert _resolve_media_segment_time(args, "audio/wav", config) is None
    assert _resolve_media_segment_time(args, "video/mp4", config) == 119


def test_resolve_media_segment_time_caps_explicit_marlin_value_to_safe_chunk_size():
    from types import SimpleNamespace

    from module.caption_pipeline.orchestrator import _resolve_media_segment_time

    args = SimpleNamespace(
        segment_time=600,
        segment_time_explicit=True,
        vlm_image_model="marlin_2b_local",
    )

    assert _resolve_media_segment_time(args, "video/mp4", {"marlin": {"video_max_seconds": 90}}) == 89


def test_process_batch_segments_marlin_video_over_hard_limit_with_media_scoped_segment_time(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import patch

    from module.caption_pipeline.orchestrator import process_batch

    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")

    class FakeScanner:
        def to_batches(self):
            return [
                {
                    "uris": pa.array([str(video_path)]),
                    "mime": pa.array(["video/mp4"]),
                    "duration": pa.array([120_500]),
                    "hash": pa.array(["hash"]),
                }
            ]

    class FakeDataset:
        def scanner(self, **_kwargs):
            return FakeScanner()

        def count_rows(self):
            return 1

    segmented_calls = []

    def fake_process_segmented_media(filepath, mime, duration, sha256hash, args, config, progress, task_id, api_process_batch_fn, console):
        segmented_calls.append(
            {
                "filepath": filepath,
                "mime": mime,
                "duration": duration,
                "segment_time": args.segment_time,
            }
        )
        return {"description": "segmented marlin summary", "caption_extension": ".txt", "provider": "marlin_2b_local"}

    args = SimpleNamespace(
        dataset_dir="ignored",
        gemini_api_key="",
        mistral_api_key="",
        not_clip_with_caption=True,
        merge_batch_size=1000,
        segment_time=600,
        segment_time_explicit=False,
        scene_detection_timeout=None,
        vlm_image_model="marlin_2b_local",
        alm_model="",
    )

    with (
        patch("module.caption_pipeline.orchestrator._resolve_dataset", return_value=FakeDataset()),
        patch("module.caption_pipeline.orchestrator.create_scene_detector", return_value=None),
        patch("module.caption_pipeline.orchestrator._process_segmented_media", side_effect=fake_process_segmented_media),
        patch("module.caption_pipeline.orchestrator.write_caption_output", return_value=(video_path.with_suffix(".txt"), None)),
        patch("module.caption_pipeline.orchestrator.update_dataset_captions") as update_mock,
    ):
        process_batch(
            args,
            {"marlin_2b_local": {"video_max_seconds": 120}},
            api_process_batch_fn=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should segment before direct Marlin call")),
            transform2lance_fn=lambda **_kwargs: None,
            extract_from_lance_fn=lambda *_args, **_kwargs: None,
            console_obj=_quiet_console(),
        )

    assert segmented_calls == [
        {
            "filepath": str(video_path),
            "mime": "video/mp4",
            "duration": 120_500,
            "segment_time": 119,
        }
    ]
    update_mock.assert_called_once()


def test_process_batch_cloud_image_jobs_run_concurrently(monkeypatch, tmp_path):
    import threading
    import time

    from module.caption_pipeline.orchestrator import process_batch

    rows = _make_rows(tmp_path, ["a.png", "b.png", "c.png"])
    _patch_process_batch_io(monkeypatch, rows)
    _fake_registry(monkeypatch, _FakeCloudProvider)

    lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0

    def fake_api_process_batch_fn(**_kwargs):
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.05)
        with lock:
            in_flight -= 1
        return "caption"

    process_batch(
        _process_batch_args(tmp_path, cloud_max_concurrency=3),
        {},
        api_process_batch_fn=fake_api_process_batch_fn,
        transform2lance_fn=lambda **_kwargs: None,
        extract_from_lance_fn=lambda *_args, **_kwargs: None,
        console_obj=_quiet_console(),
    )

    assert max_in_flight > 1


def test_process_batch_preserves_order_under_concurrency(monkeypatch, tmp_path):
    import time

    from module.caption_pipeline.orchestrator import process_batch

    rows = _make_rows(tmp_path, ["slow.png", "medium.png", "fast.png"])
    update_calls, _sidecar_calls = _patch_process_batch_io(monkeypatch, rows)
    _fake_registry(monkeypatch, _FakeCloudProvider)
    delays = {"slow.png": 0.06, "medium.png": 0.03, "fast.png": 0.0}

    def fake_api_process_batch_fn(**kwargs):
        name = Path(kwargs["uri"]).name
        time.sleep(delays[name])
        return f"caption:{name}"

    process_batch(
        _process_batch_args(tmp_path, cloud_max_concurrency=3),
        {},
        api_process_batch_fn=fake_api_process_batch_fn,
        transform2lance_fn=lambda **_kwargs: None,
        extract_from_lance_fn=lambda *_args, **_kwargs: None,
        console_obj=_quiet_console(),
    )

    assert len(update_calls) == 1
    assert update_calls[0][0] == [row["uri"] for row in rows]
    assert update_calls[0][1] == ["caption:slow.png", "caption:medium.png", "caption:fast.png"]


def test_process_batch_local_provider_remains_serial(monkeypatch, tmp_path):
    import threading
    import time

    from module.caption_pipeline.orchestrator import process_batch

    rows = _make_rows(tmp_path, ["a.png", "b.png", "c.png"])
    _patch_process_batch_io(monkeypatch, rows)
    _fake_registry(monkeypatch, _FakeLocalProvider)

    lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0

    def fake_api_process_batch_fn(**_kwargs):
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.02)
        with lock:
            in_flight -= 1
        return "caption"

    process_batch(
        _process_batch_args(tmp_path, cloud_max_concurrency=4),
        {},
        api_process_batch_fn=fake_api_process_batch_fn,
        transform2lance_fn=lambda **_kwargs: None,
        extract_from_lance_fn=lambda *_args, **_kwargs: None,
        console_obj=_quiet_console(),
    )

    assert max_in_flight == 1


def test_process_batch_video_cloud_provider_remains_serial_in_phase_1(monkeypatch, tmp_path):
    import threading
    import time

    from module.caption_pipeline.orchestrator import process_batch

    rows = _make_rows(tmp_path, ["a.mp4", "b.mp4", "c.mp4"], mime="video/mp4")
    _patch_process_batch_io(monkeypatch, rows)
    _fake_registry(monkeypatch, _FakeCloudProvider)

    lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0

    def fake_api_process_batch_fn(**_kwargs):
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.02)
        with lock:
            in_flight -= 1
        return {"description": "video caption", "caption_extension": ".txt"}

    process_batch(
        _process_batch_args(tmp_path, cloud_max_concurrency=4),
        {},
        api_process_batch_fn=fake_api_process_batch_fn,
        transform2lance_fn=lambda **_kwargs: None,
        extract_from_lance_fn=lambda *_args, **_kwargs: None,
        console_obj=_quiet_console(),
    )

    assert max_in_flight == 1


def test_process_batch_application_document_remains_serial_in_phase_1(monkeypatch, tmp_path):
    import threading
    import time

    from module.caption_pipeline.orchestrator import process_batch

    rows = _make_rows(tmp_path, ["a.pdf", "b.pdf", "c.pdf"], mime="application/pdf")
    _patch_process_batch_io(monkeypatch, rows)
    _fake_registry(monkeypatch, _FakeCloudProvider)

    lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0

    def fake_api_process_batch_fn(**_kwargs):
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.02)
        with lock:
            in_flight -= 1
        return "document caption"

    process_batch(
        _process_batch_args(tmp_path, cloud_max_concurrency=4),
        {},
        api_process_batch_fn=fake_api_process_batch_fn,
        transform2lance_fn=lambda **_kwargs: None,
        extract_from_lance_fn=lambda *_args, **_kwargs: None,
        console_obj=_quiet_console(),
    )

    assert max_in_flight == 1


def test_concurrent_failure_skips_dataset_update_but_keeps_successful_sidecars(monkeypatch, tmp_path):
    import time

    import pytest

    from module.caption_pipeline.orchestrator import process_batch

    rows = _make_rows(tmp_path, ["ok.png", "fail.png"])
    update_calls, sidecar_calls = _patch_process_batch_io(monkeypatch, rows)
    _fake_registry(monkeypatch, _FakeCloudProvider)

    def fake_api_process_batch_fn(**kwargs):
        name = Path(kwargs["uri"]).name
        if name == "fail.png":
            time.sleep(0.03)
            raise RuntimeError("provider boom")
        time.sleep(0.01)
        return "ok caption"

    with pytest.raises(RuntimeError) as exc_info:
        process_batch(
            _process_batch_args(tmp_path, cloud_max_concurrency=2),
            {},
            api_process_batch_fn=fake_api_process_batch_fn,
            transform2lance_fn=lambda **_kwargs: None,
            extract_from_lance_fn=lambda *_args, **_kwargs: None,
            console_obj=_quiet_console(),
        )

    assert "fail.png" in str(exc_info.value)
    assert "fake_cloud" in str(exc_info.value)
    assert update_calls == []
    assert [Path(call[0]).name for call in sidecar_calls] == ["ok.png"]


def test_retry_sleep_blocks_only_one_worker(monkeypatch, tmp_path):
    import time

    from module.caption_pipeline.orchestrator import process_batch

    rows = _make_rows(tmp_path, ["slow.png", "fast.png"])
    _patch_process_batch_io(monkeypatch, rows)
    _fake_registry(monkeypatch, _FakeCloudProvider)
    completion_order = []

    def fake_api_process_batch_fn(**kwargs):
        name = Path(kwargs["uri"]).name
        if name == "slow.png":
            time.sleep(0.08)
        else:
            time.sleep(0.01)
        completion_order.append(name)
        return f"caption:{name}"

    process_batch(
        _process_batch_args(tmp_path, cloud_max_concurrency=2),
        {},
        api_process_batch_fn=fake_api_process_batch_fn,
        transform2lance_fn=lambda **_kwargs: None,
        extract_from_lance_fn=lambda *_args, **_kwargs: None,
        console_obj=_quiet_console(),
    )

    assert completion_order[0] == "fast.png"


def test_codex_max_concurrency_limits_in_flight_calls(monkeypatch, tmp_path):
    import threading
    import time

    from module.caption_pipeline.orchestrator import process_batch

    class FakeCodexProvider:
        name = "codex_subscription"
        capabilities = type("Capabilities", (), {"supports_cloud_concurrency": True})()

    rows = _make_rows(tmp_path, ["a.png", "b.png", "c.png", "d.png"])
    _patch_process_batch_io(monkeypatch, rows)
    _fake_registry(monkeypatch, FakeCodexProvider)

    lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0

    def fake_api_process_batch_fn(**_kwargs):
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.04)
        with lock:
            in_flight -= 1
        return "caption"

    process_batch(
        _process_batch_args(tmp_path, codex_max_concurrency=2),
        {},
        api_process_batch_fn=fake_api_process_batch_fn,
        transform2lance_fn=lambda **_kwargs: None,
        extract_from_lance_fn=lambda *_args, **_kwargs: None,
        console_obj=_quiet_console(),
    )

    assert max_in_flight == 2


def test_codex_max_concurrency_does_not_require_cloud_max_concurrency(monkeypatch, tmp_path):
    import io

    from rich.console import Console

    from module.caption_pipeline.orchestrator import process_batch

    class FakeCodexProvider:
        name = "codex_subscription"
        capabilities = type("Capabilities", (), {"supports_cloud_concurrency": True})()

    rows = _make_rows(tmp_path, ["a.png", "b.png"])
    _patch_process_batch_io(monkeypatch, rows)
    _fake_registry(monkeypatch, FakeCodexProvider)
    buffer = io.StringIO()

    process_batch(
        _process_batch_args(tmp_path, codex_max_concurrency=2),
        {},
        api_process_batch_fn=lambda **_kwargs: "caption",
        transform2lance_fn=lambda **_kwargs: None,
        extract_from_lance_fn=lambda *_args, **_kwargs: None,
        console_obj=Console(file=buffer, force_terminal=False),
    )

    assert "requires --cloud_max_concurrency" not in buffer.getvalue()


def test_codex_default_remains_serial(monkeypatch, tmp_path):
    import threading
    import time

    from module.caption_pipeline.orchestrator import process_batch

    class FakeCodexProvider:
        name = "codex_subscription"
        capabilities = type("Capabilities", (), {"supports_cloud_concurrency": True})()

    rows = _make_rows(tmp_path, ["a.png", "b.png", "c.png"])
    _patch_process_batch_io(monkeypatch, rows)
    _fake_registry(monkeypatch, FakeCodexProvider)

    lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0

    def fake_api_process_batch_fn(**_kwargs):
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.02)
        with lock:
            in_flight -= 1
        return "caption"

    process_batch(
        _process_batch_args(tmp_path, cloud_max_concurrency=4),
        {},
        api_process_batch_fn=fake_api_process_batch_fn,
        transform2lance_fn=lambda **_kwargs: None,
        extract_from_lance_fn=lambda *_args, **_kwargs: None,
        console_obj=_quiet_console(),
    )

    assert max_in_flight == 1


def test_normalize_subtitle_timestamps_preserves_existing_hours():
    from module.caption_pipeline.postprocess import _normalize_subtitle_timestamps

    content = (
        "1\n"
        "01:02:03,456 --> 123:03:04,567\n"
        "hello\n"
        "2\n"
        "02:03,111 --> 04:05,222\n"
        "world\n"
    )

    normalized = _normalize_subtitle_timestamps(content)

    assert "01:02:03,456 --> 123:03:04,567" in normalized
    assert "00:02:03,111 --> 00:04:05,222" in normalized
