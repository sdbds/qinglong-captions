import io
import json
import sys
from pathlib import Path

import pyarrow as pa
import pysrt
from rich.console import Console


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


def _quiet_console():
    return Console(file=io.StringIO(), force_terminal=False, color_system=None)


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

    update_dataset_captions(
        FakeDataset(),
        ["a.jpg", "b.jpg"],
        [["short", "long"], {"description": "desc"}],
        merge_batch_size=10,
        console=_quiet_console(),
    )

    rows = executed_tables[0].to_pylist()
    assert rows[0]["captions"] == ["short\nlong"]
    assert json.loads(rows[1]["captions"][0])["description"] == "desc"


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

    assert merged.count("sidecar") == 1
    assert "chunk0" in merged
    assert "chunk1" in merged


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

    assert merged["caption_extension"] == ".txt"
    assert merged["provider"] == "music_flamingo_local"
    assert len(merged["segments"]) == 2
    assert "Segment 1 [00:00:00 - 00:00:30]" in merged["description"]
    assert "Slow piano intro" in merged["description"]
    assert "Full drums and vocal hook" in merged["description"]
    assert "sidecar" not in merged["description"]


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
