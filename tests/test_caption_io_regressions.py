from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pysrt
import pytest
from rich.console import Console

from module.caption_pipeline import orchestrator
from module.lanceexport import save_caption
from module.lanceImport import _find_sidecar_caption
from module.providers.base import CaptionResult, CaptionStatus
from utils.lance_rebuild import read_detected_sidecar_caption
from utils.output_writer import write_caption_output
from utils.stream_util import split_video_with_imageio_ffmpeg


@pytest.mark.parametrize(
    "payload, extension, expected",
    [
        ({"task_kind": "transcribe", "transcript": "spoken words", "caption_extension": ".txt"}, ".txt", "spoken words"),
        (
            {"task_kind": "ast", "translation_srt": "1\n00:00:00,000 --> 00:00:01,000\nhello\n", "caption_extension": ".srt"},
            ".srt",
            "1\n00:00:00,000 --> 00:00:01,000\nhello\n",
        ),
        ({"markdown": "# Document\n", "caption_extension": ".md"}, ".md", "# Document\n"),
        ({"text": "plain text", "caption_extension": ".txt"}, ".txt", "plain text"),
    ],
)
def test_export_preserves_generated_structured_sidecar(tmp_path, payload, extension, expected):
    source = tmp_path / "speech.wav"
    result = CaptionResult(raw=json.dumps(payload), parsed=payload)
    path, _ = write_caption_output(source, result, "audio/wav")
    assert path.read_text(encoding="utf-8") == expected
    assert save_caption(str(source), [result.to_dataset_caption()], "audio")
    assert source.with_suffix(extension).read_text(encoding="utf-8") == expected
    assert json.loads(source.with_suffix(".json").read_text(encoding="utf-8")) == payload


def test_export_preserves_text_around_structured_caption_line(tmp_path):
    source = tmp_path / "mixed.png"
    assert save_caption(str(source), ["before", '{"long_description":"middle"}', "after"], "image")
    content = source.with_suffix(".txt").read_text(encoding="utf-8")
    assert content.startswith("before\n")
    assert "middle" in content
    assert content.endswith("after\n")


def test_export_preserves_multiple_structured_caption_lines(tmp_path):
    source = tmp_path / "multiple.png"
    assert save_caption(str(source), ['{"description":"first"}', '{"description":"second"}'], "image")
    assert source.with_suffix(".txt").read_text(encoding="utf-8") == "firstsecond"


@pytest.mark.parametrize("filename, media_type, extension", [("invoice.pdf", "application", ".md"), ("voice.wav", "audio", ".srt")])
def test_export_preserves_plain_json_without_caption_fields(tmp_path, filename, media_type, extension):
    source = tmp_path / filename
    content = '{"invoice_number":"A-123","amount":50}'
    assert save_caption(str(source), [content], media_type)
    assert source.with_suffix(extension).read_text(encoding="utf-8") == content


@pytest.mark.parametrize("separate_root", [False, True])
def test_import_dotted_stems_selects_exact_sidecar(tmp_path, separate_root):
    source = tmp_path / "scene.001.png"
    root = tmp_path / "captions" if separate_root else tmp_path
    root.mkdir(exist_ok=True)
    (root / "scene.001.txt").write_text("correct", encoding="utf-8")
    (root / "scene.txt").write_text("wrong", encoding="utf-8")
    kwargs = {"caption_root": root, "dataset_root": tmp_path} if separate_root else {}
    assert _find_sidecar_caption(source, **kwargs) == ["correct"]


def test_rebuild_dotted_stems_selects_exact_sidecar(tmp_path):
    source = tmp_path / "scene.001.png"
    source.with_suffix(".txt").write_text("correct", encoding="utf-8")
    (tmp_path / "scene.txt").write_text("wrong", encoding="utf-8")
    assert read_detected_sidecar_caption(str(source)) == ["correct"]


class _Progress:
    def add_task(self, *args, **kwargs):
        return 1

    def update(self, *args, **kwargs):
        pass


@pytest.mark.parametrize("outcome", ["failed", "skipped", "empty"])
def test_incomplete_segments_cannot_replace_complete_caption(tmp_path, monkeypatch, outcome):
    source = tmp_path / "speech.wav"
    source.with_suffix(".txt").write_text("complete previous transcript", encoding="utf-8")
    clips = tmp_path / "speech_clip"
    clips.mkdir()
    for index in range(2):
        (clips / f"speech_{index}.wav").touch()
    def split(path, subs, *, output_dir, **kwargs):
        output_dir.mkdir(parents=True)
        paths = [output_dir / f"speech_{sub.index}.wav" for sub in subs]
        for clip in paths:
            clip.write_bytes(b"chunk")
        return paths

    monkeypatch.setattr(orchestrator, "split_video_with_imageio_ffmpeg", split)
    monkeypatch.setattr(orchestrator, "get_video_duration", lambda *a: 30000)

    def caption(**kwargs):
        if Path(kwargs["uri"]).name == "speech_0.wav":
            return CaptionResult.success(
                parsed={"task_kind": "transcribe", "transcript": "first half", "caption_extension": ".txt"}
            )
        if outcome == "empty":
            return CaptionResult.success()
        return getattr(CaptionResult, outcome)("backend did not produce a caption")

    result = orchestrator._process_segmented_media(
        str(source),
        "audio/wav",
        60000,
        "hash",
        SimpleNamespace(segment_time=30),
        {},
        _Progress(),
        1,
        caption,
        Console(file=io.StringIO()),
    )
    assert result.status is CaptionStatus.FAILED
    assert not result.is_persistable
    assert source.with_suffix(".txt").read_text(encoding="utf-8") == "complete previous transcript"


def test_fractional_tail_is_included_in_segment_plan(tmp_path, monkeypatch):
    source = tmp_path / "speech.wav"
    planned = []
    seen = []

    def split(path, subs, **kwargs):
        planned.extend((sub.start.ordinal, sub.end.ordinal) for sub in subs)
        directory = kwargs["output_dir"]
        directory.mkdir()
        paths = []
        for sub in subs:
            clip = directory / f"speech_{sub.index:03d}.wav"
            clip.write_bytes(b"chunk")
            paths.append(clip)
        return paths

    def caption(**kwargs):
        seen.append(Path(kwargs["uri"]).name)
        return {"task_kind": "transcribe", "transcript": "speech", "caption_extension": ".txt"}

    monkeypatch.setattr(orchestrator, "split_video_with_imageio_ffmpeg", split)
    result = orchestrator._process_segmented_media(
        str(source),
        "audio/wav",
        60500,
        "hash",
        SimpleNamespace(segment_time=30),
        {},
        _Progress(),
        1,
        caption,
        Console(file=io.StringIO()),
    )
    assert planned == [(0, 30000), (30000, 60000), (60000, 60500)]
    assert seen == ["speech_000.wav", "speech_001.wav", "speech_002.wav"]
    assert result.is_persistable


def test_fixed_segmentation_runs_ffmpeg_once(tmp_path, monkeypatch):
    commands = []

    def popen(command, **kwargs):
        commands.append(command)
        paths = [Path(command[-1] % index) for index in range(3)]
        for path in paths:
            path.write_bytes(b"chunk")
        import csv
        manifest = Path(command[command.index("-segment_list") + 1])
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows((path.name, index * 119, (index + 1) * 119)
                                         for index, path in enumerate(paths))
        return SimpleNamespace(returncode=0, communicate=lambda: ("", ""))

    monkeypatch.setattr("utils.stream_util.subprocess.Popen", popen)
    monkeypatch.setattr("imageio_ffmpeg.get_ffmpeg_exe", lambda: "ffmpeg")
    subs = pysrt.SubRipFile(
        [
            pysrt.SubRipItem(index=i, start=pysrt.SubRipTime(seconds=i * 119), end=pysrt.SubRipTime(seconds=(i + 1) * 119))
            for i in range(3)
        ]
    )
    split_video_with_imageio_ffmpeg(tmp_path / "video.mp4", subs, segment_time=119)
    assert len(commands) == 1
    assert "segment" in commands[0]


def test_caption_clip_export_does_not_segment_whole_video(tmp_path, monkeypatch):
    commands = []
    def popen(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"chunk")
        return SimpleNamespace(returncode=0, communicate=lambda: ("", ""))

    monkeypatch.setattr(
        "utils.stream_util.subprocess.Popen",
        popen,
    )
    monkeypatch.setattr("imageio_ffmpeg.get_ffmpeg_exe", lambda: "ffmpeg")
    subs = pysrt.SubRipFile(
        [
            pysrt.SubRipItem(index=1, start=pysrt.SubRipTime(seconds=5), end=pysrt.SubRipTime(seconds=125)),
            pysrt.SubRipItem(index=2, start=pysrt.SubRipTime(seconds=130), end=pysrt.SubRipTime(seconds=131)),
        ]
    )
    split_video_with_imageio_ffmpeg(tmp_path / "video.mp4", subs, save_caption_func=lambda *a: None)
    assert len(commands) == 2
    assert all("segment" not in command for command in commands)
