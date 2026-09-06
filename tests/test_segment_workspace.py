import io
import subprocess
import sys
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace

import av
import imageio_ffmpeg
import pysrt
import pytest
from rich.console import Console

from module.caption_pipeline import orchestrator
from module.providers.base import CaptionResult
from utils.stream_util import split_media_stream_clips, split_video_with_imageio_ffmpeg


class _Progress:
    def add_task(self, *args, **kwargs):
        return 1

    def update(self, *args, **kwargs):
        pass


def _call(source, provider, *, chunks=2):
    return orchestrator._process_segmented_media(
        str(source), "audio/wav", chunks * 1000, "hash", SimpleNamespace(segment_time=1), {},
        _Progress(), 0, provider, Console(file=io.StringIO()),
    )


def _clips(directory, count):
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"clip_{index}.wav" for index in range(1, count + 1)]
    for index, path in enumerate(paths, 1):
        path.write_text(str(index), encoding="utf-8")
    return paths


def test_manifest_order_preserves_twelve_clips_and_old_files(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    old = _clips(tmp_path / "source_clip", 12)
    for path in old:
        path.write_text("old", encoding="utf-8")
    generated = []

    def split(*args, **kwargs):
        paths = _clips(Path(kwargs.get("output_dir", tmp_path / "new")), 12)
        generated.extend(paths)
        return paths

    seen = []
    monkeypatch.setattr(orchestrator, "split_video_with_imageio_ffmpeg", split)
    result = _call(source, lambda **kw: seen.append(kw["uri"].read_text()) or
                   CaptionResult(raw="ok", parsed={"description": "ok"}), chunks=12)
    assert result.is_persistable
    assert seen == [str(index) for index in range(1, 13)]
    assert all(path.read_text() == "old" for path in old)
    assert all(not path.exists() for path in generated)


@pytest.mark.parametrize("failure", ["exception", "empty_result"])
def test_workspace_is_cleaned_when_provider_fails(tmp_path, monkeypatch, failure):
    source = tmp_path / "source.wav"
    old = _clips(tmp_path / "source_clip", 2)
    generated = []

    def split(*args, **kwargs):
        paths = _clips(Path(kwargs.get("output_dir", tmp_path / "new")), 2)
        generated.extend(paths)
        return paths

    def provider(**kwargs):
        if failure == "exception":
            raise RuntimeError("provider failed")
        return CaptionResult.failed("provider failed")

    monkeypatch.setattr(orchestrator, "split_video_with_imageio_ffmpeg", split)
    if failure == "exception":
        with pytest.raises(RuntimeError, match="provider failed"):
            _call(source, provider)
    else:
        assert not _call(source, provider).is_persistable
    assert generated and all(not path.exists() for path in generated)
    assert all(path.exists() for path in old)


@pytest.mark.parametrize("invalid", ["outside", "count", "missing", "duplicate"])
def test_invalid_segment_manifest_is_rejected_before_provider(tmp_path, monkeypatch, invalid):
    source = tmp_path / "source.wav"
    outside = _clips(tmp_path / "source_clip", 2)

    def split(*args, **kwargs):
        paths = _clips(Path(kwargs.get("output_dir", tmp_path / "new")), 2)
        if invalid == "outside":
            return outside
        if invalid == "count":
            return paths[:1]
        if invalid == "missing":
            paths[0].unlink()
            return paths
        return [paths[0], paths[0]]

    monkeypatch.setattr(orchestrator, "split_video_with_imageio_ffmpeg", split)
    monkeypatch.setattr(orchestrator, "split_media_stream_clips", split)
    with pytest.raises(ValueError, match="[Ss]egment"):
        _call(source, lambda **kw: pytest.fail("invalid manifest reached provider"))
    assert all(path.exists() for path in outside)


def test_fallback_uses_separate_workspace(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    old = _clips(tmp_path / "source_clip", 2)
    used = []

    def ffmpeg(*args, **kwargs):
        directory = Path(kwargs.get("output_dir", tmp_path / "ffmpeg"))
        used.append(directory)
        _clips(directory, 1)
        raise RuntimeError("ffmpeg failed")

    def fallback(*args, **kwargs):
        directory = Path(kwargs.get("output_dir", tmp_path / "fallback"))
        used.append(directory)
        return _clips(directory, 2)

    monkeypatch.setattr(orchestrator, "split_video_with_imageio_ffmpeg", ffmpeg)
    monkeypatch.setattr(orchestrator, "split_media_stream_clips", fallback)
    result = _call(source, lambda **kw: CaptionResult(raw="ok", parsed={"description": "ok"}))
    assert result.is_persistable
    assert used[0] != used[1]
    assert all(not directory.exists() for directory in used)
    assert all(path.exists() for path in old)


def _wave_source(tmp_path):
    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(array("h", range(16000)).tobytes())
    return source


def test_real_ffmpeg_returns_current_csv_manifest_in_order(tmp_path):
    source = _wave_source(tmp_path)
    output = tmp_path / "workspace"
    subs = pysrt.SubRipFile([pysrt.SubRipItem(index=index, start=pysrt.SubRipTime(milliseconds=index * 250),
                                            end=pysrt.SubRipTime(milliseconds=(index + 1) * 250))
                            for index in range(4)])
    paths = split_video_with_imageio_ffmpeg(source, subs, segment_time=.25, output_dir=output)
    assert isinstance(paths, list) and len(paths) == 4
    assert [path.name for path in paths] == [f"input_{index:03d}.wav" for index in range(4)]
    for path in paths:
        assert path.parent == output
        with av.open(str(path)) as container:
            assert next(container.decode(audio=0)).samples > 0


def test_real_pyav_audio_fallback_honors_milliseconds_and_resets_pts(tmp_path):
    source = _wave_source(tmp_path)
    output = tmp_path / "fallback"
    subs = pysrt.SubRipFile([pysrt.SubRipItem(index=12, start=pysrt.SubRipTime(milliseconds=125),
                                           end=pysrt.SubRipTime(milliseconds=375))])
    paths = split_media_stream_clips(source, "audio", subs, output_dir=output)
    assert paths == [output / "input_12.wav"]
    with av.open(str(paths[0])) as container:
        frames = list(container.decode(audio=0))
        assert frames[0].pts == 0
        assert frames[0].to_ndarray().reshape(-1)[0] == 2000
        assert sum(frame.samples for frame in frames) == 4000


@pytest.mark.parametrize("suffix,video_codec,audio_codec", [(".mp4", "libx264", "aac"),
                                                          (".webm", "libvpx-vp9", "libopus")])
def test_real_pyav_video_fallback_keeps_container_compatible_and_resets_pts(tmp_path, suffix, video_codec, audio_codec):
    source = tmp_path / f"video{suffix}"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=red:s=32x32:r=20:d=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
                    "-c:v", video_codec, "-threads", "1", "-c:a", audio_codec, "-shortest", str(source)],
                   check=True, capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    subs = pysrt.SubRipFile([pysrt.SubRipItem(index=index, start=pysrt.SubRipTime(milliseconds=start),
                                            end=pysrt.SubRipTime(milliseconds=start + 250))
                            for index, start in enumerate((250, 500))])
    paths = split_media_stream_clips(source, "video", subs, output_dir=tmp_path / "fallback")
    assert len(paths) == 2
    for path in paths:
        with av.open(str(path)) as container:
            assert len(container.streams.video) == len(container.streams.audio) == 1
            frames = list(container.decode(video=0))
            assert len(frames) == 5
            assert float(frames[0].time) == 0
        with av.open(str(path)) as container:
            frames = list(container.decode(audio=0))
            assert frames and abs(float(frames[0].time)) < .025


@pytest.mark.parametrize("start_ms", [25, 225])
def test_real_pyav_video_fallback_keeps_non_frame_boundary_pts_monotonic(tmp_path, start_ms):
    source = tmp_path / "video.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=size=32x32:rate=20:duration=1",
                    "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(source)],
                   check=True, capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    subs = pysrt.SubRipFile([pysrt.SubRipItem(index=0,
                                             start=pysrt.SubRipTime(milliseconds=start_ms),
                                             end=pysrt.SubRipTime(milliseconds=start_ms + 250))])

    path = split_media_stream_clips(source, "video", subs, output_dir=tmp_path / "fallback")[0]

    with av.open(str(path)) as container:
        times = [float(frame.time) for frame in container.decode(video=0)]
    assert len(times) == 5
    assert all(left < right for left, right in zip(times, times[1:]))
    assert all(delta == pytest.approx(.05, abs=.001)
               for delta in (right - left for left, right in zip(times, times[1:])))


def test_nonempty_plan_rejects_empty_ffmpeg_manifest(tmp_path, monkeypatch):
    def popen(command, **kwargs):
        Path(command[command.index("-segment_list") + 1]).write_text("", encoding="utf-8")
        return SimpleNamespace(returncode=0, communicate=lambda: ("", ""))

    monkeypatch.setattr("utils.stream_util.subprocess.Popen", popen)
    subs = pysrt.SubRipFile([pysrt.SubRipItem(index=index, start=pysrt.SubRipTime(seconds=index),
                                           end=pysrt.SubRipTime(seconds=index + 1)) for index in range(2)])
    with pytest.raises(ValueError, match="[Ss]egment"):
        split_video_with_imageio_ffmpeg(tmp_path / "source.wav", subs, segment_time=1, output_dir=tmp_path / "ffmpeg")


def test_pyav_video_fallback_preserves_delayed_audio_timestamps(tmp_path):
    source = tmp_path / "delayed.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=red:s=32x32:r=20:d=1",
                    "-itsoffset", "0.25", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.75",
                    "-c:v", "libx264", "-threads", "1", "-c:a", "aac", "-shortest", str(source)],
                   check=True, capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    with av.open(str(source)) as container:
        original_start = float(next(container.decode(audio=0)).time)
    assert original_start > .2
    subs = pysrt.SubRipFile([pysrt.SubRipItem(index=0, start=pysrt.SubRipTime(milliseconds=0),
                                           end=pysrt.SubRipTime(milliseconds=500))])
    paths = split_media_stream_clips(source, "video", subs, output_dir=tmp_path / "fallback")
    with av.open(str(paths[0])) as container:
        output_start = float(next(container.decode(audio=0)).time)
    # AAC encoder priming can shift the first decoded frame by one codec frame.
    assert abs(output_start - original_start) <= 1024 / 48000 + .001
    with av.open(str(paths[0])) as container:
        assert float(next(container.decode(video=0)).time) == 0
