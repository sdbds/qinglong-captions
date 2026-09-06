import hashlib
import subprocess
import sys
from pathlib import Path

import av
import imageio_ffmpeg
import pyarrow as pa
import pytest
from PIL import Image

from module import lanceImport as importer


def _make_video(path, *, audio=True, color="red"):
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "lavfi", "-i", f"color=c={color}:s=32x32:r=10:d=1"]
    if audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=1"]
    command += ["-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p"]
    if audio:
        command += ["-c:a", "aac", "-shortest"]
    command += [str(path)]
    subprocess.run(command, check=True, capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    return path


@pytest.fixture
def source(tmp_path):
    return _make_video(tmp_path / "clip.mp4")


def _tracks(path):
    with av.open(str(path)) as container:
        return len(container.streams.video), len(container.streams.audio)


@pytest.mark.parametrize("mode,tracks,mime", [
    (importer.VideoImportMode.ALL, (1, 1), "video/"),
    (importer.VideoImportMode.VIDEO_ONLY, (1, 0), "video/"),
    (importer.VideoImportMode.AUDIO_ONLY, (0, 1), "audio/"),
])
def test_scalar_modes_return_real_readable_components(source, mode, tracks, mime):
    original = source.read_bytes()
    metadata = importer.FileProcessor.load_metadata(str(source), save_binary=True, import_mode=mode)
    assert metadata is not None
    assert _tracks(metadata.uris) == tracks
    assert metadata.mime.startswith(mime)
    assert metadata.blob == Path(metadata.uris).read_bytes()
    assert metadata.hash == hashlib.sha256(metadata.blob).hexdigest()
    assert source.read_bytes() == original
    if mode == importer.VideoImportMode.ALL:
        assert metadata.uris == str(source)
    else:
        assert Path(metadata.uris) != source
        assert ".qinglong-import" in Path(metadata.uris).parts
    with av.open(metadata.uris) as container:
        assert next(container.decode(video=0) if tracks[0] else container.decode(audio=0)) is not None


@pytest.mark.parametrize("save_binary", [False, True])
def test_split_batch_has_two_distinct_persistent_uris(source, save_binary):
    items = [{"file_path": str(source), "caption": ["Original caption"], "chunk_offsets": [0]}]
    batches = list(importer.process(items, save_binary=save_binary,
                                    import_mode=importer.VideoImportMode.VIDEO_SPLIT_AUDIO))
    rows = [row for batch in batches for row in batch.to_pylist()]
    assert len(rows) == 2
    assert len({row["uris"] for row in rows}) == 2
    assert [_tracks(row["uris"]) for row in rows] == [(1, 0), (0, 1)]
    assert [row["has_audio"] for row in rows] == [False, True]
    for row in rows:
        assert row["captions"] == ["Original caption"]
        assert row["chunk_offsets"] == [0]
        assert Path(row["uris"]).exists()
        assert row["blob"] == (Path(row["uris"]).read_bytes() if save_binary else b"")


def test_component_import_recomputes_source_metadata_and_preserves_business_fields(source):
    schema = pa.schema([
        pa.field("uris", pa.string()),
        pa.field("mime", pa.string()),
        pa.field("hash", pa.string()),
        pa.field("size", pa.int64()),
        pa.field("has_audio", pa.bool_()),
        pa.field("blob", pa.large_binary()),
        pa.field("captions", pa.list_(pa.string())),
        pa.field("chunk_offsets", pa.list_(pa.int32())),
        pa.field("business_id", pa.string()),
    ])
    item = {
        "file_path": str(source),
        "uris": str(source),
        "mime": "video/stale",
        "hash": "source-hash",
        "size": -1,
        "has_audio": False,
        "blob": b"source-blob",
        "caption": ["Original caption"],
        "chunk_offsets": [7],
        "business_id": "asset-42",
    }

    rows = [row for batch in importer.process(
        [item], save_binary=True, import_mode=importer.VideoImportMode.AUDIO_ONLY, schema=schema
    ) for row in batch.to_pylist()]

    assert len(rows) == 1
    component = Path(rows[0]["uris"])
    content = component.read_bytes()
    assert component.suffix == ".wav"
    assert rows[0]["mime"].startswith("audio/")
    assert rows[0]["hash"] == hashlib.sha256(content).hexdigest()
    assert rows[0]["size"] == len(content)
    assert rows[0]["has_audio"] is True
    assert rows[0]["blob"] == content
    assert rows[0]["captions"] == ["Original caption"]
    assert rows[0]["chunk_offsets"] == [7]
    assert rows[0]["business_id"] == "asset-42"


def test_custom_loader_component_import_recomputes_derived_metadata(source, monkeypatch):
    monkeypatch.setattr(importer, "DATASET_SCHEMA", tuple(importer.DATASET_SCHEMA) + (
        ("filename", pa.string()),
        ("ext", pa.string()),
        ("bits_per_channel", pa.int32()),
        ("bit_rate", pa.int64()),
    ))

    def custom_loader(*_args, **_kwargs):
        return [{
            "file_path": str(source),
            "caption": [],
            "chunk_offsets": [],
            "filename": "stale-source-name",
            "ext": ".mp4",
            "bits_per_channel": 999,
            "bit_rate": 999,
        }]

    dataset = importer.transform2lance(
        str(source.parent),
        output_name="derived-metadata",
        load_condition=custom_loader,
        import_mode=importer.VideoImportMode.AUDIO_ONLY,
    )
    row = dataset.to_table(columns=[
        "uris", "filename", "ext", "bits_per_channel", "bit_rate",
    ]).to_pylist()[0]

    assert Path(row["uris"]).suffix == ".wav"
    assert row["filename"] == "audio"
    assert row["ext"] == ".wav"
    assert row["bits_per_channel"] == 16
    assert row["bit_rate"] == 256000


def test_transform_split_writes_two_component_records(source):
    dataset = importer.transform2lance(str(source.parent), import_mode=importer.VideoImportMode.VIDEO_SPLIT_AUDIO,
                                       save_binary=False, data_storage_version="2.1")
    rows = dataset.to_table(columns=["uris", "mime"]).to_pylist()
    assert len(rows) == 2
    assert [_tracks(row["uris"]) for row in rows] == [(1, 0), (0, 1)]


@pytest.mark.parametrize("external_captions", [False, True])
def test_directory_scan_never_reimports_generated_components(source, external_captions):
    cache = source.parent / ".qinglong-import" / "previous"
    cache.mkdir(parents=True)
    (cache / "audio.wav").write_bytes(b"cached media")
    captions = source.parent / "captions"
    captions.mkdir()
    items = list(importer.iter_data_items(str(source.parent), str(captions) if external_captions else None))
    assert [item["file_path"] for item in items] == [str(source)]


def test_source_content_change_does_not_reuse_old_components(source):
    first = importer.FileProcessor.load_metadata(str(source), save_binary=False,
                                                 import_mode=importer.VideoImportMode.VIDEO_ONLY)
    first_content = Path(first.uris).read_bytes()
    _make_video(source, color="blue")
    second = importer.FileProcessor.load_metadata(str(source), save_binary=False,
                                                  import_mode=importer.VideoImportMode.VIDEO_ONLY)
    assert first.uris != second.uris
    assert Path(first.uris).read_bytes() == first_content
    assert _tracks(second.uris) == (1, 0)


def test_no_audio_source_skips_audio_component(tmp_path):
    source = _make_video(tmp_path / "silent.mp4", audio=False)
    assert importer.FileProcessor.load_metadata(str(source), import_mode=importer.VideoImportMode.AUDIO_ONLY) is None
    rows = [row for batch in importer.process([{"file_path": str(source), "chunk_offsets": []}],
            import_mode=importer.VideoImportMode.VIDEO_SPLIT_AUDIO) for row in batch.to_pylist()]
    assert len(rows) == 1
    assert _tracks(rows[0]["uris"]) == (1, 0)


def test_failed_component_write_leaves_original_and_no_partial_target(source, monkeypatch):
    original = source.read_bytes()

    def fail(command, **kwargs):
        Path(command[-1]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, command, stderr=b"conversion failed")

    monkeypatch.setattr(subprocess, "run", fail)
    result = importer.FileProcessor.load_metadata(str(source), import_mode=importer.VideoImportMode.VIDEO_ONLY)
    assert result is None
    assert source.read_bytes() == original
    assert not [path for path in (source.parent / ".qinglong-import").rglob("*") if path.is_file()]


def test_scalar_split_requires_batch_entrypoint(source):
    with pytest.raises(ValueError, match="process"):
        importer.FileProcessor.load_metadata(str(source), import_mode=importer.VideoImportMode.VIDEO_SPLIT_AUDIO)


@pytest.mark.parametrize("mode", list(importer.VideoImportMode))
def test_video_modes_leave_non_video_assets_unchanged(tmp_path, mode):
    source = tmp_path / "still.png"
    Image.new("RGB", (2, 2)).save(source)
    metadata = importer.FileProcessor.load_metadata(str(source), import_mode=mode)
    assert metadata.uris == str(source)
    assert metadata.blob == source.read_bytes()
