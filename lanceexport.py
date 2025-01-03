import argparse
import lance
import io
from PIL import Image
from typing import Optional, Union, List, Dict, Any
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from config import get_supported_extensions, DATASET_SCHEMA, CONSOLE_COLORS
from pathlib import Path
import pysrt
import av

console = Console()
image_extensions = get_supported_extensions("image")
animation_extensions = get_supported_extensions("animation")
video_extensions = get_supported_extensions("video")
audio_extensions = get_supported_extensions("audio")


def format_duration(duration_ms: int) -> str:
    """将毫秒转换为分:秒格式."""
    total_seconds = duration_ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"


def save_blob(
    uri: Path,
    blob: Union[bytes, lance.BlobFile],
    metadata: Dict[str, Any],
    media_type: str,
) -> bool:
    """Save binary blob to file.

    Args:
        uri: Target path
        blob: Binary data or BlobFile
        metadata: File metadata
        media_type: Type of media (image/video/audio)

    Returns:
        bool: True if successful
    """
    try:
        uri.parent.mkdir(parents=True, exist_ok=True)

        # Handle both bytes and BlobFile
        if isinstance(blob, lance.BlobFile):
            with open(uri, "wb") as f:
                while True:
                    chunk = blob.read(8192)  # Read in chunks
                    if not chunk:
                        break
                    f.write(chunk)
        else:
            uri.write_bytes(blob)

        # Print media-specific metadata
        meta_info = []
        if media_type in ["image", "animation"]:
            meta_info.extend(
                [
                    f"{metadata.get('width', 0)}x{metadata.get('height', 0)}",
                    f"{metadata.get('channels', 0)}ch",
                    (
                        f"{metadata.get('num_frames', 1)} frames"
                        if metadata.get("num_frames", 1) > 1
                        else None
                    ),
                ]
            )
        elif media_type == "video":
            duration = metadata.get("duration", 0)
            meta_info.extend(
                [
                    f"{metadata.get('width', 0)}x{metadata.get('height', 0)}",
                    f"{format_duration(duration)}",
                    f"{metadata.get('frame_rate', 0):.1f}fps",
                ]
            )
        elif media_type == "audio":
            duration = metadata.get("duration", 0)
            meta_info.extend(
                [
                    f"{metadata.get('channels', 0)}ch",
                    f"{metadata.get('frame_rate', 0):.1f}Hz",
                    f"{format_duration(duration)}",
                ]
            )

        meta_str = ", ".join(filter(None, meta_info))
        console.print()

        # 使用配置的颜色
        color = CONSOLE_COLORS.get(media_type, "white")
        console.print(
            f"[{color}]{media_type}: {uri} ({meta_str}) saved successfully.[/{color}]"
        )
        return True
    except Exception as e:
        console.print(f"[red]Error saving {media_type} {uri}: {e}[/red]")
        return False


def save_image(
    image_path: str,
    blob: Union[bytes, lance.BlobFile],
    metadata: Dict[str, Any],
    quality: int = 100,
) -> bool:
    """Save image blob to file."""
    return save_blob(
        Path(image_path),
        blob,
        metadata,
        "animation" if metadata.get("num_frames", 1) > 1 else "image",
    )


def save_video(
    uri: Path, blob: Union[bytes, lance.BlobFile], metadata: Dict[str, Any]
) -> bool:
    """Save video file with metadata."""
    return save_blob(uri, blob, metadata, "video")


def save_audio(
    uri: Path, blob: Union[bytes, lance.BlobFile], metadata: Dict[str, Any]
) -> bool:
    """Save audio file with metadata."""
    return save_blob(uri, blob, metadata, "audio")


def save_caption(caption_path: str, caption_lines: List[str], media_type: str) -> bool:
    """Save caption data to disk.

    Args:
        caption_path: Path to save caption file
        caption_lines: List of caption lines
        media_type: Type of media (image/video/audio)

    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        caption_path = Path(caption_path)
        caption_path.parent.mkdir(parents=True, exist_ok=True)

        caption_path = (
            caption_path.parent / f"{caption_path.stem}.txt"
            if media_type == "image"
            else caption_path.parent / f"{caption_path.stem}.srt"
        )

        with open(caption_path, "w", encoding="utf-8") as f:
            if caption_path.suffix == ".srt":
                # For SRT files, preserve all lines including empty ones
                f.write("\n".join(caption_lines))
            else:
                # For TXT files, strip empty lines and whitespace
                for line in caption_lines:
                    if line and line.strip():
                        f.write(line.strip() + "\n")

            console.print()
            console.print(
                f"[{CONSOLE_COLORS['text']}]text: {caption_path} saved successfully.[/{CONSOLE_COLORS['text']}]"
            )
        return True
    except Exception as e:
        console.print(f"[red]Error saving caption: {e}[/red]")
        return False


def extract_from_lance(
    lance_or_path: Union[str, lance.LanceDataset],
    output_dir: str,
    version: str = "gemini",
    clip_with_caption: bool = True,
    caption_dir: Optional[str] = None,
) -> None:
    """
    Extract images and captions from Lance dataset.

    Args:
        lance_or_path: Path to Lance dataset or Lance dataset object
        output_dir: Directory to save extracted images
        caption_dir: Optional directory to save caption files
        save_binary: Whether to save binary data
    """
    ds = (
        lance.dataset(lance_or_path, version=version)
        if isinstance(lance_or_path, str)
        else lance_or_path
    )

    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if caption_dir:
        captions_dir_path = Path(caption_dir)
        captions_dir_path.mkdir(parents=True, exist_ok=True)

    with Progress() as progress:
        task = progress.add_task("[green]Extracting files...", total=ds.count_rows())

        for batch in ds.to_batches():
            # Get all metadata columns
            metadata_batch = {
                field[0]: batch.column(field[0]).to_pylist()
                for field in DATASET_SCHEMA
                if field[0] != "blob"  # Skip blob to save memory
            }
            indices = list(range(len(batch)))
            blobs = ds.take_blobs(indices, "blob")

            for i in range(len(batch)):
                # Create metadata dict for current item
                metadata = {key: values[i] for key, values in metadata_batch.items()}
                uri = Path(metadata["uris"])
                blob = blobs[i]

                media_type = None
                suffix = uri.suffix.lower()
                if suffix in image_extensions:
                    media_type = "image"
                elif suffix in animation_extensions:
                    media_type = "animation"
                elif suffix in video_extensions:
                    media_type = "video"
                elif suffix in audio_extensions:
                    media_type = "audio"
                if not uri.exists() and blob:
                    if media_type:
                        if not save_blob(uri, blob, metadata, media_type):
                            progress.advance(task)
                            continue
                    else:
                        console.print(
                            f"[yellow]Unsupported file format: {suffix}[/yellow]"
                        )
                        progress.advance(task)
                        continue

                # Save caption if available
                caption = metadata.get("captions", [])
                if caption:
                    caption_file_path = (
                        captions_dir_path
                        if caption_dir
                        else uri.parent / f"{uri.stem}.txt"
                    )
                    caption_file_path.parent.mkdir(parents=True, exist_ok=True)
                    save_caption(caption_file_path, caption, media_type)

                    color = CONSOLE_COLORS.get(media_type, "white")
                    if clip_with_caption and (uri.with_suffix(".srt")).exists():
                        subs = pysrt.open(uri.with_suffix(".srt"), encoding="utf-8")
                        with av.open(uri) as in_container:
                            if media_type != "video":
                                video_stream = None
                            else:
                                video_stream = next(
                                    (
                                        s
                                        for s in in_container.streams
                                        if s.type == "video"
                                    ),
                                    None,
                                )
                            # Try to get audio stream if available
                            audio_stream = next(
                                (s for s in in_container.streams if s.type == "audio"),
                                None,
                            )

                            # 添加字幕片段的进度条
                            with Progress(
                                "[progress.description]{task.description}",
                                BarColumn(),
                                "[progress.percentage]{task.percentage:>3.0f}%",
                                TimeRemainingColumn(),
                                console=console,
                            ) as sub_progress:
                                sub_task = sub_progress.add_task(
                                    f"[cyan]Processing subtitles for {uri.name}",
                                    total=len(subs),
                                )

                                for sub in subs:
                                    if len(subs) < 2:
                                        break
                                    clip_path = (
                                        uri.parent
                                        / f"{uri.stem}_clip/{uri.stem}_{sub.index}{uri.suffix}"
                                    )
                                    clip_path.parent.mkdir(parents=True, exist_ok=True)

                                    with av.open(
                                        str(clip_path), mode="w"
                                    ) as out_container:
                                        # copy encoder settings
                                        if video_stream:
                                            out_video_stream = (
                                                out_container.add_stream_from_template(
                                                    template=video_stream
                                                )
                                            )
                                        else:
                                            out_video_stream = None
                                        if audio_stream:
                                            # 为音频流使用特定的设置
                                            if media_type == "video":
                                                codec_name = "aac"
                                            elif uri.suffix == ".mp3":
                                                codec_name = "mp3"
                                            else:
                                                codec_name = "pcm_s16le"
                                            out_audio_stream = out_container.add_stream(
                                                codec_name=codec_name,
                                                rate=audio_stream.rate,  # 保持原始采样率
                                            )
                                            # 复制音频相关参数
                                            out_audio_stream.layout = (
                                                audio_stream.layout
                                            )
                                            out_audio_stream.format = (
                                                audio_stream.format
                                            )
                                        else:
                                            out_audio_stream = None

                                        # 正确计算 start 和 end 时间戳, 单位是 video_stream.time_base
                                        # 使用毫秒并根据 video_stream.time_base 转换
                                        start_seconds = (
                                            sub.start.hours * 3600
                                            + sub.start.minutes * 60
                                            + sub.start.seconds
                                        )
                                        end_seconds = (
                                            sub.end.hours * 3600
                                            + sub.end.minutes * 60
                                            + sub.end.seconds
                                        )
                                        if video_stream:
                                            start_offset = int(
                                                start_seconds
                                                * video_stream.time_base.denominator
                                                / video_stream.time_base.numerator
                                            )  # 开始时间戳偏移量 (基于 video_stream.time_base)
                                        else:
                                            start_offset = int(
                                                start_seconds
                                                * audio_stream.time_base.denominator
                                                / audio_stream.time_base.numerator
                                            )  # 开始时间戳偏移量 (基于 audio_stream.time_base)
                                        # seek to start
                                        in_container.seek(
                                            start_offset,
                                            stream=(
                                                video_stream
                                                if video_stream
                                                else audio_stream
                                            ),
                                        )

                                        # 手动跳过帧 (如果在 seek 之后需要的话)
                                        for frame in in_container.decode(
                                            video_stream, audio_stream
                                        ):
                                            if frame.time > end_seconds:
                                                break

                                            if (
                                                video_stream
                                                and isinstance(frame, av.VideoFrame)
                                                and frame.time >= start_seconds
                                            ):
                                                for packet in out_video_stream.encode(
                                                    frame
                                                ):
                                                    out_container.mux(packet)
                                            elif (
                                                audio_stream
                                                and isinstance(frame, av.AudioFrame)
                                                and frame.time >= start_seconds
                                            ):
                                                for packet in out_audio_stream.encode(
                                                    frame
                                                ):
                                                    out_container.mux(packet)

                                        # Flush streams
                                        if out_video_stream:
                                            for packet in out_video_stream.encode():
                                                out_container.mux(packet)
                                        if out_audio_stream:
                                            for packet in out_audio_stream.encode():
                                                out_container.mux(packet)

                                    console.print(
                                        f"[{color}]{media_type}: {str(clip_path)} saved successfully.[/{color}]"
                                    )
                                    console.print()
                                    console.print(f"Saved clip: {sub.text}")
                                    save_caption(clip_path, [sub.text], "image")
                                    sub_progress.advance(sub_task)

                progress.advance(task)


def main():

    parser = argparse.ArgumentParser(
        description="Extract images and captions from a Lance dataset"
    )
    parser.add_argument("lance_file", help="Path to the .lance file")
    parser.add_argument(
        "--output_dir",
        default="./dataset",
        help="Directory to save extracted data",
    )
    parser.add_argument(
        "--version",
        default="gemini",
        help="Dataset version",
    )

    parser.add_argument(
        "--not_clip_with_caption",
        action="store_true",
        help="Not clip with caption",
    )

    args = parser.parse_args()
    extract_from_lance(args.lance_file, args.output_dir, args.version, not args.not_clip_with_caption)


if __name__ == "__main__":
    main()
