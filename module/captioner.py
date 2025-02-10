import lance
from rich.progress import Progress
from rich.console import Console
import argparse
from module.lanceImport import transform2lance
from module.lanceexport import extract_from_lance
from module.api_handler import api_process_batch, process_llm_response
import pyarrow as pa
import re
from utils.stream_util import split_media_stream_clips, split_video_with_imageio_ffmpeg
from config.config import (
    BASE_VIDEO_EXTENSIONS,
    BASE_AUDIO_EXTENSIONS,
)

import toml
from PIL import Image
import pysrt
from pymediainfo import MediaInfo
from pathlib import Path

Image.MAX_IMAGE_PIXELS = None  # Disable image size limit check

console = Console()


def process_batch(args, config):
    # Load the dataset
    if not isinstance(args.dataset_dir, lance.LanceDataset):
        if args.gemini_api_key == "" and args.pixtral_api_key == "":
            dataset = transform2lance(dataset_dir=args.dataset_dir)
        else:
            dataset = transform2lance(dataset_dir=args.dataset_dir, save_binary=False)

    scanner = dataset.scanner(
        columns=["uris", "blob", "mime", "captions", "duration", "hash"],
        scan_in_order=True,
        late_materialization=["blob"],
        batch_size=1,
    )
    total_rows = dataset.count_rows()

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing media...", total=total_rows)

        results = []
        processed_filepaths = []
        for batch in scanner.to_batches():
            filepaths = batch["uris"].to_pylist()
            mime = batch["mime"].to_pylist()
            duration = batch["duration"].to_pylist()
            sha256hash = batch["hash"].to_pylist()

            for filepath, mime, duration, sha256hash in zip(
                filepaths, mime, duration, sha256hash
            ):

                if mime.startswith("image") or 0 < duration <= args.segment_time * 1000:

                    output = api_process_batch(
                        uri=filepath,
                        mime=mime,
                        config=config,
                        args=args,
                        sha256hash=sha256hash,
                    )

                    output = _postprocess_caption_content(
                        output, filepath, mode=args.mode
                    )

                else:
                    console.print(
                        f"[blue]{filepath} video > {args.segment_time} seconds[/blue]"
                    )
                    console.print(f"[blue]split video[/blue]")

                    # 创建用于分割的字幕文件
                    subs = pysrt.SubRipFile()

                    # 计算分块
                    duration_seconds = duration / 1000  # 将毫秒转换为秒
                    chunk_duration = args.segment_time
                    num_chunks = int(
                        (duration_seconds + chunk_duration - 1) // chunk_duration
                    )

                    # 创建字幕条目
                    for i in range(num_chunks):
                        start_time = i * chunk_duration
                        end_time = min((i + 1) * chunk_duration, duration_seconds)

                        # 创建字幕条目
                        sub = pysrt.SubRipItem(
                            index=i,
                            start=pysrt.SubRipTime(seconds=start_time),
                            end=pysrt.SubRipTime(seconds=end_time),
                            text=f"Chunk {i + 1}",
                        )
                        subs.append(sub)

                    for sub in subs:
                        console.print(f"[blue]Subtitles created:[/blue] {sub}")
                    try:
                        split_video_with_imageio_ffmpeg(
                            Path(filepath),
                            subs,
                            save_caption_func=None,
                            segment_time=args.segment_time,
                        )
                    except Exception as e:
                        # 使用字幕分割视频
                        meta_type = "video" if mime.startswith("video") else "audio"
                        console.print(
                            f"[red]Error splitting video with imageio-ffmpeg: {e}[/red]"
                        )
                        split_media_stream_clips(Path(filepath), meta_type, subs)

                    pathfile = Path(filepath)
                    clip_dir = pathfile.parent / f"{pathfile.stem}_clip"

                    search_pattern = f"*{pathfile.suffix}"
                    files = sorted(clip_dir.glob(search_pattern))

                    merged_subs = pysrt.SubRipFile()
                    total_duration = 0
                    for i in range(num_chunks):
                        sub_path = Path(filepath).with_suffix(".srt")
                        if sub_path.exists():
                            sub = pysrt.open(sub_path, encoding="utf-8")
                            merged_subs.extend(sub)

                        console.print(
                            f"[yellow]Processing chunk {i+1}/{num_chunks}[/yellow]"
                        )
                        uri = files[i]

                        chunk_output = api_process_batch(
                            uri=uri,
                            mime=mime,
                            config=config,
                            args=args,
                            sha256hash=sha256hash,
                        )

                        console.print(
                            f"[green]API processing complete for chunk {i+1}[/green]"
                        )

                        console.print(
                            f"[yellow]Post-processing chunk output...[/yellow]"
                        )
                        chunk_output = _postprocess_caption_content(chunk_output, uri)

                        chunk_subs = pysrt.from_string(chunk_output)
                        if i > 0:
                            for track in MediaInfo.parse(files[i - 1]).tracks:
                                if track.track_type == "Video":
                                    last_duration = track.duration
                                    break
                                elif track.track_type == "Audio":
                                    last_duration = track.duration

                            total_duration += int(float(last_duration))
                            # 将纯毫秒单位转换为分、秒、毫秒
                            last_duration_minutes = int(total_duration / 60000)
                            last_duration_seconds = int((total_duration % 60000) / 1000)
                            last_duration_milliseconds = total_duration % 1000

                            console.print(
                                f"[yellow]Total shift duration: {last_duration_minutes}m {last_duration_seconds}s {last_duration_milliseconds}ms[/yellow]"
                            )
                            # Shift all subtitles in the chunk
                            chunk_subs.shift(
                                minutes=last_duration_minutes,
                                seconds=last_duration_seconds,
                                milliseconds=last_duration_milliseconds,
                            )

                        # Extend merged subtitles with the shifted chunk
                        merged_subs.extend(chunk_subs)
                        console.print(
                            f"[green]Successfully merged chunk {i+1}. Total subtitles: {len(merged_subs)}[/green]"
                        )

                    for file in files:
                        file.unlink(missing_ok=True)

                    # Update indices to continue from the last subtitle
                    merged_subs.clean_indexes()
                    # 手动构建 SRT 格式
                    output = ""
                    for i, sub in enumerate(merged_subs, start=1):
                        # 格式: 序号 + 时间戳 + 文本
                        output += f"{i}\n"
                        output += f"{sub.start} --> {sub.end}\n"
                        output += f"{sub.text}\n\n"
                    if output:
                        console.print(
                            f"[green]All subtitles merged successfully. Total: {len(merged_subs)}[/green]"
                        )

                results.append(output)
                processed_filepaths.append(filepath)

                filepath_path = Path(filepath)
                caption_path = (
                    filepath_path.with_suffix(".srt")
                    if filepath_path.suffix in BASE_VIDEO_EXTENSIONS
                    or filepath_path.suffix in BASE_AUDIO_EXTENSIONS
                    else filepath_path.with_suffix(".txt")
                )
                console.print(f"[blue]Processing caption for:[/blue] {filepath_path}")
                console.print(f"[blue]Caption content length:[/blue] {len(output)}")

                if caption_path.suffix == ".srt":
                    try:
                        subs = pysrt.from_string(output)
                        subs.save(str(caption_path), encoding="utf-8")
                        console.print(
                            f"[green]Saved captions to {caption_path}[/green]"
                        )
                    except Exception as e:
                        console.print(
                            f"[yellow]pysrt validation failed: {e}, falling back to direct file write[/yellow]"
                        )
                        try:
                            caption_path.write_text(output, encoding="utf-8")
                            console.print(
                                f"[green]Saved captions to {caption_path}[/green]"
                            )
                        except Exception as e:
                            console.print(f"[red]Error saving SRT file: {e}[/red]")
                else:
                    try:
                        if isinstance(output, list):
                            with open(caption_path, "w", encoding="utf-8") as f:
                                for line in output:
                                    f.write(line + "\n")
                        else:
                            caption_path.write_text(output, encoding="utf-8")
                        console.print(
                            f"[green]Saved captions to {caption_path}[/green]"
                        )
                    except Exception as e:
                        console.print(f"[red]Error saving TXT file: {e}[/red]")

            progress.update(task, advance=len(batch))

    # Update dataset with new captions
    if results:
        # 确保每个caption都是单个字符串
        processed_captions = []
        for caption in results:
            if isinstance(caption, list):
                # 如果是列表，合并为单个字符串
                processed_captions.append("\n".join(caption))
            else:
                processed_captions.append(caption)

        table = pa.table(
            {
                "uris": pa.array(processed_filepaths, type=pa.string()),
                "captions": pa.array(
                    [[caption] for caption in processed_captions],
                    type=pa.list_(pa.string()),
                ),
            }
        )

        dataset.merge_insert(on="uris").when_matched_update_all().execute(table)

        try:
            dataset.tags.create("gemini", 1)
        except:
            dataset.tags.update("gemini", 1)

        console.print("[green]Successfully updated dataset with new captions[/green]")

    extract_from_lance(
        dataset, args.dataset_dir, clip_with_caption=not args.not_clip_with_caption
    )


def _postprocess_caption_content(output, filepath, mode="all"):
    """
    postprocess_caption_content
    """
    if not output:
        console.print(f"[red]No caption content generated for {filepath}[/red]")
        return ""

    if isinstance(output, list):
        output = "\n".join(output)

    # 确保字幕内容格式正确
    output = output.strip()
    if not output.strip():
        console.print(f"[red]Empty caption content for {filepath}[/red]")
        return ""

    # 格式化时间戳 - 只处理7位的时间戳 (MM:SS,ZZZ)
    if (
        Path(filepath).suffix in BASE_VIDEO_EXTENSIONS
        or Path(filepath).suffix in BASE_AUDIO_EXTENSIONS
    ):
        output = re.sub(
            r"(?<!:)(\d{2}):(\d{2})[,:.](\d{3})",
            r"00:\1:\2,\3",
            output,
            flags=re.MULTILINE,
        )
    else:
        if "###" in output:
            shortdescription, long_description = process_llm_response(output)
            if mode == "all":
                output = [shortdescription, long_description]
            elif mode == "long":
                output = long_description
            elif mode == "short":
                output = shortdescription

    return output


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_dir", type=str, help="directory for dataset")

    parser.add_argument(
        "--gemini_api_key",
        type=str,
        default="",
        help="API key for gemini API",
    )

    parser.add_argument(
        "--gemini_model_path",
        type=str,
        default="gemini-exp-1206",
        help="Model path for gemini",
    )

    parser.add_argument(
        "--step_api_key",
        type=str,
        default="",
        help="API key for step API",
    )

    parser.add_argument(
        "--step_model_path",
        type=str,
        default="step-1.5v-mini",
        help="video model for step",
    )

    parser.add_argument(
        "--pixtral_api_key",
        type=str,
        default="",
        help="API key for pixtral API",
    )

    parser.add_argument(
        "--pixtral_model_path",
        type=str,
        default="pixtral-large-2411",
        help="Model path for pixtral",
    )

    parser.add_argument(
        "--dir_name",
        action="store_true",
        help="Use the directory name as the dataset name",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        help="Mode for processing the dataset",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.toml",
        help="Path to config file",
    )

    parser.add_argument(
        "--not_clip_with_caption",
        action="store_true",
        help="Not clip with caption",
    )

    parser.add_argument(
        "--wait_time",
        type=int,
        default=1,
        help="Wait time",
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=20,
        help="Max retries",
    )

    parser.add_argument(
        "--segment_time",
        type=int,
        default=300,
        help="Segment time",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()

    config = toml.load(args.config)

    process_batch(args, config)
