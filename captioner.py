import lance
from rich_pixels import Pixels
from rich.progress import Progress
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.layout import Layout
import argparse
from lanceImport import transform2lance
from lanceexport import extract_from_lance
import pyarrow as pa
import re
from utils.stream_util import split_media_stream_clips, split_video_with_imageio_ffmpeg

# from mistralai import Mistral
import toml
from PIL import Image
import pysrt
from api_handler import api_process_batch
from pathlib import Path

Image.MAX_IMAGE_PIXELS = None  # Disable image size limit check

console = Console()


def process_batch(args, config):
    # Load the dataset
    if not isinstance(args.dataset_dir, lance.LanceDataset):
        if args.api_key == "":
            dataset = transform2lance(dataset_dir=args.dataset_dir)
        else:
            dataset = transform2lance(dataset_dir=args.dataset_dir, save_binary=False)

    scanner = dataset.scanner(
        columns=["uris", "blob", "mime", "captions", "duration"],
        scan_in_order=True,
        late_materialization=["blob"],
        batch_size=1,
    )
    total_rows = dataset.count_rows()

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing videos...", total=total_rows)

        results = []
        processed_filepaths = []
        for batch in scanner.to_batches():
            filepaths = batch["uris"].to_pylist()
            mime = batch["mime"].to_pylist()
            duration = batch["duration"].to_pylist()

            for filepath, mime, duration in zip(filepaths, mime, duration):

                if duration and 0 < duration <= 5 * 60 * 1000:

                    output = api_process_batch(
                        uri=filepath,
                        mime=mime,
                        config=config,
                        api_key=args.api_key,
                        wait_time=1,
                        max_retries=100,
                        model_path=args.model_path,
                    )

                    output = _postprocess_caption_content(output, filepath)

                else:
                    console.print(f"[blue]{filepath} video > 5 minutes [/blue]")
                    console.print(f"[blue]split video[/blue]")

                    # 创建用于分割的字幕文件
                    subs = pysrt.SubRipFile()

                    # 计算分块
                    duration_seconds = duration / 1000  # 将毫秒转换为秒
                    chunk_duration = 300  # 5分钟 = 300秒
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
                        split_video_with_imageio_ffmpeg(Path(filepath), subs)
                    except Exception as e:
                        # 使用字幕分割视频
                        meta_type = "video" if mime.startswith("video") else "audio"
                        console.print(
                            f"[red]Error splitting video with imageio-ffmpeg: {e}[/red]"
                        )
                        split_media_stream_clips(Path(filepath), meta_type, subs)

                    pathfile = Path(filepath)
                    files = sorted(pathfile.parent.glob(f"{pathfile.stem}_clip/{pathfile.stem}_*{pathfile.suffix}"))
                    merged_subs = pysrt.SubRipFile()
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
                            api_key=args.api_key,
                            wait_time=1,
                            max_retries=100,
                            model_path=args.model_path,
                        )

                        console.print(
                            f"[green]API processing complete for chunk {i+1}[/green]"
                        )

                        console.print(
                            f"[yellow]Post-processing chunk output...[/yellow]"
                        )
                        chunk_output = _postprocess_caption_content(
                            chunk_output, uri
                        )

                        uri.unlink(missing_ok=True)

                        chunk_subs = pysrt.from_string(chunk_output)
                        if len(merged_subs) > 0:
                            last_end = merged_subs[-1].end
                            console.print(
                                f"[yellow]Shifting subtitles by {last_end.hours}h {last_end.minutes}m {last_end.seconds}s {last_end.milliseconds}ms[/yellow]"
                            )
                            # Shift all subtitles in the chunk by the end time of the last subtitle
                            chunk_subs.shift(
                                minutes=5 * i,
                            )
                            # Update indices to continue from the last subtitle
                            for j, sub in enumerate(
                                chunk_subs, start=len(merged_subs) + 1
                            ):
                                sub.index = j

                        # Extend merged subtitles with the shifted chunk
                        merged_subs.extend(chunk_subs)
                        console.print(
                            f"[green]Successfully merged chunk {i+1}. Total subtitles: {len(merged_subs)}[/green]"
                        )

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
                caption_path = filepath_path.with_suffix(".srt")
                console.print(f"[blue]Processing caption for:[/blue] {filepath_path}")
                console.print(f"[blue]Caption content length:[/blue] {len(output)}")

                try:
                    subs = pysrt.from_string(output)
                    subs.save(str(caption_path), encoding="utf-8")
                    console.print(f"[green]Saved captions to {caption_path}[/green]")
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


def _postprocess_caption_content(output, filepath):
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
    output = re.sub(
        r"(?<!:)(\d{2}):(\d{2}),(\d{3})",
        r"00:\1:\2,\3",
        output,
        flags=re.MULTILINE,
    )

    return output


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_dir", type=str, help="directory for dataset")

    parser.add_argument(
        "--systemprompt",
        type=str,
        help="directory for train images",
    )

    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="API key for pixtral API",
    )

    parser.add_argument(
        "--dir_name",
        action="store_true",
        help="Use the directory name as the dataset name",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="gemini-exp-1206",
        help="Model path for gemini",
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

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()

    config = toml.load(args.config)

    process_batch(args, config)
