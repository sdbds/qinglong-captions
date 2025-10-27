import lance
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
    MofNCompleteColumn,
)
from rich.console import Console
from PIL import Image
import json
import pyarrow as pa
from module.lanceImport import transform2lance
from module.lanceexport import extract_from_lance
from module.api_handler import api_process_batch
from utils.parse_display import process_llm_response
from utils.stream_util import (
    split_media_stream_clips,
    split_video_with_imageio_ffmpeg,
    get_video_duration,
)
from config.config import (
    BASE_VIDEO_EXTENSIONS,
    BASE_AUDIO_EXTENSIONS,
    BASE_APPLICATION_EXTENSIONS,
)
import re
import argparse
import toml
import pysrt
from pathlib import Path
import base64
import asyncio

Image.MAX_IMAGE_PIXELS = None  # Disable image size limit check

console = Console(color_system="truecolor", force_terminal=True)


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

    with Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(spinner_name="dots"),
        MofNCompleteColumn(separator="/"),
        BarColumn(bar_width=40, complete_style="green", finished_style="bold green"),
        TextColumn("•"),
        TaskProgressColumn(),
        TextColumn("•"),
        TransferSpeedColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        expand=True,
    ) as progress:
        task = progress.add_task("[bold cyan]Processing media...", total=total_rows)

        results = []
        scene_detectors = {}
        processed_filepaths = []
        for batch in scanner.to_batches():
            filepaths = batch["uris"].to_pylist()
            mime = batch["mime"].to_pylist()
            duration = batch["duration"].to_pylist()
            sha256hash = batch["hash"].to_pylist()

            for filepath, mime, duration, sha256hash in zip(
                filepaths, mime, duration, sha256hash
            ):
                # 创建场景检测器，但异步初始化它（不阻塞主线程）
                scene_detector = None
                if (
                    args.scene_threshold > 0
                    and args.scene_min_len > 0
                    and mime.startswith("video")
                ):
                    from module.videospilter import SceneDetector, run_async_in_thread
                    scene_detector = SceneDetector(
                        detector=args.scene_detector,
                        threshold=args.scene_threshold,
                        min_scene_len=args.scene_min_len,
                        luma_only=args.scene_luma_only,
                        console=progress,
                    )
                    # 启动场景检测，直接使用协程对象
                    coroutine = scene_detector.detect_scenes_async(filepath)
                    run_async_in_thread(coroutine)
                    # 保存检测器实例以便后续使用
                    scene_detectors[filepath] = scene_detector

                if (
                    mime.startswith("image")
                    or duration <= (args.segment_time + 1) * 1000
                ):

                    output = api_process_batch(
                        uri=filepath,
                        mime=mime,
                        config=config,
                        args=args,
                        sha256hash=sha256hash,
                        progress=progress,
                        task_id=task,
                    )

                    output = _postprocess_caption_content(
                        output,
                        filepath,
                        args,
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

                    # 使用全局变量来存储clip_task_id
                    global clip_task_id

                    # 检查是否已经存在clip任务
                    if "clip_task_id" in globals() and clip_task_id in [
                        task.id for task in progress.tasks
                    ]:
                        # 重置已存在的clip任务
                        progress.reset(
                            clip_task_id,
                            total=num_chunks,
                            visible=True,
                            description=f"[cyan]Processing clips...",
                        )
                        clip_task = clip_task_id
                    else:
                        # 创建新的clip任务
                        clip_task = progress.add_task(
                            f"[cyan]Processing clips...", total=num_chunks
                        )
                        clip_task_id = clip_task
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
                            progress=progress,
                            task_id=task,
                        )

                        console.print(
                            f"[green]API processing complete for chunk {i+1}[/green]"
                        )

                        console.print(
                            f"[yellow]Post-processing chunk output...[/yellow]"
                        )
                        chunk_output = _postprocess_caption_content(
                            chunk_output, uri, args
                        )

                        chunk_subs = pysrt.from_string(chunk_output)
                        # 检查并删除超时的字幕
                        for sub in list(chunk_subs):  # 使用list创建副本以便安全删除
                            if (
                                sub.start.ordinal > args.segment_time * 1000
                            ):  # 转换为毫秒比较
                                chunk_subs.remove(sub)

                        if i > 0:
                            last_duration = get_video_duration(files[i - 1])

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

                        progress.update(
                            clip_task,
                            advance=1,
                            refresh=True,
                            description=f"[yellow]merging complete for chunk [/yellow]",
                        )

                    # Mark the clip task as completed and hide it
                    progress.update(clip_task, completed=num_chunks, visible=False)

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

                    for file in files:
                        file.unlink(missing_ok=True)

                results.append(output)
                processed_filepaths.append(filepath)

                filepath_path = Path(filepath)
                # Determine caption file extension based on media type
                if (
                    filepath_path.suffix in BASE_VIDEO_EXTENSIONS
                    or filepath_path.suffix in BASE_AUDIO_EXTENSIONS
                ):
                    caption_path = filepath_path.with_suffix(".srt")
                elif filepath_path.suffix in BASE_APPLICATION_EXTENSIONS:
                    caption_path = filepath_path.with_suffix(".md")
                else:
                    caption_path = filepath_path.with_suffix(".txt")
                console.print(f"[blue]Processing caption for:[/blue] {filepath_path}")
                if isinstance(output, dict):
                    console.print(
                        f"[blue]Caption content length:[/blue] {len(output['description'])}"
                    )
                else:
                    console.print(f"[blue]Caption content length:[/blue] {len(output)}")

                if caption_path.suffix == ".srt" and output:
                    try:
                        subs = pysrt.from_string(output)
                        if scene_detector:
                            # 检查场景检测是否已经完成
                            console.print(
                                f"[bold cyan]{scene_detectors[filepath].get_scene_list()}...[/bold cyan]"
                            )
                            if scene_detectors[filepath].get_scene_list() is None:
                                scene_list = asyncio.run(
                                    scene_detectors[filepath].ensure_detection_complete(
                                        filepath
                                    )
                                )
                            else:
                                scene_list = scene_detectors[filepath].get_scene_list()
                            # 使用实例方法align_subtitle，传入scene_list参数
                            console.print(
                                f"[bold cyan]Aligning subtitles with scene changes...[/bold cyan]"
                            )
                            subs = scene_detectors[filepath].align_subtitle(
                                subs,
                                scene_list=scene_list,
                                console=console,
                                segment_time=args.segment_time,
                            )
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
                elif caption_path.suffix == ".md" and output:
                    try:
                        with open(caption_path, "w", encoding="utf-8") as f:
                            f.write(output)
                        console.print(
                            f"[green]Saved captions to {caption_path}[/green]"
                        )
                    except Exception as e:
                        console.print(f"[red]Error saving MD file: {e}[/red]")
                elif output:
                    try:
                        if isinstance(output, list):
                            with open(caption_path, "w", encoding="utf-8") as f:
                                for line in output:
                                    f.write(line + "\n")
                        elif isinstance(output, dict):
                            with open(
                                filepath_path.with_suffix(".json"),
                                "w",
                                encoding="utf-8",
                            ) as f:
                                json.dump(output, f, indent=2, ensure_ascii=False)
                            with open(caption_path, "w", encoding="utf-8") as f:
                                if "description" in output and output["description"]:
                                    f.write(output["description"])
                                else:
                                    f.write("No description available")
                        else:
                            caption_path.write_text(output, encoding="utf-8")
                        console.print(
                            f"[green]Saved captions to {caption_path}[/green]"
                        )
                    except Exception as e:
                        console.print(f"[red]Error saving TXT file: {e}[/red]")

            progress.update(task, advance=len(batch))

    # 处理完所有批次后，将主任务设置为不可见
    progress.update(task, visible=False)

    # Update dataset with new captions
    if results:
        # 确保每个caption都是单个字符串
        processed_captions = []
        for caption in results:
            if isinstance(caption, list):
                # 如果是列表，合并为单个字符串
                processed_captions.append("\n".join(caption))
            elif isinstance(caption, dict):
                processed_captions.append(json.dumps(caption, ensure_ascii=False))
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


def _postprocess_caption_content(output, filepath, args):
    """
    postprocess_caption_content
    """
    if not output:
        console.print(f"[red]No caption content generated for {filepath}[/red]")
        return ""

    if isinstance(output, list):
        # 检查是否为OCRPageObject对象列表
        if (
            len(output) > 0
            and hasattr(output[0], "markdown")
            and hasattr(output[0], "index")
        ):
            combined_output = ""
            for page in output:
                # 添加页面索引作为HTML页眉和页脚
                page_index = page.index if hasattr(page, "index") else "unknown"
                # 添加HTML页眉
                combined_output += f'<header style="background-color: #f5f5f5; padding: 8px; margin-bottom: 20px; text-align: center; border-bottom: 1px solid #ddd;">\n<strong> Page {page_index+1} </strong>\n</header>\n\n'
                # 添加页面内容
                page_markdown = (
                    page.markdown if hasattr(page, "markdown") else str(page)
                )
                # 替换图片路径，将图片路径改为上一级目录
                if hasattr(page, "images") and args.document_image:
                    # 查找并替换所有图片引用格式 ![...](filename)
                    img_pattern = r"!\[(.*?)\]\(([^/)]+)\)"
                    parent_dir = Path(filepath).stem
                    page_markdown = re.sub(
                        img_pattern,
                        lambda m: f"![{m.group(1)}]({parent_dir}/{m.group(2)})",
                        page_markdown,
                    )

                if hasattr(page, "images") and args.document_image:
                    for image in page.images:
                        if hasattr(image, "image_base64") and image.image_base64:
                            try:
                                base64_str = image.image_base64
                                # 处理data URL格式
                                if base64_str.startswith("data:"):
                                    # 提取实际的base64内容
                                    base64_content = base64_str.split(",", 1)[1]
                                    image_data = base64.b64decode(base64_content)
                                else:
                                    image_data = base64.b64decode(base64_str)

                                image_filename = image.id
                                image_dir = Path(filepath).with_suffix("")
                                image_dir.mkdir(parents=True, exist_ok=True)
                                image_path = image_dir / image_filename
                                with open(image_path, "wb") as img_file:
                                    img_file.write(image_data)
                            except Exception as e:
                                console.print(
                                    f"[yellow]Error saving OCR image: {e}[/yellow]"
                                )
                # 这里添加页面内容，只添加一次
                combined_output += f"{page_markdown}\n\n"
                # 添加HTML页脚和分隔符
                combined_output += f'<footer style="background-color: #f5f5f5; padding: 8px; margin-top: 20px; text-align: center; border-top: 1px solid #ddd;">\n<strong> Page {page_index+1} </strong>\n</footer>\n\n'
                combined_output += '<div style="page-break-after: always;"></div>\n\n'
            output = combined_output
        else:
            output = "\n".join(output)

    # 确保字幕内容格式正确
    output = output.strip()
    if not output.strip():
        console.print(f"[red]Empty caption content for {filepath}[/red]")
        return ""

    # 格式化时间戳 - 只处理视频和音频文件的字幕
    if (
        Path(filepath).suffix in BASE_VIDEO_EXTENSIONS
        or Path(filepath).suffix in BASE_AUDIO_EXTENSIONS
    ):
        # 确保字幕内容格式正确
        output = output.strip()
        if not output.strip():
            console.print(f"[red]Empty caption content for {filepath}[/red]")
            return ""

        # 使用单一的正则表达式和处理函数来修复时间戳格式
        # 创建一个匹配各种时间戳格式的模式
        timestamp_pattern = re.compile(
            # 匹配格式1: M:SS,mmm (单位数分钟)
            r"(?<!:)(\d):(\d{2})[,:.](\d{3})|"
            # 匹配格式2: MM:SS,mmm (两位数分钟)
            r"(?<!:)(\d{2}):(\d{2})[,:.](\d{3})|"
            # 匹配格式3: HH:MM:SS,mmm (标准时间)
            r"(?<!:)(\d{2}):(\d{2}):(\d{2})[,:.](\d{3})|"
            # 匹配格式4: 大于99的小时值
            r"(?<![0-9:])([1-9][0-9][0-9]+):(\d{2}):(\d{2})[,:.](\d{3})",
            re.MULTILINE,
        )

        # 定义时间戳处理函数
        def normalize_timestamp(match):
            groups = match.groups()
            # 确定匹配了哪种模式
            if groups[0] is not None:  # 匹配了 M:SS,mmm
                return f"00:0{groups[0]}:{groups[1]},{groups[2]}"
            elif groups[3] is not None:  # 匹配了 MM:SS,mmm
                return f"00:{groups[3]}:{groups[4]},{groups[5]}"
            elif groups[6] is not None:  # 匹配了 HH:MM:SS,mmm
                return f"00:{groups[7]}:{groups[8]},{groups[9]}"
            elif groups[10] is not None:  # 匹配了超大小时值
                return f"00:{groups[11]}:{groups[12]},{groups[13]}"
            return match.group(0)  # 如果不匹配预期格式，返回原始文本

        # 一次性处理所有时间戳
        output = timestamp_pattern.sub(normalize_timestamp, output)

    elif Path(filepath).suffix in BASE_APPLICATION_EXTENSIONS or args.ocr_model != "none":
        pass
    else:
        try:
            json_from = json.loads(output)
            return json_from
        except json.JSONDecodeError:
            pass
        if "###" in output:
            shortdescription, long_description = process_llm_response(output)
            if args.mode == "all":
                output = [shortdescription, long_description]
            elif args.mode == "long":
                output = long_description
            elif args.mode == "short":
                output = shortdescription

    return output


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_dir", type=str, help="directory for dataset")

    parser.add_argument(
        "--pair_dir",
        type=str,
        default="",
        help="directory for pair dataset",
    )

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
        "--qwenVL_api_key",
        type=str,
        default="",
        help="API key for qwenVL API",
    )

    parser.add_argument(
        "--qwenVL_model_path",
        type=str,
        default="qwen-vl-max-latest",
        help="video model for qwenVL",
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
        "--vlm_image_model",
        type=str,
        choices=["moondream", "qwen_vl_local", ""],
        default="",
        help="VLM model for image tasks (default: empty)",
    )

    parser.add_argument(
        "--glm_api_key",
        type=str,
        default="",
        help="API key for glm API",
    )

    parser.add_argument(
        "--glm_model_path",
        type=str,
        default="glm-4v-plus-0111",
        help="Model path for glm",
    )

    # Ark (Volcano Engine) options
    parser.add_argument(
        "--ark_api_key",
        type=str,
        default="",
        help="API key for Ark (Volcano Engine) API",
    )
    parser.add_argument(
        "--ark_model_path",
        type=str,
        default="",
        help="Model ID for Ark chat.completions (e.g. your EP model id)",
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
        default=600,
        help="Segment time",
    )

    parser.add_argument(
        "--ocr_model",
        type=str,
        choices=["pixtral_ocr", "deepseek_ocr", "olmocr", "paddle_ocr", "moondream", ""],
        default="",
        help="OCR model to use for text extraction (default: empty)",
    )

    parser.add_argument(
        "--document_image",
        action="store_true",
        help="Use OCR to extract image from document",
    )

    parser.add_argument(
        "--scene_detector",
        type=str,
        choices=[
            "ContentDetector",
            "AdaptiveDetector",
            "HashDetector",
            "HistogramDetector",
            "ThresholdDetector",
        ],
        default="AdaptiveDetector",
        help="Detector to use for scene detection",
    )

    parser.add_argument(
        "--scene_threshold",
        type=float,
        default=0.0,
        help="Threshold for scene detection",
    )

    parser.add_argument(
        "--scene_min_len",
        type=int,
        default=15,
        help="Minimum length(frames) for scene detection",
    )

    parser.add_argument(
        "--scene_luma_only",
        action="store_true",
        help="Only use luma (brightness) without color changes for scene detection.",
    )

    parser.add_argument(
        "--gemini_task",
        type=str,
        default="",
        help="Task for gemini-2.0-flash-exp",
    )

    parser.add_argument(
        "--tags_highlightrate",
        type=float,
        default=0.4,
        help="tags_highlightrate for check captions",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()

    config = toml.load(args.config)

    process_batch(args, config)
