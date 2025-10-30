import math
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import imageio_ffmpeg
import toml
from rich.console import Console
from rich.progress import BarColumn, Progress, TimeRemainingColumn

console = Console()


def split_media_stream_clips(uri, media_type, subs, save_caption_func=None, **kwargs):
    """
    Process media stream and extract clips based on subtitles.

    Args:
        uri (str): Path to the media file
        media_type (str): Type of media ('video' or 'audio')
        subs (pysrt.SubRipFile): Subtitles to process
        save_caption_func (callable): Function to save captions

    Returns:
        None
    """
    import av
    from av.audio.format import AudioFormat
    from av.audio.layout import AudioLayout

    with av.open(uri) as in_container:
        if media_type != "video":
            video_stream = None
        else:
            video_stream = next(
                (s for s in in_container.streams if s.type == "video"),
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

            for i, sub in enumerate(subs):
                # if len(subs) < 2:
                #     break
                clip_path = uri.parent / f"{uri.stem}_clip/{uri.stem}_{sub.index}{uri.suffix}"
                clip_path.parent.mkdir(parents=True, exist_ok=True)

                with av.open(str(clip_path), mode="w") as out_container:
                    # copy encoder settings
                    if video_stream:
                        out_video_stream = out_container.add_stream_from_template(template=video_stream)
                    else:
                        out_video_stream = None
                    if audio_stream:
                        # 为音频流使用特定的设置
                        if media_type == "video":
                            codec_name = "aac"
                            out_audio_stream = out_container.add_stream(
                                codec_name=codec_name,
                                rate=48000,  # AAC标准采样率
                            )
                            out_audio_stream.layout = AudioLayout("mono")  # AAC通常使用立体声
                            out_audio_stream.format = AudioFormat("fltp")  # AAC使用浮点平面格式
                        elif uri.suffix == ".mp3":
                            codec_name = "mp3"
                            out_audio_stream = out_container.add_stream(
                                codec_name=codec_name,
                                rate=16000,
                            )
                            out_audio_stream.layout = AudioLayout("mono")
                            out_audio_stream.format = AudioFormat("s16p")
                        else:
                            codec_name = "pcm_s16le"
                            out_audio_stream = out_container.add_stream(
                                codec_name=codec_name,
                                rate=16000,
                            )
                            out_audio_stream.layout = AudioLayout("mono")
                            out_audio_stream.format = AudioFormat("s16p")
                    else:
                        out_audio_stream = None

                    # 正确计算 start 和 end 时间戳, 单位是 video_stream.time_base
                    # 使用毫秒并根据 video_stream.time_base 转换
                    start_seconds = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds
                    end_seconds = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds
                    if video_stream:
                        start_offset = int(
                            start_seconds * video_stream.time_base.denominator / video_stream.time_base.numerator
                        )  # 开始时间戳偏移量 (基于 video_stream.time_base)
                    else:
                        start_offset = int(
                            start_seconds * audio_stream.time_base.denominator / audio_stream.time_base.numerator
                        )  # 开始时间戳偏移量 (基于 audio_stream.time_base)
                    # seek to start
                    in_container.seek(
                        start_offset,
                        stream=(video_stream if video_stream else audio_stream),
                    )

                    # 手动跳过帧 (如果在 seek 之后需要的话)
                    for frame in in_container.decode(video_stream, audio_stream):
                        if frame.time > end_seconds:
                            break

                        if video_stream and isinstance(frame, av.VideoFrame) and frame.time >= start_seconds:
                            for packet in out_video_stream.encode(frame):
                                out_container.mux(packet)
                        elif audio_stream and isinstance(frame, av.AudioFrame) and frame.time >= start_seconds:
                            for packet in out_audio_stream.encode(frame):
                                out_container.mux(packet)

                    # Flush streams
                    if out_video_stream:
                        for packet in out_video_stream.encode():
                            out_container.mux(packet)
                    if out_audio_stream:
                        for packet in out_audio_stream.encode():
                            out_container.mux(packet)

                if save_caption_func:
                    save_caption_func(clip_path, [sub.text], "image")
                sub_progress.advance(sub_task)


def split_video_with_imageio_ffmpeg(uri, subs, save_caption_func=None, segment_time=120, **kwargs):
    """
    Process media stream and extract clips based on subtitles using ffmpeg.

    Args:
        uri (Path): Path to the media file
        subs (pysrt.SubRipFile): Subtitles to process
        save_caption_func (callable, optional): Function to save captions
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
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

        for i, sub in enumerate(subs):
            # if len(subs) < 2:
            #     break
            clip_path = uri.parent / f"{uri.stem}_clip/{uri.stem}_{sub.index}{uri.suffix}"
            clip_path.parent.mkdir(parents=True, exist_ok=True)

            # 计算开始和结束时间
            start_time = f"{int(sub.start.hours):02d}:{int(sub.start.minutes):02d}:{int(sub.start.seconds):02d}.{int(sub.start.milliseconds):03d}"
            duration = (
                (sub.end.hours - sub.start.hours) * 3600
                + (sub.end.minutes - sub.start.minutes) * 60
                + (sub.end.seconds - sub.start.seconds)
                + (sub.end.milliseconds - sub.start.milliseconds) / 1000
            )

            if duration == segment_time:
                # 使用segment模式时的输出模板
                output_template = str(uri.parent / f"{uri.stem}_clip/{uri.stem}_%03d{uri.suffix}")
                command = [
                    ffmpeg_exe,
                    "-i",
                    str(uri),  # 输入文件
                    "-f",
                    "segment",  # 使用segment模式
                    "-c",
                    "copy",  # 拷贝原始编码，速度更快
                    "-segment_time",
                    str(segment_time),  # 指定片段时长（5分钟）
                    "-reset_timestamps",
                    "1",  # 重置时间戳
                    "-y",  # 覆盖输出文件
                    "-break_non_keyframes",
                    "0",
                    output_template,  # 输出文件模板
                ]
            else:
                if uri.suffix == ".mp3":
                    audio_codec = "mp3"
                elif uri.suffix == ".wav":
                    audio_codec = "pcm_s16le"
                else:
                    audio_codec = "aac"
                # 根据是否是第一个片段来调整命令
                if i == 0:
                    # 第一个片段，-ss 放在 -i 前面以获得更精确的开始时间
                    command = [
                        ffmpeg_exe,
                        "-ss",
                        start_time,  # 开始时间
                        "-t",
                        str(duration),  # 持续时间
                        "-i",
                        str(uri),  # 输入文件
                        "-c:v",
                        "libx264",  # 重新编码视频流
                        "-c:a",
                        audio_codec,  # 重新编码音频流
                        "-vf",
                        "setpts=PTS-STARTPTS",  # 重置视频时间戳
                        "-af",
                        "asetpts=PTS-STARTPTS",  # 重置音频时间戳
                        "-y",  # 覆盖输出文件
                        str(clip_path),  # 输出文件
                    ]
                else:
                    # 其他片段，-i 放在前面以确保片段连接
                    command = [
                        ffmpeg_exe,
                        "-i",
                        str(uri),  # 输入文件
                        "-ss",
                        start_time,  # 开始时间
                        "-t",
                        str(duration),  # 持续时间
                        "-c:v",
                        "libx264",  # 重新编码视频流
                        "-c:a",
                        audio_codec,  # 重新编码音频流
                        "-vf",
                        "setpts=PTS-STARTPTS",  # 重置视频时间戳
                        "-af",
                        "asetpts=PTS-STARTPTS",  # 重置音频时间戳
                        "-y",  # 覆盖输出文件
                        str(clip_path),  # 输出文件
                    ]

            console.print(f"Running command: {' '.join(command)}")
            try:
                # 使用 subprocess.PIPE 并设置 encoding='utf-8'
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                stdout, stderr = process.communicate()

                if process.returncode != 0:
                    console.print(f"[red]Error running ffmpeg:[/red] {stderr}")
                    raise Exception(f"FFmpeg failed: {stderr}")

            except Exception as e:
                console.print(f"[red]Failed to run ffmpeg:[/red] {str(e)}")
                raise

            if save_caption_func:
                save_caption_func(clip_path, [sub.text], "image")
            sub_progress.advance(sub_task)

            if sub.end.minutes - sub.start.minutes == segment_time / 60:
                sub_progress.advance(sub_task, advance=4)
                break


def sanitize_filename(name: str) -> str:
    """Sanitizes filenames.

    Requirements:
    - Only lowercase alphanumeric characters or dashes (-)
    - Cannot begin or end with a dash
    - Max length is 40 characters
    """
    # Convert to lowercase and replace non-alphanumeric chars with dash
    sanitized = re.sub(r"[^a-z0-9-]", "-", name.lower())
    # Replace multiple dashes with single dash
    sanitized = re.sub(r"-+", "-", sanitized)
    # Remove leading and trailing dashes
    sanitized = sanitized.strip("-")
    # If empty after sanitization, use a default name
    if not sanitized:
        sanitized = "file"
    # Ensure it starts and ends with alphanumeric character
    if sanitized[0] == "-":
        sanitized = "f" + sanitized
    if sanitized[-1] == "-":
        sanitized = sanitized + "f"
    # If length exceeds 40, keep the first 20 and last 19 chars with a dash in between
    if len(sanitized) > 40:
        # Take parts that don't end with dash
        first_part = sanitized[:20].rstrip("-")
        last_part = sanitized[-19:].rstrip("-")
        sanitized = first_part + "-" + last_part
    return sanitized


def get_video_duration(file_path):
    """
    获取视频片段的精确持续时间，用于字幕偏移计算

    Args:
        file_path: 视频文件路径

    Returns:
        float: 视频持续时间（毫秒）
    """
    from pymediainfo import MediaInfo

    for track in MediaInfo.parse(file_path).tracks:
        if track.track_type == "Video":
            return track.duration
        elif track.track_type == "Audio":
            return track.duration
    return 0


def calculate_dimensions(
    width: int,
    height: int,
    max_pixels: Optional[int] = None,
    max_long_edge: Optional[int] = 2048,
    max_short_edge: Optional[int] = None,
    img_scale_num: int = 16,
) -> Tuple[int, int]:
    """
    根据最大像素数、最大边长和缩放倍数限制，计算新的图像尺寸，同时保持长宽比。
    该逻辑与 OmniGen2 中的预处理方法一致。

    Args:
        width: 原始宽度。
        height: 原始高度。
        max_pixels: 允许的最大像素总数 (宽 * 高)。
        max_long_edge: 允许的图像长边的最大长度。
        max_short_edge: 允许的图像短边的最大长度。
        img_scale_num: 最终尺寸必须是此数值的倍数。

    Returns:
        一个包含新 (宽度, 高度) 的元组。
    """
    if max_pixels is None and max_long_edge is None:
        # 如果没有提供限制，则仅将尺寸调整为 img_scale_num 的倍数
        new_width = round(width / img_scale_num) * img_scale_num
        new_height = round(height / img_scale_num) * img_scale_num
        return int(new_width), int(new_height)

    # 1. 根据 max_long_edge 计算缩放因子
    scale_factor_side = 1.0
    if max_long_edge is not None:
        long_edge = max(width, height)
        if long_edge > max_long_edge:
            scale_factor_side = max_long_edge / long_edge

    if max_short_edge is not None:
        short_edge = min(width, height)
        if short_edge > max_short_edge:
            short_edge_scale_factor = max_short_edge / short_edge
            # Use the smaller scaling factor to satisfy both constraints
            scale_factor_side = min(scale_factor_side, short_edge_scale_factor)

    # 2. 根据 max_pixels 计算缩放因子
    scale_factor_pixels = 1.0
    if max_pixels is not None:
        current_pixels = width * height
        if current_pixels > max_pixels:
            scale_factor_pixels = math.sqrt(max_pixels / current_pixels)

    # 3. 确定最终的缩放因子（取限制性更强的那个）
    final_scale_factor = min(scale_factor_side, scale_factor_pixels)

    # 4. 计算新尺寸
    new_width = width * final_scale_factor
    new_height = height * final_scale_factor

    # 5. 调整尺寸使其为 img_scale_num 的倍数
    final_width = round(new_width / img_scale_num) * img_scale_num
    final_height = round(new_height / img_scale_num) * img_scale_num

    return int(final_width), int(final_height)


def split_name_series(names: str) -> str:
    """Split and format character names and series information.

    Args:
        names: String containing character names and series info

    Returns:
        Formatted string with character names and series
    """
    name_list = []

    items = [item.strip().replace("_", ":") for item in names.split(",")]

    # --- Config Loading ---
    # Load configuration from config.toml
    # This allows for dynamic configuration of the series exclusion list.
    CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.toml"
    SERIES_EXCLUDE_LIST = set()
    try:
        if CONFIG_PATH.exists():
            config = toml.load(CONFIG_PATH)
            SERIES_EXCLUDE_LIST = set(config.get("wdtagger", {}).get("series_exclude_list", []))
        else:
            console.print(f"[yellow]Config file not found at {CONFIG_PATH}, using default empty exclude list.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error loading config file: {e}, using default empty exclude list.[/red]")
    # --- End Config Loading ---

    for item in items:
        if item.endswith(" (cosplay)"):
            item = item.replace(" (cosplay)", "")
        if ("c.c_") or ("c.c") in item:
            item = item.replace("c.c_", "c.c.")
        if ("k:da") in item:
            item = item.replace("k:da", "k/da")
        if ("ranma 1:2") in item:
            item = item.replace("ranma 1:2", "ranma 1/2")
        # 匹配最后一对括号作为系列名
        match = re.match(r"(.*)\((.*?)\)$", item)
        if match and match.group(2).strip() not in SERIES_EXCLUDE_LIST:
            # 获取除最后一个括号外的所有内容作为名字
            full_name = match.group(1).strip()
            series = match.group(2).strip()

            # 保留名字中的其他括号
            name_list.append(f"<{full_name}> from ({series})")
        else:
            name_list.append(f"<{item}>")

    return ", ".join(name_list)


def format_description(text: str, tag_description: str = "") -> str:
    """Format description text with highlighting.

    Args:
        text: Input text to format
        tag_description: Tags to highlight in blue

    Returns:
        Formatted text with rich markup
    """
    from utils.wdtagger import TagClassifier
    # 高亮<>内的内容
    text = re.sub(r"<([^>]+)>", r"[magenta]\1[/magenta]", text)
    # 高亮()内的内容
    text = re.sub(r"\(([^)]+)\)", r"[dark_magenta]\1[/dark_magenta]", text)

    words = text.split()

    tagClassifier = TagClassifier()

    blue_words = set()

    # 高亮与tag_description匹配的单词
    if tag_description:
        # 将tag_description分割成单词列表
        tag_words = set(
            word.strip().lower()
            for word in re.sub(r"\d+", "", tag_description).replace(",", " ").replace(".", " ").split()
            if word.strip()
        )
        for i, word in enumerate(words):
            highlight_word = re.sub(r"[^\w\s]", "", word.replace("'s", "").lower())
            if highlight_word in tag_words:
                blue_words.add(highlight_word)
                words[i] = tagClassifier.get_colored_tag(word)
        text = " ".join(words)

    # 统计高亮的次数
    highlight_count = 0
    has_green = False
    has_purple = False
    for word in words:
        if word.startswith("[magenta]") and word.endswith("[/magenta]"):
            has_green = True
        if word.startswith("[dark_magenta]") and word.endswith("[/dark_magenta]"):
            has_purple = True

    highlight_count = len(blue_words) + int(has_green) + int(has_purple)

    # 打印高亮率
    colors = ["red", "orange", "yellow", "green", "cyan", "blue", "magenta"]
    rate = highlight_count / len(tag_description.replace(",", " ").split()) * 100
    # 将100%平均分配给7种颜色，每个颜色约14.3%
    color_index = min(int(rate / (100 / len(colors))), len(colors) - 1)
    color = colors[color_index]
    # 根据rate值决定是否使用粗体
    style = f"{color} bold" if rate > 50 else color
    highlight_rate = f"[{style}]{rate:.2f}%[/{style}]"

    return text, highlight_rate


def pdf_to_images_high_quality(pdf_path: str, dpi: int = 144, image_format: str = "PNG"):
    import io

    import fitz
    from PIL import Image

    images = []
    pdf_document = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        if image_format.upper() != "PNG":
            if img.mode in ("RGBA", "LA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                img = background
        images.append(img)
    pdf_document.close()
    return images


def pil_to_pdf_img2pdf(pil_images, output_path: str):
    import img2pdf
    import io

    if not pil_images:
        return
    image_bytes_list = []
    for img in pil_images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="JPEG", quality=95)
        img_bytes = img_buffer.getvalue()
        image_bytes_list.append(img_bytes)
    pdf_bytes = img2pdf.convert(image_bytes_list)
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)
