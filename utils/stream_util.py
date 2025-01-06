import av
from rich.progress import Progress, BarColumn, TimeRemainingColumn
from rich.console import Console
from av.audio.format import AudioFormat
from av.audio.layout import AudioLayout
import subprocess
import imageio_ffmpeg

console = Console()


def split_media_stream_clips(uri, media_type, subs, save_caption_func=None):
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
                if len(subs) < 2:
                    break
                clip_path = (
                    uri.parent / f"{uri.stem}_clip/{uri.stem}_{sub.index}{uri.suffix}"
                )
                clip_path.parent.mkdir(parents=True, exist_ok=True)

                with av.open(str(clip_path), mode="w") as out_container:
                    # copy encoder settings
                    if video_stream:
                        out_video_stream = out_container.add_stream_from_template(
                            template=video_stream
                        )
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
                            out_audio_stream.layout = AudioLayout(
                                "mono"
                            )  # AAC通常使用立体声
                            out_audio_stream.format = AudioFormat(
                                "fltp"
                            )  # AAC使用浮点平面格式
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
                    start_seconds = (
                        sub.start.hours * 3600
                        + sub.start.minutes * 60
                        + sub.start.seconds
                    )
                    end_seconds = (
                        sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds
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
                        stream=(video_stream if video_stream else audio_stream),
                    )

                    # 手动跳过帧 (如果在 seek 之后需要的话)
                    for frame in in_container.decode(video_stream, audio_stream):
                        if frame.time > end_seconds:
                            break

                        if (
                            video_stream
                            and isinstance(frame, av.VideoFrame)
                            and frame.time >= start_seconds
                        ):
                            for packet in out_video_stream.encode(frame):
                                out_container.mux(packet)
                        elif (
                            audio_stream
                            and isinstance(frame, av.AudioFrame)
                            and frame.time >= start_seconds
                        ):
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


def split_video_with_imageio_ffmpeg(uri, subs, save_caption_func=None):
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
            if len(subs) < 2:
                break
            clip_path = (
                uri.parent / f"{uri.stem}_clip/{uri.stem}_{sub.index}{uri.suffix}"
            )
            clip_path.parent.mkdir(parents=True, exist_ok=True)

            # 计算开始和结束时间
            start_time = f"{int(sub.start.hours):02d}:{int(sub.start.minutes):02d}:{int(sub.start.seconds):02d}.{int(sub.start.milliseconds):03d}"
            end_time = f"{int(sub.end.hours):02d}:{int(sub.end.minutes):02d}:{int(sub.end.seconds):02d}.{int(sub.end.milliseconds):03d}"

            if sub.end.minutes - sub.start.minutes == 5:
                # 使用segment模式时的输出模板
                output_template = str(uri.parent / f"{uri.stem}_clip/{uri.stem}_%03d{uri.suffix}")
                command = [
                    ffmpeg_exe,
                    '-i', str(uri),                 # 输入文件
                    '-f', 'segment',                # 使用segment模式
                    '-c', 'copy',                   # 拷贝原始编码，速度更快
                    '-segment_time', '300',         # 指定片段时长（5分钟）
                    '-reset_timestamps', '1',        # 重置时间戳
                    '-y',                           # 覆盖输出文件
                    output_template                 # 输出文件模板
                ]
            else:
                # 根据是否是第一个片段来调整命令
                if i == 0:
                    # 第一个片段，-ss 放在 -i 前面以获得更精确的开始时间
                    command = [
                        ffmpeg_exe,
                        '-ss', start_time,              # 开始时间
                        '-i', str(uri),                 # 输入文件
                        '-to', end_time,                # 结束时间
                        '-c', 'copy',                   # 拷贝原始编码，速度更快
                        '-map', '0',                    # 复制所有流
                        '-force_key_frames', '0',       # 强制第一帧为关键帧
                        '-y',                           # 覆盖输出文件
                        str(clip_path)                  # 输出文件
                    ]
                else:
                    # 其他片段，-i 放在前面以确保片段连接
                    command = [
                        ffmpeg_exe,
                        '-i', str(uri),                 # 输入文件
                        '-ss', start_time,              # 开始时间
                        '-to', end_time,                # 结束时间
                        '-c', 'copy',                   # 拷贝原始编码，速度更快
                        '-map', '0',                    # 复制所有流
                        '-force_key_frames', '0',       # 强制第一帧为关键帧
                        '-y',                           # 覆盖输出文件
                        str(clip_path)                  # 输出文件
                    ]

            console.print(f"Running command: {' '.join(command)}")
            try:
                # 使用 subprocess.PIPE 并设置 encoding='utf-8'
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
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

            if sub.end.minutes - sub.start.minutes == 5:
                sub_progress.advance(sub_task, advance=4)
                break
