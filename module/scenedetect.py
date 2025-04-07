import pysrt
from pathlib import Path
from PIL import Image
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg, open_video
from scenedetect.scene_manager import write_scene_list_html, save_images
import asyncio
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor
from rich.console import Console
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
from config.config import BASE_VIDEO_EXTENSIONS


# 辅助函数：在线程中运行异步任务
def run_async_in_thread(coroutine):
    """在单独的线程中运行异步协程"""

    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(coroutine)
        finally:
            loop.close()

    thread = threading.Thread(target=run_in_thread)
    thread.daemon = True  # 使线程在主程序退出时自动结束
    thread.start()
    return thread


class SceneDetector:
    """
    视频场景检测器类，用于视频场景分割和时间戳提取
    """

    def __init__(self, threshold=3, min_scene_len=1, console=None):
        """
        初始化场景检测器

        参数:
            threshold (int): 场景变化检测的阈值，数值越低越敏感
            min_scene_len (int): 场景变化检测的最小场景长度，数值越小越敏感
        """
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.scene_list = []
        self._detection_complete = False
        self.__init_async_attrs()
        self.console = console or Console()

    async def detect_scenes_async(self, video_path):
        """
        异步检测视频中的场景变化

        参数:
            video_path (str): 视频文件路径

        返回:
            list: 场景列表
        """
        # 创建异步任务，检测场景
        scene_list = await asyncio.to_thread(self.detect_scenes, video_path)
        self._detection_complete = True
        self.scene_list = scene_list
        return scene_list

    def detect_scenes(self, video_path):
        """
        同步检测视频中的场景

        参数:
            video_path (str): 视频文件路径

        返回:
            list: 场景列表，每个场景包含开始和结束时间
        """
        try:
            detector = AdaptiveDetector(
                adaptive_threshold=self.threshold,
                min_scene_len=self.min_scene_len,
            )
            # 使用PySceneDetect的detect接口检测场景
            scene_list = detect(video_path, detector=detector, show_progress=True)

            return scene_list
        except Exception as e:
            self.console.print(f"[red]Scene detection failed: {str(e)}[/red]")
            return []

    def start_async_detection(self, video_path):
        """
        开始异步场景检测，返回协程对象

        Args:
            video_path (str): 视频文件路径

        Returns:
            coroutine: 异步场景检测的协程对象
        """
        # 不使用create_task，直接返回协程对象
        coroutine = self.detect_scenes_async(video_path)
        self._init_task = coroutine
        return self._init_task

    async def ensure_detection_complete(self, video_path=None):
        """
        确保场景检测完成，并返回场景列表

        Args:
            video_path (str, optional): 视频路径，如果提供则启动检测

        Returns:
            list: 场景列表
        """
        # 将Path对象转换为字符串
        video_path_str = str(video_path) if video_path else None
        
        # 如果没有任务或已完成，则创建新任务
        if self._init_task is None and video_path_str:
            # 直接调用异步方法，不使用create_task
            self._init_task = self.detect_scenes_async(video_path_str)

        # 等待任务完成
        try:
            if self._init_task:
                self.scene_list = await self._init_task
                self._detection_complete = True
                return self.scene_list
            return []
        except Exception as e:
            self.console.print(f"[red]scene detection failed: {str(e)}[/red]")
            return []

    def get_timestamps(self, scene_list):
        """
        获取场景变化的时间戳（秒）

        Args:
            scene_list (list): 场景列表

        Returns:
            list: 时间戳列表（秒）
        """
        timestamps = []
        for scene in scene_list:
            # 获取场景开始时间（秒）
            start_time = scene[0].get_seconds()
            timestamps.append(start_time)

        return timestamps

    def get_srt(self, scene_list):
        """
        获取场景变化的SRT字幕

        Args:
            scene_list (list): 场景列表

        Returns:
            pysrt.SubRipFile: SRT字幕文件
        """
        srt = pysrt.SubRipFile()
        for i, (start, end) in enumerate(scene_list):
            subtitle = pysrt.SubRipItem(i, start.get_timecode(), end.get_timecode(), "")
            srt.append(subtitle)
        return srt

    def split_video(
        self,
        video_path,
        scene_list,
        output_dir=None,
        save_html=False,
        video2images_min_number=0,
    ):
        """
        将视频按场景分割为多个片段

        Args:
            video_path (str): 视频文件路径
            scene_list (list): 场景列表
            output_dir (str, optional): 输出目录，默认为None使用视频所在目录
            save_html (bool, optional): 是否保存HTML报告文件
            video2images_min_number (int, optional): 每个场景保存的图像数量，为0则不保存

        Returns:
            list: 分割后的视频文件路径列表
        """
        if output_dir is None:
            output_dir = Path(video_path).parent / Path(video_path).stem
        else:
            # 确保输出目录存在并且是个目录而不是文件
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            elif not output_dir.is_dir():
                # 如果是文件，则创建同名目录
                output_parent = output_dir.parent
                output_dir_name = output_dir.name
                output_dir = output_parent / f"{output_dir_name}_dir"
                output_dir.mkdir(parents=True, exist_ok=True)

        scene_image_dict = None
        image_height = None
        image_width = None
        if video2images_min_number > 0:
            try:
                self.console.print("[green]Saving scene images...[/green]")
                scene_image_dict = save_images(
                    scene_list=scene_list,
                    video=open_video(video_path),
                    output_dir=str(output_dir / "images"),
                    num_images=video2images_min_number,
                )
            except Exception as e:
                self.console.print(f"[red]can't save scene images: {str(e)}[/red]")

        if save_html:
            # create HTML report file name
            html_output = (
                output_dir / "images" / (output_dir.stem + ".html")
                if scene_image_dict
                else output_dir / (output_dir.stem + ".html")
            )
            # get first image size from scene_image_dict and calculate height and width
            if scene_image_dict and len(scene_image_dict) > 0:
                # 获取第一个场景的第一张图片
                self.console.print("[green]Getting save image...[/green]")
                first_scene_images = next(iter(scene_image_dict.values()), [])
                if first_scene_images and len(first_scene_images) > 0:
                    try:
                        first_image = Image.open(
                            output_dir / "images" / Path(first_scene_images[0]).name
                        )
                        orig_width, orig_height = first_image.size
                        # 按比例缩放，确保最大边不超过512
                        max_dimension = max(orig_width, orig_height)
                        if max_dimension > 512:
                            scale = 512 / max_dimension
                            image_width = int(orig_width * scale)
                            image_height = int(orig_height * scale)
                    except Exception as e:
                        self.console.print(f"[red]can't get image size: {str(e)}[/red]")

            try:
                self.console.print("[green]Generating HTML report...[/green]")
                write_scene_list_html(
                    str(html_output),
                    scene_list,
                    image_filenames=scene_image_dict,
                    image_height=image_height,
                    image_width=image_width,
                )
            except Exception as e:
                self.console.print(f"[red]can't save HTML report: {str(e)}[/red]")

        # 使用ffmpeg分割视频
        try:
            return split_video_ffmpeg(
                video_path,
                scene_list,
                output_dir=str(output_dir / "clips"),
                show_progress=True,
            )
        except Exception as e:
            self.console.print(f"[red]split video failed: {str(e)}[/red]")
            return []

    async def split_video_async(
        self,
        video_path,
        scene_list,
        output_dir=None,
        save_html=False,
        video2images_min_number=0,
    ):
        """
        Asynchronously split video into multiple segments based on scenes

        Args:
            video_path (str): video file path
            scene_list (list): scene list
            output_dir (str, optional): output directory. Defaults to None.
            save_html (bool, optional): whether to save HTML report file. Defaults to False.
            video2images_min_number (int, optional): number of images to save per scene, 0 means no saving. Defaults to 0.

        Returns:
            list: output video file list
        """
        # ensure scene detection is complete
        await self.ensure_detection_complete(video_path)

        # use thread pool to handle IO密集型任务
        return await asyncio.to_thread(
            self.split_video,
            video_path,
            scene_list,
            output_dir,
            save_html,
            video2images_min_number,
        )

    def __init_async_attrs(self):
        """初始化异步相关的属性"""
        self._init_complete = False
        self._init_task = None
        self._lock = threading.Lock()

    def align_subtitle(self, srt_path, scene_list, segment_time=600, console=None):
        """
        根据视频场景变化对字幕进行智能对齐，为1帧/秒转写的字幕优化

        Args:
            srt_path: 字幕文件路径
            scene_list: 场景列表
            segment_time: 分段时长（秒），默认60秒，主要用于分析，不严格按此分段

        Returns:
            subs: 对齐后的字幕对象
        """
        # 如果没有提供console对象，则导入Rich的Console
        if console is None:
            console = Console()

        if not scene_list:
            console.print(
                "[yellow]No scene changes detected, unable to align subtitles[/yellow]"
            )
            return srt_path

        # 加载字幕文件
        if isinstance(srt_path, str):
            subs = pysrt.open(srt_path)
        else:
            subs = srt_path

        # 检测视频场景并转换时间戳
        scene_timestamps_ms = [int(ts * 1000) for ts in self.get_timestamps(scene_list)]

        # 构建场景块（每个场景的开始和结束时间）
        scene_blocks = []
        for i in range(len(scene_timestamps_ms) - 1):
            scene_blocks.append((scene_timestamps_ms[i], scene_timestamps_ms[i + 1]))

        # 如果视频很长，可能最后还有一个场景没有结束点，我们给它一个结束点
        if scene_blocks and scene_timestamps_ms[-1] > scene_blocks[-1][1]:
            # 添加最后一个场景，结束时间设为最后一个场景开始时间加上平均场景长度
            if len(scene_blocks) > 0:
                avg_scene_duration = sum(
                    end - start for start, end in scene_blocks
                ) / len(scene_blocks)
                scene_blocks.append(
                    (
                        scene_timestamps_ms[-1],
                        scene_timestamps_ms[-1] + int(avg_scene_duration),
                    )
                )

        console.print(f"[green]Detected {len(scene_blocks)} scene blocks[/green]")

        # 初始化当前场景块索引
        current_scene_index = 0

        # 记录前一个字幕的结束时间（调整后）
        previous_end_time = None

        # 预先标记所有分段的位置
        segment_boundaries = []
        last_segment = -1
        for i, sub in enumerate(subs):
            start_time_ms = sub.start.ordinal
            segment = int((start_time_ms / 1000) // segment_time)
            if segment > last_segment:
                segment_boundaries.append((i, segment))
                last_segment = segment

        if segment_boundaries:
            console.print(
                f"[green]Detected {len(segment_boundaries)} segment boundaries: {', '.join([f'Sub#{b[0]+1}(Segment{b[1]})' for b in segment_boundaries])}[/green]"
            )

        # 顺序处理每个字幕，保持连贯性
        console.print(
            f"[blue]Starting sequential subtitle processing, maintaining coherence...[/blue]"
        )

        # 定义误差阈值（毫秒）
        error_threshold_ms = 1000  # 1秒以上被视为有明显误差
        # 记录需要重新计算的字幕
        recalculate_subs = []

        for i, sub in enumerate(subs):
            # 获取当前字幕的时间范围
            start_time_ms = sub.start.ordinal  # 已经是毫秒
            end_time_ms = sub.end.ordinal  # 已经是毫秒
            subtitle_duration = end_time_ms - start_time_ms

            # 计算当前字幕所在的分段
            current_segment = int((start_time_ms / 1000) // segment_time)

            # 检查当前字幕是否是某个分段的第一个字幕
            is_segment_boundary = False
            for boundary_index, boundary_segment in segment_boundaries:
                if i == boundary_index:
                    is_segment_boundary = True
                    break

            # 调试输出
            if i < 3 or i % 40 == 0 or is_segment_boundary:
                console.print(
                    f"[blue]Subtitle #{i+1}: Start={start_time_ms/1000:.1f}s (Segment{current_segment})[/blue]"
                )

            # 检查当前字幕是否在分段边界上
            reset_coherence = False
            if previous_end_time is not None and is_segment_boundary:
                console.print(
                    f"[cyan]Subtitle #{i+1}/{len(subs)}: Processing segment boundary {current_segment}, resetting coherence[/cyan]"
                )
                reset_coherence = True

            # 基于连贯性检测结果决定是否调整开始时间
            if previous_end_time is not None and not reset_coherence:
                # 在同一分段内，保持连贯
                # 确保字幕之间有一定的间隔
                new_start_time = previous_end_time

                # 计算开始时间的调整量
                start_offset = new_start_time - start_time_ms

                # 只有当偏移量大于100毫秒时才调整开始时间
                if abs(start_offset) > 100:
                    # 应用到当前字幕
                    seconds_start = start_offset // 1000
                    milliseconds_start = start_offset % 1000

                    # 调整字幕开始时间
                    sub.start.shift(
                        seconds=seconds_start, milliseconds=milliseconds_start
                    )

                    # 如果偏移足够大则打印
                    if abs(seconds_start) >= 1 or abs(milliseconds_start) >= 500:
                        console.print(
                            f"[cyan]Subtitle #{i+1}/{len(subs)}: Adjusted start time by {seconds_start}s {milliseconds_start}ms (connecting to previous subtitle)[/cyan]"
                        )

            # 更新开始时间变量以备后续使用
            start_time_ms = sub.start.ordinal

            # 找出适合这个字幕的场景块
            # 从当前场景索引开始，避免重复使用已处理的场景
            relevant_blocks = []
            for j in range(current_scene_index, len(scene_blocks)):
                block_start, block_end = scene_blocks[j]

                # 判断场景块是否与字幕结束时间相关（考虑±2秒偏移）
                if (end_time_ms - 2000 <= block_end) and (
                    end_time_ms + 2000 >= block_start
                ):
                    relevant_blocks.append((j, block_start, block_end))

            # 如果没有找到相关场景块，使用下一个未使用的场景块
            if not relevant_blocks and current_scene_index < len(scene_blocks):
                j = current_scene_index
                block_start, block_end = scene_blocks[j]
                relevant_blocks.append((j, block_start, block_end))

            # 如果仍然没有相关场景块，跳过调整
            if not relevant_blocks:
                previous_end_time = end_time_ms  # 保持当前结束时间不变
                continue

            # 简化处理：找到字幕结束时间所在的场景块或最近的场景块
            best_scene_idx = relevant_blocks[0][0]
            best_end_time = end_time_ms  # 默认不调整

            for scene_idx, scene_start, scene_end in relevant_blocks:
                # 检查字幕结束时间是否在场景块内
                if scene_start <= end_time_ms <= scene_end:
                    # 找到了字幕结束时间所在的场景块
                    best_scene_idx = scene_idx

                    # 判断结束时间离场景块的开始还是结束更近
                    if abs(end_time_ms - scene_start) < abs(end_time_ms - scene_end):
                        # 离场景开始更近，向场景开始偏移
                        best_end_time = scene_start
                        # 如果调整超过误差阈值，记录下来以供后续分析
                        if abs(best_end_time - end_time_ms) > error_threshold_ms:
                            recalculate_subs.append(
                                (i, "start", end_time_ms, best_end_time)
                            )
                            console.print(
                                f"[yellow]Warning: Subtitle #{i+1}/{len(subs)} adjustment to scene start exceeds {error_threshold_ms//1000}s, marked as abnormal[/yellow]"
                            )
                    else:
                        # 离场景结束更近，向场景结束偏移
                        best_end_time = scene_end
                        # 如果调整超过误差阈值，记录下来以供后续分析
                        if abs(best_end_time - end_time_ms) > error_threshold_ms:
                            recalculate_subs.append(
                                (i, "end", end_time_ms, best_end_time)
                            )
                            console.print(
                                f"[yellow]Warning: Subtitle #{i+1}/{len(subs)} adjustment to scene end exceeds {error_threshold_ms//1000}s, marked as abnormal[/yellow]"
                            )

                    # 找到匹配场景块后跳出循环
                    break
                elif scene_idx > best_scene_idx:
                    # 如果没有找到正好匹配的场景块，使用索引较大的场景块
                    best_scene_idx = scene_idx

            # 如果没找到包含结束时间的场景块，选择最近的场景边界
            if best_end_time == end_time_ms and relevant_blocks:
                # 找出距离结束时间最近的场景边界点
                closest_boundary = None
                min_distance = float("inf")

                for _, scene_start, scene_end in relevant_blocks:
                    # 计算结束时间到场景开始的距离
                    start_distance = abs(end_time_ms - scene_start)
                    if start_distance < min_distance:
                        min_distance = start_distance
                        closest_boundary = scene_start

                    # 计算结束时间到场景结束的距离
                    end_distance = abs(end_time_ms - scene_end)
                    if end_distance < min_distance:
                        min_distance = end_distance
                        closest_boundary = scene_end

                if closest_boundary is not None:
                    best_end_time = closest_boundary
                    # 如果调整超过误差阈值，记录下来以供后续分析
                    if abs(best_end_time - end_time_ms) > error_threshold_ms:
                        recalculate_subs.append(
                            (i, "boundary", end_time_ms, best_end_time)
                        )
                        console.print(
                            f"[yellow]Warning: Subtitle #{i+1}/{len(subs)} adjustment to nearest boundary exceeds {error_threshold_ms//1000}s, marked as abnormal[/yellow]"
                        )

            # 计算偏移量
            end_offset = best_end_time - end_time_ms

            # 如果偏移量超过误差阈值，尝试智能修正
            if abs(end_offset) > error_threshold_ms:
                console.print(
                    f"[cyan]Subtitle #{i+1}/{len(subs)}: Attempting intelligent correction for large offset of {end_offset//1000}s[/cyan]"
                )

                # 策略1: 检查相邻场景块，尝试找到更合适的匹配
                better_match_found = False

                # 扩大搜索范围，查看更多场景块
                extended_blocks = []
                search_range = min(5, len(scene_blocks))  # 最多查看前后5个场景块
                start_idx = max(0, best_scene_idx - search_range)
                end_idx = min(len(scene_blocks), best_scene_idx + search_range)

                for j in range(start_idx, end_idx):
                    if j not in [
                        rb[0] for rb in relevant_blocks
                    ]:  # 避免重复检查已考虑的场景块
                        block_start, block_end = scene_blocks[j]
                        extended_blocks.append((j, block_start, block_end))

                # 在扩展的场景块中找最近的边界
                if extended_blocks:
                    closest_ext_boundary = None
                    min_ext_distance = abs(end_offset)  # 使用当前偏移作为基准

                    for j, ext_start, ext_end in extended_blocks:
                        # 检查场景开始边界
                        start_distance = abs(end_time_ms - ext_start)
                        if start_distance < min_ext_distance:
                            min_ext_distance = start_distance
                            closest_ext_boundary = ext_start

                        # 检查场景结束边界
                        end_distance = abs(end_time_ms - ext_end)
                        if end_distance < min_ext_distance:
                            min_ext_distance = end_distance
                            closest_ext_boundary = ext_end

                    # 如果找到更接近的边界，使用它
                    if closest_ext_boundary is not None and abs(
                        end_time_ms - closest_ext_boundary
                    ) < abs(end_offset):
                        console.print(
                            f"[green]Subtitle #{i+1}/{len(subs)}: Found closer scene boundary, reducing offset from {end_offset//1000}s to {(closest_ext_boundary-end_time_ms)//1000}s[/green]"
                        )
                        best_end_time = closest_ext_boundary
                        end_offset = best_end_time - end_time_ms
                        better_match_found = True

                # 策略2: 如果仍未找到更好的匹配，使用加权平均值
                if not better_match_found:
                    # 使用原始结束时间和检测到的场景边界的加权平均
                    # 当偏移很大时，更倾向于保留原始时间，但仍然向场景边界靠拢
                    weight_original = 0.7  # 原始时间的权重
                    weight_scene = 0.3  # 场景边界的权重

                    weighted_end_time = int(
                        end_time_ms * weight_original + best_end_time * weight_scene
                    )

                    console.print(
                        f"[green]Subtitle #{i+1}/{len(subs)}: Using weighted average time, reducing offset from {end_offset//1000}s to {(weighted_end_time-end_time_ms)//1000}s[/green]"
                    )
                    best_end_time = weighted_end_time
                    end_offset = best_end_time - end_time_ms

            seconds_end = end_offset // 1000
            milliseconds_end = end_offset % 1000

            # 调整字幕结束时间
            sub.end.shift(seconds=seconds_end, milliseconds=milliseconds_end)

            # 更新为下一个字幕处理保存的变量
            previous_end_time = sub.end.ordinal
            current_scene_index = best_scene_idx + 1  # 下一个字幕从下一个场景开始

            # 只打印偏移量绝对值较大的调整
            if abs(seconds_end) >= 1 or abs(milliseconds_end) >= 500:
                console.print(
                    f"[blue]Subtitle #{i+1}/{len(subs)}: Adjusted end time by {seconds_end}s {milliseconds_end}ms (aligned to scene)[/blue]"
                )

        return subs


    def is_detection_complete(self):
        return self._detection_complete

    def get_scene_list(self):
        return self.scene_list

def setup_parser() -> argparse.ArgumentParser:
    """Setup argument parser."""
    parser = argparse.ArgumentParser(description="split video")
    parser.add_argument("input_video_dir", type=str, help="Input video directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--threshold",
        type=float,
        default=3,
        help="Threshold (float) that score ratio must exceed to trigger a new scene.",
    )
    parser.add_argument(
        "--min_scene_len",
        type=int,
        default=15,
        help="Once a cut is detected, this many frames must pass before a new one can be added to the scene list.",
    )
    parser.add_argument(
        "--save_html",
        action="store_true",
        help="Writes the given list of scenes to an output file handle in html format.",
    )
    parser.add_argument(
        "--video2images_min_number",
        type=int,
        default=0,
        help="Number of images to generate for each scene. Minimum is 1.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for videos in subdirectories.",
    )

    return parser


if __name__ == "__main__":

    # 创建Rich控制台对象
    console = Console()

    parser = setup_parser()
    args = parser.parse_args()

    # 创建场景检测器实例
    scene_detector = SceneDetector(args.threshold, args.min_scene_len, console=console)

    # 获取所有视频文件
    video_files = []
    input_path = Path(args.input_video_dir)
    if args.recursive:
        for ext in BASE_VIDEO_EXTENSIONS:
            video_files.extend(list(input_path.rglob(f"*{ext}")))
    else:
        for ext in BASE_VIDEO_EXTENSIONS:
            video_files.extend(list(input_path.glob(f"*{ext}")))

    # remove duplicates (Windows is case-insensitive)
    video_files = list(set(video_files))

    if not video_files:
        console.print(
            f"[yellow]No video files found, supported extensions: {', '.join(BASE_VIDEO_EXTENSIONS)}[/yellow]"
        )

    # 打印找到的每个视频文件路径
    console.print(f"[green]Found {len(video_files)} video files[/green]")
    for i, video_file in enumerate(video_files):
        console.print(f"[magenta]  {i+1}. {video_file}[/magenta]")

    # Create Rich progress bar
    progress = Progress(
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
        auto_refresh=False,
    )

    async def process_video(video_path):
        """Process a single video file"""
        video_name = video_path.stem
        with progress:
            # Create scene detection task
            task_id = progress.add_task(
                f"Processing: {video_name}",
                total=2,  # 2 steps: 1. detect scenes 2. split video
                status="Detecting scenes...",
            )

            # detect scenes
            try:
                scene_list = await scene_detector.detect_scenes_async(str(video_path))
                progress.update(task_id, advance=1, status="Splitting video...")

                # split video
                if scene_list and len(scene_list) > 1:
                    # use thread pool to execute IO task
                    with ThreadPoolExecutor() as executor:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            executor,
                            scene_detector.split_video,
                            str(video_path),
                            scene_list,
                            args.output_dir,
                            args.save_html,
                            args.video2images_min_number,
                        )
                    progress.update(task_id, advance=1, status="Completed")
                else:
                    progress.update(
                        task_id, advance=1, status="No need to split (too few scenes)"
                    )
            except Exception as e:
                progress.update(task_id, status=f"Error: {str(e)}")
                raise e

    async def main():
        """Main async function"""
        # parallel process all videos
        tasks = [process_video(video_path) for video_path in video_files]
        await asyncio.gather(*tasks)
        console.print("[bold green]All videos processed successfully[/bold green]")

    # run main async function
    asyncio.run(main())
