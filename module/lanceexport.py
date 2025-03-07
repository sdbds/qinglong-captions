import argparse
import lance
import re
from typing import Optional, Union, List, Dict, Any
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
from config.config import get_supported_extensions, DATASET_SCHEMA, CONSOLE_COLORS
from utils.stream_util import split_media_stream_clips, split_video_with_imageio_ffmpeg
from pathlib import Path
import pysrt

console = Console()
image_extensions = get_supported_extensions("image")
animation_extensions = get_supported_extensions("animation")
video_extensions = get_supported_extensions("video")
audio_extensions = get_supported_extensions("audio")
application_extensions = get_supported_extensions("application")


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

        elif media_type == "application":
            meta_info.extend(
                [
                    f"{metadata.get('size', 0) / (1024 * 1024):.2f} MB",
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

        if media_type == "audio" or media_type == "video":
            caption_path = caption_path.with_suffix(".srt")
        elif media_type == "application":
            caption_path = caption_path.with_suffix(".md")
        else:
            caption_path = caption_path.with_suffix(".txt")

        with open(caption_path, "w", encoding="utf-8") as f:
            if caption_path.suffix == ".srt":
                # For SRT files, preserve all lines including empty ones
                f.write("\n".join(caption_lines))
            elif caption_path.suffix == ".md":
                # For MD files, preserve original markdown formatting
                f.write("".join(caption_lines))
            else:
                # For TXT files, strip empty lines and whitespace
                for line in caption_lines:
                    if "<font color=" in line:
                        line = line.replace('<font color="green">', "").replace(
                            "</font>", ""
                        )
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


def save_caption_by_pages(caption_path: Path, caption_lines: List[str]) -> bool:
    """将多页文档分割为单独的页面并分别保存

    Args:
        caption_path: 保存路径
        caption_lines: 包含多页内容的文本列表

    Returns:
        bool: 成功返回True，失败返回False
    """
    try:
        # 合并文本行为单个字符串
        if len(caption_lines) == 1:
            # 如果只有一个元素（来自.md或.srt文件），直接使用该元素
            combined_text = caption_lines[0]
        else:
            # 如果是多行（来自.txt文件），用换行符连接
            combined_text = "\n".join(caption_lines)

        # 使用页眉作为分隔符来分割多个页面
        header_pattern = (
            r'(?s)<header style="background-color: #f5f5f5;.*?<strong> Page (\d+) </strong>'
        )
        footer_pattern = (
            r'(?s)<footer\s+style="[^"]*">.*?<strong>\s*Page\s+(\d+)\s*</strong>.*?</footer>'
        )
        page_break_pattern = r'<div style="page-break-after: always;"></div>'

        # 分割所有页面
        page_contents = []
        page_numbers = []

        # 查找所有页头位置
        header_matches = list(re.finditer(header_pattern, combined_text))
        footer_matches = list(re.finditer(footer_pattern, combined_text))

        # 如果没有找到页头，整体保存
        if not header_matches:
            # 没有找到页头，尝试其他方式分割内容
            # 尝试使用Markdown标题作为分割点
            md_header_pattern = r'^#{1,6}\s+(.+?)$'
            md_headers = list(re.finditer(md_header_pattern, combined_text, re.MULTILINE))

            if md_headers:
                # 使用Markdown标题分割内容
                console.print(
                    f"[yellow]No HTML headers found, splitting by Markdown headers.[/yellow]"
                )

                for i in range(len(md_headers)):
                    header_match = md_headers[i]
                    header_text = header_match.group(1).strip()

                    # 计算当前部分内容的开始位置
                    start_pos = header_match.start()

                    # 计算当前部分内容的结束位置
                    if i < len(md_headers) - 1:
                        end_pos = md_headers[i + 1].start()
                    else:
                        end_pos = len(combined_text)

                    # 提取部分内容
                    section_content = combined_text[start_pos:end_pos]

                    # 创建文件名 (使用标题的前20个字符，去除特殊字符)
                    safe_header = re.sub(r'[^\w\s-]', '', header_text)[:20].strip()
                    safe_header = re.sub(r'[-\s]+', '_', safe_header)

                    section_filename = f"{caption_path.stem}_{safe_header}{caption_path.suffix}"
                    section_file_path = caption_path.with_suffix("") / section_filename

                    # 保存部分内容
                    section_file_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(section_file_path, "w", encoding="utf-8") as f:
                        f.write(section_content)

                    console.print(
                        f"[{CONSOLE_COLORS['text']}]text: {section_file_path} saved successfully.[/{CONSOLE_COLORS['text']}]"
                    )

                return True
            else:
                # 如果没有任何分割点，保存为单个文件
                output_dir = caption_path.with_suffix(".md")
                output_dir.mkdir(parents=True, exist_ok=True)
                single_file_path = output_dir / f"{caption_path.stem}.md"

                with open(single_file_path, "w", encoding="utf-8") as f:
                    f.write(combined_text)
                console.print(
                    f"[{CONSOLE_COLORS['text']}]text: {single_file_path} saved successfully.[/{CONSOLE_COLORS['text']}]"
                )
                return True
        # 分割每个页面的内容
        for i in range(len(header_matches)):
            header_match = header_matches[i]
            page_number = int(header_match.group(1))
            page_numbers.append(page_number)

            # 计算当前页面内容的开始位置（从页头开始）
            start_pos = header_match.start()

            # 计算当前页面内容的结束位置
            # 先尝试查找对应的页脚
            end_pos = None
            
            # 寻找这个页码对应的页脚
            for footer_match in footer_matches:
                footer_page = int(footer_match.group(1))
                if footer_page == page_number:
                    # 结束位置是这个页脚的结束位置
                    end_pos = footer_match.end()
                    break
            
            # 如果没找到对应页脚，则使用下一个页头作为结束位置
            if end_pos is None:
                if i < len(header_matches) - 1:
                    end_pos = header_matches[i + 1].start()
                else:
                    end_pos = len(combined_text)

            # 提取页面内容
            page_content = combined_text[start_pos:end_pos]

            # 移除页面分隔符 (确保使用多行模式)
            page_content = re.sub(page_break_pattern, "", page_content, flags=re.DOTALL)
            
            # 移除页眉
            page_content = re.sub(r'(?s)<header style="background-color: #f5f5f5;.*?</header>', "", page_content)
            
            # 移除页脚
            page_content = re.sub(r'(?s)<footer\s+style="[^"]*">.*?</footer>', "", page_content)
            
            # 清理可能的多余空行
            page_content = re.sub(r'\n{3,}', '\n\n', page_content)

            # 添加到页面内容列表
            page_contents.append((page_number, page_content))

        # 创建输出目录
        output_dir = caption_path.with_suffix("")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存每一页为独立文件
        for page_number, page_content in page_contents:
            # 处理图片路径，将路径从子文件夹改为同级
            img_pattern = r"!\[(.*?)\]\(([^/]+)/([^/)]+)\)"
            
            # 检查是否有重复引用的图片
            processed_page_content = page_content
            matches = list(re.finditer(img_pattern, page_content))
            
            if matches:
                # 使用字典记录每个图片第一次出现的位置
                first_occurrence = {}
                
                # 找出每个图片第一次出现的位置
                for match in matches:
                    alt_text = match.group(1)
                    folder = match.group(2)
                    img_name = match.group(3)
                    if img_name not in first_occurrence:
                        first_occurrence[img_name] = match
                
                # 先处理图片路径，统一格式
                processed_page_content = re.sub(img_pattern, r"![\1](\3)", processed_page_content)
                
                # 移除所有重复的图片，但保留第一次出现的位置
                for img_name, match in first_occurrence.items():
                    # 计算该图片在文本中所有出现的位置
                    all_matches = [m for m in matches if m.group(3) == img_name]
                    
                    # 如果有多次出现，移除除了第一次之外的所有引用
                    if len(all_matches) > 1:
                        # 排序匹配，按位置从前向后处理
                        sorted_matches = sorted(all_matches, key=lambda m: m.start())
                        
                        # 跳过第一次出现的匹配
                        for m in sorted_matches[1:]:
                            # 构建要移除的模式
                            alt_text = m.group(1)
                            pattern_to_remove = f"!\\[{re.escape(alt_text)}\\]\\({re.escape(img_name)}\\)"
                            # 从处理后的内容中移除该模式
                            processed_page_content = re.sub(pattern_to_remove, "", processed_page_content, count=1)
            else:
                # 如果没有匹配到图片，只进行路径格式转换
                processed_page_content = re.sub(img_pattern, r"![\1](\3)", page_content)
            
            page_filename = f"{caption_path.stem}_{page_number}.md"
            page_file_path = output_dir / page_filename

            # 保存页面内容
            with open(page_file_path, "w", encoding="utf-8") as f:
                f.write(processed_page_content)

            console.print(
                f"[{CONSOLE_COLORS['text']}]text: {page_file_path} saved successfully.[/{CONSOLE_COLORS['text']}]"
            )

        return True
    except Exception as e:
        console.print(f"[red]Error saving pages: {e}[/red]")
        return False


def split_md_document(uri: Path, caption_lines: List[str], save_caption_func) -> None:
    """分割多页Markdown文档并单独保存每一页

    Args:
        uri: 文件路径
        caption_lines: 包含多页内容的文本列表
        save_caption_func: 用于保存单页内容的函数
    """
    try:
        # 检查是否包含多页内容
        if any(
            '<header style="background-color: #f5f5f5;' in line
            for line in caption_lines
        ):
            # 调用分页保存函数
            md_path = uri.with_suffix(".md")
            save_caption_by_pages(uri, caption_lines)
            console.print(
                f"[green]Successfully split document into individual pages: {md_path}[/green]"
            )
        else:
            # 如果不是多页文档，按原样保存
            console.print(
                f"[yellow]Document does not contain multiple pages, saving as single file.[/yellow]"
            )
    except Exception as e:
        console.print(f"[red]Error splitting MD document: {e}[/red]")


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
        transient=False,  # 防止进度条随刷新滚动
    ) as progress:

        global console

        console = progress.console

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
                elif suffix in application_extensions:
                    media_type = "application"
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
                        else uri.with_suffix("")
                    )
                    caption_file_path.parent.mkdir(parents=True, exist_ok=True)
                    save_caption(caption_file_path, caption, media_type)

                    if clip_with_caption and (uri.with_suffix(".srt")).exists():
                        subs = pysrt.open(uri.with_suffix(".srt"), encoding="utf-8")
                        try:
                            split_video_with_imageio_ffmpeg(uri, subs, save_caption)
                        except Exception as e:
                            console.print(f"[red]Error splitting video: {e}[/red]")
                            split_media_stream_clips(
                                uri, media_type, subs, save_caption
                            )
                    elif clip_with_caption and (uri.with_suffix(".md")).exists():
                        split_md_document(uri, caption, save_caption)

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
    extract_from_lance(
        args.lance_file, args.output_dir, args.version, not args.not_clip_with_caption
    )


if __name__ == "__main__":
    main()
