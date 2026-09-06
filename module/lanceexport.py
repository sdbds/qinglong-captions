from __future__ import annotations

# /// script
# dependencies = [
#   "setuptools",
#   "pillow>=11.3",
#   "pylance>=2.0.1",
#   "pysrt",
#   "rich>=13.5.0",
#   "imageio>=2.31.1",
#   "imageio-ffmpeg>=0.4.8",
#   "numpy",
#   "mutagen",
#   "toml",
#   "pyarrow",
#   "filelock>=3.32.3,<4",
# ]
# ///
import argparse
import json
import re
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import lance
import pysrt
from filelock import FileLock
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from config.config import CONSOLE_COLORS, DATASET_SCHEMA, get_supported_extensions
from utils.caption_index import CAPTION_INDEX_NAME, CaptionFileTransaction, CaptionIndex, caption_file_identity, caption_index_key, load_caption_index, write_caption_index
from utils.lance_blob import take_blob_files
from utils.console_util import print_exception
from utils.output_writer import ResolvedPaths, caption_extension_from_payload, caption_path_avoiding_sources, caption_text, disambiguate_caption_bases, normalize_caption_extension
from utils.path_safety import safe_child_path, safe_leaf_name
from utils.stream_util import split_media_stream_clips, split_video_with_imageio_ffmpeg

console = Console(color_system="truecolor", force_terminal=True)
_CAPTION_INDEX_LOCK_NAME = CAPTION_INDEX_NAME + ".lock"
image_extensions = get_supported_extensions("image")
animation_extensions = get_supported_extensions("animation")
video_extensions = get_supported_extensions("video")
audio_extensions = get_supported_extensions("audio")
text_extensions = get_supported_extensions("text")
application_extensions = get_supported_extensions("application")
# Frozen sets for O(1) membership testing (all lowercase)
_image_ext_set = frozenset(image_extensions)
_animation_ext_set = frozenset(animation_extensions)
_video_ext_set = frozenset(video_extensions)
_audio_ext_set = frozenset(audio_extensions)
_text_ext_set = frozenset(text_extensions)
_application_ext_set = frozenset(application_extensions)


class _ReadableBlob(Protocol):
    def read(self, size: int = -1) -> bytes:
        ...


def format_duration(duration_ms: int) -> str:
    """将毫秒转换为分:秒格式."""
    total_seconds = duration_ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"


def save_blob(
    uri: Path,
    blob: Union[bytes, _ReadableBlob],
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

        # Handle both bytes and blob-like readers without depending on Lance runtime types.
        if hasattr(blob, "read") and callable(getattr(blob, "read")):
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
                    (f"{metadata.get('num_frames', 1)} frames" if metadata.get("num_frames", 1) > 1 else None),
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
        console.print(f"[{color}]{media_type}: {uri} ({meta_str}) saved successfully.[/{color}]")
        return True
    except Exception as e:
        print_exception(console, e, prefix=f"Error saving {media_type} {uri}")
        return False


def _resolve_caption_target_path(
    base_path: Path,
    media_type: Optional[str],
    caption_suffix: str = "",
    caption_extension: Optional[str] = None,
) -> Path:
    extension = caption_extension
    if extension:
        if not extension.startswith("."):
            extension = f".{extension}"
    elif media_type in {"audio", "video"}:
        extension = ".srt"
    elif media_type == "application":
        extension = ".md"
    else:
        extension = ".txt"

    if caption_suffix:
        return base_path.with_name(f"{base_path.stem}{caption_suffix}{extension}")
    return base_path.with_suffix(extension)


def _extract_structured_caption_payload(caption_lines: List[str]) -> Optional[Dict[str, Any]]:
    for line in caption_lines:
        if not line:
            continue
        text = line.strip()
        if not (text.startswith("{") and text.endswith("}")):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _caption_output_paths(base_path, caption_lines, media_type, caption_suffix, caption_extension, protected_paths):
    payload = _extract_structured_caption_payload(caption_lines)
    extension = normalize_caption_extension(caption_extension) or caption_extension_from_payload(payload)
    target = _resolve_caption_target_path(Path(base_path), media_type, caption_suffix, extension)
    target = caption_path_avoiding_sources(target, protected_paths)
    single_structured = bool(caption_text(payload).strip()) and sum(bool(line and line.strip()) for line in caption_lines) == 1
    has_json = single_structured or (payload is not None and target.suffix not in {".srt", ".md"})
    json_target = (
        caption_path_avoiding_sources(target.with_suffix(".json"), protected_paths, additional_paths=(target,))
        if has_json else None
    )
    return target, json_target


def save_caption(
    caption_path: str,
    caption_lines: List[str],
    media_type: Optional[str],
    caption_suffix: str = "",
    caption_extension: Optional[str] = None,
    protected_paths=None,
) -> bool:
    """Save caption data to disk."""
    try:
        has_content = any(line.strip() for line in caption_lines if line)
        if not has_content:
            console.print(f"[red]No caption content found for {caption_path}[/red]")
            return False

        structured_payload = _extract_structured_caption_payload(caption_lines)
        structured_text = caption_text(structured_payload)
        single_structured_caption = bool(structured_text.strip()) and sum(bool(line and line.strip()) for line in caption_lines) == 1
        protected_paths = ResolvedPaths(protected_paths or ())
        caption_path, json_path = _caption_output_paths(
            caption_path, caption_lines, media_type, caption_suffix, caption_extension, protected_paths,
        )
        caption_path.parent.mkdir(parents=True, exist_ok=True)

        with open(caption_path, "w", encoding="utf-8") as f:
            if single_structured_caption:
                with open(json_path, "w", encoding="utf-8") as j:
                    json.dump(structured_payload, j, indent=2, ensure_ascii=False)
                f.write(structured_text)
            elif caption_path.suffix == ".srt":
                f.write("\n".join(caption_lines))
            elif caption_path.suffix == ".md":
                f.write("".join(caption_lines))
            else:
                for line in caption_lines:
                    if "<font color=" in line:
                        line = line.replace('<font color="green">', "").replace("<font color='green'>", "").replace("</font>", "")

                    if line.strip().startswith("{") and line.strip().endswith("}"):
                        try:
                            json_content = line.strip()
                            parsed_json = json.loads(json_content)
                            with open(json_path, "w", encoding="utf-8") as j:
                                json.dump(parsed_json, j, indent=2, ensure_ascii=False)

                            desc = caption_text(parsed_json)
                            f.write(desc if desc else "")
                        except json.JSONDecodeError:
                            if line and line.strip():
                                f.write(line.strip() + "\n")
                    else:
                        if line and line.strip():
                            f.write(line.strip() + "\n")

            console.print()
            console.print(f"[{CONSOLE_COLORS['text']}]text: {caption_path} saved successfully.[/{CONSOLE_COLORS['text']}]")
        return True
    except Exception as e:
        print_exception(console, e, prefix="Error saving caption")
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
        header_pattern = r'(?s)<header style="background-color: #f5f5f5;.*?<strong> Page (\d+) </strong>'
        footer_pattern = r'(?s)<footer\s+style="[^"]*">.*?<strong>\s*Page\s+(\d+)\s*</strong>.*?</footer>'
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
            md_header_pattern = r"^#{1,6}\s+(.+?)$"
            md_headers = list(re.finditer(md_header_pattern, combined_text, re.MULTILINE))

            if md_headers:
                # 使用Markdown标题分割内容
                console.print("[yellow]No HTML headers found, splitting by Markdown headers.[/yellow]")

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
                    safe_header = re.sub(r"[^\w\s-]", "", header_text)[:20].strip()
                    safe_header = re.sub(r"[-\s]+", "_", safe_header)

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
                console.print(f"[{CONSOLE_COLORS['text']}]text: {single_file_path} saved successfully.[/{CONSOLE_COLORS['text']}]")
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
            page_content = re.sub(
                r'(?s)<header style="background-color: #f5f5f5;.*?</header>',
                "",
                page_content,
            )

            # 移除页脚
            page_content = re.sub(r'(?s)<footer\s+style="[^"]*">.*?</footer>', "", page_content)

            # 清理可能的多余空行
            page_content = re.sub(r"\n{3,}", "\n\n", page_content)

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

            console.print(f"[{CONSOLE_COLORS['text']}]text: {page_file_path} saved successfully.[/{CONSOLE_COLORS['text']}]")

        return True
    except Exception as e:
        print_exception(console, e, prefix="Error saving pages")
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
        if any('<header style="background-color: #f5f5f5;' in line for line in caption_lines):
            # 调用分页保存函数
            md_path = uri.with_suffix(".md")
            save_caption_by_pages(uri, caption_lines)
            console.print(f"[green]Successfully split document into individual pages: {md_path}[/green]")
        else:
            # 如果不是多页文档，按原样保存
            console.print("[yellow]Document does not contain multiple pages, saving as single file.[/yellow]")
    except Exception as e:
        print_exception(console, e, prefix="Error splitting MD document")


def _media_type_for_uri(uri: Path) -> Optional[str]:
    suffix = uri.suffix.lower()
    for media_type, extensions in (
        ("image", _image_ext_set), ("animation", _animation_ext_set),
        ("video", _video_ext_set), ("audio", _audio_ext_set),
        ("text", _text_ext_set), ("application", _application_ext_set),
    ):
        if suffix in extensions:
            return media_type
    return None


def _validate_export_paths(planned_paths, generated_directories=(), protected_paths=()):
    occupied = {}
    for path, owner in planned_paths:
        resolved = path.resolve()
        key = caption_file_identity(resolved)
        if key in occupied:
            raise ValueError(f"Ambiguous export path collision: {path} ({occupied[key]} and {owner})")
        if path.is_dir():
            raise ValueError(f"Export path collision with a directory: {path}")
        occupied[key] = owner
    for path, owner in planned_paths:
        for parent in path.resolve().parents:
            if caption_file_identity(parent) in occupied:
                raise ValueError(f"Export file/directory collision: {path} ({owner})")
    directory_owners = {}
    for directory, owner in generated_directories:
        key = caption_file_identity(directory.resolve())
        if key in directory_owners or directory.is_file():
            raise ValueError(f"Ambiguous generated directory collision: {directory} ({owner})")
        directory_owners[key] = owner
    for directory, owner in generated_directories:
        for parent in directory.resolve().parents:
            key = caption_file_identity(parent)
            if key in occupied or key in directory_owners:
                raise ValueError(f"Generated directory collision: {directory} ({owner})")
    for path in [path for path, _ in planned_paths] + list(protected_paths):
        resolved = path.resolve()
        if any(caption_file_identity(parent) in directory_owners for parent in (resolved, *resolved.parents)):
            raise ValueError(f"Generated directory collision with a primary or planned output: {path}")


def _plan_export_targets(dataset, output_path: Path, captions_path: Optional[Path], *,
                         caption_suffix="", caption_extension=None, allowed_caption_types=None,
                         clip_with_caption=False):
    uris = [uri for batch in dataset.scanner(columns=["uris"]).to_batches() for uri in batch["uris"].to_pylist()]
    if len(set(uris)) != len(uris):
        raise ValueError("Cannot export ambiguous duplicate source URIs")
    targets = {}
    for uri in uris:
        source = Path(uri)
        media_target = source if source.exists() else safe_child_path(output_path, safe_leaf_name(uri))
        caption_base = safe_child_path(captions_path, safe_leaf_name(uri)) if captions_path else media_target
        targets[uri] = (media_target, caption_base)
    caption_bases = disambiguate_caption_bases({uri: base for uri, (_, base) in targets.items()})
    for uri, (media_target, caption_base) in targets.items():
        if caption_bases[uri] != caption_base:
            caption_base = caption_bases[uri]
            if not Path(uri).exists():
                media_target = safe_child_path(output_path, caption_base.name)
            targets[uri] = (media_target, caption_base)
    protected = ResolvedPaths([*uris, *(target for target, _ in targets.values())])
    planned_paths = [(media_target, uri) for uri, (media_target, _) in targets.items()]
    generated_directories = []
    caption_targets = {}
    companion_targets = {}
    if "captions" in dataset.schema.names:
        for batch in dataset.scanner(columns=["uris", "captions"]).to_batches():
            for uri, lines in zip(batch["uris"].to_pylist(), batch["captions"].to_pylist()):
                media_type = _media_type_for_uri(Path(uri))
                if not lines or not any(line and line.strip() for line in lines):
                    continue
                if allowed_caption_types is not None and media_type not in allowed_caption_types:
                    continue
                target, json_target = _caption_output_paths(
                    targets[uri][1], lines, media_type, caption_suffix, caption_extension, protected,
                )
                caption_targets[uri] = target
                planned_paths.append((target, f"caption for {uri}"))
                if json_target is not None:
                    planned_paths.append((json_target, f"JSON caption for {uri}"))
                    companion_targets[uri] = json_target
                if clip_with_caption and not caption_suffix:
                    if media_type in {"audio", "video"} and target.suffix.lower() == ".srt":
                        media_target = targets[uri][0]
                        generated_directories.append((media_target.parent / f"{media_target.stem}_clip", uri))
                    elif target.suffix.lower() == ".md" and any('<header style="background-color: #f5f5f5;' in line for line in lines):
                        generated_directories.append((target.with_suffix(""), uri))
    indexes = {}

    def index_for(directory):
        directory = directory.resolve()
        if directory not in indexes:
            indexes[directory] = load_caption_index(directory, strict=True) or CaptionIndex()
            planned_paths.append((directory / CAPTION_INDEX_NAME, "caption index"))
            lock_path = directory / _CAPTION_INDEX_LOCK_NAME
            if lock_path.is_file() and lock_path.stat().st_size:
                raise ValueError(f"Cannot overwrite an unowned caption index lock: {lock_path}")
            planned_paths.append((lock_path, "caption index lock"))
        return indexes[directory]

    for uri, (media_target, _) in targets.items():
        if not Path(uri).exists():
            directory = media_target.parent.resolve()
            index = index_for(directory)
            key = caption_index_key(media_target, directory)
            index.validate_origin(key, Path(uri), require_known=media_target.exists() or key in index.captions)
    for uri, target in caption_targets.items():
        directory = target.parent.resolve()
        index = index_for(directory)
        key = caption_index_key(targets[uri][0], directory)
        if not Path(uri).exists():
            index.validate_origin(key, Path(uri), require_known=target.exists() or key in index.captions)
        index.validate_file_claim(key, target.name)
        companion = companion_targets.get(uri)
        if companion is not None:
            index.validate_file_claim(key, companion.name, require_owned=companion.exists())
    _validate_export_paths(planned_paths, generated_directories, protected)
    return targets, protected, caption_targets, indexes


def extract_from_lance(
    lance_or_path: Union[str, lance.LanceDataset],
    output_dir: str,
    version: str = "gemini",
    clip_with_caption: bool = True,
    caption_dir: Optional[str] = None,
    caption_suffix: str = "",
    caption_extension: Optional[str] = None,
    allowed_caption_types: Optional[List[str]] = None,
) -> None:
    """
    Extract images and captions from Lance dataset.
    """
    ds = lance.dataset(lance_or_path, version=version) if isinstance(lance_or_path, str) else lance_or_path

    output_path = Path(output_dir)

    captions_dir_path = None
    if caption_dir:
        captions_dir_path = Path(caption_dir)

    allowed_caption_type_set = set(allowed_caption_types) if allowed_caption_types else None
    plan_options = dict(
        caption_suffix=caption_suffix, caption_extension=caption_extension,
        allowed_caption_types=allowed_caption_type_set, clip_with_caption=clip_with_caption,
    )
    plan = _plan_export_targets(ds, output_path, captions_dir_path, **plan_options)
    directories = sorted(plan[3], key=caption_file_identity)
    with ExitStack() as locks:
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            locks.enter_context(FileLock(str(directory / _CAPTION_INDEX_LOCK_NAME), preserve_lock_file=True))
        # A waiting exporter must validate against the last committed generation.
        plan = _plan_export_targets(ds, output_path, captions_dir_path, **plan_options)
        if not set(plan[3]).issubset(directories):
            raise RuntimeError("Export source locations changed while acquiring locks; retry the export")
        _extract_planned_dataset(
            ds, output_path, captions_dir_path, plan,
            clip_with_caption=clip_with_caption, caption_suffix=caption_suffix,
            caption_extension=caption_extension, allowed_caption_type_set=allowed_caption_type_set,
        )


def _extract_planned_dataset(ds, output_path, captions_dir_path, plan, *, clip_with_caption,
                             caption_suffix, caption_extension, allowed_caption_type_set):
    export_targets, protected_paths, caption_targets, caption_indexes = plan
    output_path.mkdir(parents=True, exist_ok=True)
    if captions_dir_path is not None:
        captions_dir_path.mkdir(parents=True, exist_ok=True)

    dirty_indexes = set()
    transactions = {}
    postprocess_items = []
    with ExitStack() as publications, ExitStack() as index_writes, Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(spinner_name="dots"),
        MofNCompleteColumn(separator="/"),
        BarColumn(bar_width=40, complete_style="green", finished_style="bold green"),
        TextColumn("|"),
        TaskProgressColumn(),
        TextColumn("|"),
        TransferSpeedColumn(),
        TextColumn("|"),
        TimeElapsedColumn(),
        TextColumn("|"),
        TimeRemainingColumn(),
        expand=True,
        transient=False,
    ) as progress:
        global console

        def publication_for(directory):
            if directory not in transactions:
                transactions[directory] = publications.enter_context(CaptionFileTransaction(directory))
                transactions[directory].watch(directory / CAPTION_INDEX_NAME)
            return transactions[directory]

        def retain_origin(source, media_target, directory):
            publication_for(directory)
            entries = caption_indexes[directory]
            entries.set_origin(caption_index_key(media_target, directory), source)
            if directory not in dirty_indexes:
                index_writes.callback(write_caption_index, directory, entries)
                dirty_indexes.add(directory)

        console = progress.console
        task = progress.add_task("[green]Extracting files...", total=ds.count_rows())
        row_offset = 0

        for batch in ds.to_batches():
            batch_field_names = set(batch.schema.names)
            metadata_batch = {
                field[0]: batch.column(field[0]).to_pylist()
                for field in DATASET_SCHEMA
                if field[0] != "blob" and field[0] in batch_field_names
            }
            indices = list(range(row_offset, row_offset + len(batch)))
            blobs = take_blob_files(ds, indices, "blob") if "blob" in batch_field_names else [None] * len(batch)
            row_offset += len(batch)

            for i in range(len(batch)):
                metadata = {key: values[i] for key, values in metadata_batch.items()}
                uri = Path(metadata["uris"])
                blob = blobs[i]

                suffix = uri.suffix.lower()
                media_type = _media_type_for_uri(uri)

                blob_target, caption_file_path = export_targets[str(metadata["uris"])]
                source_missing = not uri.exists()
                media_publication = None
                if source_missing and blob:
                    if media_type:
                        media_publication = publication_for(blob_target.parent.resolve())
                        media_publication.watch(blob_target)
                        if not save_blob(blob_target, blob, metadata, media_type):
                            media_publication.restore(blob_target)
                            progress.advance(task)
                            continue
                    else:
                        console.print(f"[yellow]Unsupported file format: {suffix}[/yellow]")
                        progress.advance(task)
                        continue

                caption = metadata.get("captions", [])
                actual_caption_path = caption_targets.get(str(metadata["uris"]))
                should_save_caption = actual_caption_path is not None and bool(caption) and (
                    allowed_caption_type_set is None or media_type in allowed_caption_type_set
                )
                if should_save_caption:
                    caption_file_path.parent.mkdir(parents=True, exist_ok=True)
                    directory = actual_caption_path.parent.resolve()
                    _, json_path = _caption_output_paths(
                        caption_file_path, caption, media_type, caption_suffix, caption_extension, protected_paths,
                    )
                    output_paths = [actual_caption_path] + ([json_path] if json_path is not None else [])
                    publication = publication_for(directory)
                    publication.watch(*output_paths)
                    saved = save_caption(
                        str(caption_file_path),
                        caption,
                        media_type,
                        caption_suffix=caption_suffix,
                        caption_extension=caption_extension,
                        protected_paths=protected_paths,
                    )

                    if not saved:
                        publication.restore(*output_paths)
                        if media_publication is not None:
                            media_publication.restore(blob_target)
                        progress.advance(task)
                        continue
                    entries = caption_indexes[directory]
                    key = caption_index_key(blob_target, directory)
                    entries.set_caption(key, actual_caption_path.name)
                    if json_path is not None:
                        entries.claim_file(key, json_path.name)
                    if source_missing:
                        retain_origin(uri, blob_target, blob_target.parent.resolve())
                        retain_origin(uri, blob_target, directory)
                    if directory not in dirty_indexes:
                        index_writes.callback(write_caption_index, directory, entries)
                        dirty_indexes.add(directory)
                    if clip_with_caption and not caption_suffix:
                        postprocess_items.append((blob_target, actual_caption_path, media_type, caption))

                if media_publication is not None:
                    retain_origin(uri, blob_target, blob_target.parent.resolve())
                progress.advance(task)

    # Derived pages/clips must not precede the commit of their source captions.
    for blob_target, caption_path, media_type, caption in postprocess_items:
        if media_type in {"audio", "video"} and caption_path.suffix.lower() == ".srt" and blob_target.is_file():
            subs = pysrt.open(caption_path, encoding="utf-8")
            try:
                split_video_with_imageio_ffmpeg(blob_target, subs, save_caption)
            except Exception as e:
                print_exception(console, e, prefix="Error splitting video")
                split_media_stream_clips(blob_target, media_type, subs, save_caption)
        elif caption_path.suffix.lower() == ".md":
            split_md_document(caption_path, caption, save_caption)


def main():
    parser = argparse.ArgumentParser(description="Extract images and captions from a Lance dataset")
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
        "--caption_suffix",
        default="",
        help="Suffix inserted before the exported caption extension, for example _zh_cn",
    )
    parser.add_argument(
        "--caption_extension",
        default=None,
        help="Optional caption extension override, for example .md",
    )
    parser.add_argument(
        "--allowed_caption_types",
        default="",
        help="Comma separated caption media types to export, for example text,application",
    )
    parser.add_argument(
        "--not_clip_with_caption",
        action="store_true",
        help="Not clip with caption",
    )

    args = parser.parse_args()
    allowed_caption_types = [item.strip() for item in args.allowed_caption_types.split(",") if item.strip()]
    extract_from_lance(
        args.lance_file,
        args.output_dir,
        args.version,
        not args.not_clip_with_caption,
        caption_suffix=args.caption_suffix,
        caption_extension=args.caption_extension,
        allowed_caption_types=allowed_caption_types or None,
    )


if __name__ == "__main__":
    main()
