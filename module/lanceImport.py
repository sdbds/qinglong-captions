from __future__ import annotations

# /// script
# dependencies = [
#   "setuptools",
#   "pillow>=11.3",
#   "pylance>=2.0.1",
#   "rich>=13.5.0",
#   "imageio>=2.31.1",
#   "imageio-ffmpeg>=0.4.8",
#   "numpy",
#   "mutagen",
#   "toml",
#   "pyarrow",
# ]
# ///
"""
Dataset processing utilities for image-caption pairs using Lance format.
This module provides tools for converting image-caption datasets to Lance format
and accessing the data through PyTorch datasets.
"""

import argparse
import hashlib
import mimetypes
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import imageio.v3 as iio
import lance
import numpy as np
import pyarrow as pa
from PIL import Image, ImageMode
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

from config.config import (
    CONSOLE_COLORS,
    DATASET_SCHEMA,
    get_supported_extensions,
)
from utils.lance_utils import update_or_create_tag

console = Console(color_system="truecolor", force_terminal=True)
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
_sidecar_text_ext_set = frozenset({".txt", ".md"})
_non_text_primary_ext_set = _image_ext_set | _animation_ext_set | _video_ext_set | _audio_ext_set | _application_ext_set
_all_ext_set = _non_text_primary_ext_set | _text_ext_set


@dataclass
class Metadata:
    """Metadata for media file."""

    uris: str  # File path or URL
    mime: str  # MIME type
    width: int = 0  # Image/video width in pixels
    height: int = 0  # Image/video height in pixels
    depth: int = 0  # Sample depth/width in bits
    channels: int = 0  # Number of channels (RGB=3, RGBA=4, mono=1, stereo=2)
    hash: str = ""  # SHA256 hash
    size: int = 0  # File size in bytes
    has_audio: bool = False  # True if audio is present
    duration: Optional[int] = None  # Duration in milliseconds
    num_frames: Optional[int] = 1  # Number of frames
    frame_rate: float = 0.0  # Frames/samples per second
    blob: bytes = b""  # Binary data

    @property
    def filename(self) -> str:
        """File name without extension, derived from filepath."""
        return Path(self.uris).stem

    @property
    def ext(self) -> str:
        """File extension derived from filepath, including dot."""
        return Path(self.uris).suffix

    @property
    def bits_per_channel(self) -> int:
        """Get bits per channel."""
        return self.depth if self.channels > 0 else 0

    @property
    def bit_rate(self) -> int:
        """Calculate bit rate in bits per second.

        For audio: channels * depth * frame_rate
        For image: channels * depth * width * height * frame_rate
        """
        if self.duration == 0:
            return 0

        bits_per_sample = self.channels * self.bits_per_channel
        if self.width and self.height:  # Image/Video
            bits_per_frame = bits_per_sample * self.width * self.height
        else:  # Audio
            bits_per_frame = bits_per_sample

        return int(bits_per_frame * self.frame_rate)


class VideoImportMode(Enum):
    """Import mode for video files."""

    ALL = 0  # Import complete video with audio
    VIDEO_ONLY = 1  # Import video without audio
    AUDIO_ONLY = 2  # Import audio only without video
    VIDEO_SPLIT_AUDIO = 3  # Split and import both video and audio separately

    @classmethod
    def from_int(cls, value: int) -> "VideoImportMode":
        """Convert integer to VideoImportMode."""
        for mode in cls:
            if mode.value == value:
                return mode
        raise ValueError(f"Invalid import mode value: {value}")


class FileProcessor:
    """Utility class for processing files.

    This class provides methods for loading and processing file metadata,
    including size, format, and hash calculations.

    Example:
        processor = FileProcessor()
        metadata = processor.load_metadata("path/to/image.jpg")
        if metadata:
            print(f"Image size: {metadata.width}x{metadata.height}")
    """

    def _extract_audio_metadata(
        self, video: Any, file_path: str, meta: Dict[str, Any], save_binary: bool = True
    ) -> Optional[Tuple[Metadata, bytes]]:
        """Extract audio metadata from video file.

        Args:
            video: Video file object
            file_path: Path to video file
            meta: Video metadata
            save_binary: Whether to save binary data

        Returns:
            Tuple of (metadata, binary_data) if successful, None if failed
        """
        if not meta.get("has_audio"):
            console.print(f"[yellow]Warning: Video file {file_path} has no audio track[/yellow]")
            return None

        # Get audio data
        audio_data = video.read_audio()  # This returns numpy array of audio samples
        if audio_data is None:
            console.print(f"[yellow]Warning: Failed to extract audio data from {file_path}[/yellow]")
            return None

        # Get audio binary data and calculate hash
        binary_data = audio_data.astype(np.int16).tobytes()
        audio_hash = hashlib.sha256(binary_data).hexdigest()

        # Create audio metadata
        duration = int(meta.get("duration", 0) * 1000)  # Convert to milliseconds
        audio_metadata = Metadata(
            uris=file_path,
            mime="audio/wav",
            width=0,
            height=0,
            channels=audio_data.shape[1],
            depth=16,  # 16-bit audio
            hash=audio_hash,  # Use audio's own hash
            size=len(binary_data),
            has_audio=True,
            duration=duration,
            num_frames=int(duration * meta.get("audio_fps", 44100) / 1000),
            frame_rate=meta.get("audio_fps", 44100),
            blob=binary_data if save_binary else b"",
        )

        return audio_metadata, binary_data

    @staticmethod
    def load_metadata(
        file_path: str,
        save_binary: bool = True,
        import_mode: VideoImportMode = VideoImportMode.ALL,
    ) -> Optional[Metadata]:
        """Load and process image metadata.

        Args:
            file_path: Path to the media file
            save_binary: If True, store the binary data of the image
            import_mode: Mode for importing video components:
                        - ALL: Complete video with audio
                        - VIDEO_ONLY: Video without audio
                        - AUDIO_ONLY: Audio only
                        - VIDEO_SPLIT_AUDIO: Split video and audio

        Returns:
            Metadata object if successful, None if failed

        Raises:
            FileNotFoundError: If the image file doesn't exist
            IOError: If there's an error reading the file
            SyntaxError: If the image format is invalid
        """
        try:
            _suffix = Path(file_path).suffix.lower()
            if _suffix in _image_ext_set or _suffix in _animation_ext_set:
                with Image.open(file_path) as img:
                    # Get file pointer position
                    pos = img.fp.tell()
                    # Reset to beginning
                    img.fp.seek(0)
                    # Read data and calculate hash
                    binary_data = img.fp.read()
                    image_hash = hashlib.sha256(binary_data).hexdigest()
                    # Restore position
                    img.fp.seek(pos)

                    # Get animation info if available
                    duration = 0  # Initialize to 0 for accumulation
                    n_frames = None
                    frame_rate = 0.0

                    if hasattr(img, "n_frames") and img.n_frames > 1:
                        n_frames = img.n_frames
                        # Get duration in milliseconds
                        for frame in range(img.n_frames):
                            img.seek(frame)
                            duration += img.info.get("duration", 0)
                        # Calculate frame rate from duration and frame count
                        if duration > 0:
                            frame_rate = (n_frames * 1000) / duration  # Convert to fps
                    else:
                        duration = None

                    # Get image MIME type, fallback to PIL format
                    mime_type, _ = mimetypes.guess_type(file_path)
                    mime = mime_type or f"image/{img.format.lower()}"

                    # Get depth based on mode type
                    channels = len(img.getbands())
                    mode = img.mode

                    # Try different ways to get bit depth
                    depth = None
                    # 1. Try img.bits first (most accurate, includes 12-bit)
                    if hasattr(img, "bits"):
                        depth = img.bits
                    # 2. Try to get from tag for TIFF images (can have 12-bit)
                    elif hasattr(img, "tag_v2"):
                        bits = img.tag_v2.get(258)  # BitsPerSample tag
                        if bits:
                            depth = bits[0] if isinstance(bits, tuple) else bits
                    # 3. Fallback to mode info
                    if depth is None:
                        mode_info = ImageMode.getmode(mode)
                        if mode_info.basetype == "1":
                            depth = 1
                        else:
                            # Convert bytes to bits (note: this will show 16 for 12-bit images)
                            type_size = int(mode_info.typestr[-1])
                            depth = type_size * 8

                    return Metadata(
                        uris=file_path,
                        mime=mime,
                        width=img.size[0],
                        height=img.size[1],
                        depth=depth,
                        channels=channels,
                        hash=image_hash,
                        size=Path(file_path).stat().st_size,
                        has_audio=False,
                        duration=duration,
                        num_frames=n_frames,
                        frame_rate=frame_rate,
                        blob=binary_data if save_binary else b"",
                    )
            elif _suffix in _video_ext_set:
                try:
                    # Get video metadata first
                    meta = iio.immeta(file_path) or {}

                    # Get video MIME type
                    mime_type, _ = mimetypes.guess_type(file_path)
                    extension = Path(file_path).suffix.lstrip(".")
                    mime = mime_type or f"video/{extension}"

                    # Get basic video info with safety checks
                    size = meta.get("size", (0, 0))
                    width = int(size[0]) if isinstance(size, (tuple, list)) and len(size) > 0 else 0
                    height = int(size[1]) if isinstance(size, (tuple, list)) and len(size) > 1 else 0

                    # Handle fps with safety checks
                    fps = meta.get("fps", 0)
                    frame_rate = float(fps) if fps and not np.isinf(fps) and not np.isnan(fps) else 0.0

                    # Handle duration with safety checks
                    dur = meta.get("duration", 0)
                    duration = int(dur * 1000) if dur and not np.isinf(dur) and not np.isnan(dur) else 0

                    # Calculate frames with safety checks
                    n_frames = meta.get("nframes", 0)
                    if not n_frames or np.isinf(n_frames) or np.isnan(n_frames):
                        if frame_rate > 0 and duration > 0:
                            n_frames = int(frame_rate * (duration / 1000))
                        else:
                            n_frames = 0

                    # Get first frame for color info
                    try:
                        with iio.imopen(file_path, "r") as file:
                            first_frame = file.read(index=0)
                            channels = first_frame.shape[2] if len(first_frame.shape) > 2 else 1
                            depth = first_frame.dtype.itemsize * 8
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not read first frame from {file_path}: {e}[/yellow]")
                        channels = 3  # Assume RGB
                        depth = 8  # Assume 8-bit

                    # Read video binary data and calculate hash in chunks
                    hasher = hashlib.sha256()
                    binary_data = bytearray()
                    chunk_size = 8192  # 8KB chunks

                    with open(file_path, "rb") as f:
                        while True:
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                            hasher.update(chunk)
                            if save_binary:
                                binary_data.extend(chunk)

                    video_hash = hasher.hexdigest()

                    return Metadata(
                        uris=file_path,
                        mime=mime,
                        width=width,
                        height=height,
                        depth=depth,
                        channels=channels,
                        hash=video_hash,
                        size=Path(file_path).stat().st_size,
                        has_audio=meta.get("has_audio", False),
                        duration=duration,
                        num_frames=n_frames,
                        frame_rate=frame_rate,
                        blob=bytes(binary_data) if save_binary else b"",
                    )
                except Exception as e:
                    console.print(f"[red]Error processing video {file_path}: {str(e)}[/red]")
                    return None

            elif _suffix in _audio_ext_set:
                from mutagen import File as MutagenFile

                try:
                    # Read audio file as binary first
                    binary_data = Path(file_path).read_bytes()
                    audio_hash = hashlib.sha256(binary_data).hexdigest()

                    # Get audio MIME type
                    mime_type, _ = mimetypes.guess_type(file_path)
                    extension = Path(file_path).suffix.lstrip(".")
                    mime = mime_type or f"audio/{extension}"

                    # Try to get audio metadata using mutagen first
                    audio = MutagenFile(file_path)
                    if audio is not None:
                        # Get duration in milliseconds
                        duration = int(audio.info.length * 1000)
                        # Get sample rate
                        frame_rate = getattr(audio.info, "sample_rate", 44100)
                        # Get number of channels
                        channels = getattr(audio.info, "channels", 2)
                        # Get bit depth if available
                        depth = getattr(audio.info, "bits_per_sample", 16)
                        # Calculate number of frames
                        n_frames = int(audio.info.length * frame_rate)
                    else:
                        raise Exception("Could not read audio metadata")

                    return Metadata(
                        uris=file_path,
                        mime=mime,
                        width=0,
                        height=0,
                        depth=depth,
                        channels=channels,
                        hash=audio_hash,
                        size=Path(file_path).stat().st_size,
                        has_audio=True,
                        duration=duration,
                        num_frames=n_frames,
                        frame_rate=frame_rate,
                        blob=binary_data if save_binary else b"",
                    )
                except Exception as e:
                    console.print(f"[red]Error processing audio {file_path}: {str(e)}[/red]")
                    return None

            elif _suffix in _text_ext_set:
                try:
                    binary_data = Path(file_path).read_bytes()
                    text_hash = hashlib.sha256(binary_data).hexdigest()
                    mime_type, _ = mimetypes.guess_type(file_path)

                    if _suffix == ".md":
                        mime = "text/markdown"
                    else:
                        mime = mime_type or "text/plain"

                    return Metadata(
                        uris=file_path,
                        mime=mime,
                        width=0,
                        height=0,
                        depth=0,
                        channels=0,
                        hash=text_hash,
                        size=Path(file_path).stat().st_size,
                        has_audio=False,
                        duration=0,
                        num_frames=0,
                        frame_rate=0,
                        blob=binary_data if save_binary else b"",
                    )
                except Exception as e:
                    console.print(f"[red]Error processing text asset {file_path}: {str(e)}[/red]")
                    return None
            elif _suffix in _application_ext_set:
                try:
                    # Read application file as binary first
                    binary_data = Path(file_path).read_bytes()
                    application_hash = hashlib.sha256(binary_data).hexdigest()

                    # Get application MIME type
                    mime_type, _ = mimetypes.guess_type(file_path)
                    extension = Path(file_path).suffix.lstrip(".")
                    mime = mime_type or f"application/{extension}"

                    return Metadata(
                        uris=file_path,
                        mime=mime,
                        width=0,
                        height=0,
                        depth=0,
                        channels=0,
                        hash=application_hash,
                        size=Path(file_path).stat().st_size,
                        has_audio=False,
                        duration=0,
                        num_frames=0,
                        frame_rate=0,
                        blob=binary_data if save_binary else b"",
                    )
                except Exception as e:
                    console.print(f"[red]Error processing application {file_path}: {str(e)}[/red]")
                    return None

        except Exception as e:
            console.print(f"[red]Unexpected error processing {file_path}: {str(e)}[/red]")
            return None


def _iter_candidate_files(root: Path, recursive: bool = True) -> List[Path]:
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(path for path in iterator if path.is_file())


def _read_caption_file(path: Path) -> List[str]:
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".txt":
        return content.splitlines()
    return [content]


def _find_sidecar_caption(file_path: Path, caption_root: Optional[Path] = None, dataset_root: Optional[Path] = None) -> List[str]:
    if caption_root is None:
        sidecar_base = file_path.with_suffix("")
    else:
        relative_path = file_path.relative_to(dataset_root) if dataset_root is not None else Path(file_path.name)
        sidecar_base = (caption_root / relative_path).with_suffix("")

    for extension in (".txt", ".md", ".srt"):
        candidate = sidecar_base.with_suffix(extension)
        if candidate.exists() and candidate != file_path:
            return _read_caption_file(candidate)
    return []


def _make_data_item(file_path: Path, caption: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "file_path": str(file_path),
        "caption": caption or [],
        "chunk_offsets": [],
    }


def load_data(datasets_dir: str, texts_dir: Optional[str] = None, include_text_assets: bool = True) -> List[Dict[str, Any]]:
    """
    Load primary assets and optional sidecar text files from directories.

    Sidecar detection only applies to .txt/.md/.srt files paired with a
    non-text asset of the same stem. Standalone .txt/.md files are imported
    as primary assets by default.
    """
    dataset_root = Path(datasets_dir).absolute()
    data: List[Dict[str, Any]] = []

    if texts_dir:
        caption_root = Path(texts_dir).absolute()
        allowed_exts = _all_ext_set if include_text_assets else _non_text_primary_ext_set
        for file_path in _iter_candidate_files(dataset_root, recursive=True):
            if file_path.suffix.lower() not in allowed_exts:
                continue
            caption = _find_sidecar_caption(file_path, caption_root=caption_root, dataset_root=dataset_root)
            data.append(_make_data_item(file_path, caption))
        return data

    files = _iter_candidate_files(dataset_root, recursive=True)
    non_text_stems = {file_path.with_suffix("") for file_path in files if file_path.suffix.lower() in _non_text_primary_ext_set}

    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix in _non_text_primary_ext_set:
            data.append(_make_data_item(file_path, _find_sidecar_caption(file_path)))
            continue

        if not include_text_assets or suffix not in _text_ext_set:
            continue

        if suffix in _sidecar_text_ext_set and file_path.with_suffix("") in non_text_stems:
            continue

        data.append(_make_data_item(file_path))

    return data

def process(
    data: List[Dict[str, Any]],
    save_binary: bool = True,
    import_mode: VideoImportMode = VideoImportMode.ALL,
) -> pa.RecordBatch:
    """
    Process image-caption pairs into Lance format.

    Args:
        data: List of dictionaries containing file paths and captions.
        save_binary: Whether to save binary data.
        import_mode: Mode for importing video components:
                    - ALL: Complete video with audio
                    - VIDEO_ONLY: Video without audio
                    - AUDIO_ONLY: Audio only
                    - VIDEO_SPLIT_AUDIO: Split video and audio

    Returns:
        A PyArrow RecordBatch containing the processed data.
    """
    processor = FileProcessor()

    with Progress(
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
        transient=False,  # 防止进度条随刷新滚动
    ) as progress:
        global console

        console = progress.console

        task = progress.add_task("[green]Processing file...", total=len(data))

        for item in data:
            file_path = item["file_path"]
            caption = item.get("caption", [])

            console.print()

            # 根据文件类型选择颜色
            suffix = Path(file_path).suffix.lower()
            if suffix in _image_ext_set:
                if suffix in _animation_ext_set:
                    color = CONSOLE_COLORS["animation"]
                    media_type = "animation"
                else:
                    color = CONSOLE_COLORS["image"]
                    media_type = "image"
            elif suffix in _video_ext_set:
                color = CONSOLE_COLORS["video"]
                media_type = "video"
            elif suffix in _audio_ext_set:
                color = CONSOLE_COLORS["audio"]
                media_type = "audio"
            elif suffix in _text_ext_set:
                color = CONSOLE_COLORS["text"]
                media_type = "text"
            elif suffix in _application_ext_set:
                color = CONSOLE_COLORS["application"]
                media_type = "application"
            else:
                color = CONSOLE_COLORS["unknown"]
                media_type = "unknown"

            console.print(f"Processing {media_type} file [{color}]'{file_path}'[/{color}]")
            console.print(f"Caption: {caption}", style=CONSOLE_COLORS["caption"])

            metadata = processor.load_metadata(file_path, save_binary, import_mode)
            if not metadata:
                progress.update(task, advance=1)
                continue

            # Get field names and create arrays
            field_names = [field[0] for field in DATASET_SCHEMA]
            arrays = []
            for field_name, field_type in DATASET_SCHEMA:
                if field_name == "filepath":
                    value = str(Path(file_path).absolute())
                    array = pa.array([value], type=field_type)
                elif field_name == "captions":
                    array = pa.array([caption], type=field_type)
                elif field_name in item:
                    array = pa.array([item[field_name]], type=field_type)
                elif field_name == "blob":
                    array = pa.array([getattr(metadata, field_name)], type=field_type)
                else:
                    value = getattr(metadata, field_name)
                    # Convert None to appropriate default value based on type
                    if value is None:
                        if pa.types.is_integer(field_type):
                            value = 0
                        elif pa.types.is_floating(field_type):
                            value = 0.0
                        elif pa.types.is_boolean(field_type):
                            value = False
                        elif pa.types.is_string(field_type):
                            value = ""
                    array = pa.array([value], type=field_type)
                arrays.append(array)

            batch = pa.RecordBatch.from_arrays(
                arrays,
                names=field_names,
            )

            yield batch
            progress.update(task, advance=1)


def transform2lance(
    dataset_dir: str,
    caption_dir: Optional[str] = None,
    output_name: str = "dataset",
    save_binary: bool = True,
    not_save_disk: bool = False,
    import_mode: VideoImportMode = VideoImportMode.ALL,
    tag: str = "gemini",
    load_condition: Callable[..., List[Dict[str, Any]]] = load_data,
    include_text_assets: bool = True,
) -> Optional[lance.LanceDataset]:
    """
    Transform dataset assets into Lance format.
    """
    try:
        data = load_condition(dataset_dir, caption_dir, include_text_assets=include_text_assets)
    except TypeError:
        data = load_condition(dataset_dir, caption_dir)

    schema = pa.schema(
        [
            pa.field(
                name,
                pa_type,
                metadata={b"lance-encoding:blob": b"true"} if name == "blob" else None,
            )
            for name, pa_type in DATASET_SCHEMA
        ]
    )

    try:
        reader = pa.RecordBatchReader.from_batches(schema, process(data, save_binary, import_mode))

        dataset_path = Path(dataset_dir) / f"{output_name}.lance"
        mode = "append" if dataset_path.exists() and not not_save_disk else "create"

        lancedataset = lance.write_dataset(
            reader,
            str(dataset_path),
            schema,
            mode=mode if not_save_disk else "overwrite",
        )
        update_or_create_tag(lancedataset, tag)
        return lancedataset

    except AttributeError as e:
        console.print(f"[red]AttributeError: {e}[/red]")
        return None

def setup_parser() -> argparse.ArgumentParser:
    """Setup argument parser."""
    parser = argparse.ArgumentParser(description="Transform dataset into Lance format")
    parser.add_argument("dataset_dir", type=str, help="Directory containing training images")
    parser.add_argument(
        "--caption_dir",
        type=str,
        default=None,
        help="Directory containing caption files",
    )
    parser.add_argument("--output_name", type=str, default="dataset", help="Name of output dataset")
    parser.add_argument(
        "--no_save_binary",
        action="store_true",
        help="Don't save binary data in the dataset",
    )
    parser.add_argument(
        "--not_save_disk",
        action="store_true",
        help="Load dataset into memory instead of saving to disk",
    )
    parser.add_argument(
        "--import_mode",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Video import mode: 0=Complete video with audio, 1=Video only, 2=Audio only, 3=Split video and audio",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="gemini",
        help="Tag for the dataset",
    )
    parser.add_argument(
        "--include_text_assets",
        dest="include_text_assets",
        action="store_true",
        default=True,
        help="Import standalone .txt/.md files as primary assets (default: enabled)",
    )
    parser.add_argument(
        "--exclude_text_assets",
        dest="include_text_assets",
        action="store_false",
        help="Skip standalone .txt/.md primary assets and only import non-text primary assets",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    transform2lance(
        dataset_dir=args.dataset_dir,
        caption_dir=args.caption_dir,
        output_name=args.output_name,
        save_binary=not args.no_save_binary,
        not_save_disk=args.not_save_disk,
        import_mode=VideoImportMode.from_int(args.import_mode),
        tag=args.tag,
        include_text_assets=args.include_text_assets,
    )
