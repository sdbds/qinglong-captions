"""
Dataset processing utilities for image-caption pairs using Lance format.
This module provides tools for converting image-caption datasets to Lance format
and accessing the data through PyTorch datasets.
"""

import argparse
from contextlib import nullcontext
import hashlib
import io
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable, Tuple, Union
import wave
import imageio.v3 as iio
import lance
import pyarrow as pa
from PIL import Image, ImageMode
from rich.progress import Progress
from rich.console import Console
import mimetypes
from pathlib import Path
from enum import Enum, auto
import numpy as np

from config import get_supported_extensions, DATASET_SCHEMA


console = Console()
image_extensions = get_supported_extensions("image")
animation_extensions = get_supported_extensions("animation")
video_extensions = get_supported_extensions("video")
audio_extensions = get_supported_extensions("audio")


@dataclass
class Metadata:
    """Metadata for media file."""

    uris: str  # File path or URL
    mime: str  # MIME type
    width: int  # Image/video width in pixels
    height: int  # Image/video height in pixels
    channels: int = 0  # Number of channels (RGB=3, RGBA=4, mono=1, stereo=2)
    depth: int  # Sample depth/width in bits
    hash: str  # SHA256 hash
    size: int  # File size in bytes
    has_audio: bool = False  # True if audio is present
    duration: Optional[int] = None  # Duration in milliseconds
    num_frames: Optional[int] = None  # Number of frames
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
            console.print(
                f"[yellow]Warning: Video file {file_path} has no audio track[/yellow]"
            )
            return None

        # Get audio data
        audio_data = video.read_audio()  # This returns numpy array of audio samples
        if audio_data is None:
            console.print(
                f"[yellow]Warning: Failed to extract audio data from {file_path}[/yellow]"
            )
            return None

        # Get audio binary data and calculate hash
        binary_data = audio_data.astype(np.int16).tobytes()
        audio_hash = hashlib.sha256(binary_data).hexdigest()

        # Create audio metadata
        duration = int(meta.get("duration", 0) * 1000)  # Convert to milliseconds
        audio_metadata = Metadata(
            uris=file_path,
            mime=f"audio/wav",
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
            if file_path.endswith(image_extensions + animation_extensions):
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
                        channels=channels,
                        depth=depth,
                        hash=image_hash,
                        size=Path(file_path).stat().st_size,
                        has_audio=False,
                        duration=duration,
                        num_frames=n_frames,
                        frame_rate=frame_rate,
                        blob=binary_data if save_binary else b"",
                    )
            elif file_path.endswith(video_extensions):
                # Open video file
                with iio.imopen(file_path, "r") as video:

                    # Get video MIME type
                    mime_type, _ = mimetypes.guess_type(file_path)
                    extension = Path(file_path).suffix.lstrip(".")
                    mime = mime_type or f"video/{extension}"

                    # Get video metadata
                    meta = video.metadata
                    width = meta.get("size", (0, 0))[0]
                    height = meta.get("size", (0, 0))[1]
                    frame_rate = float(meta.get("fps", 0))
                    duration = int(
                        meta.get("duration", 0) * 1000
                    )  # Convert to milliseconds
                    n_frames = meta.get(
                        "nframes", int(frame_rate * meta.get("duration", 0))
                    )

                    # Get first frame for color info
                    first_frame = video.read(index=0)
                    channels = first_frame.shape[2] if len(first_frame.shape) > 2 else 1
                    depth = first_frame.dtype.itemsize * 8

                    # Read video binary data and calculate hash
                    video.seek(0)
                    binary_data = video.read_bytes()
                    video_hash = hashlib.sha256(binary_data).hexdigest()

                    if import_mode == VideoImportMode.ALL:
                        has_audio = meta.get("has_audio", False)
                    elif import_mode == VideoImportMode.VIDEO_ONLY:
                        # Remove audio from video
                        if meta.get("has_audio"):
                            video.audio = None
                        has_audio = False

                    elif import_mode == VideoImportMode.AUDIO_ONLY:
                        # Extract audio metadata
                        result = self._extract_audio_metadata(
                            video, file_path, meta, save_binary
                        )
                        if result is None:
                            return None
                        return result[0]  # Return just the metadata

                    elif import_mode == VideoImportMode.VIDEO_SPLIT_AUDIO:
                        # Create separate entries for video and audio
                        if meta.get("has_audio", False):
                            # Extract audio metadata
                            result = self._extract_audio_metadata(
                                video, file_path, meta, save_binary
                            )
                            if result is None:
                                # If no audio, just return video metadata
                                has_audio = False
                            else:
                                audio_metadata = result[0]
                                # Create video metadata
                                video_metadata = Metadata(
                                    uris=file_path,
                                    mime=mime,
                                    width=width,
                                    height=height,
                                    channels=channels,
                                    depth=depth,
                                    hash=video_hash,
                                    size=Path(file_path).stat().st_size,
                                    has_audio=False,
                                    duration=duration,
                                    num_frames=n_frames,
                                    frame_rate=frame_rate,
                                    blob=binary_data if save_binary else b"",
                                )
                                return [video_metadata, audio_metadata]
                        else:
                            # If no audio, just return video metadata
                            has_audio = False

                    return Metadata(
                        uris=file_path,
                        mime=mime,
                        width=width,
                        height=height,
                        channels=channels,
                        depth=depth,
                        hash=video_hash,
                        size=Path(file_path).stat().st_size,
                        has_audio=has_audio,
                        duration=duration,
                        num_frames=n_frames,
                        frame_rate=frame_rate,
                        blob=binary_data if save_binary else b"",
                    )
            elif file_path.endswith(audio_extensions):
                with wave.open(file_path, "rb") as wav:
                    # Get audio info
                    frame_rate = float(wav.getframerate())  # Hz
                    num_frames = wav.getnframes()
                    duration = int(
                        num_frames * 1000 / frame_rate
                    )  # Convert to milliseconds
                    channels = wav.getnchannels()
                    depth = wav.getsampwidth() * 8  # Convert bytes to bits

                    # Read audio data and calculate hash
                    wav.setpos(0)  # Reset position to start
                    binary_data = wav.readframes(num_frames)
                    audio_hash = hashlib.sha256(binary_data).hexdigest()

                # Get audio MIME type
                mime_type, _ = mimetypes.guess_type(file_path)
                extension = Path(file_path).suffix.lstrip(".")
                mime = mime_type or f"audio/{extension}"

                return Metadata(
                    uris=file_path,
                    mime=mime,
                    width=0,
                    height=0,
                    channels=channels,
                    depth=depth,
                    hash=audio_hash,
                    size=Path(file_path).stat().st_size,
                    has_audio=True,
                    duration=duration,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    blob=binary_data if save_binary else b"",
                )
            else:
                console.print(
                    f"[yellow]Unsupported file format for {file_path}[/yellow]"
                )
                return None
        except FileNotFoundError:
            console.print(f"[red]File not found: {file_path}[/red]")
            return None
        except IOError as e:
            console.print(f"[red]IO error reading file {file_path}: {e}[/red]")
            return None
        except SyntaxError as e:
            console.print(f"[red]Invalid file format for {file_path}: {e}[/red]")
            return None
        except Exception as e:
            console.print(f"[red]Unexpected error processing {file_path}: {e}[/red]")
            return None


def load_data(
    datasets_dir: str, texts_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Load image and caption data from directories.

    Args:
        datasets_dir: Directory containing images or videos
        texts_dir: Optional directory containing caption text files

    Returns:
        List of image-caption pairs
    """
    data = []

    if texts_dir:
        # Paired directory structure
        for file in Path(datasets_dir).iterdir():
            if not file.is_file() or not any(
                str(file).endswith(ext)
                for ext in (image_extensions + video_extensions + audio_extensions)
            ):
                continue

            text_file = file.stem + ".txt"
            text_path = Path(texts_dir) / text_file

            if not text_path.exists():
                continue

            with open(text_path, "r", encoding="utf-8") as file:
                caption = file.read().splitlines()

            data.append({"file_path": str(file), "caption": caption})
    else:
        # Single directory structure
        for file_path in Path(datasets_dir).rglob("*"):
            if not file_path.is_file() or not any(
                str(file_path).endswith(ext)
                for ext in (image_extensions + video_extensions + audio_extensions)
            ):
                continue

            text_path = file_path.with_suffix(".txt")

            caption = []
            if text_path.exists():
                with open(text_path, "r", encoding="utf-8") as file:
                    caption = file.read().splitlines()

            data.append({"file_path": str(file_path), "caption": caption})

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

    with Progress() as progress:
        task = progress.add_task("[green]Processing file...", total=len(data))

        for item in data:
            file_path = item["file_path"]
            caption = item["caption"]

            console.print(f"Processing file [yellow]'{file_path}'...", style="yellow")
            console.print(f"Caption: {caption}", style="cyan")

            metadata = processor.load_metadata(file_path, save_binary, import_mode)
            if not metadata:
                progress.update(task, advance=1)
                continue

            # Get field names and create arrays
            field_names = [field[0] for field in DATASET_SCHEMA]
            arrays = []
            for field_name in field_names:
                if field_name == "filepath":
                    value = str(Path(file_path).resolve())
                elif field_name == "captions":
                    value = caption
                else:
                    value = getattr(metadata, field_name)
                arrays.append(pa.array([value]))

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
    tag: str = "WDtagger",
    load_condition: Callable[[str, Optional[str]], List[Dict[str, Any]]] = load_data,
) -> Optional[lance.LanceDataset]:
    """
    Transform image-caption pairs into Lance dataset.

    Args:
        dataset_dir: Directory containing training images
        caption_dir: Optional directory containing captions
        output_name: Name of output dataset
        save_binary: Whether to save binary data in the dataset.
        not_save_disk: If True, don't save to disk
        import_mode: Mode for importing video components:
                    - ALL: Complete video with audio
                    - VIDEO_ONLY: Video without audio
                    - AUDIO_ONLY: Audio only
                    - VIDEO_SPLIT_AUDIO: Split video and audio
        load_condition: Function to load data

    Returns:
        Lance dataset object or None if error occurs
    """
    data = load_condition(dataset_dir, caption_dir)

    schema = pa.schema([pa.field(name, pa_type) for name, pa_type in DATASET_SCHEMA])

    try:
        reader = pa.RecordBatchReader.from_batches(
            schema, process(data, save_binary, import_mode)
        )

        if not_save_disk:
            table = reader.read_all()
            return lance.dataset(table)

        dataset_path = Path(dataset_dir) / f"{output_name}.lance"
        lancedataset = lance.write_dataset(
            reader, str(dataset_path), schema, mode="overwrite"
        )

        try:
            lancedataset.tags.create(tag, 1)
        except:
            lancedataset.tags.update(tag, 1)

        return lancedataset

    except AttributeError as e:
        console.print(f"[red]AttributeError: {e}[/red]")
        return None


def setup_parser() -> argparse.ArgumentParser:
    """Setup argument parser."""
    parser = argparse.ArgumentParser(description="Transform dataset into Lance format")
    parser.add_argument(
        "dataset_dir", type=str, help="Directory containing training images"
    )
    parser.add_argument(
        "--caption_dir",
        type=str,
        default=None,
        help="Directory containing caption files",
    )
    parser.add_argument(
        "--output_name", type=str, default="dataset", help="Name of output dataset"
    )
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
        help="Video import mode: 0=Complete video with audio, 1=Video only, "
        "2=Audio only, 3=Split video and audio",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="WDtagger",
        help="Tag for the dataset",
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
    )
