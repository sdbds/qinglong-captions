"""
Dataset processing utilities for image-caption pairs using Lance format.
This module provides tools for converting image-caption datasets to Lance format
and accessing the data through PyTorch datasets.
"""

import argparse
import os
import hashlib
import io
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable, Tuple, Union
import wave
from moviepy.video.io.VideoFileClip import VideoFileClip
import lance
import pyarrow as pa
from PIL import Image
from torch.utils.data import Dataset
from rich.progress import Progress
from rich.console import Console
import mimetypes

from config import get_supported_extensions, DATASET_SCHEMA


console = Console()
image_extensions = get_supported_extensions("image")
video_extensions = get_supported_extensions("video")
audio_extensions = get_supported_extensions("audio")


@dataclass
class Metadata:
    """Metadata for media file."""

    uris: str
    width: int
    height: int
    format: str
    channels: int = 0  # Number of channels (RGB=3, RGBA=4, mono=1, stereo=2)
    size: int
    blob: bytes
    hash: str
    has_audio: bool = False
    duration: int = 0  # Duration in milliseconds
    num_frames: int = 0
    depth: int = 0  # Sample depth/width in bits

    @property
    def filename(self) -> str:
        """File name without extension, derived from filepath."""
        return os.path.splitext(os.path.basename(self.uris))[0]

    @property
    def extension(self) -> str:
        """File extension derived from filepath, including dot."""
        return os.path.splitext(self.uris)[1]

    @property
    def frame_rate(self) -> float:
        """Calculate frames per second or samples per second."""
        if self.duration > 0:
            return self.num_frames * 1000 / self.duration  # Convert ms to seconds
        return 0.0

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

    @staticmethod
    def load_metadata(file_path: str, save_binary: bool = True) -> Optional[Metadata]:
        """Load and process image metadata.

        Args:
            file_path: Path to the image file
            save_binary: If True, store the binary data of the image

        Returns:
            Metadata object if successful, None if failed

        Raises:
            FileNotFoundError: If the image file doesn't exist
            IOError: If there's an error reading the file
            SyntaxError: If the image format is invalid
        """
        try:
            if file_path.endswith(image_extensions):
                with Image.open(file_path) as img:
                    buffer = io.BytesIO()
                    img.save(buffer, format=img.format)
                    blob = buffer.getvalue()
                    image_hash = hashlib.sha256(blob).hexdigest()

                    # Get image MIME type, fallback to PIL format
                    mime_type, _ = mimetypes.guess_type(file_path)
                    format = mime_type or img.format

                    # Get depth based on mode type
                    channels = len(img.getbands())
                    mode = img.mode
                    if hasattr(img, "bits"):
                        depth = img.bits  # Use actual bit depth if available
                    elif mode == "1":
                        depth = 1
                    elif mode in ["I", "F"]:
                        depth = 32
                    else:
                        depth = 8

                    return Metadata(
                        uris=file_path,
                        width=img.size[0],
                        height=img.size[1],
                        format=format,
                        channels=channels,
                        size=os.path.getsize(file_path),
                        blob=blob if save_binary else b"",
                        hash=image_hash,
                        depth=depth,
                    )
            elif file_path.endswith(video_extensions):
                video = VideoFileClip(
                    file_path, audio=True
                )  # Explicitly enable audio loading
                try:
                    # Get video info
                    width, height = video.size
                    fps = video.fps
                    duration = int(video.duration * 1000)  # Convert to milliseconds
                    n_frames = int(fps * video.duration)  # Calculate total frames

                    # Get color info from first frame
                    first_frame = video.get_frame(0)  # Returns numpy array
                    channels = first_frame.shape[2] if len(first_frame.shape) > 2 else 1
                    depth = first_frame.dtype.itemsize * 8  # Get actual bit depth

                    # Get video MIME type
                    mime_type, _ = mimetypes.guess_type(file_path)
                    extension = os.path.splitext(file_path)[1].lstrip(".")
                    format = mime_type or extension

                    # Read video binary data
                    with open(file_path, "rb") as f:
                        blob = f.read()
                        video_hash = hashlib.sha256(blob).hexdigest()

                    return Metadata(
                        uris=file_path,
                        width=width,
                        height=height,
                        format=format,
                        channels=channels,
                        size=os.path.getsize(file_path),
                        blob=blob if save_binary else b"",
                        hash=video_hash,
                        has_audio=video.audio is not None,
                        duration=duration,
                        num_frames=n_frames,
                        depth=depth,
                    )
                finally:
                    video.close()  # Ensure video file is properly closed
            elif file_path.endswith(audio_extensions):
                with wave.open(file_path, "rb") as wav:
                    frame_rate = wav.getframerate()
                    num_frames = wav.getnframes()
                    duration = int(
                        num_frames * 1000 / frame_rate
                    )  # Convert to milliseconds
                    channels = wav.getnchannels()
                    depth = wav.getsampwidth() * 8  # Convert bytes to bits

                    # Get audio data
                    blob = wav.readframes(num_frames)
                    audio_hash = hashlib.sha256(blob).hexdigest()

                # Get audio MIME type
                mime_type, _ = mimetypes.guess_type(file_path)
                extension = os.path.splitext(file_path)[1].lstrip(".")
                format = mime_type or extension

                return Metadata(
                    uris=file_path,
                    width=0,
                    height=0,
                    format=format,
                    channels=channels,
                    size=os.path.getsize(file_path),
                    blob=blob if save_binary else b"",
                    hash=audio_hash,
                    has_audio=True,
                    duration=duration,
                    num_frames=num_frames,
                    depth=depth,
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


class QingLongDataset(Dataset):
    """PyTorch Dataset for accessing Lance-format image-caption data."""

    def __init__(
        self,
        lance_or_path: Union[str, lance.LanceDataset],
        max_len: int = 225,
        tokenizers: Optional[List[Any]] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the dataset.

        Args:
            lance_or_path: Path to Lance dataset or Lance dataset object
            max_len: Maximum length for tokenization
            tokenizers: List of tokenizers to apply
            transform: Transform to apply to images
        """
        self.ds = (
            lance.dataset(lance_or_path)
            if isinstance(lance_or_path, str)
            else lance_or_path
        )
        self.max_len = max_len
        self.tokenizers = tokenizers
        self.transform = transform

    def __len__(self) -> int:
        return self.ds.count_rows()

    def load_data(self, idx: int) -> Optional[Image.Image]:
        """Load image data from dataset."""
        raw_data = self.ds.take([idx], columns=["data"]).to_pydict()
        if not raw_data["data"][0]:
            return None

        try:
            with Image.open(io.BytesIO(raw_data["data"][0])) as img:
                img = img.convert("RGB") if img.mode != "RGB" else img
                return self.transform(img) if self.transform else img
        except Exception as e:
            console.print(f"[red]Error loading image: {e}[/red]")
            return None

    def load_path(self, idx: int) -> str:
        """Load file path from dataset."""
        return self.ds.take([idx], columns=["filepath"]).to_pydict()["filepath"][0]

    def load_caption(self, idx: int) -> List[str]:
        """Load caption from dataset."""
        return self.ds.take([idx], columns=["captions"]).to_pydict()["captions"][0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        caption = self.load_caption(idx)
        # if self.tokenizers:
        #     for i, tokenizer in enumerate(self.tokenizers):
        #         caption = tokenizer(
        #             caption,
        #             max_length=self.max_len[i],
        #             padding="max_length",
        #             truncation=True,
        #             return_tensors="pt",
        #         )
        #     # Flatten each component of tokenized caption otherwise they will cause size mismatch errors during training
        #     caption = {k: v.flatten() for k, v in caption.items()}
        return {
            "filepath": self.load_path(idx),
            "data": self.load_data(idx),
            "captions": caption,
        }


def collate_fn_remove_corrupted(
    batch: List[Optional[Dict[str, Any]]]
) -> Optional[Dict[str, List[Any]]]:
    """
    Collate function that removes corrupted examples from the batch.

    Args:
        batch: List of dataset items

    Returns:
        Collated batch with corrupted items removed
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    result = {"filepath": [], "captions": []}

    for item in batch:
        result["filepath"].append(item["filepath"])
        result["captions"].append(item["captions"])

    if all(item["data"] is not None for item in batch):
        result["data"] = [item["data"] for item in batch]

    return result


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
        for file in os.listdir(datasets_dir):
            if not file.endswith(
                image_extensions + video_extensions + audio_extensions
            ):
                continue

            text_file = os.path.splitext(file)[0] + ".txt"
            text_path = os.path.join(texts_dir, text_file)

            if not os.path.exists(text_path):
                continue

            with open(text_path, "r", encoding="utf-8") as file:
                caption = file.read().splitlines()

            data.append(
                {"file_path": os.path.join(datasets_dir, file), "caption": caption}
            )
    else:
        # Single directory structure
        for root, _, files in os.walk(datasets_dir):
            for file in files:
                if not file.endswith(
                    image_extensions + video_extensions + audio_extensions
                ):
                    continue

                file_path = os.path.join(root, file)
                text_path = os.path.splitext(file_path)[0] + ".txt"

                caption = []
                if os.path.exists(text_path):
                    with open(text_path, "r", encoding="utf-8") as file:
                        caption = file.read().splitlines()

                data.append({"file_path": file_path, "caption": caption})

    return data


def process(data: List[Dict[str, Any]], only_save_path: bool = False) -> pa.RecordBatch:
    """
    Process image-caption pairs into Lance format.

    Args:
        data: List of image-caption pairs
        only_save_path: If True, don't store image binary data

    Yields:
        PyArrow RecordBatch objects
    """
    processor = FileProcessor()

    with Progress() as progress:
        task = progress.add_task("[green]Processing file...", total=len(data))

        for item in data:
            file_path = item["file_path"]
            caption = item["caption"]

            console.print(f"Processing file [yellow]'{file_path}'...", style="yellow")
            console.print(f"Caption: {caption}", style="cyan")

            metadata = processor.load_metadata(file_path, not only_save_path)
            if not metadata:
                progress.update(task, advance=1)
                continue

            batch = pa.RecordBatch.from_arrays(
                [
                    pa.array([os.path.realpath(file_path)]),
                    pa.array([metadata.format]),
                    pa.array([metadata.width]),
                    pa.array([metadata.height]),
                    pa.array([metadata.channels]),
                    pa.array([metadata.hash]),
                    pa.array([metadata.size]),
                    pa.array([metadata.has_audio]),
                    pa.array([metadata.duration]),
                    pa.array([metadata.num_frames]),
                    pa.array([metadata.depth]),
                    pa.array([metadata.blob]),
                    pa.array([caption]),
                ],
                names=[field[0] for field in DATASET_SCHEMA],
            )

            yield batch
            progress.update(task, advance=1)


def transform2lance(
    dataset_dir: str,
    caption_dir: Optional[str] = None,
    output_name: str = "dataset",
    only_save_path: bool = False,
    not_save_disk: bool = False,
    tag: str = "WDtagger",
    load_condition: Callable[[str, Optional[str]], List[Dict[str, Any]]] = load_data,
) -> Optional[lance.LanceDataset]:
    """
    Transform image data into Lance dataset format.

    Args:
        dataset_dir: Directory containing training images
        caption_dir: Optional directory containing captions
        output_name: Name of output dataset
        only_save_path: If True, only save file paths
        not_save_disk: If True, don't save to disk
        load_condition: Function to load data

    Returns:
        Lance dataset object or None if error occurs
    """
    data = load_condition(dataset_dir, caption_dir)

    schema = pa.schema([pa.field(name, pa_type) for name, pa_type in DATASET_SCHEMA])

    try:
        reader = pa.RecordBatchReader.from_batches(
            schema, process(data, only_save_path)
        )

        if not_save_disk:
            table = reader.read_all()
            return lance.dataset(table)

        dataset_path = os.path.join(dataset_dir, f"{output_name}.lance")
        lancedataset = lance.write_dataset(
            reader, dataset_path, schema, mode="overwrite"
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
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert image-caption dataset to Lance format"
    )

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
        "--only_save_path", action="store_true", help="Only save image file paths"
    )
    parser.add_argument(
        "--not_save_disk",
        action="store_true",
        help="Load dataset into memory instead of saving to disk",
    )

    parser.add_argument(
        "--tag", type=str, default="WDtagger", help="Tag for the dataset"
    )

    return parser


def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    transform2lance(
        dataset_dir=args.dataset_dir,
        caption_dir=args.caption_dir,
        output_name=args.output_name,
        only_save_path=args.only_save_path,
        not_save_disk=args.not_save_disk,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
