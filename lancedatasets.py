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

import lance
import pyarrow as pa
from PIL import Image
from torch.utils.data import Dataset
from rich.progress import Progress
from rich.console import Console

from config import get_supported_extensions, DATASET_SCHEMA


console = Console()


@dataclass
class ImageMetadata:
    """Metadata for an image file."""

    width: int
    height: int
    format: str
    size: int
    binary_data: bytes
    hash: str


class ImageProcessor:
    """Utility class for processing images.

    This class provides methods for loading and processing image metadata,
    including size, format, and hash calculations.

    Example:
        processor = ImageProcessor()
        metadata = processor.load_image_metadata("path/to/image.jpg")
        if metadata:
            print(f"Image size: {metadata.width}x{metadata.height}")
    """

    @staticmethod
    def load_image_metadata(
        image_path: str, save_binary: bool = True
    ) -> Optional[ImageMetadata]:
        """Load and process image metadata.

        Args:
            image_path: Path to the image file
            save_binary: If True, store the binary data of the image

        Returns:
            ImageMetadata object if successful, None if failed

        Raises:
            FileNotFoundError: If the image file doesn't exist
            IOError: If there's an error reading the file
            SyntaxError: If the image format is invalid
        """
        try:
            with Image.open(image_path) as img:
                buffer = io.BytesIO()
                img.save(buffer, format=img.format)
                binary_data = buffer.getvalue()
                image_hash = hashlib.sha256(binary_data).hexdigest()

                return ImageMetadata(
                    width=img.size[0],
                    height=img.size[1],
                    format=img.format,
                    size=os.path.getsize(image_path),
                    binary_data=binary_data if save_binary else b"",
                    hash=image_hash,
                )
        except FileNotFoundError:
            console.print(f"[red]Image file not found: {image_path}[/red]")
            return None
        except IOError as e:
            console.print(f"[red]IO error reading image {image_path}: {e}[/red]")
            return None
        except SyntaxError as e:
            console.print(f"[red]Invalid image format for {image_path}: {e}[/red]")
            return None
        except Exception as e:
            console.print(f"[red]Unexpected error processing {image_path}: {e}[/red]")
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

    def load_image(self, idx: int) -> Optional[Image.Image]:
        """Load image data from dataset."""
        raw_img = self.ds.take([idx], columns=["data"]).to_pydict()
        if not raw_img["data"][0]:
            return None

        try:
            with Image.open(io.BytesIO(raw_img["data"][0])) as img:
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
            "image": self.load_image(idx),
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

    if all(item["image"] is not None for item in batch):
        result["image"] = [item["image"] for item in batch]

    return result


def load_data(images_dir: str, texts_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load image and caption data from directories.

    Args:
        images_dir: Directory containing images
        texts_dir: Optional directory containing caption text files

    Returns:
        List of image-caption pairs
    """
    data = []
    image_extensions = get_supported_extensions()

    if texts_dir:
        # Paired directory structure
        for image_file in os.listdir(images_dir):
            if not image_file.endswith(image_extensions):
                continue

            text_file = os.path.splitext(image_file)[0] + ".txt"
            text_path = os.path.join(texts_dir, text_file)

            if not os.path.exists(text_path):
                continue

            with open(text_path, "r", encoding="utf-8") as file:
                caption = file.read().splitlines()

            data.append(
                {"image_path": os.path.join(images_dir, image_file), "caption": caption}
            )
    else:
        # Single directory structure
        for root, _, files in os.walk(images_dir):
            for image_file in files:
                if not image_file.endswith(image_extensions):
                    continue

                image_path = os.path.join(root, image_file)
                text_path = os.path.splitext(image_path)[0] + ".txt"

                caption = []
                if os.path.exists(text_path):
                    with open(text_path, "r", encoding="utf-8") as file:
                        caption = file.read().splitlines()

                data.append({"image_path": image_path, "caption": caption})

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
    processor = ImageProcessor()

    with Progress() as progress:
        task = progress.add_task("[green]Processing images...", total=len(data))

        for item in data:
            image_path = item["image_path"]
            caption = item["caption"]

            console.print(f"Processing image [yellow]'{image_path}'...", style="yellow")
            console.print(f"Caption: {caption}", style="cyan")

            metadata = processor.load_image_metadata(image_path, not only_save_path)
            if not metadata:
                progress.update(task, advance=1)
                continue

            batch = pa.RecordBatch.from_arrays(
                [
                    pa.array([os.path.realpath(image_path)]),
                    pa.array([metadata.format]),
                    pa.array([metadata.hash]),
                    pa.array([metadata.size]),
                    pa.array([metadata.width]),
                    pa.array([metadata.height]),
                    pa.array([metadata.binary_data]),
                    pa.array([caption]),
                ],
                names=[field[0] for field in DATASET_SCHEMA],
            )

            yield batch
            progress.update(task, advance=1)


def transform2lance(
    train_data_dir: str,
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
        train_data_dir: Directory containing training images
        caption_dir: Optional directory containing captions
        output_name: Name of output dataset
        only_save_path: If True, only save file paths
        not_save_disk: If True, don't save to disk
        load_condition: Function to load data

    Returns:
        Lance dataset object or None if error occurs
    """
    data = load_condition(train_data_dir, caption_dir)

    schema = pa.schema([pa.field(name, pa_type) for name, pa_type in DATASET_SCHEMA])

    try:
        reader = pa.RecordBatchReader.from_batches(
            schema, process(data, only_save_path)
        )

        if not_save_disk:
            table = reader.read_all()
            return lance.dataset(table)

        dataset_path = os.path.join(train_data_dir, f"{output_name}.lance")
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
        "train_data_dir", type=str, help="Directory containing training images"
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
        train_data_dir=args.train_data_dir,
        caption_dir=args.caption_dir,
        output_name=args.output_name,
        only_save_path=args.only_save_path,
        not_save_disk=args.not_save_disk,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
