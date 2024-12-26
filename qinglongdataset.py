"""
PyTorch Dataset for accessing Lance-format image-caption data.
"""

import io
from typing import Optional, List, Dict, Any, Callable, Union
from pathlib import Path
import lance
from PIL import Image
from torch.utils.data import Dataset
from rich.console import Console


console = Console()


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
        """Load image blob from dataset."""
        raw_data = self.ds.take([idx], columns=["blob"]).to_pydict()
        if not raw_data["blob"][0]:
            return None

        try:
            with Image.open(io.BytesIO(raw_data["blob"][0])) as img:
                img = img.convert("RGB") if img.mode != "RGB" else img
                return self.transform(img) if self.transform else img
        except Exception as e:
            console.print(f"[red]Error loading image: {e}[/red]")
            return None

    def load_path(self, idx: int) -> str:
        """Load file path from dataset."""
        path = self.ds.take([idx], columns=["filepath"]).to_pydict()["filepath"][0]
        return str(Path(path))

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
            "blob": self.load_data(idx),
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

    if all(item["blob"] is not None for item in batch):
        result["blob"] = [item["blob"] for item in batch]

    return result
