from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import lance
import pyarrow as pa

from module.lanceImport import load_data, transform2lance
from module.wdtagger import constants
from module.wdtagger.outputs import has_sidecar_caption, read_sidecar_caption
from utils.lance_utils import update_or_create_tag


@dataclass
class WDTaggerDatasetRef:
    dataset: Any
    dataset_path: Optional[Path]


def resolve_dataset(train_data_dir: Any) -> WDTaggerDatasetRef:
    dataset_path: Optional[Path] = None
    lance_dataset_type = getattr(lance, "LanceDataset", None)
    if lance_dataset_type is not None and isinstance(train_data_dir, lance_dataset_type):
        constants.console.print("[green]Using existing Lance dataset[/green]")
        return WDTaggerDatasetRef(train_data_dir, None)

    if str(train_data_dir).endswith(".lance"):
        dataset_path = Path(train_data_dir)
        return WDTaggerDatasetRef(lance.dataset(str(train_data_dir)), dataset_path)

    train_path = Path(train_data_dir)
    if any(file.suffix == ".lance" for file in train_path.glob("*")):
        lance_file = next(file for file in train_path.glob("*") if file.suffix == ".lance")
        dataset_path = lance_file
        return WDTaggerDatasetRef(lance.dataset(str(lance_file)), dataset_path)

    constants.console.print("[yellow]Converting dataset to Lance format...[/yellow]")
    dataset = transform2lance(
        str(train_data_dir),
        output_name="dataset",
        save_binary=False,
        not_save_disk=False,
        tag="WDtagger",
    )
    dataset_path = Path(train_data_dir) / "dataset.lance"
    constants.console.print("[green]Dataset converted to Lance format[/green]")
    return WDTaggerDatasetRef(dataset, dataset_path)


def is_empty_caption_list(captions: Optional[List[str]]) -> bool:
    return captions is None or len(captions) == 0


def filter_uncaptioned_batch(batch: pa.RecordBatch, args: argparse.Namespace) -> Optional[pa.RecordBatch]:
    captions = batch["captions"].to_pylist()
    uris = batch["uris"].to_pylist()
    should_skip_sidecars = not getattr(args, "append_tags", False)
    caption_extension = getattr(args, "caption_extension", ".txt")
    indices = [
        index
        for index, value in enumerate(captions)
        if is_empty_caption_list(value)
        and not (should_skip_sidecars and has_sidecar_caption(str(uris[index]), caption_extension))
    ]

    if len(indices) == batch.num_rows:
        return batch
    if not indices:
        return None

    return batch.take(pa.array(indices, type=pa.int64()))


def scan_wdtagger_candidate_batches(dataset: Any, args: argparse.Namespace) -> Iterator[pa.RecordBatch]:
    columns = ["uris", "mime"] if args.overwrite else ["uris", "mime", "captions"]
    scanner = dataset.scanner(
        columns=columns,
        filter=constants.IMAGE_MIME_FILTER,
        scan_in_order=True,
        batch_size=args.batch_size,
        batch_readahead=16,
        fragment_readahead=4,
        io_buffer_size=32 * 1024 * 1024,
        late_materialization=False,
    )

    for batch in scanner.to_batches():
        if args.overwrite:
            yield batch
            continue

        filtered_batch = filter_uncaptioned_batch(batch, args)
        if filtered_batch is not None and filtered_batch.num_rows > 0:
            yield filtered_batch


def count_wdtagger_candidate_rows(dataset: Any, args: argparse.Namespace) -> int:
    return sum(batch.num_rows for batch in scan_wdtagger_candidate_batches(dataset, args))


def resolve_lance_rebuild_source(train_data_dir: Any, dataset_path: Optional[Path]) -> Optional[Path]:
    if not isinstance(train_data_dir, str):
        return None

    input_path = Path(train_data_dir)
    if input_path.suffix == ".lance":
        return input_path.parent
    if input_path.is_dir():
        return input_path
    if dataset_path is not None:
        return dataset_path.parent
    return None


def load_rebuild_data(source_dir: Optional[Path], dataset: Any, caption_extension: str) -> List[Dict[str, Any]]:
    if source_dir is not None:
        data = load_data(str(source_dir))
        if data:
            return data

    if dataset is None:
        return []

    data = []
    scanner = dataset.scanner(
        columns=["uris"],
        scan_in_order=True,
        batch_size=1024,
        late_materialization=False,
    )
    for batch in scanner.to_batches():
        for uri in batch["uris"].to_pylist():
            data.append(
                {
                    "file_path": str(uri),
                    "caption": read_sidecar_caption(str(uri), caption_extension),
                    "chunk_offsets": [],
                }
            )
    return data


def rebuild_lance_from_sidecars(
    source_dir: Optional[Path],
    output_name: str,
    dataset: Any,
    caption_extension: str,
) -> Optional[Any]:
    if source_dir is None:
        constants.console.print("[yellow]Skipping Lance rebuild: source directory is unavailable.[/yellow]")
        return None

    data = load_rebuild_data(source_dir, dataset, caption_extension)
    if not data:
        constants.console.print("[yellow]Skipping Lance rebuild: no source media rows were found.[/yellow]")
        return None

    constants.console.print("[yellow]Rebuilding Lance dataset from sidecar caption files...[/yellow]")
    rebuilt_dataset = transform2lance(
        str(source_dir),
        output_name=output_name,
        save_binary=False,
        not_save_disk=False,
        tag="WDtagger",
        load_condition=lambda *_args, **_kwargs: data,
    )
    if rebuilt_dataset is None:
        constants.console.print("[yellow]Lance rebuild did not return a dataset; sidecar captions were still written.[/yellow]")
    else:
        constants.console.print("[green]Lance dataset rebuilt from sidecar caption files[/green]")
    return rebuilt_dataset


def merge_caption_updates(dataset: Any, results: Sequence[Tuple[str, List[str]]]) -> None:
    table = pa.table(
        {
            "uris": pa.array([str(path) for path, _ in results], type=pa.string()),
            "captions": pa.array(
                [caption for _, caption in results],
                type=pa.list_(pa.string()),
            ),
        }
    )
    dataset.merge_insert(on="uris").when_matched_update_all().execute(table)


def update_wdtagger_tag(dataset: Any) -> None:
    update_or_create_tag(dataset, "WDtagger")
