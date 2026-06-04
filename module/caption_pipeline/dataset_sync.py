from __future__ import annotations

import json
from typing import Any

import pyarrow as pa

from module.providers.base import CaptionResult
from utils.console_util import print_exception
from utils.lance_rebuild import rebuild_lance_from_sidecars, resolve_lance_rebuild_source
from utils.lance_utils import update_or_create_tag


def _normalize_dataset_captions(results) -> list[Any]:
    processed_captions = []
    for caption in results:
        if isinstance(caption, CaptionResult):
            processed_captions.append(caption.to_dataset_caption())
        elif isinstance(caption, list):
            processed_captions.append("\n".join(caption))
        elif isinstance(caption, dict):
            processed_captions.append(json.dumps(caption, ensure_ascii=False))
        else:
            processed_captions.append(caption)
    return processed_captions


def _merge_caption_batches(dataset, processed_filepaths, processed_captions, merge_batch_size: int, console) -> None:
    total_items = len(processed_filepaths)
    for batch_start in range(0, total_items, merge_batch_size):
        batch_end = min(batch_start + merge_batch_size, total_items)
        table = pa.table(
            {
                "uris": pa.array(processed_filepaths[batch_start:batch_end], type=pa.string()),
                "captions": pa.array(
                    [[caption] for caption in processed_captions[batch_start:batch_end]],
                    type=pa.list_(pa.string()),
                ),
            }
        )
        dataset.merge_insert(on="uris").when_matched_update_all().execute(table)

        if total_items > merge_batch_size:
            current_batch = batch_start // merge_batch_size + 1
            total_batches = (total_items + merge_batch_size - 1) // merge_batch_size
            console.print(f"[cyan]Merged batch {current_batch}/{total_batches}[/cyan]")


def _rebuild_after_merge_failure(
    dataset,
    *,
    dataset_dir,
    dataset_path,
    transform2lance_fn,
    load_data_fn,
    console,
    tag_name: str,
    merge_error: Exception,
):
    source_dir = resolve_lance_rebuild_source(dataset_dir, dataset_path)
    if source_dir is None or transform2lance_fn is None or load_data_fn is None:
        raise RuntimeError("Lance merge_insert failed and fallback rebuild is unavailable") from merge_error

    try:
        rebuilt_dataset = rebuild_lance_from_sidecars(
            source_dir,
            output_name="dataset",
            dataset=dataset,
            tag=tag_name,
            transform2lance_fn=transform2lance_fn,
            load_data_fn=load_data_fn,
            console=console,
            caption_extension=None,
        )
    except Exception as rebuild_error:
        print_exception(console, rebuild_error, prefix="Lance fallback rebuild failed", summary_style="yellow")
        raise RuntimeError(
            f"Lance merge_insert failed and fallback rebuild also failed: {rebuild_error}"
        ) from merge_error

    if rebuilt_dataset is None:
        raise RuntimeError("Lance merge_insert failed and fallback rebuild is unavailable") from merge_error
    return rebuilt_dataset


def update_dataset_captions(
    dataset,
    processed_filepaths,
    results,
    merge_batch_size: int,
    console,
    tag_name: str = "gemini",
    *,
    dataset_dir=None,
    dataset_path=None,
    transform2lance_fn=None,
    load_data_fn=None,
):
    if not results:
        return dataset

    processed_captions = _normalize_dataset_captions(results)

    try:
        _merge_caption_batches(dataset, processed_filepaths, processed_captions, merge_batch_size, console)
        update_or_create_tag(dataset, tag_name)
    except Exception as merge_error:
        console.print("[yellow]Lance merge_insert failed; rebuilding dataset from sidecar caption files.[/yellow]")
        print_exception(console, merge_error, prefix="Lance merge_insert failed", summary_style="yellow")
        return _rebuild_after_merge_failure(
            dataset,
            dataset_dir=dataset_dir,
            dataset_path=dataset_path,
            transform2lance_fn=transform2lance_fn,
            load_data_fn=load_data_fn,
            console=console,
            tag_name=tag_name,
            merge_error=merge_error,
        )

    console.print("[green]Successfully updated dataset with new captions[/green]")
    return dataset
