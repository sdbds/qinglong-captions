from __future__ import annotations

import json

import pyarrow as pa


def update_dataset_captions(dataset, processed_filepaths, results, merge_batch_size: int, console, tag_name: str = "gemini") -> None:
    if not results:
        return

    processed_captions = []
    for caption in results:
        if isinstance(caption, list):
            processed_captions.append("\n".join(caption))
        elif isinstance(caption, dict):
            processed_captions.append(json.dumps(caption, ensure_ascii=False))
        else:
            processed_captions.append(caption)

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

    try:
        dataset.tags.create(tag_name, 1)
    except Exception:
        dataset.tags.update(tag_name, 1)

    console.print("[green]Successfully updated dataset with new captions[/green]")
