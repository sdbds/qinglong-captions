from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from module.providers.base import CaptionResult
from utils.lance_updates import LanceRowUpdate, merge_rows_preserving_schema
from utils.lance_utils import update_or_create_tag


@dataclass(frozen=True)
class CaptionUpdate:
    uri: str
    caption: str


def _serialize_legacy_caption(value: Any) -> str | None:
    if isinstance(value, CaptionResult):
        return value.to_dataset_caption() if value.is_persistable else None
    if isinstance(value, dict):
        result = CaptionResult(raw=json.dumps(value, ensure_ascii=False), parsed=value)
        return result.to_dataset_caption() if result.is_persistable else None
    if isinstance(value, list):
        caption = "\n".join(str(item) for item in value)
        return caption if caption.strip() else None
    caption = str(value or "")
    return caption if caption.strip() else None


def _coerce_caption_updates(
    updates_or_filepaths: Sequence[CaptionUpdate] | Sequence[str],
    results: Sequence[Any] | None,
) -> list[CaptionUpdate]:
    if results is None:
        updates = list(updates_or_filepaths)
        if not all(isinstance(update, CaptionUpdate) for update in updates):
            raise TypeError("Canonical caption updates must be CaptionUpdate instances")
        return updates

    filepaths = list(updates_or_filepaths)
    result_values = list(results)
    if len(filepaths) != len(result_values):
        raise ValueError("Caption URI and result counts must match")

    updates: list[CaptionUpdate] = []
    for uri, result in zip(filepaths, result_values):
        caption = _serialize_legacy_caption(result)
        if caption is not None:
            updates.append(CaptionUpdate(uri=str(uri), caption=caption))
    return updates


def update_dataset_captions(
    dataset,
    updates_or_filepaths,
    results=None,
    merge_batch_size: int = 100,
    console=None,
    tag_name: str = "gemini",
    **_legacy_rebuild_options,
):
    """Persist paired caption updates without exposing partial rows to Lance."""
    updates = _coerce_caption_updates(updates_or_filepaths, results)
    if not updates:
        return dataset

    row_updates = [
        LanceRowUpdate(uri=update.uri, values={"captions": [update.caption]})
        for update in updates
    ]
    merge_rows_preserving_schema(dataset, row_updates, batch_size=merge_batch_size)
    update_or_create_tag(dataset, tag_name)

    if console is not None:
        console.print(f"[green]Persisted {len(updates)} caption update(s)[/green]")
    return dataset
