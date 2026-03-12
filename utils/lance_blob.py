from __future__ import annotations

import inspect
from typing import Any, Iterable, Optional


def _resolve_first_arg_name(method: Any) -> str:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return "blob_column"
    params = list(signature.parameters)
    return params[0] if params else "blob_column"


def take_blob_files(dataset: Any, ids: Iterable[int], blob_column: str = "blob") -> list[Optional[Any]]:
    method = dataset.take_blobs
    row_ids = list(ids)
    first_arg = _resolve_first_arg_name(method)
    if first_arg == "blob_column":
        try:
            return list(method(blob_column, ids=row_ids))
        except TypeError:
            pass
    return list(method(row_ids, blob_column))
