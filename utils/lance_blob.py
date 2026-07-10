from __future__ import annotations

import inspect
from typing import Any, Iterable, Optional

import pyarrow as pa


BLOB_V2_EXTENSION_NAME = "lance.blob.v2"
DEFAULT_BLOB_DATA_STORAGE_VERSION = "2.2"


def _blob_v2_api() -> tuple[Any, Any]:
    import lance

    return getattr(lance, "blob_field", None), getattr(lance, "blob_array", None)


def _uses_blob_v2(data_storage_version: str) -> bool:
    normalized = str(data_storage_version).lower()
    if normalized in {"stable", "next"}:
        return True
    if normalized in {"legacy", "0.1"}:
        return False
    parts = normalized.split(".")
    if len(parts) < 2:
        return False
    try:
        major, minor = int(parts[0]), int(parts[1])
    except ValueError:
        return False
    return (major, minor) >= (2, 2)


def build_lance_schema(
    dataset_schema: Iterable[tuple[str, pa.DataType]],
    *,
    data_storage_version: str = DEFAULT_BLOB_DATA_STORAGE_VERSION,
    blob_column: str = "blob",
) -> pa.Schema:
    """Build a Lance schema with Blob v2 for modern data storage versions."""
    blob_field, _ = _blob_v2_api()
    fields = []
    use_blob_v2 = _uses_blob_v2(data_storage_version)

    if use_blob_v2 and blob_field is None:
        raise RuntimeError("Lance Blob v2 requires a pylance build that exposes lance.blob_field")

    for name, pa_type in dataset_schema:
        if name != blob_column:
            fields.append(pa.field(name, pa_type))
            continue

        if use_blob_v2:
            fields.append(blob_field(name, nullable=True))
        else:
            fields.append(pa.field(name, pa_type, metadata={b"lance-encoding:blob": b"true"}))

    return pa.schema(fields)


def is_blob_v2_field(field: pa.Field) -> bool:
    return BLOB_V2_EXTENSION_NAME in str(field.type)


def build_blob_array(values: Iterable[Optional[bytes]], field: pa.Field) -> pa.Array:
    values = list(values)
    if is_blob_v2_field(field):
        _, blob_array = _blob_v2_api()
        if blob_array is None:
            raise RuntimeError("Lance Blob v2 requires a pylance build that exposes lance.blob_array")
        return blob_array(values)
    return pa.array(values, type=field.type)


def build_lance_value_array(values: Iterable[Any], field: pa.Field) -> pa.Array:
    if field.name == "blob" or is_blob_v2_field(field):
        return build_blob_array(values, field)
    return pa.array(list(values), type=field.type)


def _resolve_first_arg_name(method: Any) -> str:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return "blob_column"
    params = list(signature.parameters)
    return params[0] if params else "blob_column"


def _align_blob_files_by_indices(
    dataset: Any,
    blob_column: str,
    indices: list[int],
    blob_files: list[Optional[Any]],
) -> list[Optional[Any]]:
    if len(blob_files) == len(indices):
        return blob_files

    try:
        metadata = dataset.take(indices, columns=[blob_column]).column(blob_column).to_pylist()
    except Exception:
        if not blob_files:
            return [None] * len(indices)
        return blob_files

    blob_iter = iter(blob_files)
    aligned: list[Optional[Any]] = []
    for item in metadata:
        size = item.get("size", 0) if isinstance(item, dict) else 0
        aligned.append(next(blob_iter, None) if size else None)
    return aligned


def take_blob_files(
    dataset: Any,
    indices: Optional[Iterable[int]] = None,
    blob_column: str = "blob",
    *,
    ids: Optional[Iterable[int]] = None,
    addresses: Optional[Iterable[int]] = None,
) -> list[Optional[Any]]:
    method = dataset.take_blobs
    selector_values = {
        "indices": list(indices) if indices is not None else None,
        "ids": list(ids) if ids is not None else None,
        "addresses": list(addresses) if addresses is not None else None,
    }
    selectors = {key: value for key, value in selector_values.items() if value is not None}
    if len(selectors) != 1:
        raise ValueError("Exactly one of indices, ids, or addresses must be specified")

    first_arg = _resolve_first_arg_name(method)
    if first_arg == "blob_column":
        try:
            blob_files = list(method(blob_column, **selectors))
            if "indices" in selectors:
                return _align_blob_files_by_indices(dataset, blob_column, selectors["indices"], blob_files)
            return blob_files
        except TypeError:
            pass

    legacy_values = next(iter(selectors.values()))
    return list(method(legacy_values, blob_column))
