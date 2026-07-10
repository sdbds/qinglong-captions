from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc

from utils.lance_blob import build_lance_value_array, is_blob_v2_field, take_blob_files


class LanceUpdateError(RuntimeError):
    """Base error for schema-preserving Lance row updates."""


class LanceUpdateValidationError(LanceUpdateError):
    """The requested update cannot identify or construct valid target rows."""


class LanceUpdateConflictError(LanceUpdateError):
    """The update lost an optimistic-concurrency race."""


class LanceUpdateStorageError(LanceUpdateError):
    """Lance or Arrow failed while reading or committing the update."""


@dataclass(frozen=True)
class LanceRowUpdate:
    uri: str
    values: Mapping[str, Any]


def _chunks(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _filter_for_keys(key: str, values: Sequence[str], field: pa.Field) -> pc.Expression:
    try:
        return pc.field(key).isin(pa.array(values, type=field.type))
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError) as exc:
        raise LanceUpdateValidationError(f"Invalid {key} values: {exc}") from exc


def _scan_rows(dataset, *, columns: list[str], filter_expression: pc.Expression) -> list[dict[str, Any]]:
    try:
        scanner = dataset.scanner(columns=columns, filter=filter_expression, with_row_id=True)
        return scanner.to_table().to_pylist()
    except LanceUpdateError:
        raise
    except Exception as exc:
        raise LanceUpdateStorageError(f"Failed to scan Lance update targets: {exc}") from exc


def _validate_updates(dataset, updates: Sequence[LanceRowUpdate], key: str, batch_size: int) -> dict[str, int]:
    schema = dataset.schema
    if batch_size <= 0:
        raise LanceUpdateValidationError("batch_size must be greater than zero")
    if key not in schema.names:
        raise LanceUpdateValidationError(f"Target schema has no key column {key!r}")

    schema_names = set(schema.names)
    seen: set[str] = set()
    for update in updates:
        if update.uri in seen:
            raise LanceUpdateValidationError(f"Duplicate update key: {update.uri!r}")
        seen.add(update.uri)
        unknown = set(update.values) - schema_names
        if unknown:
            raise LanceUpdateValidationError(f"Unknown update columns: {sorted(unknown)!r}")
        if key in update.values and update.values[key] != update.uri:
            raise LanceUpdateValidationError(f"Update may not change key column {key!r}")

    key_field = schema.field(key)
    row_ids: dict[str, int] = {}
    for update_batch in _chunks(updates, batch_size):
        keys = [update.uri for update in update_batch]
        rows = _scan_rows(
            dataset,
            columns=[key],
            filter_expression=_filter_for_keys(key, keys, key_field),
        )
        counts = {value: 0 for value in keys}
        for row in rows:
            value = row[key]
            if value in counts:
                counts[value] += 1
                row_ids[value] = int(row["_rowid"])
        invalid = {value: count for value, count in counts.items() if count != 1}
        if invalid:
            details = ", ".join(f"{value!r}: {count} rows" for value, count in invalid.items())
            raise LanceUpdateValidationError(f"Each update key must match exactly one target row ({details})")
    return row_ids


def _read_blob_values(dataset, field: pa.Field, row_ids: list[int]) -> list[bytes | None]:
    try:
        blob_files = take_blob_files(dataset, ids=row_ids, blob_column=field.name)
        if len(blob_files) != len(row_ids):
            blob_files = []
            for row_id in row_ids:
                files = take_blob_files(dataset, ids=[row_id], blob_column=field.name)
                blob_files.append(files[0] if files else None)
        return [blob_file.readall() if blob_file is not None else None for blob_file in blob_files]
    except Exception as exc:
        raise LanceUpdateStorageError(f"Failed to read Blob v2 column {field.name!r}: {exc}") from exc


def _build_batches(
    dataset,
    updates: Sequence[LanceRowUpdate],
    *,
    key: str,
    batch_size: int,
    row_ids_by_key: Mapping[str, int],
) -> Iterator[pa.RecordBatch]:
    schema = dataset.schema
    blob_fields = [field for field in schema if is_blob_v2_field(field)]
    non_blob_columns = [field.name for field in schema if not is_blob_v2_field(field)]
    key_field = schema.field(key)

    for update_batch in _chunks(updates, batch_size):
        keys = [update.uri for update in update_batch]
        rows = _scan_rows(
            dataset,
            columns=non_blob_columns,
            filter_expression=_filter_for_keys(key, keys, key_field),
        )
        rows_by_key = {row[key]: row for row in rows}
        ordered_rows = [dict(rows_by_key[update.uri]) for update in update_batch]
        ordered_row_ids = [row_ids_by_key[update.uri] for update in update_batch]

        for field in blob_fields:
            blob_values = _read_blob_values(dataset, field, ordered_row_ids)
            for row, value in zip(ordered_rows, blob_values):
                row[field.name] = value

        for row, update in zip(ordered_rows, update_batch):
            row.pop("_rowid", None)
            row.update(update.values)
            row[key] = update.uri

        try:
            arrays = [build_lance_value_array([row[field.name] for row in ordered_rows], field) for field in schema]
            yield pa.RecordBatch.from_arrays(arrays, schema=schema)
        except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError) as exc:
            raise LanceUpdateValidationError(f"Invalid Lance update values: {exc}") from exc
        except LanceUpdateError:
            raise
        except Exception as exc:
            raise LanceUpdateStorageError(f"Failed to construct Lance update batch: {exc}") from exc


def _is_conflict_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "conflict" in message or ("commit" in message and "version" in message)


def merge_rows_preserving_schema(
    dataset,
    updates: Sequence[LanceRowUpdate],
    *,
    key: str = "uris",
    batch_size: int = 100,
):
    """Atomically merge bounded, full-schema source batches into a Lance dataset."""
    updates = tuple(updates)
    if not updates:
        return None

    row_ids_by_key = _validate_updates(dataset, updates, key, batch_size)
    batches = _build_batches(
        dataset,
        updates,
        key=key,
        batch_size=batch_size,
        row_ids_by_key=row_ids_by_key,
    )
    reader = pa.RecordBatchReader.from_batches(dataset.schema, batches)
    try:
        result = dataset.merge_insert(on=key).when_matched_update_all().execute(reader)
    except LanceUpdateError:
        raise
    except Exception as exc:
        if _is_conflict_error(exc):
            raise LanceUpdateConflictError(f"Lance update conflict: {exc}") from exc
        raise LanceUpdateStorageError(f"Failed to commit Lance row updates: {exc}") from exc

    updated = getattr(result, "num_updated_rows", None)
    if updated is not None and int(updated) != len(updates):
        raise LanceUpdateStorageError(f"Lance reported {updated} updated rows for {len(updates)} requested updates")
    return result
