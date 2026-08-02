from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from ...jcs import jcs_sha256

SPINE_SYMBOL_VIEW_VERSION = "spine-symbol-view-v1"

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class SpineSymbolError(ValueError):
    """Raised when Stage D cannot prove an exact Stage C symbol identity."""

    def __init__(self, message: str) -> None:
        super().__init__(f"invalid_spine_symbol_view: {message}")


def _error(message: str) -> SpineSymbolError:
    return SpineSymbolError(message)


@dataclass(frozen=True, slots=True)
class SpineSymbolRecord:
    symbol_id: str
    symbol_sha256: str
    typed_key_sha256: str
    format_id: str
    kind: str
    source_internal_ids: tuple[str, ...]
    base_source_internal_id: str
    component_id: str | None
    preset_id: str | None
    skin_id: str | None
    slot_id: str | None
    namespace: str
    namespace_key_sha256: str
    base_export_name: str
    export_name: str

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol_id": self.symbol_id,
            "symbol_sha256": self.symbol_sha256,
            "typed_key_sha256": self.typed_key_sha256,
            "format_id": self.format_id,
            "kind": self.kind,
            "source_internal_ids": list(self.source_internal_ids),
            "base_source_internal_id": self.base_source_internal_id,
            "component_id": self.component_id,
            "preset_id": self.preset_id,
            "skin_id": self.skin_id,
            "slot_id": self.slot_id,
            "namespace": self.namespace,
            "namespace_key_sha256": self.namespace_key_sha256,
            "base_export_name": self.base_export_name,
            "export_name": self.export_name,
        }


@dataclass(frozen=True, slots=True)
class SpineSymbolView:
    schema_version: str
    source_table_sha256: str
    symbols: tuple[SpineSymbolRecord, ...]
    view_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_table_sha256": self.source_table_sha256,
            "symbols": [symbol.to_dict() for symbol in self.symbols],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "view_sha256": self.view_sha256}


def _without(mapping: Mapping[str, object], key: str) -> dict[str, object]:
    return {name: value for name, value in mapping.items() if name != key}


def _validate_source_table(table: Mapping[str, object]) -> list[Mapping[str, object]]:
    table_sha256 = table.get("table_sha256")
    if not isinstance(table_sha256, str) or not _SHA256_RE.fullmatch(table_sha256):
        raise _error("global symbol table digest is invalid")
    if table_sha256 != jcs_sha256(_without(table, "table_sha256")):
        raise _error("global symbol table digest mismatch")
    exporter_families = table.get("exporter_families")
    if not isinstance(exporter_families, list) or "spine_4_2" not in exporter_families:
        raise _error("global symbol table lacks the Spine exporter family")
    raw_symbols = table.get("symbols")
    if not isinstance(raw_symbols, list) or not raw_symbols:
        raise _error("global symbol table has no symbols")
    if any(not isinstance(symbol, Mapping) for symbol in raw_symbols):
        raise _error("global symbol table contains a non-object symbol")
    return raw_symbols


def _parse_symbol(raw: Mapping[str, object]) -> SpineSymbolRecord | None:
    typed = raw.get("typed_primitive_key")
    namespace = raw.get("namespace_key")
    if not isinstance(typed, Mapping) or not isinstance(namespace, Mapping):
        raise _error("symbol lacks a typed key or namespace")
    if typed.get("key_sha256") != jcs_sha256(_without(typed, "key_sha256")):
        raise _error("typed-key digest mismatch")
    if namespace.get("key_sha256") != jcs_sha256(_without(namespace, "key_sha256")):
        raise _error("namespace digest mismatch")
    if raw.get("symbol_sha256") != jcs_sha256(_without(raw, "symbol_sha256")):
        raise _error("symbol digest mismatch")
    if typed.get("format_id") != namespace.get("format_id"):
        raise _error("typed key and namespace use different formats")
    if typed.get("format_id") != "spine_4_2":
        return None
    source_ids = typed.get("source_internal_ids")
    if not isinstance(source_ids, list) or not source_ids or any(
        not isinstance(value, str) or not value for value in source_ids
    ):
        raise _error("Spine typed key has invalid source identities")
    fields = {
        "symbol_id": raw.get("symbol_id"),
        "symbol_sha256": raw.get("symbol_sha256"),
        "typed_key_sha256": typed.get("key_sha256"),
        "format_id": typed.get("format_id"),
        "kind": typed.get("kind"),
        "base_source_internal_id": typed.get("base_source_internal_id"),
        "namespace": namespace.get("namespace"),
        "namespace_key_sha256": namespace.get("key_sha256"),
        "base_export_name": raw.get("base_export_name"),
        "export_name": raw.get("export_name"),
    }
    if any(not isinstance(value, str) or not value for value in fields.values()):
        raise _error("Spine symbol contains an invalid identity or export name")
    if not str(fields["export_name"]).isascii():
        raise _error("Spine export name is not ASCII")
    optional_fields = {}
    for field in ("component_id", "preset_id", "skin_id", "slot_id"):
        value = typed.get(field)
        if value is not None and (not isinstance(value, str) or not value):
            raise _error(f"Spine typed key has invalid {field}")
        optional_fields[field] = value
    return SpineSymbolRecord(
        **fields,  # type: ignore[arg-type]
        source_internal_ids=tuple(source_ids),
        **optional_fields,
    )


def build_spine_symbol_view(
    export_symbols: Mapping[str, object],
) -> SpineSymbolView:
    if not isinstance(export_symbols, Mapping):
        raise _error("export_symbols must be an object")
    raw_symbols = _validate_source_table(export_symbols)
    symbols = tuple(
        record
        for record in (_parse_symbol(raw) for raw in raw_symbols)
        if record is not None
    )
    if not symbols:
        raise _error("global symbol table has no Spine symbols")
    if tuple(symbol.symbol_id for symbol in symbols) != tuple(
        sorted(symbol.symbol_id for symbol in symbols)
    ):
        raise _error("Spine symbol subset is not in canonical order")
    provisional = SpineSymbolView(
        schema_version=SPINE_SYMBOL_VIEW_VERSION,
        source_table_sha256=str(export_symbols["table_sha256"]),
        symbols=symbols,
        view_sha256="",
    )
    view = SpineSymbolView(
        schema_version=provisional.schema_version,
        source_table_sha256=provisional.source_table_sha256,
        symbols=provisional.symbols,
        view_sha256=jcs_sha256(provisional.semantic_payload()),
    )
    return validate_spine_symbol_view(view)


def validate_spine_symbol_view(view: SpineSymbolView) -> SpineSymbolView:
    if not isinstance(view, SpineSymbolView):
        raise _error("symbol view has the wrong type")
    if view.schema_version != SPINE_SYMBOL_VIEW_VERSION:
        raise _error("symbol view version is unsupported")
    if not _SHA256_RE.fullmatch(view.source_table_sha256):
        raise _error("source table digest is invalid")
    if tuple(symbol.symbol_id for symbol in view.symbols) != tuple(
        sorted(symbol.symbol_id for symbol in view.symbols)
    ):
        raise _error("Spine symbol subset is not canonical")
    namespace_names: set[tuple[str, str]] = set()
    typed_keys: set[str] = set()
    for symbol in view.symbols:
        namespace_name = (symbol.namespace_key_sha256, symbol.export_name)
        if namespace_name in namespace_names:
            raise _error("two symbols collide within one export namespace")
        namespace_names.add(namespace_name)
        if symbol.typed_key_sha256 in typed_keys:
            raise _error("Spine typed-key identity is duplicated")
        typed_keys.add(symbol.typed_key_sha256)
    if view.view_sha256 != jcs_sha256(view.semantic_payload()):
        raise _error("symbol view digest mismatch")
    return view


def require_spine_symbol(
    view: SpineSymbolView,
    *,
    kind: str,
    source_internal_ids: tuple[str, ...],
    component_id: str | None = None,
    preset_id: str | None = None,
    skin_id: str | None = None,
    slot_id: str | None = None,
) -> SpineSymbolRecord:
    validate_spine_symbol_view(view)
    matches = [
        symbol
        for symbol in view.symbols
        if symbol.kind == kind
        and symbol.source_internal_ids == source_internal_ids
        and (component_id is None or symbol.component_id == component_id)
        and (preset_id is None or symbol.preset_id == preset_id)
        and (skin_id is None or symbol.skin_id == skin_id)
        and (slot_id is None or symbol.slot_id == slot_id)
    ]
    if not matches:
        raise _error(
            f"missing typed Spine symbol: {kind} {source_internal_ids!r}"
        )
    if len(matches) != 1:
        raise _error(
            f"ambiguous typed Spine symbol: {kind} {source_internal_ids!r}"
        )
    return matches[0]


__all__ = [
    "SPINE_SYMBOL_VIEW_VERSION",
    "SpineSymbolError",
    "SpineSymbolRecord",
    "SpineSymbolView",
    "build_spine_symbol_view",
    "require_spine_symbol",
    "validate_spine_symbol_view",
]
