from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from ...jcs import jcs_sha256

LIVE2D_SYMBOL_VIEW_VERSION = "live2d-symbol-view-v1"

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class Live2DSymbolError(ValueError):
    """Raised when Stage E cannot prove an exact Stage C symbol identity."""

    def __init__(self, message: str) -> None:
        super().__init__(f"invalid_live2d_symbol_view: {message}")


def _error(message: str) -> Live2DSymbolError:
    return Live2DSymbolError(message)


@dataclass(frozen=True, slots=True)
class Live2DSymbolRecord:
    symbol_id: str
    symbol_sha256: str
    typed_key_sha256: str
    format_id: str
    kind: str
    source_internal_ids: tuple[str, ...]
    base_source_internal_id: str
    derivation_tokens: tuple[str, ...]
    component_id: str | None
    control_id: str | None
    parameter_id: str | None
    preset_id: str | None
    directory: str | None
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
            "derivation_tokens": list(self.derivation_tokens),
            "component_id": self.component_id,
            "control_id": self.control_id,
            "parameter_id": self.parameter_id,
            "preset_id": self.preset_id,
            "directory": self.directory,
            "namespace": self.namespace,
            "namespace_key_sha256": self.namespace_key_sha256,
            "base_export_name": self.base_export_name,
            "export_name": self.export_name,
        }


@dataclass(frozen=True, slots=True)
class Live2DSymbolView:
    schema_version: str
    source_table_sha256: str
    symbols: tuple[Live2DSymbolRecord, ...]
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


def _source_symbols(table: Mapping[str, object]) -> list[Mapping[str, object]]:
    digest = table.get("table_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise _error("global symbol table digest is invalid")
    if digest != jcs_sha256(_without(table, "table_sha256")):
        raise _error("global symbol table digest mismatch")
    families = table.get("exporter_families")
    if not isinstance(families, list) or "live2d_moc3_v4_00" not in families:
        raise _error("global symbol table lacks the Live2D exporter family")
    values = table.get("symbols")
    if not isinstance(values, list) or not values or any(
        not isinstance(value, Mapping) for value in values
    ):
        raise _error("global symbol table has invalid symbols")
    return values


def _optional_id(typed: Mapping[str, object], field: str) -> str | None:
    value = typed.get(field)
    if value is not None and (not isinstance(value, str) or not value):
        raise _error(f"Live2D typed key has invalid {field}")
    return value


def _parse_symbol(raw: Mapping[str, object]) -> Live2DSymbolRecord | None:
    typed = raw.get("typed_primitive_key")
    namespace = raw.get("namespace_key")
    if not isinstance(typed, Mapping) or not isinstance(namespace, Mapping):
        raise _error("symbol lacks a typed key or namespace")
    if typed.get("key_sha256") != jcs_sha256(_without(typed, "key_sha256")):
        raise _error("typed-key digest mismatch")
    if namespace.get("key_sha256") != jcs_sha256(
        _without(namespace, "key_sha256")
    ):
        raise _error("namespace digest mismatch")
    if raw.get("symbol_sha256") != jcs_sha256(_without(raw, "symbol_sha256")):
        raise _error("symbol digest mismatch")
    if typed.get("format_id") != namespace.get("format_id"):
        raise _error("typed key and namespace use different formats")
    if typed.get("format_id") != "live2d_moc3_v4_00":
        return None
    source_ids = typed.get("source_internal_ids")
    derivation = typed.get("derivation_tokens")
    if not isinstance(source_ids, list) or not source_ids or any(
        not isinstance(value, str) or not value for value in source_ids
    ):
        raise _error("Live2D typed key has invalid source identities")
    if not isinstance(derivation, list) or any(
        not isinstance(value, str) or not value for value in derivation
    ):
        raise _error("Live2D typed key has invalid derivation tokens")
    required = {
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
    if any(not isinstance(value, str) or not value for value in required.values()):
        raise _error("Live2D symbol contains an invalid identity or export name")
    if not str(required["export_name"]).isascii():
        raise _error("Live2D export name is not ASCII")
    return Live2DSymbolRecord(
        **required,  # type: ignore[arg-type]
        source_internal_ids=tuple(source_ids),
        derivation_tokens=tuple(derivation),
        component_id=_optional_id(typed, "component_id"),
        control_id=_optional_id(typed, "control_id"),
        parameter_id=_optional_id(typed, "parameter_id"),
        preset_id=_optional_id(typed, "preset_id"),
        directory=_optional_id(typed, "directory"),
    )


def build_live2d_symbol_view(
    export_symbols: Mapping[str, object],
) -> Live2DSymbolView:
    if not isinstance(export_symbols, Mapping):
        raise _error("export_symbols must be an object")
    symbols = tuple(
        record
        for record in (_parse_symbol(raw) for raw in _source_symbols(export_symbols))
        if record is not None
    )
    provisional = Live2DSymbolView(
        schema_version=LIVE2D_SYMBOL_VIEW_VERSION,
        source_table_sha256=str(export_symbols["table_sha256"]),
        symbols=symbols,
        view_sha256="",
    )
    return validate_live2d_symbol_view(
        Live2DSymbolView(
            schema_version=provisional.schema_version,
            source_table_sha256=provisional.source_table_sha256,
            symbols=provisional.symbols,
            view_sha256=jcs_sha256(provisional.semantic_payload()),
        )
    )


def validate_live2d_symbol_view(view: Live2DSymbolView) -> Live2DSymbolView:
    if not isinstance(view, Live2DSymbolView):
        raise _error("symbol view has the wrong type")
    if view.schema_version != LIVE2D_SYMBOL_VIEW_VERSION:
        raise _error("symbol view version is unsupported")
    if not _SHA256_RE.fullmatch(view.source_table_sha256):
        raise _error("source table digest is invalid")
    if tuple(symbol.symbol_id for symbol in view.symbols) != tuple(
        sorted(symbol.symbol_id for symbol in view.symbols)
    ):
        raise _error("Live2D symbol subset is not canonical")
    names: set[tuple[str, str]] = set()
    typed_keys: set[str] = set()
    for symbol in view.symbols:
        scoped_name = (symbol.namespace_key_sha256, symbol.export_name)
        if scoped_name in names:
            raise _error("two symbols collide within one export namespace")
        names.add(scoped_name)
        if symbol.typed_key_sha256 in typed_keys:
            raise _error("Live2D typed-key identity is duplicated")
        typed_keys.add(symbol.typed_key_sha256)
    if view.view_sha256 != jcs_sha256(view.semantic_payload()):
        raise _error("symbol view digest mismatch")
    return view


def require_live2d_symbol(
    view: Live2DSymbolView,
    *,
    kind: str,
    source_internal_ids: tuple[str, ...] | None = None,
    component_id: str | None = None,
    control_id: str | None = None,
    parameter_id: str | None = None,
    preset_id: str | None = None,
) -> Live2DSymbolRecord:
    validate_live2d_symbol_view(view)
    matches = [
        symbol
        for symbol in view.symbols
        if symbol.kind == kind
        and (
            source_internal_ids is None
            or symbol.source_internal_ids == source_internal_ids
        )
        and (component_id is None or symbol.component_id == component_id)
        and (control_id is None or symbol.control_id == control_id)
        and (parameter_id is None or symbol.parameter_id == parameter_id)
        and (preset_id is None or symbol.preset_id == preset_id)
    ]
    if not matches:
        raise _error(f"missing typed Live2D symbol: {kind}")
    if len(matches) != 1:
        raise _error(f"ambiguous typed Live2D symbol: {kind}")
    return matches[0]


def symbol_by_typed_key(
    view: Live2DSymbolView, typed_key_sha256: str
) -> Live2DSymbolRecord:
    validate_live2d_symbol_view(view)
    matches = [
        symbol for symbol in view.symbols if symbol.typed_key_sha256 == typed_key_sha256
    ]
    if len(matches) != 1:
        raise _error("typed Live2D symbol is missing or ambiguous")
    return matches[0]


__all__ = [
    "LIVE2D_SYMBOL_VIEW_VERSION",
    "Live2DSymbolError",
    "Live2DSymbolRecord",
    "Live2DSymbolView",
    "build_live2d_symbol_view",
    "require_live2d_symbol",
    "symbol_by_typed_key",
    "validate_live2d_symbol_view",
]
