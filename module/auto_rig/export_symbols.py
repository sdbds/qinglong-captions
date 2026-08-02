from __future__ import annotations

import re
from dataclasses import dataclass

from .control_registry import ControlRegistryPlan, validate_control_registry_plan
from .jcs import jcs_sha256
from .primitive_candidates import PrimitiveCandidateSet, TypedPrimitiveKey

GLOBAL_EXPORT_SYMBOL_TABLE_VERSION = "global-export-symbol-table-v1"
EXPORT_NAMESPACE_SCHEMA_VERSION = "export-namespaces-v1"
INTERNAL_ID_CODEC_VERSION = "internal-id-v1"
EXPORT_NAME_CODEC_VERSION = "export-name-v1"
SYMBOL_KIND_CODEC_VERSION = "symbol-kind-v1"

_INTERNAL_ID_RE = re.compile(
    r"^[a-z][a-z0-9_-]*/[a-z0-9][a-z0-9._-]{0,126}$"
)
_EXPORTER_FAMILIES = ("live2d_moc3_v4_00", "spine_4_2")


class GlobalExportSymbolTableError(ValueError):
    """Raised when typed exporter symbols cannot be named unambiguously."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _error(message: str) -> GlobalExportSymbolTableError:
    return GlobalExportSymbolTableError(
        "invalid_global_export_symbol_table", message
    )


def _collision(message: str) -> GlobalExportSymbolTableError:
    return GlobalExportSymbolTableError("export_name_collision", message)


@dataclass(frozen=True, slots=True)
class ExportNamespaceKey:
    schema_version: str
    format_id: str
    namespace: str
    skin_id: str | None
    slot_id: str | None
    directory: str | None
    key_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "format_id": self.format_id,
            "namespace": self.namespace,
            "skin_id": self.skin_id,
            "slot_id": self.slot_id,
            "directory": self.directory,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "key_sha256": self.key_sha256}


@dataclass(frozen=True, slots=True)
class ExportSymbol:
    symbol_id: str
    typed_primitive_key: TypedPrimitiveKey
    namespace_key: ExportNamespaceKey
    base_export_name: str
    preferred_export_name: str
    export_name: str
    reserved_name: bool
    symbol_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "symbol_id": self.symbol_id,
            "typed_primitive_key": self.typed_primitive_key.to_dict(),
            "namespace_key": self.namespace_key.to_dict(),
            "base_export_name": self.base_export_name,
            "preferred_export_name": self.preferred_export_name,
            "export_name": self.export_name,
            "reserved_name": self.reserved_name,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "symbol_sha256": self.symbol_sha256}


@dataclass(frozen=True, slots=True)
class GlobalExportSymbolTable:
    schema_version: str
    namespace_schema_version: str
    internal_id_codec_version: str
    name_codec_version: str
    symbol_kind_codec_version: str
    exporter_families: tuple[str, ...]
    primitive_candidate_set_sha256: str
    candidate_universe_sha256: str
    symbol_universe_sha256: str
    symbols: tuple[ExportSymbol, ...]
    table_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "namespace_schema_version": self.namespace_schema_version,
            "internal_id_codec_version": self.internal_id_codec_version,
            "name_codec_version": self.name_codec_version,
            "symbol_kind_codec_version": self.symbol_kind_codec_version,
            "exporter_families": list(self.exporter_families),
            "primitive_candidate_set_sha256": self.primitive_candidate_set_sha256,
            "candidate_universe_sha256": self.candidate_universe_sha256,
            "symbol_universe_sha256": self.symbol_universe_sha256,
            "symbols": [symbol.to_dict() for symbol in self.symbols],
        }


def encode_base_export_name(internal_id: str) -> str:
    """Apply the frozen InternalId-to-base-name transform."""

    if not isinstance(internal_id, str) or not _INTERNAL_ID_RE.fullmatch(internal_id):
        raise _error(f"base source is not an InternalId v1 value: {internal_id!r}")
    stem = re.sub(r"[.-]+", "_", internal_id.rsplit("/", 1)[-1]).strip("_")
    if not stem or not stem.isascii():
        raise _error("base export name is empty or non-ASCII")
    return stem


def _namespace(
    format_id: str,
    namespace: str,
    *,
    skin_id: str | None = None,
    slot_id: str | None = None,
    directory: str | None = None,
) -> ExportNamespaceKey:
    values = {
        "schema_version": EXPORT_NAMESPACE_SCHEMA_VERSION,
        "format_id": format_id,
        "namespace": namespace,
        "skin_id": skin_id,
        "slot_id": slot_id,
        "directory": directory,
    }
    provisional = ExportNamespaceKey(**values, key_sha256="")
    return ExportNamespaceKey(
        **values,
        key_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def _namespace_for(key: TypedPrimitiveKey) -> ExportNamespaceKey:
    mapping = {
        "spine_skin": "skin",
        "spine_bone": "bone",
        "spine_slot": "slot",
        "spine_attachment_object": "attachment_object",
        "spine_atlas_region": "atlas_region",
        "spine_animation": "animation",
        "live2d_part": "part",
        "live2d_artmesh": "artmesh",
        "live2d_rotation_deformer": "deformer",
        "live2d_warp_deformer": "deformer",
        "live2d_parameter": "parameter",
        "live2d_motion": "motion",
        "live2d_expression": "expression",
        "texture_page": "texture_page",
    }
    if key.kind == "spine_attachment_key":
        if key.skin_id is None or key.slot_id is None:
            raise _error("Spine attachment-key namespace lacks skin/slot scope")
        return _namespace(
            key.format_id,
            "attachment_key",
            skin_id=key.skin_id,
            slot_id=key.slot_id,
        )
    namespace = mapping.get(key.kind)
    if namespace is None:
        raise _error(f"SymbolKindCodec has no namespace for {key.kind}")
    directory = key.directory if namespace in {"motion", "expression"} else None
    if namespace in {"motion", "expression"} and not directory:
        raise _error(f"{namespace} symbol lacks a directory scope")
    return _namespace(key.format_id, namespace, directory=directory)


def _validate_candidate_set_structure(plan: PrimitiveCandidateSet) -> None:
    if not isinstance(plan, PrimitiveCandidateSet):
        raise _error("primitive candidate set has the wrong type")
    if plan.plan_sha256 != jcs_sha256(plan.semantic_payload()):
        raise _error("primitive candidate set digest mismatch")
    if tuple(candidate.candidate_id for candidate in plan.candidates) != tuple(
        sorted(candidate.candidate_id for candidate in plan.candidates)
    ):
        raise _error("primitive candidates are not in canonical order")
    if len({candidate.candidate_id for candidate in plan.candidates}) != len(
        plan.candidates
    ):
        raise _error("primitive candidate IDs collide")
    if any(
        candidate.candidate_sha256 != jcs_sha256(candidate.semantic_payload())
        or candidate.typed_primitive_key.key_sha256
        != jcs_sha256(candidate.typed_primitive_key.semantic_payload())
        for candidate in plan.candidates
    ):
        raise _error("primitive candidate content digest mismatch")
    universe_payload = [candidate.to_dict() for candidate in plan.candidates] + [
        {"deterministic_absence": value}
        for value in plan.deterministic_absences
    ]
    if plan.candidate_universe_sha256 != jcs_sha256(universe_payload):
        raise _error("primitive candidate universe digest mismatch")


def _unique_keys(plan: PrimitiveCandidateSet) -> tuple[TypedPrimitiveKey, ...]:
    by_digest: dict[str, TypedPrimitiveKey] = {}
    for candidate in plan.candidates:
        key = candidate.typed_primitive_key
        previous = by_digest.get(key.key_sha256)
        if previous is not None and previous != key:
            raise _collision("two typed keys share one complete key digest")
        by_digest[key.key_sha256] = key
    return tuple(by_digest[digest] for digest in sorted(by_digest))


def _suffix_name(
    preferred: str,
    namespace: ExportNamespaceKey,
    key: TypedPrimitiveKey,
) -> str:
    prefix = preferred.encode("ascii")[:46].decode("ascii").rstrip("_")
    suffix = jcs_sha256(
        {
            "namespace_key": namespace.semantic_payload(),
            "typed_key": key.semantic_payload(),
        }
    ).removeprefix("sha256:")[:16]
    return f"{prefix}_{suffix}"


def _resolve_export_symbols(
    keys: tuple[TypedPrimitiveKey, ...],
    *,
    parameter_export_names: dict[str, str],
) -> tuple[ExportSymbol, ...]:
    """Resolve a complete key set; kept module-private so exporters cannot rename."""

    if len({key.key_sha256 for key in keys}) != len(keys):
        raise _collision("typed key list contains duplicate identities")
    raw: list[
        tuple[TypedPrimitiveKey, ExportNamespaceKey, str, str, bool]
    ] = []
    for key in keys:
        if key.key_sha256 != jcs_sha256(key.semantic_payload()):
            raise _error("typed primitive key digest mismatch")
        namespace = _namespace_for(key)
        base = encode_base_export_name(key.base_source_internal_id)
        reserved = key.kind == "live2d_parameter"
        if reserved:
            export_name = parameter_export_names.get(key.parameter_id or "")
            if export_name is None:
                raise _error(f"Live2D parameter is absent from registry: {key.parameter_id}")
            preferred = export_name
        else:
            preferred = base
            if key.derivation_tokens:
                preferred += "__" + "_".join(key.derivation_tokens)
        if not preferred or not preferred.isascii():
            raise _collision("preferred export name is empty or non-ASCII")
        raw.append((key, namespace, base, preferred, reserved))

    reserved_by_namespace: dict[str, set[str]] = {}
    groups: dict[tuple[str, str], list[int]] = {}
    for index, (_key_value, namespace, _base, preferred, reserved) in enumerate(raw):
        groups.setdefault((namespace.key_sha256, preferred), []).append(index)
        if reserved:
            reserved_by_namespace.setdefault(namespace.key_sha256, set()).add(preferred)

    resolved: list[ExportSymbol] = []
    for index, (key, namespace, base, preferred, reserved) in enumerate(raw):
        group = groups[(namespace.key_sha256, preferred)]
        if reserved and len(group) > 1:
            reserved_members = sum(raw[item][4] for item in group)
            if reserved_members > 1:
                raise _collision("reserved Live2D parameter names collide")
        needs_suffix = (
            len(preferred.encode("ascii")) >= 64
            or len(group) > 1 and not reserved
            or (
                not reserved
                and preferred
                in reserved_by_namespace.get(namespace.key_sha256, set())
            )
        )
        export_name = (
            _suffix_name(preferred, namespace, key) if needs_suffix else preferred
        )
        if (
            not export_name
            or not export_name.isascii()
            or len(export_name.encode("ascii")) >= 64
        ):
            raise _collision("resolved export name violates the 63-byte ASCII limit")
        symbol_id = "symbol/s_" + jcs_sha256(
            {
                "schema_version": SYMBOL_KIND_CODEC_VERSION,
                "typed_key_sha256": key.key_sha256,
                "namespace_key_sha256": namespace.key_sha256,
            }
        ).removeprefix("sha256:")
        values = {
            "symbol_id": symbol_id,
            "typed_primitive_key": key,
            "namespace_key": namespace,
            "base_export_name": base,
            "preferred_export_name": preferred,
            "export_name": export_name,
            "reserved_name": reserved,
        }
        provisional = ExportSymbol(**values, symbol_sha256="")
        resolved.append(
            ExportSymbol(
                **values,
                symbol_sha256=jcs_sha256(provisional.semantic_payload()),
            )
        )

    uniqueness: set[tuple[str, str]] = set()
    for symbol in resolved:
        identity = (symbol.namespace_key.key_sha256, symbol.export_name)
        if identity in uniqueness:
            raise _collision("resolved names still collide inside a namespace")
        uniqueness.add(identity)
    return tuple(resolved)


def _parameter_export_names(controls: ControlRegistryPlan) -> dict[str, str]:
    result = {
        control.live2d.parameter_id: control.live2d.export_name
        for control in controls.controls
        if control.live2d is not None
    }
    if len(result) != sum(control.live2d is not None for control in controls.controls):
        raise _error("Live2D parameter IDs collide in ControlRegistry")
    return result


def validate_global_export_symbol_table(
    table: GlobalExportSymbolTable,
    candidates: PrimitiveCandidateSet,
    controls: ControlRegistryPlan,
) -> GlobalExportSymbolTable:
    """Re-resolve names from the full, unpruned candidate universe."""

    _validate_candidate_set_structure(candidates)
    validate_control_registry_plan(controls)
    if (
        not isinstance(table, GlobalExportSymbolTable)
        or table.schema_version != GLOBAL_EXPORT_SYMBOL_TABLE_VERSION
        or table.namespace_schema_version != EXPORT_NAMESPACE_SCHEMA_VERSION
        or table.internal_id_codec_version != INTERNAL_ID_CODEC_VERSION
        or table.name_codec_version != EXPORT_NAME_CODEC_VERSION
        or table.symbol_kind_codec_version != SYMBOL_KIND_CODEC_VERSION
        or table.exporter_families != _EXPORTER_FAMILIES
    ):
        raise _error("global symbol-table version descriptor is invalid")
    if (
        table.primitive_candidate_set_sha256 != candidates.plan_sha256
        or table.candidate_universe_sha256 != candidates.candidate_universe_sha256
    ):
        raise _error("global symbol table references another candidate universe")
    expected = tuple(
        sorted(
            _resolve_export_symbols(
                _unique_keys(candidates),
                parameter_export_names=_parameter_export_names(controls),
            ),
            key=lambda symbol: symbol.symbol_id,
        )
    )
    if table.symbols != expected:
        raise _error("global symbols differ from the canonical codec output")
    if any(
        symbol.symbol_sha256 != jcs_sha256(symbol.semantic_payload())
        or symbol.namespace_key.key_sha256
        != jcs_sha256(symbol.namespace_key.semantic_payload())
        for symbol in table.symbols
    ):
        raise _error("symbol or namespace digest mismatch")
    universe_payload = [
        {
            "typed_key_sha256": symbol.typed_primitive_key.key_sha256,
            "namespace_key": symbol.namespace_key.to_dict(),
            "base_export_name": symbol.base_export_name,
            "export_name": symbol.export_name,
        }
        for symbol in table.symbols
    ]
    if table.symbol_universe_sha256 != jcs_sha256(universe_payload):
        raise _error("symbol-universe digest mismatch")
    if table.table_sha256 != jcs_sha256(table.semantic_payload()):
        raise _error("global symbol-table digest mismatch")
    return table


def build_global_export_symbol_table(
    candidates: PrimitiveCandidateSet,
    controls: ControlRegistryPlan,
) -> GlobalExportSymbolTable:
    """Name the complete typed universe once, before profile pruning."""

    _validate_candidate_set_structure(candidates)
    validate_control_registry_plan(controls)
    symbols = tuple(
        sorted(
            _resolve_export_symbols(
                _unique_keys(candidates),
                parameter_export_names=_parameter_export_names(controls),
            ),
            key=lambda symbol: symbol.symbol_id,
        )
    )
    universe_payload = [
        {
            "typed_key_sha256": symbol.typed_primitive_key.key_sha256,
            "namespace_key": symbol.namespace_key.to_dict(),
            "base_export_name": symbol.base_export_name,
            "export_name": symbol.export_name,
        }
        for symbol in symbols
    ]
    values = {
        "schema_version": GLOBAL_EXPORT_SYMBOL_TABLE_VERSION,
        "namespace_schema_version": EXPORT_NAMESPACE_SCHEMA_VERSION,
        "internal_id_codec_version": INTERNAL_ID_CODEC_VERSION,
        "name_codec_version": EXPORT_NAME_CODEC_VERSION,
        "symbol_kind_codec_version": SYMBOL_KIND_CODEC_VERSION,
        "exporter_families": _EXPORTER_FAMILIES,
        "primitive_candidate_set_sha256": candidates.plan_sha256,
        "candidate_universe_sha256": candidates.candidate_universe_sha256,
        "symbol_universe_sha256": jcs_sha256(universe_payload),
        "symbols": symbols,
    }
    provisional = GlobalExportSymbolTable(**values, table_sha256="")
    table = GlobalExportSymbolTable(
        **values,
        table_sha256=jcs_sha256(provisional.semantic_payload()),
    )
    return validate_global_export_symbol_table(table, candidates, controls)


__all__ = [
    "EXPORT_NAME_CODEC_VERSION",
    "EXPORT_NAMESPACE_SCHEMA_VERSION",
    "GLOBAL_EXPORT_SYMBOL_TABLE_VERSION",
    "INTERNAL_ID_CODEC_VERSION",
    "SYMBOL_KIND_CODEC_VERSION",
    "ExportNamespaceKey",
    "ExportSymbol",
    "GlobalExportSymbolTable",
    "GlobalExportSymbolTableError",
    "build_global_export_symbol_table",
    "encode_base_export_name",
    "validate_global_export_symbol_table",
]
