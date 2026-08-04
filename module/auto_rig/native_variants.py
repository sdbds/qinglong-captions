from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .artifacts import ArtifactContractError, sha256_file
from .component_plan import MaskComponentPlan, NormalizedMaskPart
from .contracts import AutoRigContractError, AutoRigInputContract
from .draw_order import OrdinaryDrawOrderPlan
from .jcs import jcs_sha256

NATIVE_VARIANT_MANIFEST_VERSION = "native-variant-manifest-v1"
NATIVE_VARIANT_SET_VERSION = "native-variant-set-v2"
NATIVE_VARIANT_COMPOSITE_MODE = "occluding_overlay_v1"
NATIVE_VARIANT_CROSSFADE_COMPOSITE_MODE = "crossfade_overlay_v1"
NATIVE_VARIANT_COMPOSITE_MODES = frozenset({NATIVE_VARIANT_COMPOSITE_MODE, NATIVE_VARIANT_CROSSFADE_COMPOSITE_MODE})
NATIVE_VARIANT_ROLES = frozenset(
    {
        "eye_closed.xmin",
        "eye_closed.xmax",
        "eye_closed.coupled",
        "mouth_closed",
        "mouth_open",
        "mouth_smile",
        "mouth_frown",
    }
)
NATIVE_VARIANT_EYE_BASE_TAGS = frozenset({"eyewhite", "irides", "eyelash"})
NATIVE_VARIANT_MOUTH_BASE_TAGS = frozenset({"mouth"})

_VARIANT_FIELDS = {
    "variant_id",
    "semantic_role",
    "composite_mode",
    "base_part_ids",
    "draw_anchor_part_id",
    "xyxy",
    "path",
    "rgba_mode",
    "alpha_mode",
    "color_space",
    "file_sha256",
}
_VARIANT_ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"} | {f"com{index}" for index in range(1, 10)} | {f"lpt{index}" for index in range(1, 10)}
)
_REPARSE_POINT_ATTRIBUTE = 0x400


class NativeVariantContractError(AutoRigContractError):
    """Raised when NativeVariant input violates its public contract."""


@dataclass(frozen=True, slots=True)
class NativeVariantCandidate:
    variant_id: str
    part_id: str
    semantic_role: str
    composite_mode: str
    base_part_ids: tuple[str, ...]
    draw_anchor_part_id: str
    anchor_base_tag: str
    anchor_depth_median: float
    xyxy: tuple[int, int, int, int]
    relative_path: str
    png_path: Path
    file_sha256: str
    rgba_mode: str
    alpha_mode: str
    color_space: str
    alpha_mass_u8_sum: int
    alpha_u8: bytes
    source_kind: Literal["authored", "generated"] = "authored"
    generator_version: str | None = None

    def semantic_entry(self) -> dict[str, object]:
        return {
            "variant_id": self.variant_id,
            "part_id": self.part_id,
            "semantic_role": self.semantic_role,
            "composite_mode": self.composite_mode,
            "base_part_ids": list(self.base_part_ids),
            "draw_anchor_part_id": self.draw_anchor_part_id,
            "xyxy": list(self.xyxy),
            "path": self.relative_path,
            "rgba_mode": self.rgba_mode,
            "alpha_mode": self.alpha_mode,
            "color_space": self.color_space,
            "file_sha256": self.file_sha256,
            "source_kind": self.source_kind,
            "generator_version": self.generator_version,
        }


@dataclass(frozen=True, slots=True)
class NativeVariantSet:
    schema_version: str
    present: bool
    manifest_path: Path | None
    entries: tuple[NativeVariantCandidate, ...]
    native_variant_set_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_schema_version": NATIVE_VARIANT_MANIFEST_VERSION,
            "entries": [entry.semantic_entry() for entry in self.entries],
        }


def _error(message: str) -> NativeVariantContractError:
    return NativeVariantContractError("input_contract_mismatch", message)


def _is_reparse_point(path: Path) -> bool:
    status = os.lstat(path)
    return stat.S_ISLNK(status.st_mode) or bool(getattr(status, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE)


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _variants_directory(item_root: Path) -> tuple[Path, bool]:
    rig_inputs = item_root / "rig_inputs"
    if not _lexists(rig_inputs):
        return rig_inputs / "variants", False
    if _is_reparse_point(rig_inputs) or not rig_inputs.is_dir():
        raise _error("rig_inputs must be a regular directory, not a symlink or reparse point")
    variants = rig_inputs / "variants"
    if not _lexists(variants):
        return variants, False
    if _is_reparse_point(variants) or not variants.is_dir():
        raise _error("rig_inputs/variants must be a regular directory, not a symlink or reparse point")
    return variants, True


def _duplicate_key_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _error(f"NativeVariant JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_key_object,
            parse_constant=lambda value: (_ for _ in ()).throw(_error(f"NativeVariant JSON contains non-finite number: {value}")),
        )
    except NativeVariantContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _error("NativeVariant manifest is not valid UTF-8 JSON") from exc
    if type(payload) is not dict or set(payload) != {"schema_version", "variants"}:
        raise _error("NativeVariant manifest fields must be exactly schema_version and variants")
    if payload["schema_version"] != NATIVE_VARIANT_MANIFEST_VERSION:
        raise _error("NativeVariant manifest schema_version is unsupported")
    if type(payload["variants"]) is not list:
        raise _error("NativeVariant manifest variants must be an array")
    return payload


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise _error(f"NativeVariant {field} must be a non-empty string")
    return value


def _require_variant_id(value: Any) -> str:
    variant_id = _require_string(value, field="variant_id")
    if not _VARIANT_ID_PATTERN.fullmatch(variant_id):
        raise _error("NativeVariant variant_id does not match the canonical ASCII grammar")
    if variant_id.casefold() in _WINDOWS_RESERVED:
        raise _error(f"NativeVariant variant_id uses a Windows reserved basename: {variant_id}")
    return variant_id


def _require_base_ids(value: Any) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise _error("NativeVariant base_part_ids must be a non-empty array")
    if any(not isinstance(part_id, str) or not part_id for part_id in value):
        raise _error("NativeVariant base_part_ids must contain non-empty strings")
    if len(value) != len(set(value)):
        raise _error("NativeVariant base_part_ids contains duplicate values")
    return tuple(sorted(value))


def _require_xyxy(value: Any, *, canvas_edge: int) -> tuple[int, int, int, int]:
    if type(value) is not list or len(value) != 4:
        raise _error("NativeVariant xyxy must contain four integers")
    if any(type(coordinate) is not int for coordinate in value):
        raise _error("NativeVariant xyxy must contain four integers")
    x1, y1, x2, y2 = value
    if not (0 <= x1 < x2 <= canvas_edge and 0 <= y1 < y2 <= canvas_edge):
        raise _error("NativeVariant xyxy is outside the Rig canvas")
    return x1, y1, x2, y2


def _expected_base_ids(
    role: str,
    parts: tuple[NormalizedMaskPart, ...],
) -> tuple[str, ...]:
    if role == "eye_closed.xmin":
        selected = (part.part_id for part in parts if part.base_tag in NATIVE_VARIANT_EYE_BASE_TAGS and part.side == "xmin")
    elif role == "eye_closed.xmax":
        selected = (part.part_id for part in parts if part.base_tag in NATIVE_VARIANT_EYE_BASE_TAGS and part.side == "xmax")
    elif role == "eye_closed.coupled":
        selected = (part.part_id for part in parts if part.base_tag in NATIVE_VARIANT_EYE_BASE_TAGS)
    else:
        selected = (part.part_id for part in parts if part.base_tag in NATIVE_VARIANT_MOUTH_BASE_TAGS)
    return tuple(sorted(selected))


def _decode_png(
    path: Path,
    *,
    xyxy: tuple[int, int, int, int],
    expected_sha256: str,
    variant_id: str,
) -> tuple[bytes, int]:
    from PIL import Image, UnidentifiedImageError

    try:
        actual_sha256 = sha256_file(path)
    except ArtifactContractError as exc:
        raise _error(f"NativeVariant PNG is unavailable: {variant_id}") from exc
    if actual_sha256 != expected_sha256:
        raise _error(f"NativeVariant PNG SHA mismatch: {variant_id}")
    expected_size = (xyxy[2] - xyxy[0], xyxy[3] - xyxy[1])
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG" or image.mode != "RGBA":
                raise _error(f"NativeVariant PNG must be RGBA: {variant_id}")
            if image.size != expected_size:
                raise _error(f"NativeVariant PNG dimensions differ from xyxy: {variant_id}")
            alpha_u8 = image.getchannel("A").tobytes()
    except NativeVariantContractError:
        raise
    except (OSError, UnidentifiedImageError) as exc:
        raise _error(f"NativeVariant PNG cannot be decoded: {variant_id}") from exc
    alpha_mass = sum(alpha_u8)
    if alpha_mass <= 0:
        raise _error(f"NativeVariant PNG alpha mass must be positive: {variant_id}")
    return alpha_u8, alpha_mass


def _validate_context(
    contract: AutoRigInputContract,
    component_plan: MaskComponentPlan,
    draw_order_plan: OrdinaryDrawOrderPlan,
) -> tuple[dict[str, NormalizedMaskPart], dict[str, int]]:
    if not isinstance(contract, AutoRigInputContract):
        raise _error("NativeVariant loader requires an AutoRigInputContract")
    if not isinstance(component_plan, MaskComponentPlan):
        raise _error("NativeVariant loader requires a MaskComponentPlan")
    if not isinstance(draw_order_plan, OrdinaryDrawOrderPlan):
        raise _error("NativeVariant loader requires an OrdinaryDrawOrderPlan")
    if component_plan.canvas_edge != contract.canvas.resolution:
        raise _error("NativeVariant component plan canvas differs from the input contract")
    if jcs_sha256(component_plan.semantic_payload()) != component_plan.plan_sha256:
        raise _error("NativeVariant component plan digest is invalid")
    if jcs_sha256(draw_order_plan.semantic_payload()) != draw_order_plan.plan_sha256:
        raise _error("NativeVariant draw-order plan digest is invalid")
    parts = {part.part_id: part for part in component_plan.parts}
    source_part_ids = {part.source_part_id for part in component_plan.parts}
    contract_part_ids = {part.part_id for part in contract.parts}
    if source_part_ids != contract_part_ids:
        raise _error("NativeVariant component plan belongs to different base PartSources")
    if any(part_id.startswith("part/native.") for part_id in parts):
        raise _error("NativeVariant parser received a component plan that already contains variants")
    ranks = {record.part_id: record.part_draw_rank for record in draw_order_plan.records}
    if set(parts) != set(ranks) or tuple(draw_order_plan.ordinary_part_order) != tuple(
        part_id for part_id, _ in sorted(ranks.items(), key=lambda item: item[1])
    ):
        raise _error("NativeVariant component and draw-order plans describe different Parts")
    return parts, ranks


def _parse_entries(
    raw_entries: list[Any],
    *,
    directory: Path,
    canvas_edge: int,
    ordinary_parts: dict[str, NormalizedMaskPart],
    draw_ranks: dict[str, int],
) -> tuple[NativeVariantCandidate, ...]:
    candidates: list[NativeVariantCandidate] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_roles: set[str] = set()
    for index, raw in enumerate(raw_entries):
        if type(raw) is not dict or set(raw) != _VARIANT_FIELDS:
            raise _error(f"NativeVariant entry {index} fields do not match the v1 schema")
        variant_id = _require_variant_id(raw["variant_id"])
        role = _require_string(raw["semantic_role"], field="semantic_role")
        if role not in NATIVE_VARIANT_ROLES:
            raise _error(f"NativeVariant semantic_role is unknown: {role}")
        path_value = _require_string(raw["path"], field="path")
        expected_path = f"{variant_id}.png"
        if path_value != expected_path:
            raise _error(f"NativeVariant path must be exactly {expected_path}")
        if variant_id in seen_ids or path_value in seen_paths or role in seen_roles:
            raise _error("NativeVariant manifest contains duplicate ID, path, or semantic role")
        seen_ids.add(variant_id)
        seen_paths.add(path_value)
        seen_roles.add(role)
        if raw["composite_mode"] not in NATIVE_VARIANT_COMPOSITE_MODES:
            raise _error("NativeVariant composite_mode is unsupported")
        if role == "mouth_closed" and raw["composite_mode"] != NATIVE_VARIANT_CROSSFADE_COMPOSITE_MODE:
            raise _error("NativeVariant mouth_closed requires crossfade_overlay_v1")
        if role in {"mouth_open", "mouth_smile", "mouth_frown"} and raw["composite_mode"] != NATIVE_VARIANT_COMPOSITE_MODE:
            raise _error(f"NativeVariant {role} requires occluding_overlay_v1")
        if raw["rgba_mode"] != "RGBA":
            raise _error("NativeVariant rgba_mode must be RGBA")
        if raw["alpha_mode"] != "straight":
            raise _error("NativeVariant alpha_mode must be straight")
        if raw["color_space"] != "sRGB":
            raise _error("NativeVariant color_space must be sRGB")
        file_sha256 = _require_string(raw["file_sha256"], field="file_sha256")
        if not _SHA256_PATTERN.fullmatch(file_sha256):
            raise _error("NativeVariant file SHA must be lowercase sha256:<64hex>")
        base_ids = _require_base_ids(raw["base_part_ids"])
        unknown_base = sorted(set(base_ids) - set(ordinary_parts))
        if unknown_base:
            raise _error(f"NativeVariant base references do not exist: {unknown_base}")
        expected_base_ids = _expected_base_ids(role, tuple(ordinary_parts.values()))
        if base_ids != expected_base_ids:
            raise _error(f"NativeVariant base_part_ids differ from the role registry; expected={expected_base_ids}")
        anchor = _require_string(raw["draw_anchor_part_id"], field="draw_anchor_part_id")
        if anchor not in ordinary_parts:
            raise _error(f"NativeVariant draw anchor does not exist: {anchor}")
        if anchor not in base_ids:
            raise _error("NativeVariant draw anchor must belong to base_part_ids")
        expected_anchor = max(base_ids, key=draw_ranks.__getitem__)
        if anchor != expected_anchor:
            raise _error(f"NativeVariant draw anchor is not the frontmost base Part; expected={expected_anchor}")
        xyxy = _require_xyxy(raw["xyxy"], canvas_edge=canvas_edge)
        part_id = f"part/native.{variant_id}"
        if part_id in ordinary_parts:
            raise _error(f"NativeVariant derived Part ID collides with an ordinary Part: {part_id}")
        png_path = directory / path_value
        alpha_u8, alpha_mass = _decode_png(
            png_path,
            xyxy=xyxy,
            expected_sha256=file_sha256,
            variant_id=variant_id,
        )
        anchor_part = ordinary_parts[anchor]
        candidates.append(
            NativeVariantCandidate(
                variant_id=variant_id,
                part_id=part_id,
                semantic_role=role,
                composite_mode=raw["composite_mode"],
                base_part_ids=base_ids,
                draw_anchor_part_id=anchor,
                anchor_base_tag=anchor_part.base_tag,
                anchor_depth_median=anchor_part.depth_median,
                xyxy=xyxy,
                relative_path=path_value,
                png_path=png_path,
                file_sha256=file_sha256,
                rgba_mode="RGBA",
                alpha_mode="straight",
                color_space="sRGB",
                alpha_mass_u8_sum=alpha_mass,
                alpha_u8=alpha_u8,
                source_kind="authored",
                generator_version=None,
            )
        )
    return tuple(sorted(candidates, key=lambda entry: entry.variant_id))


def _preflight_inventory(directory: Path) -> set[str]:
    actual: set[str] = set()
    try:
        filesystem_entries = tuple(directory.iterdir())
    except OSError as exc:
        raise _error("NativeVariant directory cannot be enumerated") from exc
    for entry in filesystem_entries:
        if _is_reparse_point(entry):
            raise _error(f"NativeVariant inventory contains a symlink or reparse point: {entry.name}")
        if not entry.is_file():
            raise _error(f"NativeVariant inventory contains a non-file entry: {entry.name}")
        actual.add(entry.name)
    return actual


def _validate_inventory(
    actual: set[str],
    entries: tuple[NativeVariantCandidate, ...],
) -> None:
    expected = {"manifest.json", *(entry.relative_path for entry in entries)}
    if actual != expected:
        raise _error(f"NativeVariant inventory mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")


def _validate_role_combinations(entries: tuple[NativeVariantCandidate, ...]) -> None:
    roles = {entry.semantic_role for entry in entries}
    if "eye_closed.coupled" in roles and roles & {"eye_closed.xmin", "eye_closed.xmax"}:
        raise _error("NativeVariant coupled and single-side eye coverage overlap")
    by_role = {entry.semantic_role: entry for entry in entries}
    if {"mouth_closed", "mouth_open"} <= roles:
        raise _error("NativeVariant mouth_closed and mouth_open encode opposite base-state contracts")
    if {"mouth_smile", "mouth_frown"} <= roles:
        smile = by_role["mouth_smile"]
        frown = by_role["mouth_frown"]
        if smile.base_part_ids != frown.base_part_ids or smile.draw_anchor_part_id != frown.draw_anchor_part_id:
            raise _error("NativeVariant smile/frown base or anchor is inconsistent")


def _semantic_set(
    *,
    present: bool,
    manifest_path: Path | None,
    entries: tuple[NativeVariantCandidate, ...],
) -> NativeVariantSet:
    return make_native_variant_set(
        present=present,
        manifest_path=manifest_path,
        entries=entries,
    )


def make_native_variant_set(
    *,
    present: bool,
    manifest_path: Path | None,
    entries: tuple[NativeVariantCandidate, ...],
) -> NativeVariantSet:
    """Build a canonical authored/generated variant set after role validation."""

    ordered = tuple(sorted(entries, key=lambda entry: entry.variant_id))
    if len({entry.variant_id for entry in ordered}) != len(ordered):
        raise _error("NativeVariant set contains duplicate variant IDs")
    if len({entry.part_id for entry in ordered}) != len(ordered):
        raise _error("NativeVariant set contains duplicate Part IDs")
    if len({entry.semantic_role for entry in ordered}) != len(ordered):
        raise _error("NativeVariant set contains duplicate semantic roles")
    if any(entry.semantic_role not in NATIVE_VARIANT_ROLES for entry in ordered):
        raise _error("NativeVariant set contains an unknown semantic role")
    if any(entry.composite_mode not in NATIVE_VARIANT_COMPOSITE_MODES for entry in ordered):
        raise _error("NativeVariant set contains an unsupported composite mode")
    if any(
        entry.source_kind == "generated"
        and not entry.generator_version
        or entry.source_kind == "authored"
        and entry.generator_version is not None
        for entry in ordered
    ):
        raise _error("NativeVariant source provenance is inconsistent")
    _validate_role_combinations(ordered)
    payload = {
        "schema_version": NATIVE_VARIANT_SET_VERSION,
        "manifest_schema_version": NATIVE_VARIANT_MANIFEST_VERSION,
        "entries": [entry.semantic_entry() for entry in ordered],
    }
    return NativeVariantSet(
        schema_version=NATIVE_VARIANT_SET_VERSION,
        present=present,
        manifest_path=manifest_path,
        entries=ordered,
        native_variant_set_sha256=jcs_sha256(payload),
    )


def load_native_variant_set(
    contract: AutoRigInputContract,
    component_plan: MaskComponentPlan,
    draw_order_plan: OrdinaryDrawOrderPlan,
) -> NativeVariantSet:
    """Load the optional NativeVariant directory without making quality decisions."""

    ordinary_parts, draw_ranks = _validate_context(contract, component_plan, draw_order_plan)
    directory, present = _variants_directory(contract.item_root)
    if not present:
        return _semantic_set(present=False, manifest_path=None, entries=())
    actual_inventory = _preflight_inventory(directory)
    manifest_path = directory / "manifest.json"
    if "manifest.json" not in actual_inventory:
        raise _error("NativeVariant directory is present but manifest.json is missing")
    payload = _load_manifest(manifest_path)
    entries = _parse_entries(
        payload["variants"],
        directory=directory,
        canvas_edge=contract.canvas.resolution,
        ordinary_parts=ordinary_parts,
        draw_ranks=draw_ranks,
    )
    _validate_inventory(actual_inventory, entries)
    _validate_role_combinations(entries)
    return _semantic_set(present=True, manifest_path=manifest_path, entries=entries)


__all__ = [
    "NATIVE_VARIANT_COMPOSITE_MODE",
    "NATIVE_VARIANT_COMPOSITE_MODES",
    "NATIVE_VARIANT_CROSSFADE_COMPOSITE_MODE",
    "NATIVE_VARIANT_EYE_BASE_TAGS",
    "NATIVE_VARIANT_MANIFEST_VERSION",
    "NATIVE_VARIANT_MOUTH_BASE_TAGS",
    "NATIVE_VARIANT_ROLES",
    "NATIVE_VARIANT_SET_VERSION",
    "NativeVariantCandidate",
    "NativeVariantContractError",
    "NativeVariantSet",
    "load_native_variant_set",
    "make_native_variant_set",
]
