from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping

from PIL import Image, UnidentifiedImageError

from .tag_registry import (
    AutoRigTagContractError,
    CanonicalPartTag,
    validate_v3_final_tag_set,
    validate_v3_layerdiff_part_files,
)

AUTO_RIG_INPUT_CONTRACT_VERSION = "auto-rig-input-contract-v1"
SUPPORTED_CANVAS_EDGES = frozenset({768, 1024, 1280})
_REPARSE_POINT_ATTRIBUTE = 0x400


class AutoRigContractError(ValueError):
    """Raised when a see-through item violates the public auto-rig input contract."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class AutoRigCanvasContract:
    width: int
    height: int
    resolution: int


@dataclass(frozen=True, slots=True)
class ValidatedPartSource:
    mode: Literal["png", "psd"]
    color_path: Path
    depth_path: Path
    color_sha256: str
    depth_sha256: str
    layer_name: str | None


@dataclass(frozen=True, slots=True)
class AutoRigPartContract:
    source_tag: str
    base_tag: str
    semantic_slug: str
    side: str | None
    part_id: str
    xyxy: tuple[int, int, int, int]
    depth_median: float
    source: ValidatedPartSource


@dataclass(frozen=True, slots=True)
class AutoRigInputContract:
    schema_version: str
    item_root: Path
    tag_version: str
    canvas: AutoRigCanvasContract
    save_to_psd: bool
    tblr_split: bool
    payload_mode: Literal["png", "psd"]
    source_image_path: Path
    source_image_sha256: str
    layerdiff_manifest_path: Path
    optimized_manifest_path: Path
    optimized_info_path: Path
    parts: tuple[AutoRigPartContract, ...]
    layerdiff_manifest_sha256: str | None = None
    optimized_manifest_sha256: str | None = None
    optimized_info_sha256: str | None = None


def _error(message: str, *, code: str = "input_contract_mismatch") -> AutoRigContractError:
    return AutoRigContractError(code, message)


def _is_reparse_point(path: Path) -> bool:
    status = os.lstat(path)
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE
    )


def _validated_item_root(item_root: str | Path) -> Path:
    candidate = Path(item_root).expanduser().absolute()
    try:
        if not candidate.is_dir():
            raise _error(f"item root is not a directory: {candidate}")
        if _is_reparse_point(candidate):
            raise _error(f"item root must not be a symlink or reparse point: {candidate}")
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise _error(f"item root cannot be inspected: {candidate}") from exc


def _relative_parts(relative_path: str) -> tuple[str, ...]:
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise _error("contract path must be a non-empty POSIX relative path")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise _error(f"contract path escapes the item root: {relative_path}")
    return pure.parts


def _require_regular_file(item_root: Path, relative_path: str) -> Path:
    current = item_root
    try:
        for part in _relative_parts(relative_path):
            current = current / part
            os.lstat(current)
            if _is_reparse_point(current):
                raise _error(f"required path contains a symlink or reparse point: {relative_path}")
        if not current.is_file():
            raise _error(f"required path is not a regular file: {relative_path}")
    except FileNotFoundError as exc:
        raise _error(f"required file is missing: {relative_path}") from exc
    except OSError as exc:
        raise _error(f"required file cannot be inspected: {relative_path}") from exc
    return current


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _error(f"JSON object contains a duplicate key: {key}")
        result[key] = value
    return result


def _load_json_object(path: Path, *, relative_path: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                _error(f"JSON contains a non-finite number: {value}")
            ),
        )
    except AutoRigContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _error(f"required JSON is invalid: {relative_path}") from exc
    if type(payload) is not dict:
        raise _error(f"required JSON root must be an object: {relative_path}")
    return payload


def _require_exact_fields(payload: dict[str, Any], expected: set[str], *, field: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise _error(f"{field} fields mismatch; missing={missing}, extra={extra}")


def _require_bool(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise _error(f"{field} must be a boolean")
    return value


def _require_int(value: Any, *, field: str) -> int:
    if type(value) is not int:
        raise _error(f"{field} must be an integer")
    return value


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise _error(f"{field} must be a non-empty string")
    return value


def _require_plain_file_names(value: Any, *, field: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise _error(f"{field} must be an array")
    names = tuple(value)
    if any(
        not isinstance(name, str)
        or not name
        or "/" in name
        or "\\" in name
        or name in (".", "..")
        for name in names
    ):
        raise _error(f"{field} entries must be plain file names")
    if len(names) != len(set(names)):
        raise _error(f"{field} contains duplicate file names")
    return names


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _image_metadata(path: Path, *, field: str) -> tuple[str | None, str, tuple[int, int]]:
    try:
        with Image.open(path) as image:
            image_format = image.format
            mode = image.mode
            size = image.size
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise _error(f"{field} is not a valid image") from exc
    return image_format, mode, size


def _validate_layerdiff_manifest(path: Path) -> tuple[int, str]:
    payload = _load_json_object(path, relative_path="layerdiff/manifest.json")
    _require_exact_fields(
        payload,
        {"source_path", "resolution", "tag_version", "parts"},
        field="layerdiff manifest",
    )
    _require_string(payload["source_path"], field="layerdiff.source_path")
    resolution = _require_int(payload["resolution"], field="layerdiff.resolution")
    tag_version = _require_string(payload["tag_version"], field="layerdiff.tag_version")
    if tag_version != "v3":
        raise _error(f"unsupported LayerDiff tag version: {tag_version}")
    try:
        validate_v3_layerdiff_part_files(
            _require_plain_file_names(payload["parts"], field="layerdiff.parts")
        )
    except AutoRigTagContractError as exc:
        raise _error(str(exc)) from exc
    return resolution, tag_version


def _validate_optimized_manifest(path: Path) -> tuple[bool, bool, tuple[str, ...]]:
    payload = _load_json_object(path, relative_path="optimized/manifest.json")
    _require_exact_fields(
        payload,
        {"source_path", "save_to_psd", "tblr_split", "generated_files", "final_psd"},
        field="optimized manifest",
    )
    _require_string(payload["source_path"], field="optimized.source_path")
    save_to_psd = _require_bool(payload["save_to_psd"], field="optimized.save_to_psd")
    tblr_split = _require_bool(payload["tblr_split"], field="optimized.tblr_split")
    generated_files = _require_plain_file_names(
        payload["generated_files"],
        field="optimized.generated_files",
    )
    if tuple(sorted(generated_files)) != generated_files:
        raise _error("optimized.generated_files must be sorted")
    final_psd = payload["final_psd"]
    if save_to_psd:
        final_psd_value = _require_string(final_psd, field="optimized.final_psd")
        if Path(final_psd_value).name != "final.psd":
            raise _error("optimized.final_psd must identify final.psd")
    elif final_psd is not None:
        raise _error("optimized.final_psd must be null in PNG mode")
    return save_to_psd, tblr_split, generated_files


def _validate_info(
    path: Path,
    *,
    tblr_split: bool,
    tag_aliases: Mapping[str, str] | None,
) -> tuple[
    int,
    int,
    tuple[tuple[str, CanonicalPartTag, tuple[int, int, int, int], float], ...],
]:
    payload = _load_json_object(path, relative_path="optimized/info.json")
    _require_exact_fields(payload, {"parts", "frame_size"}, field="optimized info")
    frame_size = payload["frame_size"]
    if type(frame_size) is not list or len(frame_size) != 2:
        raise _error("optimized.frame_size must be a two-integer array")
    height = _require_int(frame_size[0], field="optimized.frame_size[0]")
    width = _require_int(frame_size[1], field="optimized.frame_size[1]")
    if width <= 0 or height <= 0:
        raise _error("optimized.frame_size edges must be positive")
    if width != height:
        raise _error(
            "optimized.frame_size must remain square until upstream writer ordering is migrated",
            code="unsupported_non_square_frame",
        )
    raw_parts = payload["parts"]
    if type(raw_parts) is not dict or not raw_parts:
        raise _error("optimized.parts must be a non-empty object")
    source_tags = tuple(raw_parts)
    aliases = _validate_tag_aliases(tag_aliases)
    unused_aliases = sorted(set(aliases) - set(source_tags))
    if unused_aliases:
        raise _error(f"unused tag aliases: {unused_aliases}")
    for source_tag in source_tags:
        _validate_raw_tag(source_tag)
    canonical_source_tags = tuple(aliases.get(tag, tag) for tag in source_tags)
    try:
        canonical_tags = validate_v3_final_tag_set(
            canonical_source_tags,
            tblr_split=tblr_split,
        )
    except AutoRigTagContractError as exc:
        raise _error(str(exc)) from exc
    raw_by_canonical = dict(zip(canonical_source_tags, source_tags, strict=True))
    normalized: list[
        tuple[str, CanonicalPartTag, tuple[int, int, int, int], float]
    ] = []
    for canonical in canonical_tags:
        source_tag = raw_by_canonical[canonical.source_tag]
        raw = raw_parts[source_tag]
        if type(raw) is not dict:
            raise _error(f"optimized.parts[{source_tag!r}] must be an object")
        for required in ("tag", "xyxy", "depth_median"):
            if required not in raw:
                raise _error(f"optimized.parts[{source_tag!r}] is missing {required}")
        if raw["tag"] != source_tag:
            raise _error(f"optimized part key/tag mismatch: {source_tag}")
        xyxy = raw["xyxy"]
        if type(xyxy) is not list or len(xyxy) != 4:
            raise _error(f"optimized part xyxy must have four integers: {source_tag}")
        bbox = tuple(
            _require_int(value, field=f"optimized.parts[{source_tag!r}].xyxy")
            for value in xyxy
        )
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise _error(f"optimized part xyxy is outside the canvas: {source_tag}")
        depth = raw["depth_median"]
        if isinstance(depth, bool) or not isinstance(depth, (int, float)):
            raise _error(f"optimized part depth_median must be numeric: {source_tag}")
        depth_value = float(depth)
        if not math.isfinite(depth_value):
            raise _error(f"optimized part depth_median must be finite: {source_tag}")
        normalized.append((source_tag, canonical, bbox, depth_value))
    return width, height, tuple(normalized)


def _validate_raw_tag(source_tag: Any) -> str:
    if (
        not isinstance(source_tag, str)
        or not source_tag
        or len(source_tag) > 128
        or source_tag in {".", ".."}
        or source_tag[-1:] in {" ", "."}
        or any(character in source_tag for character in '/\\:*?"<>|')
        or any(ord(character) < 32 or ord(character) == 127 for character in source_tag)
    ):
        raise _error(f"unsafe raw tag for fixed PartSource paths: {source_tag!r}")
    return source_tag


def _validate_tag_aliases(tag_aliases: Mapping[str, str] | None) -> dict[str, str]:
    if tag_aliases is None:
        return {}
    if not isinstance(tag_aliases, Mapping):
        raise _error("tag_aliases must be a mapping")
    aliases: dict[str, str] = {}
    for source_tag, canonical_tag in tag_aliases.items():
        raw = _validate_raw_tag(source_tag)
        if not isinstance(canonical_tag, str) or not canonical_tag:
            raise _error(f"tag alias target must be a non-empty string: {raw}")
        aliases[raw] = canonical_tag
    return aliases


def _actual_optimized_inventory(item_root: Path) -> tuple[str, ...]:
    directory = item_root / "optimized"
    try:
        entries = tuple(directory.iterdir())
    except OSError as exc:
        raise _error("optimized directory cannot be enumerated") from exc
    names: list[str] = []
    for entry in entries:
        if _is_reparse_point(entry):
            raise _error(f"optimized inventory contains a symlink or reparse point: {entry.name}")
        if not entry.is_file():
            raise _error(f"optimized inventory contains a non-file entry: {entry.name}")
        if entry.name != "manifest.json":
            names.append(entry.name)
    return tuple(sorted(names))


def _validate_png_sources(
    item_root: Path,
    part_rows: tuple[
        tuple[str, CanonicalPartTag, tuple[int, int, int, int], float], ...
    ],
    generated_files: tuple[str, ...],
) -> tuple[AutoRigPartContract, ...]:
    expected_inventory = {"info.json"}
    parts: list[AutoRigPartContract] = []
    for source_tag, canonical, bbox, depth in part_rows:
        color_name = f"{source_tag}.png"
        depth_name = f"{source_tag}_depth.png"
        color_path = _require_regular_file(item_root, f"optimized/{color_name}")
        depth_path = _require_regular_file(item_root, f"optimized/{depth_name}")
        expected_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        color_format, color_mode, color_size = _image_metadata(
            color_path,
            field=f"optimized/{color_name}",
        )
        depth_format, depth_mode, depth_size = _image_metadata(
            depth_path,
            field=f"optimized/{depth_name}",
        )
        if color_format != "PNG" or color_mode != "RGBA" or color_size != expected_size:
            raise _error(f"optimized color PNG contract mismatch: {color_name}")
        if depth_format != "PNG" or depth_mode != "L" or depth_size != expected_size:
            raise _error(f"optimized depth PNG contract mismatch: {depth_name}")
        expected_inventory.update((color_name, depth_name))
        parts.append(
            AutoRigPartContract(
                source_tag=source_tag,
                base_tag=canonical.base_tag,
                semantic_slug=canonical.semantic_slug,
                side=canonical.side,
                part_id=canonical.part_id,
                xyxy=bbox,
                depth_median=depth,
                source=ValidatedPartSource(
                    mode="png",
                    color_path=color_path,
                    depth_path=depth_path,
                    color_sha256=_sha256_file(color_path),
                    depth_sha256=_sha256_file(depth_path),
                    layer_name=None,
                ),
            )
        )
    expected = tuple(sorted(expected_inventory))
    actual = _actual_optimized_inventory(item_root)
    if generated_files != expected or actual != expected:
        raise _error(
            f"optimized PNG inventory mismatch; manifest={generated_files}, actual={actual}, expected={expected}"
        )
    return tuple(parts)


def _psd_layer_map(psd: Any, *, field: str) -> dict[str, Any]:
    layers: dict[str, Any] = {}
    for layer in psd:
        if layer.is_group():
            raise _error(f"{field} must contain only flat pixel layers")
        name = layer.name
        if not isinstance(name, str) or not name or name in layers:
            raise _error(f"{field} layer names must be unique non-empty strings")
        layers[name] = layer
    return layers


def _validate_psd_sources(
    item_root: Path,
    width: int,
    height: int,
    part_rows: tuple[
        tuple[str, CanonicalPartTag, tuple[int, int, int, int], float], ...
    ],
    generated_files: tuple[str, ...],
) -> tuple[AutoRigPartContract, ...]:
    try:
        from psd_tools import PSDImage
    except ImportError as exc:
        raise _error(
            "PSD mode requires the psdexport optional dependency",
            code="missing_optional_dependency",
        ) from exc
    color_path = _require_regular_file(item_root, "final.psd")
    depth_path = _require_regular_file(item_root, "final_depth.psd")
    try:
        color_psd = PSDImage.open(color_path)
        depth_psd = PSDImage.open(depth_path)
    except Exception as exc:
        raise _error("final PSD payload cannot be parsed") from exc
    if color_psd.size != (width, height) or depth_psd.size != (width, height):
        raise _error("final PSD canvas size differs from optimized.frame_size")
    color_layers = _psd_layer_map(color_psd, field="final.psd")
    depth_layers = _psd_layer_map(depth_psd, field="final_depth.psd")
    expected_tags = {source_tag for source_tag, _, _, _ in part_rows}
    if set(color_layers) != expected_tags or set(depth_layers) != expected_tags:
        raise _error("final PSD layer names must exactly equal optimized.parts tags")
    color_sha256 = _sha256_file(color_path)
    depth_sha256 = _sha256_file(depth_path)
    parts: list[AutoRigPartContract] = []
    for source_tag, canonical, bbox, depth in part_rows:
        color_bbox = tuple(color_layers[source_tag].bbox)
        depth_bbox = tuple(depth_layers[source_tag].bbox)
        if color_bbox != bbox or depth_bbox != bbox:
            raise _error(
                f"PSD stored layer rectangle differs from xyxy: {source_tag}",
                code="psd_bbox_mismatch",
            )
        parts.append(
            AutoRigPartContract(
                source_tag=source_tag,
                base_tag=canonical.base_tag,
                semantic_slug=canonical.semantic_slug,
                side=canonical.side,
                part_id=canonical.part_id,
                xyxy=bbox,
                depth_median=depth,
                source=ValidatedPartSource(
                    mode="psd",
                    color_path=color_path,
                    depth_path=depth_path,
                    color_sha256=color_sha256,
                    depth_sha256=depth_sha256,
                    layer_name=source_tag,
                ),
            )
        )
    expected_inventory = ("info.json",)
    actual_inventory = _actual_optimized_inventory(item_root)
    if generated_files != expected_inventory or actual_inventory != expected_inventory:
        raise _error(
            "optimized PSD inventory must contain only info.json besides manifest.json"
        )
    return tuple(parts)


def load_auto_rig_input_contract(
    item_root: str | Path,
    *,
    tag_aliases: Mapping[str, str] | None = None,
) -> AutoRigInputContract:
    """Validate a completed see-through item through fixed Stage A paths only."""

    root = _validated_item_root(item_root)
    source_image_path = _require_regular_file(root, "src_img.png")
    layerdiff_manifest_path = _require_regular_file(root, "layerdiff/manifest.json")
    optimized_manifest_path = _require_regular_file(root, "optimized/manifest.json")
    optimized_info_path = _require_regular_file(root, "optimized/info.json")

    resolution, tag_version = _validate_layerdiff_manifest(layerdiff_manifest_path)
    save_to_psd, tblr_split, generated_files = _validate_optimized_manifest(
        optimized_manifest_path
    )
    width, height, part_rows = _validate_info(
        optimized_info_path,
        tblr_split=tblr_split,
        tag_aliases=tag_aliases,
    )
    if resolution not in SUPPORTED_CANVAS_EDGES:
        raise _error(
            f"auto-rig v1 does not support canvas resolution {resolution}",
            code="unsupported_auto_rig_canvas_resolution",
        )
    if (width, height) != (resolution, resolution):
        raise _error("layerdiff.resolution and optimized.frame_size differ")
    image_format, _, source_size = _image_metadata(source_image_path, field="src_img.png")
    if image_format != "PNG" or source_size != (width, height):
        raise _error("src_img.png dimensions differ from the contract canvas")

    if save_to_psd:
        parts = _validate_psd_sources(
            root,
            width,
            height,
            part_rows,
            generated_files,
        )
        payload_mode: Literal["png", "psd"] = "psd"
    else:
        parts = _validate_png_sources(root, part_rows, generated_files)
        payload_mode = "png"
    return AutoRigInputContract(
        schema_version=AUTO_RIG_INPUT_CONTRACT_VERSION,
        item_root=root,
        tag_version=tag_version,
        canvas=AutoRigCanvasContract(width=width, height=height, resolution=resolution),
        save_to_psd=save_to_psd,
        tblr_split=tblr_split,
        payload_mode=payload_mode,
        source_image_path=source_image_path,
        source_image_sha256=_sha256_file(source_image_path),
        layerdiff_manifest_path=layerdiff_manifest_path,
        optimized_manifest_path=optimized_manifest_path,
        optimized_info_path=optimized_info_path,
        parts=parts,
        layerdiff_manifest_sha256=_sha256_file(layerdiff_manifest_path),
        optimized_manifest_sha256=_sha256_file(optimized_manifest_path),
        optimized_info_sha256=_sha256_file(optimized_info_path),
    )


__all__ = [
    "AUTO_RIG_INPUT_CONTRACT_VERSION",
    "SUPPORTED_CANVAS_EDGES",
    "AutoRigCanvasContract",
    "AutoRigContractError",
    "AutoRigInputContract",
    "AutoRigPartContract",
    "ValidatedPartSource",
    "load_auto_rig_input_contract",
]
