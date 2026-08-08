from __future__ import annotations

import hashlib
import numbers
import re
import struct
from dataclasses import dataclass
from importlib.metadata import version as distribution_version
from io import BytesIO
from pathlib import Path
from typing import Iterable, Literal

from .artifacts import FileDigest, atomic_write_bytes, describe_file
from .jcs import jcs_sha256
from .texture_sources import TEXTURE_PIXEL_CONTRACT_VERSION, LoadedTextureRegion

TEXTURE_PAGE_PLAN_VERSION = "texture-page-plan-v1"
TEXTURE_RECTANGLE_BUILDER_VERSION = "texture-rectangle-content-extrude-gap-v1"
TEXTURE_PACKER_VERSION = "rectpack-maxrects-bssf-stable-v1"
TEXTURE_PAGE_SIZE = 2048
TEXTURE_MAX_PAGES = 4
TEXTURE_EXTRUSION_PX = 2
TEXTURE_SAFETY_GAP_PX = 2
CANONICAL_PNG_ENCODER_VERSION = "canonical-png-encoder-v2"
CANONICAL_TEXTURE_PAGE_SET_VERSION = "canonical-texture-page-set-v1"
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_PAGE_NAME_PATTERN = re.compile(r"^page_(0|[1-9][0-9]*)\.png$")


class TexturePlanError(ValueError):
    """Raised when a region or generated texture plan violates its contract."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class TextureRect:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def to_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    def to_dict(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    def contains(self, other: TextureRect) -> bool:
        return (
            self.x <= other.x
            and self.y <= other.y
            and self.right >= other.right
            and self.bottom >= other.bottom
        )

    def overlaps(self, other: TextureRect) -> bool:
        return not (
            self.right <= other.x
            or other.right <= self.x
            or self.bottom <= other.y
            or other.bottom <= self.y
        )


@dataclass(frozen=True, slots=True)
class TextureRegionInput:
    part_id: str
    source_kind: Literal["see_through", "native_variant"]
    variant_id: str | None
    source_xyxy: tuple[int, int, int, int]
    width: int
    height: int
    rgba_sha256: str
    alpha_mode: str
    color_space: str
    pixel_contract_version: str

    def semantic_entry(self) -> dict[str, object]:
        return {
            "part_id": self.part_id,
            "source_kind": self.source_kind,
            "variant_id": self.variant_id,
            "source_xyxy": list(self.source_xyxy),
            "width": self.width,
            "height": self.height,
            "rgba_sha256": self.rgba_sha256,
            "alpha_mode": self.alpha_mode,
            "color_space": self.color_space,
            "pixel_contract_version": self.pixel_contract_version,
        }


@dataclass(frozen=True, slots=True)
class TextureRegionPlacement:
    part_id: str
    source_kind: Literal["see_through", "native_variant"]
    variant_id: str | None
    page_index: int
    rotation: bool
    packed_footprint: TextureRect
    extrusion_rect: TextureRect
    content_rect: TextureRect
    source_xyxy: tuple[int, int, int, int]
    rgba_sha256: str
    u0: float
    v_top0: float
    u1: float
    v_top1: float

    def to_dict(self) -> dict[str, object]:
        return {
            "part_id": self.part_id,
            "source_kind": self.source_kind,
            "variant_id": self.variant_id,
            "page_index": self.page_index,
            "rotation": self.rotation,
            "packed_footprint": self.packed_footprint.to_dict(),
            "extrusion_rect": self.extrusion_rect.to_dict(),
            "content_rect": self.content_rect.to_dict(),
            "source_xyxy": list(self.source_xyxy),
            "rgba_sha256": self.rgba_sha256,
            "u0": self.u0,
            "v_top0": self.v_top0,
            "u1": self.u1,
            "v_top1": self.v_top1,
        }


@dataclass(frozen=True, slots=True)
class TexturePageRecord:
    index: int
    relative_path: str
    region_part_ids: tuple[str, ...]
    packed_area: int

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "relative_path": self.relative_path,
            "region_part_ids": list(self.region_part_ids),
            "packed_area": self.packed_area,
        }


@dataclass(frozen=True, slots=True)
class TexturePagePlan:
    schema_version: str
    rectangle_builder_version: str
    packer_version: str
    rectpack_version: str
    page_width: int
    page_height: int
    max_pages: int
    extrusion_px: int
    safety_gap_px: int
    rotation: bool
    input_part_ids: tuple[str, ...]
    input_regions_sha256: str
    fit: bool
    failure_reason: Literal["region_oversize", "page_budget_exceeded"] | None
    pages: tuple[TexturePageRecord, ...]
    placements: tuple[TextureRegionPlacement, ...]
    unplaced_part_ids: tuple[str, ...]
    used_page_count: int
    sum_padded_region_area: int
    budget_occupancy: float
    used_page_fill: float
    maximum_footprint_part_id: str | None
    maximum_footprint_width: int
    maximum_footprint_height: int
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "rectangle_builder_version": self.rectangle_builder_version,
            "packer_version": self.packer_version,
            "rectpack_version": self.rectpack_version,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "max_pages": self.max_pages,
            "extrusion_px": self.extrusion_px,
            "safety_gap_px": self.safety_gap_px,
            "rotation": self.rotation,
            "input_part_ids": list(self.input_part_ids),
            "input_regions_sha256": self.input_regions_sha256,
            "fit": self.fit,
            "failure_reason": self.failure_reason,
            "pages": [page.to_dict() for page in self.pages],
            "placements": [placement.to_dict() for placement in self.placements],
            "unplaced_part_ids": list(self.unplaced_part_ids),
            "used_page_count": self.used_page_count,
            "sum_padded_region_area": self.sum_padded_region_area,
            "budget_occupancy": self.budget_occupancy,
            "used_page_fill": self.used_page_fill,
            "maximum_footprint_part_id": self.maximum_footprint_part_id,
            "maximum_footprint_width": self.maximum_footprint_width,
            "maximum_footprint_height": self.maximum_footprint_height,
        }


@dataclass(frozen=True, slots=True)
class CanonicalPngEncoderDescriptor:
    schema_version: str
    pillow_version: str
    zlib_runtime_version: str
    optimize: bool
    compress_level: int
    metadata_policy: str
    pixel_contract_version: str
    alpha_mode: str
    color_space: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "pillow_version": self.pillow_version,
            "zlib_runtime_version": self.zlib_runtime_version,
            "optimize": self.optimize,
            "compress_level": self.compress_level,
            "metadata_policy": self.metadata_policy,
            "pixel_contract_version": self.pixel_contract_version,
            "alpha_mode": self.alpha_mode,
            "color_space": self.color_space,
        }


@dataclass(frozen=True, slots=True)
class MaterializedTexturePage:
    index: int
    width: int
    height: int
    relative_path: str
    rgba_sha256: str
    encoded_png_sha256: str
    file: FileDigest

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "width": self.width,
            "height": self.height,
            "relative_path": self.relative_path,
            "rgba_sha256": self.rgba_sha256,
            "encoded_png_sha256": self.encoded_png_sha256,
            "file": self.file.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CanonicalTexturePageSet:
    schema_version: str
    texture_page_plan_sha256: str
    encoder: CanonicalPngEncoderDescriptor
    pages: tuple[MaterializedTexturePage, ...]
    set_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "texture_page_plan_sha256": self.texture_page_plan_sha256,
            "encoder": self.encoder.to_dict(),
            "pages": [page.to_dict() for page in self.pages],
        }


def texture_region_input(region: LoadedTextureRegion) -> TextureRegionInput:
    return TextureRegionInput(
        part_id=region.part_id,
        source_kind=region.source_kind,
        variant_id=region.variant_id,
        source_xyxy=region.xyxy,
        width=region.width,
        height=region.height,
        rgba_sha256=region.rgba_sha256,
        alpha_mode=region.alpha_mode,
        color_space=region.color_space,
        pixel_contract_version=region.pixel_contract_version,
    )


def _error(code: str, message: str) -> TexturePlanError:
    return TexturePlanError(code, message)


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _validate_region(region: TextureRegionInput) -> None:
    if not isinstance(region, TextureRegionInput):
        raise _error("invalid_texture_region", "region must use TextureRegionInput")
    if not region.part_id or not region.part_id.startswith("part/"):
        raise _error("invalid_texture_region", "region part_id must use the part/ namespace")
    if region.source_kind not in {"see_through", "native_variant"}:
        raise _error("invalid_texture_region", "region source_kind is unsupported")
    if region.source_kind == "native_variant" and not region.variant_id:
        raise _error("invalid_texture_region", "native region requires variant_id")
    if region.source_kind == "see_through" and region.variant_id is not None:
        raise _error("invalid_texture_region", "see-through region cannot carry variant_id")
    if any(
        isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) <= 0
        for value in (region.width, region.height)
    ):
        raise _error("invalid_texture_region", "region dimensions must be positive integers")
    if (
        not isinstance(region.source_xyxy, tuple)
        or len(region.source_xyxy) != 4
        or any(
            isinstance(value, bool) or not isinstance(value, numbers.Integral)
            for value in region.source_xyxy
        )
    ):
        raise _error("invalid_texture_region", "region source_xyxy must contain four integers")
    x1, y1, x2, y2 = region.source_xyxy
    if (x1, y1, x2, y2) != (x1, y1, x1 + region.width, y1 + region.height):
        raise _error("invalid_texture_region", "region dimensions differ from source_xyxy")
    if not _SHA256_PATTERN.fullmatch(region.rgba_sha256):
        raise _error("invalid_texture_region", "region RGBA digest is invalid")
    if region.alpha_mode != "straight" or region.color_space != "srgb_bytes":
        raise _error("invalid_texture_region", "region alpha/color contract is unsupported")
    if region.pixel_contract_version != TEXTURE_PIXEL_CONTRACT_VERSION:
        raise _error("invalid_texture_region", "region pixel contract version is unsupported")


def _footprint_dimensions(region: TextureRegionInput) -> tuple[int, int]:
    margin = 2 * (TEXTURE_EXTRUSION_PX + TEXTURE_SAFETY_GAP_PX)
    return region.width + margin, region.height + margin


def _sort_key(region: TextureRegionInput) -> tuple[int, int, str]:
    width, height = _footprint_dimensions(region)
    return -max(width, height), -(width * height), region.part_id


def _stable_maxrects_bssf():
    from rectpack.geometry import Rectangle
    from rectpack.maxrects import MaxRectsBssf

    class StableMaxRectsBssf(MaxRectsBssf):
        def _select_position(self, width, height):
            candidates = []
            for maximum in self._max_rects:
                fitness = self._rect_fitness(maximum, width, height)
                if fitness is not None:
                    candidates.append((fitness, maximum.y, maximum.x, maximum))
            if not candidates:
                return None, None
            _, _, _, maximum = min(candidates, key=lambda item: item[:3])
            return Rectangle(maximum.x, maximum.y, width, height), maximum

    return StableMaxRectsBssf


def _pack(regions: tuple[TextureRegionInput, ...]):
    from rectpack import SORT_NONE, PackingBin, PackingMode, newPacker

    packer = newPacker(
        mode=PackingMode.Offline,
        bin_algo=PackingBin.BFF,
        pack_algo=_stable_maxrects_bssf(),
        sort_algo=SORT_NONE,
        rotation=False,
    )
    packer.add_bin(TEXTURE_PAGE_SIZE, TEXTURE_PAGE_SIZE, count=TEXTURE_MAX_PAGES)
    for region in regions:
        width, height = _footprint_dimensions(region)
        packer.add_rect(width, height, rid=region.part_id)
    packer.pack()
    return tuple(packer.rect_list())


def _placement(
    region: TextureRegionInput,
    *,
    page_index: int,
    x: int,
    y: int,
    width: int,
    height: int,
) -> TextureRegionPlacement:
    extrusion_offset = TEXTURE_SAFETY_GAP_PX
    content_offset = TEXTURE_SAFETY_GAP_PX + TEXTURE_EXTRUSION_PX
    footprint = TextureRect(x=x, y=y, width=width, height=height)
    extrusion = TextureRect(
        x=x + extrusion_offset,
        y=y + extrusion_offset,
        width=region.width + 2 * TEXTURE_EXTRUSION_PX,
        height=region.height + 2 * TEXTURE_EXTRUSION_PX,
    )
    content = TextureRect(
        x=x + content_offset,
        y=y + content_offset,
        width=region.width,
        height=region.height,
    )
    return TextureRegionPlacement(
        part_id=region.part_id,
        source_kind=region.source_kind,
        variant_id=region.variant_id,
        page_index=page_index,
        rotation=False,
        packed_footprint=footprint,
        extrusion_rect=extrusion,
        content_rect=content,
        source_xyxy=region.source_xyxy,
        rgba_sha256=region.rgba_sha256,
        u0=content.x / TEXTURE_PAGE_SIZE,
        v_top0=content.y / TEXTURE_PAGE_SIZE,
        u1=content.right / TEXTURE_PAGE_SIZE,
        v_top1=content.bottom / TEXTURE_PAGE_SIZE,
    )


def _validate_generated_plan(plan: TexturePagePlan) -> None:
    if tuple(page.index for page in plan.pages) != tuple(range(plan.used_page_count)):
        raise _error("invalid_texture_plan", "used page indices are not contiguous from zero")
    if tuple(page.relative_path for page in plan.pages) != tuple(
        f"page_{index}.png" for index in range(plan.used_page_count)
    ):
        raise _error("invalid_texture_plan", "page paths are not canonical")
    page_bounds = TextureRect(0, 0, plan.page_width, plan.page_height)
    for page in plan.pages:
        placements = tuple(item for item in plan.placements if item.page_index == page.index)
        for index, first in enumerate(placements):
            if not page_bounds.contains(first.packed_footprint):
                raise _error("invalid_texture_plan", "packed footprint escapes its page")
            if not first.packed_footprint.contains(first.extrusion_rect) or not first.extrusion_rect.contains(
                first.content_rect
            ):
                raise _error("invalid_texture_plan", "nested texture rectangles are inconsistent")
            for second in placements[index + 1 :]:
                if first.packed_footprint.overlaps(second.packed_footprint):
                    raise _error("invalid_texture_plan", "packed footprints overlap")
    placed_ids = {item.part_id for item in plan.placements}
    unplaced_ids = set(plan.unplaced_part_ids)
    if placed_ids & unplaced_ids or placed_ids | unplaced_ids != set(plan.input_part_ids):
        raise _error("invalid_texture_plan", "placed/unplaced region inventory is inconsistent")
    if plan.fit != (not plan.unplaced_part_ids):
        raise _error("invalid_texture_plan", "fit flag differs from unplaced inventory")
    if jcs_sha256(plan.semantic_payload()) != plan.plan_sha256:
        raise _error("invalid_texture_plan", "texture plan digest is invalid")


def build_texture_page_plan(
    regions: Iterable[TextureRegionInput],
) -> TexturePagePlan:
    """Pack canonical payload footprints into the frozen four-page profile."""

    normalized = tuple(regions)
    for region in normalized:
        _validate_region(region)
    part_ids = tuple(region.part_id for region in normalized)
    if len(part_ids) != len(set(part_ids)):
        raise _error("invalid_texture_region", "texture region part IDs must be unique")
    ordered = tuple(sorted(normalized, key=_sort_key))
    input_payload = {
        "schema_version": TEXTURE_PAGE_PLAN_VERSION,
        "rectangle_builder_version": TEXTURE_RECTANGLE_BUILDER_VERSION,
        "packer_version": TEXTURE_PACKER_VERSION,
        "regions": [region.semantic_entry() for region in ordered],
    }
    footprint_by_id = {region.part_id: _footprint_dimensions(region) for region in ordered}
    sum_area = sum(width * height for width, height in footprint_by_id.values())
    maximum = max(
        ordered,
        key=lambda region: (
            _footprint_dimensions(region)[0] * _footprint_dimensions(region)[1],
            region.part_id,
        ),
        default=None,
    )
    oversized = tuple(
        region.part_id
        for region in ordered
        if max(_footprint_dimensions(region)) > TEXTURE_PAGE_SIZE
    )
    packed = () if oversized else _pack(ordered)
    by_id = {region.part_id: region for region in ordered}
    placements = tuple(
        sorted(
            (
                _placement(
                    by_id[part_id],
                    page_index=int(page_index),
                    x=int(x),
                    y=int(y),
                    width=int(width),
                    height=int(height),
                )
                for page_index, x, y, width, height, part_id in packed
            ),
            key=lambda item: item.part_id,
        )
    )
    placed_ids = {item.part_id for item in placements}
    unplaced = tuple(region.part_id for region in ordered if region.part_id not in placed_ids)
    if oversized:
        failure_reason = "region_oversize"
    elif unplaced:
        failure_reason = "page_budget_exceeded"
    else:
        failure_reason = None
    used_page_count = max((item.page_index for item in placements), default=-1) + 1
    pages = tuple(
        TexturePageRecord(
            index=index,
            relative_path=f"page_{index}.png",
            region_part_ids=tuple(
                item.part_id
                for item in sorted(
                    (placement for placement in placements if placement.page_index == index),
                    key=lambda item: (
                        item.packed_footprint.y,
                        item.packed_footprint.x,
                        item.part_id,
                    ),
                )
            ),
            packed_area=sum(
                item.packed_footprint.width * item.packed_footprint.height
                for item in placements
                if item.page_index == index
            ),
        )
        for index in range(used_page_count)
    )
    budget_occupancy = sum_area / (TEXTURE_MAX_PAGES * TEXTURE_PAGE_SIZE**2)
    used_page_fill = (
        sum_area / (used_page_count * TEXTURE_PAGE_SIZE**2) if used_page_count else 0.0
    )
    maximum_width, maximum_height = (
        _footprint_dimensions(maximum) if maximum is not None else (0, 0)
    )
    values = {
        "schema_version": TEXTURE_PAGE_PLAN_VERSION,
        "rectangle_builder_version": TEXTURE_RECTANGLE_BUILDER_VERSION,
        "packer_version": TEXTURE_PACKER_VERSION,
        "rectpack_version": distribution_version("rectpack"),
        "page_width": TEXTURE_PAGE_SIZE,
        "page_height": TEXTURE_PAGE_SIZE,
        "max_pages": TEXTURE_MAX_PAGES,
        "extrusion_px": TEXTURE_EXTRUSION_PX,
        "safety_gap_px": TEXTURE_SAFETY_GAP_PX,
        "rotation": False,
        "input_part_ids": [region.part_id for region in ordered],
        "input_regions_sha256": jcs_sha256(input_payload),
        "fit": not unplaced,
        "failure_reason": failure_reason,
        "pages": [page.to_dict() for page in pages],
        "placements": [placement.to_dict() for placement in placements],
        "unplaced_part_ids": list(unplaced),
        "used_page_count": used_page_count,
        "sum_padded_region_area": sum_area,
        "budget_occupancy": budget_occupancy,
        "used_page_fill": used_page_fill,
        "maximum_footprint_part_id": maximum.part_id if maximum is not None else None,
        "maximum_footprint_width": maximum_width,
        "maximum_footprint_height": maximum_height,
    }
    plan = TexturePagePlan(
        schema_version=TEXTURE_PAGE_PLAN_VERSION,
        rectangle_builder_version=TEXTURE_RECTANGLE_BUILDER_VERSION,
        packer_version=TEXTURE_PACKER_VERSION,
        rectpack_version=distribution_version("rectpack"),
        page_width=TEXTURE_PAGE_SIZE,
        page_height=TEXTURE_PAGE_SIZE,
        max_pages=TEXTURE_MAX_PAGES,
        extrusion_px=TEXTURE_EXTRUSION_PX,
        safety_gap_px=TEXTURE_SAFETY_GAP_PX,
        rotation=False,
        input_part_ids=tuple(region.part_id for region in ordered),
        input_regions_sha256=jcs_sha256(input_payload),
        fit=not unplaced,
        failure_reason=failure_reason,
        pages=pages,
        placements=placements,
        unplaced_part_ids=unplaced,
        used_page_count=used_page_count,
        sum_padded_region_area=sum_area,
        budget_occupancy=budget_occupancy,
        used_page_fill=used_page_fill,
        maximum_footprint_part_id=maximum.part_id if maximum is not None else None,
        maximum_footprint_width=maximum_width,
        maximum_footprint_height=maximum_height,
        plan_sha256=jcs_sha256(values),
    )
    _validate_generated_plan(plan)
    return plan


def _png_chunk_types(payload: bytes) -> tuple[bytes, ...]:
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise _error("invalid_texture_materialization", "encoded page is not PNG")
    chunks = []
    offset = 8
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise _error("invalid_texture_materialization", "encoded PNG chunk is truncated")
        length = struct.unpack_from(">I", payload, offset)[0]
        end = offset + 12 + length
        if end > len(payload):
            raise _error("invalid_texture_materialization", "encoded PNG chunk is truncated")
        chunks.append(payload[offset + 4 : offset + 8])
        offset = end
    if offset != len(payload):
        raise _error("invalid_texture_materialization", "encoded PNG has trailing bytes")
    return tuple(chunks)


def _canonical_encoder_descriptor() -> CanonicalPngEncoderDescriptor:
    from PIL import features

    pillow_version = distribution_version("Pillow")
    zlib_runtime_version = features.version_codec("zlib")
    if not zlib_runtime_version:
        raise _error(
            "invalid_texture_materialization",
            "canonical PNG encoder requires Pillow zlib codec support",
        )
    return CanonicalPngEncoderDescriptor(
        schema_version=CANONICAL_PNG_ENCODER_VERSION,
        pillow_version=pillow_version,
        zlib_runtime_version=zlib_runtime_version,
        optimize=False,
        compress_level=9,
        metadata_policy="none",
        pixel_contract_version=TEXTURE_PIXEL_CONTRACT_VERSION,
        alpha_mode="straight",
        color_space="srgb_bytes",
    )


def _encode_canonical_png(rgba_u8: bytes, *, width: int, height: int) -> bytes:
    from PIL import Image

    image = Image.frombytes("RGBA", (width, height), rgba_u8)
    output = BytesIO()
    image.save(
        output,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    payload = output.getvalue()
    chunks = _png_chunk_types(payload)
    if not chunks or chunks[0] != b"IHDR" or chunks[-1] != b"IEND" or any(
        chunk not in {b"IHDR", b"IDAT", b"IEND"} for chunk in chunks
    ):
        raise _error("invalid_texture_materialization", "canonical PNG contains metadata chunks")
    return payload


def _compose_page_rgba(
    plan: TexturePagePlan,
    page_index: int,
    regions: dict[str, LoadedTextureRegion],
) -> bytes:
    import numpy as np

    page = np.zeros((plan.page_height, plan.page_width, 4), dtype=np.uint8)
    for placement in sorted(
        (item for item in plan.placements if item.page_index == page_index),
        key=lambda item: (item.packed_footprint.y, item.packed_footprint.x, item.part_id),
    ):
        source_record = regions[placement.part_id]
        source = np.frombuffer(source_record.rgba_u8, dtype=np.uint8).reshape(
            source_record.height,
            source_record.width,
            4,
        )
        extruded = np.pad(
            source,
            (
                (plan.extrusion_px, plan.extrusion_px),
                (plan.extrusion_px, plan.extrusion_px),
                (0, 0),
            ),
            mode="edge",
        )
        rect = placement.extrusion_rect
        page[rect.y : rect.bottom, rect.x : rect.right] = extruded
    return page.tobytes(order="C")


def _remove_obsolete_page_files(directory: Path, expected_names: set[str]) -> None:
    for entry in directory.iterdir():
        if not _PAGE_NAME_PATTERN.fullmatch(entry.name) or entry.name in expected_names:
            continue
        if entry.is_symlink() or not entry.is_file():
            raise _error(
                "invalid_texture_materialization",
                f"obsolete canonical page path is not a regular file: {entry.name}",
            )
        entry.unlink()


def materialize_canonical_texture_pages(
    plan: TexturePagePlan,
    loaded_regions: Iterable[LoadedTextureRegion],
    *,
    item_root: str | Path,
) -> CanonicalTexturePageSet:
    """Compose and encode each shared page exactly once under the C namespace."""

    if jcs_sha256(plan.semantic_payload()) != plan.plan_sha256 or not plan.fit:
        raise _error("invalid_texture_materialization", "texture plan is invalid or does not fit")
    loaded = tuple(loaded_regions)
    if any(not isinstance(region, LoadedTextureRegion) for region in loaded):
        raise _error("invalid_texture_materialization", "loaded regions use an invalid type")
    by_id = {region.part_id: region for region in loaded}
    if len(by_id) != len(loaded) or set(by_id) != set(plan.input_part_ids):
        raise _error("invalid_texture_materialization", "loaded region inventory differs from plan")
    rebuilt = build_texture_page_plan(texture_region_input(region) for region in loaded)
    if rebuilt != plan:
        raise _error("invalid_texture_materialization", "loaded region semantics differ from plan")
    encoder = _canonical_encoder_descriptor()
    root = Path(item_root).resolve(strict=True)
    output_dir = root / "rig" / "shared" / "textures"
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    for page_record in plan.pages:
        raw_rgba = _compose_page_rgba(plan, page_record.index, by_id)
        rgba_sha = _sha256_bytes(raw_rgba)
        encoded = _encode_canonical_png(
            raw_rgba,
            width=plan.page_width,
            height=plan.page_height,
        )
        relative_path = f"rig/shared/textures/{page_record.relative_path}"
        target = root / Path(*relative_path.split("/"))
        atomic_write_bytes(target, encoded)
        file = describe_file(root, relative_path)
        pages.append(
            MaterializedTexturePage(
                index=page_record.index,
                width=plan.page_width,
                height=plan.page_height,
                relative_path=relative_path,
                rgba_sha256=rgba_sha,
                encoded_png_sha256=_sha256_bytes(encoded),
                file=file,
            )
        )
    _remove_obsolete_page_files(
        output_dir,
        {page.relative_path.rsplit("/", 1)[1] for page in pages},
    )
    semantic_payload = {
        "schema_version": CANONICAL_TEXTURE_PAGE_SET_VERSION,
        "texture_page_plan_sha256": plan.plan_sha256,
        "encoder": encoder.to_dict(),
        "pages": [page.to_dict() for page in pages],
    }
    return CanonicalTexturePageSet(
        schema_version=CANONICAL_TEXTURE_PAGE_SET_VERSION,
        texture_page_plan_sha256=plan.plan_sha256,
        encoder=encoder,
        pages=tuple(pages),
        set_sha256=jcs_sha256(semantic_payload),
    )


__all__ = [
    "CANONICAL_PNG_ENCODER_VERSION",
    "CANONICAL_TEXTURE_PAGE_SET_VERSION",
    "TEXTURE_EXTRUSION_PX",
    "TEXTURE_MAX_PAGES",
    "TEXTURE_PACKER_VERSION",
    "TEXTURE_PAGE_PLAN_VERSION",
    "TEXTURE_PAGE_SIZE",
    "TEXTURE_RECTANGLE_BUILDER_VERSION",
    "TEXTURE_SAFETY_GAP_PX",
    "TexturePagePlan",
    "TexturePageRecord",
    "TexturePlanError",
    "TextureRect",
    "TextureRegionInput",
    "TextureRegionPlacement",
    "CanonicalPngEncoderDescriptor",
    "CanonicalTexturePageSet",
    "MaterializedTexturePage",
    "build_texture_page_plan",
    "materialize_canonical_texture_pages",
    "texture_region_input",
]
