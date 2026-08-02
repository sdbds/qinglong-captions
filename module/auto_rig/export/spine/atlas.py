from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from ...jcs import jcs_sha256
from .symbols import (
    SpineSymbolView,
    require_spine_symbol,
    validate_spine_symbol_view,
)
from .uv import SPINE_42_UV_ADAPTER_VERSION

SPINE_ATLAS_PLAN_VERSION = "spine-atlas-plan-v1"
SPINE_ATLAS_ENCODING_VERSION = "spine-atlas-ascii-lf-v1"

_PAGE_RE = re.compile(r"^rig/shared/textures/page_(0|[1-9][0-9]*)\.png$")


class SpineAtlasError(ValueError):
    """Raised when canonical texture pages cannot form the frozen Spine atlas."""

    def __init__(self, message: str) -> None:
        super().__init__(f"invalid_spine_atlas: {message}")


def _error(message: str) -> SpineAtlasError:
    return SpineAtlasError(message)


def _positive_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise _error(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class SpineAtlasRegion:
    part_id: str
    name: str
    page_index: int
    x: int
    y: int
    width: int
    height: int
    original_width: int
    original_height: int
    rotate: bool
    index: int

    def to_dict(self) -> dict[str, object]:
        return {
            "part_id": self.part_id,
            "name": self.name,
            "page_index": self.page_index,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "original_width": self.original_width,
            "original_height": self.original_height,
            "rotate": self.rotate,
            "index": self.index,
        }


@dataclass(frozen=True, slots=True)
class SpineAtlasPage:
    index: int
    path: str
    width: int
    height: int
    pma: bool
    source_encoded_png_sha256: str
    regions: tuple[SpineAtlasRegion, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "pma": self.pma,
            "source_encoded_png_sha256": self.source_encoded_png_sha256,
            "regions": [region.to_dict() for region in self.regions],
        }


@dataclass(frozen=True, slots=True)
class SpineAtlasPlan:
    schema_version: str
    encoding_version: str
    uv_adapter_version: str
    texture_pages_sha256: str
    parts_sha256: str
    symbol_view_sha256: str
    pages: tuple[SpineAtlasPage, ...]
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "encoding_version": self.encoding_version,
            "uv_adapter_version": self.uv_adapter_version,
            "texture_pages_sha256": self.texture_pages_sha256,
            "parts_sha256": self.parts_sha256,
            "symbol_view_sha256": self.symbol_view_sha256,
            "pages": [page.to_dict() for page in self.pages],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "plan_sha256": self.plan_sha256}


@dataclass(frozen=True, slots=True)
class ParsedSpineAtlasRegion:
    name: str
    bounds: tuple[int, int, int, int]
    offsets: tuple[int, int, int, int]
    rotate: bool
    index: int


@dataclass(frozen=True, slots=True)
class ParsedSpineAtlasPage:
    path: str
    width: int
    height: int
    pma: bool
    regions: tuple[ParsedSpineAtlasRegion, ...]


@dataclass(frozen=True, slots=True)
class ParsedSpineAtlas:
    pages: tuple[ParsedSpineAtlasPage, ...]


def _content_rect(placement: Mapping[str, object]) -> tuple[int, int, int, int]:
    raw = placement.get("content_rect")
    if not isinstance(raw, Mapping):
        raise _error("texture placement lacks a content rectangle")
    x = raw.get("x")
    y = raw.get("y")
    width = raw.get("width")
    height = raw.get("height")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in (x, y, width, height)):
        raise _error("content rectangle must use integers")
    assert isinstance(x, int) and isinstance(y, int)
    assert isinstance(width, int) and isinstance(height, int)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise _error("content rectangle is invalid")
    return x, y, width, height


def _build_page(
    raw_page: Mapping[str, object],
    *,
    expected_index: int,
    part_by_id: Mapping[str, Mapping[str, object]],
    symbols: SpineSymbolView,
) -> SpineAtlasPage:
    index = raw_page.get("index")
    if index != expected_index:
        raise _error("texture page indices must be consecutive from zero")
    relative_path = raw_page.get("relative_path")
    if not isinstance(relative_path, str) or not _PAGE_RE.fullmatch(relative_path):
        raise _error("texture page has a noncanonical shared path")
    if int(_PAGE_RE.fullmatch(relative_path).group(1)) != expected_index:  # type: ignore[union-attr]
        raise _error("texture page path and index disagree")
    width = _positive_int(raw_page.get("width"), field="page width")
    height = _positive_int(raw_page.get("height"), field="page height")
    if raw_page.get("alpha_mode") != "straight":
        raise _error("Spine atlas requires straight-alpha canonical pages")
    encoded_sha = raw_page.get("encoded_png_sha256")
    file_record = raw_page.get("file")
    if not isinstance(encoded_sha, str) or not isinstance(file_record, Mapping):
        raise _error("texture page lacks encoded PNG provenance")
    if file_record.get("sha256") != encoded_sha or file_record.get("path") != relative_path:
        raise _error("texture page file digest or path differs from canonical pixels")
    raw_placements = raw_page.get("placements")
    if not isinstance(raw_placements, list):
        raise _error("texture page placements must be a list")
    regions = []
    seen_parts: set[str] = set()
    for placement in raw_placements:
        if not isinstance(placement, Mapping):
            raise _error("texture placement must be an object")
        part_id = placement.get("part_id")
        if not isinstance(part_id, str) or part_id not in part_by_id:
            raise _error("texture placement references an unknown part")
        if part_id in seen_parts:
            raise _error("texture page repeats a part region")
        seen_parts.add(part_id)
        if placement.get("page_index") != expected_index:
            raise _error("texture placement references the wrong page")
        if placement.get("rotation") is not False:
            raise _error("Spine v1 atlas does not permit region rotation")
        x, y, region_width, region_height = _content_rect(placement)
        if x + region_width > width or y + region_height > height:
            raise _error("texture region exceeds its page")
        part = part_by_id[part_id]
        xyxy = part.get("xyxy")
        source_xyxy = placement.get("source_xyxy")
        if xyxy != source_xyxy or not isinstance(xyxy, list) or len(xyxy) != 4:
            raise _error("texture region differs from the final Part rectangle")
        if xyxy[2] - xyxy[0] != region_width or xyxy[3] - xyxy[1] != region_height:
            raise _error("texture region size differs from the Part rectangle")
        symbol = require_spine_symbol(
            symbols,
            kind="spine_atlas_region",
            source_internal_ids=(part_id,),
        )
        regions.append(
            SpineAtlasRegion(
                part_id=part_id,
                name=symbol.export_name,
                page_index=expected_index,
                x=x,
                y=y,
                width=region_width,
                height=region_height,
                original_width=region_width,
                original_height=region_height,
                rotate=False,
                index=-1,
            )
        )
    return SpineAtlasPage(
        index=expected_index,
        path=f"textures/page_{expected_index}.png",
        width=width,
        height=height,
        pma=False,
        source_encoded_png_sha256=encoded_sha,
        regions=tuple(sorted(regions, key=lambda region: region.part_id)),
    )


def build_spine_atlas_plan(
    texture_pages: Iterable[Mapping[str, object]],
    parts: Iterable[Mapping[str, object]],
    symbols: SpineSymbolView,
) -> SpineAtlasPlan:
    validate_spine_symbol_view(symbols)
    source_pages = tuple(texture_pages)
    source_parts = tuple(parts)
    if not source_pages or any(not isinstance(value, Mapping) for value in source_pages):
        raise _error("texture pages must be a non-empty object list")
    if not source_parts or any(not isinstance(value, Mapping) for value in source_parts):
        raise _error("parts must be a non-empty object list")
    part_by_id: dict[str, Mapping[str, object]] = {}
    for part in source_parts:
        part_id = part.get("part_id")
        if not isinstance(part_id, str) or not part_id or part_id in part_by_id:
            raise _error("Part IDs must be non-empty and unique")
        part_by_id[part_id] = part
    pages = tuple(
        _build_page(
            page,
            expected_index=index,
            part_by_id=part_by_id,
            symbols=symbols,
        )
        for index, page in enumerate(source_pages)
    )
    region_parts = [region.part_id for page in pages for region in page.regions]
    if len(region_parts) != len(set(region_parts)) or set(region_parts) != set(part_by_id):
        raise _error("atlas regions do not form an exact one-per-Part inventory")
    provisional = SpineAtlasPlan(
        schema_version=SPINE_ATLAS_PLAN_VERSION,
        encoding_version=SPINE_ATLAS_ENCODING_VERSION,
        uv_adapter_version=SPINE_42_UV_ADAPTER_VERSION,
        texture_pages_sha256=jcs_sha256(list(source_pages)),
        parts_sha256=jcs_sha256(list(source_parts)),
        symbol_view_sha256=symbols.view_sha256,
        pages=pages,
        plan_sha256="",
    )
    return SpineAtlasPlan(
        schema_version=provisional.schema_version,
        encoding_version=provisional.encoding_version,
        uv_adapter_version=provisional.uv_adapter_version,
        texture_pages_sha256=provisional.texture_pages_sha256,
        parts_sha256=provisional.parts_sha256,
        symbol_view_sha256=provisional.symbol_view_sha256,
        pages=provisional.pages,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def serialize_spine_atlas(plan: SpineAtlasPlan) -> bytes:
    if not isinstance(plan, SpineAtlasPlan) or plan.plan_sha256 != jcs_sha256(
        plan.semantic_payload()
    ):
        raise _error("atlas plan digest mismatch")
    sections = []
    for page in plan.pages:
        lines = [
            page.path,
            f"size: {page.width}, {page.height}",
            "format: RGBA8888",
            "filter: Linear, Linear",
            "repeat: none",
            "pma: false",
        ]
        for region in page.regions:
            lines.extend(
                (
                    region.name,
                    f"bounds: {region.x}, {region.y}, {region.width}, {region.height}",
                    f"offsets: 0, 0, {region.original_width}, {region.original_height}",
                    "rotate: false",
                    "index: -1",
                )
            )
        sections.append("\n".join(lines))
    payload = ("\n\n".join(sections) + "\n").encode("ascii")
    if any(byte not in b"\n" and not 32 <= byte <= 126 for byte in payload):
        raise _error("atlas bytes are not printable ASCII/LF")
    return payload


def _parse_pair(value: str, *, count: int, field: str) -> tuple[int, ...]:
    parts = tuple(item.strip() for item in value.split(","))
    if len(parts) != count:
        raise _error(f"atlas {field} has the wrong arity")
    try:
        return tuple(int(item) for item in parts)
    except ValueError as exc:
        raise _error(f"atlas {field} is not numeric") from exc


def parse_spine_atlas(payload: bytes) -> ParsedSpineAtlas:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise _error("atlas is not ASCII") from exc
    if "\r" in text or not text.endswith("\n"):
        raise _error("atlas must use LF and end with a newline")
    sections = text[:-1].split("\n\n")
    pages = []
    for section in sections:
        lines = section.split("\n")
        if len(lines) < 6:
            raise _error("atlas page section is incomplete")
        path = lines[0]
        expected_headers = ("size: ", "format: ", "filter: ", "repeat: ", "pma: ")
        if any(not lines[index + 1].startswith(prefix) for index, prefix in enumerate(expected_headers)):
            raise _error("atlas page header order differs from v1")
        width, height = _parse_pair(lines[1][6:], count=2, field="size")
        if lines[2:] and lines[2] != "format: RGBA8888":
            raise _error("atlas format must be RGBA8888")
        if lines[3] != "filter: Linear, Linear" or lines[4] != "repeat: none":
            raise _error("atlas sampler contract differs from v1")
        if lines[5] != "pma: false":
            raise _error("atlas pma must be false")
        regions = []
        cursor = 6
        while cursor < len(lines):
            if cursor + 4 >= len(lines):
                raise _error("atlas region section is incomplete")
            name = lines[cursor]
            if not name or ":" in name:
                raise _error("atlas region name is invalid")
            if not lines[cursor + 1].startswith("bounds: ") or not lines[cursor + 2].startswith("offsets: "):
                raise _error("atlas region field order differs from v1")
            bounds = _parse_pair(lines[cursor + 1][8:], count=4, field="bounds")
            offsets = _parse_pair(lines[cursor + 2][9:], count=4, field="offsets")
            if lines[cursor + 3] != "rotate: false" or lines[cursor + 4] != "index: -1":
                raise _error("atlas region rotation/index differs from v1")
            regions.append(
                ParsedSpineAtlasRegion(
                    name=name,
                    bounds=bounds,  # type: ignore[arg-type]
                    offsets=offsets,  # type: ignore[arg-type]
                    rotate=False,
                    index=-1,
                )
            )
            cursor += 5
        pages.append(
            ParsedSpineAtlasPage(
                path=path,
                width=width,
                height=height,
                pma=False,
                regions=tuple(regions),
            )
        )
    return ParsedSpineAtlas(pages=tuple(pages))


def validate_spine_atlas_plan(
    plan: SpineAtlasPlan,
    texture_pages: Iterable[Mapping[str, object]],
    parts: Iterable[Mapping[str, object]],
    symbols: SpineSymbolView,
) -> SpineAtlasPlan:
    if not isinstance(plan, SpineAtlasPlan):
        raise _error("atlas plan has the wrong type")
    if (
        plan.schema_version != SPINE_ATLAS_PLAN_VERSION
        or plan.encoding_version != SPINE_ATLAS_ENCODING_VERSION
        or plan.uv_adapter_version != SPINE_42_UV_ADAPTER_VERSION
    ):
        raise _error("atlas version is unsupported")
    if any(page.pma for page in plan.pages):
        raise _error("atlas pma must remain false")
    if plan.plan_sha256 != jcs_sha256(plan.semantic_payload()):
        raise _error("atlas plan digest mismatch")
    expected = build_spine_atlas_plan(texture_pages, parts, symbols)
    if expected != plan:
        raise _error("atlas plan differs from recomputed C facts")
    parsed = parse_spine_atlas(serialize_spine_atlas(plan))
    if len(parsed.pages) != len(plan.pages):
        raise _error("serialized atlas page count changed")
    return plan


__all__ = [
    "SPINE_ATLAS_ENCODING_VERSION",
    "SPINE_ATLAS_PLAN_VERSION",
    "ParsedSpineAtlas",
    "ParsedSpineAtlasPage",
    "ParsedSpineAtlasRegion",
    "SpineAtlasError",
    "SpineAtlasPage",
    "SpineAtlasPlan",
    "SpineAtlasRegion",
    "build_spine_atlas_plan",
    "parse_spine_atlas",
    "serialize_spine_atlas",
    "validate_spine_atlas_plan",
]
