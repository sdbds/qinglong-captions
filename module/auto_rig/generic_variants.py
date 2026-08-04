from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .artifacts import atomic_write_bytes
from .component_plan import MaskComponentPlan, NormalizedMaskPart
from .contracts import AutoRigInputContract
from .draw_order import OrdinaryDrawOrderPlan
from .jcs import jcs_sha256
from .native_variants import (
    NATIVE_VARIANT_COMPOSITE_MODE,
    NATIVE_VARIANT_CROSSFADE_COMPOSITE_MODE,
    NativeVariantCandidate,
    NativeVariantSet,
    make_native_variant_set,
)
from .texture_sources import LoadedTextureRegion

if TYPE_CHECKING:
    from PIL import Image

GENERIC_VARIANT_SYNTHESIS_VERSION = "generic-variant-synthesis-plan-v1"
GENERIC_VARIANT_GENERATOR_VERSION = "item-pixel-eye-mouth-generator-v2"
GENERIC_VARIANT_OUTPUT_DIRECTORY = "rig/cache/A/generated_variants"

_TARGETS = (
    ("eye_closed.xmin", NATIVE_VARIANT_CROSSFADE_COMPOSITE_MODE),
    ("eye_closed.xmax", NATIVE_VARIANT_CROSSFADE_COMPOSITE_MODE),
    ("mouth_closed", NATIVE_VARIANT_CROSSFADE_COMPOSITE_MODE),
    ("mouth_frown", NATIVE_VARIANT_COMPOSITE_MODE),
    ("mouth_smile", NATIVE_VARIANT_COMPOSITE_MODE),
)


class GenericVariantSynthesisError(ValueError):
    """Raised when deterministic variant synthesis receives inconsistent facts."""


@dataclass(frozen=True, slots=True)
class GenericVariantSynthesisRecord:
    semantic_role: str
    status: Literal["generated", "authored_override", "unavailable"]
    composite_mode: str
    variant_id: str | None
    source_part_ids: tuple[str, ...]
    xyxy: tuple[int, int, int, int] | None
    output_path: str | None
    file_sha256: str | None
    reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "semantic_role": self.semantic_role,
            "status": self.status,
            "composite_mode": self.composite_mode,
            "variant_id": self.variant_id,
            "source_part_ids": list(self.source_part_ids),
            "xyxy": list(self.xyxy) if self.xyxy is not None else None,
            "output_path": self.output_path,
            "file_sha256": self.file_sha256,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class GenericVariantSynthesisPlan:
    schema_version: str
    generator_version: str
    authored_variant_set_sha256: str
    source_rgba_sha256: tuple[str, ...]
    mouth_state_observation: Literal["open_candidate", "closed_or_ambiguous", "missing"]
    records: tuple[GenericVariantSynthesisRecord, ...]
    generated_variant_ids: tuple[str, ...]
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generator_version": self.generator_version,
            "authored_variant_set_sha256": self.authored_variant_set_sha256,
            "source_rgba_sha256": list(self.source_rgba_sha256),
            "mouth_state_observation": self.mouth_state_observation,
            "records": [record.to_dict() for record in self.records],
            "generated_variant_ids": list(self.generated_variant_ids),
        }


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _union_bbox(
    parts: tuple[NormalizedMaskPart, ...],
    *,
    canvas_edge: int,
    padding: int,
) -> tuple[int, int, int, int]:
    x1 = max(0, min(part.xyxy[0] for part in parts) - padding)
    y1 = max(0, min(part.xyxy[1] for part in parts) - padding)
    x2 = min(canvas_edge, max(part.xyxy[2] for part in parts) + padding)
    y2 = min(canvas_edge, max(part.xyxy[3] for part in parts) + padding)
    return x1, y1, x2, y2


def _alpha_weighted_color(
    parts: tuple[NormalizedMaskPart, ...],
    regions: dict[str, LoadedTextureRegion],
) -> tuple[int, int, int, int]:
    red = green = blue = weight = 0
    for part in parts:
        region = regions[part.source_part_id]
        pixels = memoryview(region.rgba_u8)
        for offset in range(0, len(pixels), 4):
            alpha = int(pixels[offset + 3])
            if alpha:
                red += int(pixels[offset]) * alpha
                green += int(pixels[offset + 1]) * alpha
                blue += int(pixels[offset + 2]) * alpha
                weight += alpha
    if weight == 0:
        return 48, 32, 42, 255
    return red // weight, green // weight, blue // weight, 255


def _curve_layer(
    size: tuple[int, int],
    *,
    color: tuple[int, int, int, int],
    curvature: float,
    thickness_ratio: float,
    center_y_ratio: float = 0.52,
) -> "Image.Image":
    from PIL import Image, ImageDraw

    width, height = size
    scale = 4
    layer = Image.new("RGBA", (width * scale, height * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    left = max(1.0, width * 0.08)
    right = min(width - 1.0, width * 0.92)
    center_y = height * center_y_ratio
    points = []
    steps = max(16, width * 2)
    for index in range(steps + 1):
        t = index / steps
        x = left + (right - left) * t
        normalized = 2.0 * t - 1.0
        y = center_y + curvature * height * (1.0 - normalized * normalized)
        points.append((round(x * scale), round(y * scale)))
    thickness = max(scale, round(max(1.0, min(width, height) * thickness_ratio) * scale))
    draw.line(points, fill=color, width=thickness, joint="curve")
    return layer.resize((width, height), Image.Resampling.LANCZOS)


def _crop_region_to_canvas(
    destination: "Image.Image",
    destination_xyxy: tuple[int, int, int, int],
    source: LoadedTextureRegion,
) -> None:
    from PIL import Image

    x1, y1, x2, y2 = destination_xyxy
    sx1, sy1, sx2, sy2 = source.xyxy
    ix1, iy1 = max(x1, sx1), max(y1, sy1)
    ix2, iy2 = min(x2, sx2), min(y2, sy2)
    if ix1 >= ix2 or iy1 >= iy2:
        return
    image = Image.frombytes("RGBA", (source.width, source.height), source.rgba_u8)
    crop = image.crop((ix1 - sx1, iy1 - sy1, ix2 - sx1, iy2 - sy1))
    destination.alpha_composite(crop, (ix1 - x1, iy1 - y1))


def _face_underlay(
    xyxy: tuple[int, int, int, int],
    face_parts: tuple[NormalizedMaskPart, ...],
    regions: dict[str, LoadedTextureRegion],
) -> "Image.Image" | None:
    from PIL import Image

    width, height = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
    result = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for part in face_parts:
        _crop_region_to_canvas(result, xyxy, regions[part.source_part_id])
    if result.getchannel("A").getbbox() is None:
        return None
    return result


def _png_bytes(image: "Image.Image") -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=False, compress_level=9)
    return stream.getvalue()


def _candidate(
    *,
    root: Path,
    role: str,
    composite_mode: str,
    image: "Image.Image",
    xyxy: tuple[int, int, int, int],
    base_parts: tuple[NormalizedMaskPart, ...],
    draw_ranks: dict[str, int],
) -> NativeVariantCandidate:
    token = role.replace(".", "-").replace("_", "-")
    variant_id = f"generated-{token}"
    relative_path = f"{GENERIC_VARIANT_OUTPUT_DIRECTORY}/{variant_id}.png"
    payload = _png_bytes(image)
    output = root / Path(*relative_path.split("/"))
    if not output.is_file() or output.read_bytes() != payload:
        atomic_write_bytes(output, payload)
    alpha = image.getchannel("A").tobytes()
    base_ids = tuple(sorted(part.part_id for part in base_parts))
    anchor = max(base_ids, key=draw_ranks.__getitem__)
    anchor_part = next(part for part in base_parts if part.part_id == anchor)
    return NativeVariantCandidate(
        variant_id=variant_id,
        part_id=f"part/native.{variant_id}",
        semantic_role=role,
        composite_mode=composite_mode,
        base_part_ids=base_ids,
        draw_anchor_part_id=anchor,
        anchor_base_tag=anchor_part.base_tag,
        anchor_depth_median=anchor_part.depth_median,
        xyxy=xyxy,
        relative_path=relative_path,
        png_path=output,
        file_sha256=_sha256_bytes(payload),
        rgba_mode="RGBA",
        alpha_mode="straight",
        color_space="sRGB",
        alpha_mass_u8_sum=sum(alpha),
        alpha_u8=alpha,
        source_kind="generated",
        generator_version=GENERIC_VARIANT_GENERATOR_VERSION,
    )


def _mouth_state(
    mouth_parts: tuple[NormalizedMaskPart, ...],
    regions: dict[str, LoadedTextureRegion],
) -> Literal["open_candidate", "closed_or_ambiguous", "missing"]:
    if not mouth_parts:
        return "missing"
    source_regions = {part.source_part_id: regions[part.source_part_id] for part in mouth_parts}
    x1 = min(region.xyxy[0] for region in source_regions.values())
    y1 = min(region.xyxy[1] for region in source_regions.values())
    x2 = max(region.xyxy[2] for region in source_regions.values())
    y2 = max(region.xyxy[3] for region in source_regions.values())
    union_width = x2 - x1
    union_height = y2 - y1
    alpha = [0] * (union_width * union_height)
    for region in source_regions.values():
        offset_x = region.xyxy[0] - x1
        offset_y = region.xyxy[1] - y1
        for local_y in range(region.height):
            for local_x in range(region.width):
                source = (local_y * region.width + local_x) * 4 + 3
                target = (offset_y + local_y) * union_width + offset_x + local_x
                alpha[target] = max(alpha[target], region.rgba_u8[source])

    occupied = [(index % union_width, index // union_width) for index, value in enumerate(alpha) if value > 0]
    if not occupied:
        return "closed_or_ambiguous"
    support_x1 = min(point[0] for point in occupied)
    support_y1 = min(point[1] for point in occupied)
    support_x2 = max(point[0] for point in occupied) + 1
    support_y2 = max(point[1] for point in occupied) + 1
    width = support_x2 - support_x1
    height = support_y2 - support_y1
    support = [alpha[y * union_width + x] for y in range(support_y1, support_y2) for x in range(support_x1, support_x2)]
    nonzero = sum(value > 0 for value in support)
    fill_ratio = nonzero / len(support)

    exterior: set[int] = set()
    frontier = [
        index
        for index, value in enumerate(support)
        if value == 0 and (index < width or index >= len(support) - width or index % width == 0 or index % width == width - 1)
    ]
    while frontier:
        index = frontier.pop()
        if index in exterior or support[index] != 0:
            continue
        exterior.add(index)
        x = index % width
        y = index // width
        if x > 0:
            frontier.append(index - 1)
        if x + 1 < width:
            frontier.append(index + 1)
        if y > 0:
            frontier.append(index - width)
        if y + 1 < height:
            frontier.append(index + width)
    enclosed_transparent = sum(value == 0 and index not in exterior for index, value in enumerate(support))
    has_interior_gap = enclosed_transparent >= max(2, nonzero // 20)
    has_area_support = fill_ratio >= 0.45
    return (
        "open_candidate"
        if height >= 4 and height / max(width, 1) >= 0.20 and (has_interior_gap or has_area_support)
        else "closed_or_ambiguous"
    )


def _validate_inputs(
    contract: AutoRigInputContract,
    component_plan: MaskComponentPlan,
    draw_order: OrdinaryDrawOrderPlan,
    base_regions: tuple[LoadedTextureRegion, ...],
    authored: NativeVariantSet,
    root: Path,
) -> dict[str, LoadedTextureRegion]:
    if root != contract.item_root.resolve(strict=True):
        raise GenericVariantSynthesisError("item root differs from the validated contract")
    if component_plan.native_variant_set_sha256 is not None or component_plan.variant_partitions:
        raise GenericVariantSynthesisError("generic synthesis requires the base component plan")
    if component_plan.canvas_edge != contract.canvas.resolution:
        raise GenericVariantSynthesisError("component canvas differs from the validated contract")
    if set(draw_order.ordinary_part_order) != {part.part_id for part in component_plan.parts}:
        raise GenericVariantSynthesisError("draw order differs from the base component plan")
    regions = {region.part_id: region for region in base_regions}
    expected = {part.source_part_id for part in component_plan.parts}
    if set(regions) != expected or any(region.source_kind != "see_through" for region in regions.values()):
        raise GenericVariantSynthesisError("base RGBA regions differ from component sources")
    if jcs_sha256(authored.semantic_payload()) != authored.native_variant_set_sha256:
        raise GenericVariantSynthesisError("authored variant-set digest is invalid")
    return regions


def build_generic_variant_synthesis_plan(
    contract: AutoRigInputContract,
    component_plan: MaskComponentPlan,
    draw_order: OrdinaryDrawOrderPlan,
    base_regions: tuple[LoadedTextureRegion, ...],
    authored_variant_set: NativeVariantSet,
    *,
    item_root: str | Path,
) -> tuple[NativeVariantSet, GenericVariantSynthesisPlan]:
    """Synthesize deterministic item-colored facial variants and merge authored roles."""

    root = Path(item_root).resolve(strict=True)
    regions = _validate_inputs(
        contract,
        component_plan,
        draw_order,
        base_regions,
        authored_variant_set,
        root,
    )
    parts = component_plan.parts
    face_parts = tuple(part for part in parts if part.base_tag == "face")
    mouth_parts = tuple(part for part in parts if part.base_tag == "mouth")
    mouth_state = _mouth_state(mouth_parts, regions)
    draw_ranks = {record.part_id: record.part_draw_rank for record in draw_order.records}
    authored_by_role = {entry.semantic_role: entry for entry in authored_variant_set.entries}
    generated: list[NativeVariantCandidate] = []
    records: list[GenericVariantSynthesisRecord] = []

    for role, mode in _TARGETS:
        authored_entry = authored_by_role.get(role)
        if authored_entry is None and role.startswith("eye_closed."):
            authored_entry = authored_by_role.get("eye_closed.coupled")
        if authored_entry is None and role == "mouth_closed":
            authored_entry = authored_by_role.get("mouth_open")
        if authored_entry is not None:
            records.append(
                GenericVariantSynthesisRecord(
                    semantic_role=role,
                    status="authored_override",
                    composite_mode=mode,
                    variant_id=authored_entry.variant_id,
                    source_part_ids=(),
                    xyxy=None,
                    output_path=None,
                    file_sha256=None,
                    reason=None,
                )
            )
            continue

        if role.startswith("eye_closed."):
            side = role.rsplit(".", 1)[1]
            base_parts = tuple(part for part in parts if part.base_tag in {"eyewhite", "irides", "eyelash"} and part.side == side)
            color_parts = tuple(part for part in base_parts if part.base_tag == "eyelash")
            available = {part.base_tag for part in base_parts} == {"eyewhite", "irides", "eyelash"}
            reason = None if available else "missing_layered_eye_parts"
        else:
            base_parts = mouth_parts
            color_parts = mouth_parts
            available = bool(base_parts and face_parts)
            reason = None if available else "missing_mouth_or_face_part"
            if role == "mouth_closed" and mouth_state != "open_candidate":
                available = False
                reason = "mouth_base_not_open"

        if not available:
            records.append(
                GenericVariantSynthesisRecord(
                    semantic_role=role,
                    status="unavailable",
                    composite_mode=mode,
                    variant_id=None,
                    source_part_ids=tuple(sorted(part.part_id for part in base_parts)),
                    xyxy=None,
                    output_path=None,
                    file_sha256=None,
                    reason=reason,
                )
            )
            continue

        padding = 2 if role.startswith("eye_closed.") or role == "mouth_closed" else 0
        xyxy = _union_bbox(base_parts, canvas_edge=component_plan.canvas_edge, padding=padding)
        size = (xyxy[2] - xyxy[0], xyxy[3] - xyxy[1])
        color = _alpha_weighted_color(color_parts, regions)
        if role.startswith("eye_closed."):
            image = _curve_layer(size, color=color, curvature=0.16, thickness_ratio=0.10)
        elif role == "mouth_closed":
            image = _curve_layer(size, color=color, curvature=0.02, thickness_ratio=0.10)
        else:
            image = _face_underlay(xyxy, face_parts, regions)
            if image is None:
                records.append(
                    GenericVariantSynthesisRecord(
                        semantic_role=role,
                        status="unavailable",
                        composite_mode=mode,
                        variant_id=None,
                        source_part_ids=tuple(sorted(part.part_id for part in base_parts)),
                        xyxy=None,
                        output_path=None,
                        file_sha256=None,
                        reason="face_underlay_empty",
                    )
                )
                continue
            smile = role == "mouth_smile"
            curvature = 0.50 if smile else -0.50
            image.alpha_composite(
                _curve_layer(
                    size,
                    color=color,
                    curvature=curvature,
                    thickness_ratio=0.16,
                    center_y_ratio=0.25 if smile else 0.75,
                )
            )
        candidate = _candidate(
            root=root,
            role=role,
            composite_mode=mode,
            image=image,
            xyxy=xyxy,
            base_parts=base_parts,
            draw_ranks=draw_ranks,
        )
        generated.append(candidate)
        records.append(
            GenericVariantSynthesisRecord(
                semantic_role=role,
                status="generated",
                composite_mode=mode,
                variant_id=candidate.variant_id,
                source_part_ids=tuple(sorted(part.part_id for part in base_parts)),
                xyxy=xyxy,
                output_path=candidate.relative_path,
                file_sha256=candidate.file_sha256,
                reason=None,
            )
        )

    merged = make_native_variant_set(
        present=authored_variant_set.present,
        manifest_path=authored_variant_set.manifest_path,
        entries=(*authored_variant_set.entries, *generated),
    )
    values = {
        "schema_version": GENERIC_VARIANT_SYNTHESIS_VERSION,
        "generator_version": GENERIC_VARIANT_GENERATOR_VERSION,
        "authored_variant_set_sha256": authored_variant_set.native_variant_set_sha256,
        "source_rgba_sha256": tuple(sorted(region.rgba_sha256 for region in base_regions)),
        "mouth_state_observation": mouth_state,
        "records": tuple(sorted(records, key=lambda record: record.semantic_role)),
        "generated_variant_ids": tuple(sorted(candidate.variant_id for candidate in generated)),
    }
    provisional = GenericVariantSynthesisPlan(**values, plan_sha256="")
    plan = GenericVariantSynthesisPlan(
        **values,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )
    return merged, plan


__all__ = [
    "GENERIC_VARIANT_GENERATOR_VERSION",
    "GENERIC_VARIANT_OUTPUT_DIRECTORY",
    "GENERIC_VARIANT_SYNTHESIS_VERSION",
    "GenericVariantSynthesisError",
    "GenericVariantSynthesisPlan",
    "GenericVariantSynthesisRecord",
    "build_generic_variant_synthesis_plan",
]
