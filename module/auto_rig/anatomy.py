from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Literal

from .component_geometry import LoadedComponentGeometry
from .jcs import jcs_sha256

ANATOMY_MASK_PLAN_VERSION = "anatomy-mask-plan-v1"
ANATOMY_MASK_REGISTRY_VERSION = "anatomy-mask-registry-v1"

HEAD_CORE_BASE_TAGS = frozenset(
    {
        "face",
        "irides",
        "eyebrow",
        "eyewhite",
        "eyelash",
        "ears",
        "nose",
        "mouth",
    }
)
TORSO_CORE_BASE_TAGS = frozenset({"topwear", "bottomwear"})
NECK_BASE_TAGS = frozenset({"neck"})
LIMB_MASK_FAMILIES = ("handwear", "legwear", "footwear")
POSE_BODY_BASE_TAGS = frozenset(
    HEAD_CORE_BASE_TAGS
    | TORSO_CORE_BASE_TAGS
    | NECK_BASE_TAGS
    | frozenset(LIMB_MASK_FAMILIES)
)

LimbState = Literal[
    "split",
    "partial",
    "merged-separable",
    "merged-ambiguous",
    "missing",
]


class AnatomyMaskError(ValueError):
    """Raised when normalized component geometry cannot form anatomy evidence."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class AnatomyMaskRegistry:
    schema_version: str
    head_core_base_tags: tuple[str, ...]
    torso_core_base_tags: tuple[str, ...]
    neck_base_tags: tuple[str, ...]
    pose_body_base_tags: tuple[str, ...]
    limb_families: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "head_core_base_tags": list(self.head_core_base_tags),
            "torso_core_base_tags": list(self.torso_core_base_tags),
            "neck_base_tags": list(self.neck_base_tags),
            "pose_body_base_tags": list(self.pose_body_base_tags),
            "limb_families": list(self.limb_families),
        }


@dataclass(frozen=True, slots=True)
class LimbStateRecord:
    family: str
    state: LimbState
    xmin_part_id: str | None
    xmax_part_id: str | None
    unsided_part_ids: tuple[str, ...]
    source_part_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "state": self.state,
            "xmin_part_id": self.xmin_part_id,
            "xmax_part_id": self.xmax_part_id,
            "unsided_part_ids": list(self.unsided_part_ids),
            "source_part_ids": list(self.source_part_ids),
        }


@dataclass(frozen=True, slots=True)
class AnatomyMaskMetric:
    metric_id: str
    status: Literal["available", "missing"]
    base_tags: tuple[str, ...]
    part_ids: tuple[str, ...]
    bbox: tuple[int, int, int, int] | None
    pixel_count: int
    x_sum: int
    y_sum: int
    cleaned_binary_mask_sha256: str | None

    @property
    def width(self) -> int:
        return 0 if self.bbox is None else self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return 0 if self.bbox is None else self.bbox[3] - self.bbox[1]

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_id": self.metric_id,
            "status": self.status,
            "base_tags": list(self.base_tags),
            "part_ids": list(self.part_ids),
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "pixel_count": self.pixel_count,
            "x_sum": self.x_sum,
            "y_sum": self.y_sum,
            "cleaned_binary_mask_sha256": self.cleaned_binary_mask_sha256,
        }


@dataclass(frozen=True, slots=True)
class AnatomyMaskPixels:
    metric_id: str
    bbox: tuple[int, int, int, int] | None
    width: int
    height: int
    binary_mask_u8: bytes


@dataclass(frozen=True, slots=True)
class AnatomyMaskPlan:
    schema_version: str
    canvas_edge: int
    registry: AnatomyMaskRegistry
    limb_states: tuple[LimbStateRecord, ...]
    metrics: tuple[AnatomyMaskMetric, ...]
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "canvas_edge": self.canvas_edge,
            "registry": self.registry.to_dict(),
            "limb_states": [record.to_dict() for record in self.limb_states],
            "metrics": [record.to_dict() for record in self.metrics],
        }


@dataclass(frozen=True, slots=True)
class AnatomyMaskGeometry:
    plan: AnatomyMaskPlan
    masks: tuple[AnatomyMaskPixels, ...]

    def mask(self, metric_id: str) -> AnatomyMaskPixels:
        for record in self.masks:
            if record.metric_id == metric_id:
                return record
        raise KeyError(metric_id)


def _error(message: str) -> AnatomyMaskError:
    return AnatomyMaskError("invalid_anatomy_mask_plan", message)


def _registry() -> AnatomyMaskRegistry:
    return AnatomyMaskRegistry(
        schema_version=ANATOMY_MASK_REGISTRY_VERSION,
        head_core_base_tags=tuple(sorted(HEAD_CORE_BASE_TAGS)),
        torso_core_base_tags=tuple(sorted(TORSO_CORE_BASE_TAGS)),
        neck_base_tags=tuple(sorted(NECK_BASE_TAGS)),
        pose_body_base_tags=tuple(sorted(POSE_BODY_BASE_TAGS)),
        limb_families=tuple(LIMB_MASK_FAMILIES),
    )


def _classify_limb(
    family: str,
    parts: tuple[LoadedComponentGeometry, ...],
) -> LimbStateRecord:
    family_parts = tuple(item for item in parts if item.part.base_tag == family)
    if not family_parts:
        return LimbStateRecord(
            family=family,
            state="missing",
            xmin_part_id=None,
            xmax_part_id=None,
            unsided_part_ids=(),
            source_part_ids=(),
        )
    by_side = {
        side: tuple(item for item in family_parts if item.part.side == side)
        for side in ("xmin", "xmax")
    }
    unsided = tuple(item for item in family_parts if item.part.side is None)
    if any(unsided) and any(by_side.values()):
        raise _error(f"limb family mixes sided and unsided Parts: {family}")
    if len(unsided) > 1 or any(len(items) > 1 for items in by_side.values()):
        raise _error(f"limb family has duplicate normalized sides: {family}")
    source_part_ids = tuple(sorted({item.part.source_part_id for item in family_parts}))
    if unsided:
        state: LimbState = "merged-ambiguous"
    elif not by_side["xmin"] or not by_side["xmax"]:
        state = "partial"
    else:
        xmin = by_side["xmin"][0].part
        xmax = by_side["xmax"][0].part
        provenance = {xmin.side_provenance, xmax.side_provenance}
        if provenance == {"source_tag"}:
            state = "split"
        elif provenance == {"component_pair"} and xmin.source_part_id == xmax.source_part_id:
            state = "merged-separable"
        else:
            raise _error(f"limb family has incoherent side provenance: {family}")
    return LimbStateRecord(
        family=family,
        state=state,
        xmin_part_id=by_side["xmin"][0].part.part_id if by_side["xmin"] else None,
        xmax_part_id=by_side["xmax"][0].part.part_id if by_side["xmax"] else None,
        unsided_part_ids=tuple(sorted(item.part.part_id for item in unsided)),
        source_part_ids=source_part_ids,
    )


def _compose_metric(
    metric_id: str,
    parts: tuple[LoadedComponentGeometry, ...],
    *,
    canvas_edge: int,
) -> tuple[AnatomyMaskMetric, AnatomyMaskPixels]:
    import numpy as np

    base_tags = tuple(sorted({item.part.base_tag for item in parts}))
    part_ids = tuple(sorted(item.part.part_id for item in parts))
    if not parts:
        metric = AnatomyMaskMetric(
            metric_id=metric_id,
            status="missing",
            base_tags=(),
            part_ids=(),
            bbox=None,
            pixel_count=0,
            x_sum=0,
            y_sum=0,
            cleaned_binary_mask_sha256=None,
        )
        return metric, AnatomyMaskPixels(
            metric_id=metric_id,
            bbox=None,
            width=0,
            height=0,
            binary_mask_u8=b"",
        )
    canvas = np.zeros((canvas_edge, canvas_edge), dtype=np.bool_)
    for item in parts:
        x1, y1, x2, y2 = item.part.xyxy
        if not (0 <= x1 < x2 <= canvas_edge and 0 <= y1 < y2 <= canvas_edge):
            raise _error(f"Part bbox escapes anatomy canvas: {item.part.part_id}")
        binary = np.asarray(item.labels, dtype=np.uint32).reshape(item.height, item.width) > 0
        canvas[y1:y2, x1:x2] |= binary
    ys, xs = np.nonzero(canvas)
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    tight = canvas[y1:y2, x1:x2].astype(np.uint8)
    payload = tight.tobytes(order="C")
    metric = AnatomyMaskMetric(
        metric_id=metric_id,
        status="available",
        base_tags=base_tags,
        part_ids=part_ids,
        bbox=(x1, y1, x2, y2),
        pixel_count=int(tight.sum()),
        x_sum=int(xs.sum()),
        y_sum=int(ys.sum()),
        cleaned_binary_mask_sha256=f"sha256:{hashlib.sha256(payload).hexdigest()}",
    )
    return metric, AnatomyMaskPixels(
        metric_id=metric_id,
        bbox=metric.bbox,
        width=x2 - x1,
        height=y2 - y1,
        binary_mask_u8=payload,
    )


def build_anatomy_mask_geometry(
    loaded_parts: Iterable[LoadedComponentGeometry],
    *,
    canvas_edge: int,
) -> AnatomyMaskGeometry:
    """Build versioned anatomy unions solely from authenticated ordinary QCL labels."""

    if isinstance(canvas_edge, bool) or not isinstance(canvas_edge, int) or canvas_edge <= 0:
        raise _error("canvas_edge must be a positive integer")
    untrusted = tuple(loaded_parts)
    if any(not isinstance(item, LoadedComponentGeometry) for item in untrusted):
        raise _error("anatomy geometry requires authenticated component records")
    parts = tuple(sorted(untrusted, key=lambda item: item.part.part_id))
    part_ids = tuple(item.part.part_id for item in parts)
    if len(part_ids) != len(set(part_ids)):
        raise _error("loaded anatomy geometry contains duplicate Part IDs")
    if any(part_id.startswith("part/native.") for part_id in part_ids):
        raise _error("NativeVariant Parts cannot enter anatomy geometry")
    registry = _registry()
    limb_states = tuple(_classify_limb(family, parts) for family in LIMB_MASK_FAMILIES)
    metric_inputs: list[tuple[str, tuple[LoadedComponentGeometry, ...]]] = [
        (
            "mask/head_core",
            tuple(item for item in parts if item.part.base_tag in HEAD_CORE_BASE_TAGS),
        ),
        (
            "mask/torso_core",
            tuple(item for item in parts if item.part.base_tag in TORSO_CORE_BASE_TAGS),
        ),
        (
            "mask/neck",
            tuple(item for item in parts if item.part.base_tag in NECK_BASE_TAGS),
        ),
        (
            "mask/pose_body",
            tuple(item for item in parts if item.part.base_tag in POSE_BODY_BASE_TAGS),
        ),
    ]
    for family in LIMB_MASK_FAMILIES:
        family_parts = tuple(item for item in parts if item.part.base_tag == family)
        for side in ("xmin", "xmax"):
            metric_inputs.append(
                (
                    f"mask/limb/{family}.{side}",
                    tuple(item for item in family_parts if item.part.side == side),
                )
            )
        metric_inputs.append(
            (
                f"mask/limb/{family}.merged",
                tuple(item for item in family_parts if item.part.side is None),
            )
        )

    metrics: list[AnatomyMaskMetric] = []
    masks: list[AnatomyMaskPixels] = []
    for metric_id, selected in metric_inputs:
        metric, mask = _compose_metric(metric_id, selected, canvas_edge=canvas_edge)
        metrics.append(metric)
        masks.append(mask)
    values = {
        "schema_version": ANATOMY_MASK_PLAN_VERSION,
        "canvas_edge": canvas_edge,
        "registry": registry,
        "limb_states": limb_states,
        "metrics": tuple(metrics),
    }
    provisional = AnatomyMaskPlan(**values, plan_sha256="")
    plan = AnatomyMaskPlan(
        **values,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )
    return AnatomyMaskGeometry(plan=plan, masks=tuple(masks))


__all__ = [
    "ANATOMY_MASK_PLAN_VERSION",
    "ANATOMY_MASK_REGISTRY_VERSION",
    "HEAD_CORE_BASE_TAGS",
    "LIMB_MASK_FAMILIES",
    "NECK_BASE_TAGS",
    "POSE_BODY_BASE_TAGS",
    "TORSO_CORE_BASE_TAGS",
    "AnatomyMaskError",
    "AnatomyMaskGeometry",
    "AnatomyMaskMetric",
    "AnatomyMaskPixels",
    "AnatomyMaskPlan",
    "AnatomyMaskRegistry",
    "LimbStateRecord",
    "build_anatomy_mask_geometry",
]
