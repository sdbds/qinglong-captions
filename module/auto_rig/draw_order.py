from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterable

from .component_plan import NormalizedMaskPart
from .jcs import jcs_sha256
from .tag_registry import V3_BASE_TAGS

DRAW_ORDER_POLICY_VERSION = "draw-order-policy-v1"
DEPTH_BUCKET_COUNT = 256
DEPTH_QUANTIZATION_RULE = "item-min-max-round-half-up-v1"
READY_SET_ORDER = "depth-bucket-desc-part-id-asc-v1"
HEAD_CORE_DRAWABLE_TAGS = frozenset(
    {"ears", "face", "eyewhite", "irides", "eyelash", "eyebrow", "nose", "mouth"}
)


def _builtin_edges() -> tuple[tuple[str, str], ...]:
    edges: set[tuple[str, str]] = set()
    for tag in HEAD_CORE_DRAWABLE_TAGS:
        edges.add(("back hair", tag))
        edges.add((tag, "front hair"))
    edges.update(
        {
            ("ears", "face"),
            ("face", "eyewhite"),
            ("face", "nose"),
            ("face", "mouth"),
            ("face", "eyebrow"),
            ("eyewhite", "irides"),
            ("irides", "eyelash"),
            ("ears", "earwear"),
        }
    )
    for tag in ("face", "eyewhite", "irides", "eyelash", "eyebrow"):
        edges.add((tag, "eyewear"))
    return tuple(sorted(edges))


BUILTIN_BASE_TAG_EDGES = _builtin_edges()


class DrawOrderPolicyError(ValueError):
    """Raised when the draw-order registry or an item graph is invalid."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class DrawOrderPolicyDescriptor:
    schema_version: str
    depth_bucket_count: int
    depth_quantization_rule: str
    ready_set_order: str
    base_tag_edges: tuple[tuple[str, str], ...]
    registry_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "depth_bucket_count": self.depth_bucket_count,
            "depth_quantization_rule": self.depth_quantization_rule,
            "ready_set_order": self.ready_set_order,
            "base_tag_edges": [list(edge) for edge in self.base_tag_edges],
            "registry_sha256": self.registry_sha256,
        }


@dataclass(frozen=True, slots=True)
class PartDrawOrderRecord:
    part_id: str
    base_tag: str
    depth_median: float
    depth_bucket: int
    part_draw_rank: int

    def to_dict(self) -> dict[str, object]:
        return {
            "part_id": self.part_id,
            "base_tag": self.base_tag,
            "depth_median": self.depth_median,
            "depth_bucket": self.depth_bucket,
            "part_draw_rank": self.part_draw_rank,
        }


@dataclass(frozen=True, slots=True)
class OrdinaryDrawOrderPlan:
    schema_version: str
    policy: DrawOrderPolicyDescriptor
    expanded_edges: tuple[tuple[str, str], ...]
    ordinary_part_order: tuple[str, ...]
    records: tuple[PartDrawOrderRecord, ...]
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy.to_dict(),
            "expanded_edges": [list(edge) for edge in self.expanded_edges],
            "ordinary_part_order": list(self.ordinary_part_order),
            "records": [record.to_dict() for record in self.records],
        }


def _error(message: str, *, code: str = "invalid_draw_order_registry") -> DrawOrderPolicyError:
    return DrawOrderPolicyError(code, message)


def _base_graph_is_acyclic(edges: tuple[tuple[str, str], ...]) -> bool:
    nodes = set(V3_BASE_TAGS)
    adjacency = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for behind, front in edges:
        if front not in adjacency[behind]:
            adjacency[behind].add(front)
            indegree[front] += 1
    ready = [node for node in nodes if indegree[node] == 0]
    heapq.heapify(ready)
    visited = 0
    while ready:
        node = heapq.heappop(ready)
        visited += 1
        for neighbor in sorted(adjacency[node]):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                heapq.heappush(ready, neighbor)
    return visited == len(nodes)


def validate_draw_order_registry(
    edges: Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    raw = tuple(edges)
    if any(
        type(edge) is not tuple
        or len(edge) != 2
        or not all(isinstance(tag, str) and tag in V3_BASE_TAGS for tag in edge)
        for edge in raw
    ):
        raise _error("draw-order edges must be registered v3 base-tag pairs")
    if len(raw) != len(set(raw)):
        raise _error("draw-order registry contains duplicate edges")
    normalized = tuple(sorted(raw))
    missing = sorted(set(BUILTIN_BASE_TAG_EDGES) - set(normalized))
    if missing:
        raise _error(f"draw-order registry is missing required edges: {missing}")
    if not _base_graph_is_acyclic(normalized):
        raise _error("draw-order base-tag registry contains a cycle")
    return normalized


_VALIDATED_BUILTIN_EDGES = validate_draw_order_registry(BUILTIN_BASE_TAG_EDGES)


def _policy_descriptor() -> DrawOrderPolicyDescriptor:
    payload = {
        "schema_version": DRAW_ORDER_POLICY_VERSION,
        "depth_bucket_count": DEPTH_BUCKET_COUNT,
        "depth_quantization_rule": DEPTH_QUANTIZATION_RULE,
        "ready_set_order": READY_SET_ORDER,
        "base_tag_edges": [list(edge) for edge in _VALIDATED_BUILTIN_EDGES],
    }
    return DrawOrderPolicyDescriptor(
        schema_version=DRAW_ORDER_POLICY_VERSION,
        depth_bucket_count=DEPTH_BUCKET_COUNT,
        depth_quantization_rule=DEPTH_QUANTIZATION_RULE,
        ready_set_order=READY_SET_ORDER,
        base_tag_edges=_VALIDATED_BUILTIN_EDGES,
        registry_sha256=jcs_sha256(payload),
    )


def _depth_buckets(parts: tuple[NormalizedMaskPart, ...]) -> dict[str, int]:
    depths = [part.depth_median for part in parts]
    if any(not math.isfinite(depth) for depth in depths):
        raise _error("Part depth must be finite", code="invalid_draw_order_input")
    minimum = min(depths)
    maximum = max(depths)
    if maximum == minimum:
        return {part.part_id: 0 for part in parts}
    span = maximum - minimum
    result: dict[str, int] = {}
    for part in parts:
        normalized = (part.depth_median - minimum) / span
        bucket = math.floor(normalized * (DEPTH_BUCKET_COUNT - 1) + 0.5)
        result[part.part_id] = min(DEPTH_BUCKET_COUNT - 1, max(0, bucket))
    return result


def _expanded_edges(
    parts: tuple[NormalizedMaskPart, ...],
) -> tuple[tuple[str, str], ...]:
    by_base_tag: dict[str, list[str]] = {}
    for part in parts:
        by_base_tag.setdefault(part.base_tag, []).append(part.part_id)
    expanded: set[tuple[str, str]] = set()
    for behind_tag, front_tag in _VALIDATED_BUILTIN_EDGES:
        for behind_id in sorted(by_base_tag.get(behind_tag, ())):
            for front_id in sorted(by_base_tag.get(front_tag, ())):
                expanded.add((behind_id, front_id))
    return tuple(sorted(expanded))


def _kahn_order(
    part_ids: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
    depth_buckets: dict[str, int],
) -> tuple[str, ...]:
    adjacency = {part_id: set() for part_id in part_ids}
    indegree = {part_id: 0 for part_id in part_ids}
    for behind, front in edges:
        if front not in adjacency[behind]:
            adjacency[behind].add(front)
            indegree[front] += 1
    ready = [(-depth_buckets[part_id], part_id) for part_id in part_ids if indegree[part_id] == 0]
    heapq.heapify(ready)
    ordered: list[str] = []
    while ready:
        _, part_id = heapq.heappop(ready)
        ordered.append(part_id)
        for neighbor in sorted(adjacency[part_id]):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                heapq.heappush(ready, (-depth_buckets[neighbor], neighbor))
    if len(ordered) != len(part_ids):
        raise _error("expanded item draw graph contains a cycle", code="invalid_draw_order_input")
    return tuple(ordered)


def build_ordinary_draw_order(
    parts: Iterable[NormalizedMaskPart],
) -> OrdinaryDrawOrderPlan:
    """Build the canonical back-to-front order for ordinary base Parts."""

    normalized_parts = tuple(sorted(parts, key=lambda item: item.part_id))
    if not normalized_parts:
        raise _error("draw ordering requires at least one Part", code="invalid_draw_order_input")
    part_ids = tuple(part.part_id for part in normalized_parts)
    if len(part_ids) != len(set(part_ids)):
        raise _error("draw-order Part IDs must be unique", code="invalid_draw_order_input")
    if any(part.base_tag not in V3_BASE_TAGS for part in normalized_parts):
        raise _error("draw-order Part has an unknown base tag", code="invalid_draw_order_input")

    buckets = _depth_buckets(normalized_parts)
    edges = _expanded_edges(normalized_parts)
    order = _kahn_order(part_ids, edges, buckets)
    part_by_id = {part.part_id: part for part in normalized_parts}
    records = tuple(
        PartDrawOrderRecord(
            part_id=part_id,
            base_tag=part_by_id[part_id].base_tag,
            depth_median=part_by_id[part_id].depth_median,
            depth_bucket=buckets[part_id],
            part_draw_rank=rank,
        )
        for rank, part_id in enumerate(order)
    )
    policy = _policy_descriptor()
    semantic_payload = {
        "schema_version": DRAW_ORDER_POLICY_VERSION,
        "policy": policy.to_dict(),
        "expanded_edges": [list(edge) for edge in edges],
        "ordinary_part_order": list(order),
        "records": [record.to_dict() for record in records],
    }
    return OrdinaryDrawOrderPlan(
        schema_version=DRAW_ORDER_POLICY_VERSION,
        policy=policy,
        expanded_edges=edges,
        ordinary_part_order=order,
        records=records,
        plan_sha256=jcs_sha256(semantic_payload),
    )


__all__ = [
    "BUILTIN_BASE_TAG_EDGES",
    "DEPTH_BUCKET_COUNT",
    "DEPTH_QUANTIZATION_RULE",
    "DRAW_ORDER_POLICY_VERSION",
    "HEAD_CORE_DRAWABLE_TAGS",
    "READY_SET_ORDER",
    "DrawOrderPolicyDescriptor",
    "DrawOrderPolicyError",
    "OrdinaryDrawOrderPlan",
    "PartDrawOrderRecord",
    "build_ordinary_draw_order",
    "validate_draw_order_registry",
]
