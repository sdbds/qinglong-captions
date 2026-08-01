from __future__ import annotations

from pathlib import Path

import pytest

from module.auto_rig.artifacts import FileDigest
from module.auto_rig.component_plan import MaskComponentRecord, NormalizedMaskPart
from module.auto_rig.draw_order import (
    BUILTIN_BASE_TAG_EDGES,
    HEAD_CORE_DRAWABLE_TAGS,
    DrawOrderPolicyError,
    build_ordinary_draw_order,
    validate_draw_order_registry,
)

_SHA256_ZERO = "sha256:" + "0" * 64


def _part(
    part_id: str,
    base_tag: str,
    depth: float,
    *,
    side: str | None = None,
) -> NormalizedMaskPart:
    component_id = "component/c_" + (part_id.encode("ascii").hex() + "0" * 64)[:64]
    bbox = (0, 0, 2, 2)
    component = MaskComponentRecord(
        component_id=component_id,
        label=1,
        part_id=part_id,
        side=side,
        bbox=bbox,
        cleaned_binary_mask_sha256=_SHA256_ZERO,
        pixel_count=4,
    )
    return NormalizedMaskPart(
        part_id=part_id,
        source_part_id=part_id,
        source_tag=base_tag,
        base_tag=base_tag,
        semantic_slug=base_tag.replace(" ", "-"),
        side=side,
        side_provenance="source_tag" if side is not None else "none",
        source_xyxy=bbox,
        xyxy=bbox,
        depth_median=depth,
        cleaned_binary_mask_sha256=_SHA256_ZERO,
        qcl_file=FileDigest(
            path=f"rig/cache/A/components/{component_id.removeprefix('component/c_')}.qcl",
            size=28,
            sha256=_SHA256_ZERO,
        ),
        components=(component,),
    )


def _ordered_ids(*parts: NormalizedMaskPart) -> tuple[str, ...]:
    return build_ordinary_draw_order(parts).ordinary_part_order


def test_draw_order_uses_farther_depth_first_without_semantic_edges() -> None:
    front = _part("part/topwear", "topwear", 0.1)
    back = _part("part/bottomwear", "bottomwear", 0.9)

    plan = build_ordinary_draw_order((front, back))

    assert plan.ordinary_part_order == ("part/bottomwear", "part/topwear")
    assert tuple(record.part_draw_rank for record in plan.records) == (0, 1)
    assert {record.part_id: record.depth_bucket for record in plan.records} == {
        "part/bottomwear": 255,
        "part/topwear": 0,
    }


def test_draw_order_equal_depth_uses_stable_part_id_and_zero_bucket() -> None:
    z = _part("part/wings", "wings", 0.5)
    a = _part("part/objects", "objects", 0.5)

    plan = build_ordinary_draw_order((z, a))

    assert plan.ordinary_part_order == ("part/objects", "part/wings")
    assert {record.depth_bucket for record in plan.records} == {0}


def test_semantic_head_edges_override_deliberately_conflicting_depth() -> None:
    back_hair = _part("part/back-hair", "back hair", 0.0)
    face = _part("part/face", "face", 0.5)
    front_hair = _part("part/front-hair", "front hair", 1.0)

    assert _ordered_ids(front_hair, face, back_hair) == (
        "part/back-hair",
        "part/face",
        "part/front-hair",
    )


def test_eyewear_edges_override_depth_for_every_near_coplanar_face_part() -> None:
    base_tags = ("face", "eyewhite", "irides", "eyelash", "eyebrow")
    core = tuple(_part(f"part/{tag}", tag, 0.0) for tag in base_tags)
    eyewear = _part("part/eyewear", "eyewear", 1.0)

    order = _ordered_ids(eyewear, *core)

    eyewear_rank = order.index("part/eyewear")
    assert all(order.index(f"part/{tag}") < eyewear_rank for tag in base_tags)
    for tag in base_tags:
        assert (tag, "eyewear") in BUILTIN_BASE_TAG_EDGES


def test_base_tag_edges_expand_over_split_parts_without_phantom_nodes() -> None:
    white_min = _part("part/eyewhite.xmin", "eyewhite", 0.0, side="xmin")
    white_max = _part("part/eyewhite.xmax", "eyewhite", 0.0, side="xmax")
    iris_min = _part("part/irides.xmin", "irides", 1.0, side="xmin")
    iris_max = _part("part/irides.xmax", "irides", 1.0, side="xmax")

    plan = build_ordinary_draw_order((iris_min, white_max, iris_max, white_min))

    white_ranks = [plan.ordinary_part_order.index(part.part_id) for part in (white_min, white_max)]
    iris_ranks = [plan.ordinary_part_order.index(part.part_id) for part in (iris_min, iris_max)]
    assert max(white_ranks) < min(iris_ranks)
    assert set(plan.ordinary_part_order) == {
        "part/eyewhite.xmin",
        "part/eyewhite.xmax",
        "part/irides.xmin",
        "part/irides.xmax",
    }


def test_headwear_intentionally_has_no_hard_edge_and_follows_depth() -> None:
    headwear = _part("part/headwear", "headwear", 1.0)
    front_hair = _part("part/front-hair", "front hair", 0.0)

    plan = build_ordinary_draw_order((front_hair, headwear))

    assert plan.ordinary_part_order == ("part/headwear", "part/front-hair")
    assert not any("headwear" in edge for edge in BUILTIN_BASE_TAG_EDGES)


def test_registry_is_exactly_versioned_and_rejects_missing_required_edge() -> None:
    assert HEAD_CORE_DRAWABLE_TAGS == frozenset(
        {"ears", "face", "eyewhite", "irides", "eyelash", "eyebrow", "nose", "mouth"}
    )
    missing_eyewear = tuple(
        edge for edge in BUILTIN_BASE_TAG_EDGES if edge != ("eyelash", "eyewear")
    )

    with pytest.raises(DrawOrderPolicyError) as captured:
        validate_draw_order_registry(missing_eyewear)

    assert captured.value.code == "invalid_draw_order_registry"


def test_registry_cycle_is_a_startup_error() -> None:
    cyclic = (*BUILTIN_BASE_TAG_EDGES, ("front hair", "back hair"))

    with pytest.raises(DrawOrderPolicyError) as captured:
        validate_draw_order_registry(cyclic)

    assert captured.value.code == "invalid_draw_order_registry"


def test_draw_order_digest_is_independent_of_input_iteration_order() -> None:
    parts = (
        _part("part/back-hair", "back hair", 0.8),
        _part("part/face", "face", 0.2),
        _part("part/front-hair", "front hair", 0.5),
        _part("part/topwear", "topwear", 0.6),
    )

    forward = build_ordinary_draw_order(parts)
    reversed_plan = build_ordinary_draw_order(reversed(parts))

    assert reversed_plan == forward
    assert reversed_plan.plan_sha256 == forward.plan_sha256


def test_depth_quantization_uses_round_half_up_across_256_buckets() -> None:
    minimum = _part("part/objects", "objects", 0.0)
    midpoint = _part("part/tail", "tail", 0.5)
    maximum = _part("part/wings", "wings", 1.0)

    records = {
        record.part_id: record
        for record in build_ordinary_draw_order((minimum, midpoint, maximum)).records
    }

    assert records["part/objects"].depth_bucket == 0
    assert records["part/tail"].depth_bucket == 128
    assert records["part/wings"].depth_bucket == 255
