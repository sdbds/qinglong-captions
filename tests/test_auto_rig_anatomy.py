from __future__ import annotations

from pathlib import Path

import pytest

from module.auto_rig.anatomy import (
    ANATOMY_MASK_PLAN_VERSION,
    ANATOMY_MASK_REGISTRY_VERSION,
    AnatomyMaskError,
    build_anatomy_mask_geometry,
)
from module.auto_rig.component_geometry import load_component_geometry
from module.auto_rig.component_plan import (
    build_mask_component_plan,
    extend_mask_component_plan_with_variants,
)
from tests.test_auto_rig_component_plan import _loaded_part, _variant, _variant_set


def _part(
    base_tag: str,
    *,
    xyxy: tuple[int, int, int, int],
    points: set[tuple[int, int]],
    side: str | None = None,
):
    suffix = f".{side}" if side is not None else ""
    source_suffix = "-r" if side == "xmin" else "-l" if side == "xmax" else ""
    return _loaded_part(
        source_tag=f"{base_tag}{source_suffix}",
        base_tag=base_tag,
        semantic_slug=base_tag.replace(" ", "-"),
        part_id=f"part/{base_tag.replace(' ', '-')}{suffix}",
        side=side,
        xyxy=xyxy,
        points=points,
    )


def _rectangle(width: int, height: int) -> set[tuple[int, int]]:
    return {(x, y) for y in range(height) for x in range(width)}


def _geometry(tmp_path: Path, *parts):
    plan = build_mask_component_plan(parts, canvas_edge=768, item_root=tmp_path)
    return build_anatomy_mask_geometry(
        load_component_geometry(plan, item_root=tmp_path),
        canvas_edge=768,
    )


@pytest.mark.parametrize(
    ("parts", "expected"),
    (
        (
            (
                _part(
                    "handwear",
                    side="xmin",
                    xyxy=(10, 10, 14, 14),
                    points=_rectangle(4, 4),
                ),
                _part(
                    "handwear",
                    side="xmax",
                    xyxy=(30, 10, 34, 14),
                    points=_rectangle(4, 4),
                ),
            ),
            "split",
        ),
        (
            (
                _part(
                    "handwear",
                    side="xmin",
                    xyxy=(10, 10, 14, 14),
                    points=_rectangle(4, 4),
                ),
            ),
            "partial",
        ),
        (
            (
                _part(
                    "legwear",
                    xyxy=(10, 10, 34, 14),
                    points=_rectangle(4, 4)
                    | {(x, y) for y in range(4) for x in range(20, 24)},
                ),
            ),
            "merged-separable",
        ),
        (
            (
                _part(
                    "legwear",
                    xyxy=(10, 10, 14, 14),
                    points=_rectangle(4, 4),
                ),
            ),
            "merged-ambiguous",
        ),
        ((), "missing"),
    ),
)
def test_anatomy_plan_freezes_explicit_limb_observability_states(
    tmp_path: Path,
    parts: tuple,
    expected: str,
) -> None:
    base_parts = parts or (
        _part(
            "face",
            xyxy=(100, 100, 104, 104),
            points=_rectangle(4, 4),
        ),
    )
    anatomy = _geometry(tmp_path, *base_parts)
    family = "handwear" if expected in {"split", "partial"} else "legwear"
    state = {record.family: record for record in anatomy.plan.limb_states}[family]

    assert state.state == expected


def test_anatomy_metrics_exclude_hair_from_head_core_and_pose_body(
    tmp_path: Path,
) -> None:
    anatomy = _geometry(
        tmp_path,
        _part(
            "front hair",
            xyxy=(0, 0, 10, 10),
            points=_rectangle(10, 10),
        ),
        _part(
            "face",
            xyxy=(100, 50, 110, 60),
            points=_rectangle(10, 10),
        ),
        _part(
            "neck",
            xyxy=(103, 60, 107, 66),
            points=_rectangle(4, 6),
        ),
        _part(
            "topwear",
            xyxy=(90, 66, 120, 96),
            points=_rectangle(30, 30),
        ),
    )
    metrics = {record.metric_id: record for record in anatomy.plan.metrics}

    assert anatomy.plan.schema_version == ANATOMY_MASK_PLAN_VERSION
    assert anatomy.plan.registry.schema_version == ANATOMY_MASK_REGISTRY_VERSION
    assert metrics["mask/head_core"].bbox == (100, 50, 110, 60)
    assert metrics["mask/torso_core"].bbox == (90, 66, 120, 96)
    assert metrics["mask/neck"].bbox == (103, 60, 107, 66)
    assert metrics["mask/pose_body"].bbox == (90, 50, 120, 96)
    assert "part/front-hair" not in metrics["mask/head_core"].part_ids
    assert "part/front-hair" not in metrics["mask/pose_body"].part_ids


def test_anatomy_metrics_are_independent_of_native_variant_partitions(
    tmp_path: Path,
) -> None:
    base = build_mask_component_plan(
        (
            _part(
                "mouth",
                xyxy=(20, 30, 24, 34),
                points=_rectangle(4, 4),
            ),
        ),
        canvas_edge=768,
        item_root=tmp_path,
    )
    extended = extend_mask_component_plan_with_variants(
        base,
        _variant_set(_variant("mouth-open")),
        item_root=tmp_path,
    )

    ordinary = build_anatomy_mask_geometry(
        load_component_geometry(base, item_root=tmp_path),
        canvas_edge=768,
    )
    with_variant = build_anatomy_mask_geometry(
        load_component_geometry(extended, item_root=tmp_path),
        canvas_edge=768,
    )

    assert extended.variant_partitions
    assert with_variant.plan == ordinary.plan


def test_anatomy_rejects_duplicate_loaded_part_ids(tmp_path: Path) -> None:
    plan = build_mask_component_plan(
        (
            _part(
                "face",
                xyxy=(10, 10, 14, 14),
                points=_rectangle(4, 4),
            ),
        ),
        canvas_edge=768,
        item_root=tmp_path,
    )
    loaded = load_component_geometry(plan, item_root=tmp_path)

    with pytest.raises(AnatomyMaskError, match="duplicate"):
        build_anatomy_mask_geometry(loaded + loaded, canvas_edge=768)


def test_anatomy_rejects_untrusted_geometry_records() -> None:
    with pytest.raises(AnatomyMaskError, match="authenticated"):
        build_anatomy_mask_geometry((object(),), canvas_edge=768)
