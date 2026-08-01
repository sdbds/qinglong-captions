from __future__ import annotations

import re
from pathlib import Path

import pytest

import module.auto_rig.component_plan as component_plan_module
from module.auto_rig.component_plan import (
    MASK_COMPONENT_ID_SCHEMA,
    MaskComponentPlanError,
    build_mask_component_plan,
    cleanup_area_threshold,
)
from module.auto_rig.contracts import AutoRigPartContract, ValidatedPartSource
from module.auto_rig.jcs import jcs_sha256
from module.auto_rig.mask_sources import LoadedPartAlpha
from module.auto_rig.qcl import decode_qcl

_SHA256_ZERO = "sha256:" + "0" * 64


def _alpha(width: int, height: int, points: set[tuple[int, int]]) -> bytes:
    values = bytearray(width * height)
    for x, y in points:
        values[y * width + x] = 255
    return bytes(values)


def _loaded_part(
    *,
    source_tag: str,
    base_tag: str,
    semantic_slug: str,
    part_id: str,
    side: str | None,
    xyxy: tuple[int, int, int, int],
    points: set[tuple[int, int]],
    depth: float = 0.5,
) -> LoadedPartAlpha:
    width = xyxy[2] - xyxy[0]
    height = xyxy[3] - xyxy[1]
    part = AutoRigPartContract(
        source_tag=source_tag,
        base_tag=base_tag,
        semantic_slug=semantic_slug,
        side=side,
        part_id=part_id,
        xyxy=xyxy,
        depth_median=depth,
        source=ValidatedPartSource(
            mode="png",
            color_path=Path("unused-color.png"),
            depth_path=Path("unused-depth.png"),
            color_sha256=_SHA256_ZERO,
            depth_sha256=_SHA256_ZERO,
            layer_name=None,
        ),
    )
    return LoadedPartAlpha(
        part=part,
        width=width,
        height=height,
        alpha_u8=_alpha(width, height, points),
    )


def _qcl_bytes(item_root: Path, relative_path: str) -> bytes:
    return (item_root / Path(*relative_path.split("/"))).read_bytes()


def test_component_plan_cleans_tiny_noise_fills_holes_and_tightens_crop(
    tmp_path: Path,
) -> None:
    ring = {
        (2, 2),
        (3, 2),
        (4, 2),
        (2, 3),
        (4, 3),
        (2, 4),
        (3, 4),
        (4, 4),
    }
    loaded = _loaded_part(
        source_tag="face",
        base_tag="face",
        semantic_slug="face",
        part_id="part/face",
        side=None,
        xyxy=(100, 200, 110, 210),
        points=ring | {(9, 9)},
    )

    plan = build_mask_component_plan((loaded,), canvas_edge=768, item_root=tmp_path)

    assert plan.cleanup.min_component_area_px == 4
    assert plan.cleanup.max_hole_area_px == 4
    assert plan.projected_component_count == 1
    part = plan.parts[0]
    assert part.xyxy == (102, 202, 105, 205)
    assert part.components[0].pixel_count == 9
    label_map = decode_qcl(_qcl_bytes(tmp_path, part.qcl_file.path))
    assert (label_map.width, label_map.height) == (3, 3)
    assert label_map.labels == (1,) * 9


def test_component_plan_uses_eight_connectivity_and_keeps_components_separate(
    tmp_path: Path,
) -> None:
    loaded = _loaded_part(
        source_tag="front hair",
        base_tag="front hair",
        semantic_slug="front-hair",
        part_id="part/front-hair",
        side=None,
        xyxy=(0, 0, 12, 8),
        points={(1, 1), (2, 2), (3, 3), (4, 4), (8, 1), (8, 2), (9, 1), (9, 2)},
    )

    plan = build_mask_component_plan((loaded,), canvas_edge=1024, item_root=tmp_path)

    assert len(plan.parts) == 1
    assert len(plan.parts[0].components) == 2
    assert {component.side for component in plan.parts[0].components} == {None}
    assert set(decode_qcl(_qcl_bytes(tmp_path, plan.parts[0].qcl_file.path)).labels) == {
        0,
        1,
        2,
    }


def test_component_plan_rejects_a_mask_that_is_empty_after_cleanup(tmp_path: Path) -> None:
    loaded = _loaded_part(
        source_tag="face",
        base_tag="face",
        semantic_slug="face",
        part_id="part/face",
        side=None,
        xyxy=(0, 0, 4, 4),
        points={(1, 1)},
    )

    with pytest.raises(MaskComponentPlanError) as captured:
        build_mask_component_plan((loaded,), canvas_edge=768, item_root=tmp_path)

    assert captured.value.code == "empty_cleaned_mask"


def test_component_plan_promotes_exactly_two_unsplit_family_components_to_image_sides(
    tmp_path: Path,
) -> None:
    loaded = _loaded_part(
        source_tag="handwear",
        base_tag="handwear",
        semantic_slug="handwear",
        part_id="part/handwear",
        side=None,
        xyxy=(10, 20, 22, 24),
        points={(0, 0), (1, 0), (0, 1), (1, 1), (9, 1), (10, 1), (9, 2), (10, 2)},
    )

    plan = build_mask_component_plan((loaded,), canvas_edge=1024, item_root=tmp_path)

    assert tuple(part.part_id for part in plan.parts) == (
        "part/handwear.xmax",
        "part/handwear.xmin",
    )
    by_side = {part.side: part for part in plan.parts}
    assert by_side["xmin"].xyxy == (10, 20, 12, 22)
    assert by_side["xmax"].xyxy == (19, 21, 21, 23)
    assert {part.side_provenance for part in plan.parts} == {"component_pair"}
    assert by_side["xmin"].components[0].side == "xmin"
    assert by_side["xmax"].components[0].side == "xmax"


def test_component_plan_preserves_source_side_for_every_component(tmp_path: Path) -> None:
    loaded = _loaded_part(
        source_tag="handwear-r",
        base_tag="handwear",
        semantic_slug="handwear",
        part_id="part/handwear.xmin",
        side="xmin",
        xyxy=(0, 0, 12, 4),
        points={(0, 0), (1, 0), (0, 1), (1, 1), (9, 1), (10, 1), (9, 2), (10, 2)},
    )

    plan = build_mask_component_plan((loaded,), canvas_edge=1024, item_root=tmp_path)

    assert tuple(part.part_id for part in plan.parts) == ("part/handwear.xmin",)
    assert plan.parts[0].side_provenance == "source_tag"
    assert {component.side for component in plan.parts[0].components} == {"xmin"}


def test_component_plan_does_not_promote_a_tiny_surviving_speck_to_a_side(
    tmp_path: Path,
) -> None:
    dominant = {(x, y) for y in range(10) for x in range(10)}
    loaded = _loaded_part(
        source_tag="handwear",
        base_tag="handwear",
        semantic_slug="handwear",
        part_id="part/handwear",
        side=None,
        xyxy=(0, 0, 20, 10),
        points=dominant | {(18, 0), (19, 0), (18, 1), (19, 1)},
    )

    plan = build_mask_component_plan((loaded,), canvas_edge=1024, item_root=tmp_path)

    assert tuple(part.part_id for part in plan.parts) == ("part/handwear",)
    assert {component.side for component in plan.parts[0].components} == {None}


@pytest.mark.parametrize(
    ("base_tag", "semantic_slug", "part_id", "expected_part_ids"),
    (
        ("handwear", "handwear", "part/handwear", ("part/handwear",)),
        ("front hair", "front-hair", "part/front-hair", ("part/front-hair",)),
    ),
)
def test_component_plan_keeps_ambiguous_or_non_split_multi_component_parts_unsided(
    tmp_path: Path,
    base_tag: str,
    semantic_slug: str,
    part_id: str,
    expected_part_ids: tuple[str, ...],
) -> None:
    loaded = _loaded_part(
        source_tag=base_tag,
        base_tag=base_tag,
        semantic_slug=semantic_slug,
        part_id=part_id,
        side=None,
        xyxy=(0, 0, 14, 4),
        points={
            (0, 0), (1, 0), (0, 1), (1, 1),
            (5, 0), (6, 0), (5, 1), (6, 1),
            (10, 0), (11, 0), (10, 1), (11, 1),
        },
    )

    plan = build_mask_component_plan((loaded,), canvas_edge=1024, item_root=tmp_path)

    assert tuple(part.part_id for part in plan.parts) == expected_part_ids
    assert {component.side for component in plan.parts[0].components} == {None}


def test_component_ids_use_full_jcs_identity_digest(tmp_path: Path) -> None:
    loaded = _loaded_part(
        source_tag="face",
        base_tag="face",
        semantic_slug="face",
        part_id="part/face",
        side=None,
        xyxy=(10, 20, 14, 24),
        points={(1, 1), (2, 1), (1, 2), (2, 2)},
    )

    component = build_mask_component_plan(
        (loaded,), canvas_edge=1024, item_root=tmp_path
    ).parts[0].components[0]
    identity = {
        "schema": MASK_COMPONENT_ID_SCHEMA,
        "part_id": "part/face",
        "component_bbox": list(component.bbox),
        "cleaned_binary_mask_sha256": component.cleaned_binary_mask_sha256,
    }
    expected = "component/c_" + jcs_sha256(identity).removeprefix("sha256:")

    assert component.component_id == expected
    assert re.fullmatch(r"component/c_[0-9a-f]{64}", component.component_id)


def test_component_plan_is_independent_of_source_and_library_label_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    face = _loaded_part(
        source_tag="face",
        base_tag="face",
        semantic_slug="face",
        part_id="part/face",
        side=None,
        xyxy=(0, 0, 8, 4),
        points={(0, 0), (1, 0), (0, 1), (1, 1), (6, 1), (7, 1), (6, 2), (7, 2)},
    )
    hair = _loaded_part(
        source_tag="front hair",
        base_tag="front hair",
        semantic_slug="front-hair",
        part_id="part/front-hair",
        side=None,
        xyxy=(20, 10, 24, 14),
        points={(1, 1), (2, 1), (1, 2), (2, 2)},
    )
    normal_root = tmp_path / "normal"
    normal_root.mkdir()
    normal = build_mask_component_plan(
        (face, hair), canvas_edge=1024, item_root=normal_root
    )
    original_label = component_plan_module._label_components

    def reverse_positive_labels(mask):
        labels = original_label(mask)
        maximum = int(labels.max())
        reversed_labels = labels.copy()
        positive = reversed_labels > 0
        reversed_labels[positive] = maximum + 1 - reversed_labels[positive]
        return reversed_labels

    monkeypatch.setattr(component_plan_module, "_label_components", reverse_positive_labels)
    reversed_root = tmp_path / "reversed"
    reversed_root.mkdir()
    reversed_plan = build_mask_component_plan(
        (hair, face), canvas_edge=1024, item_root=reversed_root
    )

    assert reversed_plan.plan_sha256 == normal.plan_sha256
    assert reversed_plan.parts == normal.parts
    for part in normal.parts:
        assert _qcl_bytes(normal_root, part.qcl_file.path) == _qcl_bytes(
            reversed_root,
            part.qcl_file.path,
        )


@pytest.mark.parametrize(
    ("edge", "expected"),
    ((768, 4), (1024, 4), (1280, 6)),
)
def test_cleanup_area_threshold_is_resolution_scaled(edge: int, expected: int) -> None:
    assert cleanup_area_threshold(edge) == expected
