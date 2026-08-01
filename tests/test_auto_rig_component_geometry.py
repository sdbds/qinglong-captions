from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from module.auto_rig.component_geometry import (
    COMPONENT_GEOMETRY_LOADER_VERSION,
    ComponentGeometryError,
    load_component_geometry,
)
from module.auto_rig.component_plan import build_mask_component_plan
from module.auto_rig.jcs import jcs_sha256
from tests.test_auto_rig_component_plan import _loaded_part


def _face(*, x: int = 10):
    return _loaded_part(
        source_tag="face",
        base_tag="face",
        semantic_slug="face",
        part_id="part/face",
        side=None,
        xyxy=(x, 20, x + 6, 26),
        points={(1, 1), (2, 1), (1, 2), (2, 2)},
    )


def _topwear():
    return _loaded_part(
        source_tag="topwear",
        base_tag="topwear",
        semantic_slug="topwear",
        part_id="part/topwear",
        side=None,
        xyxy=(30, 40, 36, 46),
        points={(2, 2), (3, 2), (2, 3), (3, 3)},
    )


def test_component_geometry_authenticates_qcl_and_loads_canonical_part_order(
    tmp_path: Path,
) -> None:
    plan = build_mask_component_plan(
        (_topwear(), _face()),
        canvas_edge=768,
        item_root=tmp_path,
    )

    geometry = load_component_geometry(plan, item_root=tmp_path)

    assert COMPONENT_GEOMETRY_LOADER_VERSION == "component-geometry-loader-v1"
    assert tuple(item.part.part_id for item in geometry) == (
        "part/face",
        "part/topwear",
    )
    assert geometry[0].width == 2
    assert geometry[0].height == 2
    assert geometry[0].labels == (1, 1, 1, 1)


def test_component_geometry_rejects_qcl_bytes_changed_after_planning(
    tmp_path: Path,
) -> None:
    plan = build_mask_component_plan((_face(),), canvas_edge=768, item_root=tmp_path)
    qcl_path = tmp_path / Path(*plan.parts[0].qcl_file.path.split("/"))
    qcl_path.write_bytes(qcl_path.read_bytes() + b"changed")

    with pytest.raises(ComponentGeometryError) as captured:
        load_component_geometry(plan, item_root=tmp_path)

    assert captured.value.code == "input_contract_mismatch"


def test_component_geometry_rejects_a_stale_plan_digest(tmp_path: Path) -> None:
    plan = build_mask_component_plan((_face(),), canvas_edge=768, item_root=tmp_path)

    with pytest.raises(ComponentGeometryError) as captured:
        load_component_geometry(
            replace(plan, plan_sha256="sha256:" + "0" * 64),
            item_root=tmp_path,
        )

    assert captured.value.code == "invalid_component_plan"


def test_component_geometry_rejects_label_records_that_disagree_with_qcl(
    tmp_path: Path,
) -> None:
    plan = build_mask_component_plan((_face(),), canvas_edge=768, item_root=tmp_path)
    part = plan.parts[0]
    component = replace(
        part.components[0],
        pixel_count=part.components[0].pixel_count + 1,
    )
    provisional = replace(
        plan,
        parts=(replace(part, components=(component,)),),
        plan_sha256="",
    )
    tampered = replace(
        provisional,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )

    with pytest.raises(ComponentGeometryError, match="pixel count") as captured:
        load_component_geometry(tampered, item_root=tmp_path)

    assert captured.value.code == "invalid_component_plan"
