from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from module.auto_rig.export.spine.atlas import (
    SPINE_ATLAS_PLAN_VERSION,
    SpineAtlasError,
    build_spine_atlas_plan,
    parse_spine_atlas,
    serialize_spine_atlas,
    validate_spine_atlas_plan,
)
from module.auto_rig.export.spine.symbols import build_spine_symbol_view
from module.auto_rig.export.spine.uv import (
    SPINE_42_UV_ADAPTER_VERSION,
    canvas_to_spine_region_uv,
)
from tests.test_auto_rig_rig_document import _build


def test_spine_atlas_is_deterministic_multi_region_straight_alpha(
    tmp_path: Path,
) -> None:
    *_, rig = _build(tmp_path)
    payload = rig.to_dict()
    symbols = build_spine_symbol_view(payload["export_symbols"])
    plan = build_spine_atlas_plan(
        payload["texture_pages"], payload["parts"], symbols
    )

    assert plan.schema_version == SPINE_ATLAS_PLAN_VERSION
    assert plan.uv_adapter_version == SPINE_42_UV_ADAPTER_VERSION
    assert [page.path for page in plan.pages] == ["textures/page_0.png"]
    assert all(page.pma is False for page in plan.pages)
    assert {region.part_id for page in plan.pages for region in page.regions} == {
        part["part_id"] for part in payload["parts"]
    }
    encoded = serialize_spine_atlas(plan)
    assert encoded.endswith(b"\n")
    assert b"\r" not in encoded
    assert b"pma: false" in encoded
    assert all(byte in b"\n" or 32 <= byte <= 126 for byte in encoded)
    parsed = parse_spine_atlas(encoded)
    assert [page.path for page in parsed.pages] == [page.path for page in plan.pages]
    assert [
        region.name for page in parsed.pages for region in page.regions
    ] == [region.name for page in plan.pages for region in page.regions]
    assert all(page.pma is False for page in parsed.pages)
    assert validate_spine_atlas_plan(plan, payload["texture_pages"], payload["parts"], symbols) is plan

    first_mesh = payload["meshes"][0]
    first_part = next(
        part for part in payload["parts"] if part["part_id"] == first_mesh["part_id"]
    )
    first_vertex = first_mesh["vertices"][0]
    assert canvas_to_spine_region_uv(first_vertex["position"], first_part["xyxy"]) == pytest.approx(
        first_vertex["uv"], abs=1e-9
    )


def test_spine_atlas_rejects_pma_rotation_and_noncanonical_page_index(
    tmp_path: Path,
) -> None:
    *_, rig = _build(tmp_path)
    payload = rig.to_dict()
    symbols = build_spine_symbol_view(payload["export_symbols"])
    plan = build_spine_atlas_plan(
        payload["texture_pages"], payload["parts"], symbols
    )
    with pytest.raises(SpineAtlasError, match="digest|pma"):
        validate_spine_atlas_plan(
            replace(
                plan,
                pages=(replace(plan.pages[0], pma=True), *plan.pages[1:]),
            ),
            payload["texture_pages"],
            payload["parts"],
            symbols,
        )

    changed_pages = [dict(page) for page in payload["texture_pages"]]
    changed_pages[0]["index"] = 1
    with pytest.raises(SpineAtlasError, match="consecutive|page"):
        build_spine_atlas_plan(changed_pages, payload["parts"], symbols)

    changed_pages = [dict(page) for page in payload["texture_pages"]]
    changed_pages[0]["placements"] = [
        {**placement, "rotation": True}
        for placement in changed_pages[0]["placements"]
    ]
    with pytest.raises(SpineAtlasError, match="rotation"):
        build_spine_atlas_plan(changed_pages, payload["parts"], symbols)
