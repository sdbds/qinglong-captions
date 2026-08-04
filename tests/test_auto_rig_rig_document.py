from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from module.auto_rig.artifacts import FileDigest
from module.auto_rig.format_plans import build_format_plan_set
from module.auto_rig.jcs import jcs_sha256
from module.auto_rig.rig_document import (
    RIG_DOCUMENT_SCHEMA_VERSION,
    RigDocumentError,
    _public_meshes,
    _public_parts,
    build_rig_document,
    load_rig_document,
    rig_document_bytes,
    validate_rig_document,
    validate_rig_document_payload,
)
from module.auto_rig.texture_plan import (
    CANONICAL_PNG_ENCODER_VERSION,
    CANONICAL_TEXTURE_PAGE_SET_VERSION,
    CanonicalPngEncoderDescriptor,
    CanonicalTexturePageSet,
    MaterializedTexturePage,
    TextureRegionInput,
    build_texture_page_plan,
)
from module.auto_rig.texture_sources import TEXTURE_PIXEL_CONTRACT_VERSION
from tests.test_auto_rig_format_plans import _fixture


def _texture_inputs(cache):
    return tuple(
        TextureRegionInput(
            part_id=part.part_id,
            source_kind=part.source_kind,
            variant_id=part.variant_id,
            source_xyxy=part.xyxy,
            width=part.xyxy[2] - part.xyxy[0],
            height=part.xyxy[3] - part.xyxy[1],
            rgba_sha256=jcs_sha256({"part_id": part.part_id, "rgba": "fixture"}),
            alpha_mode="straight",
            color_space="srgb_bytes",
            pixel_contract_version=TEXTURE_PIXEL_CONTRACT_VERSION,
        )
        for part in cache.parts
    )


def _texture_pages(cache):
    plan = build_texture_page_plan(_texture_inputs(cache))
    assert plan.fit and len(plan.pages) == 1
    page = plan.pages[0]
    encoded_sha = jcs_sha256({"page": page.index, "encoded": "fixture"})
    materialized = MaterializedTexturePage(
        index=page.index,
        width=plan.page_width,
        height=plan.page_height,
        relative_path=f"rig/shared/textures/{page.relative_path}",
        rgba_sha256=jcs_sha256({"page": page.index, "rgba": "fixture"}),
        encoded_png_sha256=encoded_sha,
        file=FileDigest(
            path=f"rig/shared/textures/{page.relative_path}",
            size=123,
            sha256=encoded_sha,
        ),
    )
    encoder = CanonicalPngEncoderDescriptor(
        schema_version=CANONICAL_PNG_ENCODER_VERSION,
        pillow_version="fixture",
        zlib_runtime_version="fixture",
        optimize=False,
        compress_level=9,
        metadata_policy="none",
        pixel_contract_version=TEXTURE_PIXEL_CONTRACT_VERSION,
        alpha_mode="straight",
        color_space="srgb_bytes",
    )
    provisional = CanonicalTexturePageSet(
        schema_version=CANONICAL_TEXTURE_PAGE_SET_VERSION,
        texture_page_plan_sha256=plan.plan_sha256,
        encoder=encoder,
        pages=(materialized,),
        set_sha256="",
    )
    page_set = replace(
        provisional,
        set_sha256=jcs_sha256(provisional.semantic_payload()),
    )
    return plan, page_set


def _build(tmp_path: Path, **fixture_kwargs):
    cache, controls, presets, capabilities, bindings, candidates, symbols = _fixture(
        tmp_path,
        **fixture_kwargs,
    )
    texture_plan, page_set = _texture_pages(cache)
    assert candidates.texture_page_ids == ("texture-page/page_0",)
    formats = build_format_plan_set(
        cache,
        controls,
        presets,
        capabilities,
        bindings,
        candidates,
        symbols,
        profile_id="dual_runtime_core_v1",
    )
    rig = build_rig_document(
        cache,
        controls,
        presets,
        capabilities,
        bindings,
        formats,
        candidates,
        symbols,
        texture_plan,
        page_set,
    )
    return (
        cache,
        controls,
        presets,
        capabilities,
        bindings,
        formats,
        candidates,
        symbols,
        texture_plan,
        page_set,
        rig,
    )


def test_complete_rig_document_round_trips_canonical_bytes(tmp_path: Path) -> None:
    *inputs, rig = _build(tmp_path)

    assert rig.schema_version == RIG_DOCUMENT_SCHEMA_VERSION
    assert validate_rig_document(rig) is rig
    encoded = rig_document_bytes(rig)
    path = tmp_path / "rig.json"
    path.write_bytes(encoded)
    loaded = load_rig_document(path)

    assert loaded == rig
    assert rig_document_bytes(loaded) == encoded
    assert build_rig_document(*inputs) == rig


def test_public_rig_contains_complete_groups_without_private_qcl_paths(
    tmp_path: Path,
) -> None:
    cache, *_rest, rig = _build(tmp_path)
    payload = rig.to_dict()

    for field in (
        "capabilities",
        "control_specs",
        "control_bindings",
        "clips",
        "expressions",
        "format_plans",
        "primitive_candidates",
        "export_symbols",
        "texture_pages",
    ):
        assert payload[field]
    assert b".qcl" not in rig_document_bytes(rig)
    assert [part["part_draw_rank"] for part in payload["parts"]] == list(range(len(cache.parts)))
    assert [mesh["component_draw_rank"] for mesh in payload["meshes"]] == list(range(len(cache.skinning_plan.weighted_meshes)))


def test_public_rig_omits_component_draw_records_without_a_mesh(tmp_path: Path) -> None:
    cache, *_rest = _fixture(tmp_path)
    omitted = cache.component_draw_order.records[0]
    component_draw = replace(
        cache.component_draw_order,
        records=(
            replace(omitted, mesh_id=None),
            *cache.component_draw_order.records[1:],
        ),
    )
    skinning = replace(
        cache.skinning_plan,
        weighted_meshes=tuple(mesh for mesh in cache.skinning_plan.weighted_meshes if mesh.component_id != omitted.component_id),
    )
    degraded = replace(
        cache,
        component_draw_order=component_draw,
        skinning_plan=skinning,
    )

    meshes = _public_meshes(degraded)
    parts = _public_parts(
        degraded,
        native_quality_by_part={},
        native_composite_mode_by_part={},
    )

    assert omitted.component_id not in {mesh["component_id"] for mesh in meshes}
    assert [mesh["component_draw_rank"] for mesh in meshes] == list(range(len(meshes)))
    assert omitted.component_id not in {component_id for part in parts for component_id in part["component_ids"]}


@pytest.mark.parametrize(
    "mutation",
    ("missing_group", "extra_field", "bone_parent", "binding_target", "symbol_name"),
)
def test_payload_validator_rejects_partial_and_broken_reference_closure(
    tmp_path: Path,
    mutation: str,
) -> None:
    cache, *_rest, rig = _build(tmp_path)
    payload = rig.to_dict()
    if mutation == "missing_group":
        del payload["control_bindings"]
    elif mutation == "extra_field":
        payload["late_exporter_patch"] = True
    elif mutation == "bone_parent":
        non_root = next(bone for bone in payload["bones"] if bone["parent_id"])
        non_root["parent_id"] = "bone/missing"
    elif mutation == "binding_target":
        payload["control_bindings"][0]["target_id"] = "mesh/missing"
    else:
        payload["export_symbols"]["symbols"][0]["export_name"] = "renamed"

    with pytest.raises(RigDocumentError) as exc_info:
        validate_rig_document_payload(payload)

    assert exc_info.value.code == "invalid_rig_document"
    with pytest.raises(RigDocumentError):
        validate_rig_document_payload(cache.semantic_payload())
