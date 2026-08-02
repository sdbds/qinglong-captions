from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from module.auto_rig.artifacts import canonical_json_sha256
from module.auto_rig.manifests import (
    manifest_relative_path,
    read_stage_manifest,
    scan_stage_public_output_paths,
)
from module.auto_rig.projections import (
    validate_motion_manifest_projection,
    validate_rig_report_projection,
)
from module.auto_rig.rig_document import load_rig_document
from module.auto_rig.stage_c import (
    STAGE_C_ALGORITHM_VERSION,
    StageCError,
    execute_stage_c,
)
from module.auto_rig.texture_plan import (
    build_texture_page_plan,
    texture_region_input,
)
from module.auto_rig.texture_sources import (
    TEXTURE_PIXEL_CONTRACT_VERSION,
    LoadedTextureRegion,
)
from tests.test_auto_rig_format_plans import _fixture


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _regions(cache):
    values = []
    for index, part in enumerate(cache.parts):
        width = part.xyxy[2] - part.xyxy[0]
        height = part.xyxy[3] - part.xyxy[1]
        rgba = bytes((index % 251, 64, 128, 255)) * (width * height)
        values.append(
            LoadedTextureRegion(
                part_id=part.part_id,
                source_kind=part.source_kind,
                variant_id=part.variant_id,
                xyxy=part.xyxy,
                width=width,
                height=height,
                rgba_u8=rgba,
                rgba_sha256=_sha(rgba),
                source_file_sha256=_sha(part.part_id.encode("ascii")),
                alpha_mode="straight",
                color_space="srgb_bytes",
                pixel_contract_version=TEXTURE_PIXEL_CONTRACT_VERSION,
            )
        )
    return tuple(values)


def _execute(tmp_path: Path):
    cache, controls, presets, capabilities, bindings, _candidates, _symbols = _fixture(
        tmp_path
    )
    regions = _regions(cache)
    expected_plan = build_texture_page_plan(
        texture_region_input(region) for region in regions
    )
    result = execute_stage_c(
        tmp_path,
        cache=cache,
        controls=controls,
        presets=presets,
        capabilities=capabilities,
        bindings=bindings,
        loaded_regions=regions,
        expected_texture_plan=expected_plan,
        profile_id="dual_runtime_core_v1",
        upstream_manifests={"B": canonical_json_sha256({"manifest": "B"})},
        relevant_config_fingerprint=canonical_json_sha256({"config": "C"}),
    )
    return cache, controls, presets, capabilities, bindings, regions, expected_plan, result


def test_stage_c_commits_complete_public_inventory_and_reloads_projections(
    tmp_path: Path,
) -> None:
    *_inputs, result = _execute(tmp_path)

    manifest = read_stage_manifest(tmp_path, "C")
    assert result.manifest == manifest
    assert manifest.algorithm_version == STAGE_C_ALGORITHM_VERSION
    assert manifest.status == "stage_validated"
    assert set(scan_stage_public_output_paths(tmp_path, "C")) == {
        digest.path for digest in manifest.output_file_sha256
    }
    assert {
        "rig/rig.json",
        "rig/report.json",
        "rig/motion_manifest.json",
        "rig/shared/textures/page_0.png",
    } == {digest.path for digest in manifest.output_file_sha256}

    rig = load_rig_document(tmp_path / "rig" / "rig.json")
    assert rig == result.rig
    assert validate_motion_manifest_projection(result.motion_manifest, rig) is (
        result.motion_manifest
    )
    assert validate_rig_report_projection(result.report, rig) is result.report
    assert not (tmp_path / "rig" / "cache" / "C" / "failure.json").exists()


def test_stage_c_removes_stale_pages_and_replaces_hand_edited_projection(
    tmp_path: Path,
) -> None:
    *inputs, first = _execute(tmp_path)
    stale = tmp_path / "rig" / "shared" / "textures" / "page_9.png"
    stale.write_bytes(b"stale")
    (tmp_path / "rig" / "report.json").write_text('{"edited":true}', encoding="utf-8")

    cache, controls, presets, capabilities, bindings, regions, expected, _result = (
        *inputs,
        first,
    )
    second = execute_stage_c(
        tmp_path,
        cache=cache,
        controls=controls,
        presets=presets,
        capabilities=capabilities,
        bindings=bindings,
        loaded_regions=regions,
        expected_texture_plan=expected,
        profile_id="dual_runtime_core_v1",
        upstream_manifests={"B": canonical_json_sha256({"manifest": "B"})},
        relevant_config_fingerprint=canonical_json_sha256({"config": "C"}),
    )

    assert not stale.exists()
    assert (tmp_path / "rig" / "report.json").read_bytes() == second.report_bytes


def test_stage_c_failure_invalidates_old_commit_marker_without_publishing_partial(
    tmp_path: Path,
) -> None:
    cache, controls, presets, capabilities, bindings, regions, expected, _result = (
        _execute(tmp_path)
    )
    mismatched = replace(
        expected,
        input_regions_sha256="sha256:" + "f" * 64,
        plan_sha256="sha256:" + "e" * 64,
    )

    with pytest.raises(StageCError) as exc_info:
        execute_stage_c(
            tmp_path,
            cache=cache,
            controls=controls,
            presets=presets,
            capabilities=capabilities,
            bindings=bindings,
            loaded_regions=regions,
            expected_texture_plan=mismatched,
            profile_id="dual_runtime_core_v1",
            upstream_manifests={"B": canonical_json_sha256({"manifest": "B"})},
            relevant_config_fingerprint=canonical_json_sha256({"config": "C"}),
        )

    assert exc_info.value.code == "texture_plan_mismatch"
    assert not (tmp_path / Path(*manifest_relative_path("C").split("/"))).exists()
    assert (tmp_path / "rig" / "cache" / "C" / "failure.json").is_file()
