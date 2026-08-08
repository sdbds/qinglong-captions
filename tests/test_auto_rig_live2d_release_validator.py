from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from module.auto_rig.export.live2d.animations import (
    build_live2d_animation_plan,
    encode_live2d_animation_asset,
)
from module.auto_rig.export.live2d.artmesh import build_live2d_artmesh_plan
from module.auto_rig.export.live2d.attestation import (
    load_packaged_live2d_frame_attestation,
    select_runtime_attestation,
)
from module.auto_rig.export.live2d.binding_plan import build_live2d_binding_plan
from module.auto_rig.export.live2d.coordinates import build_live2d_coordinate_plan
from module.auto_rig.export.live2d.document import build_live2d_moc3_document
from module.auto_rig.export.live2d.keyforms import build_live2d_keyform_plan
from module.auto_rig.export.live2d.moc3_codec import encode_moc3_v400
from module.auto_rig.export.live2d.release_validator import (
    LIVE2D_RELEASE_VALIDATOR_VERSION,
    Live2DExpressionReleaseEvidence,
    Live2DReleaseGateError,
    Live2DReleaseValidationReport,
    _motion_sample_times,
    evaluate_facial_render_semantics,
    validate_live2d_release_bundle,
)
from module.auto_rig.export.live2d.rigid_drivers import build_rigid_driver_registry
from module.auto_rig.export.live2d.runtime_assets import (
    build_live2d_runtime_asset_plan,
    encode_live2d_runtime_asset,
)
from module.auto_rig.export.live2d.runtime_toolchain import (
    LIVE2D_VALIDATOR_PROTOCOL_DIGEST,
    Live2DRuntimeToolchain,
    ensure_live2d_runtime_toolchain,
)
from module.auto_rig.export.live2d.symbols import build_live2d_symbol_view
from module.auto_rig.export.live2d.validator import (
    build_live2d_structure_validation_report,
)
from tests.test_auto_rig_stage_c import _execute


def test_release_report_schema_excludes_machine_specific_binary_paths() -> None:
    fields = Live2DReleaseValidationReport.__dataclass_fields__

    assert "core_path" not in fields
    assert "renderer_path" not in fields
    assert "renderer_protocol_digest" in fields
    expression_fields = Live2DExpressionReleaseEvidence.__dataclass_fields__
    assert "blink_composition_tested" in expression_fields
    assert "blink_composition_parameter_maximum_residual" in expression_fields
    assert "blink_composition_visible" in expression_fields


def _rgba(
    width: int,
    height: int,
    changed: set[tuple[int, int]] | None = None,
) -> bytes:
    values = bytearray((20, 30, 40, 255) * (width * height))
    for x, y in changed or ():
        offset = (y * width + x) * 4
        values[offset : offset + 4] = bytes((220, 10, 40, 255))
    return bytes(values)


def test_facial_semantics_rejects_a_hash_change_confined_to_one_pixel() -> None:
    width = height = 64
    parts = ({"base_tag": "mouth", "side": None, "xyxy": [12, 14, 20, 18]},)

    metrics, passed = evaluate_facial_render_semantics(
        "talk",
        baseline_rgba=_rgba(width, height),
        effect_rgba=_rgba(width, height, {(28, 30)}),
        render_width=width,
        render_height=height,
        canvas_width=32,
        canvas_height=32,
        parts=parts,
    )

    assert passed is False
    assert metrics[0].changed_pixel_count == 1
    assert metrics[0].passed is False


def test_facial_semantics_accepts_a_mouth_change_with_area_and_height() -> None:
    width = height = 64
    parts = ({"base_tag": "mouth", "side": None, "xyxy": [12, 14, 20, 18]},)
    changed = {(x, y) for y in range(30, 34) for x in range(28, 35)}

    metrics, passed = evaluate_facial_render_semantics(
        "talk",
        baseline_rgba=_rgba(width, height),
        effect_rgba=_rgba(width, height, changed),
        render_width=width,
        render_height=height,
        canvas_width=32,
        canvas_height=32,
        parts=parts,
    )

    assert passed is True
    assert metrics[0].changed_fraction >= 0.20
    assert metrics[0].changed_bbox_height >= 4


def test_facial_semantics_accepts_emotion_when_one_symmetric_brow_is_occluded() -> None:
    width = height = 64
    parts = (
        {"base_tag": "mouth", "side": None, "xyxy": [12, 20, 20, 24]},
        {"base_tag": "eyebrow", "side": "xmin", "xyxy": [8, 8, 12, 10]},
        {"base_tag": "eyebrow", "side": "xmax", "xyxy": [20, 8, 24, 10]},
    )
    changed = {
        *((x, y) for y in range(36, 39) for x in range(28, 36)),
        *((x, y) for y in range(24, 26) for x in range(36, 40)),
    }

    metrics, passed = evaluate_facial_render_semantics(
        "happy",
        baseline_rgba=_rgba(width, height),
        effect_rgba=_rgba(width, height, changed),
        render_width=width,
        render_height=height,
        canvas_width=32,
        canvas_height=32,
        parts=parts,
    )

    by_roi = {metric.roi_id: metric for metric in metrics}
    assert passed is True
    assert by_roi["mouth"].passed is True
    assert by_roi["brow.xmin"].passed is False
    assert by_roi["brow.xmax"].passed is True


@pytest.mark.parametrize("preset_id", ("happy", "sad", "unimpressed"))
def test_mouth_form_semantics_accepts_one_substantial_raster_row(
    preset_id: str,
) -> None:
    width = height = 64
    parts = (
        {"base_tag": "mouth", "side": None, "xyxy": [12, 20, 20, 24]},
        {"base_tag": "eyebrow", "side": "xmin", "xyxy": [8, 8, 12, 10]},
        {"base_tag": "eyebrow", "side": "xmax", "xyxy": [20, 8, 24, 10]},
    )
    changed = {
        *((x, 36) for x in range(28, 36)),
        *((x, 24) for x in range(36, 40)),
    }

    metrics, passed = evaluate_facial_render_semantics(
        preset_id,
        baseline_rgba=_rgba(width, height),
        effect_rgba=_rgba(width, height, changed),
        render_width=width,
        render_height=height,
        canvas_width=32,
        canvas_height=32,
        parts=parts,
    )

    by_roi = {metric.roi_id: metric for metric in metrics}
    assert passed is True
    assert by_roi["mouth"].changed_bbox_height == 1
    assert by_roi["mouth"].changed_fraction >= 0.08
    assert by_roi["brow.xmax"].passed is True


def test_unimpressed_semantics_do_not_require_a_held_eye_crossfade() -> None:
    width = height = 64
    parts = (
        {"base_tag": "mouth", "side": None, "xyxy": [12, 20, 20, 24]},
        {"base_tag": "eyebrow", "side": "xmin", "xyxy": [8, 8, 12, 10]},
        {"base_tag": "eyebrow", "side": "xmax", "xyxy": [20, 8, 24, 10]},
        {"base_tag": "eyewhite", "side": "xmin", "xyxy": [8, 12, 12, 16]},
        {"base_tag": "eyewhite", "side": "xmax", "xyxy": [20, 12, 24, 16]},
    )
    changed = {
        *((x, y) for y in range(36, 39) for x in range(28, 36)),
        *((x, y) for y in range(24, 26) for x in range(36, 40)),
    }

    metrics, passed = evaluate_facial_render_semantics(
        "unimpressed",
        baseline_rgba=_rgba(width, height),
        effect_rgba=_rgba(width, height, changed),
        render_width=width,
        render_height=height,
        canvas_width=32,
        canvas_height=32,
        parts=parts,
    )

    by_roi = {metric.roi_id: metric for metric in metrics}
    assert passed is True
    assert by_roi["mouth"].passed is True
    assert by_roi["brow.xmax"].passed is True
    assert not any(roi_id.startswith("eye.") for roi_id in by_roi)


def test_facial_semantics_rejects_old_three_pixel_talk_height() -> None:
    width = height = 64
    parts = ({"base_tag": "mouth", "side": None, "xyxy": [12, 14, 20, 18]},)
    changed = {(x, y) for y in range(30, 33) for x in range(28, 35)}

    metrics, passed = evaluate_facial_render_semantics(
        "talk",
        baseline_rgba=_rgba(width, height),
        effect_rgba=_rgba(width, height, changed),
        render_width=width,
        render_height=height,
        canvas_width=32,
        canvas_height=32,
        parts=parts,
    )

    assert passed is False
    assert metrics[0].changed_fraction >= 0.20
    assert metrics[0].changed_bbox_height == 3


def test_render_semantics_measures_breath_inside_the_torso_roi() -> None:
    width = height = 64
    parts = ({"base_tag": "topwear", "side": None, "xyxy": [8, 8, 24, 28]},)
    changed = {(x, y) for y in range(22, 28) for x in range(22, 40)}

    metrics, passed = evaluate_facial_render_semantics(
        "breath",
        baseline_rgba=_rgba(width, height),
        effect_rgba=_rgba(width, height, changed),
        render_width=width,
        render_height=height,
        canvas_width=32,
        canvas_height=32,
        parts=parts,
    )

    assert passed is True
    assert len(metrics) == 1
    assert metrics[0].roi_id == "torso"
    assert metrics[0].changed_fraction >= 0.05
    assert metrics[0].changed_bbox_height >= 4


@pytest.fixture(scope="module")
def release_fixture(tmp_path_factory: pytest.TempPathFactory):
    root = Path(tmp_path_factory.mktemp("live2d-release"))
    *_inputs, stage_c = _execute(root)
    rig = stage_c.rig
    payload = rig.to_dict()
    symbols = build_live2d_symbol_view(payload["export_symbols"])
    registry = build_rigid_driver_registry(payload["control_specs"])
    bindings = build_live2d_binding_plan(rig, symbols, registry)
    coordinates = build_live2d_coordinate_plan(rig, bindings)
    artmeshes = build_live2d_artmesh_plan(rig, symbols, bindings, coordinates)
    keyforms = build_live2d_keyform_plan(rig, bindings, coordinates, artmeshes)
    document = build_live2d_moc3_document(rig, bindings, coordinates, artmeshes, keyforms)
    moc_payload = encode_moc3_v400(document)
    animations = build_live2d_animation_plan(rig, symbols, bindings, keyforms)
    runtime = build_live2d_runtime_asset_plan(rig, symbols, bindings, artmeshes, animations)
    structure = build_live2d_structure_validation_report(moc_payload, rig, bindings, coordinates, artmeshes, keyforms)

    bundle = root / "rig" / "live2d"
    (bundle / "textures").mkdir(parents=True)
    (bundle / "motions").mkdir()
    (bundle / "expressions").mkdir()
    (bundle / "model.moc3").write_bytes(moc_payload)
    (bundle / runtime.model3.relative_path).write_bytes(encode_live2d_runtime_asset(runtime.model3))
    (bundle / runtime.cdi3.relative_path).write_bytes(encode_live2d_runtime_asset(runtime.cdi3))
    for asset in (*animations.motion_assets, *animations.expression_assets):
        output = bundle / Path(*asset.relative_path.split("/"))
        output.write_bytes(encode_live2d_animation_asset(asset))
    for relative_path in runtime.referenced_texture_paths:
        source = root / "rig" / "shared" / "textures" / Path(relative_path).name
        destination = bundle / Path(*relative_path.split("/"))
        shutil.copyfile(source, destination)
    return (
        bundle,
        rig,
        bindings,
        coordinates,
        artmeshes,
        keyforms,
        animations,
        runtime,
        structure,
    )


def test_release_gate_requires_resolved_toolchain_and_attestation(release_fixture) -> None:
    (
        bundle,
        rig,
        bindings,
        coordinates,
        artmeshes,
        keyforms,
        animations,
        runtime,
        structure,
    ) = release_fixture
    with pytest.raises(Live2DReleaseGateError, match="Core|attest|release gate|renderer"):
        validate_live2d_release_bundle(
            bundle,
            runtime_toolchain=None,  # type: ignore[arg-type]
            runtime_attestation=None,  # type: ignore[arg-type]
            rig=rig,
            bindings=bindings,
            coordinates=coordinates,
            artmeshes=artmeshes,
            keyforms=keyforms,
            animations=animations,
            runtime_assets=runtime,
            structure_report=structure,
        )


def test_release_gate_rejects_validator_source_attestation_mismatch(release_fixture) -> None:
    (
        bundle,
        rig,
        bindings,
        coordinates,
        artmeshes,
        keyforms,
        animations,
        runtime,
        structure,
    ) = release_fixture
    payload = load_packaged_live2d_frame_attestation()
    record = payload["runtime_attestations"][0]
    runtime_attestation = select_runtime_attestation(
        payload,
        platform_id=record["platform_id"],
        backend_id=record["backend_id"],
        core_sha256=record["core_sha256"],
        validator_protocol_digest=record["validator_protocol_digest"],
        validator_source_sha256=record["validator_source_sha256"],
    )
    wrong_source = "sha256:" + ("0" * 64)
    assert wrong_source != runtime_attestation.validator_source_sha256
    runtime_toolchain = Live2DRuntimeToolchain(
        sdk_root=bundle,
        core_path=bundle / "model.moc3",
        validator_path=bundle / "model.moc3",
        platform_id=runtime_attestation.platform_id,
        backend_id=runtime_attestation.backend_id,
        cache_key="test-runtime-cache-key",
        core_sha256=runtime_attestation.core_sha256,
        validator_sha256="sha256:" + ("1" * 64),
        validator_source_sha256=wrong_source,
    )

    with pytest.raises(Live2DReleaseGateError, match="identities differ"):
        validate_live2d_release_bundle(
            bundle,
            runtime_toolchain=runtime_toolchain,
            runtime_attestation=runtime_attestation,
            rig=rig,
            bindings=bindings,
            coordinates=coordinates,
            artmeshes=artmeshes,
            keyforms=keyforms,
            animations=animations,
            runtime_assets=runtime,
            structure_report=structure,
        )


def test_blink_release_samples_rest_before_motion_completion(release_fixture) -> None:
    animations = release_fixture[6]
    blink = next(asset for asset in animations.motion_assets if asset.preset_id == "blink")
    defaults = {parameter_id: 1.0 for parameter_id in blink.parameter_ids}

    assert 23 / 30 in _motion_sample_times(blink, defaults=defaults)


@pytest.mark.optional_runtime
def test_official_sdk_validates_every_parameter_motion_and_expression(
    release_fixture,
) -> None:
    sdk_root = os.environ.get("CUBISM_SDK_ROOT") or os.environ.get("LIVE2D_SDK_ROOT")
    if not sdk_root:
        pytest.skip("CUBISM_SDK_ROOT is required")
    (
        bundle,
        rig,
        bindings,
        coordinates,
        artmeshes,
        keyforms,
        animations,
        runtime,
        structure,
    ) = release_fixture

    runtime_toolchain = ensure_live2d_runtime_toolchain(sdk_root=sdk_root)
    runtime_attestation = select_runtime_attestation(
        load_packaged_live2d_frame_attestation(),
        platform_id=runtime_toolchain.platform_id,
        backend_id=runtime_toolchain.backend_id,
        core_sha256=runtime_toolchain.core_sha256,
        validator_protocol_digest=LIVE2D_VALIDATOR_PROTOCOL_DIGEST,
        validator_source_sha256=runtime_toolchain.validator_source_sha256,
    )
    report = validate_live2d_release_bundle(
        bundle,
        runtime_toolchain=runtime_toolchain,
        runtime_attestation=runtime_attestation,
        rig=rig,
        bindings=bindings,
        coordinates=coordinates,
        artmeshes=artmeshes,
        keyforms=keyforms,
        animations=animations,
        runtime_assets=runtime,
        structure_report=structure,
    )

    assert report.validator_version == LIVE2D_RELEASE_VALIDATOR_VERSION
    assert report.core_consistency is True
    assert report.default_rest_maximum_residual <= 0.1
    assert len(report.parameter_evidence) == len(bindings.parameters)
    assert all(record.visible_change for record in report.parameter_evidence)
    assert {record.preset_id for record in report.motion_evidence} == {asset.preset_id for asset in animations.motion_assets}
    assert {record.preset_id for record in report.expression_evidence} == {
        asset.preset_id for asset in animations.expression_assets
    }
    assert all(record.nonzero_alpha for record in report.motion_evidence)
    assert all(record.visible_change for record in report.motion_evidence)
    assert all(record.restored_after_clear for record in report.expression_evidence)
    blink_compositions = [record for record in report.expression_evidence if record.blink_composition_tested]
    assert blink_compositions
    assert all(record.blink_composition_visible for record in blink_compositions)
    assert all(record.blink_composition_parameter_maximum_residual <= 1e-6 for record in blink_compositions)
