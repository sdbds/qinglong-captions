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
from module.auto_rig.export.live2d.binding_plan import build_live2d_binding_plan
from module.auto_rig.export.live2d.coordinates import build_live2d_coordinate_plan
from module.auto_rig.export.live2d.document import build_live2d_moc3_document
from module.auto_rig.export.live2d.keyforms import build_live2d_keyform_plan
from module.auto_rig.export.live2d.moc3_codec import encode_moc3_v400
from module.auto_rig.export.live2d.release_validator import (
    LIVE2D_RELEASE_VALIDATOR_VERSION,
    Live2DReleaseGateError,
    Live2DReleaseValidationReport,
    validate_live2d_release_bundle,
)
from module.auto_rig.export.live2d.rigid_drivers import build_rigid_driver_registry
from module.auto_rig.export.live2d.runtime_assets import (
    build_live2d_runtime_asset_plan,
    encode_live2d_runtime_asset,
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
    artmeshes = build_live2d_artmesh_plan(
        rig, symbols, bindings, coordinates
    )
    keyforms = build_live2d_keyform_plan(
        rig, bindings, coordinates, artmeshes
    )
    document = build_live2d_moc3_document(
        rig, bindings, coordinates, artmeshes, keyforms
    )
    moc_payload = encode_moc3_v400(document)
    animations = build_live2d_animation_plan(
        rig, symbols, bindings, keyforms
    )
    runtime = build_live2d_runtime_asset_plan(
        rig, symbols, bindings, artmeshes, animations
    )
    structure = build_live2d_structure_validation_report(
        moc_payload, rig, bindings, coordinates, artmeshes, keyforms
    )

    bundle = root / "rig" / "live2d"
    (bundle / "textures").mkdir(parents=True)
    (bundle / "motions").mkdir()
    (bundle / "expressions").mkdir()
    (bundle / "model.moc3").write_bytes(moc_payload)
    (bundle / runtime.model3.relative_path).write_bytes(
        encode_live2d_runtime_asset(runtime.model3)
    )
    (bundle / runtime.cdi3.relative_path).write_bytes(
        encode_live2d_runtime_asset(runtime.cdi3)
    )
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


def test_release_gate_requires_configured_attested_core_and_renderer(
    release_fixture, tmp_path: Path
) -> None:
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
    fake_core = tmp_path / "Live2DCubismCore.dll"
    fake_core.write_bytes(b"not-a-core")
    with pytest.raises(Live2DReleaseGateError, match="Core|attest|release gate"):
        validate_live2d_release_bundle(
            bundle,
            core_path=fake_core,
            renderer_path=tmp_path / "missing-renderer.exe",
            rig=rig,
            bindings=bindings,
            coordinates=coordinates,
            artmeshes=artmeshes,
            keyforms=keyforms,
            animations=animations,
            runtime_assets=runtime,
            structure_report=structure,
        )


@pytest.mark.optional_runtime
def test_official_sdk_validates_every_parameter_motion_and_expression(
    release_fixture,
) -> None:
    core_path = os.environ.get("LIVE2D_CUBISM_CORE_PATH")
    renderer_path = os.environ.get("LIVE2D_E0_RENDERER_PATH")
    if not core_path or not renderer_path:
        pytest.skip("official Core and SDK renderer paths are required")
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

    report = validate_live2d_release_bundle(
        bundle,
        core_path=core_path,
        renderer_path=renderer_path,
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
    assert {record.preset_id for record in report.motion_evidence} == {
        asset.preset_id for asset in animations.motion_assets
    }
    assert {record.preset_id for record in report.expression_evidence} == {
        asset.preset_id for asset in animations.expression_assets
    }
    assert all(record.nonzero_alpha for record in report.motion_evidence)
    assert all(record.visible_change for record in report.motion_evidence)
    assert all(record.restored_after_clear for record in report.expression_evidence)
