from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

import module.auto_rig.export.live2d as live2d
from module.auto_rig.export.live2d.cubism_core import (
    E0_CORE_EXPORTS,
    CubismCoreError,
    CubismCoreVersion,
    assess_core_capabilities,
    decode_core_version,
    exercise_moc_with_core,
    probe_cubism_core,
)


def test_live2d_package_exports_core_probe_api() -> None:
    assert live2d.CubismCoreError is not None
    assert live2d.probe_cubism_core is probe_cubism_core
    assert live2d.exercise_moc_with_core is exercise_moc_with_core


def test_decode_core_version_uses_two_byte_revision() -> None:
    version = decode_core_version(0x04020002)

    assert version == CubismCoreVersion(raw=0x04020002, major=4, minor=2, revision=2)
    assert version.label == "04.02.0002"


def test_420002_is_runtime_capable_but_cannot_be_a_consistency_gate() -> None:
    capabilities = assess_core_capabilities(0x04020002, E0_CORE_EXPORTS)

    assert capabilities.e0_runtime_api_available is True
    assert capabilities.consistency_api_available is False
    assert capabilities.consistency_floor_met is False
    assert capabilities.attestation_candidate is False
    assert capabilities.blockers == (
        "missing_export:csmHasMocConsistency",
        "core_version_below_consistency_floor:04.02.0004",
    )


def test_420004_with_consistency_export_is_only_an_attestation_candidate() -> None:
    capabilities = assess_core_capabilities(
        0x04020004,
        (*E0_CORE_EXPORTS, "csmHasMocConsistency"),
    )

    assert capabilities.e0_runtime_api_available is True
    assert capabilities.consistency_api_available is True
    assert capabilities.consistency_floor_met is True
    assert capabilities.attestation_candidate is True
    assert capabilities.blockers == ()


def test_missing_runtime_exports_are_reported_in_stable_order() -> None:
    capabilities = assess_core_capabilities(
        0x04020004,
        ("csmGetVersion", "csmHasMocConsistency"),
    )

    assert capabilities.e0_runtime_api_available is False
    assert capabilities.attestation_candidate is False
    assert capabilities.missing_e0_exports == tuple(sorted(set(E0_CORE_EXPORTS) - {"csmGetVersion"}))
    assert capabilities.blockers[0].startswith("missing_e0_exports:")


@pytest.mark.parametrize(
    "parameter_values",
    [
        {"": 1.0},
        {"ParamAngleX": True},
        {"ParamAngleX": float("nan")},
        {"ParamAngleX": float("inf")},
        [("ParamAngleX", 1.0)],
    ],
)
def test_parameter_requests_are_validated_before_native_loading(parameter_values: object) -> None:
    with pytest.raises(CubismCoreError, match="parameter"):
        exercise_moc_with_core(
            "missing-core.dll",
            "missing-model.moc3",
            parameter_values=parameter_values,  # type: ignore[arg-type]
        )


@pytest.mark.optional_runtime
def test_configured_cubism_core_probe() -> None:
    core_path = os.environ.get("LIVE2D_CUBISM_CORE_PATH")
    if not core_path:
        pytest.skip("LIVE2D_CUBISM_CORE_PATH is not configured")

    probe = probe_cubism_core(core_path)

    assert probe.path == str(Path(core_path).resolve())
    assert len(probe.sha256) == 64
    assert probe.file_size > 0
    assert probe.version.raw > 0
    assert probe.latest_moc_version > 0
    assert probe.capabilities.e0_runtime_api_available is True


@pytest.mark.optional_runtime
def test_configured_cubism_core_loads_and_updates_official_moc() -> None:
    core_path = os.environ.get("LIVE2D_CUBISM_CORE_PATH")
    moc_path = os.environ.get("LIVE2D_TEST_MOC_PATH")
    if not core_path or not moc_path:
        pytest.skip("LIVE2D_CUBISM_CORE_PATH and LIVE2D_TEST_MOC_PATH are required")

    probe = probe_cubism_core(core_path)
    result = exercise_moc_with_core(core_path, moc_path)

    assert result.core_sha256 == probe.sha256
    assert result.detected_moc_version <= probe.latest_moc_version
    assert result.model_size > 0
    assert result.parameter_count > 0
    assert result.part_count > 0
    assert result.drawable_count > 0
    assert result.total_vertex_count > 0
    assert result.nonzero_drawable_count == result.drawable_count
    assert result.finite_vertices is True
    if probe.capabilities.consistency_api_available and probe.capabilities.consistency_floor_met:
        assert result.consistency is True
    else:
        assert result.consistency is None


@pytest.mark.optional_runtime
def test_configured_cubism_core_captures_and_mutates_model_state() -> None:
    core_path = os.environ.get("LIVE2D_CUBISM_CORE_PATH")
    moc_path = os.environ.get("LIVE2D_TEST_MOC_PATH")
    if not core_path or not moc_path:
        pytest.skip("LIVE2D_CUBISM_CORE_PATH and LIVE2D_TEST_MOC_PATH are required")

    rest = exercise_moc_with_core(core_path, moc_path, capture_model_state=True)
    moved = exercise_moc_with_core(
        core_path,
        moc_path,
        parameter_values={"ParamAngleX": 30.0},
        capture_model_state=True,
    )

    assert rest.model_state is not None
    assert moved.model_state is not None
    rest_state = rest.model_state
    moved_state = moved.model_state
    rest_angle = next(parameter for parameter in rest_state.parameters if parameter.id == "ParamAngleX")
    moved_angle = next(parameter for parameter in moved_state.parameters if parameter.id == "ParamAngleX")
    assert rest_angle.value == pytest.approx(rest_angle.default_value)
    assert moved_angle.value == pytest.approx(30.0)
    assert tuple(drawable.id for drawable in rest_state.drawables) == tuple(
        drawable.id for drawable in moved_state.drawables
    )
    assert rest_state.vertex_uv_sha256 == moved_state.vertex_uv_sha256
    assert rest_state.vertex_position_sha256 != moved_state.vertex_position_sha256
    assert all(
        math.isfinite(coordinate)
        for drawable in moved_state.drawables
        for point in drawable.vertex_positions
        for coordinate in point
    )
    assert all(len(drawable.vertex_positions) == len(drawable.vertex_uvs) for drawable in moved_state.drawables)


@pytest.mark.optional_runtime
def test_configured_cubism_core_rejects_unknown_parameter() -> None:
    core_path = os.environ.get("LIVE2D_CUBISM_CORE_PATH")
    moc_path = os.environ.get("LIVE2D_TEST_MOC_PATH")
    if not core_path or not moc_path:
        pytest.skip("LIVE2D_CUBISM_CORE_PATH and LIVE2D_TEST_MOC_PATH are required")

    with pytest.raises(CubismCoreError, match="unknown parameter"):
        exercise_moc_with_core(
            core_path,
            moc_path,
            parameter_values={"ParamDefinitelyMissing": 1.0},
        )
