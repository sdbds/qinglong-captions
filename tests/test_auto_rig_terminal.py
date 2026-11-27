import hashlib
import json
from pathlib import Path

import pytest

from module.auto_rig.artifacts import sha256_file
from module.auto_rig.manifests import (
    StageManifest,
    build_stage_manifest,
    manifest_relative_path,
    read_stage_manifest,
    write_stage_manifest,
)
from module.auto_rig.terminal import (
    ERROR_RECORD_PATH,
    EXPORT_MANIFEST_PATH,
    FormatValidation,
    StageFailureRecord,
    TerminalFinalizationError,
    TextureRuntimeContract,
    finalize_failure,
    finalize_success,
    invalidate_terminal,
    is_item_completed,
)

DEPENDENCIES = {"A": (), "B": ("A",), "C": ("A", "B"), "D": ("C",), "E": ("C",)}
C_OUTPUTS = ("rig/rig.json", "rig/report.json", "rig/motion_manifest.json", "rig/shared/textures/page_0.png")
D_OUTPUTS = ("rig/spine/skeleton.json", "rig/spine/skeleton.atlas", "rig/spine/textures/page_0.png", "rig/spine/export_report.json")
E_OUTPUTS = (
    "rig/live2d/model.moc3",
    "rig/live2d/model.model3.json",
    "rig/live2d/model.cdi3.json",
    "rig/live2d/textures/page_0.png",
    "rig/live2d/motions/idle.motion3.json",
    "rig/live2d/export_report.json",
)


def digest(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def item_path(root: Path, relative: str) -> Path:
    return root / Path(*relative.split("/"))


def stage_marker(root: Path, stage: str) -> Path:
    return item_path(root, manifest_relative_path(stage))


def write_files(root: Path, paths: tuple[str, ...], stage: str) -> None:
    for relative in paths:
        path = item_path(root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith(".json"):
            path.write_text(json.dumps({"stage": stage, "path": relative}), encoding="utf-8")
        else:
            path.write_bytes(f"{stage}:{relative}".encode("utf-8"))


def write_stage(root: Path, stage: str, outputs: tuple[str, ...], *, status: str = "stage_validated") -> StageManifest:
    write_files(root, outputs, stage)
    upstream = {name: sha256_file(stage_marker(root, name)) for name in DEPENDENCIES[stage]}
    manifest = build_stage_manifest(
        root,
        stage_name=stage,
        stage_schema_version=1,
        algorithm_version=f"stage-{stage.lower()}-v1",
        upstream_manifests=upstream,
        input_file_sha256=[],
        target_input_fingerprint=digest("input"),
        native_variant_set_sha256=digest("native-set"),
        native_variant_eligibility_sha256=digest("native-eligibility"),
        relevant_config_fingerprint=digest("config"),
        rig_overrides_sha256=digest("overrides"),
        output_paths=outputs,
        status=status,
    )
    write_stage_manifest(root, manifest)
    return manifest


def write_preterminal_graph(root: Path, *, degraded_stage: str | None = None) -> dict[str, StageManifest]:
    def stage_status(stage: str) -> str:
        return "stage_validated_with_degradation" if stage == degraded_stage else "stage_validated"

    manifests = {
        "A": write_stage(root, "A", ("rig/cache/A/geometry.json",), status=stage_status("A")),
        "B": write_stage(root, "B", ("rig/cache/B/rig_geometry.json",), status=stage_status("B")),
    }
    manifests["C"] = write_stage(root, "C", C_OUTPUTS, status=stage_status("C"))
    manifests["D"] = write_stage(root, "D", D_OUTPUTS, status=stage_status("D"))
    manifests["E"] = write_stage(root, "E", E_OUTPUTS, status=stage_status("E"))
    return manifests


def format_validations() -> tuple[FormatValidation, ...]:
    return (
        FormatValidation(
            format_id="spine_4_2",
            stage_name="D",
            status="validated",
            files=D_OUTPUTS,
            validator_fingerprint=digest("spine-validator"),
        ),
        FormatValidation(
            format_id="live2d_moc3_v4_00",
            stage_name="E",
            status="validated",
            files=E_OUTPUTS,
            validator_fingerprint=digest("live2d-validator"),
        ),
    )


def texture_runtime_contract() -> TextureRuntimeContract:
    return TextureRuntimeContract(
        schema_version="shared-texture-v1",
        canonical_uv_space="page_top_left_v_down",
        alpha_mode="straight",
        color_space="srgb_bytes",
        spine_uv_adapter="spine-4.2-uv-v1",
        spine_atlas_pma=False,
        live2d_uv_adapter="cubism-v4.00-uv-v1",
        live2d_runtime_loader_contract=digest("live2d-texture-loader"),
    )


def finalize_valid_item(root: Path, *, degraded_stage: str | None = None):
    manifests = write_preterminal_graph(root, degraded_stage=degraded_stage)
    expected = {stage: manifest.stage_fingerprint for stage, manifest in manifests.items()}
    result = finalize_success(
        root,
        input_fingerprint=digest("input"),
        native_variant_set_sha256=digest("native-set"),
        native_variant_eligibility_sha256=digest("native-eligibility"),
        config_fingerprint=digest("config"),
        profile="dual_runtime_core_v1",
        profile_fingerprint=digest("profile"),
        rig_overrides_sha256=digest("overrides"),
        expected_stage_fingerprints=expected,
        formats=format_validations(),
        validation_tier="release",
        motion_runtime_contract_sha256=digest("motion-runtime"),
        global_symbol_table_sha256=digest("symbols"),
        texture_contract=texture_runtime_contract(),
    )
    return manifests, result, expected


def test_success_finalization_and_revalidation(tmp_path: Path) -> None:
    stale_error = item_path(tmp_path, ERROR_RECORD_PATH)
    stale_error.parent.mkdir(parents=True, exist_ok=True)
    stale_error.write_text("stale", encoding="utf-8")

    _, result, expected = finalize_valid_item(tmp_path)
    export_path = item_path(tmp_path, EXPORT_MANIFEST_PATH)

    assert export_path.is_file()
    assert not stale_error.exists()
    assert result.g_manifest == read_stage_manifest(tmp_path, "G")
    assert result.g_manifest.status == "completed"
    assert is_item_completed(tmp_path, expected_stage_fingerprints=expected) is True

    item_path(tmp_path, "rig/spine/skeleton.json").write_text("changed", encoding="utf-8")
    assert is_item_completed(tmp_path, expected_stage_fingerprints=expected) is False


def test_success_propagates_degradation_and_refuses_missing_format(tmp_path: Path) -> None:
    degraded_root = tmp_path / "degraded"
    _, result, expected = finalize_valid_item(degraded_root, degraded_stage="B")
    assert result.g_manifest.status == "completed_with_degradation"
    assert is_item_completed(degraded_root, expected_stage_fingerprints=expected) is True

    missing_root = tmp_path / "missing"
    manifests = write_preterminal_graph(missing_root)
    with pytest.raises(TerminalFinalizationError, match="exactly"):
        finalize_success(
            missing_root,
            input_fingerprint=digest("input"),
            native_variant_set_sha256=digest("native-set"),
            native_variant_eligibility_sha256=digest("native-eligibility"),
            config_fingerprint=digest("config"),
            profile="dual_runtime_core_v1",
            profile_fingerprint=digest("profile"),
            rig_overrides_sha256=digest("overrides"),
            expected_stage_fingerprints={s: m.stage_fingerprint for s, m in manifests.items()},
            formats=format_validations()[:1],
            validation_tier="release",
            motion_runtime_contract_sha256=digest("motion-runtime"),
            global_symbol_table_sha256=digest("symbols"),
            texture_contract=texture_runtime_contract(),
        )


def test_failure_finalization_and_invalidation(tmp_path: Path) -> None:
    finalize_valid_item(tmp_path)
    write_files(tmp_path, ("rig/cache/E/failure.json",), "E")
    result = finalize_failure(
        tmp_path,
        item_id="alice.png",
        target_input_fingerprint=digest("input"),
        observed_input_set_sha256=digest("observed-inputs"),
        native_variant_set_sha256=digest("native-set"),
        native_variant_eligibility_sha256=digest("native-eligibility"),
        config_fingerprint=digest("config"),
        rig_overrides_sha256=digest("overrides"),
        profile="dual_runtime_core_v1",
        profile_fingerprint=digest("profile"),
        validation_tier="release",
        failure_records=(
            StageFailureRecord(
                stage_name="E",
                record_path="rig/cache/E/failure.json",
                diagnostics=({"code": "live2d_failed"},),
                retryable=True,
            ),
        ),
    )

    assert result.g_manifest.status == "failed"
    assert not item_path(tmp_path, EXPORT_MANIFEST_PATH).exists()
    assert item_path(tmp_path, ERROR_RECORD_PATH).is_file()

    invalidate_terminal(tmp_path)
    assert not stage_marker(tmp_path, "G").exists()
    assert not item_path(tmp_path, ERROR_RECORD_PATH).exists()
    assert item_path(tmp_path, "rig/rig.json").is_file()
