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

DEPENDENCIES = {
    "A": (),
    "B": ("A",),
    "C": ("A", "B"),
    "D": ("C",),
    "E": ("C",),
}

C_OUTPUTS = (
    "rig/rig.json",
    "rig/report.json",
    "rig/motion_manifest.json",
    "rig/shared/textures/page_0.png",
)
D_OUTPUTS = (
    "rig/spine/skeleton.json",
    "rig/spine/skeleton.atlas",
    "rig/spine/textures/page_0.png",
    "rig/spine/export_report.json",
)
E_OUTPUTS = (
    "rig/live2d/avatar.moc3",
    "rig/live2d/avatar.model3.json",
    "rig/live2d/avatar.cdi3.json",
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


def write_stage(
    root: Path,
    stage: str,
    outputs: tuple[str, ...],
    *,
    status: str = "stage_validated",
) -> StageManifest:
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


def write_preterminal_graph(
    root: Path,
    *,
    degraded_stage: str | None = None,
) -> dict[str, StageManifest]:
    def stage_status(stage: str) -> str:
        return (
            "stage_validated_with_degradation"
            if stage == degraded_stage
            else "stage_validated"
        )

    manifests = {
        "A": write_stage(
            root,
            "A",
            ("rig/cache/A/geometry.json",),
            status=stage_status("A"),
        ),
        "B": write_stage(
            root,
            "B",
            ("rig/cache/B/rig_geometry.json",),
            status=stage_status("B"),
        ),
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


def success_kwargs(manifests: dict[str, StageManifest]) -> dict[str, object]:
    return {
        "input_fingerprint": digest("input"),
        "native_variant_set_sha256": digest("native-set"),
        "native_variant_eligibility_sha256": digest("native-eligibility"),
        "config_fingerprint": digest("config"),
        "profile": "dual_runtime_core_v1",
        "profile_fingerprint": digest("profile"),
        "rig_overrides_sha256": digest("overrides"),
        "expected_stage_fingerprints": {
            stage: manifest.stage_fingerprint for stage, manifest in manifests.items()
        },
        "formats": format_validations(),
        "validation_tier": "release",
        "motion_runtime_contract_sha256": digest("motion-runtime"),
        "global_symbol_table_sha256": digest("symbols"),
        "texture_contract": texture_runtime_contract(),
    }


def finalize_valid_item(root: Path, *, degraded_stage: str | None = None):
    manifests = write_preterminal_graph(root, degraded_stage=degraded_stage)
    result = finalize_success(
        root,
        **success_kwargs(manifests),
    )
    expected = {stage: manifest.stage_fingerprint for stage, manifest in manifests.items()}
    return manifests, result, expected


def test_success_finalization_publishes_dual_runtime_terminal_state(tmp_path: Path) -> None:
    stale_error = item_path(tmp_path, ERROR_RECORD_PATH)
    stale_error.parent.mkdir(parents=True, exist_ok=True)
    stale_error.write_text("stale", encoding="utf-8")

    manifests, result, expected = finalize_valid_item(tmp_path)
    export_path = item_path(tmp_path, EXPORT_MANIFEST_PATH)
    payload = json.loads(export_path.read_text(encoding="utf-8"))

    assert export_path.is_file()
    assert not stale_error.exists()
    assert stage_marker(tmp_path, "G").is_file()
    assert result.g_manifest == read_stage_manifest(tmp_path, "G")
    assert result.g_manifest.status == "completed"
    assert result.g_manifest.input_file_sha256 == ()
    assert [item.path for item in result.g_manifest.output_file_sha256] == [EXPORT_MANIFEST_PATH]
    assert payload["producer_stage"] == "G"
    assert payload["input_fingerprint"] == digest("input")
    assert payload["native_variant_set_sha256"] == digest("native-set")
    assert payload["native_variant_eligibility_sha256"] == digest("native-eligibility")
    assert payload["config_fingerprint"] == digest("config")
    assert payload["rig_overrides_sha256"] == digest("overrides")
    assert payload["required_formats"] == ["spine_4_2", "live2d_moc3_v4_00"]
    assert payload["formats"]["spine_4_2"]["status"] == "validated"
    assert payload["formats"]["live2d_moc3_v4_00"]["status"] == "validated"
    assert payload["upstream_stage_manifests"] == {stage: sha256_file(stage_marker(tmp_path, stage)) for stage in ("C", "D", "E")}
    assert payload["motion_manifest_sha256"] == sha256_file(item_path(tmp_path, "rig/motion_manifest.json"))
    assert payload["motion_runtime_contract_sha256"] == digest("motion-runtime")
    assert payload["texture_contract"] == texture_runtime_contract().to_dict()
    assert is_item_completed(tmp_path, expected_stage_fingerprints=expected) is True
    assert manifests["C"].output_file_sha256


def test_completion_revalidates_exporter_artifacts_not_just_marker_existence(tmp_path: Path) -> None:
    _, _, expected = finalize_valid_item(tmp_path)
    item_path(tmp_path, "rig/spine/skeleton.json").write_text("tampered", encoding="utf-8")

    assert item_path(tmp_path, EXPORT_MANIFEST_PATH).exists()
    assert is_item_completed(tmp_path, expected_stage_fingerprints=expected) is False


def test_success_propagates_degradation_to_terminal_status(tmp_path: Path) -> None:
    _, result, expected = finalize_valid_item(tmp_path, degraded_stage="B")

    assert result.g_manifest.status == "completed_with_degradation"
    assert result.payload["status"] == "completed_with_degradation"
    assert is_item_completed(tmp_path, expected_stage_fingerprints=expected) is True


def test_completion_rejects_undeclared_exporter_artifact(tmp_path: Path) -> None:
    _, _, expected = finalize_valid_item(tmp_path)
    stale = item_path(tmp_path, "rig/spine/textures/page_9.png")
    stale.write_bytes(b"stale")

    assert is_item_completed(tmp_path, expected_stage_fingerprints=expected) is False


def test_completion_fingerprint_contract_rejects_terminal_fingerprint(tmp_path: Path) -> None:
    _, result, expected = finalize_valid_item(tmp_path)
    expected_with_g = {**expected, "G": result.g_manifest.stage_fingerprint}

    with pytest.raises(TerminalFinalizationError, match="exactly A, B, C, D, and E"):
        is_item_completed(tmp_path, expected_stage_fingerprints=expected_with_g)


def test_completion_rejects_tampered_export_manifest(tmp_path: Path) -> None:
    _, _, expected = finalize_valid_item(tmp_path)
    export_path = item_path(tmp_path, EXPORT_MANIFEST_PATH)
    export_path.write_text(export_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    assert is_item_completed(tmp_path, expected_stage_fingerprints=expected) is False


def test_export_manifest_without_g_marker_is_not_completed(tmp_path: Path) -> None:
    _, _, expected = finalize_valid_item(tmp_path)
    stage_marker(tmp_path, "G").unlink()

    assert item_path(tmp_path, EXPORT_MANIFEST_PATH).exists()
    assert is_item_completed(tmp_path, expected_stage_fingerprints=expected) is False


def test_success_refuses_missing_live2d_artifact(tmp_path: Path) -> None:
    manifests = write_preterminal_graph(tmp_path)
    item_path(tmp_path, E_OUTPUTS[0]).unlink()

    with pytest.raises(TerminalFinalizationError, match="preterminal stage graph"):
        finalize_success(
            tmp_path,
            **success_kwargs(manifests),
        )
    assert not item_path(tmp_path, EXPORT_MANIFEST_PATH).exists()


def test_success_refuses_unvalidated_or_missing_required_format(tmp_path: Path) -> None:
    manifests = write_preterminal_graph(tmp_path)
    invalid = list(format_validations())
    invalid[1] = FormatValidation(
        format_id=invalid[1].format_id,
        stage_name=invalid[1].stage_name,
        status="stage_validated",
        files=invalid[1].files,
        validator_fingerprint=invalid[1].validator_fingerprint,
    )

    with pytest.raises(TerminalFinalizationError, match="validated"):
        finalize_success(
            tmp_path,
            **{**success_kwargs(manifests), "formats": invalid},
        )

    with pytest.raises(TerminalFinalizationError, match="exactly"):
        finalize_success(
            tmp_path,
            **{**success_kwargs(manifests), "formats": format_validations()[:1]},
        )


def test_failure_finalization_orders_records_and_publishes_error_only(tmp_path: Path) -> None:
    export_path = item_path(tmp_path, EXPORT_MANIFEST_PATH)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text("stale export", encoding="utf-8")
    d_record = item_path(tmp_path, "rig/cache/D/failure.json")
    e_record = item_path(tmp_path, "rig/cache/E/failure.json")
    write_files(tmp_path, ("rig/cache/D/failure.json",), "D")
    write_files(tmp_path, ("rig/cache/E/failure.json",), "E")

    result = finalize_failure(
        tmp_path,
        item_id="characters/alice.png",
        target_input_fingerprint=None,
        observed_input_set_sha256=digest("observed-inputs"),
        native_variant_set_sha256=None,
        native_variant_eligibility_sha256=None,
        config_fingerprint=digest("config"),
        rig_overrides_sha256=digest("overrides"),
        profile="dual_runtime_core_v1",
        profile_fingerprint=digest("profile"),
        validation_tier="release",
        failure_records=(
            StageFailureRecord(
                stage_name="E",
                record_path=e_record.relative_to(tmp_path).as_posix(),
                diagnostics=({"code": "live2d_failed", "severity": "error"},),
                retryable=False,
            ),
            StageFailureRecord(
                stage_name="D",
                record_path=d_record.relative_to(tmp_path).as_posix(),
                diagnostics=({"code": "spine_failed", "severity": "error"},),
                retryable=True,
            ),
        ),
    )
    error_path = item_path(tmp_path, ERROR_RECORD_PATH)
    payload = json.loads(error_path.read_text(encoding="utf-8"))

    assert error_path.is_file()
    assert not export_path.exists()
    assert payload["terminal_state"] == "failed"
    assert payload["target_input_fingerprint"] is None
    assert payload["observed_input_set_sha256"] == digest("observed-inputs")
    assert payload["native_variant_set_sha256"] is None
    assert payload["native_variant_eligibility_sha256"] is None
    assert payload["profile"] == "dual_runtime_core_v1"
    assert payload["validation_tier"] == "release"
    assert payload["failed_stages"] == ["D", "E"]
    assert [record["stage"] for record in payload["failure_records"]] == ["D", "E"]
    assert [record["code"] for record in payload["diagnostics"]] == ["spine_failed", "live2d_failed"]
    assert payload["retryable"] is False
    assert result.g_manifest.status == "failed"
    assert [item.path for item in result.g_manifest.output_file_sha256] == [ERROR_RECORD_PATH]
    assert [item.path for item in result.g_manifest.input_file_sha256] == [
        "rig/cache/D/failure.json",
        "rig/cache/E/failure.json",
    ]


def test_failure_upstream_uses_manifest_identity_without_duplicate_file_input(
    tmp_path: Path,
) -> None:
    a_manifest = write_stage(tmp_path, "A", ("rig/cache/A/geometry.json",))
    write_files(tmp_path, ("rig/cache/B/failure.json",), "B")
    failure = StageFailureRecord(
        stage_name="B",
        record_path="rig/cache/B/failure.json",
        diagnostics=({"code": "mesh_failed"},),
        retryable=True,
    )
    common = {
        "item_id": "alice.png",
        "observed_input_set_sha256": digest("observed-inputs"),
        "config_fingerprint": digest("config"),
        "rig_overrides_sha256": digest("overrides"),
        "profile": "dual_runtime_core_v1",
        "profile_fingerprint": digest("profile"),
        "validation_tier": "release",
        "failure_records": (failure,),
        "completed_stage_names": ("A",),
    }

    with pytest.raises(TerminalFinalizationError, match="semantic identities"):
        finalize_failure(
            tmp_path,
            target_input_fingerprint=None,
            native_variant_set_sha256=None,
            native_variant_eligibility_sha256=None,
            **common,
        )

    result = finalize_failure(
        tmp_path,
        target_input_fingerprint=digest("input"),
        native_variant_set_sha256=digest("native-set"),
        native_variant_eligibility_sha256=digest("native-eligibility"),
        **common,
    )

    assert dict(result.g_manifest.upstream_manifests) == {
        "A": sha256_file(stage_marker(tmp_path, "A"))
    }
    assert [item.path for item in result.g_manifest.input_file_sha256] == [
        "rig/cache/B/failure.json"
    ]
    assert a_manifest.target_input_fingerprint == digest("input")


def test_terminal_files_are_strict_xor_across_success_then_failure(tmp_path: Path) -> None:
    finalize_valid_item(tmp_path)
    failure_path = item_path(tmp_path, "rig/cache/E/failure.json")
    write_files(tmp_path, ("rig/cache/E/failure.json",), "E")

    finalize_failure(
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
                record_path=failure_path.relative_to(tmp_path).as_posix(),
                diagnostics=({"code": "live2d_failed"},),
                retryable=True,
            ),
        ),
    )

    assert not item_path(tmp_path, EXPORT_MANIFEST_PATH).exists()
    assert item_path(tmp_path, ERROR_RECORD_PATH).exists()


def test_invalidate_terminal_removes_only_g_owned_files(tmp_path: Path) -> None:
    finalize_valid_item(tmp_path)
    c_output = item_path(tmp_path, "rig/rig.json")

    invalidate_terminal(tmp_path)

    assert not stage_marker(tmp_path, "G").exists()
    assert not item_path(tmp_path, EXPORT_MANIFEST_PATH).exists()
    assert not item_path(tmp_path, ERROR_RECORD_PATH).exists()
    assert c_output.exists()
