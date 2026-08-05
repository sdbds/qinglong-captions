from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from module.auto_rig.artifacts import (
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)
from module.auto_rig.export.live2d.attestation import (
    Live2DRuntimeAttestation,
    load_packaged_live2d_frame_attestation,
    select_runtime_attestation,
)
from module.auto_rig.export.live2d.cubism_renderer import (
    LIVE2D_E0_VALIDATOR_PROTOCOL_DIGEST,
)
from module.auto_rig.export.live2d.release_validator import (
    LIVE2D_RELEASE_REPORT_VERSION,
    LIVE2D_RELEASE_VALIDATOR_VERSION,
)
from module.auto_rig.export.live2d.runtime_toolchain import (
    LIVE2D_VALIDATOR_PROTOCOL_DIGEST,
    Live2DRuntimeToolchain,
    ensure_live2d_runtime_toolchain,
)
from module.auto_rig.jcs import jcs_bytes, jcs_sha256
from module.auto_rig.manifests import (
    StageManifest,
    build_stage_manifest,
    manifest_relative_path,
    read_stage_manifest,
    write_stage_manifest,
)
from module.auto_rig.rig_document import load_rig_document
from module.auto_rig.stage_c import execute_stage_c
from module.auto_rig.stage_d import execute_stage_d
from module.auto_rig.stage_e import execute_stage_e
from module.auto_rig.stage_g import StageGError, execute_stage_g_success
from module.auto_rig.terminal import is_item_completed
from module.auto_rig.texture_plan import (
    build_texture_page_plan,
    texture_region_input,
)
from tests.test_auto_rig_format_plans import _fixture
from tests.test_auto_rig_stage_c import _regions


def _digest(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _marker(root: Path, stage: str) -> Path:
    return root / Path(*manifest_relative_path(stage).split("/"))


def _write_foundation_stage(
    root: Path,
    *,
    stage: str,
    upstream: dict[str, str],
    output_path: str,
    cache,
) -> StageManifest:
    atomic_write_json(
        root / Path(*output_path.split("/")),
        {"schema_version": 1, "stage": stage},
    )
    manifest = build_stage_manifest(
        root,
        stage_name=stage,
        stage_schema_version=1,
        algorithm_version=f"test-stage-{stage.lower()}-v1",
        upstream_manifests=upstream,
        input_file_sha256=(),
        target_input_fingerprint=cache.target_input_fingerprint,
        native_variant_set_sha256=cache.native_variant_set_sha256,
        native_variant_eligibility_sha256=(
            cache.native_variant_eligibility_sha256
        ),
        relevant_config_fingerprint=_digest(f"{stage}-config"),
        rig_overrides_sha256=cache.rig_overrides_sha256,
        output_paths=(output_path,),
        status="stage_validated",
    )
    write_stage_manifest(root, manifest)
    return manifest


def _build_preterminal_graph(
    root: Path,
    *,
    live2d_validation_tier: str,
    runtime_toolchain: Live2DRuntimeToolchain | None = None,
    runtime_attestation: Live2DRuntimeAttestation | None = None,
) -> dict[str, StageManifest]:
    cache, controls, presets, capabilities, bindings, _candidates, _symbols = (
        _fixture(root)
    )
    stage_a = _write_foundation_stage(
        root,
        stage="A",
        upstream={},
        output_path="rig/cache/A/geometry.json",
        cache=cache,
    )
    stage_b = _write_foundation_stage(
        root,
        stage="B",
        upstream={"A": sha256_file(_marker(root, "A"))},
        output_path="rig/cache/B/rig_geometry.json",
        cache=cache,
    )
    regions = _regions(cache)
    texture_plan = build_texture_page_plan(
        texture_region_input(region) for region in regions
    )
    stage_c = execute_stage_c(
        root,
        cache=cache,
        controls=controls,
        presets=presets,
        capabilities=capabilities,
        bindings=bindings,
        loaded_regions=regions,
        expected_texture_plan=texture_plan,
        profile_id="dual_runtime_core_v1",
        upstream_manifests={
            "A": sha256_file(_marker(root, "A")),
            "B": sha256_file(_marker(root, "B")),
        },
        relevant_config_fingerprint=_digest("C-config"),
    )
    stage_d = execute_stage_d(
        root,
        upstream_manifests={"C": sha256_file(_marker(root, "C"))},
        relevant_config_fingerprint=_digest("D-config"),
    )
    stage_e = execute_stage_e(
        root,
        upstream_manifests={"C": sha256_file(_marker(root, "C"))},
        relevant_config_fingerprint=_digest(
            f"E-config-{live2d_validation_tier}"
        ),
        validation_tier=live2d_validation_tier,
        runtime_toolchain=runtime_toolchain,
        runtime_attestation=runtime_attestation,
    )
    return {
        "A": stage_a,
        "B": stage_b,
        "C": stage_c.manifest,
        "D": stage_d.manifest,
        "E": stage_e.manifest,
    }


def _expected_fingerprints(
    manifests: dict[str, StageManifest],
) -> dict[str, str]:
    return {
        stage: manifest.stage_fingerprint
        for stage, manifest in manifests.items()
    }


@pytest.fixture(scope="module")
def structural_graph_seed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = Path(tmp_path_factory.mktemp("stage-g-structural-seed"))
    _build_preterminal_graph(root, live2d_validation_tier="structural")
    return root


def _copy_graph(seed: Path, root: Path) -> dict[str, StageManifest]:
    shutil.copytree(seed, root, dirs_exist_ok=True)
    return {stage: read_stage_manifest(root, stage) for stage in "ABCDE"}


def _rewrite_report(
    root: Path,
    *,
    stage: str,
    relative_path: str,
    mutate,
) -> StageManifest:
    path = root / Path(*relative_path.split("/"))
    report = json.loads(path.read_text(encoding="utf-8"))
    mutate(report)
    path.write_bytes(jcs_bytes(report))
    old = read_stage_manifest(root, stage)
    manifest = build_stage_manifest(
        root,
        stage_name=old.stage_name,
        stage_schema_version=old.stage_schema_version,
        algorithm_version=old.algorithm_version,
        upstream_manifests=dict(old.upstream_manifests),
        input_file_sha256=old.input_file_sha256,
        target_input_fingerprint=old.target_input_fingerprint,
        native_variant_set_sha256=old.native_variant_set_sha256,
        native_variant_eligibility_sha256=(
            old.native_variant_eligibility_sha256
        ),
        relevant_config_fingerprint=old.relevant_config_fingerprint,
        rig_overrides_sha256=old.rig_overrides_sha256,
        output_paths=tuple(item.path for item in old.output_file_sha256),
        status=old.status,
    )
    write_stage_manifest(root, manifest)
    return manifest


def _promote_live2d_report_for_unit_test(
    root: Path,
    manifests: dict[str, StageManifest],
) -> None:
    def promote(report: dict[str, object]) -> None:
        structure = report["structure_validation"]
        assert isinstance(structure, dict)
        evidence = {
            "schema_version": LIVE2D_RELEASE_REPORT_VERSION,
            "validator_version": LIVE2D_RELEASE_VALIDATOR_VERSION,
            "structure_report_sha256": structure["report_sha256"],
            "core_sha256": _digest("test-core"),
            "core_version": "06.00.0001",
            "renderer_sha256": _digest("test-renderer"),
            "renderer_protocol_digest": LIVE2D_E0_VALIDATOR_PROTOCOL_DIGEST,
            "moc_sha256": structure["moc_sha256"],
            "core_consistency": True,
            "default_rest_maximum_residual": 0,
            "baseline_rgba_sha256": _digest("baseline-rgba"),
            "parameter_evidence": [],
            "motion_evidence": [],
            "expression_evidence": [],
        }
        evidence["report_sha256"] = jcs_sha256(evidence)
        report["validation_tier"] = "release"
        report["formal_release_eligible"] = True
        report["release_validation"] = {
            "status": "passed",
            "report": evidence,
        }
        report["report_sha256"] = jcs_sha256(
            {key: value for key, value in report.items() if key != "report_sha256"}
        )

    manifests["E"] = _rewrite_report(
        root,
        stage="E",
        relative_path="rig/live2d/export_report.json",
        mutate=promote,
    )


def test_stage_g_rejects_structural_live2d_as_formal_completion(
    structural_graph_seed: Path,
    tmp_path: Path,
) -> None:
    manifests = _copy_graph(structural_graph_seed, tmp_path)

    with pytest.raises(StageGError) as exc_info:
        execute_stage_g_success(
            tmp_path,
            config_fingerprint=_digest("G-config"),
            expected_stage_fingerprints=_expected_fingerprints(manifests),
        )

    assert exc_info.value.code == "live2d_release_validation_missing"
    assert not (tmp_path / "rig/export_manifest.json").exists()
    assert not _marker(tmp_path, "G").exists()


def test_stage_g_rejects_forged_live2d_release_evidence(
    structural_graph_seed: Path,
    tmp_path: Path,
) -> None:
    manifests = _copy_graph(structural_graph_seed, tmp_path)

    def forge_release(report: dict[str, object]) -> None:
        report["validation_tier"] = "release"
        report["formal_release_eligible"] = True
        report["release_validation"] = {"status": "passed", "report": {}}
        report["report_sha256"] = jcs_sha256(
            {key: value for key, value in report.items() if key != "report_sha256"}
        )

    manifests["E"] = _rewrite_report(
        tmp_path,
        stage="E",
        relative_path="rig/live2d/export_report.json",
        mutate=forge_release,
    )

    with pytest.raises(StageGError) as exc_info:
        execute_stage_g_success(
            tmp_path,
            config_fingerprint=_digest("G-config"),
            expected_stage_fingerprints=_expected_fingerprints(manifests),
        )

    assert exc_info.value.code == "live2d_release_validation_invalid"


def test_stage_g_derives_terminal_facts_from_current_c_d_e_graph(
    structural_graph_seed: Path,
    tmp_path: Path,
) -> None:
    manifests = _copy_graph(structural_graph_seed, tmp_path)
    _promote_live2d_report_for_unit_test(tmp_path, manifests)
    expected = _expected_fingerprints(manifests)
    rig = load_rig_document(tmp_path / "rig/rig.json")
    rig_payload = rig.to_dict()

    result = execute_stage_g_success(
        tmp_path,
        config_fingerprint=_digest("G-config"),
        expected_stage_fingerprints=expected,
    )

    assert result.payload["input_fingerprint"] == rig.input_fingerprint
    assert result.payload["profile"] == "dual_runtime_core_v1"
    assert result.payload["profile_fingerprint"] == (
        rig_payload["format_plans"]["profile"]["profile_sha256"]
    )
    assert result.payload["motion_runtime_contract_sha256"] == jcs_sha256(
        rig_payload["runtime_application"]
    )
    assert result.payload["global_symbol_table_sha256"] == (
        rig_payload["export_symbols"]["table_sha256"]
    )
    assert result.payload["texture_contract"][
        "live2d_runtime_loader_contract"
    ] == LIVE2D_E0_VALIDATOR_PROTOCOL_DIGEST
    assert {
        item["path"]
        for item in result.payload["formats"]["spine_4_2"]["files"]
    } == {
        item.path for item in manifests["D"].output_file_sha256
    }
    assert {
        item["path"]
        for item in result.payload["formats"]["live2d_moc3_v4_00"]["files"]
    } == {
        item.path for item in manifests["E"].output_file_sha256
    }
    assert is_item_completed(
        tmp_path,
        expected_stage_fingerprints=expected,
    ) is True


@pytest.mark.optional_runtime
def test_stage_g_commits_real_official_sdk_validated_dual_runtime_item(
    tmp_path: Path,
) -> None:
    sdk_root = os.environ.get("CUBISM_SDK_ROOT") or os.environ.get("LIVE2D_SDK_ROOT")
    if not sdk_root:
        pytest.skip("CUBISM_SDK_ROOT is required")
    runtime_toolchain = ensure_live2d_runtime_toolchain(sdk_root=sdk_root)
    runtime_attestation = select_runtime_attestation(
        load_packaged_live2d_frame_attestation(),
        platform_id=runtime_toolchain.platform_id,
        backend_id=runtime_toolchain.backend_id,
        core_sha256=runtime_toolchain.core_sha256,
        validator_protocol_digest=LIVE2D_VALIDATOR_PROTOCOL_DIGEST,
    )
    manifests = _build_preterminal_graph(
        tmp_path,
        live2d_validation_tier="release",
        runtime_toolchain=runtime_toolchain,
        runtime_attestation=runtime_attestation,
    )
    expected = _expected_fingerprints(manifests)

    result = execute_stage_g_success(
        tmp_path,
        config_fingerprint=_digest("G-release-config"),
        expected_stage_fingerprints=expected,
    )

    assert result.payload["status"] == "completed"
    assert (tmp_path / "rig/export_manifest.json").is_file()
    assert not (tmp_path / "rig/error.json").exists()
    assert _marker(tmp_path, "G").is_file()
    assert result.payload["required_formats"] == [
        "spine_4_2",
        "live2d_moc3_v4_00",
    ]
    assert result.payload["validation"]["tier"] == "release"
    assert is_item_completed(
        tmp_path,
        expected_stage_fingerprints=expected,
    ) is True
    assert (tmp_path / "rig/shared/textures/page_0.png").read_bytes() == (
        tmp_path / "rig/spine/textures/page_0.png"
    ).read_bytes()
    assert (tmp_path / "rig/shared/textures/page_0.png").read_bytes() == (
        tmp_path / "rig/live2d/textures/page_0.png"
    ).read_bytes()
    live2d_report_bytes = (
        tmp_path / "rig/live2d/export_report.json"
    ).read_bytes()
    assert b"core_path" not in live2d_report_bytes
    assert b"renderer_path" not in live2d_report_bytes
