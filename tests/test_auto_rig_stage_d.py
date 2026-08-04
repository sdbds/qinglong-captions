from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from module.auto_rig.artifacts import canonical_json_sha256, sha256_file
from module.auto_rig.manifests import (
    manifest_relative_path,
    read_stage_manifest,
    scan_stage_public_output_paths,
)
from module.auto_rig.stage_d import (
    STAGE_D_ALGORITHM_VERSION,
    StageDError,
    execute_stage_d,
)
from tests.test_auto_rig_stage_c import _execute


def _run_stage_d(root: Path):
    *_inputs, stage_c = _execute(root)
    c_marker = root / Path(*manifest_relative_path("C").split("/"))
    stage_d = execute_stage_d(
        root,
        upstream_manifests={"C": sha256_file(c_marker)},
        relevant_config_fingerprint=canonical_json_sha256({"profile": "dual_runtime_core_v1", "tier": "spine-4.2"}),
    )
    return stage_c, stage_d


def test_stage_d_commits_exact_spine_inventory_and_validated_report(
    tmp_path: Path,
) -> None:
    stage_c, result = _run_stage_d(tmp_path)

    manifest = read_stage_manifest(tmp_path, "D")
    assert result.manifest == manifest
    assert manifest.algorithm_version == STAGE_D_ALGORITHM_VERSION
    assert manifest.status == "stage_validated"
    expected = {
        "rig/spine/skeleton.json",
        "rig/spine/skeleton.atlas",
        "rig/spine/export_report.json",
        "rig/spine/textures/page_0.png",
    }
    assert set(scan_stage_public_output_paths(tmp_path, "D")) == expected
    assert {item.path for item in manifest.output_file_sha256} == expected
    assert result.validation.validated is True
    assert result.report["validation"]["report_sha256"] == result.validation.report_sha256
    assert result.report["rig_json_sha256"] == stage_c.rig.document_sha256
    assert (tmp_path / "rig" / "spine" / "skeleton.json").read_bytes() == result.skeleton_bytes
    assert (tmp_path / "rig" / "spine" / "skeleton.atlas").read_bytes() == result.atlas_bytes
    assert (tmp_path / "rig" / "spine" / "textures" / "page_0.png").read_bytes() == (
        tmp_path / "rig" / "shared" / "textures" / "page_0.png"
    ).read_bytes()
    assert not (tmp_path / "rig" / "cache" / "D" / "failure.json").exists()


def test_stage_d_replaces_edited_outputs_and_removes_obsolete_owned_files(
    tmp_path: Path,
) -> None:
    stage_c, first = _run_stage_d(tmp_path)
    stale = tmp_path / "rig" / "spine" / "textures" / "page_9.png"
    stale.write_bytes(b"stale")
    (tmp_path / "rig" / "spine" / "skeleton.json").write_bytes(b"{}")

    second = execute_stage_d(
        tmp_path,
        upstream_manifests={"C": sha256_file(tmp_path / Path(*manifest_relative_path("C").split("/")))},
        relevant_config_fingerprint=canonical_json_sha256({"profile": "dual_runtime_core_v1", "tier": "spine-4.2"}),
    )

    assert not stale.exists()
    assert second.skeleton_bytes == first.skeleton_bytes
    assert (tmp_path / "rig" / "spine" / "skeleton.json").read_bytes() == first.skeleton_bytes
    assert read_stage_manifest(tmp_path, "D") == second.manifest


def test_stage_d_failure_invalidates_marker_and_writes_private_evidence(
    tmp_path: Path,
) -> None:
    stage_c, _first = _run_stage_d(tmp_path)
    shared_page = tmp_path / "rig" / "shared" / "textures" / "page_0.png"
    shared_page.write_bytes(shared_page.read_bytes() + b"tampered")

    with pytest.raises(StageDError):
        execute_stage_d(
            tmp_path,
            upstream_manifests={"C": sha256_file(tmp_path / Path(*manifest_relative_path("C").split("/")))},
            relevant_config_fingerprint=canonical_json_sha256({"profile": "dual_runtime_core_v1", "tier": "spine-4.2"}),
        )

    assert not (tmp_path / Path(*manifest_relative_path("D").split("/"))).exists()
    assert (tmp_path / "rig" / "cache" / "D" / "failure.json").is_file()


def test_stage_d_records_opt_in_official_runtime_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    *_inputs, stage_c = _execute(tmp_path)
    runtime = tmp_path / "spine-runtime.exe"
    runtime.write_bytes(b"official-runtime-probe")
    evidence = {
        "validated": True,
        "validator_version": "spine-runtime-validator-v3",
        "runtime_version": "4.2",
        "animation_count": 1,
        "report_sha256": "sha256:" + ("1" * 64),
    }
    runtime_report = SimpleNamespace(to_dict=lambda: evidence)
    observed: dict[str, object] = {}

    def validate(executable, skeleton, atlas, *, expected_animation_names):
        observed.update(
            {
                "executable": Path(executable),
                "skeleton_name": Path(skeleton).name,
                "atlas_name": Path(atlas).name,
                "animations": expected_animation_names,
            }
        )
        return runtime_report

    monkeypatch.setattr(
        "module.auto_rig.stage_d.validate_spine_runtime_bundle",
        validate,
    )
    result = execute_stage_d(
        tmp_path,
        upstream_manifests={"C": sha256_file(tmp_path / Path(*manifest_relative_path("C").split("/")))},
        relevant_config_fingerprint=canonical_json_sha256({"profile": "dual_runtime_core_v1", "tier": "spine-4.2"}),
        spine_runtime_path=runtime,
    )

    assert observed["executable"] == runtime.resolve()
    assert observed["skeleton_name"] == "skeleton.json"
    assert observed["atlas_name"] == "skeleton.atlas"
    assert observed["animations"] == tuple(record.artifact_export_name for record in result.animation_plan.records)
    assert result.runtime_validation is runtime_report
    assert result.report["official_spine_runtime_gate"] == {
        "status": "passed",
        "validation": evidence,
    }
    assert str(runtime.resolve()) not in result.report_bytes.decode("ascii")
