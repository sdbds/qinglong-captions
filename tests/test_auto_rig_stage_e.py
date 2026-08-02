from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from module.auto_rig.artifacts import canonical_json_sha256, sha256_file
from module.auto_rig.manifests import (
    manifest_relative_path,
    read_stage_manifest,
    scan_stage_public_output_paths,
)
from module.auto_rig.stage_e import (
    STAGE_E_ALGORITHM_VERSION,
    StageEError,
    execute_stage_e,
)
from tests.test_auto_rig_stage_c import _execute


@pytest.fixture(scope="module")
def stage_c_seed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = Path(tmp_path_factory.mktemp("stage-e-seed"))
    _execute(root)
    return root


def _copy_seed(seed: Path, root: Path) -> None:
    shutil.copytree(seed, root, dirs_exist_ok=True)


def _run_stage_e(
    root: Path,
    *,
    validation_tier: str = "structural",
    core_path: str | Path | None = None,
    renderer_path: str | Path | None = None,
):
    c_marker = root / Path(*manifest_relative_path("C").split("/"))
    return execute_stage_e(
        root,
        upstream_manifests={"C": sha256_file(c_marker)},
        relevant_config_fingerprint=canonical_json_sha256(
            {
                "profile": "dual_runtime_core_v1",
                "tier": validation_tier,
            }
        ),
        validation_tier=validation_tier,
        core_path=core_path,
        renderer_path=renderer_path,
    )


def test_stage_e_structural_tier_commits_exact_live2d_inventory(
    stage_c_seed: Path,
    tmp_path: Path,
) -> None:
    _copy_seed(stage_c_seed, tmp_path)
    spine_sentinel = tmp_path / "rig/spine/keep.bin"
    spine_sentinel.parent.mkdir(parents=True)
    spine_sentinel.write_bytes(b"spine")
    terminal_sentinel = tmp_path / "rig/error.json"
    terminal_sentinel.write_bytes(b"terminal")
    result = _run_stage_e(tmp_path)

    manifest = read_stage_manifest(tmp_path, "E")
    expected = {
        "rig/live2d/model.moc3",
        f"rig/live2d/{result.runtime_assets.model3.relative_path}",
        f"rig/live2d/{result.runtime_assets.cdi3.relative_path}",
        *(f"rig/live2d/{path}" for path in result.runtime_assets.referenced_texture_paths),
        *(
            f"rig/live2d/{asset.relative_path}"
            for asset in result.animations.motion_assets
        ),
        *(
            f"rig/live2d/{asset.relative_path}"
            for asset in result.animations.expression_assets
        ),
        "rig/live2d/export_report.json",
    }
    assert result.manifest == manifest
    assert manifest.algorithm_version == STAGE_E_ALGORITHM_VERSION
    assert manifest.status == "stage_validated"
    assert set(scan_stage_public_output_paths(tmp_path, "E")) == expected
    assert {item.path for item in manifest.output_file_sha256} == expected
    assert result.report["validation_tier"] == "structural"
    assert result.report["formal_release_eligible"] is False
    assert result.report["release_validation"]["status"] == "not_run"
    assert result.report["rig_json_sha256"] == result.rig.document_sha256
    assert (tmp_path / "rig/live2d/textures/page_0.png").read_bytes() == (
        tmp_path / "rig/shared/textures/page_0.png"
    ).read_bytes()
    assert not (tmp_path / "rig/cache/E/failure.json").exists()
    assert not (tmp_path / "rig/export_manifest.json").exists()
    assert spine_sentinel.read_bytes() == b"spine"
    assert terminal_sentinel.read_bytes() == b"terminal"


def test_stage_e_replaces_outputs_and_removes_obsolete_owned_files(
    stage_c_seed: Path,
    tmp_path: Path,
) -> None:
    _copy_seed(stage_c_seed, tmp_path)
    first = _run_stage_e(tmp_path)
    stale = tmp_path / "rig/live2d/motions/stale.motion3.json"
    stale.write_bytes(b"stale")
    (tmp_path / "rig/live2d/model.moc3").write_bytes(b"edited")

    second = _run_stage_e(tmp_path)

    assert not stale.exists()
    assert second.moc_bytes == first.moc_bytes
    assert (tmp_path / "rig/live2d/model.moc3").read_bytes() == first.moc_bytes
    assert read_stage_manifest(tmp_path, "E") == second.manifest


def test_stage_e_failure_invalidates_marker_and_writes_private_evidence(
    stage_c_seed: Path,
    tmp_path: Path,
) -> None:
    _copy_seed(stage_c_seed, tmp_path)
    _run_stage_e(tmp_path)
    shared_page = tmp_path / "rig/shared/textures/page_0.png"
    shared_page.write_bytes(shared_page.read_bytes() + b"tampered")

    with pytest.raises(StageEError):
        _run_stage_e(tmp_path)

    assert not (
        tmp_path / Path(*manifest_relative_path("E").split("/"))
    ).exists()
    assert (tmp_path / "rig/cache/E/failure.json").is_file()
    assert not (tmp_path / "rig/export_manifest.json").exists()


def test_stage_e_release_tier_requires_core_and_renderer_before_commit(
    stage_c_seed: Path,
    tmp_path: Path,
) -> None:
    _copy_seed(stage_c_seed, tmp_path)

    with pytest.raises(StageEError, match="live2d_release_gate_unavailable"):
        _run_stage_e(tmp_path, validation_tier="release")

    assert not (
        tmp_path / Path(*manifest_relative_path("E").split("/"))
    ).exists()
    assert (tmp_path / "rig/cache/E/failure.json").is_file()


@pytest.mark.optional_runtime
def test_stage_e_release_tier_commits_only_after_official_runtime_validation(
    stage_c_seed: Path,
    tmp_path: Path,
) -> None:
    core_path = os.environ.get("LIVE2D_CUBISM_CORE_PATH")
    renderer_path = os.environ.get("LIVE2D_E0_RENDERER_PATH")
    if not core_path or not renderer_path:
        pytest.skip("official Core and SDK renderer paths are required")
    _copy_seed(stage_c_seed, tmp_path)

    result = _run_stage_e(
        tmp_path,
        validation_tier="release",
        core_path=core_path,
        renderer_path=renderer_path,
    )

    assert result.release_validation is not None
    assert result.report["validation_tier"] == "release"
    assert result.report["formal_release_eligible"] is True
    assert result.report["release_validation"]["status"] == "passed"
    assert read_stage_manifest(tmp_path, "E") == result.manifest
