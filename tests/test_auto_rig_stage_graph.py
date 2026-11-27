import hashlib
from pathlib import Path

from module.auto_rig.artifacts import sha256_file
from module.auto_rig.manifests import (
    StageManifest,
    build_stage_manifest,
    manifest_relative_path,
    read_stage_manifest,
    write_stage_manifest,
)
from module.auto_rig.stage_graph import StageGraphValidator

RELEASE_DEPENDENCIES = {"A": (), "B": ("A",), "C": ("A", "B"), "D": ("C",), "E": ("C",), "G": ("C", "D", "E")}


def digest(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def marker_path(root: Path, stage: str) -> Path:
    return root / Path(*manifest_relative_path(stage).split("/"))


def write_stage(root: Path, stage: str) -> StageManifest:
    output = f"rig/cache/{stage}/payload.bin"
    target = root / Path(*output.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(f"payload-{stage}".encode("ascii"))
    upstream = {dep: sha256_file(marker_path(root, dep)) for dep in RELEASE_DEPENDENCIES[stage]}
    manifest = build_stage_manifest(
        root,
        stage_name=stage,
        stage_schema_version=1,
        algorithm_version=f"stage-{stage.lower()}-v1",
        upstream_manifests=upstream,
        input_file_sha256=[],
        target_input_fingerprint=digest("target-input"),
        native_variant_set_sha256=digest("native-set"),
        native_variant_eligibility_sha256=digest("native-eligibility"),
        relevant_config_fingerprint=digest("config"),
        rig_overrides_sha256=digest("overrides"),
        output_paths=[output],
        status="completed" if stage == "G" else "stage_validated",
    )
    write_stage_manifest(root, manifest)
    return manifest


def write_release_graph(root: Path) -> dict[str, StageManifest]:
    return {stage: write_stage(root, stage) for stage in ("A", "B", "C", "D", "E", "G")}


def expected_fingerprints(manifests: dict[str, StageManifest]) -> dict[str, str]:
    return {stage: manifest.stage_fingerprint for stage, manifest in manifests.items()}


def test_release_graph_is_reusable_when_every_contract_matches(tmp_path: Path) -> None:
    manifests = write_release_graph(tmp_path)
    result = StageGraphValidator(tmp_path).validate(target_stage="G", expected_fingerprints=expected_fingerprints(manifests))

    assert result.reusable is True
    assert result.issues == ()
    assert [stage for stage, _ in result.manifests] == ["A", "B", "C", "D", "E", "G"]


def test_release_graph_reports_missing_marker_and_changed_output(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"
    manifests = write_release_graph(missing_root)
    marker_path(missing_root, "B").unlink()
    result = StageGraphValidator(missing_root).validate(target_stage="G", expected_fingerprints=expected_fingerprints(manifests))
    assert not result.reusable
    assert "manifest_missing" in {issue.code for issue in result.issues}

    changed_root = tmp_path / "changed"
    manifests = write_release_graph(changed_root)
    output = changed_root / read_stage_manifest(changed_root, "D").output_file_sha256[0].path
    output.write_bytes(b"changed")
    result = StageGraphValidator(changed_root).validate(target_stage="G", expected_fingerprints=expected_fingerprints(manifests))
    assert "output_digest_mismatch" in {issue.code for issue in result.issues}


def test_release_graph_rejects_undeclared_public_file(tmp_path: Path) -> None:
    manifests = write_release_graph(tmp_path)
    stale = tmp_path / "rig" / "spine" / "textures" / "page_9.png"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"stale")

    result = StageGraphValidator(tmp_path).validate(target_stage="G", expected_fingerprints=expected_fingerprints(manifests))
    extras = [issue for issue in result.issues if issue.code == "undeclared_public_output"]
    assert [(issue.stage_name, issue.path) for issue in extras] == [("D", "rig/spine/textures/page_9.png")]
