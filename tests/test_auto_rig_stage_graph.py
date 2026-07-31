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
from module.auto_rig.stage_graph import (
    StageGraphContractError,
    StageGraphValidator,
    StageNode,
)


RELEASE_DEPENDENCIES = {
    "A": (),
    "B": ("A",),
    "C": ("A", "B"),
    "D": ("C",),
    "E": ("C",),
    "G": ("C", "D", "E"),
}


def digest(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def marker_path(root: Path, stage: str) -> Path:
    return root / Path(*manifest_relative_path(stage).split("/"))


def write_stage(
    root: Path,
    stage: str,
    *,
    upstream: tuple[str, ...] | None = None,
    output_path: str | None = None,
    status: str = "completed",
) -> StageManifest:
    dependencies = RELEASE_DEPENDENCIES[stage] if upstream is None else upstream
    output = output_path or f"rig/cache/{stage}/payload.bin"
    target = root / Path(*output.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(f"payload-{stage}".encode("ascii"))
    upstream_manifests = {dependency: sha256_file(marker_path(root, dependency)) for dependency in dependencies}
    manifest = build_stage_manifest(
        root,
        stage_name=stage,
        stage_schema_version=1,
        algorithm_version=f"stage-{stage.lower()}-v1",
        upstream_manifests=upstream_manifests,
        input_file_sha256=[],
        relevant_config_fingerprint=digest("config"),
        rig_overrides_sha256=digest("overrides"),
        output_paths=[output],
        status=status,
    )
    write_stage_manifest(root, manifest)
    return manifest


def write_release_graph(root: Path) -> dict[str, StageManifest]:
    manifests: dict[str, StageManifest] = {}
    for stage in ("A", "B", "C", "D", "E", "G"):
        manifests[stage] = write_stage(root, stage)
    return manifests


def expected_fingerprints(manifests: dict[str, StageManifest]) -> dict[str, str]:
    return {stage: manifest.stage_fingerprint for stage, manifest in manifests.items()}


def issue_codes(result) -> set[str]:
    return {issue.code for issue in result.issues}


def test_release_graph_is_reusable_when_every_contract_matches(tmp_path: Path) -> None:
    manifests = write_release_graph(tmp_path)

    result = StageGraphValidator(tmp_path).validate(
        target_stage="G",
        expected_fingerprints=expected_fingerprints(manifests),
    )

    assert result.reusable is True
    assert result.issues == ()
    assert [stage for stage, _ in result.manifests] == ["A", "B", "C", "D", "E", "G"]


def test_release_graph_reports_missing_marker(tmp_path: Path) -> None:
    manifests = write_release_graph(tmp_path)
    marker_path(tmp_path, "B").unlink()

    result = StageGraphValidator(tmp_path).validate(
        target_stage="G",
        expected_fingerprints=expected_fingerprints(manifests),
    )

    assert result.reusable is False
    assert "manifest_missing" in issue_codes(result)


def test_release_graph_reports_changed_output_bytes(tmp_path: Path) -> None:
    manifests = write_release_graph(tmp_path)
    output = tmp_path / read_stage_manifest(tmp_path, "D").output_file_sha256[0].path
    output.write_bytes(b"tampered")

    result = StageGraphValidator(tmp_path).validate(
        target_stage="G",
        expected_fingerprints=expected_fingerprints(manifests),
    )

    assert "output_digest_mismatch" in issue_codes(result)


def test_release_graph_reports_expected_fingerprint_mismatch(tmp_path: Path) -> None:
    manifests = write_release_graph(tmp_path)
    expected = expected_fingerprints(manifests)
    expected["C"] = digest("new-compiler-version")

    result = StageGraphValidator(tmp_path).validate(target_stage="G", expected_fingerprints=expected)

    assert "stage_fingerprint_mismatch" in issue_codes(result)


def test_release_graph_reports_changed_upstream_marker_digest(tmp_path: Path) -> None:
    manifests = write_release_graph(tmp_path)
    a_marker = marker_path(tmp_path, "A")
    a_marker.write_text(a_marker.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    result = StageGraphValidator(tmp_path).validate(
        target_stage="G",
        expected_fingerprints=expected_fingerprints(manifests),
    )

    mismatches = [issue for issue in result.issues if issue.code == "upstream_manifest_mismatch"]
    assert any(issue.stage_name == "B" and issue.path == manifest_relative_path("A") for issue in mismatches)


def test_release_graph_reports_undeclared_required_upstream(tmp_path: Path) -> None:
    manifests = write_release_graph(tmp_path)
    manifests["B"] = write_stage(tmp_path, "B", upstream=())
    manifests["C"] = write_stage(tmp_path, "C")
    manifests["D"] = write_stage(tmp_path, "D")
    manifests["E"] = write_stage(tmp_path, "E")
    manifests["G"] = write_stage(tmp_path, "G")

    result = StageGraphValidator(tmp_path).validate(
        target_stage="G",
        expected_fingerprints=expected_fingerprints(manifests),
    )

    assert any(issue.code == "upstream_set_mismatch" and issue.stage_name == "B" for issue in result.issues)


def test_release_graph_reports_output_ownership_overlap(tmp_path: Path) -> None:
    manifests = write_release_graph(tmp_path)
    shared = "rig/shared/conflict.bin"
    manifests["D"] = write_stage(tmp_path, "D", output_path=shared)
    manifests["E"] = write_stage(tmp_path, "E", output_path=shared)
    manifests["G"] = write_stage(tmp_path, "G")

    result = StageGraphValidator(tmp_path).validate(
        target_stage="G",
        expected_fingerprints=expected_fingerprints(manifests),
    )

    conflicts = [issue for issue in result.issues if issue.code == "output_ownership_conflict"]
    assert len(conflicts) == 1
    assert conflicts[0].path == shared
    assert "D" in conflicts[0].detail and "E" in conflicts[0].detail


def test_release_graph_rejects_output_that_owns_another_stage_marker(tmp_path: Path) -> None:
    manifests: dict[str, StageManifest] = {}
    for stage in ("A", "B", "C"):
        manifests[stage] = write_stage(tmp_path, stage)
    manifests["D"] = write_stage(tmp_path, "D", output_path=manifest_relative_path("E"))
    manifests["E"] = write_stage(tmp_path, "E")
    manifests["G"] = write_stage(tmp_path, "G")

    result = StageGraphValidator(tmp_path).validate(
        target_stage="G",
        expected_fingerprints=expected_fingerprints(manifests),
    )

    assert "stage_marker_owned_as_output" in issue_codes(result)


def test_completed_g_requires_c_d_and_e_upstreams(tmp_path: Path) -> None:
    manifests: dict[str, StageManifest] = {}
    for stage in ("A", "B", "C", "D", "E"):
        manifests[stage] = write_stage(tmp_path, stage)
    manifests["G"] = write_stage(tmp_path, "G", upstream=("C",))

    result = StageGraphValidator(tmp_path).validate(
        target_stage="G",
        expected_fingerprints=expected_fingerprints(manifests),
    )

    assert "terminal_dependencies_missing" in issue_codes(result)


def test_failed_stage_is_not_reusable(tmp_path: Path) -> None:
    manifests = write_release_graph(tmp_path)
    manifests["G"] = write_stage(tmp_path, "G", status="failed")

    result = StageGraphValidator(tmp_path).validate(
        target_stage="G",
        expected_fingerprints=expected_fingerprints(manifests),
    )

    assert "stage_status_not_completed" in issue_codes(result)


def test_invalid_manifest_is_reported_without_raising(tmp_path: Path) -> None:
    manifests = write_release_graph(tmp_path)
    g_marker = marker_path(tmp_path, "G")
    payload = json.loads(g_marker.read_text(encoding="utf-8"))
    payload["unknown"] = True
    g_marker.write_text(json.dumps(payload), encoding="utf-8")

    result = StageGraphValidator(tmp_path).validate(
        target_stage="G",
        expected_fingerprints=expected_fingerprints(manifests),
    )

    assert "manifest_invalid" in issue_codes(result)


def test_graph_constructor_rejects_dependency_cycles() -> None:
    with pytest.raises(StageGraphContractError, match="cycle"):
        StageGraphValidator(
            Path("item"),
            nodes=(StageNode("A", ("B",)), StageNode("B", ("A",))),
        )
