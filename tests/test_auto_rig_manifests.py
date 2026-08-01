import hashlib
import json
from pathlib import Path

import pytest

from module.auto_rig.artifacts import FileDigest, canonical_json_sha256, describe_file, sha256_file
from module.auto_rig.manifests import (
    STAGE_MANIFEST_SCHEMA_VERSION,
    StageManifest,
    StageManifestError,
    build_stage_fingerprint,
    build_stage_manifest,
    manifest_relative_path,
    read_stage_manifest,
    write_stage_manifest,
)


def digest(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def stage_identity(stage: str, *, status: str | None = None) -> dict[str, object]:
    return {
        "target_input_fingerprint": digest("target-input"),
        "native_variant_set_sha256": digest("native-set"),
        "native_variant_eligibility_sha256": digest("native-eligibility"),
        "status": status or ("completed" if stage == "G" else "stage_validated"),
    }


def write_output(root: Path, relative_path: str, payload: bytes = b"payload") -> FileDigest:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return describe_file(root, relative_path)


def build_a_manifest(root: Path) -> StageManifest:
    input_digest = write_output(root, "optimized/info.json", b'{"parts":{}}')
    write_output(root, "rig/cache/A/geometry_observations.json", b"geometry")
    return build_stage_manifest(
        root,
        stage_name="A",
        stage_schema_version=1,
        algorithm_version="geometry-observations-v1",
        upstream_manifests={},
        input_file_sha256=[input_digest],
        **stage_identity("A"),
        relevant_config_fingerprint=digest("config"),
        rig_overrides_sha256=digest("no-overrides"),
        output_paths=["rig/cache/A/geometry_observations.json"],
    )


def test_stage_manifest_round_trips_as_strict_canonical_payload(tmp_path: Path) -> None:
    manifest = build_a_manifest(tmp_path)

    marker = write_stage_manifest(tmp_path, manifest)
    loaded = read_stage_manifest(tmp_path, "A")
    payload = json.loads(marker.read_text(encoding="utf-8"))

    assert loaded == manifest
    assert payload == manifest.to_dict()
    assert payload["schema_version"] == STAGE_MANIFEST_SCHEMA_VERSION == 3
    assert payload["stage_name"] == "A"
    assert payload["status"] == "stage_validated"
    assert payload["target_input_fingerprint"] == digest("target-input")
    assert payload["native_variant_set_sha256"] == digest("native-set")
    assert payload["native_variant_eligibility_sha256"] == digest("native-eligibility")
    assert payload["output_file_sha256"] == [describe_file(tmp_path, "rig/cache/A/geometry_observations.json").to_dict()]
    assert payload["output_inventory_sha256"] == canonical_json_sha256(payload["output_file_sha256"])
    assert marker.relative_to(tmp_path).as_posix() == manifest_relative_path("A")
    assert sha256_file(marker).startswith("sha256:")


def test_stage_fingerprint_is_stable_across_input_order(tmp_path: Path) -> None:
    first = write_output(tmp_path, "inputs/a.json", b"a")
    second = write_output(tmp_path, "inputs/b.json", b"b")
    common = {
        "stage_name": "B",
        "stage_schema_version": 1,
        "algorithm_version": "rig-geometry-v1",
        "upstream_manifests": {"A": digest("manifest-a")},
        **stage_identity("B"),
        "relevant_config_fingerprint": digest("config"),
        "rig_overrides_sha256": digest("overrides"),
    }

    assert build_stage_fingerprint(input_file_sha256=[first, second], **common) == build_stage_fingerprint(
        input_file_sha256=[second, first], **common
    )


@pytest.mark.parametrize("stage", ["", "a", "H", "join", "AA"])
def test_manifest_rejects_unknown_stage_names(tmp_path: Path, stage: str) -> None:
    with pytest.raises(StageManifestError, match="stage"):
        manifest_relative_path(stage)


def test_manifest_parser_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = build_a_manifest(tmp_path).to_dict()
    payload["surprise"] = True

    with pytest.raises(StageManifestError, match="exactly"):
        StageManifest.from_dict(payload)


def test_manifest_parser_rejects_tampered_output_inventory_digest(tmp_path: Path) -> None:
    payload = build_a_manifest(tmp_path).to_dict()
    payload["output_inventory_sha256"] = digest("not-the-output-inventory")

    with pytest.raises(StageManifestError, match="output_inventory_sha256"):
        StageManifest.from_dict(payload)


def test_manifest_parser_rejects_unsorted_and_duplicate_file_records(tmp_path: Path) -> None:
    first = write_output(tmp_path, "inputs/a.json", b"a")
    second = write_output(tmp_path, "inputs/b.json", b"b")
    payload = build_a_manifest(tmp_path).to_dict()
    payload["input_file_sha256"] = [second.to_dict(), first.to_dict()]

    with pytest.raises(StageManifestError, match="sorted"):
        StageManifest.from_dict(payload)

    payload["input_file_sha256"] = [first.to_dict(), first.to_dict()]
    with pytest.raises(StageManifestError, match="duplicate"):
        StageManifest.from_dict(payload)


def test_manifest_rejects_its_own_commit_marker_as_output(tmp_path: Path) -> None:
    marker_path = manifest_relative_path("A")
    write_output(tmp_path, marker_path, b"not-a-marker-yet")

    with pytest.raises(StageManifestError, match="commit marker"):
        build_stage_manifest(
            tmp_path,
            stage_name="A",
            stage_schema_version=1,
            algorithm_version="geometry-observations-v1",
            upstream_manifests={},
            input_file_sha256=[],
            **stage_identity("A"),
            relevant_config_fingerprint=digest("config"),
            rig_overrides_sha256=digest("overrides"),
            output_paths=[marker_path],
        )


def test_manifest_rejects_input_output_overlap(tmp_path: Path) -> None:
    shared = write_output(tmp_path, "rig/cache/A/shared.json")

    with pytest.raises(StageManifestError, match="both input and output"):
        build_stage_manifest(
            tmp_path,
            stage_name="A",
            stage_schema_version=1,
            algorithm_version="geometry-observations-v1",
            upstream_manifests={},
            input_file_sha256=[shared],
            **stage_identity("A"),
            relevant_config_fingerprint=digest("config"),
            rig_overrides_sha256=digest("overrides"),
            output_paths=[shared.path],
        )


def test_manifest_build_requires_all_outputs_to_exist(tmp_path: Path) -> None:
    with pytest.raises(StageManifestError, match="regular file"):
        build_stage_manifest(
            tmp_path,
            stage_name="A",
            stage_schema_version=1,
            algorithm_version="geometry-observations-v1",
            upstream_manifests={},
            input_file_sha256=[],
            **stage_identity("A"),
            relevant_config_fingerprint=digest("config"),
            rig_overrides_sha256=digest("overrides"),
            output_paths=["rig/cache/A/missing.json"],
        )


def test_manifest_write_rejects_output_changed_after_build(tmp_path: Path) -> None:
    manifest = build_a_manifest(tmp_path)
    (tmp_path / "rig/cache/A/geometry_observations.json").write_bytes(b"changed")

    with pytest.raises(StageManifestError, match="changed before commit"):
        write_stage_manifest(tmp_path, manifest)
    assert not (tmp_path / manifest_relative_path("A")).exists()


@pytest.mark.parametrize(
    "stage,current_path,stale_path",
    [
        ("C", "rig/shared/textures/page_0.png", "rig/shared/textures/page_3.png"),
        ("D", "rig/spine/skeleton.json", "rig/spine/textures/page_3.png"),
        ("E", "rig/live2d/model.moc3", "rig/live2d/motions/stale.motion3.json"),
        ("G", "rig/export_manifest.json", "rig/error.json"),
    ],
)
def test_manifest_commit_removes_obsolete_files_from_owner_public_namespace(
    tmp_path: Path,
    stage: str,
    current_path: str,
    stale_path: str,
) -> None:
    current = write_output(tmp_path, current_path, b"current")
    stale = tmp_path / Path(*stale_path.split("/"))
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"stale")
    manifest = build_stage_manifest(
        tmp_path,
        stage_name=stage,
        stage_schema_version=1,
        algorithm_version=f"stage-{stage.lower()}-v1",
        upstream_manifests={},
        input_file_sha256=[],
        **stage_identity(stage),
        relevant_config_fingerprint=digest("config"),
        rig_overrides_sha256=digest("overrides"),
        output_paths=[current.path],
    )

    write_stage_manifest(tmp_path, manifest)

    assert current == describe_file(tmp_path, current.path)
    assert not stale.exists()


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("schema_version", 1, "schema_version"),
        ("stage_schema_version", 0, "stage_schema_version"),
        ("algorithm_version", "", "algorithm_version"),
        ("stage_fingerprint", "not-a-digest", "SHA-256"),
        ("target_input_fingerprint", "not-a-digest", "SHA-256"),
        ("native_variant_set_sha256", "not-a-digest", "SHA-256"),
        ("native_variant_eligibility_sha256", "not-a-digest", "SHA-256"),
        ("relevant_config_fingerprint", "not-a-digest", "SHA-256"),
        ("rig_overrides_sha256", "not-a-digest", "SHA-256"),
        ("status", "completed", "status"),
    ],
)
def test_manifest_parser_rejects_invalid_scalar_contracts(tmp_path: Path, field: str, value: object, match: str) -> None:
    payload = build_a_manifest(tmp_path).to_dict()
    payload[field] = value

    with pytest.raises(StageManifestError, match=match):
        StageManifest.from_dict(payload)


def test_manifest_rejects_unexpected_upstream_and_fingerprint_tampering(tmp_path: Path) -> None:
    payload = build_a_manifest(tmp_path).to_dict()
    payload["upstream_manifests"] = {"A": digest("self")}

    with pytest.raises(StageManifestError, match="itself"):
        StageManifest.from_dict(payload)

    payload = build_a_manifest(tmp_path).to_dict()
    payload["stage_fingerprint"] = digest("tampered")
    with pytest.raises(StageManifestError, match="fingerprint"):
        StageManifest.from_dict(payload)


@pytest.mark.parametrize(
    "stage,status",
    (
        ("A", "completed"),
        ("F", "failed"),
        ("G", "stage_validated"),
    ),
)
def test_manifest_rejects_status_from_another_stage_class(
    tmp_path: Path,
    stage: str,
    status: str,
) -> None:
    output = write_output(tmp_path, f"rig/cache/{stage}/payload.json")

    with pytest.raises(StageManifestError, match="status"):
        build_stage_manifest(
            tmp_path,
            stage_name=stage,
            stage_schema_version=1,
            algorithm_version="stage-v1",
            upstream_manifests={},
            input_file_sha256=[],
            **stage_identity(stage, status=status),
            relevant_config_fingerprint=digest("config"),
            rig_overrides_sha256=digest("overrides"),
            output_paths=[output.path],
        )


def test_g_failed_manifest_allows_unresolved_semantic_identities(tmp_path: Path) -> None:
    error = write_output(tmp_path, "rig/error.json", b"{}")

    manifest = build_stage_manifest(
        tmp_path,
        stage_name="G",
        stage_schema_version=1,
        algorithm_version="terminal-v1",
        upstream_manifests={},
        input_file_sha256=[],
        target_input_fingerprint=None,
        native_variant_set_sha256=None,
        native_variant_eligibility_sha256=None,
        relevant_config_fingerprint=digest("config"),
        rig_overrides_sha256=digest("overrides"),
        output_paths=[error.path],
        status="failed",
    )

    assert manifest.status == "failed"
    assert manifest.target_input_fingerprint is None


def test_stage_fingerprint_tracks_each_semantic_identity(tmp_path: Path) -> None:
    common = {
        "stage_name": "C",
        "stage_schema_version": 1,
        "algorithm_version": "rig-document-v1",
        "upstream_manifests": {"B": digest("manifest-b")},
        "input_file_sha256": [],
        **stage_identity("C"),
        "relevant_config_fingerprint": digest("config"),
        "rig_overrides_sha256": digest("overrides"),
    }
    baseline = build_stage_fingerprint(**common)

    for field in (
        "target_input_fingerprint",
        "native_variant_set_sha256",
        "native_variant_eligibility_sha256",
        "rig_overrides_sha256",
    ):
        mutated = {**common, field: digest(f"changed-{field}")}
        assert build_stage_fingerprint(**mutated) != baseline


def test_stage_fingerprint_does_not_depend_on_result_status() -> None:
    common = {
        "stage_name": "B",
        "stage_schema_version": 1,
        "algorithm_version": "rig-geometry-v1",
        "upstream_manifests": {"A": digest("manifest-a")},
        "input_file_sha256": [],
        "target_input_fingerprint": digest("target-input"),
        "native_variant_set_sha256": digest("native-set"),
        "native_variant_eligibility_sha256": digest("native-eligibility"),
        "relevant_config_fingerprint": digest("config"),
        "rig_overrides_sha256": digest("overrides"),
    }

    assert build_stage_fingerprint(
        **common,
        status="stage_validated",
    ) == build_stage_fingerprint(
        **common,
        status="stage_validated_with_degradation",
    )
