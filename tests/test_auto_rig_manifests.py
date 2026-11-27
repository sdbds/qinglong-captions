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


def test_stage_manifest_round_trips(tmp_path: Path) -> None:
    manifest = build_a_manifest(tmp_path)
    marker = write_stage_manifest(tmp_path, manifest)
    loaded = read_stage_manifest(tmp_path, "A")
    payload = json.loads(marker.read_text(encoding="utf-8"))

    assert loaded == manifest
    assert payload["schema_version"] == STAGE_MANIFEST_SCHEMA_VERSION == 3
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


def test_manifest_commit_removes_obsolete_files_from_owner_public_namespace(tmp_path: Path) -> None:
    current = write_output(tmp_path, "rig/spine/skeleton.json", b"current")
    stale = tmp_path / "rig" / "spine" / "textures" / "page_3.png"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"stale")
    manifest = build_stage_manifest(
        tmp_path,
        stage_name="D",
        stage_schema_version=1,
        algorithm_version="stage-d-v1",
        upstream_manifests={},
        input_file_sha256=[],
        **stage_identity("D"),
        relevant_config_fingerprint=digest("config"),
        rig_overrides_sha256=digest("overrides"),
        output_paths=[current.path],
    )

    write_stage_manifest(tmp_path, manifest)

    assert current == describe_file(tmp_path, current.path)
    assert not stale.exists()


def test_manifest_rejects_invalid_stage_or_changed_output(tmp_path: Path) -> None:
    with pytest.raises(StageManifestError, match="stage"):
        manifest_relative_path("H")

    manifest = build_a_manifest(tmp_path)
    (tmp_path / "rig/cache/A/geometry_observations.json").write_bytes(b"changed")
    with pytest.raises(StageManifestError, match="changed before commit"):
        write_stage_manifest(tmp_path, manifest)

    (tmp_path / "rig/cache/A/geometry_observations.json").unlink()
    with pytest.raises(StageManifestError, match="missing before commit"):
        write_stage_manifest(tmp_path, manifest)
