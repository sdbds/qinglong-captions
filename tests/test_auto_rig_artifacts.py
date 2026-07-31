import hashlib
import json
from pathlib import Path

import pytest

from module.auto_rig.artifacts import (
    ArtifactContractError,
    FileDigest,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    canonical_json_sha256,
    describe_file,
    normalize_relative_path,
    sha256_file,
)


def test_canonical_json_is_order_independent_and_ascii() -> None:
    first = {"z": [3, 2, 1], "label": "\u56fe\u5c42"}
    second = {"label": "\u56fe\u5c42", "z": [3, 2, 1]}

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_json_bytes(first) == b'{"label":"\\u56fe\\u5c42","z":[3,2,1]}'
    assert canonical_json_sha256(first) == canonical_json_sha256(second)


def test_canonical_json_digest_excludes_file_formatting_newline() -> None:
    payload = {"schema_version": 1, "status": "completed"}
    canonical = canonical_json_bytes(payload)
    formatted_digest = hashlib.sha256(canonical + b"\n").hexdigest()

    assert canonical_json_sha256(payload) == f"sha256:{hashlib.sha256(canonical).hexdigest()}"
    assert canonical_json_sha256(payload) != f"sha256:{formatted_digest}"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(ValueError, match="JSON compliant"):
        canonical_json_bytes({"value": value})


def test_describe_file_records_normalized_path_size_and_sha256(tmp_path: Path) -> None:
    root = tmp_path / "item"
    path = root / "rig" / "cache" / "A" / "geometry.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"geometry")

    described = describe_file(root, Path("rig/cache/A/geometry.json"))

    assert described == FileDigest(
        path="rig/cache/A/geometry.json",
        size=8,
        sha256=f"sha256:{hashlib.sha256(b'geometry').hexdigest()}",
    )
    assert sha256_file(path) == described.sha256


@pytest.mark.parametrize(
    "path",
    ["", ".", "../outside.json", "rig/../outside.json", "/absolute.json", "C:/absolute.json"],
)
def test_normalize_relative_path_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ArtifactContractError):
        normalize_relative_path(path)


def test_describe_file_rejects_missing_and_directory_paths(tmp_path: Path) -> None:
    root = tmp_path / "item"
    directory = root / "rig"
    directory.mkdir(parents=True)

    with pytest.raises(ArtifactContractError, match="regular file"):
        describe_file(root, "rig")
    with pytest.raises(ArtifactContractError, match="regular file"):
        describe_file(root, "missing.json")


def test_file_digest_rejects_malformed_values() -> None:
    with pytest.raises(ArtifactContractError, match="size"):
        FileDigest(path="rig/a.json", size=-1, sha256="sha256:" + "0" * 64)
    with pytest.raises(ArtifactContractError, match="SHA-256"):
        FileDigest(path="rig/a.json", size=1, sha256="0" * 64)


def test_atomic_writes_replace_target_without_part_residue(tmp_path: Path) -> None:
    binary = tmp_path / "nested" / "payload.bin"
    document = tmp_path / "nested" / "payload.json"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"old")

    atomic_write_bytes(binary, b"new")
    atomic_write_json(document, {"b": 2, "a": 1})

    assert binary.read_bytes() == b"new"
    assert document.read_text(encoding="utf-8") == '{"a":1,"b":2}\n'
    assert json.loads(document.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert not list(tmp_path.rglob("*.part*"))
