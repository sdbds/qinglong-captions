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


def test_canonical_json_and_describe_file(tmp_path: Path) -> None:
    first = {"z": [3, 2, 1], "label": "\u56fe\u5c42"}
    second = {"label": "\u56fe\u5c42", "z": [3, 2, 1]}
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_json_sha256(first) == canonical_json_sha256(second)

    path = tmp_path / "rig" / "cache" / "A" / "geometry.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"geometry")
    described = describe_file(tmp_path, Path("rig/cache/A/geometry.json"))
    assert described == FileDigest(
        path="rig/cache/A/geometry.json",
        size=8,
        sha256=f"sha256:{hashlib.sha256(b'geometry').hexdigest()}",
    )
    assert sha256_file(path) == described.sha256


@pytest.mark.parametrize("path", ["", "../outside.json", "/absolute.json"])
def test_normalize_relative_path_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ArtifactContractError):
        normalize_relative_path(path)


def test_atomic_writes_replace_target_without_part_residue(tmp_path: Path) -> None:
    binary = tmp_path / "nested" / "payload.bin"
    document = tmp_path / "nested" / "payload.json"
    atomic_write_bytes(binary, b"new")
    atomic_write_json(document, {"b": 2, "a": 1})

    assert binary.read_bytes() == b"new"
    assert json.loads(document.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert not list(tmp_path.rglob("*.part*"))
