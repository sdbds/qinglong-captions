from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from module.auto_rig.contracts import load_auto_rig_input_contract
from module.auto_rig.input_identity import (
    TARGET_INPUT_IDENTITY_VERSION,
    AutoRigInputIdentityError,
    build_target_input_identity,
)
from module.auto_rig.jcs import jcs_sha256
from module.auto_rig.tag_registry import CANONICAL_TAG_REGISTRY_VERSION
from tests.test_auto_rig_contracts import _write_item


def test_target_input_identity_freezes_sorted_raw_inventory_without_absolute_paths(
    tmp_path: Path,
) -> None:
    _write_item(tmp_path, tags=("front hair", "face"))
    contract = load_auto_rig_input_contract(tmp_path)

    identity = build_target_input_identity(contract)

    assert identity.schema_version == TARGET_INPUT_IDENTITY_VERSION
    assert identity.canonical_tag_registry_version == CANONICAL_TAG_REGISTRY_VERSION
    assert identity.target_input_fingerprint == jcs_sha256(identity.semantic_payload())
    assert tuple(item.path for item in identity.input_files) == tuple(
        sorted(item.path for item in identity.input_files)
    )
    assert {item.path for item in identity.input_files} == {
        "src_img.png",
        "layerdiff/manifest.json",
        "optimized/manifest.json",
        "optimized/info.json",
        "optimized/face.png",
        "optimized/face_depth.png",
        "optimized/front hair.png",
        "optimized/front hair_depth.png",
    }
    assert str(tmp_path.resolve()) not in str(identity.semantic_payload())


def test_target_input_identity_is_independent_of_contract_part_iteration_order(
    tmp_path: Path,
) -> None:
    _write_item(tmp_path, tags=("face", "front hair"))
    contract = load_auto_rig_input_contract(tmp_path)

    forward = build_target_input_identity(contract)
    reversed_parts = build_target_input_identity(
        replace(contract, parts=tuple(reversed(contract.parts)))
    )

    assert reversed_parts == forward


@pytest.mark.parametrize(
    "relative_path",
    (
        "src_img.png",
        "layerdiff/manifest.json",
        "optimized/manifest.json",
        "optimized/info.json",
        "optimized/face.png",
        "optimized/face_depth.png",
    ),
)
def test_target_input_identity_rejects_snapshot_mutation(
    tmp_path: Path,
    relative_path: str,
) -> None:
    _write_item(tmp_path, tags=("face",))
    contract = load_auto_rig_input_contract(tmp_path)
    path = tmp_path / Path(*relative_path.split("/"))
    path.write_bytes(path.read_bytes() + b"changed")

    with pytest.raises(AutoRigInputIdentityError) as captured:
        build_target_input_identity(contract)

    assert captured.value.code == "input_contract_mismatch"


def test_target_input_identity_deduplicates_shared_psd_payload_files(
    tmp_path: Path,
) -> None:
    _write_item(tmp_path, mode="psd", tags=("face", "front hair"))
    contract = load_auto_rig_input_contract(tmp_path)

    identity = build_target_input_identity(contract)

    assert tuple(item.path for item in identity.input_files) == (
        "final.psd",
        "final_depth.psd",
        "layerdiff/manifest.json",
        "optimized/info.json",
        "optimized/manifest.json",
        "src_img.png",
    )
    assert [part.source_tag for part in identity.parts] == ["face", "front hair"]


def test_target_identity_excludes_override_derived_canonical_aliases(tmp_path: Path) -> None:
    _write_item(
        tmp_path,
        edge=768,
        tags=("objects_1", "objects_2"),
        tblr_split=True,
    )

    handwear = build_target_input_identity(
        load_auto_rig_input_contract(
            tmp_path,
            tag_aliases={
                "objects_1": "handwear-l",
                "objects_2": "handwear-r",
            },
        )
    )
    eyes = build_target_input_identity(
        load_auto_rig_input_contract(
            tmp_path,
            tag_aliases={
                "objects_1": "eyewhite-l",
                "objects_2": "eyewhite-r",
            },
        )
    )

    assert handwear.target_input_fingerprint == eyes.target_input_fingerprint
