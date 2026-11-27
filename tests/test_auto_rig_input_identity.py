from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from module.auto_rig.contracts import load_auto_rig_input_contract
from module.auto_rig.input_identity import (
    AutoRigInputIdentityError,
    build_target_input_identity,
)
from tests.test_auto_rig_contracts import _write_item


def test_target_input_identity_is_deterministic_and_path_free(tmp_path: Path) -> None:
    _write_item(tmp_path, tags=("front hair", "face"))
    contract = load_auto_rig_input_contract(tmp_path)

    forward = build_target_input_identity(contract)
    reversed_parts = build_target_input_identity(replace(contract, parts=tuple(reversed(contract.parts))))

    assert reversed_parts == forward
    assert str(tmp_path.resolve()) not in str(forward.semantic_payload())


def test_target_input_identity_rejects_changed_input_file(tmp_path: Path) -> None:
    _write_item(tmp_path, tags=("face",))
    contract = load_auto_rig_input_contract(tmp_path)
    (tmp_path / "src_img.png").write_bytes((tmp_path / "src_img.png").read_bytes() + b"changed")

    with pytest.raises(AutoRigInputIdentityError) as captured:
        build_target_input_identity(contract)

    assert captured.value.code == "input_contract_mismatch"


def test_target_input_identity_deduplicates_psd_files(tmp_path: Path) -> None:
    _write_item(tmp_path, mode="psd", tags=("face", "front hair"))
    identity = build_target_input_identity(load_auto_rig_input_contract(tmp_path))

    assert "final.psd" in {item.path for item in identity.input_files}
    assert [part.source_tag for part in identity.parts] == ["face", "front hair"]
