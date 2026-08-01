from __future__ import annotations

import json
from pathlib import Path

import pytest

from module.auto_rig.contracts import load_auto_rig_input_contract
from module.auto_rig.input_identity import build_target_input_identity
from module.auto_rig.jcs import jcs_sha256
from module.auto_rig.overrides import (
    OVERRIDE_INPUT_IDENTITY_VERSION,
    RIG_OVERRIDES_PATH,
    RigOverrideContractError,
    identify_rig_override_input,
    load_rig_override_source,
    validate_rig_override_source,
)
from tests.test_auto_rig_contracts import _write_item


def _target(root: Path):
    _write_item(root, edge=768, tags=("face", "handwear"), tblr_split=True)
    return build_target_input_identity(load_auto_rig_input_contract(root))


def _payload(target_fingerprint: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "target_input_fingerprint": target_fingerprint,
        "joints": {
            "joint/elbow.xmin": {"x": 120.5, "y": 240},
        },
        "tag_aliases": {"objects_2": "handwear-r"},
    }


def _write_override(root: Path, payload: object, *, indent: int | None = None) -> Path:
    path = root / RIG_OVERRIDES_PATH
    path.write_text(json.dumps(payload, indent=indent, sort_keys=True), encoding="utf-8")
    return path


def test_missing_override_has_total_non_null_identity(tmp_path: Path) -> None:
    identity = identify_rig_override_input(tmp_path)
    source = load_rig_override_source(tmp_path)

    assert identity.schema_version == OVERRIDE_INPUT_IDENTITY_VERSION
    assert identity.present is False
    assert identity.file_sha256 is None
    assert identity.semantic_payload() == {
        "schema": OVERRIDE_INPUT_IDENTITY_VERSION,
        "present": False,
    }
    assert identity.rig_overrides_sha256 == jcs_sha256(identity.semantic_payload())
    assert source.identity == identity
    assert source.target_input_fingerprint is None
    assert source.joints == ()
    assert source.tag_aliases == ()


def test_override_identity_tracks_raw_bytes_not_only_json_semantics(tmp_path: Path) -> None:
    target = _target(tmp_path)
    payload = _payload(target.target_input_fingerprint)
    path = _write_override(tmp_path, payload, indent=None)
    compact = identify_rig_override_input(tmp_path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pretty = identify_rig_override_input(tmp_path)

    assert compact.file_sha256 != pretty.file_sha256
    assert compact.rig_overrides_sha256 != pretty.rig_overrides_sha256


def test_override_parser_rejects_bytes_that_changed_after_identification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target(tmp_path)
    path = _write_override(tmp_path, _payload(target.target_input_fingerprint))
    replacement = json.dumps(
        {
            **_payload(target.target_input_fingerprint),
            "joints": {"joint/neck": {"x": 10, "y": 20}},
        },
        sort_keys=True,
    ).encode("utf-8")
    original_read_bytes = Path.read_bytes

    def swapped_read_bytes(candidate: Path) -> bytes:
        if candidate == path:
            return replacement
        return original_read_bytes(candidate)

    monkeypatch.setattr(Path, "read_bytes", swapped_read_bytes)

    with pytest.raises(RigOverrideContractError, match="changed during parsing"):
        load_rig_override_source(tmp_path)


def test_override_source_parses_stable_aliases_and_validates_joint_bounds(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    _write_override(tmp_path, _payload(target.target_input_fingerprint), indent=2)

    source = load_rig_override_source(tmp_path)
    validated = validate_rig_override_source(source, target)

    assert source.tag_aliases == (("objects_2", "handwear-r"),)
    assert validated.target_input_fingerprint == target.target_input_fingerprint
    assert validated.joints[0].joint_id == "joint/elbow.xmin"
    assert (validated.joints[0].x, validated.joints[0].y) == (120.5, 240.0)
    assert validated.outside_joint_ids == ()


@pytest.mark.parametrize(
    "raw",
    (
        '{"schema_version":1,"schema_version":1}',
        '{"schema_version":1,"target_input_fingerprint":"bad","joints":{},"tag_aliases":{},"extra":1}',
        '{not-json',
        '{"schema_version":1,"target_input_fingerprint":"bad","joints":{"joint/elbow.xmin":{"x":NaN,"y":2}},"tag_aliases":{}}',
    ),
)
def test_invalid_override_reports_the_already_computed_file_identity(
    tmp_path: Path,
    raw: str,
) -> None:
    (tmp_path / RIG_OVERRIDES_PATH).write_text(raw, encoding="utf-8")
    identity = identify_rig_override_input(tmp_path)

    with pytest.raises(RigOverrideContractError) as captured:
        load_rig_override_source(tmp_path)

    assert captured.value.identity == identity
    assert captured.value.code == "input_contract_mismatch"


def test_override_rejects_stale_target_fingerprint(tmp_path: Path) -> None:
    target = _target(tmp_path)
    payload = _payload("sha256:" + "0" * 64)
    _write_override(tmp_path, payload)
    source = load_rig_override_source(tmp_path)

    with pytest.raises(RigOverrideContractError, match="target_input_fingerprint"):
        validate_rig_override_source(source, target)


def test_override_requires_explicit_allow_outside_and_records_warning_scope(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    payload = _payload(target.target_input_fingerprint)
    payload["joints"] = {"joint/head_top": {"x": 768, "y": -1}}
    _write_override(tmp_path, payload)
    source = load_rig_override_source(tmp_path)

    with pytest.raises(RigOverrideContractError, match="outside"):
        validate_rig_override_source(source, target)

    payload["joints"] = {
        "joint/head_top": {"x": 768, "y": -1, "allow_outside": True}
    }
    _write_override(tmp_path, payload)
    validated = validate_rig_override_source(load_rig_override_source(tmp_path), target)

    assert validated.outside_joint_ids == ("joint/head_top",)


@pytest.mark.parametrize(
    "joints,aliases",
    (
        ({"joint/unknown": {"x": 1, "y": 2}}, {}),
        ({"joint/elbow.xmin": {"x": True, "y": 2}}, {}),
        ({}, {"../escape": "face"}),
        ({}, {"objects_2": "unknown-tag"}),
    ),
)
def test_override_rejects_unknown_joint_or_unsafe_alias_contract(
    tmp_path: Path,
    joints: dict[str, object],
    aliases: dict[str, str],
) -> None:
    target = _target(tmp_path)
    payload = _payload(target.target_input_fingerprint)
    payload["joints"] = joints
    payload["tag_aliases"] = aliases
    _write_override(tmp_path, payload)

    with pytest.raises(RigOverrideContractError):
        load_rig_override_source(tmp_path)


def test_override_aliases_are_parsed_before_target_binding_without_a_digest_cycle(
    tmp_path: Path,
) -> None:
    _write_item(
        tmp_path,
        edge=768,
        tags=("objects_1", "objects_2"),
        tblr_split=True,
    )
    payload = _payload("sha256:" + "0" * 64)
    payload["tag_aliases"] = {
        "objects_1": "handwear-l",
        "objects_2": "handwear-r",
    }
    _write_override(tmp_path, payload)

    unbound = load_rig_override_source(tmp_path)
    contract = load_auto_rig_input_contract(
        tmp_path,
        tag_aliases=dict(unbound.tag_aliases),
    )
    target = build_target_input_identity(contract)
    payload["target_input_fingerprint"] = target.target_input_fingerprint
    _write_override(tmp_path, payload, indent=2)

    rebound = load_rig_override_source(tmp_path)
    rebound_contract = load_auto_rig_input_contract(
        tmp_path,
        tag_aliases=dict(rebound.tag_aliases),
    )
    rebound_target = build_target_input_identity(rebound_contract)
    validated = validate_rig_override_source(rebound, rebound_target)

    assert rebound_target.target_input_fingerprint == target.target_input_fingerprint
    assert validated.target_input_fingerprint == target.target_input_fingerprint
    assert {part.part_id for part in rebound_contract.parts} == {
        "part/handwear.xmin",
        "part/handwear.xmax",
    }
