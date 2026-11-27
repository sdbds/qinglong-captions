from __future__ import annotations

import hashlib

import pytest

from module.auto_rig.jcs import JcsContractError, jcs_bytes, jcs_sha256


def test_jcs_serializes_numbers_keys_and_escapes() -> None:
    assert jcs_bytes([1.0, -42.0, -0.0, 1e-6, 1e21]) == b"[1,-42,0,0.000001,1e+21]"
    assert jcs_bytes({"b": 1, "a": '\b\t\n"\\\u20ac'}) == b'{"a":"\\b\\t\\n\\"\\\\\xe2\x82\xac","b":1}'
    payload = {"z": 1, "a": [True, None]}
    assert jcs_sha256(payload) == f"sha256:{hashlib.sha256(jcs_bytes(payload)).hexdigest()}"


@pytest.mark.parametrize("payload", [float("nan"), float("inf"), {1: "bad"}, b"bytes"])
def test_jcs_rejects_non_ijson_values(payload: object) -> None:
    with pytest.raises(JcsContractError):
        jcs_bytes(payload)
