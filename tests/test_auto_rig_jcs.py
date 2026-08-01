from __future__ import annotations

import hashlib
import struct

import pytest

from module.auto_rig.jcs import JcsContractError, jcs_bytes, jcs_sha256


def test_jcs_serializes_ecmascript_number_boundaries() -> None:
    payload = [
        333333333.33333329,
        1e30,
        4.50,
        2e-3,
        1e-27,
        1e-6,
        1e-7,
        1e20,
        1e21,
        -0.0,
    ]

    assert jcs_bytes(payload) == (b"[333333333.3333333,1e+30,4.5,0.002,1e-27,0.000001,1e-7,100000000000000000000,1e+21,0]")


def test_jcs_serializes_integral_floats_without_decimal_suffix() -> None:
    assert jcs_bytes([1.0, -42.0, 1_000_000.0]) == b"[1,-42,1000000]"


@pytest.mark.parametrize(
    ("ieee754", "expected"),
    [
        ("0000000000000000", "0"),
        ("8000000000000000", "0"),
        ("0000000000000001", "5e-324"),
        ("8000000000000001", "-5e-324"),
        ("7fefffffffffffff", "1.7976931348623157e+308"),
        ("ffefffffffffffff", "-1.7976931348623157e+308"),
        ("4340000000000000", "9007199254740992"),
        ("c340000000000000", "-9007199254740992"),
        ("4430000000000000", "295147905179352830000"),
        ("44b52d02c7e14af5", "9.999999999999997e+22"),
        ("44b52d02c7e14af6", "1e+23"),
        ("44b52d02c7e14af7", "1.0000000000000001e+23"),
        ("444b1ae4d6e2ef4e", "999999999999999700000"),
        ("444b1ae4d6e2ef4f", "999999999999999900000"),
        ("444b1ae4d6e2ef50", "1e+21"),
        ("3eb0c6f7a0b5ed8c", "9.999999999999997e-7"),
        ("3eb0c6f7a0b5ed8d", "0.000001"),
        ("41b3de4355555553", "333333333.3333332"),
        ("41b3de4355555554", "333333333.33333325"),
        ("41b3de4355555555", "333333333.3333333"),
        ("41b3de4355555556", "333333333.3333334"),
        ("41b3de4355555557", "333333333.33333343"),
        ("becbf647612f3696", "-0.0000033333333333333333"),
        ("43143ff3c1cb0959", "1424953923781206.2"),
    ],
)
def test_jcs_matches_rfc8785_appendix_b_number_vectors(ieee754: str, expected: str) -> None:
    value = struct.unpack(">d", bytes.fromhex(ieee754))[0]

    assert jcs_bytes(value) == expected.encode("ascii")


def test_jcs_sorts_object_keys_by_utf16_code_units() -> None:
    payload = {
        "\ufb33": "hebrew",
        "1": "ascii",
        "\U0001f600": "astral",
        "\r": "control",
        "\u20ac": "euro",
        "\u0080": "control-extended",
        "\u00f6": "latin",
    }

    assert jcs_bytes(payload).decode("utf-8") == (
        '{"\\r":"control","1":"ascii","\u0080":"control-extended",'
        '"\u00f6":"latin","\u20ac":"euro","\U0001f600":"astral","\ufb33":"hebrew"}'
    )


def test_jcs_uses_json_control_escapes_without_escaping_non_ascii() -> None:
    payload = {"s": '\b\t\n\f\r"\\\x00/\u20ac'}

    assert jcs_bytes(payload) == b'{"s":"\\b\\t\\n\\f\\r\\"\\\\\\u0000/\xe2\x82\xac"}'


@pytest.mark.parametrize(
    "payload",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        9_007_199_254_740_992,
        -9_007_199_254_740_992,
        "\ud800",
        {1: "not-a-string-key"},
        {"not": {"json"}},
        ("tuple",),
        b"bytes",
    ],
)
def test_jcs_rejects_values_outside_the_ijson_data_model(payload: object) -> None:
    with pytest.raises(JcsContractError):
        jcs_bytes(payload)


def test_jcs_sha256_hashes_exact_canonical_bytes() -> None:
    payload = {"z": 1, "a": [True, None, "\u20ac"]}
    expected = hashlib.sha256(jcs_bytes(payload)).hexdigest()

    assert jcs_sha256(payload) == f"sha256:{expected}"
