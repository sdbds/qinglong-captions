from __future__ import annotations

import hashlib
import math
from typing import Any

_IEEE754_SAFE_INTEGER = 9_007_199_254_740_991
_SHORT_ESCAPES = {
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
    '"': '\\"',
    "\\": "\\\\",
}


class JcsContractError(ValueError):
    """Raised when a value cannot be represented as RFC 8785 JSON."""


def _validate_unicode(value: str) -> None:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise JcsContractError("JCS strings must not contain lone UTF-16 surrogates")


def _encode_string(value: str) -> str:
    _validate_unicode(value)
    encoded = ['"']
    for character in value:
        escaped = _SHORT_ESCAPES.get(character)
        if escaped is not None:
            encoded.append(escaped)
            continue
        codepoint = ord(character)
        if codepoint <= 0x1F:
            encoded.append(f"\\u{codepoint:04x}")
        else:
            encoded.append(character)
    encoded.append('"')
    return "".join(encoded)


def _utf16_sort_key(value: str) -> bytes:
    _validate_unicode(value)
    return value.encode("utf-16-be")


def _expand_scientific(mantissa: str, exponent: int) -> str:
    digits = mantissa.replace(".", "")
    decimal_position = 1 + exponent
    if decimal_position <= 0:
        return f"0.{('0' * -decimal_position)}{digits}"
    if decimal_position >= len(digits):
        return f"{digits}{'0' * (decimal_position - len(digits))}"
    return f"{digits[:decimal_position]}.{digits[decimal_position:]}"


def _encode_float(value: float) -> str:
    if not math.isfinite(value):
        raise JcsContractError("JCS numbers must be finite")
    if value == 0.0:
        return "0"

    sign = "-" if value < 0.0 else ""
    magnitude = abs(value)
    rendered = repr(magnitude).lower()
    if "e" not in rendered:
        if rendered.endswith(".0"):
            rendered = rendered[:-2]
        return f"{sign}{rendered}"

    mantissa, raw_exponent = rendered.split("e", 1)
    exponent = int(raw_exponent)
    if 1e-6 <= magnitude < 1e21:
        return f"{sign}{_expand_scientific(mantissa, exponent)}"

    exponent_sign = "+" if exponent >= 0 else "-"
    return f"{sign}{mantissa}e{exponent_sign}{abs(exponent)}"


def _encode_value(value: Any, active_containers: set[int]) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        if abs(value) > _IEEE754_SAFE_INTEGER:
            raise JcsContractError("JCS integer exceeds the I-JSON safe integer range")
        return str(value)
    if isinstance(value, float):
        return _encode_float(value)
    if isinstance(value, str):
        return _encode_string(value)

    if type(value) is list:
        identity = id(value)
        if identity in active_containers:
            raise JcsContractError("JCS arrays must not contain cycles")
        active_containers.add(identity)
        try:
            return "[" + ",".join(_encode_value(item, active_containers) for item in value) + "]"
        finally:
            active_containers.remove(identity)

    if type(value) is dict:
        identity = id(value)
        if identity in active_containers:
            raise JcsContractError("JCS objects must not contain cycles")
        if any(not isinstance(key, str) for key in value):
            raise JcsContractError("JCS object keys must be strings")
        active_containers.add(identity)
        try:
            members = []
            for key in sorted(value, key=_utf16_sort_key):
                members.append(f"{_encode_string(key)}:{_encode_value(value[key], active_containers)}")
            return "{" + ",".join(members) + "}"
        finally:
            active_containers.remove(identity)

    raise JcsContractError(f"unsupported JCS value type: {type(value).__name__}")


def jcs_bytes(payload: Any) -> bytes:
    """Serialize a JSON-compatible value using RFC 8785 canonicalization."""

    return _encode_value(payload, set()).encode("utf-8")


def jcs_sha256(payload: Any) -> str:
    """Return the repository digest form for RFC 8785 canonical bytes."""

    return f"sha256:{hashlib.sha256(jcs_bytes(payload)).hexdigest()}"


__all__ = ["JcsContractError", "jcs_bytes", "jcs_sha256"]
