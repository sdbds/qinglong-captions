from __future__ import annotations

import hashlib
import math
from json.encoder import encode_basestring as _c_encode_string
from typing import Any

_IEEE754_SAFE_INTEGER = 9_007_199_254_740_991


class JcsContractError(ValueError):
    """Raised when a value cannot be represented as RFC 8785 JSON."""


def _utf16_sort_key(value: str) -> bytes:
    try:
        return value.encode("utf-16-be")
    except UnicodeEncodeError as exc:
        raise JcsContractError("JCS strings must not contain lone UTF-16 surrogates") from exc


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
    t = type(value)
    if t is str:
        return _c_encode_string(value)
    if t is int:
        if value > _IEEE754_SAFE_INTEGER or value < -_IEEE754_SAFE_INTEGER:
            raise JcsContractError("JCS integer exceeds the I-JSON safe integer range")
        return str(value)
    if t is float:
        return _encode_float(value)
    if value is None:
        return "null"
    if t is bool:
        return "true" if value else "false"

    if t is list:
        identity = id(value)
        if identity in active_containers:
            raise JcsContractError("JCS arrays must not contain cycles")
        active_containers.add(identity)
        try:
            return "[" + ",".join(_encode_value(item, active_containers) for item in value) + "]"
        finally:
            active_containers.remove(identity)

    if t is dict:
        identity = id(value)
        if identity in active_containers:
            raise JcsContractError("JCS objects must not contain cycles")
        all_ascii = True
        for key in value:
            kt = type(key)
            if kt is str:
                if all_ascii and not key.isascii():
                    all_ascii = False
            elif isinstance(key, str):
                if all_ascii and not key.isascii():
                    all_ascii = False
            else:
                raise JcsContractError("JCS object keys must be strings")
        active_containers.add(identity)
        try:
            keys = sorted(value) if all_ascii else sorted(value, key=_utf16_sort_key)
            return "{" + ",".join(
                f"{_c_encode_string(key)}:{_encode_value(value[key], active_containers)}" for key in keys
            ) + "}"
        finally:
            active_containers.remove(identity)

    # Subclasses of int / float / str (e.g. IntEnum, StrEnum)
    if isinstance(value, int):
        if value > _IEEE754_SAFE_INTEGER or value < -_IEEE754_SAFE_INTEGER:
            raise JcsContractError("JCS integer exceeds the I-JSON safe integer range")
        return str(int(value))
    if isinstance(value, float):
        return _encode_float(float(value))
    if isinstance(value, str):
        return _c_encode_string(str(value))

    raise JcsContractError(f"unsupported JCS value type: {t.__name__}")


def jcs_bytes(payload: Any) -> bytes:
    """Serialize a JSON-compatible value using RFC 8785 canonicalization."""

    rendered = _encode_value(payload, set())
    try:
        return rendered.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise JcsContractError("JCS strings must not contain lone UTF-16 surrogates") from exc


def jcs_sha256(payload: Any) -> str:
    """Return the repository digest form for RFC 8785 canonical bytes."""

    return f"sha256:{hashlib.sha256(jcs_bytes(payload)).hexdigest()}"


__all__ = ["JcsContractError", "jcs_bytes", "jcs_sha256"]
