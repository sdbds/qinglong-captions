from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ArtifactContractError, sha256_file
from .input_identity import TargetInputIdentity
from .jcs import jcs_sha256
from .joint_registry import JOINT_ID_SET
from .tag_registry import AutoRigTagContractError, decode_v3_source_tag

RIG_OVERRIDES_PATH = "rig_overrides.json"
OVERRIDE_INPUT_IDENTITY_VERSION = "rig-overrides-input-v1"
RIG_OVERRIDES_SCHEMA_VERSION = 1

_REPARSE_POINT_ATTRIBUTE = 0x400
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class OverrideInputIdentity:
    schema_version: str
    present: bool
    file_sha256: str | None
    rig_overrides_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema_version,
            "present": self.present,
        }
        if self.present:
            payload["file_sha256"] = self.file_sha256
        return payload


class RigOverrideContractError(ValueError):
    """Raised when the fixed auto-rig override input is not safe to apply."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        identity: OverrideInputIdentity | None = None,
    ) -> None:
        self.code = code
        self.identity = identity
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class JointOverride:
    joint_id: str
    x: float
    y: float
    allow_outside: bool


@dataclass(frozen=True, slots=True)
class RigOverrideSource:
    identity: OverrideInputIdentity
    target_input_fingerprint: str | None
    joints: tuple[JointOverride, ...]
    tag_aliases: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ValidatedRigOverrides:
    identity: OverrideInputIdentity
    target_input_fingerprint: str | None
    joints: tuple[JointOverride, ...]
    tag_aliases: tuple[tuple[str, str], ...]
    outside_joint_ids: tuple[str, ...]


def _error(
    message: str,
    *,
    identity: OverrideInputIdentity | None = None,
) -> RigOverrideContractError:
    return RigOverrideContractError(
        "input_contract_mismatch",
        message,
        identity=identity,
    )


def _is_reparse_point(path: Path) -> bool:
    status = os.lstat(path)
    return stat.S_ISLNK(status.st_mode) or bool(
        getattr(status, "st_file_attributes", 0) & _REPARSE_POINT_ATTRIBUTE
    )


def _build_identity(*, present: bool, file_sha256: str | None) -> OverrideInputIdentity:
    values = {
        "schema_version": OVERRIDE_INPUT_IDENTITY_VERSION,
        "present": present,
        "file_sha256": file_sha256,
    }
    provisional = OverrideInputIdentity(**values, rig_overrides_sha256="")
    return OverrideInputIdentity(
        **values,
        rig_overrides_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def identify_rig_override_input(item_root: str | Path) -> OverrideInputIdentity:
    """Identify the fixed override path without requiring its JSON to be valid."""

    root = Path(item_root).expanduser().absolute()
    path = root / RIG_OVERRIDES_PATH
    try:
        if not root.is_dir() or _is_reparse_point(root):
            raise _error("item root must be a non-link directory")
        try:
            os.lstat(path)
        except FileNotFoundError:
            return _build_identity(present=False, file_sha256=None)
        if _is_reparse_point(path) or not path.is_file():
            raise _error("rig_overrides.json must be a non-link regular file")
        return _build_identity(present=True, file_sha256=sha256_file(path))
    except RigOverrideContractError:
        raise
    except (ArtifactContractError, OSError) as exc:
        raise _error("rig_overrides.json cannot be inspected") from exc


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key: {key}")
        result[key] = value
    return result


def _require_exact_fields(
    payload: dict[str, Any],
    expected: set[str],
    *,
    field: str,
) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ValueError(f"{field} fields mismatch; missing={missing}, extra={extra}")


def _coordinate(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be finite")
    return 0.0 if normalized == 0.0 else normalized


def _parse_joint_overrides(payload: Any) -> tuple[JointOverride, ...]:
    if type(payload) is not dict:
        raise ValueError("joints must be an object")
    joints: list[JointOverride] = []
    for joint_id in sorted(payload):
        if joint_id not in JOINT_ID_SET:
            raise ValueError(f"unknown joint override ID: {joint_id}")
        record = payload[joint_id]
        if type(record) is not dict:
            raise ValueError(f"joint override must be an object: {joint_id}")
        if set(record) not in ({"x", "y"}, {"x", "y", "allow_outside"}):
            raise ValueError(f"joint override fields mismatch: {joint_id}")
        allow_outside = record.get("allow_outside", False)
        if type(allow_outside) is not bool:
            raise ValueError(f"joint allow_outside must be a boolean: {joint_id}")
        joints.append(
            JointOverride(
                joint_id=joint_id,
                x=_coordinate(record["x"], field=f"{joint_id}.x"),
                y=_coordinate(record["y"], field=f"{joint_id}.y"),
                allow_outside=allow_outside,
            )
        )
    return tuple(joints)


def _parse_tag_aliases(payload: Any) -> tuple[tuple[str, str], ...]:
    if type(payload) is not dict:
        raise ValueError("tag_aliases must be an object")
    aliases: list[tuple[str, str]] = []
    for raw_tag in sorted(payload):
        canonical_tag = payload[raw_tag]
        if (
            not isinstance(raw_tag, str)
            or not raw_tag
            or len(raw_tag) > 128
            or raw_tag in {".", ".."}
            or "/" in raw_tag
            or "\\" in raw_tag
            or any(ord(character) < 32 or ord(character) == 127 for character in raw_tag)
        ):
            raise ValueError(f"unsafe raw tag alias key: {raw_tag!r}")
        if not isinstance(canonical_tag, str):
            raise ValueError(f"tag alias target must be a string: {raw_tag}")
        try:
            decode_v3_source_tag(canonical_tag)
        except AutoRigTagContractError as exc:
            raise ValueError(f"invalid canonical tag alias target: {canonical_tag}") from exc
        aliases.append((raw_tag, canonical_tag))
    return tuple(aliases)


def load_rig_override_source(item_root: str | Path) -> RigOverrideSource:
    """Strictly parse override syntax while retaining a raw-byte identity on errors."""

    identity = identify_rig_override_input(item_root)
    if not identity.present:
        return RigOverrideSource(
            identity=identity,
            target_input_fingerprint=None,
            joints=(),
            tag_aliases=(),
        )
    path = Path(item_root).expanduser().absolute() / RIG_OVERRIDES_PATH
    try:
        raw_bytes = path.read_bytes()
        parsed_sha256 = f"sha256:{hashlib.sha256(raw_bytes).hexdigest()}"
        if parsed_sha256 != identity.file_sha256:
            raise _error(
                "rig_overrides.json changed during parsing",
                identity=identity,
            )
        payload = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"JSON contains non-finite number: {value}")
            ),
        )
        if type(payload) is not dict:
            raise ValueError("rig_overrides.json root must be an object")
        _require_exact_fields(
            payload,
            {"schema_version", "target_input_fingerprint", "joints", "tag_aliases"},
            field="rig override",
        )
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ValueError("rig override schema_version must equal 1")
        target = payload["target_input_fingerprint"]
        if not isinstance(target, str) or not _SHA256_PATTERN.fullmatch(target):
            raise ValueError("target_input_fingerprint must be a lowercase SHA-256 digest")
        joints = _parse_joint_overrides(payload["joints"])
        aliases = _parse_tag_aliases(payload["tag_aliases"])
    except RigOverrideContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise _error("rig_overrides.json is invalid", identity=identity) from exc
    return RigOverrideSource(
        identity=identity,
        target_input_fingerprint=target,
        joints=joints,
        tag_aliases=aliases,
    )


def validate_rig_override_source(
    source: RigOverrideSource,
    target: TargetInputIdentity,
) -> ValidatedRigOverrides:
    """Bind parsed overrides to a target snapshot and validate canvas coordinates."""

    if not isinstance(source, RigOverrideSource) or not isinstance(target, TargetInputIdentity):
        raise _error("override validation requires typed source and target identities")
    if source.identity.present and source.target_input_fingerprint != target.target_input_fingerprint:
        raise _error(
            "rig override target_input_fingerprint does not match this item",
            identity=source.identity,
        )
    outside: list[str] = []
    for joint in source.joints:
        is_outside = not (
            0.0 <= joint.x < target.canvas_width
            and 0.0 <= joint.y < target.canvas_height
        )
        if is_outside and not joint.allow_outside:
            raise _error(
                f"joint override is outside the Rig canvas: {joint.joint_id}",
                identity=source.identity,
            )
        if is_outside:
            outside.append(joint.joint_id)
    return ValidatedRigOverrides(
        identity=source.identity,
        target_input_fingerprint=source.target_input_fingerprint,
        joints=source.joints,
        tag_aliases=source.tag_aliases,
        outside_joint_ids=tuple(sorted(outside)),
    )


__all__ = [
    "OVERRIDE_INPUT_IDENTITY_VERSION",
    "RIG_OVERRIDES_PATH",
    "RIG_OVERRIDES_SCHEMA_VERSION",
    "JointOverride",
    "OverrideInputIdentity",
    "RigOverrideContractError",
    "RigOverrideSource",
    "ValidatedRigOverrides",
    "identify_rig_override_input",
    "load_rig_override_source",
    "validate_rig_override_source",
]
