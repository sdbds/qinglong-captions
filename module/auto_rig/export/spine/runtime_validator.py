from __future__ import annotations

import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ...artifacts import sha256_file
from ...jcs import jcs_sha256

SPINE_RUNTIME_VALIDATOR_VERSION = "spine-runtime-validator-v3"
SPINE_RUNTIME_REPORT_SCHEMA_VERSION = "auto-rig-spine-runtime-v3"
SPINE_RUNTIME_VALIDATOR_PROTOCOL_DIGEST = "sha256:fa260823ed4e0ac2028263d36132f730e2791bf0b508beae36e6e68daa950b62"
SPINE_RUNTIME_EXPECTED_VERSION = "4.2"
SPINE_RUNTIME_SAMPLE_RATE_HZ = 60
SPINE_RUNTIME_SETUP_RESTORE_TOLERANCE = 1.0e-4
SPINE_RUNTIME_BREATH_MIN_WORLD_VERTEX_DELTA_RATIO = 0.005
SPINE_RUNTIME_TALK_MIN_WORLD_VERTEX_DELTA_RATIO = 0.002
SPINE_RUNTIME_TALK_MIN_SLOT_ALPHA_DELTA = 0.8


class SpineRuntimeValidationError(RuntimeError):
    """Raised when the external official Spine Runtime gate does not pass."""


@dataclass(frozen=True, slots=True)
class SpineRuntimeAnimationEvidence:
    name: str
    duration: float
    sample_count: int
    finite: bool
    visible_change: bool
    maximum_numeric_state_delta: float
    maximum_world_vertex_displacement: float
    maximum_world_vertex_displacement_ratio: float
    maximum_slot_alpha_delta: float
    maximum_setup_restore_residual: float

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "duration": self.duration,
            "sample_count": self.sample_count,
            "finite": self.finite,
            "visible_change": self.visible_change,
            "maximum_numeric_state_delta": self.maximum_numeric_state_delta,
            "maximum_world_vertex_displacement": (self.maximum_world_vertex_displacement),
            "maximum_world_vertex_displacement_ratio": (self.maximum_world_vertex_displacement_ratio),
            "maximum_slot_alpha_delta": self.maximum_slot_alpha_delta,
            "maximum_setup_restore_residual": (self.maximum_setup_restore_residual),
        }


@dataclass(frozen=True, slots=True)
class SpineRuntimeValidationReport:
    validated: bool
    validator_version: str
    validator_protocol_digest: str
    runtime_executable_sha256: str
    runtime_executable_size: int
    runtime_version: str
    skeleton_version: str
    skeleton_sha256: str
    atlas_sha256: str
    sample_rate_hz: int
    bone_count: int
    slot_count: int
    skin_count: int
    setup_attachment_count: int
    animation_count: int
    atlas_page_count: int
    atlas_region_count: int
    animations: tuple[SpineRuntimeAnimationEvidence, ...]
    report_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "validated": self.validated,
            "validator_version": self.validator_version,
            "validator_protocol_digest": self.validator_protocol_digest,
            "runtime_executable_sha256": self.runtime_executable_sha256,
            "runtime_executable_size": self.runtime_executable_size,
            "runtime_version": self.runtime_version,
            "skeleton_version": self.skeleton_version,
            "skeleton_sha256": self.skeleton_sha256,
            "atlas_sha256": self.atlas_sha256,
            "sample_rate_hz": self.sample_rate_hz,
            "bone_count": self.bone_count,
            "slot_count": self.slot_count,
            "skin_count": self.skin_count,
            "setup_attachment_count": self.setup_attachment_count,
            "animation_count": self.animation_count,
            "atlas_page_count": self.atlas_page_count,
            "atlas_region_count": self.atlas_region_count,
            "animations": [record.to_dict() for record in self.animations],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "report_sha256": self.report_sha256}


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise SpineRuntimeValidationError(f"{context} fields do not match protocol; missing={missing}, extra={extra}")


def _require_string(payload: Mapping[str, Any], key: str, *, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise SpineRuntimeValidationError(f"{context}.{key} must be a non-empty string")
    return value


def _require_bool(payload: Mapping[str, Any], key: str, *, context: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise SpineRuntimeValidationError(f"{context}.{key} must be a boolean")
    return value


def _require_integer(
    payload: Mapping[str, Any],
    key: str,
    *,
    context: str,
    minimum: int = 0,
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SpineRuntimeValidationError(f"{context}.{key} must be an integer >= {minimum}")
    return value


def _require_finite_number(
    payload: Mapping[str, Any],
    key: str,
    *,
    context: str,
    minimum: float = 0.0,
) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpineRuntimeValidationError(f"{context}.{key} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < minimum:
        raise SpineRuntimeValidationError(f"{context}.{key} must be finite and >= {minimum}")
    return normalized


def _load_skeleton_version(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SpineRuntimeValidationError(f"Spine skeleton JSON cannot be read: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("skeleton"), dict):
        raise SpineRuntimeValidationError("Spine skeleton metadata is missing")
    version = payload["skeleton"].get("spine")
    if not isinstance(version, str) or not version:
        raise SpineRuntimeValidationError("Spine skeleton version is missing")
    return version


def _parse_animation(payload: Any, *, index: int) -> SpineRuntimeAnimationEvidence:
    context = f"runtime report animation[{index}]"
    if not isinstance(payload, dict):
        raise SpineRuntimeValidationError(f"{context} must be an object")
    _require_exact_keys(
        payload,
        {
            "name",
            "duration",
            "sample_count",
            "finite",
            "visible_change",
            "maximum_numeric_state_delta",
            "maximum_world_vertex_displacement",
            "maximum_world_vertex_displacement_ratio",
            "maximum_slot_alpha_delta",
            "maximum_setup_restore_residual",
        },
        context=context,
    )
    evidence = SpineRuntimeAnimationEvidence(
        name=_require_string(payload, "name", context=context),
        duration=_require_finite_number(payload, "duration", context=context, minimum=0.0),
        sample_count=_require_integer(payload, "sample_count", context=context, minimum=2),
        finite=_require_bool(payload, "finite", context=context),
        visible_change=_require_bool(payload, "visible_change", context=context),
        maximum_numeric_state_delta=_require_finite_number(
            payload,
            "maximum_numeric_state_delta",
            context=context,
        ),
        maximum_world_vertex_displacement=_require_finite_number(
            payload,
            "maximum_world_vertex_displacement",
            context=context,
        ),
        maximum_world_vertex_displacement_ratio=_require_finite_number(
            payload,
            "maximum_world_vertex_displacement_ratio",
            context=context,
        ),
        maximum_slot_alpha_delta=_require_finite_number(
            payload,
            "maximum_slot_alpha_delta",
            context=context,
        ),
        maximum_setup_restore_residual=_require_finite_number(
            payload,
            "maximum_setup_restore_residual",
            context=context,
            minimum=0.0,
        ),
    )
    if evidence.duration <= 0.0:
        raise SpineRuntimeValidationError(f"animation {evidence.name!r} has no positive runtime duration")
    if not evidence.finite:
        raise SpineRuntimeValidationError(f"animation {evidence.name!r} contains non-finite runtime state")
    if not evidence.visible_change:
        raise SpineRuntimeValidationError(f"animation {evidence.name!r} has no runtime-visible state change")
    if (
        evidence.name == "breath"
        and evidence.maximum_world_vertex_displacement_ratio < SPINE_RUNTIME_BREATH_MIN_WORLD_VERTEX_DELTA_RATIO
    ):
        raise SpineRuntimeValidationError(
            f"animation {evidence.name!r} misses the semantic motion threshold: "
            "world vertex displacement ratio "
            f"{evidence.maximum_world_vertex_displacement_ratio} "
            f"< {SPINE_RUNTIME_BREATH_MIN_WORLD_VERTEX_DELTA_RATIO}"
        )
    if evidence.name == "talk":
        geometry_visible = evidence.maximum_world_vertex_displacement_ratio >= SPINE_RUNTIME_TALK_MIN_WORLD_VERTEX_DELTA_RATIO
        opacity_visible = evidence.maximum_slot_alpha_delta >= SPINE_RUNTIME_TALK_MIN_SLOT_ALPHA_DELTA
        if not geometry_visible and not opacity_visible:
            raise SpineRuntimeValidationError(
                f"animation {evidence.name!r} misses the semantic motion threshold: "
                "neither normalized geometry nor opacity changes are visible; "
                f"ratio={evidence.maximum_world_vertex_displacement_ratio} "
                f"(< {SPINE_RUNTIME_TALK_MIN_WORLD_VERTEX_DELTA_RATIO}), "
                f"slot alpha delta={evidence.maximum_slot_alpha_delta} "
                f"(< {SPINE_RUNTIME_TALK_MIN_SLOT_ALPHA_DELTA})"
            )
    if evidence.maximum_setup_restore_residual > SPINE_RUNTIME_SETUP_RESTORE_TOLERANCE:
        raise SpineRuntimeValidationError(f"animation {evidence.name!r} does not restore setup pose")
    return evidence


def _parse_native_report(
    payload: Any,
    *,
    executable: Path,
    skeleton: Path,
    atlas: Path,
    expected_animation_names: tuple[str, ...],
) -> SpineRuntimeValidationReport:
    if not isinstance(payload, dict):
        raise SpineRuntimeValidationError("runtime report must be a JSON object")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "validator_protocol_digest",
            "runtime_version",
            "skeleton_version",
            "sample_rate_hz",
            "bone_count",
            "slot_count",
            "skin_count",
            "setup_attachment_count",
            "animation_count",
            "atlas_page_count",
            "atlas_region_count",
            "animations",
        },
        context="runtime report",
    )
    schema = _require_string(payload, "schema_version", context="runtime report")
    if schema != SPINE_RUNTIME_REPORT_SCHEMA_VERSION:
        raise SpineRuntimeValidationError(f"runtime report schema version mismatch: {schema}")
    protocol = _require_string(payload, "validator_protocol_digest", context="runtime report")
    if protocol != SPINE_RUNTIME_VALIDATOR_PROTOCOL_DIGEST:
        raise SpineRuntimeValidationError("runtime validator protocol digest mismatch")

    runtime_version = _require_string(payload, "runtime_version", context="runtime report")
    skeleton_version = _require_string(payload, "skeleton_version", context="runtime report")
    if runtime_version != SPINE_RUNTIME_EXPECTED_VERSION:
        raise SpineRuntimeValidationError(
            f"official Spine Runtime version mismatch: expected {SPINE_RUNTIME_EXPECTED_VERSION}, got {runtime_version}"
        )
    if skeleton_version != SPINE_RUNTIME_EXPECTED_VERSION:
        raise SpineRuntimeValidationError(
            f"Spine skeleton version mismatch: expected {SPINE_RUNTIME_EXPECTED_VERSION}, got {skeleton_version}"
        )

    raw_animations = payload.get("animations")
    if not isinstance(raw_animations, list) or not raw_animations:
        raise SpineRuntimeValidationError("runtime report must contain animation evidence")
    animations = tuple(_parse_animation(record, index=index) for index, record in enumerate(raw_animations))
    actual_names = tuple(record.name for record in animations)
    if len(set(actual_names)) != len(actual_names):
        raise SpineRuntimeValidationError("runtime animation names must be unique")
    if set(actual_names) != set(expected_animation_names):
        raise SpineRuntimeValidationError(
            f"runtime animation set mismatch: expected={sorted(expected_animation_names)}, actual={sorted(actual_names)}"
        )

    animation_count = _require_integer(payload, "animation_count", context="runtime report", minimum=1)
    if animation_count != len(animations):
        raise SpineRuntimeValidationError("runtime animation_count does not match animation evidence")
    sample_rate = _require_integer(payload, "sample_rate_hz", context="runtime report", minimum=1)
    if sample_rate != SPINE_RUNTIME_SAMPLE_RATE_HZ:
        raise SpineRuntimeValidationError(f"runtime sample rate must be {SPINE_RUNTIME_SAMPLE_RATE_HZ} Hz")

    base: dict[str, object] = {
        "validated": True,
        "validator_version": SPINE_RUNTIME_VALIDATOR_VERSION,
        "validator_protocol_digest": protocol,
        "runtime_executable_sha256": sha256_file(executable),
        "runtime_executable_size": executable.stat().st_size,
        "runtime_version": runtime_version,
        "skeleton_version": skeleton_version,
        "skeleton_sha256": sha256_file(skeleton),
        "atlas_sha256": sha256_file(atlas),
        "sample_rate_hz": sample_rate,
        "bone_count": _require_integer(payload, "bone_count", context="runtime report", minimum=1),
        "slot_count": _require_integer(payload, "slot_count", context="runtime report", minimum=1),
        "skin_count": _require_integer(payload, "skin_count", context="runtime report", minimum=1),
        "setup_attachment_count": _require_integer(
            payload,
            "setup_attachment_count",
            context="runtime report",
            minimum=0,
        ),
        "animation_count": animation_count,
        "atlas_page_count": _require_integer(payload, "atlas_page_count", context="runtime report", minimum=1),
        "atlas_region_count": _require_integer(payload, "atlas_region_count", context="runtime report", minimum=1),
        "animations": [record.to_dict() for record in animations],
    }
    return SpineRuntimeValidationReport(
        validated=True,
        validator_version=SPINE_RUNTIME_VALIDATOR_VERSION,
        validator_protocol_digest=protocol,
        runtime_executable_sha256=str(base["runtime_executable_sha256"]),
        runtime_executable_size=int(base["runtime_executable_size"]),
        runtime_version=runtime_version,
        skeleton_version=skeleton_version,
        skeleton_sha256=str(base["skeleton_sha256"]),
        atlas_sha256=str(base["atlas_sha256"]),
        sample_rate_hz=sample_rate,
        bone_count=int(base["bone_count"]),
        slot_count=int(base["slot_count"]),
        skin_count=int(base["skin_count"]),
        setup_attachment_count=int(base["setup_attachment_count"]),
        animation_count=animation_count,
        atlas_page_count=int(base["atlas_page_count"]),
        atlas_region_count=int(base["atlas_region_count"]),
        animations=animations,
        report_sha256=jcs_sha256(base),
    )


def validate_spine_runtime_bundle(
    executable_path: str | Path,
    skeleton_path: str | Path,
    atlas_path: str | Path,
    *,
    expected_animation_names: tuple[str, ...],
    timeout_seconds: float = 120.0,
) -> SpineRuntimeValidationReport:
    """Load and exercise a Spine 4.2 bundle through the official C++ runtime."""

    executable = Path(executable_path).resolve(strict=True)
    skeleton = Path(skeleton_path).resolve(strict=True)
    atlas = Path(atlas_path).resolve(strict=True)
    if not executable.is_file() or not skeleton.is_file() or not atlas.is_file():
        raise SpineRuntimeValidationError("runtime executable, skeleton, and atlas must be regular files")
    if not expected_animation_names or len(set(expected_animation_names)) != len(expected_animation_names):
        raise SpineRuntimeValidationError("expected animation names must be a non-empty unique tuple")
    source_version = _load_skeleton_version(skeleton)
    if source_version != SPINE_RUNTIME_EXPECTED_VERSION:
        raise SpineRuntimeValidationError(
            f"Spine skeleton version mismatch: expected {SPINE_RUNTIME_EXPECTED_VERSION}, got {source_version}"
        )
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise SpineRuntimeValidationError("runtime timeout must be positive and finite")

    with tempfile.TemporaryDirectory(prefix="auto-rig-spine-runtime-") as temporary:
        report_path = Path(temporary) / "report.json"
        command = [
            str(executable),
            "--skeleton",
            str(skeleton),
            "--atlas",
            str(atlas),
            "--report",
            str(report_path),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SpineRuntimeValidationError(f"official Spine Runtime invocation failed: {exc}") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise SpineRuntimeValidationError("official Spine Runtime rejected the bundle" + (f": {detail}" if detail else ""))
        if not report_path.is_file():
            raise SpineRuntimeValidationError("official Spine Runtime did not produce its evidence report")
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SpineRuntimeValidationError(f"official Spine Runtime report cannot be read: {exc}") from exc

    report = _parse_native_report(
        payload,
        executable=executable,
        skeleton=skeleton,
        atlas=atlas,
        expected_animation_names=expected_animation_names,
    )
    if report.skeleton_version != source_version:
        raise SpineRuntimeValidationError("runtime-reported skeleton version differs from source metadata")
    return report


__all__ = [
    "SPINE_RUNTIME_BREATH_MIN_WORLD_VERTEX_DELTA_RATIO",
    "SPINE_RUNTIME_EXPECTED_VERSION",
    "SPINE_RUNTIME_REPORT_SCHEMA_VERSION",
    "SPINE_RUNTIME_SAMPLE_RATE_HZ",
    "SPINE_RUNTIME_SETUP_RESTORE_TOLERANCE",
    "SPINE_RUNTIME_TALK_MIN_SLOT_ALPHA_DELTA",
    "SPINE_RUNTIME_TALK_MIN_WORLD_VERTEX_DELTA_RATIO",
    "SPINE_RUNTIME_VALIDATOR_PROTOCOL_DIGEST",
    "SPINE_RUNTIME_VALIDATOR_VERSION",
    "SpineRuntimeAnimationEvidence",
    "SpineRuntimeValidationError",
    "SpineRuntimeValidationReport",
    "validate_spine_runtime_bundle",
]
