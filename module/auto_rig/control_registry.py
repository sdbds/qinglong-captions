from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

from .jcs import jcs_sha256

CONTROL_REGISTRY_VERSION = "control-registry-v1"
CONTROL_REGISTRY_PLAN_VERSION = "control-registry-plan-v1"

_ID_RE = re.compile(r"^[a-z][a-z0-9_-]*/[a-z0-9][a-z0-9._-]{0,126}$")
_EXPORT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,62}$")
_UNITS = frozenset({"degree", "normalized"})
_FORBIDDEN_IMAGE_SIDE_ALIASES = frozenset(
    {"ParamEyeLOpen", "ParamEyeROpen", "ParamBrowLY", "ParamBrowRY"}
)


class ControlRegistryError(ValueError):
    """Raised when the packaged ControlRegistry is internally inconsistent."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _error(message: str) -> ControlRegistryError:
    return ControlRegistryError("invalid_control_registry", message)


@dataclass(frozen=True, slots=True)
class Live2DParameterBinding:
    parameter_id: str
    export_name: str
    standard_id: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "parameter_id": self.parameter_id,
            "export_name": self.export_name,
            "standard_id": self.standard_id,
        }


@dataclass(frozen=True, slots=True)
class ControlSpec:
    control_id: str
    minimum: float
    default: float
    maximum: float
    unit: str
    live2d: Live2DParameterBinding | None
    registry_sha256: str

    def registry_record(self) -> dict[str, object]:
        return {
            "control_id": self.control_id,
            "minimum": self.minimum,
            "default": self.default,
            "maximum": self.maximum,
            "unit": self.unit,
            "format_bindings": {
                "spine_4_2": {"control_id": self.control_id},
                "live2d_moc3_v4_00": (
                    None if self.live2d is None else self.live2d.to_dict()
                ),
            },
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.registry_record(), "registry_sha256": self.registry_sha256}


@dataclass(frozen=True, slots=True)
class ControlRegistryPlan:
    schema_version: str
    registry_version: str
    registry_sha256: str
    controls: tuple[ControlSpec, ...]
    plan_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "registry_version": self.registry_version,
            "registry_sha256": self.registry_sha256,
            "controls": [control.to_dict() for control in self.controls],
        }


def _live2d(
    parameter_id: str,
    export_name: str,
    *,
    standard: bool,
) -> Live2DParameterBinding:
    return Live2DParameterBinding(
        parameter_id=parameter_id,
        export_name=export_name,
        standard_id=export_name if standard else None,
    )


def _spec(
    control_id: str,
    minimum: float,
    default: float,
    maximum: float,
    unit: str,
    live2d: Live2DParameterBinding | None,
) -> ControlSpec:
    return ControlSpec(
        control_id=control_id,
        minimum=minimum,
        default=default,
        maximum=maximum,
        unit=unit,
        live2d=live2d,
        registry_sha256="",
    )


_CONTROL_ROWS = (
    _spec(
        "control/body_sway",
        -10.0,
        0.0,
        10.0,
        "degree",
        _live2d("parameter/body_angle_x", "ParamBodyAngleX", standard=True),
    ),
    _spec(
        "control/breath",
        0.0,
        0.0,
        1.0,
        "normalized",
        _live2d("parameter/breath", "ParamBreath", standard=True),
    ),
    _spec(
        "control/brow_y.xmax",
        -1.0,
        0.0,
        1.0,
        "normalized",
        _live2d("parameter/brow_y.xmax", "ParamBrowYXMax", standard=False),
    ),
    _spec(
        "control/brow_y.xmin",
        -1.0,
        0.0,
        1.0,
        "normalized",
        _live2d("parameter/brow_y.xmin", "ParamBrowYXMin", standard=False),
    ),
    _spec(
        "control/eye_open.xmax",
        0.0,
        1.0,
        1.0,
        "normalized",
        _live2d("parameter/eye_open.xmax", "ParamEyeOpenXMax", standard=False),
    ),
    _spec(
        "control/eye_open.xmin",
        0.0,
        1.0,
        1.0,
        "normalized",
        _live2d("parameter/eye_open.xmin", "ParamEyeOpenXMin", standard=False),
    ),
    _spec(
        "control/head_nod",
        -30.0,
        0.0,
        30.0,
        "degree",
        _live2d("parameter/angle_y", "ParamAngleY", standard=True),
    ),
    _spec(
        "control/head_shake",
        -30.0,
        0.0,
        30.0,
        "degree",
        _live2d("parameter/angle_x", "ParamAngleX", standard=True),
    ),
    _spec(
        "control/idle",
        -1.0,
        0.0,
        1.0,
        "normalized",
        _live2d("parameter/auto_idle", "ParamAutoIdle", standard=False),
    ),
    _spec(
        "control/mouth_form",
        -1.0,
        0.0,
        1.0,
        "normalized",
        _live2d("parameter/mouth_form", "ParamMouthForm", standard=True),
    ),
    _spec(
        "control/mouth_open",
        0.0,
        0.0,
        1.0,
        "normalized",
        _live2d("parameter/mouth_open_y", "ParamMouthOpenY", standard=True),
    ),
    _spec("control/wave_lift.xmax", 0.0, 0.0, 1.0, "normalized", None),
    _spec("control/wave_lift.xmin", 0.0, 0.0, 1.0, "normalized", None),
    _spec("control/wave_osc.xmax", -1.0, 0.0, 1.0, "normalized", None),
    _spec("control/wave_osc.xmin", -1.0, 0.0, 1.0, "normalized", None),
)


def _registry_digest(controls: tuple[ControlSpec, ...]) -> str:
    return jcs_sha256(
        {
            "registry_version": CONTROL_REGISTRY_VERSION,
            "controls": [control.registry_record() for control in controls],
        }
    )


def validate_control_registry_plan(
    plan: ControlRegistryPlan,
) -> ControlRegistryPlan:
    """Validate the complete, profile-independent control universe."""

    if not isinstance(plan, ControlRegistryPlan):
        raise _error("registry must use ControlRegistryPlan")
    if (
        plan.schema_version != CONTROL_REGISTRY_PLAN_VERSION
        or plan.registry_version != CONTROL_REGISTRY_VERSION
    ):
        raise _error("registry version is unsupported")
    if tuple(sorted(plan.controls, key=lambda item: item.control_id)) != plan.controls:
        raise _error("controls must be sorted by control_id")
    control_ids = tuple(control.control_id for control in plan.controls)
    if not control_ids or len(control_ids) != len(set(control_ids)):
        raise _error("control IDs must be non-empty and unique")
    parameter_ids: list[str] = []
    export_names: list[str] = []
    for control in plan.controls:
        if not _ID_RE.fullmatch(control.control_id):
            raise _error(f"invalid control ID: {control.control_id!r}")
        if control.unit not in _UNITS:
            raise _error(f"invalid unit for {control.control_id}")
        values = (control.minimum, control.default, control.maximum)
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise _error(f"non-finite domain for {control.control_id}")
        if not control.minimum < control.maximum:
            raise _error(f"minimum must be below maximum for {control.control_id}")
        if not control.minimum <= control.default <= control.maximum:
            raise _error(f"default lies outside the domain for {control.control_id}")
        binding = control.live2d
        if binding is None:
            if not control.control_id.startswith(("control/wave_lift.", "control/wave_osc.")):
                raise _error(f"unexpected null Live2D binding for {control.control_id}")
            continue
        if not _ID_RE.fullmatch(binding.parameter_id):
            raise _error(f"invalid parameter ID for {control.control_id}")
        if not _EXPORT_NAME_RE.fullmatch(binding.export_name):
            raise _error(f"invalid parameter export name for {control.control_id}")
        if binding.export_name in _FORBIDDEN_IMAGE_SIDE_ALIASES:
            raise _error("image-side controls cannot use anatomical L/R parameter IDs")
        if binding.standard_id is not None and binding.standard_id != binding.export_name:
            raise _error(f"standard parameter ID differs from export name for {control.control_id}")
        parameter_ids.append(binding.parameter_id)
        export_names.append(binding.export_name)
    if len(parameter_ids) != len(set(parameter_ids)):
        raise _error("Live2D parameter IDs must be globally unique")
    if len(export_names) != len(set(export_names)):
        raise _error("Live2D parameter export names must be globally unique")
    expected_registry_sha256 = _registry_digest(plan.controls)
    if plan.registry_sha256 != expected_registry_sha256:
        raise _error("registry digest mismatch")
    if any(control.registry_sha256 != plan.registry_sha256 for control in plan.controls):
        raise _error("materialized ControlSpec has a different registry digest")
    if plan.plan_sha256 != jcs_sha256(plan.semantic_payload()):
        raise _error("control registry plan digest mismatch")
    return plan


def build_control_registry_plan() -> ControlRegistryPlan:
    """Materialize the complete ControlRegistry v1 in canonical ID order."""

    ordered = tuple(sorted(_CONTROL_ROWS, key=lambda item: item.control_id))
    registry_sha256 = _registry_digest(ordered)
    controls = tuple(
        replace(control, registry_sha256=registry_sha256) for control in ordered
    )
    values = {
        "schema_version": CONTROL_REGISTRY_PLAN_VERSION,
        "registry_version": CONTROL_REGISTRY_VERSION,
        "registry_sha256": registry_sha256,
        "controls": controls,
    }
    provisional = ControlRegistryPlan(**values, plan_sha256="")
    plan = ControlRegistryPlan(
        **values,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )
    return validate_control_registry_plan(plan)


__all__ = [
    "CONTROL_REGISTRY_PLAN_VERSION",
    "CONTROL_REGISTRY_VERSION",
    "ControlRegistryError",
    "ControlRegistryPlan",
    "ControlSpec",
    "Live2DParameterBinding",
    "build_control_registry_plan",
    "validate_control_registry_plan",
]
