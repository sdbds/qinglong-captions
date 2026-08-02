from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ...jcs import jcs_sha256

RIGID_DRIVER_REGISTRY_VERSION = "rigid-driver-registry-v1"
RIGID_DRIVER_PRIMITIVE_KIND = "live2d_rotation_deformer"

_ROWS = (
    ("control/body_sway", 100),
    ("control/idle", 200),
    ("control/head_shake", 300),
    ("control/head_nod", 400),
)


class RigidDriverRegistryError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(f"invalid_rigid_driver_registry: {message}")


def _error(message: str) -> RigidDriverRegistryError:
    return RigidDriverRegistryError(message)


@dataclass(frozen=True, slots=True)
class RigidDriverRow:
    control_id: str
    parameter_id: str
    primitive_kind: str
    stack_rank: int
    row_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "control_id": self.control_id,
            "parameter_id": self.parameter_id,
            "primitive_kind": self.primitive_kind,
            "stack_rank": self.stack_rank,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "row_sha256": self.row_sha256}


@dataclass(frozen=True, slots=True)
class RigidDriverRegistry:
    schema_version: str
    rows: tuple[RigidDriverRow, ...]
    registry_sha256: str

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic_payload(), "registry_sha256": self.registry_sha256}


def _control_parameters(
    control_specs: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for index, control in enumerate(control_specs):
        if not isinstance(control, Mapping):
            raise _error(f"control_specs[{index}] is not an object")
        control_id = control.get("control_id")
        formats = control.get("format_bindings")
        if not isinstance(control_id, str) or not isinstance(formats, Mapping):
            raise _error(f"control_specs[{index}] has invalid identity")
        live2d = formats.get("live2d_moc3_v4_00")
        if live2d is None:
            continue
        if not isinstance(live2d, Mapping):
            raise _error(f"control_specs[{index}] has invalid Live2D binding")
        parameter_id = live2d.get("parameter_id")
        if not isinstance(parameter_id, str) or not parameter_id:
            raise _error(f"control_specs[{index}] has invalid parameter identity")
        if control_id in result:
            raise _error("control IDs are duplicated")
        result[control_id] = parameter_id
    return result


def build_rigid_driver_registry(
    control_specs: Sequence[Mapping[str, object]],
) -> RigidDriverRegistry:
    parameters = _control_parameters(control_specs)
    rows = []
    for control_id, rank in _ROWS:
        parameter_id = parameters.get(control_id)
        if parameter_id is None:
            raise _error(f"rigid control is absent: {control_id}")
        provisional = RigidDriverRow(
            control_id=control_id,
            parameter_id=parameter_id,
            primitive_kind=RIGID_DRIVER_PRIMITIVE_KIND,
            stack_rank=rank,
            row_sha256="",
        )
        rows.append(
            RigidDriverRow(
                control_id=provisional.control_id,
                parameter_id=provisional.parameter_id,
                primitive_kind=provisional.primitive_kind,
                stack_rank=provisional.stack_rank,
                row_sha256=jcs_sha256(provisional.semantic_payload()),
            )
        )
    provisional_registry = RigidDriverRegistry(
        schema_version=RIGID_DRIVER_REGISTRY_VERSION,
        rows=tuple(rows),
        registry_sha256="",
    )
    return validate_rigid_driver_registry(
        RigidDriverRegistry(
            schema_version=provisional_registry.schema_version,
            rows=provisional_registry.rows,
            registry_sha256=jcs_sha256(provisional_registry.semantic_payload()),
        ),
        control_specs,
    )


def validate_rigid_driver_registry(
    registry: RigidDriverRegistry,
    control_specs: Sequence[Mapping[str, object]],
) -> RigidDriverRegistry:
    if not isinstance(registry, RigidDriverRegistry):
        raise _error("registry has the wrong type")
    if registry.schema_version != RIGID_DRIVER_REGISTRY_VERSION:
        raise _error("registry version is unsupported")
    parameters = _control_parameters(control_specs)
    if tuple(row.control_id for row in registry.rows) != tuple(
        control_id for control_id, _rank in _ROWS
    ):
        raise _error("registry rows differ from the built-in control order")
    ranks = [row.stack_rank for row in registry.rows]
    if len(ranks) != len(set(ranks)):
        raise _error("rigid driver stack rank values must be globally unique")
    expected_ranks = dict(_ROWS)
    for row in registry.rows:
        if (
            row.parameter_id != parameters.get(row.control_id)
            or row.primitive_kind != RIGID_DRIVER_PRIMITIVE_KIND
            or row.stack_rank != expected_ranks[row.control_id]
            or row.row_sha256 != jcs_sha256(row.semantic_payload())
        ):
            raise _error("rigid driver row differs from its control/parameter contract")
    if registry.registry_sha256 != jcs_sha256(registry.semantic_payload()):
        raise _error("registry digest mismatch")
    return registry


__all__ = [
    "RIGID_DRIVER_PRIMITIVE_KIND",
    "RIGID_DRIVER_REGISTRY_VERSION",
    "RigidDriverRegistry",
    "RigidDriverRegistryError",
    "RigidDriverRow",
    "build_rigid_driver_registry",
    "validate_rigid_driver_registry",
]
