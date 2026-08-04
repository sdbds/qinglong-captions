from __future__ import annotations

from dataclasses import replace

import pytest

from module.auto_rig.control_registry import (
    CONTROL_REGISTRY_PLAN_VERSION,
    CONTROL_REGISTRY_VERSION,
    ControlRegistryError,
    build_control_registry_plan,
    validate_control_registry_plan,
)
from module.auto_rig.jcs import jcs_sha256

EXPECTED_CONTROLS = (
    (
        "control/arm_sway",
        "parameter/arm_sway",
        "ParamArmSway",
        -1.0,
        0.0,
        1.0,
        "normalized",
    ),
    (
        "control/body_sway",
        "parameter/body_angle_x",
        "ParamBodyAngleX",
        -10.0,
        0.0,
        10.0,
        "degree",
    ),
    (
        "control/breath",
        "parameter/breath",
        "ParamBreath",
        0.0,
        0.0,
        1.0,
        "normalized",
    ),
    (
        "control/brow_y.xmax",
        "parameter/brow_y.xmax",
        "ParamBrowYXMax",
        -1.0,
        0.0,
        1.0,
        "normalized",
    ),
    (
        "control/brow_y.xmin",
        "parameter/brow_y.xmin",
        "ParamBrowYXMin",
        -1.0,
        0.0,
        1.0,
        "normalized",
    ),
    (
        "control/eye_open.xmax",
        "parameter/eye_open.xmax",
        "ParamEyeOpenXMax",
        0.0,
        1.0,
        1.0,
        "normalized",
    ),
    (
        "control/eye_open.xmin",
        "parameter/eye_open.xmin",
        "ParamEyeOpenXMin",
        0.0,
        1.0,
        1.0,
        "normalized",
    ),
    (
        "control/head_nod",
        "parameter/angle_y",
        "ParamAngleY",
        -30.0,
        0.0,
        30.0,
        "degree",
    ),
    (
        "control/head_shake",
        "parameter/angle_x",
        "ParamAngleX",
        -30.0,
        0.0,
        30.0,
        "degree",
    ),
    (
        "control/idle",
        "parameter/auto_idle",
        "ParamAutoIdle",
        -1.0,
        0.0,
        1.0,
        "normalized",
    ),
    (
        "control/leg_sway",
        "parameter/leg_sway",
        "ParamLegSway",
        -1.0,
        0.0,
        1.0,
        "normalized",
    ),
    (
        "control/mouth_form",
        "parameter/mouth_form",
        "ParamMouthForm",
        -1.0,
        0.0,
        1.0,
        "normalized",
    ),
    (
        "control/mouth_open",
        "parameter/mouth_open_y",
        "ParamMouthOpenY",
        0.0,
        0.0,
        1.0,
        "normalized",
    ),
    ("control/wave_lift.xmax", None, None, 0.0, 0.0, 1.0, "normalized"),
    ("control/wave_lift.xmin", None, None, 0.0, 0.0, 1.0, "normalized"),
    ("control/wave_osc.xmax", None, None, -1.0, 0.0, 1.0, "normalized"),
    ("control/wave_osc.xmin", None, None, -1.0, 0.0, 1.0, "normalized"),
)


def _rehash(plan, **changes):
    provisional = replace(plan, **changes, plan_sha256="")
    return replace(
        provisional,
        plan_sha256=jcs_sha256(provisional.semantic_payload()),
    )


def test_control_registry_freezes_exact_domains_and_parameter_names() -> None:
    plan = build_control_registry_plan()

    assert plan.schema_version == CONTROL_REGISTRY_PLAN_VERSION
    assert plan.registry_version == CONTROL_REGISTRY_VERSION
    assert validate_control_registry_plan(plan) is plan
    assert (
        tuple(
            (
                spec.control_id,
                None if spec.live2d is None else spec.live2d.parameter_id,
                None if spec.live2d is None else spec.live2d.export_name,
                spec.minimum,
                spec.default,
                spec.maximum,
                spec.unit,
            )
            for spec in plan.controls
        )
        == EXPECTED_CONTROLS
    )
    assert all(spec.registry_sha256 == plan.registry_sha256 for spec in plan.controls)


def test_control_registry_is_byte_stable_and_does_not_alias_image_sides_to_lr() -> None:
    first = build_control_registry_plan()
    second = build_control_registry_plan()

    assert first == second
    assert first.plan_sha256 == second.plan_sha256
    exported = {spec.live2d.export_name for spec in first.controls if spec.live2d is not None}
    assert "ParamEyeLOpen" not in exported
    assert "ParamEyeROpen" not in exported
    assert "ParamBrowLY" not in exported
    assert "ParamBrowRY" not in exported


@pytest.mark.parametrize(
    "mutation",
    (
        "duplicate_control",
        "duplicate_parameter",
        "invalid_domain",
        "non_finite",
        "illegal_side_alias",
    ),
)
def test_control_registry_rejects_rehashed_semantic_mutations(mutation: str) -> None:
    plan = build_control_registry_plan()
    controls = list(plan.controls)
    if mutation == "duplicate_control":
        controls[1] = replace(controls[1], control_id=controls[0].control_id)
    elif mutation == "duplicate_parameter":
        controls[1] = replace(controls[1], live2d=controls[0].live2d)
    elif mutation == "invalid_domain":
        controls[1] = replace(controls[1], minimum=2.0, maximum=1.0)
    elif mutation == "non_finite":
        controls[1] = replace(controls[1], default=float("inf"))
    else:
        eye_index = next(index for index, spec in enumerate(controls) if spec.control_id == "control/eye_open.xmin")
        controls[eye_index] = replace(
            controls[eye_index],
            live2d=replace(controls[eye_index].live2d, export_name="ParamEyeLOpen"),
        )

    tampered = replace(plan, controls=tuple(controls)) if mutation == "non_finite" else _rehash(plan, controls=tuple(controls))
    with pytest.raises(ControlRegistryError) as exc_info:
        validate_control_registry_plan(tampered)

    assert exc_info.value.code == "invalid_control_registry"
