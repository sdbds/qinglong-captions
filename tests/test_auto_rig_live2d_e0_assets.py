from __future__ import annotations

import json

from module.auto_rig.export.live2d.e0_assets import (
    build_e0_expression_json_bytes,
    build_e0_motion_json_bytes,
    cubism_runtime_json_bytes,
)


def test_cubism_runtime_json_uses_framework_compatible_numeric_terminators() -> None:
    encoded = cubism_runtime_json_bytes({"Values": [0.0, 10.0], "Version": 3})

    assert json.loads(encoded) == {"Values": [0.0, 10.0], "Version": 3}
    assert b"10.0\n  ]" in encoded
    assert b"3\n}" in encoded
    assert encoded.endswith(b"\n")


def test_e0_motion_and_expression_assets_cover_the_frozen_parameter_order() -> None:
    motion = json.loads(build_e0_motion_json_bytes())
    expression = json.loads(build_e0_expression_json_bytes())

    assert motion["Curves"][0]["Id"] == "ParamOuter"
    assert motion["Meta"]["FadeInTime"] == 0.0
    assert motion["Meta"]["FadeOutTime"] == 0.0
    assert [parameter["Blend"] for parameter in expression["Parameters"]] == [
        "Add",
        "Multiply",
        "Overwrite",
    ]
    assert expression["FadeInTime"] == 0.0
    assert expression["FadeOutTime"] == 0.0
