from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

E0_RUNTIME_ASSET_VERSION = "live2d-e0-runtime-assets-v1"
E0_ORIENTATION_TEXTURE_SIZE = (64, 64)
E0_BASE_PARAMETER_VALUES = {
    "ParamBreath": 0.25,
    "ParamOuter": 4.0,
    "ParamInner": 3.0,
}
E0_EXPECTED_PARAMETER_VALUES = {
    "ParamBreath": 0.75,
    "ParamOuter": 15.0,
    "ParamInner": -8.0,
}


def cubism_runtime_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize Cubism JSON with numeric terminators accepted by SDK Framework 5-r.5."""

    if not isinstance(payload, Mapping):
        raise TypeError("Cubism runtime JSON payload must be a mapping")
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def build_e0_motion_json_bytes() -> bytes:
    return cubism_runtime_json_bytes(
        {
            "Version": 3,
            "Meta": {
                "Duration": 1.0,
                "Fps": 30.0,
                "Loop": False,
                "AreBeziersRestricted": True,
                "CurveCount": 1,
                "TotalSegmentCount": 1,
                "TotalPointCount": 2,
                "UserDataCount": 0,
                "TotalUserDataSize": 0,
            },
            "Curves": [
                {
                    "Target": "Parameter",
                    "Id": "ParamOuter",
                    "Segments": [0.0, 4.0, 0, 1.0, 10.0],
                }
            ],
            "UserData": [],
        }
    )


def build_e0_expression_json_bytes() -> bytes:
    return cubism_runtime_json_bytes(
        {
            "Type": "Live2D Expression",
            "FadeInTime": 0.0,
            "FadeOutTime": 0.0,
            "Parameters": [
                {"Id": "ParamBreath", "Value": 0.5, "Blend": "Add"},
                {"Id": "ParamOuter", "Value": 1.5, "Blend": "Multiply"},
                {"Id": "ParamInner", "Value": -8.0, "Blend": "Overwrite"},
            ],
        }
    )


def build_e0_orientation_rgba() -> bytes:
    width, height = E0_ORIENTATION_TEXTURE_SIZE
    pixels = bytearray(width * height * 4)
    for y in range(height):
        for x in range(width):
            if x < width // 2 and y < height // 2:
                color = (255, 0, 0, 255)
            elif y < height // 2:
                color = (0, 255, 0, 255)
            elif x < width // 2:
                color = (0, 0, 255, 255)
            else:
                color = (255, 255, 0, 255)
            if 24 <= x <= 39 and 24 <= y <= 39:
                color = (255, 0, 255, 128)
            offset = (y * width + x) * 4
            pixels[offset : offset + 4] = bytes(color)
    return bytes(pixels)


def write_e0_orientation_texture(path: str | Path) -> None:
    output = Path(path)
    image = Image.frombytes("RGBA", E0_ORIENTATION_TEXTURE_SIZE, build_e0_orientation_rgba())
    image.save(output)


__all__ = [
    "E0_BASE_PARAMETER_VALUES",
    "E0_EXPECTED_PARAMETER_VALUES",
    "E0_ORIENTATION_TEXTURE_SIZE",
    "E0_RUNTIME_ASSET_VERSION",
    "build_e0_expression_json_bytes",
    "build_e0_motion_json_bytes",
    "build_e0_orientation_rgba",
    "cubism_runtime_json_bytes",
    "write_e0_orientation_texture",
]
