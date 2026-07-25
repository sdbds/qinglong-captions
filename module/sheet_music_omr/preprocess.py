"""Strict image preprocessing for the pinned MuSViT OMR model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

_PREPROCESSOR_KEYS = frozenset(
    {
        "color",
        "do_normalize",
        "do_rescale",
        "do_resize",
        "image_size",
        "input_layout",
        "interpolation",
        "rescale_factor",
    }
)
_EXPECTED_IMAGE_SIZE = (1024, 1024)
_EXPECTED_RESCALE_FACTOR = 1 / 255


@dataclass(frozen=True)
class MuSViTPreprocessorConfig:
    image_size: tuple[int, int]
    interpolation: str
    rescale_factor: float
    color: str = "RGB"
    input_layout: str = "NCHW"
    do_resize: bool = True
    do_rescale: bool = True
    do_normalize: bool = False


def _require_exact_keys(payload: Mapping[str, Any]) -> None:
    actual = set(payload)
    missing = sorted(_PREPROCESSOR_KEYS - actual)
    unknown = sorted(actual - _PREPROCESSOR_KEYS)
    if missing:
        raise ValueError(f"preprocessor config missing keys: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"preprocessor config has unknown keys: {', '.join(unknown)}")


def parse_preprocessor_config(payload: Mapping[str, Any]) -> MuSViTPreprocessorConfig:
    if not isinstance(payload, Mapping):
        raise TypeError("preprocessor config must be a JSON object")
    _require_exact_keys(payload)

    image_size = payload["image_size"]
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in image_size)
    ):
        raise ValueError("preprocessor image_size must be the integer list [1024, 1024]")
    parsed_size = (image_size[0], image_size[1])
    if parsed_size != _EXPECTED_IMAGE_SIZE:
        raise ValueError(
            f"preprocessor image_size must be {_EXPECTED_IMAGE_SIZE!r}, got {parsed_size!r}"
        )

    exact_values = {
        "color": "RGB",
        "input_layout": "NCHW",
        "interpolation": "bilinear",
        "do_resize": True,
        "do_rescale": True,
        "do_normalize": False,
    }
    for key, expected in exact_values.items():
        if payload[key] != expected or type(payload[key]) is not type(expected):
            raise ValueError(
                f"preprocessor {key} must be {expected!r}, got {payload[key]!r}"
            )

    raw_factor = payload["rescale_factor"]
    if isinstance(raw_factor, bool) or not isinstance(raw_factor, (int, float)):
        raise ValueError("preprocessor rescale_factor must be numeric")
    factor = float(raw_factor)
    if factor != _EXPECTED_RESCALE_FACTOR:
        raise ValueError(
            "preprocessor rescale_factor must be "
            f"{_EXPECTED_RESCALE_FACTOR!r}, got {factor!r}"
        )

    return MuSViTPreprocessorConfig(
        image_size=parsed_size,
        interpolation="bilinear",
        rescale_factor=factor,
    )


def load_preprocessor_config(path: str | Path) -> MuSViTPreprocessorConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"preprocessor config must contain a JSON object: {config_path}")
    return parse_preprocessor_config(payload)


def preprocess_pil_image(
    image: Image.Image,
    config: MuSViTPreprocessorConfig,
) -> np.ndarray:
    height, width = config.image_size
    prepared = image.convert(config.color)
    prepared = prepared.resize((width, height), Image.Resampling.BILINEAR)
    pixels = np.asarray(prepared, dtype=np.float32)
    pixels /= np.float32(255.0)
    pixels = pixels.transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(pixels, dtype=np.float32)


def preprocess_image(
    image_path: str | Path,
    config: MuSViTPreprocessorConfig,
) -> np.ndarray:
    with Image.open(Path(image_path)) as image:
        return preprocess_pil_image(image, config)
