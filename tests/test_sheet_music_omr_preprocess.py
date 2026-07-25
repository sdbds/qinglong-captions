import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _valid_preprocessor_payload() -> dict:
    return {
        "color": "RGB",
        "do_normalize": False,
        "do_rescale": True,
        "do_resize": True,
        "image_size": [1024, 1024],
        "input_layout": "NCHW",
        "interpolation": "bilinear",
        "rescale_factor": 1 / 255,
    }


def _write_payload(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_strict_preprocessor_config_accepts_published_contract(tmp_path: Path):
    from module.sheet_music_omr.preprocess import load_preprocessor_config

    config = load_preprocessor_config(
        _write_payload(tmp_path / "preprocessor_config.json", _valid_preprocessor_payload())
    )

    assert config.image_size == (1024, 1024)
    assert config.interpolation == "bilinear"
    assert config.rescale_factor == 1 / 255


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.pop("image_size"), "missing keys.*image_size"),
        (lambda payload: payload.__setitem__("size", [1024, 1024]), "unknown keys.*size"),
        (lambda payload: payload.__setitem__("interpolation", "bicubic"), "interpolation"),
        (lambda payload: payload.__setitem__("input_layout", "NHWC"), "input_layout"),
    ],
)
def test_strict_preprocessor_config_rejects_drift(tmp_path: Path, mutation, match: str):
    from module.sheet_music_omr.preprocess import load_preprocessor_config

    payload = _valid_preprocessor_payload()
    mutation(payload)

    with pytest.raises(ValueError, match=match):
        load_preprocessor_config(_write_payload(tmp_path / "preprocessor_config.json", payload))


def test_preprocess_matches_bilinear_float32_reference_exactly(tmp_path: Path):
    from module.sheet_music_omr.preprocess import (
        load_preprocessor_config,
        preprocess_pil_image,
    )

    config = load_preprocessor_config(
        _write_payload(tmp_path / "preprocessor_config.json", _valid_preprocessor_payload())
    )
    source = np.array(
        [
            [[0, 10, 255], [255, 30, 0], [20, 220, 80]],
            [[50, 60, 70], [80, 90, 100], [110, 120, 130]],
        ],
        dtype=np.uint8,
    )
    image = Image.fromarray(source, mode="RGB")

    actual = preprocess_pil_image(image, config)

    expected_image = image.convert("RGB").resize((1024, 1024), Image.Resampling.BILINEAR)
    expected = np.asarray(expected_image, dtype=np.float32)
    expected /= np.float32(255.0)
    expected = np.ascontiguousarray(expected.transpose(2, 0, 1)[None, ...], dtype=np.float32)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (1, 3, 1024, 1024)
    assert actual.dtype == np.float32
