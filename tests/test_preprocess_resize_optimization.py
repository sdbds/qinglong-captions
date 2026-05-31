import io
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

fake_cv2 = types.SimpleNamespace(
    cuda=types.SimpleNamespace(getCudaEnabledDeviceCount=lambda: 0),
    INTER_AREA=3,
    COLOR_BGR2RGB=1,
    COLOR_GRAY2BGR=2,
)
sys.modules.setdefault("cv2", fake_cv2)

import utils.preprocess_datasets as preprocess_datasets


def _quiet_console():
    return Console(file=io.StringIO(), force_terminal=False, color_system=None)


def _patch_cpu_cv2(monkeypatch):
    calls = {"resize": 0, "cvtColor": 0}

    def resize(array, size, interpolation=None):
        calls["resize"] += 1
        return np.asarray(Image.fromarray(array).resize(size, resample=Image.Resampling.NEAREST))

    def cvt_color(array, code):
        calls["cvtColor"] += 1
        if code == fake_cv2.COLOR_BGR2RGB:
            return array[..., ::-1]
        if code == fake_cv2.COLOR_GRAY2BGR:
            return np.stack([array, array, array], axis=-1)
        raise AssertionError(f"unexpected cvtColor code: {code}")

    patched_cv2 = types.SimpleNamespace(
        cuda=types.SimpleNamespace(getCudaEnabledDeviceCount=lambda: 0),
        INTER_AREA=fake_cv2.INTER_AREA,
        COLOR_BGR2RGB=fake_cv2.COLOR_BGR2RGB,
        COLOR_GRAY2BGR=fake_cv2.COLOR_GRAY2BGR,
        resize=resize,
        cvtColor=cvt_color,
    )
    monkeypatch.setattr(preprocess_datasets, "cv2", patched_cv2)
    return calls


def _processor(max_workers=1, **kwargs):
    return preprocess_datasets.ImageProcessor(
        max_workers=max_workers,
        console=_quiet_console(),
        **kwargs,
    )


def test_rgb_jpeg_that_already_fits_skips_before_resize(monkeypatch, tmp_path):
    calls = _patch_cpu_cv2(monkeypatch)
    image_path = tmp_path / "fits.jpg"
    Image.new("RGB", (64, 64), (255, 0, 0)).save(image_path)

    result = _processor().resize_image(str(image_path), max_long_edge=2048)

    assert result.ok is True
    assert result.skipped is True
    assert calls["resize"] == 0
    assert calls["cvtColor"] == 0


def test_rgb_image_resizes_to_calculated_dimensions(monkeypatch, tmp_path):
    calls = _patch_cpu_cv2(monkeypatch)
    image_path = tmp_path / "large.png"
    Image.new("RGB", (80, 40), (10, 20, 30)).save(image_path)

    result = _processor().resize_image(str(image_path), max_long_edge=32)

    assert result.ok is True
    assert result.resized is True
    assert calls["resize"] == 1
    with Image.open(image_path) as image:
        assert image.size == preprocess_datasets.calculate_dimensions(80, 40, max_long_edge=32)


def test_palette_png_that_fits_converts_mode_without_early_skip(monkeypatch, tmp_path):
    calls = _patch_cpu_cv2(monkeypatch)
    image_path = tmp_path / "palette.png"
    image = Image.new("P", (64, 64))
    image.putpalette([0, 0, 0, 255, 0, 0] + [0, 0, 0] * 254)
    image.save(image_path)

    result = _processor().resize_image(str(image_path), max_long_edge=2048)

    assert result.ok is True
    assert result.skipped is False
    assert result.converted is True
    assert calls["resize"] == 0
    with Image.open(image_path) as image:
        assert image.mode == "RGB"
        assert image.size == (64, 64)


def test_rgba_crop_transparent_crops_and_does_not_early_skip(monkeypatch, tmp_path):
    calls = _patch_cpu_cv2(monkeypatch)
    image_path = tmp_path / "transparent.png"
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    image.paste((255, 0, 0, 255), (16, 16, 32, 32))
    image.save(image_path)

    result = _processor(crop_transparent=True).resize_image(str(image_path), max_long_edge=2048)

    assert result.ok is True
    assert result.skipped is False
    assert result.cropped is True
    assert result.converted is True
    assert calls["resize"] == 0
    with Image.open(image_path) as image:
        assert image.mode == "RGB"
        assert image.size == (16, 16)


def test_short_edge_and_pixel_limits_use_strictest_dimension(monkeypatch, tmp_path):
    calls = _patch_cpu_cv2(monkeypatch)
    image_path = tmp_path / "limited.png"
    Image.new("RGB", (300, 200), (1, 2, 3)).save(image_path)

    result = _processor().resize_image(
        str(image_path),
        max_short_edge=80,
        max_pixels=6400,
    )

    assert result.ok is True
    assert result.resized is True
    assert calls["resize"] == 1
    with Image.open(image_path) as image:
        assert image.size == preprocess_datasets.calculate_dimensions(
            300,
            200,
            max_short_edge=80,
            max_pixels=6400,
            max_long_edge=None,
        )


def test_cpu_resize_rgb_fast_path_preserves_channel_order(monkeypatch, tmp_path):
    _patch_cpu_cv2(monkeypatch)
    image_path = tmp_path / "channels.png"
    image = Image.new("RGB", (64, 64))
    image.paste((255, 0, 0), (0, 0, 32, 32))
    image.paste((0, 255, 0), (32, 0, 64, 32))
    image.paste((0, 0, 255), (0, 32, 32, 64))
    image.save(image_path)

    result = _processor().resize_image(str(image_path), max_long_edge=32)

    assert result.ok is True
    with Image.open(image_path) as resized:
        assert resized.size == (32, 32)
        assert resized.getpixel((8, 8)) == (255, 0, 0)
        assert resized.getpixel((24, 8)) == (0, 255, 0)
        assert resized.getpixel((8, 24)) == (0, 0, 255)


def test_process_directory_keeps_success_count_and_collects_summary(monkeypatch, tmp_path):
    _patch_cpu_cv2(monkeypatch)
    Image.new("RGB", (32, 32), (1, 2, 3)).save(tmp_path / "skip.jpg")
    Image.new("RGB", (80, 40), (4, 5, 6)).save(tmp_path / "resize.png")
    (tmp_path / "broken.jpg").write_text("not an image", encoding="utf-8")

    captured_summaries = []
    processor = _processor(max_workers=1)
    monkeypatch.setattr(processor, "_print_resize_summary", captured_summaries.append)

    successful, total = processor.process_directory(str(tmp_path), max_long_edge=32)

    assert (successful, total) == (2, 3)
    assert len(captured_summaries) == 1
    summary = captured_summaries[0]
    assert summary.skipped == 1
    assert summary.resized == 1
    assert summary.failed == 1
