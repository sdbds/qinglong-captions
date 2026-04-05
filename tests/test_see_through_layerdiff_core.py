import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import module.see_through.extracted.layerdiff_core as layerdiff_core
from module.see_through.vendor_bootstrap import VENDOR_ROOT


VALID_BODY_PARTS_V2 = [
    "hair",
    "headwear",
    "face",
    "eyes",
    "eyewear",
    "ears",
    "earwear",
    "nose",
    "mouth",
    "neck",
    "neckwear",
    "topwear",
    "handwear",
    "bottomwear",
    "legwear",
    "footwear",
    "tail",
    "wings",
    "objects",
]


class _FakeOutput:
    def __init__(self, images):
        self.images = images


class _FakeUnet:
    def get_tag_version(self):
        return "v3"


class _FakePipeline:
    def __init__(self, body_images, head_images):
        self.unet = _FakeUnet()
        self._body_images = body_images
        self._head_images = head_images

    def __call__(self, **kwargs):
        group_index = kwargs.get("group_index", 0)
        if group_index == 0:
            return _FakeOutput(self._body_images)
        if group_index == 1:
            return _FakeOutput(self._head_images)
        raise AssertionError(f"unexpected group_index: {group_index}")


def _alpha_bbox(path: Path) -> tuple[int, int, int, int]:
    arr = np.array(Image.open(path).convert("RGBA"))
    ys, xs = np.where(arr[..., 3] > 15)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def test_run_layerdiff_phase_uses_fullpage_padding_for_head_anchor(monkeypatch, tmp_path):
    source_path = tmp_path / "source.png"
    Image.new("RGBA", (60, 100), (0, 0, 0, 0)).save(source_path)

    resolution = 100
    blank = np.zeros((resolution, resolution, 4), dtype=np.uint8)

    head_img = blank.copy()
    head_img[10:22, 45:55, :3] = 255
    head_img[10:22, 45:55, 3] = 255
    body_images = [blank.copy(), blank.copy(), head_img] + [blank.copy() for _ in range(10)]

    face_img = blank.copy()
    face_img[..., 0] = 255
    face_img[..., 3] = 255
    head_images = [blank.copy(), face_img] + [blank.copy() for _ in range(9)]

    fake_pipeline = _FakePipeline(body_images=body_images, head_images=head_images)

    fake_cv = types.ModuleType("utils.cv")
    fake_cv2 = types.ModuleType("cv2")
    fake_inference_utils = types.ModuleType("utils.inference_utils")
    fake_inference_utils.VALID_BODY_PARTS_V2 = VALID_BODY_PARTS_V2

    def smart_resize(array, size):
        target_height, target_width = int(size[0]), int(size[1])
        resized = Image.fromarray(np.asarray(array, dtype=np.uint8)).resize((target_width, target_height), Image.BILINEAR)
        return np.array(resized)

    def center_square_pad_resize(img, target_size, pad_value=0, return_pad_info=False, **kwargs):
        h, w = img.shape[:2]
        pad_size = (w, h)
        pad_pos = (0, 0)
        padded = img
        if h != w:
            sz = max(h, w)
            px1 = (sz - w) // 2
            py1 = (sz - h) // 2
            shape = (sz, sz) if img.ndim == 2 else (sz, sz, img.shape[-1])
            padded = np.full(shape, pad_value, dtype=img.dtype)
            padded[py1 : py1 + h, px1 : px1 + w] = img
            pad_size = (sz, sz)
            pad_pos = (px1, py1)
        if padded.shape[0] != target_size or padded.shape[1] != target_size:
            padded = smart_resize(padded, (target_size, target_size))
        if return_pad_info:
            return padded, pad_size, pad_pos
        return padded

    fake_cv.smart_resize = smart_resize
    fake_cv.center_square_pad_resize = center_square_pad_resize

    def find_non_zero(mask):
        ys, xs = np.where(np.asarray(mask) > 0)
        if len(xs) == 0:
            return None
        return np.array([[[int(x), int(y)]] for y, x in zip(ys, xs)], dtype=np.int32)

    def bounding_rect(points):
        coords = np.asarray(points).reshape(-1, 2)
        xs = coords[:, 0]
        ys = coords[:, 1]
        x0 = int(xs.min())
        y0 = int(ys.min())
        return x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1

    fake_cv2.findNonZero = find_non_zero
    fake_cv2.boundingRect = bounding_rect

    def _ensure_vendor_root_for_test():
        monkeypatch.syspath_prepend(str(VENDOR_ROOT))
        monkeypatch.delitem(sys.modules, "utils", raising=False)
        monkeypatch.delitem(sys.modules, "cv2", raising=False)
        monkeypatch.delitem(sys.modules, "utils.cv", raising=False)
        monkeypatch.delitem(sys.modules, "utils.inference_utils", raising=False)
        monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
        monkeypatch.setitem(sys.modules, "utils.cv", fake_cv)
        monkeypatch.setitem(sys.modules, "utils.inference_utils", fake_inference_utils)
        return VENDOR_ROOT

    monkeypatch.setattr(
        layerdiff_core,
        "ensure_vendor_imports",
        _ensure_vendor_root_for_test,
    )

    output_dir = tmp_path / "outputs"

    layerdiff_core.run_layerdiff_phase(
        source_path=source_path,
        output_dir=output_dir,
        pipeline=fake_pipeline,
        resolution=resolution,
        generator_device="cpu",
    )

    assert _alpha_bbox(output_dir / "face.png") == (43, 8, 57, 24)
