import json
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

fake_transformer_loader = types.ModuleType("utils.transformer_loader")
fake_transformer_loader.load_pretrained_component = lambda *args, **kwargs: None
sys.modules.setdefault("utils.transformer_loader", fake_transformer_loader)

import module.see_through.extracted.marigold_core as marigold_core
from module.see_through.vendor_bootstrap import VENDOR_ROOT


VALID_BODY_PARTS_V2 = ["face", "head"]


class _FakeTensor:
    def __init__(self, array):
        self._array = np.asarray(array, dtype=np.float32)

    def to(self, device=None, dtype=None):
        return self

    def numpy(self):
        return self._array


class _FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        img_list = kwargs["img_list"]
        height, width = img_list[0].shape[:2]
        depth_tensor = np.linspace(0.1, 0.9, len(img_list), dtype=np.float32).reshape(len(img_list), 1, 1)
        depth_tensor = np.broadcast_to(depth_tensor, (len(img_list), height, width)).copy()
        return types.SimpleNamespace(depth_tensor=_FakeTensor(depth_tensor))


def _install_fake_vendor_imports(monkeypatch, seeded):
    fake_cv = types.ModuleType("utils.cv")
    fake_inference_utils = types.ModuleType("utils.inference_utils")
    fake_io_utils = types.ModuleType("utils.io_utils")
    fake_inference_utils.VALID_BODY_PARTS_V2 = VALID_BODY_PARTS_V2

    fake_torch_utils = types.ModuleType("utils.torch_utils")
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"

    def seed_everything(seed):
        seeded.append(seed)

    def validate_resolution(resolution):
        if isinstance(resolution, int):
            return [resolution, resolution]
        if isinstance(resolution, str):
            return [int(part.strip()) for part in resolution.split(",")[:2]]
        return [int(resolution[0]), int(resolution[1])]

    def smart_resize(array, size):
        target_height, target_width = int(size[0]), int(size[1])
        if array.ndim == 2:
            resized = Image.fromarray(np.asarray(array, dtype=np.float32), mode="F").resize(
                (target_width, target_height),
                Image.BILINEAR,
            )
            return np.array(resized, dtype=np.float32)
        resized = Image.fromarray(np.asarray(array, dtype=np.uint8)).resize((target_width, target_height), Image.BILINEAR)
        return np.array(resized)

    def img_alpha_blending(items, premultiplied=False):
        arrays = [item["img"] for item in items] if items and isinstance(items[0], dict) else items
        canvas = np.zeros_like(arrays[0], dtype=np.float32)
        for array in arrays:
            alpha = array[..., 3:4].astype(np.float32) / 255.0
            canvas[..., :3] = canvas[..., :3] * (1.0 - alpha) + array[..., :3].astype(np.float32) * alpha
            canvas[..., 3:4] = np.clip(canvas[..., 3:4] + array[..., 3:4].astype(np.float32), 0.0, 255.0)
        return canvas.astype(np.uint8)

    def dict2json(payload, path):
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def json2dict(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    fake_cv.validate_resolution = validate_resolution
    fake_cv.smart_resize = smart_resize
    fake_cv.img_alpha_blending = img_alpha_blending
    fake_io_utils.dict2json = dict2json
    fake_io_utils.json2dict = json2dict
    fake_torch_utils.seed_everything = seed_everything

    def _ensure_vendor_root_for_test():
        monkeypatch.syspath_prepend(str(VENDOR_ROOT))
        monkeypatch.delitem(sys.modules, "utils", raising=False)
        monkeypatch.delitem(sys.modules, "utils.cv", raising=False)
        monkeypatch.delitem(sys.modules, "utils.io_utils", raising=False)
        monkeypatch.delitem(sys.modules, "utils.inference_utils", raising=False)
        monkeypatch.delitem(sys.modules, "utils.torch_utils", raising=False)
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        monkeypatch.setitem(sys.modules, "utils.cv", fake_cv)
        monkeypatch.setitem(sys.modules, "utils.inference_utils", fake_inference_utils)
        monkeypatch.setitem(sys.modules, "utils.io_utils", fake_io_utils)
        monkeypatch.setitem(sys.modules, "utils.torch_utils", fake_torch_utils)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        return VENDOR_ROOT

    monkeypatch.setattr(marigold_core, "ensure_vendor_imports", _ensure_vendor_root_for_test)


def test_run_marigold_phase_uses_src_img_canvas_and_default_depth_steps(monkeypatch, tmp_path):
    seeded = []
    _install_fake_vendor_imports(monkeypatch, seeded)

    source_path = tmp_path / "source.png"
    Image.new("RGBA", (16, 32), (0, 255, 0, 255)).save(source_path)

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    Image.new("RGBA", (64, 64), (255, 0, 0, 255)).save(output_dir / "src_img.png")

    pipeline = _FakePipeline()

    marigold_core.run_marigold_phase(
        source_path=source_path,
        output_dir=output_dir,
        pipeline=pipeline,
        resolution_depth=-1,
        inference_steps_depth=-1,
        seed=777,
    )

    assert seeded == [777]
    assert len(pipeline.calls) == 1
    call = pipeline.calls[0]
    assert "denoising_steps" not in call
    assert all(image.shape[:2] == (64, 64) for image in call["img_list"])
    assert tuple(call["img_list"][-1][0, 0, :3]) == (255, 0, 0)

    manifest = json.loads((output_dir / "depth" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["resolution"] == 64
    assert manifest["resolution_depth"] == 64


def test_run_marigold_phase_resizes_to_requested_depth_resolution(monkeypatch, tmp_path):
    seeded = []
    _install_fake_vendor_imports(monkeypatch, seeded)

    source_path = tmp_path / "source.png"
    Image.new("RGBA", (32, 32), (0, 255, 0, 255)).save(source_path)

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    Image.new("RGBA", (64, 64), (255, 0, 0, 255)).save(output_dir / "src_img.png")

    pipeline = _FakePipeline()

    marigold_core.run_marigold_phase(
        source_path=source_path,
        output_dir=output_dir,
        pipeline=pipeline,
        resolution_depth=32,
        inference_steps_depth=6,
        seed=5,
    )

    assert seeded == [5]
    assert len(pipeline.calls) == 1
    call = pipeline.calls[0]
    assert call["denoising_steps"] == 6
    assert all(image.shape[:2] == (32, 32) for image in call["img_list"])
    assert Image.open(output_dir / "depth" / "depth.png").size == (64, 64)
    assert Image.open(output_dir / "face_depth.png").size == (64, 64)

    manifest = json.loads((output_dir / "depth" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["resolution"] == 64
    assert manifest["resolution_depth"] == 32
