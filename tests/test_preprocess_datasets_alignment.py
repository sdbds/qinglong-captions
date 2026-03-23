import sys
import types
from pathlib import Path

from PIL import Image

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

fake_cv2 = types.SimpleNamespace(
    cuda=types.SimpleNamespace(getCudaEnabledDeviceCount=lambda: 0),
)
sys.modules.setdefault("cv2", fake_cv2)

import utils.preprocess_datasets as preprocess_datasets


def test_choose_matcher_backend_auto_prefers_affine_steerers_on_cuda(monkeypatch):
    monkeypatch.setattr(preprocess_datasets.cv2.cuda, "getCudaEnabledDeviceCount", lambda: 1)

    assert preprocess_datasets._choose_matcher_backend("auto") == "affine-steerers"


def test_choose_matcher_backend_auto_prefers_xfeat_without_cuda(monkeypatch):
    monkeypatch.setattr(preprocess_datasets.cv2.cuda, "getCudaEnabledDeviceCount", lambda: 0)

    assert preprocess_datasets._choose_matcher_backend("auto") == "xfeat"


def test_choose_matcher_backend_honors_explicit_backend(monkeypatch):
    monkeypatch.setattr(preprocess_datasets.cv2.cuda, "getCudaEnabledDeviceCount", lambda: 1)

    assert preprocess_datasets._choose_matcher_backend("orb") == "orb"


def test_transform_type_affine_never_escalates_to_homography():
    assert preprocess_datasets._transform_candidates("affine") == ("affine_partial", "affine")


def test_transform_type_auto_tries_simpler_models_before_homography():
    assert preprocess_datasets._transform_candidates("auto") == (
        "affine_partial",
        "affine",
        "homography",
    )


def test_load_matcher_falls_back_to_orb_when_vismatch_is_unavailable(monkeypatch):
    def _raise_import_error(*_args, **_kwargs):
        raise ImportError("vismatch not installed")

    monkeypatch.setattr(preprocess_datasets, "_load_vismatch_matcher", _raise_import_error)
    monkeypatch.setattr(preprocess_datasets.cv2.cuda, "getCudaEnabledDeviceCount", lambda: 0)

    matcher = preprocess_datasets._build_matcher("auto")

    assert matcher["kind"] == "orb"
    assert matcher["name"] == "orb"


def test_estimate_transform_from_points_auto_tries_simpler_models_first(monkeypatch):
    calls = []

    def _partial(*_args, **_kwargs):
        calls.append("affine_partial")
        return None, None

    def _affine(*_args, **_kwargs):
        calls.append("affine")
        return None, None

    def _homography(*_args, **_kwargs):
        calls.append("homography")
        return object(), None

    monkeypatch.setattr(preprocess_datasets.cv2, "estimateAffinePartial2D", _partial, raising=False)
    monkeypatch.setattr(preprocess_datasets.cv2, "estimateAffine2D", _affine, raising=False)
    monkeypatch.setattr(preprocess_datasets.cv2, "findHomography", _homography, raising=False)

    src_pts = preprocess_datasets.np.zeros((4, 1, 2), dtype=preprocess_datasets.np.float32)
    dst_pts = preprocess_datasets.np.zeros((4, 1, 2), dtype=preprocess_datasets.np.float32)

    matrix, transform_name = preprocess_datasets._estimate_transform_from_points(src_pts, dst_pts, "auto")

    assert calls == ["affine_partial", "affine", "homography"]
    assert transform_name == "homography"
    assert matrix is not None


def test_estimate_transform_from_points_affine_never_calls_homography(monkeypatch):
    calls = []

    def _partial(*_args, **_kwargs):
        calls.append("affine_partial")
        return object(), None

    def _affine(*_args, **_kwargs):
        calls.append("affine")
        return object(), None

    def _homography(*_args, **_kwargs):
        calls.append("homography")
        return object(), None

    monkeypatch.setattr(preprocess_datasets.cv2, "estimateAffinePartial2D", _partial, raising=False)
    monkeypatch.setattr(preprocess_datasets.cv2, "estimateAffine2D", _affine, raising=False)
    monkeypatch.setattr(preprocess_datasets.cv2, "findHomography", _homography, raising=False)

    src_pts = preprocess_datasets.np.zeros((4, 1, 2), dtype=preprocess_datasets.np.float32)
    dst_pts = preprocess_datasets.np.zeros((4, 1, 2), dtype=preprocess_datasets.np.float32)

    _matrix, transform_name = preprocess_datasets._estimate_transform_from_points(src_pts, dst_pts, "affine")

    assert calls == ["affine_partial"]
    assert transform_name == "affine_partial"


def test_image_processor_stores_matcher_backend():
    processor = preprocess_datasets.ImageProcessor(matcher_backend="orb")

    assert processor.matcher_backend == "orb"


def test_match_points_with_backend_uses_vismatch_keypoints(monkeypatch):
    class FakeMatcher:
        def __call__(self, img0, img1):
            assert isinstance(img0, Image.Image)
            assert isinstance(img1, Image.Image)
            assert img0.mode == "RGB"
            assert img1.mode == "RGB"
            return {
                "matched_kpts0": [[1.0, 2.0], [3.0, 4.0]],
                "matched_kpts1": [[5.0, 6.0], [7.0, 8.0]],
            }

    monkeypatch.setattr(
        preprocess_datasets,
        "_build_matcher",
        lambda _preferred: {"kind": "vismatch", "name": "xfeat", "matcher": FakeMatcher()},
    )

    src_pts, dst_pts, backend_name = preprocess_datasets._match_points_with_backend(
        preprocess_datasets.np.zeros((8, 8, 3), dtype=preprocess_datasets.np.uint8),
        preprocess_datasets.np.zeros((8, 8, 3), dtype=preprocess_datasets.np.uint8),
        "auto",
    )

    assert backend_name == "xfeat"
    assert src_pts.shape == (2, 1, 2)
    assert dst_pts.shape == (2, 1, 2)


def test_pyproject_declares_image_align_extra():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    optional_deps = pyproject["project"]["optional-dependencies"]

    assert "image-align" in optional_deps
    assert "vismatch" in optional_deps["image-align"]


def test_preprocess_wrapper_exposes_matcher_backend():
    content = (ROOT / "2.2.preprocess_images.ps1").read_text(encoding="utf-8")

    assert 'matcher_backend' in content
    assert '--matcher-backend=$($Config.matcher_backend)' in content
    assert 'Install-UvExtraPatch @("image-align")' in content
    assert 'runtime dependency profile: extra:image-align' in content
