import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import module.see_through.extracted.postprocess_core as postprocess_core


def _install_fake_vendor_modules(monkeypatch, *, fail: bool = False):
    calls: list[str | None] = []

    fake_utils = types.ModuleType("utils")
    fake_utils.__path__ = []
    fake_torchcv = types.ModuleType("utils.torchcv")
    fake_inference_utils = types.ModuleType("utils.inference_utils")

    def original_cluster_inpaint_part(*args, **kwargs):
        calls.append(kwargs.get("inpaint"))
        return {"ok": True}

    def further_extr(*args, **kwargs):
        fake_torchcv.cluster_inpaint_part(None, None, None)
        fake_inference_utils.cluster_inpaint_part(None, None, None)
        if fail:
            raise RuntimeError("postprocess boom")

    fake_torchcv.cluster_inpaint_part = original_cluster_inpaint_part
    fake_inference_utils.cluster_inpaint_part = original_cluster_inpaint_part
    fake_inference_utils.further_extr = further_extr

    monkeypatch.setattr(postprocess_core, "ensure_vendor_imports", lambda: None)
    monkeypatch.delitem(sys.modules, "utils", raising=False)
    monkeypatch.delitem(sys.modules, "utils.torchcv", raising=False)
    monkeypatch.delitem(sys.modules, "utils.inference_utils", raising=False)
    monkeypatch.setitem(sys.modules, "utils", fake_utils)
    monkeypatch.setitem(sys.modules, "utils.torchcv", fake_torchcv)
    monkeypatch.setitem(sys.modules, "utils.inference_utils", fake_inference_utils)

    return fake_torchcv, fake_inference_utils, original_cluster_inpaint_part, calls


def test_run_postprocess_core_restores_vendor_patch_after_success(monkeypatch, tmp_path):
    fake_torchcv, fake_inference_utils, original_cluster_inpaint_part, calls = _install_fake_vendor_modules(
        monkeypatch
    )

    output_dir = tmp_path / "outputs"
    result = postprocess_core.run_postprocess_core(
        source_path=tmp_path / "source.png",
        output_dir=output_dir,
        save_to_psd=False,
        tblr_split=False,
    )

    assert calls == ["cv2", "cv2"]
    assert fake_torchcv.cluster_inpaint_part is original_cluster_inpaint_part
    assert fake_inference_utils.cluster_inpaint_part is original_cluster_inpaint_part
    assert result["manifest"] == output_dir / "optimized" / "manifest.json"


def test_run_postprocess_core_restores_vendor_patch_after_failure(monkeypatch, tmp_path):
    fake_torchcv, fake_inference_utils, original_cluster_inpaint_part, _ = _install_fake_vendor_modules(
        monkeypatch,
        fail=True,
    )

    with pytest.raises(RuntimeError, match="postprocess boom"):
        postprocess_core.run_postprocess_core(
            source_path=tmp_path / "source.png",
            output_dir=tmp_path / "outputs",
            save_to_psd=False,
            tblr_split=False,
        )

    assert fake_torchcv.cluster_inpaint_part is original_cluster_inpaint_part
    assert fake_inference_utils.cluster_inpaint_part is original_cluster_inpaint_part
