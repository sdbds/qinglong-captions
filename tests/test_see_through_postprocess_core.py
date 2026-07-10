import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

import module.see_through.extracted.postprocess_core as postprocess_core


VENDOR_PREFIX = "module.see_through.vendor"
VENDOR_DIR = ROOT / "module" / "see_through" / "vendor"


def _vendor_backed_top_level_utils() -> set[str]:
    aliases: set[str] = set()
    vendor_dir = VENDOR_DIR.resolve()
    for name, loaded_module in sys.modules.items():
        if name != "utils" and not name.startswith("utils."):
            continue
        module_file = getattr(loaded_module, "__file__", None)
        if module_file and Path(module_file).resolve().is_relative_to(vendor_dir):
            aliases.add(name)
    return aliases


def _install_fake_vendor_modules(monkeypatch, *, fail: bool = False):
    calls: list[str | None] = []

    fake_torchcv = types.ModuleType(f"{VENDOR_PREFIX}.utils.torchcv")
    fake_inference_utils = types.ModuleType(f"{VENDOR_PREFIX}.utils.inference_utils")

    def original_cluster_inpaint_part(*args, **kwargs):
        calls.append(kwargs.get("inpaint"))
        return {"ok": True}

    def further_extr(*args, **kwargs):
        calls.append(kwargs.get("inpaint"))
        if fail:
            raise RuntimeError("postprocess boom")

    fake_torchcv.cluster_inpaint_part = original_cluster_inpaint_part
    fake_inference_utils.cluster_inpaint_part = original_cluster_inpaint_part
    fake_inference_utils.further_extr = further_extr

    vendor_utils_package = importlib.import_module(f"{VENDOR_PREFIX}.utils")
    monkeypatch.setitem(sys.modules, f"{VENDOR_PREFIX}.utils.torchcv", fake_torchcv)
    monkeypatch.setitem(sys.modules, f"{VENDOR_PREFIX}.utils.inference_utils", fake_inference_utils)
    monkeypatch.setattr(vendor_utils_package, "torchcv", fake_torchcv, raising=False)
    monkeypatch.setattr(vendor_utils_package, "inference_utils", fake_inference_utils, raising=False)

    return fake_torchcv, fake_inference_utils, original_cluster_inpaint_part, calls


def test_run_postprocess_core_passes_inpaint_mode_without_mutating_vendor_modules(monkeypatch, tmp_path):
    path_safety_before = importlib.import_module("utils.path_safety")
    module_names_before = set(sys.modules)
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

    assert calls == ["cv2"]
    assert fake_torchcv.cluster_inpaint_part is original_cluster_inpaint_part
    assert fake_inference_utils.cluster_inpaint_part is original_cluster_inpaint_part
    assert result["manifest"] == output_dir / "optimized" / "manifest.json"
    assert importlib.import_module("utils.path_safety") is path_safety_before
    created_conflicts = {
        name
        for name in set(sys.modules) - module_names_before
        if name == "modules"
        or name.startswith("modules.")
        or name == "annotators"
        or name.startswith("annotators.")
    }
    assert not created_conflicts
    assert not _vendor_backed_top_level_utils()


def test_run_postprocess_core_leaves_vendor_modules_unchanged_after_failure(monkeypatch, tmp_path):
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
