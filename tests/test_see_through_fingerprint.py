import sys
import types
import importlib
from types import SimpleNamespace

import pytest


@pytest.fixture
def build_config_fingerprint(monkeypatch):
    fake_layerdiff = types.ModuleType("module.see_through.pipelines.layerdiff")
    fake_layerdiff.LayerDiffPhase = object
    monkeypatch.setitem(sys.modules, "module.see_through.pipelines.layerdiff", fake_layerdiff)

    fake_marigold = types.ModuleType("module.see_through.pipelines.marigold")
    fake_marigold.MarigoldPhase = object
    monkeypatch.setitem(sys.modules, "module.see_through.pipelines.marigold", fake_marigold)

    fake_model_manager = types.ModuleType("module.see_through.model_manager")
    fake_model_manager.SeeThroughModelManager = object
    monkeypatch.setitem(sys.modules, "module.see_through.model_manager", fake_model_manager)

    fake_postprocess = types.ModuleType("module.see_through.postprocess")
    fake_postprocess.run_postprocess = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "module.see_through.postprocess", fake_postprocess)

    fake_runtime = types.ModuleType("module.see_through.runtime")
    fake_runtime.resolve_attention_backend = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "module.see_through.runtime", fake_runtime)

    sys.modules.pop("module.see_through.runner", None)
    return importlib.import_module("module.see_through.runner").build_config_fingerprint


def _make_config(**overrides):
    base = {
        "repo_id_layerdiff": "layerdifforg/seethroughv0.0.2_layerdiff3d",
        "repo_id_depth": "24yearsold/seethroughv0.0.1_marigold",
        "resolution": 1024,
        "resolution_depth": 720,
        "inference_steps_depth": -1,
        "seed": 42,
        "dtype": "bfloat16",
        "quant_mode": "none",
        "group_offload": False,
        "save_to_psd": True,
        "tblr_split": False,
        "vae_ckpt": None,
        "unet_ckpt": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_config_fingerprint_changes_when_output_affecting_see_through_settings_change(build_config_fingerprint):
    baseline = _make_config()
    changed_quant_mode = _make_config(quant_mode="nf4")
    changed_resolution_depth = _make_config(resolution_depth=-1)
    changed_inference_steps_depth = _make_config(inference_steps_depth=8)
    changed_seed = _make_config(seed=123)

    assert build_config_fingerprint(baseline) != build_config_fingerprint(changed_quant_mode)
    assert build_config_fingerprint(baseline) != build_config_fingerprint(changed_resolution_depth)
    assert build_config_fingerprint(baseline) != build_config_fingerprint(changed_inference_steps_depth)
    assert build_config_fingerprint(baseline) != build_config_fingerprint(changed_seed)


def test_build_config_fingerprint_ignores_group_offload_runtime_only_switch(build_config_fingerprint):
    baseline = _make_config(group_offload=False)
    changed_group_offload = _make_config(group_offload=True)

    assert build_config_fingerprint(baseline) == build_config_fingerprint(changed_group_offload)
