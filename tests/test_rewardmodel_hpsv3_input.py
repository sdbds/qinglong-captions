import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _class_stub(name):
    return type(name, (), {})


def _install_import_stubs(monkeypatch):
    for package_name in [
        "imscore",
        "imscore.aesthetic",
        "imscore.cyclereward",
        "imscore.evalmuse",
        "imscore.hps",
        "imscore.hpsv3",
        "imscore.imreward",
        "imscore.mps",
        "imscore.pickscore",
        "imscore.preference",
        "imscore.vqascore",
    ]:
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    model_modules = {
        "imscore.aesthetic.model": [
            "CLIPAestheticScorer",
            "Dinov2AestheticScorer",
            "LAIONAestheticScorer",
            "ShadowAesthetic",
            "SiglipAestheticScorer",
        ],
        "imscore.cyclereward.model": ["CycleReward"],
        "imscore.evalmuse.model": ["EvalMuse"],
        "imscore.hps.model": ["HPSv2"],
        "imscore.hpsv3.model": ["HPSv3"],
        "imscore.imreward.model": ["ImageReward"],
        "imscore.mps.model": ["MPS"],
        "imscore.pickscore.model": ["PickScorer"],
        "imscore.preference.model": ["CLIPPreferenceScorer", "CLIPScore", "SiglipPreferenceScorer"],
        "imscore.vqascore.model": ["VQAScore"],
    }
    for module_name, class_names in model_modules.items():
        module = types.ModuleType(module_name)
        for class_name in class_names:
            setattr(module, class_name, _class_stub(class_name))
        monkeypatch.setitem(sys.modules, module_name, module)

    monkeypatch.setitem(sys.modules, "lance", types.ModuleType("lance"))

    lance_import = types.ModuleType("module.lanceImport")
    lance_import.transform2lance = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "module.lanceImport", lance_import)


@pytest.fixture()
def rewardmodel(monkeypatch):
    _install_import_stubs(monkeypatch)
    sys.modules.pop("module.rewardmodel", None)
    import module.rewardmodel as rewardmodel_module

    return rewardmodel_module


class HPSv3:
    def __init__(self):
        self.calls = []
        self._param = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))

    def parameters(self):
        yield self._param

    def score(self, images, prompts):
        self.calls.append((images, prompts))
        return torch.tensor([0.75])


def test_hpsv3_process_batch_converts_bfloat16_tensor_and_pads(rewardmodel):
    model = HPSv3()
    pixels = torch.ones((3, 1, 392), dtype=torch.bfloat16)

    scores = rewardmodel.process_batch([pixels], model, [""])

    assert scores.tolist() == [0.75]
    images, prompts = model.calls[0]
    assert prompts == [" "]
    assert isinstance(images, list)
    assert len(images) == 1
    image = images[0]
    assert image.dtype == torch.float32
    assert image.device.type == "cpu"
    assert tuple(image.shape) == (3, 2, 392)
    _, height, width = image.shape
    assert max(width / height, height / width) < 200


def test_hpsv3_process_batch_passes_numpy_as_tensor_list(rewardmodel):
    model = HPSv3()
    pixels = np.zeros((8, 9, 3), dtype=np.uint8)

    scores = rewardmodel.process_batch([pixels], model, ["a photo"])

    assert scores.tolist() == [0.75]
    images, prompts = model.calls[0]
    assert prompts == ["a photo"]
    assert isinstance(images, list)
    assert tuple(images[0].shape) == (3, 8, 9)
    assert images[0].dtype == torch.float32


def test_reward_parser_accepts_indexed_cuda_device(rewardmodel):
    args = rewardmodel.setup_parser().parse_args(["datasets", "--device=cuda:1"])

    assert args.device == "cuda:1"


def test_normalize_device_dtype_uses_explicit_cuda_index(rewardmodel, monkeypatch):
    calls = []
    monkeypatch.setattr(rewardmodel.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rewardmodel.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(rewardmodel.torch.cuda, "set_device", lambda index: calls.append(index))
    args = types.SimpleNamespace(device="cuda:1", dtype="auto", repo_id="RE-N-Y/pickscore")

    device, dtype = rewardmodel._normalize_device_dtype(args)

    assert device == "cuda:1"
    assert dtype == torch.float16
    assert calls == [1]


def test_normalize_device_dtype_auto_uses_cuda_zero_for_hpsv3(rewardmodel, monkeypatch):
    calls = []
    monkeypatch.setattr(rewardmodel.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rewardmodel.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(rewardmodel.torch.cuda, "set_device", lambda index: calls.append(index))
    args = types.SimpleNamespace(device="auto", dtype="auto", repo_id="RE-N-Y/hpsv3")

    device, dtype = rewardmodel._normalize_device_dtype(args)

    assert device == "cuda:0"
    assert dtype == torch.bfloat16
    assert calls == [0]
