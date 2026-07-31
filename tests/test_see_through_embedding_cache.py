import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = ROOT / "module" / "see_through" / "vendor"

BODY_TAGS_V3 = [
    "front hair",
    "back hair",
    "head",
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
HEAD_TAGS_V3 = [
    "headwear",
    "face",
    "irides",
    "eyebrow",
    "eyewhite",
    "eyelash",
    "eyewear",
    "ears",
    "earwear",
    "nose",
    "mouth",
]


class _FakeDiffusionPipeline:
    def __init__(self, **components):
        self.register_modules(**components)

    def register_modules(self, **components):
        for name, component in components.items():
            setattr(self, name, component)

    def register_to_config(self, **config):
        self.config = types.SimpleNamespace(**config)


class _TrackedTextEncoder:
    def __init__(self):
        self.cpu_calls = 0

    def cpu(self):
        self.cpu_calls += 1
        return self


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_module(monkeypatch, *, module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _load_layerdiff_vendor_module(monkeypatch):
    pipeline_module_name = (
        "diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img"
    )
    pipeline_module = types.ModuleType(pipeline_module_name)
    pipeline_module.__all__ = [
        "torch",
        "StableDiffusionXLImg2ImgPipeline",
        "CLIPVisionModelWithProjection",
        "CLIPImageProcessor",
    ]
    pipeline_module.torch = torch
    pipeline_module.StableDiffusionXLImg2ImgPipeline = _FakeDiffusionPipeline
    pipeline_module.CLIPVisionModelWithProjection = type("CLIPVisionModelWithProjection", (), {})
    pipeline_module.CLIPImageProcessor = type("CLIPImageProcessor", (), {})

    diffusers = _package("diffusers")
    for name in (
        "StableDiffusionXLPipeline",
        "StableDiffusionPipeline",
        "DPMSolverMultistepScheduler",
        "DPMSolverSinglestepScheduler",
        "EulerDiscreteScheduler",
    ):
        setattr(diffusers, name, type(name, (), {}))

    diffusers_utils = _package("diffusers.utils")
    diffusers_outputs = types.ModuleType("diffusers.utils.outputs")
    diffusers_outputs.BaseOutput = object

    fake_vae = types.ModuleType("module.see_through.vendor.modules.layerdiffuse.vae")
    fake_vae.TransparentVAEDecoder = object
    fake_vae.TransparentVAEEncoder = object
    fake_vae.vae_encode = lambda *args, **kwargs: None

    fake_layerdiff3d = types.ModuleType("module.see_through.vendor.modules.layerdiffuse.layerdiff3d")
    fake_layerdiff3d.UNetFrameConditionModel = type("UNetFrameConditionModel", (), {})

    fake_torch_utils = types.ModuleType("module.see_through.vendor.utils.torch_utils")
    fake_torch_utils.seed_everything = lambda *args, **kwargs: None
    fake_torch_utils.img2tensor = lambda *args, **kwargs: None
    fake_torch_utils.tensor2img = lambda *args, **kwargs: None

    modules = {
        "diffusers": diffusers,
        "diffusers.pipelines": _package("diffusers.pipelines"),
        "diffusers.pipelines.stable_diffusion_xl": _package("diffusers.pipelines.stable_diffusion_xl"),
        pipeline_module_name: pipeline_module,
        "diffusers.utils": diffusers_utils,
        "diffusers.utils.outputs": diffusers_outputs,
        "module.see_through.vendor.modules.layerdiffuse.vae": fake_vae,
        "module.see_through.vendor.modules.layerdiffuse.layerdiff3d": fake_layerdiff3d,
        "module.see_through.vendor.utils.torch_utils": fake_torch_utils,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    return _load_module(
        monkeypatch,
        module_name="module.see_through.vendor.modules.layerdiffuse._embedding_cache_test",
        path=VENDOR_DIR / "modules" / "layerdiffuse" / "diffusers_kdiffusion_sdxl.py",
    )


def _load_marigold_vendor_module(monkeypatch):
    diffusers = _package("diffusers")
    diffusers.DiffusionPipeline = _FakeDiffusionPipeline
    for name in ("AutoencoderKL", "DDIMScheduler", "LCMScheduler", "UNet2DConditionModel"):
        setattr(diffusers, name, type(name, (), {}))

    diffusers_utils = _package("diffusers.utils")
    diffusers_utils.BaseOutput = object

    torchvision = _package("torchvision")
    torchvision_transforms = _package("torchvision.transforms")
    torchvision_transforms.InterpolationMode = type("InterpolationMode", (), {})
    torchvision_functional = types.ModuleType("torchvision.transforms.functional")
    torchvision_functional.pil_to_tensor = lambda image: image
    torchvision_functional.resize = lambda image, *args, **kwargs: image

    transformers = types.ModuleType("transformers")
    transformers.CLIPTextModel = type("CLIPTextModel", (), {})
    transformers.CLIPTokenizer = type("CLIPTokenizer", (), {})

    fake_layerdiff3d = types.ModuleType("module.see_through.vendor.modules.layerdiffuse.layerdiff3d")
    fake_layerdiff3d.UNetFrameConditionModel = type("UNetFrameConditionModel", (), {})

    fake_batchsize = types.ModuleType("module.see_through.vendor.modules.marigold.util.batchsize")
    fake_batchsize.find_batch_size = lambda **kwargs: 1
    fake_ensemble = types.ModuleType("module.see_through.vendor.modules.marigold.util.ensemble")
    fake_ensemble.ensemble_depth = lambda *args, **kwargs: None
    fake_image_util = types.ModuleType("module.see_through.vendor.modules.marigold.util.image_util")
    fake_image_util.chw2hwc = lambda value: value
    fake_image_util.colorize_depth_maps = lambda *args, **kwargs: None
    fake_image_util.get_tv_resample_method = lambda value: value
    fake_image_util.resize_max_res = lambda value, **kwargs: value

    fake_torchcv = types.ModuleType("module.see_through.vendor.utils.torchcv")
    fake_torchcv.pad_rgb_torch = lambda value, **kwargs: value
    fake_torch_utils = types.ModuleType("module.see_through.vendor.utils.torch_utils")
    fake_torch_utils.img2tensor = lambda value, **kwargs: value

    modules = {
        "diffusers": diffusers,
        "diffusers.utils": diffusers_utils,
        "torchvision": torchvision,
        "torchvision.transforms": torchvision_transforms,
        "torchvision.transforms.functional": torchvision_functional,
        "transformers": transformers,
        "module.see_through.vendor.modules.layerdiffuse.layerdiff3d": fake_layerdiff3d,
        "module.see_through.vendor.modules.marigold.util.batchsize": fake_batchsize,
        "module.see_through.vendor.modules.marigold.util.ensemble": fake_ensemble,
        "module.see_through.vendor.modules.marigold.util.image_util": fake_image_util,
        "module.see_through.vendor.utils.torchcv": fake_torchcv,
        "module.see_through.vendor.utils.torch_utils": fake_torch_utils,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    return _load_module(
        monkeypatch,
        module_name="module.see_through.vendor.modules.marigold._embedding_cache_test",
        path=VENDOR_DIR / "modules" / "marigold" / "marigold_depth_pipeline.py",
    )


def test_layerdiff_caches_v3_tag_embeddings_before_unloading_text_encoders(monkeypatch):
    module = _load_layerdiff_vendor_module(monkeypatch)
    unet = module.UNetFrameConditionModel()
    unet.device = torch.device("cpu")
    unet.dtype = torch.float32
    unet.get_tag_version = lambda: "v3"
    text_encoder = _TrackedTextEncoder()
    text_encoder_2 = _TrackedTextEncoder()
    pipeline = module.KDiffusionStableDiffusionXLPipeline(
        vae=object(),
        text_encoder=text_encoder,
        tokenizer=object(),
        text_encoder_2=text_encoder_2,
        tokenizer_2=object(),
        unet=unet,
        scheduler=object(),
        trans_vae=object(),
    )

    encoded_batches = []

    def fake_encode(prompts):
        encoded_batches.append(list(prompts))
        offset = 100 * (len(encoded_batches) - 1)
        values = torch.arange(offset, offset + len(prompts), dtype=torch.float32)
        return values.reshape(-1, 1, 1), (values + 1000).reshape(-1, 1)

    pipeline.encode_cropped_prompt_77tokens = fake_encode
    empty_cache_calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True))

    pipeline.cache_tag_embeds()

    assert encoded_batches == [BODY_TAGS_V3, HEAD_TAGS_V3]
    assert set(pipeline._cached_prompt_embeds) == set(BODY_TAGS_V3 + HEAD_TAGS_V3)
    prompt_embeds, pooled_prompt_embeds = pipeline.encode_cropped_prompt_77tokens_cached(
        ["mouth", "front hair"]
    )
    assert prompt_embeds.flatten().tolist() == [110.0, 0.0]
    assert pooled_prompt_embeds.flatten().tolist() == [1110.0, 1000.0]
    assert text_encoder.cpu_calls == 1
    assert text_encoder_2.cpu_calls == 1
    assert isinstance(pipeline.text_encoder, torch.nn.Identity)
    assert isinstance(pipeline.text_encoder_2, torch.nn.Identity)
    assert pipeline.device == torch.device("cpu")
    assert empty_cache_calls == [True]


def test_marigold_caches_empty_embedding_before_unloading_text_encoder(monkeypatch):
    module = _load_marigold_vendor_module(monkeypatch)
    unet = module.UNetFrameConditionModel()
    unet.device = torch.device("cpu")
    text_encoder = _TrackedTextEncoder()
    pipeline = module.MarigoldDepthPipeline(
        unet=unet,
        vae=object(),
        scheduler=object(),
        text_encoder=text_encoder,
        tokenizer=object(),
    )

    encode_calls = []

    def fake_encode_empty_text():
        encode_calls.append(True)
        pipeline.empty_text_embed = torch.ones((1, 2, 3))

    pipeline.encode_empty_text = fake_encode_empty_text
    empty_cache_calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True))

    pipeline.cache_tag_embeds()
    pipeline.cache_tag_embeds()

    assert encode_calls == [True]
    assert text_encoder.cpu_calls == 1
    assert isinstance(pipeline.text_encoder, torch.nn.Identity)
    assert pipeline.device == torch.device("cpu")
    assert empty_cache_calls == [True]
