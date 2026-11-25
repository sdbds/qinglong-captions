import io
import json
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from packaging.requirements import Requirement
from PIL import Image
from rich.console import Console

from module.providers.base import MediaContext, MediaModality, PromptContext, ProviderContext
from module.providers.local_vlm import lfm_vl_local as lfm

ROOT = Path(__file__).resolve().parents[1]


def provider(**config):
    return lfm.LFMVLLocalProvider(
        ProviderContext(
            console=Console(file=io.StringIO()),
            config={"lfm_vl_local": config},
            args=SimpleNamespace(vlm_image_model="lfm_vl_local"),
        )
    )


@pytest.mark.parametrize(
    ("variant", "cuda", "decoder", "vision"),
    [
        ("auto", False, "decoder_q4", "vision_encoder_q8"),
        ("auto", True, "decoder_q4f16", "vision_encoder_fp16"),
        ("q8", False, "decoder_q8", "vision_encoder_fp16"),
        ("q8f16", True, "decoder_q8f16", "vision_encoder_fp16"),
        ("fp16", True, "decoder_fp16", "vision_encoder_fp16"),
        ("fp32", False, "decoder", "vision_encoder"),
        ("float32", False, "decoder", "vision_encoder"),
        ("base", False, "decoder", "vision_encoder"),
        ("default", True, "decoder", "vision_encoder"),
    ],
)
def test_load_published_graph_pair_and_sidecars(monkeypatch, tmp_path, variant, cuda, decoder, vision):
    metadata = {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json"}
    graphs = {"vision_encoder": f"onnx/{vision}.onnx", "decoder": f"onnx/{decoder}.onnx"}
    files = metadata | set(graphs.values()) | {path + "_data" for path in graphs.values()}
    downloaded = {}

    def download(repo_id, filename, local_dir, force_download):
        assert filename in files
        downloaded[filename] = force_download
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if filename == "processor_config.json":
            target.write_text(
                json.dumps(
                    {
                        "image_processor": {
                            "patch_size": 16,
                            "merge_size": 2,
                            "size": {"shortest_edge": 65536, "longest_edge": 16777216},
                        }
                    }
                )
            )
        else:
            target.touch()
        return str(target)

    def load_tokenizer(path, **kwargs):
        assert (Path(path) / "chat_template.jinja").is_file()
        # Transformers 4.x calls .keys() on this field (the 5.x export contains an empty list).
        assert isinstance(kwargs.get("extra_special_tokens", []), dict)
        return "tokenizer"

    def qwen_processor(min_pixels=3136, max_pixels=1003520, **kwargs):
        # The 4.56 constructor ignores a valid size dict unless pixel limits are also passed.
        return {**kwargs, "size": {"shortest_edge": min_pixels, "longest_edge": max_pixels}}

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            list_repo_files=lambda _: sorted(files),
            hf_hub_download=download,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            PreTrainedTokenizerFast=SimpleNamespace(from_pretrained=load_tokenizer),
            Qwen2VLImageProcessor=qwen_processor,
        ),
    )
    eps = ["CPUExecutionProvider"]
    if cuda:
        eps = ["TensorrtExecutionProvider", "CUDAExecutionProvider", *eps]
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(get_available_providers=lambda: eps))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda)))
    monkeypatch.setattr(lfm, "load_session_bundle", lambda **kw: SimpleNamespace(sessions=kw["session_paths"]))

    settings = {"decoder_variant": variant, "model_cache_dir": str(tmp_path)}
    cached = provider(**settings)._load_model()
    assert downloaded == dict.fromkeys(files, False)
    assert {key: path.name for key, path in cached["sessions"].items()} == {
        "vision_encoder": f"{vision}.onnx",
        "decoder": f"{decoder}.onnx",
    }
    assert cached["processor"]["patch_size"] == 16
    assert cached["processor"]["size"] == {"shortest_edge": 65536, "longest_edge": 16777216}
    assert cached["runtime_config"].execution_provider == ("cuda" if cuda else "cpu")

    # A forced refresh must use the published listing, not the (possibly stale) local headers.
    downloaded.clear()
    provider(**settings, force_download=True)._load_model()
    assert downloaded == dict.fromkeys(files, True)


def test_auto_uses_cpu_bundle_when_cuda_wheel_has_no_visible_gpu(monkeypatch):
    from module.onnx_runtime.config import OnnxRuntimeConfig

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    runtime, files = lfm.lfm3_vl_onnx.resolve_bundle(OnnxRuntimeConfig(), "auto")
    assert runtime.execution_provider == "cpu"
    assert files == {"vision_encoder": "onnx/vision_encoder_q8.onnx", "decoder": "onnx/decoder_q4.onnx"}


def test_cached_lfm3_graphs_repair_only_the_referenced_missing_shard(tmp_path, monkeypatch):
    onnx = pytest.importorskip("onnx")
    repo = "local-mirror/LFM3-VL-2B-ONNX"
    cache = tmp_path / "local-mirror_LFM3-VL-2B-ONNX" / "onnx"
    cache.mkdir(parents=True)
    for name, suffixes in (("vision_encoder_fp16", ("_data",)), ("decoder_fp16", ("_data", "_data_1"))):
        graph = onnx.ModelProto()
        for i, suffix in enumerate(suffixes):
            location = name + ".onnx" + suffix
            if suffix == "_data":
                weight = graph.graph.initializer.add()
            else:
                node = graph.graph.node.add(op_type="Constant")
                weight = node.attribute.add(name="value", type=onnx.AttributeProto.TENSOR).t
            weight.name = f"weight{i}"
            weight.data_type = onnx.TensorProto.FLOAT
            weight.dims.append(1)
            weight.data_location = onnx.TensorProto.EXTERNAL
            weight.external_data.add(key="location", value=location)
            if suffix == "_data":
                (cache / location).write_bytes(b"\0" * 4)
        (cache / (name + ".onnx")).write_bytes(graph.SerializeToString())

    downloads = []

    def download(repo_id, filename, local_dir, force_download):
        assert repo_id == repo and not force_download
        downloads.append(filename)
        path = Path(local_dir) / filename
        path.write_bytes(b"\0" * 4)
        return str(path)

    hub = SimpleNamespace(list_repo_files=lambda *_: pytest.fail("Unexpected Hub listing"), hf_hub_download=download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"]))
    monkeypatch.setattr(lfm.lfm3_vl_onnx, "load_preprocessing", lambda *args, **kw: {})
    monkeypatch.setattr(lfm, "load_session_bundle", lambda **kw: SimpleNamespace(sessions=kw["session_paths"]))
    settings = dict(model_id=repo, execution_provider="cpu", decoder_variant="fp16", model_cache_dir=str(tmp_path))
    provider(**settings)._load_model()
    assert downloads == ["onnx/decoder_fp16.onnx_data_1"]
    downloads.clear()
    provider(**settings)._load_model()
    assert not downloads


class Tokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and not enable_thinking
        assert messages[0]["content"] == [{"type": "text", "text": "system"}]
        return "".join("<|image|>" if p["type"] == "image" else p["text"] for p in messages[1]["content"])

    def encode(self, prompt, *, add_special_tokens):
        assert not add_special_tokens  # BOS is already in the published template.
        return [10] + [90, 99, 91] * prompt.count("<|image|>") + [11]

    def convert_tokens_to_ids(self, token):
        assert token == "<|image|>"
        return 99

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens
        return ",".join(str(i) for i in ids if i != self.eos_token_id)


class ImageProcessor:
    size = {"shortest_edge": 65536, "longest_edge": 16777216}
    patch_size = 16
    merge_size = 2

    def __call__(self, *, images, return_tensors, size):
        assert size == {"shortest_edge": 65536, "longest_edge": 262144}
        assert return_tensors == "np"
        width, height = images.size
        rows = np.arange(width * height, dtype=np.float32) + images.getpixel((0, 0))[0]
        return {"image_grid_thw": np.array([[1, height, width]]), "pixel_values": rows[:, None]}


class VisionEncoder:
    def run(self, _, feed):
        blocks = feed["pixel_values"]
        assert blocks.shape == (1, 2 if blocks.flat[0] == 10 else 1, 2, 2, 1)
        return [blocks[:, :, 0, 0, :].reshape(-1, 1)]


def meta(name, type_name, shape):
    return SimpleNamespace(name=name, type=f"tensor({type_name})", shape=shape)


class Decoder:
    def __init__(self):
        self.feeds = []

    def get_inputs(self):
        return [
            meta("input_ids", "int64", [1, "sequence_length"]),
            meta("position_ids", "int64", [1, "sequence_length"]),
            meta("attention_mask", "int64", [1, "total_sequence_length"]),
            meta("image_features", "float", ["num_image_tokens", 1]),
            meta("video_features", "float", ["num_video_tokens", 1]),
            meta("logits_to_keep", "int64", [1]),
            meta("past_conv.0", "float16", ["batch_size", 2, 3]),
            meta("past_key_values.1.key", "float16", ["batch_size", 1, "past_sequence_length", 2]),
        ]

    def get_outputs(self):
        return [meta("logits", "float", []), meta("present_conv.0", "float16", []), meta("present.1.key", "float16", [])]

    def run(self, _, feed):
        self.feeds.append(feed)
        logits = np.zeros((1, 1, 50), np.float32)
        logits[0, 0, 42 if len(self.feeds) == 1 else 2] = 1
        length = feed["attention_mask"].shape[1]
        return [logits, np.ones((1, 2, 3), np.float16), np.ones((1, 1, length, 2), np.float16)]


@pytest.fixture
def caption(monkeypatch, tmp_path):
    instance = provider(min_image_tokens=64, max_image_tokens=256, max_new_tokens=3)
    decoder = Decoder()
    monkeypatch.setattr(
        instance,
        "_get_or_load_model",
        lambda: {
            "processor": ImageProcessor(),
            "tokenizer": Tokenizer(),
            "sessions": {"vision_encoder": VisionEncoder(), "decoder": decoder},
        },
    )

    def run(*, pair=False, prompt="describe"):
        path = tmp_path / "image.png"
        Image.new("RGB", (4, 2), (10, 0, 0)).save(path)
        extras = {}
        if pair:
            other = tmp_path / "pair.png"
            Image.new("RGB", (2, 2), (20, 0, 0)).save(other)
            extras["pair_uri"] = str(other)
        media = MediaContext(
            uri=str(path),
            mime="image/png",
            modality=MediaModality.IMAGE,
            sha256hash="",
            file_size=path.stat().st_size,
            extras=extras,
        )
        return instance.attempt(media, PromptContext(system="system", user=prompt)), decoder.feeds

    return run


@pytest.mark.parametrize(
    ("pair", "ids", "features"),
    [
        (False, [10, 90, 99, 99, 91, 11], [10, 14]),
        (True, [10, 90, 99, 99, 91, 90, 99, 91, 11], [10, 14, 20]),
    ],
)
def test_caption_prefills_ordered_image_features_then_decodes_cached_tokens(caption, pair, ids, features):
    result, (first, second) = caption(pair=pair)
    assert result.raw == "42"
    assert result.metadata["model_id"] == "LiquidAI/LFM3-VL-2B-ONNX"
    np.testing.assert_array_equal(first["input_ids"], [ids])
    np.testing.assert_array_equal(first["image_features"][:, 0], features)
    assert first["video_features"].shape == second["image_features"].shape == (0, 1)
    np.testing.assert_array_equal(first["logits_to_keep"], [1])
    assert first["past_key_values.1.key"].shape == (1, 1, 0, 2)
    assert not first["past_conv.0"].any()
    np.testing.assert_array_equal(second["input_ids"], [[42]])
    np.testing.assert_array_equal(second["position_ids"], [[len(ids)]])
    np.testing.assert_array_equal(second["attention_mask"], np.ones((1, len(ids) + 1)))
    np.testing.assert_array_equal(second["past_key_values.1.key"], np.ones((1, 1, len(ids), 2), np.float16))
    np.testing.assert_array_equal(second["past_conv.0"], np.ones((1, 2, 3), np.float16))


def test_caption_rejects_an_extra_image_placeholder(caption):
    with pytest.raises(ValueError, match="placeholder"):
        caption(prompt="describe <|image|>")


def test_lfm_extra_includes_graph_reader_and_compatible_linux_runtime():
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]["optional-dependencies"]
    assert "onnx" in {Requirement(dep).name for dep in extras["lfm-vl-local"]}
    deps = (Requirement(dep) for group in ("lfm-vl-local", "onnx-base") for dep in extras[group])
    constraints = [
        dep.specifier
        for dep in deps
        if dep.name == "onnxruntime-gpu" and (dep.marker is None or dep.marker.evaluate({"sys_platform": "linux"}))
    ]
    assert any("1.21.0" not in spec for spec in constraints)
    assert all("1.24.4" in spec for spec in constraints)


@pytest.mark.optional_runtime
@pytest.mark.integration
def test_offline_q4_captions_real_images_in_prompt_order(tmp_path, monkeypatch, request):
    from module.onnx_runtime.session import clear_session_bundle_cache

    repo = "LiquidAI/LFM3-VL-2B-ONNX"
    root = ROOT / "huggingface"
    cache = root / "LiquidAI_LFM3-VL-2B-ONNX"
    files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "chat_template.jinja",
        "onnx/vision_encoder_q8.onnx",
        "onnx/vision_encoder_q8.onnx_data",
        "onnx/decoder_q4.onnx",
        "onnx/decoder_q4.onnx_data",
    ]
    if not all((cache / filename).is_file() for filename in files):
        pytest.skip("LFM3 CPU q4 bundle is not cached")
    hub = pytest.importorskip("huggingface_hub")
    pytest.importorskip("transformers")
    pytest.importorskip("onnxruntime")
    monkeypatch.setattr(hub, "list_repo_files", lambda *args, **kw: pytest.fail("Unexpected Hub listing during offline startup"))
    monkeypatch.setattr(hub, "hf_hub_download", lambda **kw: pytest.fail(f"Unexpected download: {kw['filename']}"))
    instance = provider(
        model_id=repo,
        execution_provider="cpu",
        model_cache_dir=str(root),
        min_image_tokens=64,
        max_image_tokens=256,
        max_new_tokens=16,
    )
    request.addfinalizer(lambda: clear_session_bundle_cache(instance._model_key))
    request.addfinalizer(lambda: instance.unload_model(repo))
    paths = [tmp_path / "red.png", tmp_path / "blue.png"]
    for path in paths:
        Image.new("RGB", (256, 256), path.stem).save(path)
    prompts = PromptContext(system="Answer in English.", user="Name the two colors in the order of the images.")
    for first, second in (paths, paths[::-1]):
        media = MediaContext(
            uri=str(first),
            mime="image/png",
            modality=MediaModality.IMAGE,
            sha256hash="",
            file_size=first.stat().st_size,
            extras={"pair_uri": str(second)},
        )
        answer = instance.attempt(media, prompts).raw.lower()
        assert first.stem in answer and second.stem in answer and answer.index(first.stem) < answer.index(second.stem), answer
