import io
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


def test_resolve_runtime_backend_prefers_args_over_config():
    from providers.backends import resolve_runtime_backend

    args = SimpleNamespace(
        openai_base_url="http://127.0.0.1:8000/v1",
        openai_api_key="test-key",
        openai_model_name="shared-served-model",
    )
    section = {
        "runtime_backend": "direct",
        "runtime_base_url": "http://ignored",
        "runtime_api_key": "ignored",
        "runtime_model_id": "ignored-model",
        "runtime_temperature": 0.8,
        "runtime_max_tokens": 999,
    }

    runtime = resolve_runtime_backend(
        args,
        section,
        arg_prefix="local_runtime",
        shared_prefix="openai",
        default_model_id="fallback-model",
        default_temperature=0.0,
        default_max_tokens=128,
    )

    assert runtime.mode == "openai"
    assert runtime.base_url == "http://127.0.0.1:8000/v1"
    assert runtime.api_key == "test-key"
    assert runtime.model_id == "shared-served-model"
    assert runtime.temperature == 0.8
    assert runtime.max_tokens == 999


def test_openai_compatible_does_not_steal_explicit_local_route():
    from providers.registry import get_registry

    args = SimpleNamespace(
        openai_base_url="http://127.0.0.1:8000/v1",
        openai_api_key="sk-no-key",
        openai_model_name="shared-model",
        vlm_image_model="qwen_vl_local",
        ocr_model="",
        document_image=False,
        pair_dir="",
        step_api_key="",
        ark_api_key="",
        qwenVL_api_key="",
        glm_api_key="",
        kimi_code_api_key="",
        kimi_api_key="",
        mistral_api_key="",
        pixtral_api_key="",
        gemini_api_key="",
        local_runtime_backend="openai",
    )

    provider = get_registry().find_provider(args, "image/jpeg")
    assert provider is not None
    assert provider.name == "qwen_vl_local"


def test_explicit_vlm_video_route_does_not_fall_back_to_openai_compatible():
    from providers.registry import ProviderSelectionError, get_registry

    args = SimpleNamespace(
        openai_base_url="http://127.0.0.1:8000/v1",
        openai_api_key="sk-no-key",
        openai_model_name="shared-model",
        vlm_image_model="qwen_vl_local",
        ocr_model="olmocr",
        document_image=True,
        pair_dir="",
        step_api_key="",
        ark_api_key="",
        qwenVL_api_key="",
        glm_api_key="",
        kimi_code_api_key="",
        kimi_api_key="",
        mistral_api_key="",
        pixtral_api_key="",
        gemini_api_key="",
        local_runtime_backend="openai",
    )

    with pytest.raises(ProviderSelectionError, match="cannot handle mime=video/mp4"):
        get_registry().find_provider(args, "video/mp4")


def test_explicit_reka_video_route_still_wins_with_openai_compatible_configured():
    from providers.registry import get_registry

    args = SimpleNamespace(
        openai_base_url="http://127.0.0.1:8000/v1",
        openai_api_key="sk-no-key",
        openai_model_name="shared-model",
        vlm_image_model="reka_edge_local",
        ocr_model="olmocr",
        document_image=True,
        pair_dir="",
        step_api_key="",
        ark_api_key="",
        qwenVL_api_key="",
        glm_api_key="",
        kimi_code_api_key="",
        kimi_api_key="",
        mistral_api_key="",
        pixtral_api_key="",
        gemini_api_key="",
        local_runtime_backend="openai",
    )

    provider = get_registry().find_provider(args, "video/mp4")
    assert provider is not None
    assert provider.name == "reka_edge_local"


def test_local_vlm_prepare_media_populates_pair_image(tmp_path):
    from providers.base import CaptionResult, ProviderContext
    from providers.local_vlm_base import LocalVLMProvider

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"image")
    pair_dir = tmp_path / "pair"
    pair_dir.mkdir()
    pair_path = pair_dir / image_path.name
    pair_path.write_bytes(b"pair")

    class DummyLocalVLM(LocalVLMProvider):
        name = "dummy_local_vlm"
        default_model_id = "dummy/model"

        @classmethod
        def can_handle(cls, args, mime):
            return True

        def attempt(self, media, prompts):
            return CaptionResult(raw="")

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={},
        args=SimpleNamespace(pair_dir=str(pair_dir)),
    )

    with patch(
        "providers.local_vlm_base.encode_image_to_blob",
        side_effect=[("primary-blob", "primary-pixels"), ("pair-blob", "pair-pixels")],
    ):
        media = DummyLocalVLM(ctx).prepare_media(str(image_path), "image/png", ctx.args)

    assert media.blob == "primary-blob"
    assert media.pair_blob == "pair-blob"
    assert media.pair_pixels == "pair-pixels"
    assert media.extras["pair_uri"] == str(pair_path.resolve())


def test_local_vlm_prepare_media_preserves_video_metadata(tmp_path):
    from providers.base import CaptionResult, MediaModality, ProviderContext
    from providers.local_vlm_base import LocalVLMProvider

    video_path = tmp_path / "sample.mp4"
    video_bytes = b"fake-video-bytes"
    video_path.write_bytes(video_bytes)

    class DummyLocalVLM(LocalVLMProvider):
        name = "dummy_local_vlm"
        default_model_id = "dummy/model"

        @classmethod
        def can_handle(cls, args, mime):
            return True

        def attempt(self, media, prompts):
            return CaptionResult(raw="")

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={},
        args=SimpleNamespace(pair_dir=""),
    )

    media = DummyLocalVLM(ctx).prepare_media(str(video_path), "video/mp4", ctx.args)

    assert media.modality is MediaModality.VIDEO
    assert media.file_size == len(video_bytes)
    assert media.blob is None


def test_local_vlm_openai_backend_supports_video_messages(tmp_path):
    from providers.base import CaptionResult, PromptContext, ProviderContext
    from providers.local_vlm_base import LocalVLMProvider

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake-video-bytes")

    class DummyLocalVLM(LocalVLMProvider):
        name = "dummy_local_vlm"
        default_model_id = "dummy/model"

        @classmethod
        def can_handle(cls, args, mime):
            return True

        def attempt(self, media, prompts):
            return CaptionResult(raw="")

    args = SimpleNamespace(
        pair_dir="",
        local_runtime_backend="openai",
        openai_base_url="http://127.0.0.1:8000/v1",
        openai_api_key="",
        openai_model_name="served-reka",
    )
    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={},
        args=args,
    )
    provider = DummyLocalVLM(ctx)
    media = provider.prepare_media(str(video_path), "video/mp4", ctx.args)
    prompts = PromptContext(system="system prompt", user="describe the clip")

    with patch("providers.local_vlm_base.OpenAIChatRuntime.complete", return_value="video answer") as mock_complete:
        result = provider.attempt_via_openai_backend(media, prompts, stop=["<sep>"])

    messages = mock_complete.call_args.args[0]
    assert messages[0]["role"] == "system"
    assert messages[1]["content"][0]["type"] == "video_url"
    assert messages[1]["content"][1]["text"] == "describe the clip"
    assert mock_complete.call_args.kwargs["stop"] == ["<sep>"]
    assert result.raw == "video answer"


def test_attempt_qwenvl_local_normalizes_file_image_uris(tmp_path):
    from module.providers.cloud_vlm import qwenvl as qwenvl_module

    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake-image")
    file_uri = f"file://{image_path.resolve().as_posix()}"

    captured = {}

    class FakeLoader:
        def get_or_load_processor(self, *args, **kwargs):
            return fake_processor

        def get_or_load_model(self, *args, **kwargs):
            captured["model_cls"] = args[1]
            return fake_model

        def prepare_image_inputs(self, processor, messages, **kwargs):
            captured["messages"] = messages
            return {"input_ids": [[101]], "pixel_values": "pixels"}

    class FakeModel:
        device = "cpu"

        def generate(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return [[101, 202]]

    fake_loader = FakeLoader()
    fake_model = FakeModel()
    fake_processor = SimpleNamespace(batch_decode=MagicMock(return_value=["caption"]))
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = object
    fake_transformers.AutoModel = object
    fake_transformers.AutoModelForImageTextToText = type("FakeAutoModelForImageTextToText", (), {})

    loader_attr = "_TRANS_LOADER"
    original_loader = getattr(qwenvl_module.attempt_qwenvl, loader_attr, None)
    had_original_loader = hasattr(qwenvl_module.attempt_qwenvl, loader_attr)
    if had_original_loader:
        delattr(qwenvl_module.attempt_qwenvl, loader_attr)

    try:
        with (
            patch.dict(sys.modules, {"transformers": fake_transformers}),
            patch("utils.transformer_loader.resolve_device_dtype", return_value=("cpu", "float32", "eager")),
            patch("utils.transformer_loader.transformerLoader", return_value=fake_loader),
        ):
            result = qwenvl_module.attempt_qwenvl(
                model_path="Qwen/Qwen3.5-2B",
                api_key="",
                messages=[
                    {"role": "system", "content": [{"text": "system"}]},
                    {"role": "user", "content": [{"image": file_uri}, {"text": "describe"}]},
                ],
                console=Console(file=io.StringIO(), force_terminal=False),
                progress=None,
                task_id=None,
                local_config={"model_id": "Qwen/Qwen3.5-2B"},
            )
    finally:
        if had_original_loader:
            setattr(qwenvl_module.attempt_qwenvl, loader_attr, original_loader)
        elif hasattr(qwenvl_module.attempt_qwenvl, loader_attr):
            delattr(qwenvl_module.attempt_qwenvl, loader_attr)

    assert result == "caption"
    assert captured["model_cls"] is fake_transformers.AutoModelForImageTextToText
    assert captured["messages"][1]["content"][0]["image"] == str(image_path.resolve())


def test_attempt_chandra_ocr_uses_chandra2_hf_contract(tmp_path):
    from PIL import Image

    from module.providers.ocr import chandra as chandra_module

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), "white").save(image_path)

    captured = {}

    class FakeLoader:
        def get_or_load_processor(self, *args, **kwargs):
            captured["processor_id"] = args[0]
            return fake_processor

        def get_or_load_model(self, *args, **kwargs):
            captured["model_id"] = args[0]
            captured["model_cls"] = args[1]
            return fake_model

    class FakeModel:
        device = "cpu"

    class FakeBatchInputItem:
        def __init__(self, *, image, prompt=None, prompt_type=None):
            self.image = image
            self.prompt = prompt
            self.prompt_type = prompt_type

    def fake_generate_hf(batch, model, max_output_tokens=None, **kwargs):
        captured["generate_batch"] = batch
        captured["generate_model"] = model
        captured["max_output_tokens"] = max_output_tokens
        return [SimpleNamespace(raw="<div>hello</div>")]

    def fake_parse_markdown(raw):
        captured["parse_markdown_raw"] = raw
        return "hello"

    fake_loader = FakeLoader()
    fake_model = FakeModel()
    fake_processor = SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right"))

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = object
    fake_transformers.AutoModelForImageTextToText = type("FakeAutoModelForImageTextToText", (), {})

    fake_chandra = types.ModuleType("chandra")
    fake_chandra_model = types.ModuleType("chandra.model")
    fake_chandra_hf = types.ModuleType("chandra.model.hf")
    fake_chandra_schema = types.ModuleType("chandra.model.schema")
    fake_chandra_output = types.ModuleType("chandra.output")

    fake_chandra_hf.generate_hf = fake_generate_hf
    fake_chandra_schema.BatchInputItem = FakeBatchInputItem
    fake_chandra_output.parse_markdown = fake_parse_markdown
    fake_chandra.model = fake_chandra_model
    fake_chandra_model.hf = fake_chandra_hf
    fake_chandra_model.schema = fake_chandra_schema

    original_loader = chandra_module._TRANS_LOADER
    chandra_module._TRANS_LOADER = None

    try:
        with (
            patch.dict(
                sys.modules,
                {
                    "transformers": fake_transformers,
                    "chandra": fake_chandra,
                    "chandra.model": fake_chandra_model,
                    "chandra.model.hf": fake_chandra_hf,
                    "chandra.model.schema": fake_chandra_schema,
                    "chandra.output": fake_chandra_output,
                },
            ),
            patch("module.providers.ocr.chandra.resolve_device_dtype", return_value=("cpu", "float32", "eager")),
            patch("module.providers.ocr.chandra.transformerLoader", return_value=fake_loader),
            patch("module.providers.ocr.chandra.write_markdown_output"),
            patch("module.providers.ocr.chandra.display_markdown"),
        ):
            result = chandra_module.attempt_chandra_ocr(
                uri=str(image_path),
                console=Console(file=io.StringIO(), force_terminal=False),
                progress=None,
                task_id=None,
            )
    finally:
        chandra_module._TRANS_LOADER = original_loader

    assert result == "hello"
    assert captured["processor_id"] == "datalab-to/chandra-ocr-2"
    assert captured["model_id"] == "datalab-to/chandra-ocr-2"
    assert captured["model_cls"] is fake_transformers.AutoModelForImageTextToText
    assert fake_processor.tokenizer.padding_side == "left"
    assert captured["generate_batch"][0].prompt_type == "ocr_layout"
    assert captured["generate_model"] is fake_model
    assert captured["max_output_tokens"] == 12384
    assert captured["parse_markdown_raw"] == "<div>hello</div>"


def test_local_vlm_runtime_uses_model_config_for_shared_openai_model():
    from providers.base import ProviderContext
    from providers.local_vlm_base import LocalVLMProvider

    class DummyLocalVLM(LocalVLMProvider):
        name = "dummy_local_vlm"
        default_model_id = "dummy/model"

        @classmethod
        def can_handle(cls, args, mime):
            return True

        def attempt(self, media, prompts):
            raise AssertionError("not used")

    args = SimpleNamespace(
        pair_dir="",
        openai_base_url="http://127.0.0.1:8000/v1",
        openai_api_key="",
        openai_model_name="served-reka",
    )
    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={
            "dummy_local_vlm": {"model_id": "dummy/model", "temperature": 0.1, "out_seq_length": 111},
            "served_reka": {"model_id": "served-reka", "temperature": 0.35, "max_new_tokens": 456},
        },
        args=args,
    )

    runtime = DummyLocalVLM(ctx).get_runtime_backend()

    assert runtime.mode == "openai"
    assert runtime.model_id == "served-reka"
    assert runtime.temperature == 0.35
    assert runtime.max_tokens == 456


def test_hy_mt_openai_backend_uses_chat_runtime():
    from module.providers.local_llm.hy_mt import HYMTProvider

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="translated output"))]
    )

    with patch("openai.OpenAI", return_value=mock_client):
        provider = HYMTProvider(
            model_id="served-hy-mt",
            console=Console(file=io.StringIO(), force_terminal=False),
            max_new_tokens=777,
            temperature=0.3,
            backend="openai",
            openai_base_url="http://127.0.0.1:9000/v1",
            openai_api_key="",
            openai_model_name="served-hy-mt",
        )
        result = provider.translate("Hello", "en", "zh_cn")

    assert result == "translated output"
    request = mock_client.chat.completions.create.call_args.kwargs
    assert request["model"] == "served-hy-mt"
    assert request["max_tokens"] == 777
    assert request["temperature"] == 0.3
    assert request["messages"][0]["role"] == "user"


def test_reka_edge_local_prefers_fp16_on_cuda():
    import torch

    from providers.base import ProviderContext
    from providers.local_vlm.reka_edge_local import RekaEdgeLocalProvider

    captured = {}

    class FakeProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            captured["processor_args"] = args
            captured["processor_kwargs"] = kwargs
            return "processor"

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            captured["model_args"] = args
            captured["model_kwargs"] = kwargs

            class _FakeModel:
                def eval(self):
                    return self

            return _FakeModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = FakeProcessor
    fake_transformers.AutoModelForImageTextToText = FakeModelLoader

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={"reka_edge_local": {"model_id": "RekaAI/reka-edge-2603"}},
        args=SimpleNamespace(vlm_image_model="reka_edge_local"),
    )

    with (
        patch.dict(sys.modules, {"transformers": fake_transformers}),
        patch("utils.transformer_loader.resolve_device_dtype", return_value=("cuda", torch.bfloat16, "eager")),
    ):
        cached = RekaEdgeLocalProvider(ctx)._load_model()

    assert cached["device"] == "cuda"
    assert cached["dtype"] == torch.float16
    assert captured["model_kwargs"]["torch_dtype"] == torch.float16
    assert captured["model_kwargs"]["device_map"] == "auto"


def test_transformer_loader_passes_torch_dtype_instead_of_dtype():
    import json

    import torch

    from utils.transformer_loader import transformerLoader

    captured = {}

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

            class _FakeModel:
                def eval(self):
                    return self

            return _FakeModel()

    loader = transformerLoader(attn_kw="attn_implementation", device_map="auto")
    loader.load_model(
        "dummy/model",
        FakeModelLoader,
        dtype=torch.float32,
        attn_impl="eager",
        trust_remote_code=False,
        low_cpu_mem_usage=False,
        device_map="cpu",
    )

    assert captured["kwargs"]["torch_dtype"] == "float32"
    assert "dtype" not in captured["kwargs"]
    json.dumps({"torch_dtype": captured["kwargs"]["torch_dtype"]})


def test_resolve_device_dtype_prefers_sdpa_when_flash_attn_missing_on_cuda(monkeypatch):
    import torch

    from utils import transformer_loader

    monkeypatch.setattr(transformer_loader.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(transformer_loader, "_has_flash_attn", lambda: False)
    monkeypatch.setattr(transformer_loader, "_has_sdpa", lambda: True)

    device, dtype, attn_impl = transformer_loader.resolve_device_dtype()

    assert device == "cuda"
    assert dtype == getattr(torch, "bfloat16", torch.float16)
    assert attn_impl == "sdpa"


def test_resolve_device_dtype_prefers_flex_when_supported_and_enabled_on_cuda(monkeypatch):
    import torch

    from utils import transformer_loader

    monkeypatch.setattr(transformer_loader.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(transformer_loader, "_has_flex_attention", lambda: True)
    monkeypatch.setattr(transformer_loader, "_has_flash_attn", lambda: True)
    monkeypatch.setattr(transformer_loader, "_has_sdpa", lambda: True)

    device, dtype, attn_impl = transformer_loader.resolve_device_dtype(supports_flex_attn=True)

    assert device == "cuda"
    assert dtype == getattr(torch, "bfloat16", torch.float16)
    assert attn_impl == "flex_attention"


def test_transformer_loader_retries_with_sdpa_when_flash_attn_load_fails(monkeypatch):
    import torch

    from utils.transformer_loader import transformerLoader

    attempts = []

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            attempts.append(kwargs.get("attn_implementation"))
            if kwargs.get("attn_implementation") == "flash_attention_2":
                raise ImportError("flash_attn import failed")

            class _FakeModel:
                def eval(self):
                    return self

            return _FakeModel()

    monkeypatch.setattr("utils.transformer_loader._has_sdpa", lambda: True)

    loader = transformerLoader(attn_kw="attn_implementation", device_map="auto")
    model = loader.load_model(
        "dummy/model",
        FakeModelLoader,
        dtype=torch.float16,
        attn_impl="flash_attention_2",
        device_map="cpu",
    )

    assert model is not None
    assert attempts == ["flash_attention_2", "sdpa"]


def test_transformer_loader_prefers_flex_then_flash_then_sdpa_then_eager(monkeypatch):
    import torch

    from utils.transformer_loader import transformerLoader

    attempts = []

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            attn_impl = kwargs.get("attn_implementation")
            attempts.append(attn_impl)
            if attn_impl == "flex_attention":
                raise ImportError("flex_attention backend unavailable")
            if attn_impl == "flash_attention_2":
                raise ImportError("flash_attn import failed")

            class _FakeModel:
                def eval(self):
                    return self

            return _FakeModel()

    monkeypatch.setattr("utils.transformer_loader._has_flex_attention", lambda: True)
    monkeypatch.setattr("utils.transformer_loader._has_flash_attn", lambda: True)
    monkeypatch.setattr("utils.transformer_loader._has_sdpa", lambda: True)

    loader = transformerLoader(attn_kw="attn_implementation", device_map="auto", supports_flex_attn=True)
    model = loader.load_model(
        "dummy/model",
        FakeModelLoader,
        dtype=torch.float16,
        attn_impl="flash_attention_2",
        device_map="cpu",
    )

    assert model is not None
    assert attempts == ["flex_attention", "flash_attention_2", "sdpa"]


def test_transformer_loader_retries_with_eager_when_sdpa_unavailable(monkeypatch):
    import torch

    from utils.transformer_loader import transformerLoader

    attempts = []

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            attempts.append(kwargs.get("attn_implementation"))
            if kwargs.get("attn_implementation") == "flash_attention_2":
                raise ImportError("flash_attn import failed")

            class _FakeModel:
                def eval(self):
                    return self

            return _FakeModel()

    monkeypatch.setattr("utils.transformer_loader._has_sdpa", lambda: False)

    loader = transformerLoader(attn_kw="attn_implementation", device_map="auto")
    model = loader.load_model(
        "dummy/model",
        FakeModelLoader,
        dtype=torch.float16,
        attn_impl="flash_attention_2",
        device_map="cpu",
    )

    assert model is not None
    assert attempts == ["flash_attention_2", "eager"]


def test_move_pretrained_component_skips_dtype_for_quantized_models():
    import torch

    from utils.transformer_loader import move_pretrained_component

    calls = []

    class FakeQuantizedModel:
        quantization_method = "bitsandbytes"

        def to(self, *args, **kwargs):
            calls.append((args, kwargs))
            if "dtype" in kwargs:
                raise AssertionError("dtype should not be passed to quantized models")
            return self

    model = FakeQuantizedModel()
    moved = move_pretrained_component(model, device="cpu", dtype=torch.float16)

    assert moved is model
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ()
    assert kwargs["device"] == "cpu"
    assert "dtype" not in kwargs


def test_prepare_multimodal_inputs_moves_batch_feature_like_result():
    from utils.transformer_loader import prepare_multimodal_inputs

    calls = []

    class FakeBatchFeature:
        def to(self, device, dtype=None):
            calls.append((device, dtype))
            return {"input_ids": f"moved:{device}", "dtype": dtype}

    class FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "hello"}]
            assert kwargs["tokenize"] is True
            assert kwargs["return_dict"] is True
            assert kwargs["return_tensors"] == "pt"
            return FakeBatchFeature()

    inputs = prepare_multimodal_inputs(
        FakeProcessor(),
        [{"role": "user", "content": "hello"}],
        device="cuda:0",
        dtype="bfloat16",
    )

    assert inputs == {"input_ids": "moved:cuda:0", "dtype": "bfloat16"}
    assert calls == [("cuda:0", "bfloat16")]


def test_prepare_multimodal_inputs_retries_without_optional_chat_template_kwargs():
    from utils.transformer_loader import prepare_multimodal_inputs

    seen_kwargs = []

    class FakeBatchFeature:
        def to(self, device, dtype=None):
            return {"device": device, "dtype": dtype}

    class FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            seen_kwargs.append(dict(kwargs))
            if "video_fps" in kwargs:
                raise TypeError("unexpected keyword argument 'video_fps'")
            return FakeBatchFeature()

    inputs = prepare_multimodal_inputs(
        FakeProcessor(),
        [{"role": "user", "content": "hello"}],
        device="cpu",
        chat_template_kwargs={"video_fps": 1.0},
    )

    assert inputs == {"device": "cpu", "dtype": None}
    assert len(seen_kwargs) == 2
    assert seen_kwargs[0]["video_fps"] == 1.0
    assert "video_fps" not in seen_kwargs[1]


def test_reka_edge_local_tolerates_quantized_cpu_model_move():
    import torch

    from providers.base import ProviderContext
    from providers.local_vlm.reka_edge_local import RekaEdgeLocalProvider

    captured = {}

    class FakeProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            captured["processor_args"] = args
            captured["processor_kwargs"] = kwargs
            return "processor"

    class FakeQuantizedModel:
        quantization_method = "bitsandbytes"

        def eval(self):
            return self

        def to(self, *args, **kwargs):
            captured.setdefault("to_calls", []).append((args, kwargs))
            raise ValueError(".to is not supported for 4-bit or 8-bit bitsandbytes models")

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            captured["model_args"] = args
            captured["model_kwargs"] = kwargs
            return FakeQuantizedModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = FakeProcessor
    fake_transformers.AutoModelForImageTextToText = FakeModelLoader

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={"reka_edge_local": {"model_id": "RekaAI/reka-edge-2603"}},
        args=SimpleNamespace(vlm_image_model="reka_edge_local"),
    )

    with (
        patch.dict(sys.modules, {"transformers": fake_transformers}),
        patch("utils.transformer_loader.resolve_device_dtype", return_value=("cpu", torch.float32, "eager")),
    ):
        cached = RekaEdgeLocalProvider(ctx)._load_model()

    assert cached["device"] == "cpu"
    assert isinstance(cached["model"], FakeQuantizedModel)
    assert captured["model_kwargs"]["torch_dtype"] == torch.float32
    assert len(captured["to_calls"]) == 1
    args, kwargs = captured["to_calls"][0]
    assert "dtype" not in kwargs
    assert kwargs.get("device") == "cpu" or args == ("cpu",)


def test_load_pretrained_component_wraps_hf_progress_context(monkeypatch):
    from contextlib import contextmanager

    from utils.transformer_loader import load_pretrained_component

    state = {}
    console_buffer = io.StringIO()

    @contextmanager
    def original_progress_context(**kwargs):
        state.setdefault("original_calls", []).append(kwargs)

        class _OriginalBar:
            def update(self, advance):
                state.setdefault("original_updates", []).append(advance)

        yield _OriginalBar()

    fake_file_download = types.SimpleNamespace(_get_progress_bar_context=original_progress_context)
    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.file_download = fake_file_download
    fake_utils = types.ModuleType("huggingface_hub.utils")

    def fake_enable_progress_bars():
        state["progress_bars_enabled"] = True

    fake_utils.enable_progress_bars = fake_enable_progress_bars

    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", fake_utils)

    class FakeLoader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            state["patched_during_call"] = fake_file_download._get_progress_bar_context is not original_progress_context
            state["kwargs"] = kwargs
            with fake_file_download._get_progress_bar_context(
                desc="model.safetensors",
                log_level=20,
                total=100,
                initial=0,
                name="huggingface_hub.http_get",
            ) as progress:
                progress.update(40)
                progress.update(60)
            return {"model_id": model_id}

    result = load_pretrained_component(
        FakeLoader,
        "demo/model",
        console=Console(file=console_buffer, force_terminal=False),
        component_name="model",
        trust_remote_code=True,
    )

    assert result == {"model_id": "demo/model"}
    assert state["progress_bars_enabled"] is True
    assert state["patched_during_call"] is True
    assert state["kwargs"]["trust_remote_code"] is True
    assert fake_file_download._get_progress_bar_context is original_progress_context
    output = console_buffer.getvalue()
    assert "Resolving Hugging Face model" in output
    assert "Hugging Face model ready" in output


def test_snapshot_download_with_reporting_wraps_hf_progress_context(monkeypatch):
    from contextlib import contextmanager

    from utils.transformer_loader import snapshot_download_with_reporting

    state = {}

    @contextmanager
    def original_progress_context(**kwargs):
        state.setdefault("original_calls", []).append(kwargs)

        class _OriginalBar:
            def update(self, advance):
                state.setdefault("original_updates", []).append(advance)

        yield _OriginalBar()

    fake_file_download = types.SimpleNamespace(_get_progress_bar_context=original_progress_context)
    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.file_download = fake_file_download
    fake_utils = types.ModuleType("huggingface_hub.utils")

    def fake_enable_progress_bars():
        state["progress_bars_enabled"] = True

    def fake_snapshot_download(*, repo_id, **kwargs):
        state["patched_during_call"] = fake_file_download._get_progress_bar_context is not original_progress_context
        state["kwargs"] = kwargs
        with fake_file_download._get_progress_bar_context(
            desc="weights.safetensors",
            log_level=20,
            total=64,
            initial=0,
            name="huggingface_hub.http_get",
        ) as progress:
            progress.update(64)
        return f"C:/models/{repo_id.replace('/', '_')}"

    fake_hf.snapshot_download = fake_snapshot_download
    fake_utils.enable_progress_bars = fake_enable_progress_bars

    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", fake_utils)

    path = snapshot_download_with_reporting("demo/model", local_dir="C:/cache")

    assert path == "C:/models/demo_model"
    assert state["progress_bars_enabled"] is True
    assert state["patched_during_call"] is True
    assert state["kwargs"]["local_dir"] == "C:/cache"
    assert fake_file_download._get_progress_bar_context is original_progress_context


def test_load_pretrained_component_temporarily_disables_library_progress_bars(monkeypatch):
    from utils.transformer_loader import load_pretrained_component

    state = {}

    def _make_progress_module(module_name):
        enabled = {"value": True}
        module = types.ModuleType(module_name)

        def is_progress_bar_enabled():
            return enabled["value"]

        def disable_progress_bar():
            enabled["value"] = False
            state.setdefault("disabled", []).append(module_name)

        def enable_progress_bar():
            enabled["value"] = True
            state.setdefault("enabled", []).append(module_name)

        module.is_progress_bar_enabled = is_progress_bar_enabled
        module.disable_progress_bar = disable_progress_bar
        module.enable_progress_bar = enable_progress_bar
        return module, enabled

    diffusers_pkg = types.ModuleType("diffusers")
    diffusers_utils_pkg = types.ModuleType("diffusers.utils")
    diffusers_logging_mod, diffusers_enabled = _make_progress_module("diffusers.utils.logging")

    transformers_pkg = types.ModuleType("transformers")
    transformers_utils_pkg = types.ModuleType("transformers.utils")
    transformers_logging_mod, transformers_enabled = _make_progress_module("transformers.utils.logging")

    monkeypatch.setitem(sys.modules, "diffusers", diffusers_pkg)
    monkeypatch.setitem(sys.modules, "diffusers.utils", diffusers_utils_pkg)
    monkeypatch.setitem(sys.modules, "diffusers.utils.logging", diffusers_logging_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_pkg)
    monkeypatch.setitem(sys.modules, "transformers.utils", transformers_utils_pkg)
    monkeypatch.setitem(sys.modules, "transformers.utils.logging", transformers_logging_mod)

    class FakeLoader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            state["during_call"] = (
                diffusers_enabled["value"],
                transformers_enabled["value"],
            )
            return {"model_id": model_id, "kwargs": kwargs}

    result = load_pretrained_component(FakeLoader, "demo/model")

    assert result["model_id"] == "demo/model"
    assert state["during_call"] == (False, False)
    assert diffusers_enabled["value"] is True
    assert transformers_enabled["value"] is True


def test_load_pretrained_component_translates_missing_bitsandbytes_error():
    from importlib.metadata import PackageNotFoundError

    from utils.transformer_loader import load_pretrained_component

    class FakeLoader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            raise PackageNotFoundError("bitsandbytes")

    with pytest.raises(RuntimeError, match="bitsandbytes"):
        load_pretrained_component(FakeLoader, "demo/model")
