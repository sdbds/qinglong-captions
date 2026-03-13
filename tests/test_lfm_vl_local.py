import io
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from rich.console import Console
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


def test_lfm_model_config_defaults_match_vl_recommendations():
    model_config = tomllib.loads((ROOT / "config" / "model.toml").read_text(encoding="utf-8"))
    lfm = model_config["lfm_vl_local"]

    assert lfm["max_new_tokens"] == 256
    assert lfm["temperature"] == 0.1
    assert lfm["min_p"] == 0.15
    assert lfm["repetition_penalty"] == 1.05
    assert lfm["min_image_tokens"] == 64
    assert lfm["max_image_tokens"] == 256
    assert lfm["do_image_splitting"] is True


def test_lfm_provider_loads_expected_artifact_bundle(monkeypatch):
    from providers.base import ProviderContext
    from providers.local_vlm.lfm_vl_local import LFMVLLocalProvider

    captured = {}
    console_buffer = io.StringIO()

    class FakeProcessor:
        @staticmethod
        def from_pretrained(model_id, trust_remote_code=False):
            captured["processor"] = (model_id, trust_remote_code)
            return "processor"

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = FakeProcessor
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    def fake_download(repo_id, artifacts, **kwargs):
        captured["download"] = (repo_id, artifacts, kwargs)
        return {name: Path(f"C:/models/{Path(path).name}") for name, path in artifacts.items()}

    def fake_load_bundle(**kwargs):
        captured["bundle"] = kwargs
        return SimpleNamespace(sessions={"decoder": "decoder-session"}, providers=("CPUExecutionProvider",))

    monkeypatch.setattr("providers.local_vlm.lfm_vl_local.download_onnx_artifact_set", fake_download, raising=False)
    monkeypatch.setattr("providers.local_vlm.lfm_vl_local.load_session_bundle", fake_load_bundle, raising=False)

    ctx = ProviderContext(
        console=Console(file=console_buffer, force_terminal=False),
        config={
            "lfm_vl_local": {
                "model_id": "LiquidAI/LFM2.5-VL-1.6B-ONNX",
                "encoder_variant": "fp16",
                "decoder_variant": "q4",
                "execution_provider": "cpu",
            }
        },
        args=SimpleNamespace(vlm_image_model="lfm_vl_local"),
    )

    cached = LFMVLLocalProvider(ctx)._load_model()

    assert captured["processor"] == ("LiquidAI/LFM2.5-VL-1.6B-ONNX", True)
    assert captured["download"][1] == {
        "embed_tokens": "onnx/embed_tokens_fp16.onnx",
        "embed_images": "onnx/embed_images_fp16.onnx",
        "decoder": "onnx/decoder_q4.onnx",
    }
    assert captured["bundle"]["bundle_key"] == "lfm_vl_local:LiquidAI/LFM2.5-VL-1.6B-ONNX"
    assert cached["processor"] == "processor"
    assert cached["sessions"]["decoder"] == "decoder-session"
    output = console_buffer.getvalue()
    assert "embed_tokens_fp16.onnx" in output
    assert "embed_images_fp16.onnx" in output
    assert "decoder_q4.onnx" in output


def test_lfm_provider_attempt_merges_image_embeddings_and_decodes_until_eos(tmp_path, monkeypatch):
    from providers.base import MediaContext, MediaModality, PromptContext, ProviderContext
    from providers.local_vlm.lfm_vl_local import LFMVLLocalProvider

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake-image")

    image_token_id = 99
    eos_token_id = 2
    eos_id = eos_token_id
    decoder_feeds = []
    processor_calls = []

    class FakeTokenizer:
        eos_token_id = eos_id

        @staticmethod
        def convert_tokens_to_ids(token):
            assert token == "<image>"
            return image_token_id

        @staticmethod
        def decode(tokens, skip_special_tokens=True):
            assert skip_special_tokens is True
            return f"decoded:{','.join(str(token) for token in tokens)}"

    class FakeProcessor:
        tokenizer = FakeTokenizer()

        @staticmethod
        def apply_chat_template(messages, add_generation_prompt=False, tokenize=False):
            assert add_generation_prompt is True
            assert tokenize is False
            assert messages[0]["role"] == "system"
            assert messages[1]["content"][0]["type"] == "image"
            return "prompt"

        @staticmethod
        def __call__(
            images=None,
            text=None,
            return_tensors=None,
            min_image_tokens=None,
            max_image_tokens=None,
            do_image_splitting=None,
        ):
            assert text == "prompt"
            assert return_tensors == "np"
            assert len(images) == 1
            processor_calls.append(
                {
                    "min_image_tokens": min_image_tokens,
                    "max_image_tokens": max_image_tokens,
                    "do_image_splitting": do_image_splitting,
                }
            )
            return {
                "input_ids": np.array([[10, image_token_id, 11]], dtype=np.int64),
                "pixel_values": np.ones((1, 3, 2, 2), dtype=np.float32),
                "pixel_attention_mask": np.ones((1, 4), dtype=np.int64),
                "spatial_shapes": np.array([[1, 1]], dtype=np.int64),
            }

    class FakeSessionMeta:
        def __init__(self, name, type_name="tensor(float)", shape=None):
            self.name = name
            self.type = type_name
            self.shape = shape or [1, 1, 0, 4]

    class FakeEmbedImagesSession:
        @staticmethod
        def get_inputs():
            return [
                FakeSessionMeta("pixel_values", "tensor(float)", [1, 3, 2, 2]),
                FakeSessionMeta("pixel_attention_mask", "tensor(int64)", [1, 4]),
                FakeSessionMeta("spatial_shapes", "tensor(int64)", [1, 2]),
            ]

        @staticmethod
        def run(_output_names, feed):
            assert set(feed) == {"pixel_values", "pixel_attention_mask", "spatial_shapes"}
            return [np.array([[7.0, 8.0, 9.0, 10.0]], dtype=np.float32)]

    class FakeEmbedTokensSession:
        @staticmethod
        def get_inputs():
            return [FakeSessionMeta("input_ids", "tensor(int64)", [1, 3])]

        @staticmethod
        def run(_output_names, feed):
            input_ids = feed["input_ids"]
            if input_ids.shape[1] == 3:
                return [np.zeros((1, 3, 4), dtype=np.float32)]
            return [np.full((1, 1, 4), float(input_ids[0, 0]), dtype=np.float32)]

    class FakeDecoderSession:
        @staticmethod
        def get_inputs():
            return [
                FakeSessionMeta("inputs_embeds", "tensor(float)", [1, "sequence_length", 4]),
                FakeSessionMeta("attention_mask", "tensor(int64)", [1, "total_sequence_length"]),
                FakeSessionMeta("past_key_values.0.key", "tensor(float)", [1, 1, "past_sequence_length", 4]),
            ]

        @staticmethod
        def get_outputs():
            return [
                FakeSessionMeta("logits", "tensor(float)", [1, "sequence_length", 128]),
                FakeSessionMeta("present.0.key", "tensor(float)", [1, 1, "present_sequence_length", 4]),
            ]

        @staticmethod
        def run(_output_names, feed):
            decoder_feeds.append(feed)
            logits = np.zeros((1, feed["inputs_embeds"].shape[1], 128), dtype=np.float32)
            if len(decoder_feeds) == 1:
                logits[0, -1, 42] = 1.0
                present = np.ones((1, 1, 1, 4), dtype=np.float32)
            else:
                logits[0, -1, eos_token_id] = 1.0
                present = np.ones((1, 1, 2, 4), dtype=np.float32)
            return [logits, present]

    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={
            "lfm_vl_local": {
                "model_id": "LiquidAI/LFM2.5-VL-1.6B-ONNX",
                "max_new_tokens": 4,
                "min_image_tokens": 64,
                "max_image_tokens": 256,
                "do_image_splitting": True,
            }
        },
        args=SimpleNamespace(vlm_image_model="lfm_vl_local"),
    )
    provider = LFMVLLocalProvider(ctx)
    monkeypatch.setattr(
        provider,
        "_get_or_load_model",
        lambda: {
            "processor": FakeProcessor(),
            "sessions": {
                "embed_images": FakeEmbedImagesSession(),
                "embed_tokens": FakeEmbedTokensSession(),
                "decoder": FakeDecoderSession(),
            },
        },
    )
    monkeypatch.setattr(provider, "_load_images", lambda _media: ["image-object"])

    media = MediaContext(
        uri=str(image_path),
        mime="image/png",
        sha256hash="",
        modality=MediaModality.IMAGE,
        file_size=image_path.stat().st_size,
    )
    result = provider.attempt(media, PromptContext(system="system", user="describe"))

    np.testing.assert_allclose(decoder_feeds[0]["inputs_embeds"][0, 1], np.array([7.0, 8.0, 9.0, 10.0], dtype=np.float32))
    assert decoder_feeds[1]["attention_mask"].shape == (1, 4)
    assert processor_calls == [
        {
            "min_image_tokens": 64,
            "max_image_tokens": 256,
            "do_image_splitting": True,
        }
    ]
    assert result.raw == "decoded:42,2"
