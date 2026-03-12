import io
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
