import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from tests.provider_v2_helpers import make_provider_args


def _make_mimo_provider(**arg_overrides):
    from providers.base import ProviderContext
    from providers.registry import get_registry

    provider_cls = get_registry().get_provider("mimo")
    return provider_cls(
        ProviderContext(
            console=Console(file=io.StringIO()),
            config={"mimo": {}, "prompts": {}},
            args=make_provider_args(mimo_api_key="test-key", **arg_overrides),
        )
    )


def test_mimo_provider_uses_openai_compatible_client_and_completion_token_param():
    from providers.base import MediaContext, MediaModality, PromptContext, ProviderContext
    from providers.registry import get_registry

    provider_cls = get_registry().get_provider("mimo")
    ctx = ProviderContext(
        console=Console(file=io.StringIO()),
        config={"mimo": {"thinking": "disabled", "max_completion_tokens": 4096}, "prompts": {}},
        args=make_provider_args(
            mimo_api_key="test-key",
            mimo_base_url="https://token-plan-sgp.xiaomimimo.com/v1",
            mimo_model_path="mimo-v2.5",
            mode="long",
            max_retries=1,
            wait_time=0.01,
        ),
    )
    provider = provider_cls(ctx)

    mock_openai_cls = MagicMock()
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    with (
        patch("openai.OpenAI", mock_openai_cls),
        patch("module.providers.cloud_vlm.mimo.attempt_kimi_vl", return_value='{"long_description": "ok"}') as mock_attempt,
    ):
        media = MediaContext(
            uri="/fake.jpg",
            mime="image/jpeg",
            sha256hash="",
            modality=MediaModality.IMAGE,
            blob="base64data",
            pixels=object(),
        )
        result = provider.attempt(media, PromptContext(system="Only output one long paragraph.", user="Describe."))

    assert result.parsed == {"long_description": "ok"}
    assert mock_openai_cls.call_args.kwargs["api_key"] == "test-key"
    assert mock_openai_cls.call_args.kwargs["base_url"] == "https://token-plan-sgp.xiaomimimo.com/v1"
    assert mock_attempt.call_args.kwargs["model_path"] == "mimo-v2.5"
    assert mock_attempt.call_args.kwargs["thinking"] == "disabled"
    assert mock_attempt.call_args.kwargs["max_tokens"] == 4096
    assert mock_attempt.call_args.kwargs["max_tokens_param"] == "max_completion_tokens"
    assert mock_attempt.call_args.kwargs["temperature"] == 1.0


def test_mimo_rejects_video_that_would_exceed_base64_limit_before_reading_file():
    from module.providers.cloud_vlm.mimo import MIMO_BASE64_VIDEO_LIMIT_BYTES
    from providers.base import MediaContext, MediaModality, PromptContext

    provider = _make_mimo_provider()
    media = MediaContext(
        uri="/does/not/exist.mp4",
        mime="video/mp4",
        sha256hash="",
        modality=MediaModality.VIDEO,
        file_size=MIMO_BASE64_VIDEO_LIMIT_BYTES,
    )

    with pytest.raises(ValueError, match="MiMo video base64 limit exceeded"):
        provider._build_messages(media, PromptContext(system="sys", user="usr"))


def test_mimo_base64_limit_error_does_not_retry():
    provider = _make_mimo_provider(wait_time=0.01, max_retries=2)
    cfg = provider.get_retry_config()

    assert cfg.classify_error(ValueError("MiMo video base64 limit exceeded: encoded input would be 51.0 MB")) is None


def test_mimo_high_risk_rejection_is_converted_to_skip_result():
    from providers.base import CaptionResult, MediaContext, MediaModality

    provider = _make_mimo_provider()
    media = MediaContext(uri="/fake.jpg", mime="image/jpeg", sha256hash="", modality=MediaModality.IMAGE)

    result = provider.post_validate(
        CaptionResult(raw="The request was rejected because it was considered high risk"),
        media,
        provider.ctx.args,
    )

    assert result.raw == ""
    assert result.parsed is None
    assert result.metadata["provider"] == "mimo"
    assert result.metadata["skip_reason"] == "mimo_high_risk_rejection"


def test_mimo_does_not_skip_normal_caption_with_high_risk_words():
    from providers.base import CaptionResult, MediaContext, MediaModality

    provider = _make_mimo_provider()
    media = MediaContext(uri="/fake.jpg", mime="image/jpeg", sha256hash="", modality=MediaModality.IMAGE)
    result = CaptionResult(raw="A warning label says high risk near a machine.")

    assert provider.post_validate(result, media, provider.ctx.args) is result


def test_mimo_prompt_resolver_reuses_kimi_image_prompt_fallbacks():
    from providers.resolver import PromptResolver

    resolver = PromptResolver(
        {
            "prompts": {
                "image_system_prompt": "base system",
                "image_prompt": "base user",
                "kimi_image_system_prompt": "kimi system",
                "kimi_image_prompt": "kimi user",
            }
        },
        "mimo",
    )

    prompts = resolver.resolve("image/jpeg", SimpleNamespace(pair_dir=""))

    assert prompts.system == "kimi system"
    assert prompts.user == "kimi user"
