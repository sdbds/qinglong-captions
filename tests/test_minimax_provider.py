import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console


def test_m31_preview_forces_reasoning_split_when_disabled_in_config():
    from module.providers.cloud_vlm.minimax_code import attempt_minimax_code

    client = MagicMock()
    client.chat.completions.create.return_value = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        reasoning_content=None,
                        content='{"long_description": "ok"}',
                    )
                )
            ]
        )
    ]

    with patch("utils.parse_display.display_caption_and_rate"):
        attempt_minimax_code(
            client=client,
            model_path="MiniMax-M3.1-Flash-Preview",
            messages=[{"role": "user", "content": "describe"}],
            console=Console(file=io.StringIO()),
            progress=None,
            task_id=None,
            uri="sample.jpg",
            reasoning_split=False,
        )

    assert client.chat.completions.create.call_args.kwargs["extra_body"] == {
        "reasoning_split": True,
    }


def _delta(content=None, reasoning=None, finish_reason=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=content, reasoning_content=reasoning),
        finish_reason=finish_reason,
    )])


def _attempt(provider):
    from module.providers.cloud_vlm.minimax_api import attempt_minimax
    from module.providers.cloud_vlm.minimax_code import attempt_minimax_code

    return attempt_minimax if provider == "api" else attempt_minimax_code


@pytest.mark.parametrize("provider", ["api", "code"])
@pytest.mark.parametrize("effort", [None, "low", "medium", "high", "xhigh", "max"])
@pytest.mark.parametrize("mixed_reasoning", ["", "remaining private reasoning"])
def test_m31_request_uses_supported_parameters_and_keeps_only_answer(provider, effort, mixed_reasoning):
    client = MagicMock()
    client.chat.completions.create.return_value = [
        _delta(reasoning="private reasoning"),
        _delta(content='{"long_description": "ok"}', reasoning=mixed_reasoning),
        _delta(finish_reason="stop"),
    ]
    kwargs = {} if effort is None else {"reasoning_effort": effort}
    with patch("utils.parse_display.display_caption_and_rate"):
        result = _attempt(provider)(
            client=client, model_path="MiniMax-M3.1-Flash",
            messages=[{"role": "user", "content": "describe"}],
            console=Console(file=io.StringIO()), uri="sample.jpg",
            reasoning_split=False, **kwargs,
        )

    request = client.chat.completions.create.call_args.kwargs
    assert request["extra_body"] == {"reasoning_split": True}
    assert request["reasoning_effort"] == (effort or "medium")
    assert request["max_completion_tokens"] == 8192
    assert {"temperature", "top_p", "max_tokens", "thinking"}.isdisjoint(request)
    assert result == '{"long_description": "ok"}'


@pytest.mark.parametrize("provider", ["api", "code"])
@pytest.mark.parametrize("effort", ["none", "disabled", "ultra", ""])
def test_m31_rejects_unsupported_effort_before_sending(provider, effort):
    client = MagicMock()
    with pytest.raises(ValueError, match="reasoning effort"):
        _attempt(provider)(
            client=client, model_path="MiniMax-M3.1-Flash", messages=[],
            console=Console(file=io.StringIO()), reasoning_effort=effort,
        )
    client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("provider", ["api", "code"])
@pytest.mark.parametrize("content,reason,pattern", [
    (None, "stop", "RETRY_EMPTY_CONTENT"),
    ("  ", "stop", "RETRY_EMPTY_CONTENT"),
    (None, "length", "max_completion_tokens"),
    ("partial caption", "length", "max_completion_tokens"),
])
def test_minimax_does_not_accept_empty_or_truncated_captions(provider, content, reason, pattern):
    client = MagicMock()
    client.chat.completions.create.return_value = [
        _delta(reasoning="reasoning"), _delta(content, finish_reason=reason),
    ]
    with patch("utils.parse_display.display_caption_and_rate"), pytest.raises(
        (RuntimeError, ValueError), match=pattern
    ):
        _attempt(provider)(
            client=client, model_path="MiniMax-M3.1-Flash", messages=[],
            console=Console(file=io.StringIO()),
        )


@pytest.mark.parametrize("provider", ["api", "code"])
@pytest.mark.parametrize("model", ["MiniMax-M3", "MiniMax-M2.7"])
def test_legacy_minimax_models_do_not_receive_m31_effort(provider, model):
    client = MagicMock()
    client.chat.completions.create.return_value = [_delta("caption", finish_reason="stop")]
    with patch("utils.parse_display.display_caption_and_rate"):
        result = _attempt(provider)(
            client=client, model_path=model, messages=[],
            console=Console(file=io.StringIO()), reasoning_split=False,
        )
    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == model
    assert "reasoning_effort" not in request
    if model == "MiniMax-M3":
        assert {"temperature", "top_p", "max_tokens"}.isdisjoint(request)
        assert request["max_completion_tokens"] == 8192
    else:
        assert request["temperature"] == 0.7
        assert request["max_tokens"] == 8192
    assert result == "caption"


@pytest.mark.parametrize("provider", ["api", "code"])
def test_minimax_completion_budget_is_configurable(provider):
    client = MagicMock()
    client.chat.completions.create.return_value = [_delta("caption", finish_reason="stop")]
    with patch("utils.parse_display.display_caption_and_rate"):
        _attempt(provider)(
            client=client, model_path="MiniMax-M3.1-Flash", messages=[],
            console=Console(file=io.StringIO()), max_completion_tokens=16384,
        )
    assert client.chat.completions.create.call_args.kwargs["max_completion_tokens"] == 16384


@pytest.mark.parametrize("provider", ["api", "code"])
def test_minimax_provider_forwards_its_own_config(provider):
    from module.providers.base import MediaContext, MediaModality, PromptContext, ProviderContext
    from module.providers.cloud_vlm.minimax_api import MiniMaxAPIProvider
    from module.providers.cloud_vlm.minimax_code import MiniMaxCodeProvider

    section = "minimax_api" if provider == "api" else "minimax_code"
    cls = MiniMaxAPIProvider if provider == "api" else MiniMaxCodeProvider
    model_arg = "minimax_model_path" if provider == "api" else "minimax_code_model_path"
    key_arg = "minimax_api_key" if provider == "api" else "minimax_code_api_key"
    client = MagicMock()
    client.chat.completions.create.return_value = [_delta("caption", finish_reason="stop")]
    context = ProviderContext(
        console=Console(file=io.StringIO()),
        config={section: {"reasoning_effort": "low", "max_completion_tokens": 16384}},
        args=SimpleNamespace(**{model_arg: "MiniMax-M3.1-Flash", key_arg: "test-key"}),
    )
    with patch("openai.OpenAI", return_value=client), patch("utils.parse_display.display_caption_and_rate"):
        cls(context).attempt(
            MediaContext(uri="sample.jpg", mime="image/jpeg", sha256hash="", modality=MediaModality.IMAGE, blob="data"),
            PromptContext(system="system", user="describe"),
        )
    request = client.chat.completions.create.call_args.kwargs
    assert request["reasoning_effort"] == "low"
    assert request["max_completion_tokens"] == 16384


@pytest.mark.parametrize("provider", ["api", "code"])
@pytest.mark.parametrize("config,error,network_calls", [
    ({}, "truncated", 1),
    ({"reasoning_effort": "none"}, "reasoning effort", 0),
    ({"max_completion_tokens": 0}, "positive integer", 0),
    ({"max_completion_tokens": True}, "positive integer", 0),
])
def test_minimax_terminal_errors_do_not_retry_paid_requests(provider, config, error, network_calls):
    from module.providers.base import ProviderContext
    from module.providers.cloud_vlm.minimax_api import MiniMaxAPIProvider
    from module.providers.cloud_vlm.minimax_code import MiniMaxCodeProvider
    from module.providers.utils import with_retry_impl

    cls = MiniMaxAPIProvider if provider == "api" else MiniMaxCodeProvider
    console = Console(file=io.StringIO())
    instance = cls(ProviderContext(
        console=console, config={}, args=SimpleNamespace(max_retries=3, wait_time=0),
    ))
    retry_config = instance.get_retry_config()
    client = MagicMock()
    client.chat.completions.create.return_value = [_delta(finish_reason="length")]
    with pytest.raises(ValueError, match=error):
        with_retry_impl(
            lambda: _attempt(provider)(
                client=client, model_path="MiniMax-M3.1-Flash", messages=[],
                console=console, **config,
            ),
            retry_config,
        )
    assert client.chat.completions.create.call_count == network_calls


@pytest.mark.parametrize("provider", ["api", "code"])
def test_minimax_stream_transport_error_does_not_return_partial_caption(provider):
    def broken_stream():
        yield _delta(content="partial caption")
        raise ConnectionError("connection reset")

    client = MagicMock()
    client.chat.completions.create.return_value = broken_stream()
    with pytest.raises(ConnectionError, match="connection reset"):
        _attempt(provider)(
            client=client, model_path="MiniMax-M3.1-Flash", messages=[],
            console=Console(file=io.StringIO()),
        )
