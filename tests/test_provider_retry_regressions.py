"""OpenAI-compatible retries with the real SDK and an offline HTTP transport."""

import io
import json
from copy import deepcopy

import httpx
import openai
import pytest
from PIL import Image
from rich.console import Console

from module.providers.base import CaptionStatus, MediaContext, MediaModality, PromptContext, ProviderContext
from module.providers.cloud_vlm.openai_compatible import OpenAICompatibleProvider
from tests.provider_v2_helpers import make_provider_args


def _error(status, message, *, param=None, code=None):
    error_type = "server_error" if status >= 500 else "rate_limit_error" if status == 429 else "invalid_request_error"
    return (status, {"error": {"message": message, "type": error_type, "param": param, "code": code}})


def _completion(content):
    return (200, {
        "id": "offline-completion", "object": "chat.completion", "created": 0, "model": "offline-model",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })


UNSUPPORTED_JSON = _error(
    400, "response_format of type json_object is not supported with this model.",
    param="response_format", code="unsupported_value",
)


def _provider(tmp_path, **overrides):
    source = tmp_path / "input.jpg"
    Image.new("RGB", (32, 32), "red").save(source)
    args = make_provider_args(
        openai_api_key="offline-placeholder", openai_base_url="https://offline.invalid/v1",
        openai_model_name="offline-model", openai_json_mode=True, max_retries=3, wait_time=0, mode="all",
    )
    vars(args).update(overrides)
    provider = OpenAICompatibleProvider(ProviderContext(
        console=Console(file=io.StringIO()), args=args,
        config={"prompts": {"image_prompt": "Describe the image."}},
    ))
    return provider, source


def _endpoint(monkeypatch, responses, *, validate_json=False):
    from module.providers import utils as provider_utils

    requests = []
    clients = []
    client_options = []
    responses = iter(responses)
    original_client = openai.OpenAI

    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if validate_json and payload.get("response_format") == {"type": "json_object"}:
            text = []
            for message in payload["messages"]:
                content = message.get("content")
                if isinstance(content, str):
                    text.append(content)
                elif isinstance(content, list):
                    text.extend(part["text"] for part in content if part.get("type") == "text")
            if not any("json" in value.lower() for value in text):
                status, error = _error(400, "Messages must contain the word JSON", param="messages")
                return httpx.Response(status, json=error, request=request)
        response = next(responses)
        if response == "timeout":
            raise httpx.ReadTimeout("Timed out", request=request)
        if response == "connection":
            raise httpx.ConnectError("Connection refused", request=request)
        status, payload = response
        return httpx.Response(status, json=payload, request=request)

    def client_factory(**kwargs):
        client_options.append(dict(kwargs))
        # The endpoint must expose each failed request to the provider, not SDK backoff.
        kwargs.setdefault("max_retries", 0)
        client = original_client(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        clients.append(client)
        return client

    monkeypatch.setattr(openai, "OpenAI", client_factory)
    monkeypatch.setattr(provider_utils.time, "sleep", lambda seconds: None)
    return requests, clients, client_options


@pytest.mark.parametrize("error", [
    _error(502, "Upstream service temporarily unavailable"),
    _error(503, "Service Unavailable"),
    _error(429, "Rate limit exceeded"),
    _error(503, "response_format json_object is not supported while the service is unavailable"),
    "timeout",
    "connection",
])
def test_openai_transient_failure_uses_all_configured_attempts_without_json_downgrade(monkeypatch, tmp_path, error):
    requests, _, _ = _endpoint(monkeypatch, [error, error, _completion("A red square.")])
    provider, source = _provider(tmp_path)

    result = provider.execute(str(source), "image/jpeg", "offline-hash")

    assert result.status is CaptionStatus.SUCCESS
    assert result.raw == "A red square."
    assert result.is_persistable
    assert len(requests) == 3
    assert all(request["response_format"] == {"type": "json_object"} for request in requests)
    assert all(request["messages"] == requests[0]["messages"] for request in requests)


def test_openai_transient_exhaustion_is_failed_not_empty_success(monkeypatch, tmp_path):
    requests, _, _ = _endpoint(monkeypatch, [_error(503, "Service Unavailable")] * 3)
    provider, source = _provider(tmp_path)

    result = provider.execute(str(source), "image/jpeg", "offline-hash")

    assert result.status is CaptionStatus.FAILED
    assert result.metadata["retry_exhausted"] is True
    assert "503" in result.error
    assert not result.is_persistable
    assert len(requests) == 3


def test_openai_explicit_json_incompatibility_can_downgrade(monkeypatch, tmp_path):
    requests, _, _ = _endpoint(monkeypatch, [UNSUPPORTED_JSON, _completion("A red square.")])
    provider, source = _provider(tmp_path)

    result = provider.execute(str(source), "image/jpeg", "offline-hash")

    assert result.raw == "A red square."
    assert result.is_persistable
    assert len(requests) == 2
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in requests[1]


def test_openai_failed_format_fallback_returns_to_unified_retry(monkeypatch, tmp_path):
    requests, _, _ = _endpoint(monkeypatch, [
        UNSUPPORTED_JSON, _error(503, "Service Unavailable"),
        UNSUPPORTED_JSON, _completion("A red square."),
    ])
    provider, source = _provider(tmp_path)

    result = provider.execute(str(source), "image/jpeg", "offline-hash")

    assert result.status is CaptionStatus.SUCCESS
    assert result.raw == "A red square."
    assert len(requests) == 4
    assert ["response_format" in request for request in requests] == [True, False, True, False]


def test_openai_format_fallback_exhaustion_is_failed(monkeypatch, tmp_path):
    requests, _, _ = _endpoint(monkeypatch, [UNSUPPORTED_JSON, _error(503, "Service Unavailable")] * 3)
    provider, source = _provider(tmp_path)

    result = provider.execute(str(source), "image/jpeg", "offline-hash")

    assert result.status is CaptionStatus.FAILED
    assert result.metadata["retry_exhausted"] is True
    assert not result.is_persistable
    assert len(requests) == 6


@pytest.mark.parametrize("error", [
    _error(400, "max_tokens is not supported with this model", param="max_tokens"),
    _error(401, "Unauthorized"),
    _error(400, "Messages must contain the word JSON when response_format is json_object", param="messages"),
])
def test_openai_other_client_errors_are_not_retried_as_format_downgrades(monkeypatch, tmp_path, error):
    requests, _, _ = _endpoint(monkeypatch, [error, _completion("Should not be accepted.")])
    provider, source = _provider(tmp_path)

    with pytest.raises(openai.APIStatusError):
        provider.execute(str(source), "image/jpeg", "offline-hash")

    assert len(requests) == 1


@pytest.mark.parametrize(("response", "expected", "parsed"), [
    ("A red square.", "A red square.", None),
    ('{"short": "red square", "long": "A red square."}', None,
     {"short_description": "red square", "long_description": "A red square."}),
])
def test_openai_success_preserves_text_and_structured_captions(monkeypatch, tmp_path, response, expected, parsed):
    requests, _, _ = _endpoint(monkeypatch, [_completion(response)])
    provider, source = _provider(tmp_path)

    result = provider.execute(str(source), "image/jpeg", "offline-hash")

    assert result.status is CaptionStatus.SUCCESS
    assert result.parsed == parsed
    assert result.is_persistable
    if parsed is None:
        assert result.raw == expected
    else:
        assert json.loads(result.raw) == parsed
    assert len(requests) == 1


def test_openai_plaintext_mode_still_retries_transient_failures(monkeypatch, tmp_path):
    requests, _, _ = _endpoint(monkeypatch, [_error(503, "Service Unavailable"), _completion("A red square.")])
    provider, source = _provider(tmp_path, openai_json_mode=False)

    result = provider.execute(str(source), "image/jpeg", "offline-hash")

    assert result.raw == "A red square."
    assert result.is_persistable
    assert len(requests) == 2
    assert all("response_format" not in request for request in requests)


@pytest.mark.parametrize("responses", [
    [_completion("A red square.")],
    [_error(503, "Service Unavailable"), _completion("A red square.")],
    [UNSUPPORTED_JSON, _completion("A red square.")],
    [_error(503, "Service Unavailable")] * 3,
])
def test_openai_attempts_close_clients_and_disable_nested_sdk_retries(monkeypatch, tmp_path, responses):
    _, clients, options = _endpoint(monkeypatch, responses)
    provider, source = _provider(tmp_path)

    provider.execute(str(source), "image/jpeg", "offline-hash")

    assert all(client.is_closed() for client in clients)
    assert all(option.get("max_retries") == 0 for option in options)


@pytest.mark.parametrize("system", ["Describe only visible details.", ""])
@pytest.mark.parametrize("media_kind", ["image", "pair", "video"])
def test_json_mode_supplies_instruction_for_plain_prompts(monkeypatch, tmp_path, system, media_kind):
    requests, clients, _ = _endpoint(monkeypatch, [_completion('{"description": "A red square."}')], validate_json=True)
    provider, source = _provider(tmp_path, mode="long", pair_dir="pairs" if media_kind == "pair" else "")
    prompts = PromptContext(system=system, user="\u63cf\u8ff0\u8fd9\u4e2a\u56fe\u6807\u3002")
    media = MediaContext(
        uri=str(source), mime="video/mp4" if media_kind == "video" else "image/jpeg", sha256hash="offline",
        modality=MediaModality.VIDEO if media_kind == "video" else MediaModality.IMAGE,
        blob="json", pair_blob="json" if media_kind == "pair" else None,
    )
    original = provider._build_messages(media, prompts)

    result = provider.attempt(media, prompts)

    assert result.text == "A red square."
    assert len(requests) == 1
    instruction = requests[0]["messages"][0]["content"]
    assert "json" in instruction.lower()
    assert "description" in instruction
    assert requests[0]["messages"][1:] == original[1:]
    assert prompts.system == system
    assert all(client.is_closed() for client in clients)


@pytest.mark.parametrize("prompts", [
    PromptContext(system='Return JSON with a "custom_caption" field.', user="Describe the image."),
    PromptContext(system="Describe only visible details.", user='Return JsOn with a "custom_caption" field.'),
])
def test_json_mode_preserves_existing_json_contract(monkeypatch, tmp_path, prompts):
    requests, _, _ = _endpoint(monkeypatch, [_completion('{"custom_caption": "A red square."}')], validate_json=True)
    provider, source = _provider(tmp_path)
    media = MediaContext(str(source), "image/jpeg", "offline", MediaModality.IMAGE, blob="image-data")
    original = provider._build_messages(media, prompts)

    result = provider.attempt(media, prompts)

    assert result.parsed == {"custom_caption": "A red square."}
    assert requests[0]["messages"] == original


@pytest.mark.parametrize("json_mode", [False, True])
def test_plaintext_and_format_fallback_keep_original_messages(monkeypatch, tmp_path, json_mode):
    responses = [UNSUPPORTED_JSON, _completion("A red square.")] if json_mode else [_completion("A red square.")]
    requests, _, _ = _endpoint(monkeypatch, responses, validate_json=True)
    provider, source = _provider(tmp_path, openai_json_mode=json_mode)
    prompts = PromptContext(system="Describe visible details.", user="Describe the image.")
    media = MediaContext(str(source), "image/jpeg", "offline", MediaModality.IMAGE, blob="image-data")
    original = provider._build_messages(media, prompts)

    result = provider.attempt(media, prompts)

    assert result.raw == "A red square."
    assert len(requests) == (2 if json_mode else 1)
    assert requests[-1]["messages"] == original
    assert "response_format" not in requests[-1]


@pytest.mark.parametrize("system", [
    [],
    [{"role": "system", "content": "Describe visible details."}],
    [{"role": "system", "content": [{"type": "text", "text": "Describe visible details."}]}],
])
def test_json_instruction_does_not_mutate_reused_messages(monkeypatch, tmp_path, system):
    requests, _, _ = _endpoint(monkeypatch, [_completion("Caption"), _completion("Caption")], validate_json=True)
    provider, source = _provider(tmp_path)
    messages = [*system, {"role": "user", "content": "Describe the image."}]
    original = deepcopy(messages)
    monkeypatch.setattr(provider, "_build_messages", lambda *_: messages)
    media = MediaContext(str(source), "image/jpeg", "offline", MediaModality.IMAGE, blob="image-data")

    for _ in range(2):
        provider.attempt(media, PromptContext(system="", user=""))

    assert messages == original
    assert requests[0]["messages"] == requests[1]["messages"]
