"""Exercise Kimi Code effort routing through the real SDK, without network IO."""

import io
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
import toml
from rich.console import Console

from module.providers.base import MediaContext, MediaModality, ProviderContext
from module.providers.cloud_vlm.kimi_code import KimiCodeProvider


@pytest.fixture
def captured_requests(monkeypatch):
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        chunk = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": json.dumps(
                            {
                                "short_description": "A figure.",
                                "long_description": "A figure in a pink outfit on a white background.",
                            }
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        )

    with httpx.Client(transport=httpx.MockTransport(respond), trust_env=False) as client:
        monkeypatch.setattr(openai, "OpenAI", partial(openai.OpenAI, http_client=client))
        yield requests


def _caption(tmp_path, model_path, settings, *, prompts_config=None):
    provider = KimiCodeProvider(
        ProviderContext(
            console=Console(file=io.StringIO()),
            config={"kimi_code": settings, "prompts": prompts_config or {}},
            args=SimpleNamespace(
                kimi_code_api_key="test-key",
                kimi_code_base_url="https://kimi.test/coding/v1",
                kimi_code_model_path=model_path,
                pair_dir="",
                mode="long",
                dir_name=True,
            ),
        )
    )
    media = MediaContext(
        uri=str(tmp_path / "eve lovecall (pink charm)" / "sample.jpg"),
        mime="image/jpeg",
        sha256hash="",
        modality=MediaModality.IMAGE,
        blob="base64data",
    )
    prompts = provider.resolve_prompts(media.uri, media.mime, media=media)
    return provider.attempt(media, prompts)


@pytest.mark.parametrize(
    ("model_path", "settings", "expected_model", "expected_effort"),
    [
        ("kimi-for-coding", {"reasoning_effort": "low"}, "kimi-for-coding", "low"),
        ("kimi-for-coding", {"reasoning_effort": "high"}, "kimi-for-coding", "high"),
        ("kimi-for-coding", {"reasoning_effort": "max"}, "kimi-for-coding", "max"),
        ("kimi-for-coding", {}, "kimi-for-coding", "high"),
        ("k2p5", {"reasoning_effort": "max"}, "kimi-for-coding", "max"),
        (" kimi-code ", {"reasoning_effort": "high"}, "kimi-for-coding", "high"),
        ("k3-256k", {"reasoning_effort": "high"}, "k3-256k", "high"),
    ],
)
def test_effort_reaches_http_payload_with_long_and_directory_context(
    tmp_path, captured_requests, model_path, settings, expected_model, expected_effort
):
    # A stale legacy toggle must not override the selected effort.
    result = _caption(tmp_path, model_path, {"thinking": "disabled", **settings})

    assert len(captured_requests) == 1
    body = captured_requests[0]
    assert body["model"] == expected_model
    assert body.get("reasoning_effort") == expected_effort
    assert "thinking" not in body
    assert "extra_body" not in body
    assert body["temperature"] == 1.0
    user_text = next(part["text"] for part in body["messages"][1]["content"] if part["type"] == "text")
    assert "<eve lovecall> from (pink charm)" in user_text
    assert result.parsed == {"long_description": "A figure in a pink outfit on a white background."}
    assert result.metadata["model"] == expected_model
    assert result.metadata["reasoning_effort"] == expected_effort
    assert "thinking" not in result.metadata


@pytest.mark.parametrize("thinking", ["enabled", "disabled"])
def test_k27_highspeed_still_sends_only_thinking_toggle(tmp_path, captured_requests, thinking):
    result = _caption(tmp_path, "kimi-for-coding-highspeed", {"thinking": thinking, "reasoning_effort": "max"})

    assert len(captured_requests) == 1
    body = captured_requests[0]
    assert body["model"] == "kimi-for-coding-highspeed"
    assert body["thinking"] == {"type": thinking}
    assert "reasoning_effort" not in body
    assert result.metadata["thinking"] == thinking
    assert "reasoning_effort" not in result.metadata


@pytest.mark.parametrize("effort", ["unknown", "", False, 0])
def test_k28_rejects_invalid_effort_before_http_request(tmp_path, captured_requests, effort):
    with pytest.raises(ValueError, match="reasoning effort"):
        _caption(tmp_path, "kimi-for-coding", {"reasoning_effort": effort})

    assert captured_requests == []


@pytest.mark.parametrize("config_name", ["prompts.toml", "config.toml"])
@pytest.mark.parametrize("model_path", ["kimi-for-coding", "k3-256k"])
def test_shipped_anti_tag_echo_instruction_reaches_request(tmp_path, captured_requests, config_name, model_path):
    prompts = toml.load(Path(__file__).resolve().parents[1] / "config" / config_name)["prompts"]
    _caption(tmp_path, model_path, {"reasoning_effort": "high"}, prompts_config=prompts)

    instruction = (
        "Use the supplied tags only as reference for writing natural-language captions, "
        "and never quote, enumerate, copy, or append the tag list or any tag-list heading in your output."
    )
    system_prompt = captured_requests[0]["messages"][0]["content"]
    assert system_prompt.count(instruction) == 1
    assert "###Short:" in system_prompt
    assert "###Long:" in system_prompt
