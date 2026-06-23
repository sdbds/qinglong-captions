import json
import os
import stat
import sys
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.provider_v2_helpers import make_provider_args


def _make_fake_codex(tmp_path: Path, script_body: str) -> Path:
    script = tmp_path / "fake_codex.py"
    script.write_text(script_body, encoding="utf-8")
    if os.name == "nt":
        wrapper = tmp_path / "codex.cmd"
        wrapper.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        wrapper = tmp_path / "codex"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return wrapper


def test_codex_progress_coalescer_merges_stream_event_lines():
    from module.providers.cloud_vlm.codex_subscription import _CodexProgressCoalescer

    emitted = []
    progress = _CodexProgressCoalescer(emitted.append)

    progress("Codex app-server: event turn/started")
    progress("Codex app-server: event item/started (reasoning)")
    progress("Codex app-server: event item/agentMessage/delta: {")
    progress("Codex app-server: event item/agentMessage/delta: }")
    assert emitted == []

    progress("Codex app-server: caption_image still waiting (15s/180s)")

    assert emitted == [
        "Codex app-server stream: turn/started | item/started (reasoning) | item/agentMessage/delta x2",
        "Codex app-server: caption_image still waiting (15s/180s)",
    ]


def test_codex_app_server_invalid_json_rpc_is_transport_failure():
    from module.providers.codex_app_server import classify_codex_app_server_failure

    assert (
        classify_codex_app_server_failure(
            "Invalid JSON-RPC line: 'SUCCESS: The process with PID 384100 has been terminated.\\n'"
        )
        == "transport"
    )


def test_codex_app_server_429_is_retryable_rate_limit():
    from module.providers.codex_app_server import CodexAppServerConfig, _coerce_sdk_exception

    exc = _coerce_sdk_exception(
        RuntimeError(
            "exceeded retry limit, last status: "
            "429 Too Many Requests, request id: 2e0b6dc2-af63-49fe-a5ca-4c5ae5aed66a"
        ),
        CodexAppServerConfig(),
        stage="request",
    )

    assert exc.kind == "rate_limited"
    assert exc.retryable is True


def test_codex_app_server_closed_client_and_pool_errors_are_retryable(monkeypatch, tmp_path):
    from module.providers import codex_app_server
    from module.providers.codex_app_server import (
        CodexAppServerCaptionClient,
        CodexAppServerClientPool,
        CodexAppServerConfig,
        CodexAppServerError,
    )

    class FakeSdkClient:
        def close(self):
            pass

    monkeypatch.setattr(codex_app_server, "_create_sdk_client", lambda _config: FakeSdkClient())
    config = CodexAppServerConfig(auth_mode="chatgpt", isolated_cwd=str(tmp_path / "work"))

    client = CodexAppServerCaptionClient(config)
    client.close()
    with pytest.raises(CodexAppServerError) as client_exc:
        client.caption_image(image_path=tmp_path / "image.png", prompt="caption")

    assert client_exc.value.kind == "closed"
    assert client_exc.value.retryable is True

    pool = CodexAppServerClientPool(config, 1)
    pool.close()
    with pytest.raises(CodexAppServerError) as pool_exc:
        pool.caption_image(image_path=tmp_path / "image.png", prompt="caption")

    assert pool_exc.value.kind == "closed"
    assert pool_exc.value.retryable is True


def test_codex_sdk_stdout_noise_filter_skips_windows_taskkill_success():
    from module.providers.codex_app_server import load_openai_codex_sdk

    load_openai_codex_sdk()

    try:
        from openai_codex.client import AppServerClient
    except ImportError:
        pytest.skip("PyPI openai-codex no longer exposes the old AppServerClient stdout reader")

    client = AppServerClient()
    client._proc = SimpleNamespace(
        stdout=io.StringIO(
            "SUCCESS: The process with PID 209968 (child process of PID 66372) has been terminated.\n"
            '{"id":"ok","result":{}}\n'
        )
    )

    assert client._read_message() == {"id": "ok", "result": {}}


def test_codex_sdk_stdout_noise_filter_skips_windows_taskkill_success_for_codex_client():
    from module.providers.codex_app_server import load_openai_codex_sdk

    load_openai_codex_sdk()

    try:
        from openai_codex.client import CodexClient
    except ImportError:
        pytest.skip("PyPI openai-codex does not expose the current CodexClient stdout reader")

    client = CodexClient()
    client._proc = SimpleNamespace(
        stdout=io.StringIO(
            "SUCCESS: The process with PID 209968 (child process of PID 66372) has been terminated.\n"
            '{"id":"ok","result":{}}\n'
        )
    )

    assert client._read_message() == {"id": "ok", "result": {}}


def test_codex_subscription_default_timeout_is_60_seconds():
    from module.providers.codex_app_server import CodexAppServerConfig
    from module.providers.codex_exec import CodexExecConfig

    assert CodexAppServerConfig().timeout == 60.0
    assert CodexExecConfig().timeout == 60.0
    assert make_provider_args().codex_timeout == 60.0


def test_codex_exec_command_uses_argument_list(tmp_path):
    from module.providers.codex_exec import CodexExecConfig, build_codex_exec_command

    cwd = tmp_path / "work"
    cwd.mkdir()
    command = build_codex_exec_command(
        CodexExecConfig(command="codex-test-bin", model="gpt-5.4", isolated_cwd=str(cwd)),
        image_path=tmp_path / "image one.png",
        prompt="return json",
        schema_path=tmp_path / "schema.json",
        output_path=tmp_path / "out.json",
    )

    assert command[0:4] == ["codex-test-bin", "-c", 'model_reasoning_effort="none"', "exec"]
    assert "--image" in command
    assert str((tmp_path / "image one.png").resolve()) in command
    assert command[-1] == "-"
    assert "return json" not in command
    assert all(" " not in item or item.endswith("image one.png") for item in command)


def test_codex_exec_command_can_set_fast_service_tier(tmp_path):
    from module.providers.codex_exec import CodexExecConfig, build_codex_exec_command

    cwd = tmp_path / "work"
    cwd.mkdir()
    command = build_codex_exec_command(
        CodexExecConfig(command="codex-test-bin", service_tier="fast", isolated_cwd=str(cwd)),
        image_path=tmp_path / "image.png",
        prompt="return json",
        schema_path=tmp_path / "schema.json",
        output_path=tmp_path / "out.json",
    )

    assert command[0:6] == [
        "codex-test-bin",
        "-c",
        'model_reasoning_effort="none"',
        "-c",
        'service_tier="fast"',
        "exec",
    ]


def test_codex_exec_command_can_set_reasoning_effort(tmp_path):
    from module.providers.codex_exec import CodexExecConfig, build_codex_exec_command

    cwd = tmp_path / "work"
    cwd.mkdir()
    command = build_codex_exec_command(
        CodexExecConfig(command="codex-test-bin", reasoning_effort="minimal", isolated_cwd=str(cwd)),
        image_path=tmp_path / "image.png",
        prompt="return json",
        schema_path=tmp_path / "schema.json",
        output_path=tmp_path / "out.json",
    )

    assert command[0:4] == ["codex-test-bin", "-c", 'model_reasoning_effort="minimal"', "exec"]


def test_codex_exec_env_clears_api_keys_and_keeps_codex_home(tmp_path):
    from module.providers.codex_exec import build_codex_exec_env

    env = build_codex_exec_env(
        {
            "OPENAI_API_KEY": "sk-api",
            "CODEX_API_KEY": "sk-codex",
            "HTTPS_PROXY": "http://proxy.local",
        },
        codex_home=str(tmp_path / "codex-home"),
    )

    assert "OPENAI_API_KEY" not in env
    assert "CODEX_API_KEY" not in env
    assert env["HTTPS_PROXY"] == "http://proxy.local"
    assert env["CODEX_HOME"].endswith("codex-home")


def test_parse_codex_caption_output_accepts_fenced_json():
    from module.providers.codex_exec import parse_codex_caption_output

    parsed = parse_codex_caption_output(
        """```json
{"short_description":"short","long_description":"long","tags":"a, b","rating":"general","confidence":0.8}
```"""
    )

    assert parsed["short_description"] == "short"
    assert parsed["long_description"] == "long"
    assert parsed["tags"] == ["a", "b"]
    assert parsed["confidence"] == 0.8
    assert parsed["scores"] == {}
    assert parsed["average_score"] == 0


def test_codex_caption_schema_supports_gemini_rate_scores():
    from module.providers.codex_schema import CODEX_CAPTION_SCHEMA, CODEX_OVERALL_SCORE_DIMENSION, CODEX_SCORE_DIMENSIONS

    required = set(CODEX_CAPTION_SCHEMA["required"])

    assert {"scores", "average_score"} <= required
    assert "total_score" not in required
    assert "total_score" not in CODEX_CAPTION_SCHEMA["properties"]
    assert "description" not in CODEX_CAPTION_SCHEMA["properties"]
    assert CODEX_CAPTION_SCHEMA["properties"]["scores"]["additionalProperties"] is False
    assert set(CODEX_CAPTION_SCHEMA["properties"]["scores"]["required"]) == set(CODEX_SCORE_DIMENSIONS)
    assert CODEX_OVERALL_SCORE_DIMENSION in CODEX_SCORE_DIMENSIONS
    for dimension in CODEX_SCORE_DIMENSIONS:
        assert dimension in CODEX_CAPTION_SCHEMA["properties"]["scores"]["properties"]


def test_parse_codex_caption_output_preserves_rate_scores():
    from module.providers.codex_exec import parse_codex_caption_output

    parsed = parse_codex_caption_output(
        json.dumps(
            {
                "short_description": "short",
                "long_description": "long",
                "tags": ["tag"],
                "rating": "questionable",
                "confidence": 0.8,
                "scores": {
                    "Costume & Makeup & Prop Presentation/Accuracy": 8,
                    "Lighting & Mood": 9.5,
                    "Overall Impact & Uniqueness": 8.75,
                },
                "average_score": 8.75,
            }
        )
    )

    assert parsed["scores"]["Costume & Makeup & Prop Presentation/Accuracy"] == 8
    assert parsed["scores"]["Lighting & Mood"] == 9.5
    assert parsed["average_score"] == 8.75


def test_parse_codex_caption_output_defaults_average_score_to_overall_score():
    from module.providers.codex_exec import parse_codex_caption_output

    parsed = parse_codex_caption_output(
        json.dumps(
            {
                "short_description": "short",
                "long_description": "long",
                "tags": ["tag"],
                "rating": "general",
                "confidence": 0.8,
                "scores": {
                    "Costume & Makeup & Prop Presentation/Accuracy": 8,
                    "Overall Impact & Uniqueness": 9,
                },
            }
        )
    )

    assert parsed["average_score"] == 9


def test_codex_caption_prompt_mentions_score_schema_contract():
    from module.providers.codex_schema import build_codex_caption_prompt

    prompt = build_codex_caption_prompt(system_prompt="Rate each dimension.", user_prompt="Give scores.")

    assert "scores and use the Overall Impact & Uniqueness score as average_score" in prompt
    assert "Fill short_description, long_description, tags, rating, confidence, scores, and average_score" in prompt
    assert "total_score" not in prompt


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        ("Missing optional dependency @openai/codex-linux-x64", "environment"),
        ("Codex is not logged in. Please sign in.", "auth"),
        ("You've reached your Codex subscription usage limit.", "usage_limit"),
        ("request timed out", "timeout"),
    ],
)
def test_classify_codex_failure(message, kind):
    from module.providers.codex_exec import classify_codex_failure

    assert classify_codex_failure(message) == kind


def test_run_codex_exec_caption_uses_output_last_message_and_clears_api_keys(tmp_path):
    from module.providers.codex_exec import (
        CodexExecConfig,
        run_codex_exec_caption,
        write_default_caption_schema,
    )

    fake_codex = _make_fake_codex(
        tmp_path,
        """
import json
import os
import sys
from pathlib import Path

prompt = sys.stdin.read()
if prompt != "caption":
    sys.exit(43)
if os.environ.get("OPENAI_API_KEY") or os.environ.get("CODEX_API_KEY"):
    sys.exit(42)
output = Path(sys.argv[sys.argv.index("--output-last-message") + 1])
output.write_text(json.dumps({
    "short_description": "short",
    "long_description": "long",
    "tags": ["tag"],
    "rating": "general",
    "confidence": 0.9,
}), encoding="utf-8")
print("progress noise")
""",
    )
    work = tmp_path / "work"
    work.mkdir()
    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    schema = write_default_caption_schema(tmp_path / "schema.json")
    output = tmp_path / "out.json"

    result = run_codex_exec_caption(
        CodexExecConfig(command=str(fake_codex), isolated_cwd=str(work), timeout=10),
        image_path=image,
        prompt="caption",
        schema_path=schema,
        output_path=output,
        env={"OPENAI_API_KEY": "sk-api", "CODEX_API_KEY": "sk-codex"},
    )

    assert result.raw.startswith("{")
    assert result.parsed["long_description"] == "long"
    assert result.stdout.strip() == "progress noise"


def test_provider_route_requires_explicit_codex_flag():
    from module.providers.registry import get_registry

    reg = get_registry()

    assert reg.find_provider(make_provider_args(), "image/jpeg") is None
    provider = reg.find_provider(make_provider_args(codex_subscription=True), "image/jpeg")
    assert provider is not None
    assert provider.name == "codex_subscription"


def test_codex_subscription_ignores_default_document_image_flag_without_ocr_route():
    from module.providers.registry import get_registry

    reg = get_registry()

    provider = reg.find_provider(
        make_provider_args(codex_subscription=True, document_image=True),
        "image/jpeg",
    )
    assert provider is not None
    assert provider.name == "codex_subscription"


def test_codex_subscription_does_not_steal_explicit_ocr_or_vlm_routes():
    from module.providers.registry import get_registry

    reg = get_registry()

    provider = reg.find_provider(
        make_provider_args(codex_subscription=True, ocr_model="paddle_ocr", document_image=True),
        "image/jpeg",
    )
    assert provider is not None
    assert provider.name == "paddle_ocr"

    original_provider = reg._providers.get("moondream")
    original_failure = reg._import_failures.get("moondream")

    class FakeMoondreamProvider:
        name = "moondream"

        @classmethod
        def can_handle(cls, args, mime):
            return getattr(args, "vlm_image_model", "") == "moondream" and mime.startswith("image")

    try:
        reg.register("moondream", FakeMoondreamProvider)
        provider = reg.find_provider(
            make_provider_args(codex_subscription=True, vlm_image_model="moondream"),
            "image/jpeg",
        )
        assert provider is not None
        assert provider.name == "moondream"
    finally:
        if original_provider is None:
            reg._providers.pop("moondream", None)
        else:
            reg._providers["moondream"] = original_provider
        if original_failure is None:
            reg._import_failures.pop("moondream", None)
        else:
            reg._import_failures["moondream"] = original_failure


def test_codex_app_server_uses_thread_and_turn_payloads_without_api_key_env(monkeypatch, tmp_path):
    from module.providers.codex_app_server import CodexAppServerConfig, caption_image_with_app_server

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    captured = {}

    class FakeClient:
        def __init__(self):
            self.calls = []
            self.api_key = ""

        def login_api_key(self, api_key):
            self.api_key = api_key

        def login_chatgpt(self):
            self.calls.append(("login_chatgpt", {}))

        def thread_start(self, **payload):
            self.calls.append(("thread_start", payload))
            return FakeThread()

    class FakeThread:
        id = "thread-1"

        def run(self, input, *, model=None, service_tier=None, effort=None, output_schema=None):
            captured["input"] = input
            captured["model"] = model
            captured["service_tier"] = service_tier
            captured["effort"] = effort
            captured["output_schema"] = output_schema
            return SimpleNamespace(
                id="turn-1",
                final_response=json.dumps(
                    {
                        "short_description": "short",
                        "long_description": "long",
                        "tags": ["tag"],
                        "rating": "general",
                        "confidence": 0.7,
                    }
                ),
            )

    fake = FakeClient()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-be-used")

    progress_events = []
    result = caption_image_with_app_server(
        CodexAppServerConfig(auth_mode="chatgpt", service_tier="fast", isolated_cwd=str(tmp_path / "work")),
        image_path=image,
        prompt="caption",
        output_schema={"type": "object"},
        client_factory=lambda _config: fake,
        progress_callback=progress_events.append,
    )

    assert fake.api_key == ""
    assert any("acquiring client" in event for event in progress_events)
    assert any("starting ephemeral thread" in event for event in progress_events)
    assert any("turn completed" in event for event in progress_events)
    thread_call = [payload for name, payload in fake.calls if name == "thread_start"][0]
    assert thread_call["config"] == {
        "tools": {},
    }
    assert captured["model"] == "gpt-5.4"
    assert captured["output_schema"] == {"type": "object"}
    assert captured["service_tier"] == "fast"
    assert captured["effort"] == "none"
    assert getattr(captured["input"][0], "text") == "caption"
    assert getattr(captured["input"][1], "path") == str(image.resolve())
    assert result.parsed["long_description"] == "long"


def test_codex_app_server_explicit_api_key_mode_uses_explicit_key(tmp_path):
    from module.providers.codex_app_server import CodexAppServerConfig, caption_image_with_app_server

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")

    class FakeClient:
        def __init__(self):
            self.api_key = ""

        def login_api_key(self, api_key):
            self.api_key = api_key

        def thread_start(self, **_payload):
            return FakeThread()

    class FakeThread:
        id = "thread-1"

        def run(self, _input):
            return SimpleNamespace(
                id="turn-1",
                parsed={
                    "short_description": "short",
                    "long_description": "long",
                    "tags": ["tag"],
                    "rating": "general",
                    "confidence": 0.7,
                },
            )

    fake = FakeClient()
    result = caption_image_with_app_server(
        CodexAppServerConfig(auth_mode="api_key", api_key="sk-explicit", isolated_cwd=str(tmp_path / "work")),
        image_path=image,
        prompt="caption",
        client_factory=lambda _config: fake,
    )

    assert fake.api_key == "sk-explicit"
    assert result.metadata["auth_mode"] == "api_key"


def test_codex_app_server_env_clears_api_keys_unless_explicit_api_key_mode(tmp_path):
    from module.providers.codex_app_server import CodexAppServerConfig, build_codex_app_server_env

    base_env = {
        "OPENAI_API_KEY": "sk-env",
        "CODEX_API_KEY": "codex-env",
        "PATH": "demo-path",
    }

    subscription_env = build_codex_app_server_env(
        CodexAppServerConfig(auth_mode="chatgpt", codex_home=str(tmp_path / "codex-home")),
        base_env=base_env,
    )
    assert "OPENAI_API_KEY" not in subscription_env
    assert "CODEX_API_KEY" not in subscription_env
    assert subscription_env["PATH"] == "demo-path"
    assert subscription_env["CODEX_HOME"].endswith("codex-home")
    assert subscription_env["CODEX_LOG_LEVEL"] == "ERROR"
    assert "RUST_LOG" not in subscription_env

    noisy_env = build_codex_app_server_env(
        CodexAppServerConfig(auth_mode="chatgpt"),
        base_env={**base_env, "CODEX_LOG_LEVEL": "TRACE", "RUST_LOG": "trace"},
    )
    assert noisy_env["CODEX_LOG_LEVEL"] == "ERROR"
    assert "RUST_LOG" not in noisy_env

    api_key_env = build_codex_app_server_env(
        CodexAppServerConfig(auth_mode="api_key", api_key="sk-explicit"),
        base_env=base_env,
    )
    assert api_key_env["OPENAI_API_KEY"] == "sk-explicit"
    assert "CODEX_API_KEY" not in api_key_env
    assert api_key_env["CODEX_LOG_LEVEL"] == "ERROR"
    assert "RUST_LOG" not in api_key_env

    api_key_env_from_source = build_codex_app_server_env(
        CodexAppServerConfig(auth_mode="api_key"),
        base_env=base_env,
    )
    assert api_key_env_from_source["OPENAI_API_KEY"] == "sk-env"


def test_codex_app_server_disable_mcp_overrides_do_not_materialize_server_tables(tmp_path):
    from module.providers.codex_app_server import (
        CODEX_CAPTION_DISABLE_MCP_CONFIG_OVERRIDES,
        build_codex_disable_mcp_config_overrides,
    )

    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        """
[mcp_servers.node_repl]
command = "node_repl"

[mcp_servers.local-tool]
command = "node"
args = ["./mcp/server.mjs", "--stdio"]

[plugins."gmail@openai-curated"]
enabled = true

[plugins."browser@openai-bundled".mcp_servers.browser]
enabled = true
""",
        encoding="utf-8",
    )

    overrides = build_codex_disable_mcp_config_overrides(str(codex_home))

    assert CODEX_CAPTION_DISABLE_MCP_CONFIG_OVERRIDES[0] == "mcp_servers.node_repl.enabled=false"
    assert "mcp_servers.node_repl.enabled=false" in overrides
    assert "mcp_servers.local-tool.enabled=false" in overrides
    assert "mcp_servers.node_repl.enabled=false" in overrides
    assert "features.plugins=false" in overrides
    assert "mcp_servers={}" not in overrides
    assert not any('"node_repl"' in override or '"local-tool"' in override for override in overrides)
    assert "plugins={}" not in overrides


def test_codex_app_server_config_passes_mcp_disable_overrides_to_sdk(tmp_path):
    from module.providers.codex_app_server import CodexAppServerConfig, _create_app_server_config

    captured = {}

    class FakeCodexConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    sdk = SimpleNamespace(CodexConfig=FakeCodexConfig)
    from module.providers.codex_app_server import CODEX_CAPTION_DISABLE_MCP_CONFIG_OVERRIDES

    overrides = CODEX_CAPTION_DISABLE_MCP_CONFIG_OVERRIDES

    _create_app_server_config(
        sdk,
        CodexAppServerConfig(
            auth_mode="chatgpt",
            isolated_cwd=str(tmp_path / "work"),
            config_overrides=overrides,
        ),
    )

    assert captured["config_overrides"] == overrides


def test_codex_app_server_supports_typed_thread_run_shape(tmp_path):
    from module.providers.codex_app_server import CodexAppServerConfig, caption_image_with_app_server

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    captured = {}

    def _input_item_value(item, key):
        if isinstance(item, dict):
            return item[key]
        return getattr(item, key)

    class FakeThread:
        id = "thread-sdk"

        def run(self, input, *, approval_mode=None, model=None, output_schema=None):
            captured["input"] = input
            captured["approval_mode"] = approval_mode
            captured["model"] = model
            captured["output_schema"] = output_schema
            return SimpleNamespace(
                id="turn-sdk",
                final_response=json.dumps(
                    {
                        "short_description": "short",
                        "long_description": "long",
                        "tags": ["tag"],
                        "rating": "general",
                        "confidence": 0.7,
                    }
                ),
            )

    class FakeClient:
        def thread_start(self, **payload):
            captured["thread_start"] = payload
            return FakeThread()

    result = caption_image_with_app_server(
        CodexAppServerConfig(auth_mode="chatgpt", isolated_cwd=str(tmp_path / "work")),
        image_path=image,
        prompt="caption",
        output_schema={"type": "object"},
        client_factory=lambda _config: FakeClient(),
    )

    assert captured["thread_start"]["ephemeral"] is True
    assert captured["thread_start"]["config"] == {
        "tools": {},
    }
    assert captured["model"] == "gpt-5.4"
    assert captured["output_schema"] == {"type": "object"}
    assert _input_item_value(captured["input"][0], "text") == "caption"
    assert _input_item_value(captured["input"][1], "path") == str(image.resolve())
    assert result.thread_id == "thread-sdk"
    assert result.turn_id == "turn-sdk"
    assert result.parsed["long_description"] == "long"


def test_codex_app_server_uses_public_thread_run(monkeypatch, tmp_path):
    from module.providers.codex_app_server import CodexAppServerConfig, caption_image_with_app_server

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    captured = {}
    progress_events = []

    class FakeThread:
        id = "thread-run"

        def run(self, input, *, approval_mode=None, model=None, output_schema=None):
            captured["input"] = input
            captured["approval_mode"] = approval_mode
            captured["model"] = model
            captured["output_schema"] = output_schema
            return SimpleNamespace(
                id="turn-run",
                final_response=json.dumps(
                    {
                        "short_description": "short",
                        "long_description": "public run",
                        "tags": ["tag"],
                        "rating": "general",
                        "confidence": 0.7,
                    }
                ),
            )

    class FakeClient:
        def thread_start(self, **_payload):
            return FakeThread()

    result = caption_image_with_app_server(
        CodexAppServerConfig(auth_mode="chatgpt", isolated_cwd=str(tmp_path / "work")),
        image_path=image,
        prompt="caption",
        output_schema={"type": "object"},
        client_factory=lambda _config: FakeClient(),
        progress_callback=progress_events.append,
    )

    assert captured["model"] == "gpt-5.4"
    assert captured["output_schema"] == {"type": "object"}
    assert any("running image turn" in event for event in progress_events)
    assert result.turn_id == "turn-run"
    assert result.parsed["long_description"] == "public run"


def test_codex_app_server_passes_service_tier_to_thread_run(monkeypatch, tmp_path):
    from module.providers.codex_app_server import CodexAppServerConfig, caption_image_with_app_server

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    captured = {}

    class FakeThread:
        id = "thread-fast"

        def run(self, input, *, model=None, service_tier=None, effort=None, output_schema=None, approval_mode=None):
            captured["input"] = input
            captured["model"] = model
            captured["service_tier"] = service_tier
            captured["effort"] = effort
            captured["output_schema"] = output_schema
            captured["approval_mode"] = approval_mode
            return SimpleNamespace(
                id="turn-fast",
                parsed={
                    "short_description": "short",
                    "long_description": "fast",
                    "tags": ["tag"],
                    "rating": "general",
                    "confidence": 0.7,
                },
            )

    class FakeClient:
        def thread_start(self, **_payload):
            return FakeThread()

    result = caption_image_with_app_server(
        CodexAppServerConfig(auth_mode="chatgpt", service_tier="fast", isolated_cwd=str(tmp_path / "work")),
        image_path=image,
        prompt="caption",
        output_schema={"type": "object"},
        client_factory=lambda _config: FakeClient(),
    )

    assert captured["model"] == "gpt-5.4"
    assert captured["service_tier"] == "fast"
    assert captured["effort"] == "none"
    assert captured["output_schema"] == {"type": "object"}
    assert result.metadata["service_tier"] == "fast"
    assert result.parsed["long_description"] == "fast"


def test_codex_app_server_passes_reasoning_effort_to_thread_run(monkeypatch, tmp_path):
    from module.providers.codex_app_server import CodexAppServerConfig, caption_image_with_app_server

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    captured = {}

    class FakeThread:
        id = "thread-low"

        def run(self, input, *, model=None, effort=None, output_schema=None, approval_mode=None):
            captured["input"] = input
            captured["model"] = model
            captured["effort"] = effort
            captured["output_schema"] = output_schema
            captured["approval_mode"] = approval_mode
            return SimpleNamespace(
                id="turn-low",
                parsed={
                    "short_description": "short",
                    "long_description": "low effort",
                    "tags": ["tag"],
                    "rating": "general",
                    "confidence": 0.7,
                },
            )

    class FakeClient:
        def thread_start(self, **_payload):
            return FakeThread()

    result = caption_image_with_app_server(
        CodexAppServerConfig(auth_mode="chatgpt", reasoning_effort="low", isolated_cwd=str(tmp_path / "work")),
        image_path=image,
        prompt="caption",
        output_schema={"type": "object"},
        client_factory=lambda _config: FakeClient(),
    )

    assert captured["model"] == "gpt-5.4"
    assert captured["effort"] == "low"
    assert captured["output_schema"] == {"type": "object"}
    assert result.metadata["reasoning_effort"] == "low"
    assert result.parsed["long_description"] == "low effort"


def test_codex_sdk_requires_public_release_symbols(monkeypatch):
    from module.providers.codex_app_server import CodexAppServerError, load_openai_codex_sdk

    fake_sdk = SimpleNamespace(
        Codex=object,
        AppServerConfig=object,
        TextInput=object,
        LocalImageInput=object,
        is_retryable_error=lambda _exc: False,
    )
    monkeypatch.setitem(sys.modules, "openai_codex", fake_sdk)

    with pytest.raises(CodexAppServerError) as exc_info:
        load_openai_codex_sdk()

    assert exc_info.value.kind == "sdk_missing"
    assert "TurnResult" in str(exc_info.value)


def test_codex_sdk_sandbox_symbol_is_soft_capability(monkeypatch):
    from module.providers.codex_app_server import load_openai_codex_sdk

    fake_sdk = SimpleNamespace(
        Codex=object,
        CodexConfig=object,
        TextInput=object,
        LocalImageInput=object,
        TurnResult=object,
        retry_on_overload=lambda fn: fn,
    )
    monkeypatch.setitem(sys.modules, "openai_codex", fake_sdk)

    assert load_openai_codex_sdk() is fake_sdk


def test_codex_sdk_accepts_legacy_app_server_config_symbol(monkeypatch):
    from module.providers.codex_app_server import load_openai_codex_sdk

    fake_sdk = SimpleNamespace(
        Codex=object,
        AppServerConfig=object,
        TextInput=object,
        LocalImageInput=object,
        TurnResult=object,
        is_retryable_error=lambda _exc: False,
    )
    monkeypatch.setitem(sys.modules, "openai_codex", fake_sdk)

    assert load_openai_codex_sdk() is fake_sdk


def test_codex_sandbox_prefers_public_sandbox_for_thread_and_run():
    from module.providers.codex_app_server import _resolve_run_sandbox_kwargs, _resolve_thread_sandbox_value

    sdk = SimpleNamespace(
        Sandbox=SimpleNamespace(read_only="public-read", workspace_write="public-write", full_access="public-full")
    )

    class FakeThread:
        def run(self, _input, *, sandbox=None):
            return sandbox

    assert _resolve_thread_sandbox_value(sdk, "read-only") == "public-read"
    assert _resolve_run_sandbox_kwargs(sdk, FakeThread(), "workspace-write") == {"sandbox": "public-write"}


def test_codex_sandbox_uses_legacy_mode_and_constructible_policy():
    from module.providers.codex_app_server import _resolve_run_sandbox_kwargs, _resolve_thread_sandbox_value

    class ReadOnlySandboxPolicy:
        def __init__(self, *, type):
            self.type = type

    class SandboxPolicy:
        def __init__(self, root):
            self.root = root

    sdk = SimpleNamespace(
        types=SimpleNamespace(
            SandboxMode=SimpleNamespace(
                read_only="mode-read",
                workspace_write="mode-write",
                danger_full_access="mode-full",
            )
        ),
        generated=SimpleNamespace(
            v2_all=SimpleNamespace(SandboxPolicy=SandboxPolicy, ReadOnlySandboxPolicy=ReadOnlySandboxPolicy)
        ),
    )

    class FakeThread:
        def run(self, _input, *, sandbox_policy=None):
            return sandbox_policy

    kwargs = _resolve_run_sandbox_kwargs(sdk, FakeThread(), "read-only")

    assert _resolve_thread_sandbox_value(sdk, "full-access") == "mode-full"
    assert list(kwargs) == ["sandbox_policy"]
    assert kwargs["sandbox_policy"].root.type == "readOnly"


def test_codex_sandbox_omits_run_policy_when_policy_cannot_be_constructed():
    from module.providers.codex_app_server import _resolve_run_sandbox_kwargs, _resolve_thread_sandbox_value

    sdk = SimpleNamespace(
        types=SimpleNamespace(SandboxMode=SimpleNamespace(read_only="mode-read")),
        generated=SimpleNamespace(v2_all=SimpleNamespace()),
    )

    class FakeThread:
        def run(self, _input, *, sandbox_policy=None):
            return sandbox_policy

    assert _resolve_thread_sandbox_value(sdk, "read-only") == "mode-read"
    assert _resolve_run_sandbox_kwargs(sdk, FakeThread(), "read-only") == {}


def test_codex_app_server_reads_assistant_final_item_when_final_response_empty():
    from module.providers.codex_app_server import _extract_turn_output

    raw, parsed = _extract_turn_output(
        SimpleNamespace(
            final_response="",
            items=[
                SimpleNamespace(type="reasoning", content="ignore"),
                SimpleNamespace(
                    type="agentMessage",
                    role="assistant",
                    phase="final_answer",
                    content=[
                        SimpleNamespace(
                            text=json.dumps(
                                {
                                    "short_description": "short",
                                    "long_description": "from item",
                                    "tags": ["tag"],
                                    "rating": "general",
                                    "confidence": 0.7,
                                }
                            )
                        )
                    ],
                ),
            ],
        )
    )

    assert parsed is None
    assert json.loads(raw)["long_description"] == "from item"


def test_codex_app_server_empty_turn_output_is_output_error(tmp_path):
    from module.providers.codex_app_server import (
        CodexAppServerConfig,
        CodexAppServerError,
        caption_image_with_app_server,
    )

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")

    class FakeThread:
        id = "thread-empty"

        def run(self, _input):
            return SimpleNamespace(id="turn-empty", final_response="", items=[])

    class FakeClient:
        def thread_start(self, **_payload):
            return FakeThread()

    with pytest.raises(CodexAppServerError) as exc_info:
        caption_image_with_app_server(
            CodexAppServerConfig(auth_mode="chatgpt", isolated_cwd=str(tmp_path / "work")),
            image_path=image,
            prompt="caption",
            client_factory=lambda _config: FakeClient(),
        )

    assert exc_info.value.kind == "output"


def test_codex_app_server_stream_event_summary_includes_item_type_and_agent_delta():
    from module.providers.codex_app_server import _summarize_stream_event

    started = SimpleNamespace(
        method="item/started",
        payload=SimpleNamespace(item=SimpleNamespace(root=SimpleNamespace(type="agentMessage", phase="final_answer"))),
    )
    delta = SimpleNamespace(
        method="item/agentMessage/delta",
        payload=SimpleNamespace(delta="partial structured caption"),
    )
    reasoning = SimpleNamespace(
        method="item/reasoning/textDelta",
        payload=SimpleNamespace(delta="hidden internal reasoning"),
    )

    assert _summarize_stream_event(started) == "Codex app-server: event item/started (agentMessage, final_answer)"
    assert _summarize_stream_event(delta) == "Codex app-server: event item/agentMessage/delta: partial structured caption"
    assert _summarize_stream_event(reasoning) == "Codex app-server: event item/reasoning/textDelta"


def test_codex_app_server_starts_ephemeral_thread_per_image(tmp_path):
    from module.providers.codex_app_server import CodexAppServerCaptionClient, CodexAppServerConfig

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")

    class FakeClient:
        def __init__(self):
            self.thread_count = 0

        def thread_start(self, **_payload):
            self.thread_count += 1
            return FakeThread(f"thread-{self.thread_count}")

    class FakeThread:
        def __init__(self, thread_id):
            self.id = thread_id

        def run(self, _input):
            return SimpleNamespace(
                id=self.id.replace("thread", "turn"),
                final_response=json.dumps(
                    {
                        "short_description": "short",
                        "long_description": "long",
                        "tags": ["tag"],
                        "rating": "general",
                        "confidence": 0.7,
                    }
                ),
            )

    fake = FakeClient()
    client = CodexAppServerCaptionClient(
        CodexAppServerConfig(auth_mode="chatgpt", isolated_cwd=str(tmp_path / "work")),
        client_factory=lambda _config: fake,
    )

    first = client.caption_image(image_path=image, prompt="caption")
    second = client.caption_image(image_path=image, prompt="caption")

    assert fake.thread_count == 2
    assert first.thread_id == "thread-1"
    assert second.thread_id == "thread-2"


def test_codex_app_server_pool_uses_distinct_slots(monkeypatch, tmp_path):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from module.providers import codex_app_server
    from module.providers.codex_app_server import CodexAppServerConfig, caption_image_with_app_server

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")

    created_lock = threading.Lock()
    turn_lock = threading.Lock()
    created_slots = []
    in_flight = 0
    max_in_flight = 0

    class FakeClient:
        def __init__(self):
            with created_lock:
                self.slot = len(created_slots) + 1
                created_slots.append(self.slot)
            self.thread_count = 0
            self.current_thread_id = ""

        def thread_start(self, **_payload):
            self.thread_count += 1
            self.current_thread_id = f"thread-{self.slot}-{self.thread_count}"
            return FakeThread(self, self.current_thread_id)

    class FakeThread:
        def __init__(self, client, thread_id):
            self.client = client
            self.id = thread_id

        def run(self, _input):
            nonlocal in_flight, max_in_flight
            if self.id != self.client.current_thread_id:
                raise AssertionError("request-specific thread state was shared")
            with turn_lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            time.sleep(0.05)
            with turn_lock:
                in_flight -= 1
            return SimpleNamespace(
                id=self.id.replace("thread", "turn"),
                parsed={
                    "short_description": "short",
                    "long_description": self.id,
                    "tags": ["tag"],
                    "rating": "general",
                    "confidence": 0.7,
                },
            )

    codex_app_server.reset_codex_app_server_client_cache()
    monkeypatch.setattr(codex_app_server, "_create_sdk_client", lambda _config: FakeClient())
    config = CodexAppServerConfig(auth_mode="chatgpt", isolated_cwd=str(tmp_path / "work"))

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    caption_image_with_app_server,
                    config,
                    image_path=image,
                    prompt="caption",
                    max_concurrency=2,
                )
                for _ in range(2)
            ]
            results = [future.result() for future in futures]
    finally:
        codex_app_server.reset_codex_app_server_client_cache()

    assert created_slots == [1, 2]
    assert max_in_flight == 2
    assert {result.thread_id for result in results} == {"thread-1-1", "thread-2-1"}


def test_codex_app_server_cache_reset_closes_cached_clients(monkeypatch, tmp_path):
    from module.providers import codex_app_server
    from module.providers.codex_app_server import (
        CodexAppServerConfig,
        get_codex_app_server_client,
        get_codex_app_server_client_pool,
    )

    created = []

    class FakeSdkClient:
        def __init__(self):
            self.closed = False
            created.append(self)

        def close(self):
            self.closed = True

    codex_app_server.reset_codex_app_server_client_cache()
    monkeypatch.setattr(codex_app_server, "_create_sdk_client", lambda _config: FakeSdkClient())
    config = CodexAppServerConfig(auth_mode="chatgpt", isolated_cwd=str(tmp_path / "work"))

    try:
        get_codex_app_server_client(config)
        get_codex_app_server_client_pool(config, max_concurrency=2)

        assert len(created) == 3

        codex_app_server.reset_codex_app_server_client_cache()

        assert all(client.closed for client in created)
    finally:
        codex_app_server.reset_codex_app_server_client_cache()


def test_codex_app_server_transport_failure_discards_cached_client(monkeypatch, tmp_path):
    from module.providers import codex_app_server
    from module.providers.codex_app_server import CodexAppServerConfig, CodexAppServerError, caption_image_with_app_server

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    created = []

    class FakeSdkClient:
        def __init__(self, *, fail: bool):
            self.fail = fail
            self.closed = False
            created.append(self)

        def thread_start(self, **_payload):
            return FakeThread(self, f"thread-{len(created)}")

        def close(self):
            self.closed = True

    class FakeThread:
        def __init__(self, client, thread_id):
            self.client = client
            self.id = thread_id

        def run(self, _input):
            if self.client.fail:
                raise RuntimeError(
                    "Invalid JSON-RPC line: 'SUCCESS: The process with PID 384100 has been terminated.\\n'"
                )
            return SimpleNamespace(
                id="turn-ok",
                parsed={
                    "short_description": "short",
                    "long_description": "recovered",
                    "tags": ["tag"],
                    "rating": "general",
                    "confidence": 0.7,
                },
            )

    def make_client(_config):
        return FakeSdkClient(fail=len(created) == 0)

    codex_app_server.reset_codex_app_server_client_cache()
    monkeypatch.setattr(codex_app_server, "_create_sdk_client", make_client)
    config = CodexAppServerConfig(auth_mode="chatgpt", isolated_cwd=str(tmp_path / "work"))

    try:
        with pytest.raises(CodexAppServerError) as exc_info:
            caption_image_with_app_server(config, image_path=image, prompt="caption")

        assert exc_info.value.kind == "transport"
        assert exc_info.value.retryable is True
        assert len(created) == 1
        assert created[0].closed is True

        result = caption_image_with_app_server(config, image_path=image, prompt="caption")

        assert result.parsed["long_description"] == "recovered"
        assert len(created) == 2
        assert created[1].closed is False
    finally:
        codex_app_server.reset_codex_app_server_client_cache()


def test_codex_app_server_late_startup_client_is_closed_after_timeout(monkeypatch, tmp_path):
    import threading
    import time

    from module.providers import codex_app_server
    from module.providers.codex_app_server import CodexAppServerConfig, CodexAppServerError, caption_image_with_app_server

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    closed = threading.Event()
    created = []

    class FakeSdkClient:
        def __init__(self):
            self.closed = False
            created.append(self)

        def close(self):
            self.closed = True
            closed.set()

    def slow_create(_config):
        time.sleep(0.12)
        return FakeSdkClient()

    codex_app_server.reset_codex_app_server_client_cache()
    monkeypatch.setattr(codex_app_server, "_create_sdk_client", slow_create)

    try:
        with pytest.raises(CodexAppServerError) as exc_info:
            caption_image_with_app_server(
                CodexAppServerConfig(timeout=0.02, auth_mode="chatgpt", isolated_cwd=str(tmp_path / "work")),
                image_path=image,
                prompt="caption",
            )

        assert exc_info.value.kind == "timeout"
        assert closed.wait(timeout=2)
        assert len(created) == 1
        assert created[0].closed is True
    finally:
        codex_app_server.reset_codex_app_server_client_cache()


def test_codex_app_server_pool_replaces_reset_slot(monkeypatch, tmp_path):
    from module.providers import codex_app_server
    from module.providers.codex_app_server import (
        CodexAppServerClientPool,
        CodexAppServerConfig,
        CodexAppServerError,
        CodexAppServerResult,
    )

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    created = []
    calls = 0

    class FakeSdkClient:
        def __init__(self):
            self.closed = False
            created.append(self)

        def close(self):
            self.closed = True

    def fake_caption_image(self, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise CodexAppServerError("transport failed", kind="transport")
        return CodexAppServerResult(
            raw='{"long_description": "recovered"}',
            parsed={"long_description": "recovered"},
            thread_id="thread-2",
            turn_id="turn-2",
            metadata={},
        )

    monkeypatch.setattr(codex_app_server, "_create_sdk_client", lambda _config: FakeSdkClient())
    monkeypatch.setattr(codex_app_server.CodexAppServerCaptionClient, "caption_image", fake_caption_image)
    pool = CodexAppServerClientPool(
        CodexAppServerConfig(auth_mode="chatgpt", isolated_cwd=str(tmp_path / "work")),
        1,
    )

    with pytest.raises(CodexAppServerError) as exc_info:
        pool.caption_image(image_path=image, prompt="caption")

    assert exc_info.value.kind == "transport"
    assert len(created) == 2
    assert created[0].closed is True
    assert created[1].closed is False

    result = pool.caption_image(image_path=image, prompt="caption", timeout=0.5)

    assert result.parsed["long_description"] == "recovered"
    pool.close()


def test_codex_app_server_pool_replaces_closed_client(monkeypatch, tmp_path):
    from module.providers import codex_app_server
    from module.providers.codex_app_server import (
        CodexAppServerClientPool,
        CodexAppServerConfig,
        CodexAppServerError,
        CodexAppServerResult,
    )

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    created = []
    calls = 0

    class FakeSdkClient:
        def __init__(self):
            self.closed = False
            created.append(self)

        def close(self):
            self.closed = True

    def fake_caption_image(self, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise CodexAppServerError("client closed", kind="closed", retryable=True)
        return CodexAppServerResult(
            raw='{"long_description": "recovered"}',
            parsed={"long_description": "recovered"},
            thread_id="thread-2",
            turn_id="turn-2",
            metadata={},
        )

    monkeypatch.setattr(codex_app_server, "_create_sdk_client", lambda _config: FakeSdkClient())
    monkeypatch.setattr(codex_app_server.CodexAppServerCaptionClient, "caption_image", fake_caption_image)
    pool = CodexAppServerClientPool(
        CodexAppServerConfig(auth_mode="chatgpt", isolated_cwd=str(tmp_path / "work")),
        1,
    )

    with pytest.raises(CodexAppServerError) as exc_info:
        pool.caption_image(image_path=image, prompt="caption")

    assert exc_info.value.kind == "closed"
    assert exc_info.value.retryable is True
    assert len(created) == 2
    assert created[0].closed is True
    assert created[1].closed is False

    result = pool.caption_image(image_path=image, prompt="caption", timeout=0.5)

    assert result.parsed["long_description"] == "recovered"
    pool.close()


def test_codex_app_server_direct_client_timeout_closes_sdk_client(tmp_path):
    import time

    from module.providers.codex_app_server import CodexAppServerCaptionClient, CodexAppServerConfig, CodexAppServerError

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")

    class FakeClient:
        def __init__(self):
            self.closed = False

        def thread_start(self, **_payload):
            return FakeThread()

        def close(self):
            self.closed = True

    class FakeThread:
        id = "thread-1"

        def run(self, _input):
            time.sleep(0.2)
            return SimpleNamespace(parsed={"long_description": "late"})

    fake = FakeClient()
    client = CodexAppServerCaptionClient(
        CodexAppServerConfig(timeout=0.03, auth_mode="chatgpt", isolated_cwd=str(tmp_path / "work")),
        client_factory=lambda _config: fake,
    )

    with pytest.raises(CodexAppServerError) as exc_info:
        client.caption_image(image_path=image, prompt="caption")

    assert exc_info.value.kind == "timeout"
    assert fake.closed is True


def test_codex_app_server_factory_client_closes_after_success(tmp_path):
    from module.providers.codex_app_server import CodexAppServerConfig, caption_image_with_app_server

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")

    class FakeClient:
        def __init__(self):
            self.closed = False

        def thread_start(self, **_payload):
            return FakeThread()

        def close(self):
            self.closed = True

    class FakeThread:
        id = "thread-1"

        def run(self, _input):
            return SimpleNamespace(
                id="turn-1",
                parsed={
                    "short_description": "short",
                    "long_description": "long",
                    "tags": ["tag"],
                    "rating": "general",
                    "confidence": 0.7,
                },
            )

    fake = FakeClient()
    result = caption_image_with_app_server(
        CodexAppServerConfig(auth_mode="chatgpt", isolated_cwd=str(tmp_path / "work")),
        image_path=image,
        prompt="caption",
        client_factory=lambda _config: fake,
    )

    assert result.parsed["long_description"] == "long"
    assert fake.closed is True


def test_codex_app_server_timeout_applies_to_sdk_client_startup(tmp_path):
    import time

    from module.providers.codex_app_server import CodexAppServerConfig, CodexAppServerError, caption_image_with_app_server

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")

    def slow_client_factory(_config):
        time.sleep(0.2)
        raise AssertionError("timeout should fire before the client factory returns")

    with pytest.raises(CodexAppServerError) as exc_info:
        caption_image_with_app_server(
            CodexAppServerConfig(timeout=0.01, auth_mode="chatgpt", isolated_cwd=str(tmp_path / "work")),
            image_path=image,
            prompt="caption",
            client_factory=slow_client_factory,
        )

    assert exc_info.value.kind == "timeout"
    assert "client_startup timed out" in str(exc_info.value)


def test_codex_app_server_startup_transport_failure_is_wrapped(tmp_path):
    from module.providers.codex_app_server import CodexAppServerConfig, CodexAppServerError, caption_image_with_app_server

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")

    def broken_client_factory(_config):
        raise RuntimeError(
            "Invalid JSON-RPC line: 'SUCCESS: The process with PID 384100 has been terminated.\\n'"
        )

    with pytest.raises(CodexAppServerError) as exc_info:
        caption_image_with_app_server(
            CodexAppServerConfig(auth_mode="chatgpt", isolated_cwd=str(tmp_path / "work")),
            image_path=image,
            prompt="caption",
            client_factory=broken_client_factory,
        )

    assert exc_info.value.kind == "transport"
    assert exc_info.value.retryable is True
    assert "client_startup failed" in str(exc_info.value)


def test_codex_app_server_timeout_applies_to_image_turn(tmp_path):
    import time

    from module.providers.codex_app_server import CodexAppServerConfig, CodexAppServerError, caption_image_with_app_server

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")

    class FakeClient:
        def thread_start(self, **_payload):
            return FakeThread()

    class FakeThread:
        id = "thread-1"

        def run(self, _input):
            time.sleep(0.2)
            return SimpleNamespace(parsed={"long_description": "late"})

    with pytest.raises(CodexAppServerError) as exc_info:
        caption_image_with_app_server(
            CodexAppServerConfig(timeout=0.03, auth_mode="chatgpt", isolated_cwd=str(tmp_path / "work")),
            image_path=image,
            prompt="caption",
            client_factory=lambda _config: FakeClient(),
        )

    assert exc_info.value.kind == "timeout"
    assert "caption_image timed out" in str(exc_info.value)


def test_codex_app_server_timeout_emits_waiting_heartbeat():
    import time

    from module.providers.codex_app_server import CodexAppServerError, _run_with_timeout

    progress_events = []

    with pytest.raises(CodexAppServerError) as exc_info:
        _run_with_timeout(
            lambda: time.sleep(0.08),
            timeout=0.04,
            stage="caption_image",
            progress_callback=progress_events.append,
            heartbeat_seconds=0.01,
        )

    assert exc_info.value.kind == "timeout"
    assert any("caption_image still waiting" in event for event in progress_events)


def test_codex_app_server_timeout_runs_cleanup_before_returning():
    import threading

    from module.providers.codex_app_server import CodexAppServerError, _run_with_timeout

    cleanup_started = threading.Event()
    worker_finished_after_cleanup = threading.Event()

    def blocking_call():
        cleanup_started.wait(timeout=1.0)
        worker_finished_after_cleanup.set()

    with pytest.raises(CodexAppServerError) as exc_info:
        _run_with_timeout(
            blocking_call,
            timeout=0.02,
            stage="caption_image",
            on_timeout=cleanup_started.set,
            timeout_cleanup_grace=0.5,
        )

    assert exc_info.value.kind == "timeout"
    assert cleanup_started.is_set()
    assert worker_finished_after_cleanup.is_set()


def test_codex_app_server_missing_sdk_error_mentions_optional_extra(monkeypatch):
    import builtins

    from module.providers.codex_app_server import CodexAppServerError, load_openai_codex_sdk

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "openai_codex":
            raise ImportError("missing openai_codex")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(CodexAppServerError) as exc_info:
        load_openai_codex_sdk()

    assert exc_info.value.kind == "sdk_missing"
    assert "uv sync --extra codex-subscription" in str(exc_info.value)


def test_codex_subscription_defaults_to_sdk_app_server(monkeypatch, tmp_path):
    from rich.console import Console

    from module.providers.base import MediaContext, MediaModality, ProviderContext, PromptContext
    from module.providers.cloud_vlm import codex_subscription
    from module.providers.cloud_vlm.codex_subscription import CodexSubscriptionProvider

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    captured = {}

    def fake_caption(config, *, image_path, prompt, output_schema, structured=True, progress_callback=None):
        captured["config"] = config
        captured["image_path"] = image_path
        captured["prompt"] = prompt
        captured["output_schema"] = output_schema
        captured["progress_callback"] = progress_callback
        return SimpleNamespace(
            parsed={
                "short_description": "short",
                "long_description": "long",
                "tags": ["tag"],
                "rating": "general",
                "confidence": 0.8,
            }
        )

    monkeypatch.setattr(codex_subscription, "caption_image_with_app_server", fake_caption)
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        """
[mcp_servers.node_repl]
command = "node_repl"

[plugins."gmail@openai-curated"]
enabled = true
""",
        encoding="utf-8",
    )
    args = make_provider_args(codex_subscription=True, codex_home=str(codex_home))
    provider = CodexSubscriptionProvider(ProviderContext(console=Console(), args=args))
    media = MediaContext(uri=str(image), mime="image/png", sha256hash="", modality=MediaModality.IMAGE)

    result = provider.attempt(media, PromptContext(system="sys", user="user"))

    assert result.metadata["backend"] == "sdk_app_server"
    assert result.metadata["auth_mode"] == "chatgpt"
    assert result.metadata["reasoning_effort"] == "none"
    assert captured["config"].auth_mode == "chatgpt"
    assert captured["config"].api_key == ""
    assert captured["config"].reasoning_effort == "none"
    assert "mcp_servers.node_repl.enabled=false" in captured["config"].config_overrides
    assert "features.plugins=false" in captured["config"].config_overrides
    assert "mcp_servers={}" not in captured["config"].config_overrides
    assert not any('"node_repl"' in override for override in captured["config"].config_overrides)
    assert "plugins={}" not in captured["config"].config_overrides
    assert captured["output_schema"]["type"] == "object"
    assert captured["progress_callback"] is None
    assert result.parsed["long_description"] == "long"


def test_codex_subscription_can_allow_user_mcp_config(monkeypatch, tmp_path):
    from rich.console import Console

    from module.providers.base import MediaContext, MediaModality, ProviderContext, PromptContext
    from module.providers.cloud_vlm import codex_subscription
    from module.providers.cloud_vlm.codex_subscription import CodexSubscriptionProvider

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    captured = {}

    def fake_caption(config, *, image_path, prompt, output_schema, structured=True, progress_callback=None):
        captured["config"] = config
        return SimpleNamespace(
            parsed={
                "short_description": "short",
                "long_description": "long",
                "tags": ["tag"],
                "rating": "general",
                "confidence": 0.8,
            }
        )

    monkeypatch.setattr(codex_subscription, "caption_image_with_app_server", fake_caption)
    args = make_provider_args(codex_subscription=True, codex_disable_mcp=False)
    provider = CodexSubscriptionProvider(ProviderContext(console=Console(), args=args))
    media = MediaContext(uri=str(image), mime="image/png", sha256hash="", modality=MediaModality.IMAGE)

    provider.attempt(media, PromptContext(system="sys", user="user"))

    assert captured["config"].config_overrides == ()


def test_codex_subscription_retries_retryable_app_server_transport(monkeypatch, tmp_path):
    from PIL import Image
    from rich.console import Console

    from module.providers.base import ProviderContext
    from module.providers.cloud_vlm import codex_subscription
    from module.providers.cloud_vlm.codex_subscription import CodexSubscriptionProvider
    from module.providers.codex_app_server import CodexAppServerError

    image = tmp_path / "image.jpg"
    Image.new("RGB", (32, 32), color=(20, 40, 80)).save(image)
    calls = 0

    def fake_caption(config, *, image_path, prompt, output_schema, structured=True, progress_callback=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise CodexAppServerError(
                "Codex app-server request failed (transport): Invalid JSON-RPC line",
                kind="transport",
                retryable=True,
            )
        return SimpleNamespace(
            parsed={
                "long_description": "retry recovered",
                "tags": ["tag"],
                "rating": "general",
                "confidence": 0.8,
            }
        )

    monkeypatch.setattr(codex_subscription, "caption_image_with_app_server", fake_caption)
    args = make_provider_args(codex_subscription=True)
    provider = CodexSubscriptionProvider(ProviderContext(console=Console(), config={"prompts": {}}, args=args))

    result = provider.execute(str(image), "image/jpeg", "hash")

    assert calls == 2
    assert result.parsed["long_description"] == "retry recovered"


def test_codex_subscription_rate_limit_exhaustion_returns_skip_result(monkeypatch, tmp_path):
    from PIL import Image
    from rich.console import Console

    from module.providers.base import ProviderContext
    from module.providers.cloud_vlm import codex_subscription
    from module.providers.cloud_vlm.codex_subscription import CodexSubscriptionProvider
    from module.providers.codex_app_server import CodexAppServerError

    image = tmp_path / "image.jpg"
    Image.new("RGB", (32, 32), color=(20, 40, 80)).save(image)
    calls = 0

    def fake_caption(config, *, image_path, prompt, output_schema, structured=True, progress_callback=None):
        nonlocal calls
        calls += 1
        raise CodexAppServerError(
            "Codex app-server request failed (rate_limited): exceeded retry limit, "
            "last status: 429 Too Many Requests",
            kind="rate_limited",
            retryable=True,
        )

    monkeypatch.setattr(codex_subscription, "caption_image_with_app_server", fake_caption)
    args = make_provider_args(codex_subscription=True, max_retries=2, wait_time=0)
    provider = CodexSubscriptionProvider(
        ProviderContext(
            console=Console(file=io.StringIO(), force_terminal=False),
            config={"prompts": {}},
            args=args,
        )
    )

    result = provider.execute(str(image), "image/jpeg", "hash")

    assert calls == 2
    assert result.raw == ""
    assert result.parsed is None
    assert result.metadata["skip_reason"] == "rate_limited"
    assert result.metadata["error_kind"] == "rate_limited"
    assert result.metadata["retry_exhausted"] is True


def test_codex_subscription_execute_displays_structured_rating(monkeypatch, tmp_path):
    from unittest.mock import patch

    from PIL import Image
    from rich.console import Console

    from module.providers.base import ProviderContext
    from module.providers.cloud_vlm import codex_subscription
    from module.providers.cloud_vlm.codex_subscription import CodexSubscriptionProvider

    image = tmp_path / "image.jpg"
    Image.new("RGB", (32, 32), color=(20, 40, 80)).save(image)
    payload = {
        "long_description": "structured codex caption",
        "tags": ["tag"],
        "rating": "general",
        "confidence": 0.8,
        "scores": {"Composition & Framing": 8},
        "average_score": 8.0,
    }

    def fake_caption(config, *, image_path, prompt, output_schema, structured=True, progress_callback=None):
        return SimpleNamespace(parsed=payload)

    monkeypatch.setattr(codex_subscription, "caption_image_with_app_server", fake_caption)
    args = make_provider_args(codex_subscription=True)
    provider = CodexSubscriptionProvider(ProviderContext(console=Console(), config={"prompts": {}}, args=args))

    with patch("module.providers.base.display_caption_and_rate") as mocked_display:
        result = provider.execute(str(image), "image/jpeg", "hash")

    assert result.parsed == payload
    mocked_display.assert_called_once()
    kwargs = mocked_display.call_args.kwargs
    assert kwargs["title"] == image.name
    assert kwargs["long_description"] == "structured codex caption"
    assert kwargs["rating"] == {"Composition & Framing": 8}
    assert kwargs["average_score"] == 8.0
    assert kwargs["pixels"] is not None


def test_codex_subscription_fast_mode_sets_app_server_service_tier(monkeypatch, tmp_path):
    from rich.console import Console

    from module.providers.base import MediaContext, MediaModality, ProviderContext, PromptContext
    from module.providers.cloud_vlm import codex_subscription
    from module.providers.cloud_vlm.codex_subscription import CodexSubscriptionProvider

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    captured = {}

    def fake_caption(config, *, image_path, prompt, output_schema, structured=True, progress_callback=None):
        captured["config"] = config
        return SimpleNamespace(
            parsed={
                "short_description": "short",
                "long_description": "fast",
                "tags": ["tag"],
                "rating": "general",
                "confidence": 0.8,
            }
        )

    monkeypatch.setattr(codex_subscription, "caption_image_with_app_server", fake_caption)
    args = make_provider_args(codex_subscription=True, codex_fast=True)
    provider = CodexSubscriptionProvider(ProviderContext(console=Console(), args=args))
    media = MediaContext(uri=str(image), mime="image/png", sha256hash="", modality=MediaModality.IMAGE)

    result = provider.attempt(media, PromptContext(system="sys", user="user"))

    assert captured["config"].service_tier == "fast"
    assert result.metadata["service_tier"] == "fast"


def test_codex_subscription_explicit_service_tier_overrides_fast_shortcut(monkeypatch, tmp_path):
    from rich.console import Console

    from module.providers.base import MediaContext, MediaModality, ProviderContext, PromptContext
    from module.providers.cloud_vlm import codex_subscription
    from module.providers.cloud_vlm.codex_subscription import CodexSubscriptionProvider

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    captured = {}

    def fake_caption(config, *, image_path, prompt, output_schema, structured=True, progress_callback=None):
        captured["config"] = config
        return SimpleNamespace(
            parsed={
                "short_description": "short",
                "long_description": "custom tier",
                "tags": ["tag"],
                "rating": "general",
                "confidence": 0.8,
            }
        )

    monkeypatch.setattr(codex_subscription, "caption_image_with_app_server", fake_caption)
    args = make_provider_args(codex_subscription=True, codex_fast=True, codex_service_tier="standard")
    provider = CodexSubscriptionProvider(ProviderContext(console=Console(), args=args))
    media = MediaContext(uri=str(image), mime="image/png", sha256hash="", modality=MediaModality.IMAGE)

    result = provider.attempt(media, PromptContext(system="sys", user="user"))

    assert captured["config"].service_tier == "standard"
    assert result.metadata["service_tier"] == "standard"


def test_codex_subscription_explicit_reasoning_effort_overrides_default(monkeypatch, tmp_path):
    from rich.console import Console

    from module.providers.base import MediaContext, MediaModality, ProviderContext, PromptContext
    from module.providers.cloud_vlm import codex_subscription
    from module.providers.cloud_vlm.codex_subscription import CodexSubscriptionProvider

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    captured = {}

    def fake_caption(config, *, image_path, prompt, output_schema, structured=True, progress_callback=None):
        captured["config"] = config
        return SimpleNamespace(
            parsed={
                "short_description": "short",
                "long_description": "minimal effort",
                "tags": ["tag"],
                "rating": "general",
                "confidence": 0.8,
            }
        )

    monkeypatch.setattr(codex_subscription, "caption_image_with_app_server", fake_caption)
    args = make_provider_args(codex_subscription=True, codex_reasoning_effort="minimal")
    provider = CodexSubscriptionProvider(ProviderContext(console=Console(), args=args))
    media = MediaContext(uri=str(image), mime="image/png", sha256hash="", modality=MediaModality.IMAGE)

    result = provider.attempt(media, PromptContext(system="sys", user="user"))

    assert captured["config"].reasoning_effort == "minimal"
    assert result.metadata["reasoning_effort"] == "minimal"


def test_codex_subscription_app_server_timeout_returns_empty_result(monkeypatch, tmp_path):
    from rich.console import Console

    from module.providers.base import MediaContext, MediaModality, ProviderContext, PromptContext
    from module.providers.cloud_vlm import codex_subscription
    from module.providers.cloud_vlm.codex_subscription import CodexSubscriptionProvider
    from module.providers.codex_app_server import CodexAppServerError

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")

    def fake_caption(config, *, image_path, prompt, output_schema, structured=True, progress_callback=None):
        raise CodexAppServerError("Codex app-server caption_image timed out after 180s.", kind="timeout")

    monkeypatch.setattr(codex_subscription, "caption_image_with_app_server", fake_caption)
    args = make_provider_args(codex_subscription=True)
    provider = CodexSubscriptionProvider(ProviderContext(console=Console(), args=args))
    media = MediaContext(uri=str(image), mime="image/png", sha256hash="", modality=MediaModality.IMAGE)

    result = provider.attempt(media, PromptContext(system="sys", user="user"))

    assert result.raw == ""
    assert result.parsed is None
    assert result.metadata["backend"] == "sdk_app_server"
    assert result.metadata["skip_reason"] == "timeout"
    assert result.metadata["error_kind"] == "timeout"
    assert result.metadata["structured"] is False


def test_codex_subscription_exec_timeout_returns_empty_result(monkeypatch, tmp_path):
    from rich.console import Console

    from module.providers.base import MediaContext, MediaModality, ProviderContext, PromptContext
    from module.providers.cloud_vlm import codex_subscription
    from module.providers.cloud_vlm.codex_subscription import CodexSubscriptionProvider
    from module.providers.codex_exec import CodexExecError

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")

    def fake_exec(config, *, image_path, prompt, output_path, schema_path=None, structured=True):
        raise CodexExecError("Codex exec timed out after 180s.", kind="timeout")

    monkeypatch.setattr(codex_subscription, "run_codex_exec_caption", fake_exec)
    args = make_provider_args(codex_subscription=True, codex_backend="exec")
    provider = CodexSubscriptionProvider(ProviderContext(console=Console(), args=args))
    media = MediaContext(uri=str(image), mime="image/png", sha256hash="", modality=MediaModality.IMAGE)

    result = provider.attempt(media, PromptContext(system="sys", user="user"))

    assert result.raw == ""
    assert result.parsed is None
    assert result.metadata["backend"] == "exec"
    assert result.metadata["skip_reason"] == "timeout"
    assert result.metadata["error_kind"] == "timeout"
    assert result.metadata["structured"] is False


def test_codex_subscription_passes_effective_app_server_concurrency(monkeypatch, tmp_path):
    from rich.console import Console

    from module.providers.base import MediaContext, MediaModality, ProviderContext, PromptContext
    from module.providers.cloud_vlm import codex_subscription
    from module.providers.cloud_vlm.codex_subscription import CodexSubscriptionProvider

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    captured = {}

    def fake_caption(config, *, image_path, prompt, output_schema, structured=True, max_concurrency=None, progress_callback=None):
        captured["max_concurrency"] = max_concurrency
        return SimpleNamespace(
            parsed={
                "short_description": "short",
                "long_description": "long",
                "tags": ["tag"],
                "rating": "general",
                "confidence": 0.8,
            }
        )

    monkeypatch.setattr(codex_subscription, "caption_image_with_app_server", fake_caption)
    args = make_provider_args(codex_subscription=True, codex_max_concurrency=2)
    provider = CodexSubscriptionProvider(ProviderContext(console=Console(), args=args))
    media = MediaContext(uri=str(image), mime="image/png", sha256hash="", modality=MediaModality.IMAGE)

    provider.attempt(media, PromptContext(system="sys", user="user"))

    assert captured["max_concurrency"] == 2


def test_codex_subscription_passes_folder_name_into_app_server_prompt(monkeypatch, tmp_path):
    from rich.console import Console

    from module.providers.base import ProviderContext
    from module.providers.cloud_vlm import codex_subscription
    from module.providers.cloud_vlm.codex_subscription import CodexSubscriptionProvider

    folder = tmp_path / "Alice (Wonderland)"
    folder.mkdir()
    image = folder / "sample.jpg"
    image.write_bytes(b"fake")
    captured = {}

    def fake_caption(config, *, image_path, prompt, output_schema, structured=True, progress_callback=None):
        captured["config"] = config
        captured["image_path"] = image_path
        captured["prompt"] = prompt
        captured["output_schema"] = output_schema
        return SimpleNamespace(
            parsed={
                "short_description": "short",
                "long_description": "long",
                "tags": ["tag"],
                "rating": "general",
                "confidence": 0.8,
            }
        )

    monkeypatch.setattr(codex_subscription, "caption_image_with_app_server", fake_caption)
    args = make_provider_args(codex_subscription=True, dir_name=True)
    provider = CodexSubscriptionProvider(ProviderContext(console=Console(), config={"prompts": {}}, args=args))

    result = provider.execute(str(image), "image/jpeg", "hash")

    assert result.parsed["long_description"] == "long"
    assert captured["image_path"] == str(image)
    assert "Project user prompt:" in captured["prompt"]
    assert (
        "If there is a person/character or more in the image you must refer to them as "
        "<Alice> from (Wonderland)."
    ) in captured["prompt"]
    assert "Character name hint:" not in captured["prompt"]
    assert captured["prompt"].find("<Alice> from (Wonderland)") < captured["prompt"].find("Task:")


def test_codex_subscription_transcodes_avif_for_app_server(monkeypatch, tmp_path):
    from rich.console import Console

    from module.providers.base import MediaContext, MediaModality, ProviderContext, PromptContext
    from module.providers.cloud_vlm import codex_subscription
    from module.providers.cloud_vlm.codex_subscription import CodexSubscriptionProvider

    image = tmp_path / "sample.avif"
    image.write_bytes(b"fake-avif")
    converted = tmp_path / "converted.jpg"
    captured = {}

    def fake_convert(path, *, quality):
        captured["converted_from"] = path
        captured["quality"] = quality
        converted.write_bytes(b"jpeg")
        return converted

    def fake_caption(config, *, image_path, prompt, output_schema, structured=True, progress_callback=None):
        captured["image_path"] = Path(image_path)
        captured["converted_existed_during_call"] = Path(image_path).exists()
        return SimpleNamespace(
            parsed={
                "short_description": "short",
                "long_description": "long",
                "tags": ["tag"],
                "rating": "general",
                "confidence": 0.8,
            }
        )

    monkeypatch.setattr(codex_subscription, "_convert_to_temp_jpeg", fake_convert)
    monkeypatch.setattr(codex_subscription, "caption_image_with_app_server", fake_caption)
    args = make_provider_args(codex_subscription=True)
    provider = CodexSubscriptionProvider(ProviderContext(console=Console(), args=args))
    media = MediaContext(uri=str(image), mime="image/avif", sha256hash="", modality=MediaModality.IMAGE)

    result = provider.attempt(media, PromptContext(system="sys", user="user"))

    assert result.parsed["long_description"] == "long"
    assert captured["converted_from"] == image
    assert captured["image_path"] == converted
    assert captured["converted_existed_during_call"] is True
    assert image.exists()
    assert not converted.exists()


def test_codex_subscription_exec_backend_is_explicit(monkeypatch, tmp_path):
    from rich.console import Console

    from module.providers.base import MediaContext, MediaModality, ProviderContext, PromptContext
    from module.providers.cloud_vlm import codex_subscription
    from module.providers.cloud_vlm.codex_subscription import CodexSubscriptionProvider

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    called = {}

    def fake_exec(config, *, image_path, prompt, output_path, schema_path=None, structured=True):
        called["config"] = config
        called["image_path"] = image_path
        return SimpleNamespace(
            parsed={
                "short_description": "short",
                "long_description": "long",
                "tags": ["tag"],
                "rating": "general",
                "confidence": 0.8,
            }
        )

    monkeypatch.setattr(codex_subscription, "run_codex_exec_caption", fake_exec)
    args = make_provider_args(codex_subscription=True, codex_backend="exec", codex_service_tier="fast")
    provider = CodexSubscriptionProvider(ProviderContext(console=Console(), args=args))
    media = MediaContext(uri=str(image), mime="image/png", sha256hash="", modality=MediaModality.IMAGE)

    result = provider.attempt(media, PromptContext(system="", user="user"))

    assert result.metadata["backend"] == "exec"
    assert result.metadata["reasoning_effort"] == "none"
    assert called["config"].command == "codex"
    assert called["config"].service_tier == "fast"
    assert called["config"].reasoning_effort == "none"
    assert called["image_path"] == str(image)


def test_codex_subscription_sdk_missing_does_not_fallback_to_exec(monkeypatch, tmp_path):
    from rich.console import Console

    from module.providers.base import MediaContext, MediaModality, ProviderContext, PromptContext
    from module.providers.cloud_vlm import codex_subscription
    from module.providers.cloud_vlm.codex_subscription import CodexSubscriptionProvider
    from module.providers.codex_app_server import CodexAppServerError

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")

    def fake_caption(*_args, **_kwargs):
        raise CodexAppServerError("Codex Python SDK is not installed. uv sync --extra codex-subscription", kind="sdk_missing")

    def fake_exec(*_args, **_kwargs):
        raise AssertionError("exec fallback must not run")

    monkeypatch.setattr(codex_subscription, "caption_image_with_app_server", fake_caption)
    monkeypatch.setattr(codex_subscription, "run_codex_exec_caption", fake_exec)
    args = make_provider_args(codex_subscription=True)
    provider = CodexSubscriptionProvider(ProviderContext(console=Console(), args=args))
    media = MediaContext(uri=str(image), mime="image/png", sha256hash="", modality=MediaModality.IMAGE)

    with pytest.raises(CodexAppServerError) as exc_info:
        provider.attempt(media, PromptContext(system="", user="user"))

    assert exc_info.value.kind == "sdk_missing"
