import json
import os
import stat
import sys
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


def test_codex_exec_command_uses_argument_list(tmp_path):
    from module.providers.codex_exec import CodexExecConfig, build_codex_exec_command

    cwd = tmp_path / "work"
    cwd.mkdir()
    command = build_codex_exec_command(
        CodexExecConfig(command="codex-test-bin", model="gpt-5.4-mini", isolated_cwd=str(cwd)),
        image_path=tmp_path / "image one.png",
        prompt="return json",
        schema_path=tmp_path / "schema.json",
        output_path=tmp_path / "out.json",
    )

    assert command[0:2] == ["codex-test-bin", "exec"]
    assert "--image" in command
    assert str((tmp_path / "image one.png").resolve()) in command
    assert command[-1] == "-"
    assert "return json" not in command
    assert all(" " not in item or item.endswith("image one.png") for item in command)


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
    from providers.registry import get_registry

    reg = get_registry()

    assert reg.find_provider(make_provider_args(), "image/jpeg") is None
    provider = reg.find_provider(make_provider_args(codex_subscription=True), "image/jpeg")
    assert provider is not None
    assert provider.name == "codex_subscription"


def test_codex_subscription_ignores_default_document_image_flag_without_ocr_route():
    from providers.registry import get_registry

    reg = get_registry()

    provider = reg.find_provider(
        make_provider_args(codex_subscription=True, document_image=True),
        "image/jpeg",
    )
    assert provider is not None
    assert provider.name == "codex_subscription"


def test_codex_subscription_does_not_steal_explicit_ocr_or_vlm_routes():
    from providers.registry import get_registry

    reg = get_registry()

    provider = reg.find_provider(
        make_provider_args(codex_subscription=True, ocr_model="paddle_ocr", document_image=True),
        "image/jpeg",
    )
    assert provider is not None
    assert provider.name == "paddle_ocr"

    provider = reg.find_provider(
        make_provider_args(codex_subscription=True, vlm_image_model="moondream"),
        "image/jpeg",
    )
    assert provider is not None
    assert provider.name == "moondream"


def test_codex_app_server_uses_thread_and_turn_payloads_without_api_key_env(monkeypatch, tmp_path):
    from module.providers.codex_app_server import CodexAppServerConfig, caption_image_with_app_server

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")

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
            return {"threadId": "thread-1"}

        def turn_start(self, **payload):
            self.calls.append(("turn_start", payload))
            return {
                "turnId": "turn-1",
                "final_response": json.dumps(
                    {
                        "short_description": "short",
                        "long_description": "long",
                        "tags": ["tag"],
                        "rating": "general",
                        "confidence": 0.7,
                    }
                ),
            }

    fake = FakeClient()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-be-used")

    result = caption_image_with_app_server(
        CodexAppServerConfig(auth_mode="chatgpt", isolated_cwd=str(tmp_path / "work")),
        image_path=image,
        prompt="caption",
        output_schema={"type": "object"},
        client_factory=lambda _config: fake,
    )

    assert fake.api_key == ""
    turn_call = [payload for name, payload in fake.calls if name == "turn_start"][0]
    assert turn_call["threadId"] == "thread-1"
    assert turn_call["input"] == [
        {"type": "text", "text": "caption"},
        {"type": "localImage", "path": str(image.resolve())},
    ]
    assert turn_call["outputSchema"] == {"type": "object"}
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
            return {"threadId": "thread-1"}

        def turn_start(self, **_payload):
            return {
                "parsed": {
                    "short_description": "short",
                    "long_description": "long",
                    "tags": ["tag"],
                    "rating": "general",
                    "confidence": 0.7,
                }
            }

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

    api_key_env = build_codex_app_server_env(
        CodexAppServerConfig(auth_mode="api_key", api_key="sk-explicit"),
        base_env=base_env,
    )
    assert api_key_env["OPENAI_API_KEY"] == "sk-explicit"
    assert "CODEX_API_KEY" not in api_key_env

    api_key_env_from_source = build_codex_app_server_env(
        CodexAppServerConfig(auth_mode="api_key"),
        base_env=base_env,
    )
    assert api_key_env_from_source["OPENAI_API_KEY"] == "sk-env"


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
    assert captured["model"] == "gpt-5.4-mini"
    assert captured["output_schema"] == {"type": "object"}
    assert _input_item_value(captured["input"][0], "text") == "caption"
    assert _input_item_value(captured["input"][1], "path") == str(image.resolve())
    assert result.thread_id == "thread-sdk"
    assert result.turn_id == "turn-sdk"
    assert result.parsed["long_description"] == "long"


def test_codex_app_server_starts_ephemeral_thread_per_image(tmp_path):
    from module.providers.codex_app_server import CodexAppServerCaptionClient, CodexAppServerConfig

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")

    class FakeClient:
        def __init__(self):
            self.thread_count = 0

        def thread_start(self, **_payload):
            self.thread_count += 1
            return {"threadId": f"thread-{self.thread_count}"}

        def turn_start(self, **payload):
            return {
                "turnId": payload["threadId"].replace("thread", "turn"),
                "final_response": json.dumps(
                    {
                        "short_description": "short",
                        "long_description": "long",
                        "tags": ["tag"],
                        "rating": "general",
                        "confidence": 0.7,
                    }
                ),
            }

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
            return {"threadId": self.current_thread_id}

        def turn_start(self, **payload):
            nonlocal in_flight, max_in_flight
            if payload["threadId"] != self.current_thread_id:
                raise AssertionError("request-specific thread state was shared")
            with turn_lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            time.sleep(0.05)
            with turn_lock:
                in_flight -= 1
            return {
                "turnId": payload["threadId"].replace("thread", "turn"),
                "parsed": {
                    "short_description": "short",
                    "long_description": payload["threadId"],
                    "tags": ["tag"],
                    "rating": "general",
                    "confidence": 0.7,
                },
            }

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

    def fake_caption(config, *, image_path, prompt, output_schema):
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
    args = make_provider_args(codex_subscription=True)
    provider = CodexSubscriptionProvider(ProviderContext(console=Console(), args=args))
    media = MediaContext(uri=str(image), mime="image/png", sha256hash="", modality=MediaModality.IMAGE)

    result = provider.attempt(media, PromptContext(system="sys", user="user"))

    assert result.metadata["backend"] == "sdk_app_server"
    assert result.metadata["auth_mode"] == "chatgpt"
    assert captured["config"].auth_mode == "chatgpt"
    assert captured["config"].api_key == ""
    assert captured["output_schema"]["type"] == "object"
    assert result.parsed["long_description"] == "long"


def test_codex_subscription_passes_effective_app_server_concurrency(monkeypatch, tmp_path):
    from rich.console import Console

    from module.providers.base import MediaContext, MediaModality, ProviderContext, PromptContext
    from module.providers.cloud_vlm import codex_subscription
    from module.providers.cloud_vlm.codex_subscription import CodexSubscriptionProvider

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    captured = {}

    def fake_caption(config, *, image_path, prompt, output_schema, max_concurrency=None):
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
    args = make_provider_args(codex_subscription=True, cloud_max_concurrency=4, codex_max_concurrency=2)
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

    def fake_caption(config, *, image_path, prompt, output_schema):
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
    assert "<Alice> from (Wonderland)" in captured["prompt"]
    assert captured["prompt"].find("<Alice> from (Wonderland)") < captured["prompt"].find("Task:")


def test_codex_subscription_exec_backend_is_explicit(monkeypatch, tmp_path):
    from rich.console import Console

    from module.providers.base import MediaContext, MediaModality, ProviderContext, PromptContext
    from module.providers.cloud_vlm import codex_subscription
    from module.providers.cloud_vlm.codex_subscription import CodexSubscriptionProvider

    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    called = {}

    def fake_exec(config, *, image_path, prompt, schema_path, output_path):
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
    args = make_provider_args(codex_subscription=True, codex_backend="exec")
    provider = CodexSubscriptionProvider(ProviderContext(console=Console(), args=args))
    media = MediaContext(uri=str(image), mime="image/png", sha256hash="", modality=MediaModality.IMAGE)

    result = provider.attempt(media, PromptContext(system="", user="user"))

    assert result.metadata["backend"] == "exec"
    assert called["config"].command == "codex"
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
