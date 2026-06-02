import io
import json
import os
import stat
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from rich.console import Console

from tests.provider_v2_helpers import ROOT, make_provider_args


def _caption_payload(short: str = "short", long: str = "long") -> dict:
    return {
        "short_description": short,
        "long_description": long,
        "tags": ["tag"],
        "rating": "general",
        "confidence": 0.9,
    }


def _make_fake_grok(tmp_path: Path, script_body: str) -> Path:
    script = tmp_path / "fake_grok.py"
    script.write_text(script_body, encoding="utf-8")
    if os.name == "nt":
        wrapper = tmp_path / "grok.cmd"
        wrapper.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        wrapper = tmp_path / "grok"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return wrapper


def _write_image(path: Path, color: tuple[int, int, int] = (255, 0, 0)) -> Path:
    image = Image.new("RGB", (32, 24), color)
    image.save(path)
    return path


def test_grok_build_env_clears_api_keys_and_keeps_proxy():
    from module.providers.grok_build_headless import build_grok_build_env

    env = build_grok_build_env(
        {
            "XAI_API_KEY": "xai-api",
            "GROK_CODE_XAI_API_KEY": "grok-code-api",
            "HTTPS_PROXY": "http://proxy.local",
        }
    )

    assert "XAI_API_KEY" not in env
    assert "GROK_CODE_XAI_API_KEY" not in env
    assert env["HTTPS_PROXY"] == "http://proxy.local"


def test_grok_build_command_uses_prompt_json_argument_list(tmp_path):
    from module.providers.grok_build_headless import GrokBuildHeadlessConfig, build_grok_build_command

    cwd = tmp_path / "work"
    cwd.mkdir()
    prompt_json = json.dumps([{"type": "text", "text": "return json"}])
    command = build_grok_build_command(
        GrokBuildHeadlessConfig(command="grok-test-bin", model="grok-build", isolated_cwd=str(cwd)),
        prompt_json=prompt_json,
    )

    assert command[0] == "grok-test-bin"
    assert "--prompt-json" in command
    assert command[command.index("--prompt-json") + 1] == prompt_json
    assert "--output-format" in command
    assert "json" in command
    assert "--no-auto-update" in command
    assert "--disable-web-search" in command
    assert "--max-turns" in command


def test_build_grok_build_prompt_json_downscales_under_default_limit(tmp_path):
    from module.providers.grok_build_headless import build_grok_build_prompt_json

    image = Image.new("RGB", (1024, 768), (255, 0, 0))
    image_path = tmp_path / "large.png"
    image.save(image_path)

    prepared = build_grok_build_prompt_json(
        image_path=image_path,
        prompt="Return JSON.",
        mime="image/png",
        max_chars=24000,
    )

    blocks = json.loads(prepared.prompt_json)
    assert prepared.prompt_json_chars <= 24000
    assert blocks[1]["type"] == "image"
    assert blocks[1]["mimeType"] == "image/jpeg"
    assert blocks[1]["data"]


@pytest.mark.parametrize("source_mime", ["image/avif", "image/webp", "image/heic", "image/svg+xml"])
def test_build_grok_build_prompt_json_transcodes_image_mime_to_jpeg(source_mime, tmp_path):
    from module.providers.grok_build_headless import build_grok_build_prompt_json

    image_path = _write_image(tmp_path / "image.png")

    prepared = build_grok_build_prompt_json(
        image_path=image_path,
        prompt="Return JSON.",
        mime=source_mime,
        max_chars=24000,
    )

    blocks = json.loads(prepared.prompt_json)
    assert prepared.mime_type == "image/jpeg"
    assert blocks[1]["mimeType"] == "image/jpeg"
    assert blocks[1]["data"]


def test_build_grok_build_prompt_json_rejects_non_image_mime(tmp_path):
    from module.providers.grok_build_headless import GrokBuildHeadlessError, build_grok_build_prompt_json

    image_path = _write_image(tmp_path / "image.png")

    with pytest.raises(GrokBuildHeadlessError) as exc_info:
        build_grok_build_prompt_json(
            image_path=image_path,
            prompt="Return JSON.",
            mime="application/pdf",
            max_chars=24000,
        )

    assert exc_info.value.kind == "unsupported_media"


def test_run_grok_build_headless_caption_success_and_clears_api_keys(monkeypatch, tmp_path):
    from module.providers.grok_build_headless import GrokBuildHeadlessConfig, run_grok_build_headless_caption

    fake_grok = _make_fake_grok(
        tmp_path,
        """
import json
import os
import sys

if os.environ.get("XAI_API_KEY") or os.environ.get("GROK_CODE_XAI_API_KEY"):
    sys.exit(42)
prompt_json = sys.argv[sys.argv.index("--prompt-json") + 1]
blocks = json.loads(prompt_json)
if blocks[0]["type"] != "text":
    sys.exit(43)
if blocks[1]["type"] != "image" or blocks[1]["mimeType"] != "image/jpeg" or not blocks[1]["data"]:
    sys.exit(44)
payload = {
    "short_description": "fake short",
    "long_description": "fake long",
    "tags": ["fake"],
    "rating": "general",
    "confidence": 0.8,
}
print(json.dumps({"text": json.dumps(payload)}))
""",
    )
    image = _write_image(tmp_path / "image.png")
    monkeypatch.setenv("XAI_API_KEY", "ambient-api-key")

    result = run_grok_build_headless_caption(
        GrokBuildHeadlessConfig(command=str(fake_grok), timeout=10),
        image_path=image,
        prompt="caption",
        mime="image/png",
        env={"GROK_CODE_XAI_API_KEY": "base-env-api-key"},
    )

    assert result.parsed["short_description"] == "fake short"
    assert result.parsed["long_description"] == "fake long"
    assert result.returncode == 0


def test_run_grok_build_headless_caption_classifies_image_input_failure(tmp_path):
    from module.providers.grok_build_headless import (
        GrokBuildHeadlessConfig,
        GrokBuildHeadlessError,
        run_grok_build_headless_caption,
    )

    fake_grok = _make_fake_grok(
        tmp_path,
        """
import sys
sys.stderr.write("Invalid ACP content blocks: missing field `data`")
sys.exit(2)
""",
    )
    image = _write_image(tmp_path / "image.jpg")

    with pytest.raises(GrokBuildHeadlessError) as exc_info:
        run_grok_build_headless_caption(
            GrokBuildHeadlessConfig(command=str(fake_grok), timeout=10),
            image_path=image,
            prompt="caption",
            mime="image/jpeg",
        )

    assert exc_info.value.kind == "image_input_unsupported"


def test_run_grok_build_headless_caption_timeout_returns_timeout_kind(tmp_path):
    from module.providers.grok_build_headless import (
        GrokBuildHeadlessConfig,
        GrokBuildHeadlessError,
        run_grok_build_headless_caption,
    )

    fake_grok = _make_fake_grok(
        tmp_path,
        """
import time
time.sleep(2)
""",
    )
    image = _write_image(tmp_path / "image.jpg")

    with pytest.raises(GrokBuildHeadlessError) as exc_info:
        run_grok_build_headless_caption(
            GrokBuildHeadlessConfig(command=str(fake_grok), timeout=0.01),
            image_path=image,
            prompt="caption",
            mime="image/jpeg",
        )

    assert exc_info.value.kind == "timeout"


def test_run_grok_build_headless_caption_timeout_kills_child_process(tmp_path):
    from module.providers.grok_build_headless import (
        GrokBuildHeadlessConfig,
        GrokBuildHeadlessError,
        run_grok_build_headless_caption,
    )

    fake_grok = _make_fake_grok(
        tmp_path,
        """
import os
import subprocess
import sys
import time

child_code = (
    "import os, time;"
    "from pathlib import Path;"
    "time.sleep(0.8);"
    "Path(os.environ['GROK_CHILD_MARKER']).write_text('child survived', encoding='utf-8')"
)
subprocess.Popen([sys.executable, "-c", child_code])
time.sleep(5)
""",
    )
    marker = tmp_path / "child-survived.txt"
    image = _write_image(tmp_path / "image.jpg")

    with pytest.raises(GrokBuildHeadlessError) as exc_info:
        run_grok_build_headless_caption(
            GrokBuildHeadlessConfig(command=str(fake_grok), timeout=0.2),
            image_path=image,
            prompt="caption",
            mime="image/jpeg",
            env={"GROK_CHILD_MARKER": str(marker)},
        )

    assert exc_info.value.kind == "timeout"
    time.sleep(1.2)
    assert not marker.exists()


def test_parse_grok_build_output_accepts_json_envelope():
    from module.providers.grok_build_headless import parse_grok_build_output

    parsed = parse_grok_build_output(json.dumps({"text": json.dumps(_caption_payload())}))

    assert parsed["short_description"] == "short"
    assert parsed["long_description"] == "long"
    assert parsed["tags"] == ["tag"]


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        ("grok: command not found", "environment"),
        ("Not logged in. Please sign in.", "auth"),
        ("You've reached your usage limit. Try again at 10:00.", "usage_limit"),
        ("Too many requests, rate limit exceeded.", "rate_limited"),
        ("unknown variant localImage, expected one of text, image", "image_input_unsupported"),
        ("request timed out", "timeout"),
    ],
)
def test_classify_grok_build_failure(message, kind):
    from module.providers.grok_build_headless import classify_grok_build_failure

    assert classify_grok_build_failure(message) == kind


def test_grok_build_subscription_route_requires_explicit_flag():
    from module.providers.registry import get_registry

    reg = get_registry()

    assert reg.find_provider(make_provider_args(), "image/jpeg") is None
    provider = reg.find_provider(make_provider_args(grok_build_subscription=True), "image/jpeg")
    assert provider is not None
    assert provider.name == "grok_build_subscription"


@pytest.mark.parametrize("source_mime", ["image/avif", "image/webp", "image/heic", "image/svg+xml"])
def test_grok_build_subscription_route_accepts_image_mime_for_transcode(source_mime):
    from module.providers.registry import get_registry

    reg = get_registry()

    provider = reg.find_provider(make_provider_args(grok_build_subscription=True), source_mime)
    assert provider is not None
    assert provider.name == "grok_build_subscription"


def test_grok_build_subscription_does_not_steal_explicit_ocr_or_vlm_routes():
    from module.providers.registry import get_registry

    reg = get_registry()

    provider = reg.find_provider(
        make_provider_args(grok_build_subscription=True, ocr_model="paddle_ocr", document_image=True),
        "image/jpeg",
    )
    assert provider is not None
    assert provider.name == "paddle_ocr"

    provider = reg.find_provider(
        make_provider_args(grok_build_subscription=True, vlm_image_model="moondream"),
        "image/jpeg",
    )
    assert provider is not None
    assert provider.name == "moondream"


def test_grok_build_subscription_attempt_uses_headless_backend(monkeypatch, tmp_path):
    from module.providers.base import MediaContext, MediaModality, PromptContext, ProviderContext
    from module.providers.cloud_vlm import grok_build_subscription
    from module.providers.cloud_vlm.grok_build_subscription import GrokBuildSubscriptionProvider

    image = _write_image(tmp_path / "image.jpg")
    calls = {}

    def fake_caption(config, *, image_path, prompt, mime):
        calls["config"] = config
        calls["image_path"] = image_path
        calls["prompt"] = prompt
        calls["mime"] = mime
        return SimpleNamespace(parsed=_caption_payload("fake short", "fake long"), prompt_json_chars=123)

    monkeypatch.setattr(grok_build_subscription, "run_grok_build_headless_caption", fake_caption)
    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={},
        args=make_provider_args(grok_build_subscription=True),
    )
    provider = GrokBuildSubscriptionProvider(ctx)
    result = provider.attempt(
        MediaContext(uri=str(image), mime="image/jpeg", sha256hash="", modality=MediaModality.IMAGE),
        PromptContext(system="system", user="user"),
    )

    assert result.parsed["short_description"] == "fake short"
    assert result.metadata["provider"] == "grok_build_subscription"
    assert result.metadata["backend"] == "headless"
    assert result.metadata["auth_mode"] == "cached_token"
    assert result.metadata["prompt_json_chars"] == 123
    assert calls["config"].model == "grok-build"
    assert calls["mime"] == "image/jpeg"
    assert "Do not read files" in calls["prompt"]


def test_grok_build_subscription_timeout_returns_empty_result(monkeypatch, tmp_path):
    from module.providers.base import MediaContext, MediaModality, PromptContext, ProviderContext
    from module.providers.cloud_vlm import grok_build_subscription
    from module.providers.cloud_vlm.grok_build_subscription import GrokBuildSubscriptionProvider
    from module.providers.grok_build_headless import GrokBuildHeadlessError

    image = _write_image(tmp_path / "image.jpg")

    def fake_caption(config, *, image_path, prompt, mime):
        raise GrokBuildHeadlessError("timeout", kind="timeout")

    monkeypatch.setattr(grok_build_subscription, "run_grok_build_headless_caption", fake_caption)
    ctx = ProviderContext(
        console=Console(file=io.StringIO(), force_terminal=False),
        config={},
        args=make_provider_args(grok_build_subscription=True),
    )
    provider = GrokBuildSubscriptionProvider(ctx)
    result = provider.attempt(
        MediaContext(uri=str(image), mime="image/jpeg", sha256hash="", modality=MediaModality.IMAGE),
        PromptContext(system="", user=""),
    )

    assert result.raw == ""
    assert result.metadata["skip_reason"] == "timeout"
    assert result.metadata["error_kind"] == "timeout"


@pytest.mark.skipif(os.environ.get("RUN_LIVE_GROK_BUILD") != "1", reason="live Grok Build smoke is opt-in")
def test_live_grok_build_headless_caption_smoke():
    from module.providers.grok_build_headless import GrokBuildHeadlessConfig, run_grok_build_headless_caption

    image = ROOT / "third_party" / "dots.ocr" / "demo" / "demo_image1.jpg"
    result = run_grok_build_headless_caption(
        GrokBuildHeadlessConfig(timeout=180),
        image_path=image,
        prompt="Return JSON with short_description, long_description, tags, rating, and confidence for this image.",
        mime="image/jpeg",
    )

    assert result.parsed["short_description"]
