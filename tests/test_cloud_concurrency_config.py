import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_captioner_parser_accepts_cloud_max_concurrency():
    from module.captioner import setup_parser

    parser = setup_parser()

    default_args = parser.parse_args(["dataset"])
    assert default_args.cloud_max_concurrency == 1
    assert default_args.grok_build_max_concurrency == 1
    assert default_args.grok_build_model_name == "grok-4.5"
    assert default_args.grok_build_disable_web_search is True

    args = parser.parse_args(["dataset", "--cloud_max_concurrency=3"])
    assert args.cloud_max_concurrency == 3


def test_captioner_parser_uses_current_kimi_models_and_k3_effort_mode():
    from module.captioner import setup_parser

    parser = setup_parser()

    default_args = parser.parse_args(["dataset"])
    assert default_args.kimi_model_path == "kimi-k2.6"
    assert default_args.kimi_code_model_path == "k3"
    assert default_args.kimi_code_thinking == "thinking.effort:max"

    args = parser.parse_args(["dataset", "--kimi_code_thinking=reasoning_effort:max"])
    assert args.kimi_code_thinking == "reasoning_effort:max"


def test_captioner_parser_accepts_grok_build_subscription_options():
    from module.captioner import setup_parser

    parser = setup_parser()

    args = parser.parse_args(
        [
            "dataset",
            "--grok_build_subscription",
            "--grok_build_backend=headless",
            "--grok_build_auth_mode=existing",
            "--grok_build_command=grok-test",
            "--grok_build_model_name=grok-build-custom",
            "--grok_build_reasoning_effort=none",
            "--no-grok_build_disable_web_search",
            "--grok_build_timeout=9",
            "--grok_build_isolated_cwd=work",
            "--grok_build_permission_mode=dontAsk",
            "--grok_build_sandbox=read-only",
            "--grok_build_prompt_json_max_chars=12000",
            "--grok_build_max_concurrency=1",
        ]
    )

    assert args.grok_build_subscription is True
    assert args.grok_build_backend == "headless"
    assert args.grok_build_auth_mode == "existing"
    assert args.grok_build_command == "grok-test"
    assert args.grok_build_model_name == "grok-build-custom"
    assert args.grok_build_reasoning_effort == "none"
    assert args.grok_build_disable_web_search is False
    assert args.grok_build_timeout == 9
    assert args.grok_build_isolated_cwd == "work"
    assert args.grok_build_permission_mode == "dontAsk"
    assert args.grok_build_sandbox == "read-only"
    assert args.grok_build_prompt_json_max_chars == 12000
    assert args.grok_build_max_concurrency == 1


def test_captioner_parser_rejects_removed_grok_build_effort_option():
    from module.captioner import setup_parser

    parser = setup_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["dataset", "--grok_build_effort=low"])


def test_captioner_powershell_defaults_python_utf8_without_overriding_user_value():
    script = (ROOT / "4.captioner.ps1").read_text(encoding="utf-8")

    assert "if (-not $Env:PYTHONUTF8) {" in script
    assert '$Env:PYTHONUTF8 = "1"' in script


def test_captioner_powershell_passes_cloud_max_concurrency_only_when_gt_one():
    script = (ROOT / "4.captioner.ps1").read_text(encoding="utf-8")

    assert "$cloud_max_concurrency = 1" in script
    assert "if ($cloud_max_concurrency -gt 1)" in script
    assert '--cloud_max_concurrency=$cloud_max_concurrency' in script


def test_captioner_powershell_uses_current_kimi_models_and_k3_effort():
    script = (ROOT / "4.captioner.ps1").read_text(encoding="utf-8")

    assert '$kimi_model_path = "kimi-k2.6"' in script
    assert '$kimi_code_model_path = "k3"' in script
    assert '$kimi_code_thinking = "thinking.effort:max"' in script
    assert "if ($kimi_code_thinking)" in script
    assert "--kimi_code_thinking=$kimi_code_thinking" in script


def test_captioner_powershell_passes_grok_build_options_only_when_enabled():
    script = (ROOT / "4.captioner.ps1").read_text(encoding="utf-8")

    assert "$grok_build_subscription = $false" in script
    assert '$grok_build_backend = "headless"' in script
    assert '$grok_build_model_name = "grok-4.5"' in script
    assert "$grok_build_disable_web_search = $true" in script
    assert 'if ($grok_build_subscription)' in script
    assert '--grok_build_subscription' in script
    assert '--grok_build_backend=$grok_build_backend' in script
    assert '--grok_build_auth_mode=$grok_build_auth_mode' in script
    assert '--grok_build_command=$grok_build_command' in script
    assert '--grok_build_model_name=$grok_build_model_name' in script
    assert "grok_build_effort" not in script
    assert '--grok_build_reasoning_effort=$grok_build_reasoning_effort' in script
    assert "--grok_build_disable_web_search" in script
    assert "--no-grok_build_disable_web_search" in script
    assert '--grok_build_prompt_json_max_chars=$grok_build_prompt_json_max_chars' in script
