import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_captioner_parser_accepts_cloud_max_concurrency():
    from module.captioner import setup_parser

    parser = setup_parser()

    default_args = parser.parse_args(["dataset"])
    assert default_args.cloud_max_concurrency == 1
    assert default_args.grok_build_max_concurrency == 1

    args = parser.parse_args(["dataset", "--cloud_max_concurrency=3"])
    assert args.cloud_max_concurrency == 3


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
    assert args.grok_build_timeout == 9
    assert args.grok_build_isolated_cwd == "work"
    assert args.grok_build_permission_mode == "dontAsk"
    assert args.grok_build_sandbox == "read-only"
    assert args.grok_build_prompt_json_max_chars == 12000
    assert args.grok_build_max_concurrency == 1


def test_captioner_powershell_passes_cloud_max_concurrency_only_when_gt_one():
    script = (ROOT / "4.captioner.ps1").read_text(encoding="utf-8")

    assert "$cloud_max_concurrency = 1" in script
    assert "if ($cloud_max_concurrency -gt 1)" in script
    assert '--cloud_max_concurrency=$cloud_max_concurrency' in script


def test_captioner_powershell_passes_grok_build_options_only_when_enabled():
    script = (ROOT / "4.captioner.ps1").read_text(encoding="utf-8")

    assert "$grok_build_subscription = $false" in script
    assert '$grok_build_backend = "headless"' in script
    assert 'if ($grok_build_subscription)' in script
    assert '--grok_build_subscription' in script
    assert '--grok_build_backend=$grok_build_backend' in script
    assert '--grok_build_auth_mode=$grok_build_auth_mode' in script
    assert '--grok_build_command=$grok_build_command' in script
    assert '--grok_build_model_name=$grok_build_model_name' in script
    assert '--grok_build_prompt_json_max_chars=$grok_build_prompt_json_max_chars' in script
