import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console


def _console_buffer():
    buffer = io.StringIO()
    return Console(file=buffer, force_terminal=False, color_system=None, width=200), buffer


def _job(mime: str = "image/png"):
    return SimpleNamespace(mime=mime)


def _args(**overrides):
    defaults = {
        "codex_subscription": False,
        "codex_home": "",
        "grok_build_subscription": False,
        "kimi_code_api_key": "",
        "ocr_model": "",
        "document_image": False,
        "vlm_image_model": "",
        "subscription_quota_timeout": 10.0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_startup_quota_reports_codex_live_cli_rate_limit(monkeypatch):
    from module.providers import subscription_quota

    def fake_run(command, **kwargs):
        assert command[0] == "codex-test"
        assert "exec" in command
        assert "--json" in command
        assert kwargs["input"] == subscription_quota.CODEX_QUOTA_PROMPT
        assert kwargs["timeout"] == 3.0
        stdout = json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "rate_limits": {
                        "primary": {"used_percent": 24.0, "resets_at": 1773173847},
                        "secondary": {"used_percent": 6.0, "resets_at": 1773760647},
                        "plan_type": "team",
                    }
                },
            }
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subscription_quota.subprocess, "run", fake_run)

    console, buffer = _console_buffer()
    subscription_quota.report_startup_subscription_quota(
        _args(
            codex_subscription=True,
            codex_command="codex-test",
            subscription_quota_timeout=3.0,
        ),
        [_job()],
        console,
    )

    output = buffer.getvalue()
    assert "Subscription quota (codex_subscription)" in output
    assert "Status: live command" in output
    assert "5-hour window" in output
    assert "[■■■■■■■■■■■■■■■░░░░░] 76.0% remaining (24.0% used)" in output
    assert "Weekly window" in output
    assert "[■■■■■■■■■■■■■■■■■■■░] 94.0% remaining (6.0% used)" in output
    assert "Plan: Team" in output
    assert "plan=team" not in output
    assert "#" not in output


def test_startup_quota_reports_codex_last_known_rate_limit(tmp_path, monkeypatch):
    from module.providers import subscription_quota

    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    codex_home = tmp_path / ".codex"
    session_dir = codex_home / "sessions" / "2026" / "06" / "03"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "rollout.jsonl"
    session_file.write_text(
        json.dumps({"type": "message", "payload": {"text": "older"}})
        + "\n"
        + json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "rate_limits": {
                        "primary": {"used_percent": 24.0, "resets_at": 1773173847},
                        "secondary": {"used_percent": 6.0, "resets_at": 1773760647},
                        "plan_type": "team",
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(subscription_quota.subprocess, "run", fake_run)
    monkeypatch.setattr(subscription_quota.Path, "home", lambda: tmp_path)

    console, buffer = _console_buffer()
    subscription_quota.report_startup_subscription_quota(
        _args(codex_subscription=True, codex_home=str(codex_home)),
        [_job()],
        console,
    )

    output = buffer.getvalue()
    assert "Checking subscription quota at startup" in output
    assert "Subscription quota (codex_subscription)" in output
    assert "Status: live command timed out after 10s; fallback last known" in output
    assert "5-hour window" in output
    assert "[■■■■■■■■■■■■■■■░░░░░] 76.0% remaining (24.0% used)" in output
    assert "Weekly window" in output
    assert "[■■■■■■■■■■■■■■■■■■■░] 94.0% remaining (6.0% used)" in output
    assert "Plan: Team" in output
    assert "plan=team" not in output
    assert "#" not in output


def test_startup_quota_rejects_live_rate_limits_without_percent(tmp_path, monkeypatch):
    from module.providers import subscription_quota

    def fake_run(command, **kwargs):
        stdout = json.dumps({"payload": {"rate_limits": {"credits": None, "plan_type": "team"}}})
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subscription_quota.subprocess, "run", fake_run)
    monkeypatch.setattr(subscription_quota.Path, "home", lambda: tmp_path)

    console, buffer = _console_buffer()
    subscription_quota.report_startup_subscription_quota(
        _args(codex_subscription=True, codex_command="codex-test"),
        [_job()],
        console,
    )

    output = buffer.getvalue()
    assert "live command returned rate_limits without percent fields" in output
    assert "fallback no local last-known rate limit found" in output


def test_startup_quota_reports_unavailable_sources_without_blocking():
    from module.providers.subscription_quota import report_startup_subscription_quota

    console, buffer = _console_buffer()
    report_startup_subscription_quota(
        _args(grok_build_subscription=True, kimi_code_api_key="sk-test"),
        [_job()],
        console,
    )

    output = buffer.getvalue()
    assert "Subscription quota (grok_build_subscription)" in output
    assert "no stable quota CLI command found" in output
    assert "Subscription quota (kimi_code)" in output
    assert "no stable Kimi Code quota CLI/API endpoint found" in output


def test_startup_quota_ignores_subscription_when_explicit_vlm_route_selected():
    from module.providers.subscription_quota import report_startup_subscription_quota

    console, buffer = _console_buffer()
    report_startup_subscription_quota(
        _args(codex_subscription=True, grok_build_subscription=True, vlm_image_model="gemini"),
        [_job()],
        console,
    )

    assert buffer.getvalue() == ""
