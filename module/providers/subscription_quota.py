"""Startup subscription quota reporting helpers."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from module.providers.catalog import route_provider_name
from module.providers.codex_exec import (
    build_codex_exec_env,
    normalize_codex_reasoning_effort,
    resolve_codex_command,
)


DEFAULT_SUBSCRIPTION_QUOTA_TIMEOUT_SECONDS = 10.0
CODEX_QUOTA_PROMPT = "Return exactly OK."


@dataclass(frozen=True)
class StartupQuotaReport:
    provider: str
    message: str
    ok: bool = False


@dataclass(frozen=True)
class _CodexCliQuotaProbe:
    rate_limits: dict[str, Any] | None
    message: str
    ok: bool = False


def report_startup_subscription_quota(args: Any, jobs: Iterable[Any], console: Any) -> None:
    providers = _active_subscription_providers(args, jobs)
    if not providers:
        return

    console.print("[bold]Checking subscription quota at startup...[/bold]")
    for provider in providers:
        report = _query_subscription_quota(provider, args)
        style = "green" if report.ok else "yellow"
        console.print(f"[bold]Subscription quota[/bold] [dim]({report.provider})[/dim]")
        console.print(report.message, style=style, markup=False)


def _active_subscription_providers(args: Any, jobs: Iterable[Any]) -> list[str]:
    mimes = [str(getattr(job, "mime", "") or "") for job in jobs]
    has_image = any(mime.startswith("image") for mime in mimes)
    has_image_or_video = any(mime.startswith(("image", "video")) for mime in mimes)
    explicit_ocr = bool(getattr(args, "ocr_model", "") or getattr(args, "document_image", False))
    explicit_vlm = route_provider_name("vlm_image_model", getattr(args, "vlm_image_model", ""))

    providers: list[str] = []
    if bool(getattr(args, "codex_subscription", False)) and has_image and not explicit_ocr and not explicit_vlm:
        providers.append("codex_subscription")
    if bool(getattr(args, "grok_build_subscription", False)) and has_image and not explicit_ocr and not explicit_vlm:
        providers.append("grok_build_subscription")

    kimi_key = bool(getattr(args, "kimi_code_api_key", ""))
    if kimi_key and has_image_or_video and not explicit_ocr and not explicit_vlm:
        providers.append("kimi_code")
    elif kimi_key and has_image_or_video and explicit_vlm == "kimi_code":
        providers.append("kimi_code")

    return providers


def _query_subscription_quota(provider: str, args: Any) -> StartupQuotaReport:
    if provider == "codex_subscription":
        return _query_codex_subscription_quota(args)
    if provider == "grok_build_subscription":
        return StartupQuotaReport(
            provider=provider,
            message="no stable quota CLI command found; ordinary Grok prompts do not expose remaining quota, continuing without blocking",
            ok=False,
        )
    if provider == "kimi_code":
        return StartupQuotaReport(
            provider=provider,
            message="no stable Kimi Code quota CLI/API endpoint found; continuing without blocking",
            ok=False,
        )
    return StartupQuotaReport(provider=provider, message="quota query is not implemented", ok=False)


def _query_codex_subscription_quota(args: Any) -> StartupQuotaReport:
    live = _query_codex_cli_quota(args)
    if live.ok and live.rate_limits:
        formatted = _format_rate_limits(live.rate_limits)
        if not formatted:
            live = _CodexCliQuotaProbe(None, "live command returned rate_limits without percent fields", ok=False)
        else:
            return StartupQuotaReport(
                provider="codex_subscription",
                message="Status: live command\n" + formatted,
                ok=True,
            )

    fallback = _query_codex_last_known_quota(args)
    if fallback.ok:
        return StartupQuotaReport(
            provider="codex_subscription",
            message=f"Status: {live.message}; fallback {fallback.message}",
            ok=True,
        )
    return StartupQuotaReport(
        provider="codex_subscription",
        message=f"{live.message}; fallback {fallback.message}",
        ok=False,
    )


def _query_codex_cli_quota(args: Any) -> _CodexCliQuotaProbe:
    timeout = _subscription_quota_timeout_seconds(args)
    try:
        command = _build_codex_quota_command(args)
    except Exception as exc:
        return _CodexCliQuotaProbe(None, f"live command config failed: {exc}", ok=False)
    try:
        completed = subprocess.run(
            command,
            input=CODEX_QUOTA_PROMPT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            env=build_codex_exec_env(codex_home=str(getattr(args, "codex_home", "") or "")),
            check=False,
        )
    except FileNotFoundError:
        return _CodexCliQuotaProbe(None, "live command not found", ok=False)
    except OSError as exc:
        return _CodexCliQuotaProbe(None, f"live command could not start: {exc}", ok=False)
    except subprocess.TimeoutExpired as exc:
        rate_limits = _find_rate_limits_in_text(_timeout_output(exc))
        if rate_limits:
            return _CodexCliQuotaProbe(rate_limits, "live command returned rate_limits before timeout", ok=True)
        return _CodexCliQuotaProbe(None, f"live command timed out after {timeout:g}s", ok=False)

    output = "\n".join(part for part in (completed.stdout or "", completed.stderr or "") if part)
    rate_limits = _find_rate_limits_in_text(output)
    if rate_limits:
        return _CodexCliQuotaProbe(rate_limits, "live command returned rate_limits", ok=True)
    if completed.returncode != 0:
        return _CodexCliQuotaProbe(
            None,
            f"live command exited {completed.returncode} without rate_limits",
            ok=False,
        )
    return _CodexCliQuotaProbe(None, "live command completed without rate_limits", ok=False)


def _build_codex_quota_command(args: Any) -> list[str]:
    command = [
        resolve_codex_command(str(getattr(args, "codex_command", "") or "codex")),
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--sandbox",
        str(getattr(args, "codex_sandbox", "") or "read-only"),
        "--ephemeral",
        "--model",
        str(getattr(args, "codex_model_name", "") or "gpt-5.4"),
        "-",
    ]
    isolated_cwd = str(getattr(args, "codex_isolated_cwd", "") or "").strip()
    if isolated_cwd:
        command[2:2] = ["--cd", str(Path(isolated_cwd).expanduser().resolve())]

    config_overrides: list[tuple[str, str]] = []
    reasoning_effort = normalize_codex_reasoning_effort(str(getattr(args, "codex_reasoning_effort", "") or "none"))
    if reasoning_effort:
        config_overrides.append(("model_reasoning_effort", reasoning_effort))
    service_tier = str(getattr(args, "codex_service_tier", "") or "").strip()
    if not service_tier and bool(getattr(args, "codex_fast", False)):
        service_tier = "fast"
    if service_tier:
        config_overrides.append(("service_tier", service_tier))

    for key, value in reversed(config_overrides):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        command[1:1] = ["-c", f'{key}="{escaped}"']
    return command


def _query_codex_last_known_quota(args: Any) -> StartupQuotaReport:
    codex_home = _codex_home(args)
    rate_limits = _find_latest_codex_rate_limits(codex_home)
    if not rate_limits:
        return StartupQuotaReport(
            provider="codex_subscription",
            message=f"no local last-known rate limit found under {codex_home}",
            ok=False,
        )

    formatted = _format_rate_limits(rate_limits)
    if not formatted:
        return StartupQuotaReport(
            provider="codex_subscription",
            message="last-known rate limit record did not include percent fields",
            ok=False,
        )
    return StartupQuotaReport(provider="codex_subscription", message="last known\n" + formatted, ok=True)


def _format_rate_limits(rate_limits: dict[str, Any]) -> str:
    parts: list[str] = []
    windows: list[str] = []
    plan_type = rate_limits.get("plan_type")
    if plan_type:
        parts.append(f"Plan: {_format_plan_name(plan_type)}")

    primary = _format_limit_window("5-hour window", rate_limits.get("primary"))
    secondary = _format_limit_window("Weekly window", rate_limits.get("secondary"))
    if primary:
        windows.append(primary)
    if secondary:
        windows.append(secondary)
    if not windows:
        return ""
    parts.extend(windows)

    return "\n".join(parts)


def _codex_home(args: Any) -> Path:
    configured = str(getattr(args, "codex_home", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".codex"


def _find_latest_codex_rate_limits(codex_home: Path) -> dict[str, Any] | None:
    for path in _candidate_codex_jsonl_files(codex_home):
        for line in _tail_lines(path):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            rate_limits = _find_rate_limits(payload)
            if isinstance(rate_limits, dict):
                return rate_limits
    return None


def _candidate_codex_jsonl_files(codex_home: Path, *, limit: int = 50) -> list[Path]:
    candidates: list[Path] = []
    for directory_name in ("sessions", "archived_sessions"):
        directory = codex_home / directory_name
        if not directory.exists():
            continue
        try:
            candidates.extend(path for path in directory.rglob("*.jsonl") if path.is_file())
        except OSError:
            continue
    session_index = codex_home / "session_index.jsonl"
    if session_index.exists():
        candidates.append(session_index)

    candidates = sorted(set(candidates), key=lambda path: _safe_mtime(path), reverse=True)
    return candidates[:limit]


def _tail_lines(path: Path, *, max_bytes: int = 256 * 1024) -> Iterable[str]:
    try:
        with path.open("rb") as file:
            file.seek(0, 2)
            size = file.tell()
            file.seek(max(0, size - max_bytes))
            data = file.read()
    except OSError:
        return []
    text = data.decode("utf-8", errors="replace")
    return reversed([line for line in text.splitlines() if line.strip()])


def _find_rate_limits(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        rate_limits = value.get("rate_limits")
        if isinstance(rate_limits, dict):
            return rate_limits
        for child in value.values():
            found = _find_rate_limits(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_rate_limits(child)
            if found is not None:
                return found
    return None


def _find_rate_limits_in_text(text: str) -> dict[str, Any] | None:
    stripped = str(text or "").strip()
    if not stripped:
        return None

    for line in reversed([line for line in stripped.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        rate_limits = _find_rate_limits(payload)
        if isinstance(rate_limits, dict):
            return rate_limits

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return _find_rate_limits(payload)


def _timeout_output(exc: subprocess.TimeoutExpired) -> str:
    parts: list[str] = []
    for value in (exc.stdout, exc.stderr, getattr(exc, "output", None)):
        if isinstance(value, bytes):
            parts.append(value.decode("utf-8", errors="replace"))
        elif isinstance(value, str):
            parts.append(value)
    return "\n".join(part for part in parts if part)


def _subscription_quota_timeout_seconds(args: Any) -> float:
    for name in ("subscription_quota_timeout", "quota_timeout"):
        value = getattr(args, name, None)
        try:
            timeout = float(value)
        except (TypeError, ValueError):
            continue
        if timeout > 0:
            return timeout
    return DEFAULT_SUBSCRIPTION_QUOTA_TIMEOUT_SECONDS


def _format_limit_window(name: str, payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    try:
        used = float(payload["used_percent"])
    except (KeyError, TypeError, ValueError):
        return ""
    remaining = max(0.0, 100.0 - used)
    pieces = [f"{name:<14} {_quota_bar(remaining)} {remaining:.1f}% remaining ({used:.1f}% used)"]
    reset_at = _format_reset_at(payload.get("resets_at"))
    if reset_at:
        pieces.append(f"resets {reset_at}")
    return ", ".join(pieces)


def _quota_bar(remaining_percent: float, *, cells: int = 20) -> str:
    remaining = min(100.0, max(0.0, remaining_percent))
    filled = int(round((remaining / 100.0) * cells))
    filled = min(cells, max(0, filled))
    return "[" + "■" * filled + "░" * (cells - filled) + "]"


def _format_plan_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = text.replace("_", " ").replace("-", " ")
    return " ".join(part.capitalize() for part in normalized.split())


def _format_reset_at(value: Any) -> str:
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return ""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
