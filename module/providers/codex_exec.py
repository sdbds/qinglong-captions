"""Helpers for the Codex CLI subscription caption provider."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from module.providers.codex_schema import (
    CODEX_CAPTION_SCHEMA,
    CODEX_CAPTION_SCHEMA_VERSION,
    CodexCaptionOutputError,
    build_codex_caption_prompt,
    filter_caption_payload_by_mode,
    load_caption_schema,
    normalize_codex_caption_payload,
    parse_codex_caption_output,
    strip_markdown_json_fence,
    write_default_caption_schema,
)


DEFAULT_CODEX_COMMAND = "codex"
DEFAULT_CODEX_MODEL = "gpt-5.4-mini"
DEFAULT_CODEX_TIMEOUT_SECONDS = 180.0
DEFAULT_CODEX_SANDBOX = "read-only"

API_KEY_ENV_VARS = ("OPENAI_API_KEY", "CODEX_API_KEY")
PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


@dataclass(frozen=True)
class CodexExecConfig:
    command: str = DEFAULT_CODEX_COMMAND
    model: str = DEFAULT_CODEX_MODEL
    timeout: float = DEFAULT_CODEX_TIMEOUT_SECONDS
    sandbox: str = DEFAULT_CODEX_SANDBOX
    codex_home: str = ""
    isolated_cwd: str = ""


@dataclass(frozen=True)
class CodexExecResult:
    raw: str
    parsed: dict
    stdout: str
    stderr: str
    returncode: int
    output_file: str


class CodexExecError(RuntimeError):
    """Raised when a Codex CLI caption request fails."""

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        stdout: str = "",
        stderr: str = "",
        returncode: int | None = None,
    ):
        super().__init__(message)
        self.kind = kind
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def build_codex_exec_command(
    config: CodexExecConfig,
    *,
    image_path: str | Path,
    prompt: str,
    schema_path: str | Path,
    output_path: str | Path,
) -> list[str]:
    return [
        resolve_codex_command(config.command),
        "exec",
        "--cd",
        str(Path(config.isolated_cwd).resolve()),
        "--skip-git-repo-check",
        "--sandbox",
        config.sandbox,
        "--ephemeral",
        "--model",
        config.model,
        "--image",
        str(Path(image_path).resolve()),
        "--output-schema",
        str(Path(schema_path).resolve()),
        "--output-last-message",
        str(Path(output_path).resolve()),
        "-",
    ]


def resolve_codex_command(command: str) -> str:
    trimmed = command.strip() or DEFAULT_CODEX_COMMAND
    if os.name == "nt" and Path(trimmed).suffix == "":
        resolved = shutil.which(trimmed)
        if resolved:
            return resolved
    return trimmed


def build_codex_exec_env(
    base_env: Mapping[str, str] | None = None,
    *,
    codex_home: str = "",
) -> dict[str, str]:
    env = dict(os.environ)
    if base_env:
        env.update(base_env)
    for key in API_KEY_ENV_VARS:
        env.pop(key, None)
    if codex_home.strip():
        env["CODEX_HOME"] = str(Path(codex_home).expanduser())
    for key in PROXY_ENV_VARS:
        lower_key = key.lower()
        if key not in env and lower_key in env:
            env[key] = env[lower_key]
    return env


def classify_codex_failure(output: str, returncode: int | None = None) -> str:
    lower = output.lower()
    if "missing optional dependency" in lower or "@openai/codex-linux" in lower:
        return "environment"
    if "not found" in lower and "codex" in lower:
        return "environment"
    if "not logged in" in lower or "sign in" in lower or "login" in lower and "codex" in lower:
        return "auth"
    if "authentication" in lower or "unauthorized" in lower or "oauth" in lower:
        return "auth"
    if "usage limit" in lower or "subscription usage limit" in lower or "try again at" in lower:
        return "usage_limit"
    if "timed out" in lower or "timeout" in lower:
        return "timeout"
    if returncode == 127:
        return "environment"
    return "execution"


def run_codex_exec_caption(
    config: CodexExecConfig,
    *,
    image_path: str | Path,
    prompt: str,
    schema_path: str | Path,
    output_path: str | Path,
    env: Mapping[str, str] | None = None,
) -> CodexExecResult:
    command = build_codex_exec_command(
        config,
        image_path=image_path,
        prompt=prompt,
        schema_path=schema_path,
        output_path=output_path,
    )
    run_env = build_codex_exec_env(env, codex_home=config.codex_home)
    try:
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=config.timeout,
            env=run_env,
            check=False,
        )
    except FileNotFoundError as exc:
        raise CodexExecError(
            f"Codex command not found: {config.command}",
            kind="environment",
        ) from exc
    except OSError as exc:
        raise CodexExecError(
            f"Codex command could not start: {exc}",
            kind="environment",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        raise CodexExecError(
            f"Codex exec timed out after {config.timeout:g}s.",
            kind="timeout",
            stdout=stdout,
            stderr=stderr,
        ) from exc

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if completed.returncode != 0:
        combined = "\n".join(part for part in (stdout, stderr) if part)
        kind = classify_codex_failure(combined, completed.returncode)
        raise CodexExecError(
            f"Codex exec failed ({kind}, exit={completed.returncode}).",
            kind=kind,
            stdout=stdout,
            stderr=stderr,
            returncode=completed.returncode,
        )

    output_file = Path(output_path)
    raw = output_file.read_text(encoding="utf-8").strip() if output_file.exists() else ""
    if not raw:
        raw = stdout.strip()
    try:
        parsed = parse_codex_caption_output(raw)
    except CodexCaptionOutputError as exc:
        raise CodexExecError(str(exc), kind="output", stdout=exc.raw or raw) from exc
    return CodexExecResult(
        raw=raw,
        parsed=parsed,
        stdout=stdout,
        stderr=stderr,
        returncode=completed.returncode,
        output_file=str(output_file),
    )
