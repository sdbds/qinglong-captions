"""Helpers for the Grok Build headless subscription caption provider."""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from PIL import Image

from module.providers.codex_schema import CODEX_CAPTION_SCHEMA, CodexCaptionOutputError, parse_codex_caption_output

DEFAULT_GROK_BUILD_COMMAND = "grok"
DEFAULT_GROK_BUILD_MODEL = "grok-4.5"
DEFAULT_GROK_BUILD_TIMEOUT_SECONDS = 180.0
DEFAULT_GROK_BUILD_PERMISSION_MODE = "dontAsk"
DEFAULT_GROK_BUILD_SANDBOX = "read-only"
DEFAULT_GROK_BUILD_PROMPT_JSON_MAX_CHARS = 24000
GROK_BUILD_PROMPT_IMAGE_MIME = "image/jpeg"

API_KEY_ENV_VARS = ("XAI_API_KEY", "GROK_CODE_XAI_API_KEY")
PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
_ENCODE_MAX_SIZES = (512, 384, 256, 192, 128)
_ENCODE_QUALITIES = (85, 75, 65, 55)


@dataclass(frozen=True)
class GrokBuildHeadlessConfig:
    command: str = DEFAULT_GROK_BUILD_COMMAND
    model: str = DEFAULT_GROK_BUILD_MODEL
    reasoning_effort: str = ""
    disable_web_search: bool = True
    timeout: float = DEFAULT_GROK_BUILD_TIMEOUT_SECONDS
    isolated_cwd: str = ""
    permission_mode: str = DEFAULT_GROK_BUILD_PERMISSION_MODE
    sandbox: str = DEFAULT_GROK_BUILD_SANDBOX
    prompt_json_max_chars: int = DEFAULT_GROK_BUILD_PROMPT_JSON_MAX_CHARS


@dataclass(frozen=True)
class GrokBuildHeadlessPrompt:
    prompt_json: str
    mime_type: str
    image_base64: str
    prompt_json_chars: int


@dataclass(frozen=True)
class GrokBuildHeadlessResult:
    raw: str
    parsed: dict
    stdout: str
    stderr: str
    returncode: int
    prompt_json_chars: int


class GrokBuildHeadlessError(RuntimeError):
    """Raised when a Grok Build headless caption request fails."""

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


def resolve_grok_build_command(command: str) -> str:
    trimmed = command.strip() or DEFAULT_GROK_BUILD_COMMAND
    if os.name == "nt" and Path(trimmed).suffix == "":
        resolved = shutil.which(trimmed)
        if resolved:
            return resolved
    return trimmed


def build_grok_build_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    if base_env:
        env.update(base_env)
    for key in API_KEY_ENV_VARS:
        env.pop(key, None)
    for key in PROXY_ENV_VARS:
        lower_key = key.lower()
        if key not in env and lower_key in env:
            env[key] = env[lower_key]
    return env


def normalize_grok_build_mime(mime: str) -> str:
    normalized = str(mime or "").split(";", 1)[0].strip().lower()
    if normalized == "image/jpg":
        return "image/jpeg"
    return normalized


def is_grok_build_source_image_mime(mime: str) -> bool:
    return normalize_grok_build_mime(mime).startswith("image/")


def build_grok_build_prompt_blocks(prompt: str, image_base64: str, mime_type: str) -> list[dict[str, str]]:
    return [
        {"type": "text", "text": prompt},
        {"type": "image", "data": image_base64, "mimeType": mime_type},
    ]


def build_grok_build_prompt_json(
    *,
    image_path: str | Path,
    prompt: str,
    mime: str,
    max_chars: int = DEFAULT_GROK_BUILD_PROMPT_JSON_MAX_CHARS,
) -> GrokBuildHeadlessPrompt:
    if not is_grok_build_source_image_mime(mime):
        raise GrokBuildHeadlessError(
            f"Unsupported Grok Build source mime: {mime}; expected image/*",
            kind="unsupported_media",
        )

    max_chars = _positive_int(max_chars, DEFAULT_GROK_BUILD_PROMPT_JSON_MAX_CHARS)
    best_prompt_json = ""
    image = _load_rgb_image(image_path)
    for max_size in _ENCODE_MAX_SIZES:
        resized = _resize_image(image, max_size)
        for quality in _ENCODE_QUALITIES:
            image_base64 = _encode_jpeg_base64(resized, quality=quality)
            blocks = build_grok_build_prompt_blocks(prompt, image_base64, GROK_BUILD_PROMPT_IMAGE_MIME)
            prompt_json = json.dumps(blocks, ensure_ascii=False, separators=(",", ":"))
            best_prompt_json = prompt_json
            if len(prompt_json) <= max_chars:
                return GrokBuildHeadlessPrompt(
                    prompt_json=prompt_json,
                    mime_type=GROK_BUILD_PROMPT_IMAGE_MIME,
                    image_base64=image_base64,
                    prompt_json_chars=len(prompt_json),
                )

    raise GrokBuildHeadlessError(
        f"Grok Build prompt-json payload is too large for safe inline argv ({len(best_prompt_json)} chars > {max_chars}).",
        kind="input_too_large",
    )


def build_grok_build_command(
    config: GrokBuildHeadlessConfig,
    *,
    prompt_json: str,
    json_schema: str | None = None,
) -> list[str]:
    command = [resolve_grok_build_command(config.command)]
    model = (config.model or "").strip()
    if model:
        command.extend(["--model", model])
    reasoning_effort = (config.reasoning_effort or "").strip()
    if reasoning_effort:
        command.extend(["--reasoning-effort", reasoning_effort])
    isolated_cwd = (config.isolated_cwd or "").strip()
    if isolated_cwd:
        command.extend(["--cwd", str(Path(isolated_cwd).expanduser().resolve())])
    permission_mode = (config.permission_mode or "").strip()
    if permission_mode:
        command.extend(["--permission-mode", permission_mode])
    sandbox = (config.sandbox or "").strip()
    if sandbox:
        command.extend(["--sandbox", sandbox])
    command.extend(
        [
            "--prompt-json",
            prompt_json,
        ]
    )
    if json_schema:
        command.extend(["--json-schema", json_schema])
    command.extend(
        [
            "--output-format",
            "json",
            "--no-auto-update",
        ]
    )
    if config.disable_web_search:
        command.append("--disable-web-search")
    command.extend(["--max-turns", "1"])
    return command


def parse_grok_build_output(text: str) -> dict:
    raw = str(text or "").strip()
    if not raw:
        raise CodexCaptionOutputError("Grok Build output was empty.", raw=text)
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        return parse_codex_caption_output(raw)

    if isinstance(envelope, dict):
        structured_output = envelope.get("structuredOutput")
        if isinstance(structured_output, dict):
            return parse_codex_caption_output(json.dumps(structured_output, ensure_ascii=False))
        text_value = envelope.get("text")
        if text_value not in (None, ""):
            return parse_codex_caption_output(str(text_value))
        for key in ("output", "content", "response", "message"):
            value = envelope.get(key)
            if isinstance(value, str) and value.strip():
                return parse_codex_caption_output(value)
        return parse_codex_caption_output(json.dumps(envelope, ensure_ascii=False))
    raise CodexCaptionOutputError("Grok Build output JSON must be an object.", raw=text)


def classify_grok_build_failure(output: str, returncode: int | None = None) -> str:
    lower = output.lower()
    if "not found" in lower and "grok" in lower:
        return "environment"
    if returncode == 127:
        return "environment"
    if "not logged in" in lower or "sign in" in lower or "login required" in lower:
        return "auth"
    if "authentication" in lower or "unauthorized" in lower or "oauth" in lower:
        return "auth"
    if "xai_api_key" in lower or "grok_code_xai_api_key" in lower:
        return "api_key_billing_mode"
    if "usage limit" in lower or "try again at" in lower:
        return "usage_limit"
    if "rate limit" in lower or "too many requests" in lower:
        return "rate_limited"
    if "invalid acp content blocks" in lower or "missing field `data`" in lower:
        return "image_input_unsupported"
    if "unknown variant" in lower and "localimage" in lower:
        return "image_input_unsupported"
    if "timed out" in lower or "timeout" in lower:
        return "timeout"
    return "execution"


def run_grok_build_headless_caption(
    config: GrokBuildHeadlessConfig,
    *,
    image_path: str | Path,
    prompt: str,
    mime: str,
    structured: bool = True,
    env: Mapping[str, str] | None = None,
) -> GrokBuildHeadlessResult:
    prepared = build_grok_build_prompt_json(
        image_path=image_path,
        prompt=prompt,
        mime=mime,
        max_chars=config.prompt_json_max_chars,
    )
    json_schema = (
        json.dumps(CODEX_CAPTION_SCHEMA, ensure_ascii=False, separators=(",", ":")) if structured else None
    )
    command = build_grok_build_command(config, prompt_json=prepared.prompt_json, json_schema=json_schema)
    run_env = build_grok_build_env(env)
    try:
        completed = _run_grok_build_process(command, timeout=config.timeout, env=run_env)
    except FileNotFoundError as exc:
        raise GrokBuildHeadlessError(
            f"Grok Build command not found: {config.command}",
            kind="environment",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        raise GrokBuildHeadlessError(
            f"Grok Build headless request timed out after {config.timeout:g}s.",
            kind="timeout",
            stdout=stdout,
            stderr=stderr,
        ) from exc
    except OSError as exc:
        kind = "input_too_large" if getattr(exc, "winerror", None) == 206 else "environment"
        raise GrokBuildHeadlessError(
            f"Grok Build command could not start: {exc}",
            kind=kind,
        ) from exc

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if completed.returncode != 0:
        combined = "\n".join(part for part in (stdout, stderr) if part)
        kind = classify_grok_build_failure(combined, completed.returncode)
        raise GrokBuildHeadlessError(
            f"Grok Build headless request failed ({kind}, exit={completed.returncode}).",
            kind=kind,
            stdout=stdout,
            stderr=stderr,
            returncode=completed.returncode,
        )

    raw = stdout.strip()
    if not structured:
        # Freeform template path: return raw output verbatim, no rating schema parse.
        return GrokBuildHeadlessResult(
            raw=raw,
            parsed={},
            stdout=stdout,
            stderr=stderr,
            returncode=completed.returncode,
            prompt_json_chars=prepared.prompt_json_chars,
        )
    try:
        parsed = parse_grok_build_output(raw)
    except CodexCaptionOutputError as exc:
        raise GrokBuildHeadlessError(
            str(exc),
            kind="output",
            stdout=exc.raw or raw,
            stderr=stderr,
            returncode=completed.returncode,
        ) from exc
    return GrokBuildHeadlessResult(
        raw=raw,
        parsed=parsed,
        stdout=stdout,
        stderr=stderr,
        returncode=completed.returncode,
        prompt_json_chars=prepared.prompt_json_chars,
    )


def _run_grok_build_process(command: list[str], *, timeout: float, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    popen_kwargs = {
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "env": env,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    process = subprocess.Popen(command, **popen_kwargs)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_tree(process)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            exc.cmd,
            exc.timeout,
            output=stdout or exc.stdout,
            stderr=stderr or exc.stderr,
        ) from exc

    return subprocess.CompletedProcess(
        args=command,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if _terminate_process_tree_with_psutil(process.pid):
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        process.kill()


def _terminate_process_tree_with_psutil(pid: int) -> bool:
    try:
        import psutil
    except ImportError:
        return False

    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return True

    processes = parent.children(recursive=True)
    processes.append(parent)
    for child in processes:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    _gone, alive = psutil.wait_procs(processes, timeout=5)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    return True


def _load_rgb_image(image_path: str | Path) -> Image.Image:
    try:
        with Image.open(image_path) as image:
            image.load()
            if "xmp" in image.info:
                del image.info["xmp"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image.copy()
    except FileNotFoundError as exc:
        raise GrokBuildHeadlessError(f"Grok Build image file not found: {image_path}", kind="image") from exc
    except Image.UnidentifiedImageError as exc:
        raise GrokBuildHeadlessError(f"Grok Build image file is not a supported image: {image_path}", kind="image") from exc
    except OSError as exc:
        raise GrokBuildHeadlessError(f"Grok Build could not read image: {image_path}: {exc}", kind="image") from exc


def _resize_image(image: Image.Image, max_size: int) -> Image.Image:
    width, height = image.size
    largest = max(width, height)
    if largest <= max_size:
        return image.copy()
    scale = max_size / largest
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return image.resize((new_width, new_height), Image.LANCZOS)


def _encode_jpeg_base64(image: Image.Image, *, quality: int) -> str:
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _positive_int(value: object, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default
