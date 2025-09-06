# -*- coding: utf-8 -*-
"""
Utilities for Gemini file upload and activation waiting.
Logs are printed via the provided Rich Console when available.
"""
from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, List, Tuple

from google.genai import types
from utils.stream_util import sanitize_filename


def upload_to_gemini(client: Any, path: str, mime_type: str | None = None, name: str | None = None):
    """Uploads the given file to Gemini.

    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    original_name = Path(path).name
    safe_name = sanitize_filename(original_name if name is None else name)

    file = client.files.upload(
        file=path,
        config=types.UploadFileConfig(
            name=safe_name,
            mime_type=mime_type,
            display_name=original_name,
        ),
    )
    return file


def wait_for_files_active(client: Any, files: List[types.File], output_console: Any | None = None) -> None:
    """Waits for the given files to be ACTIVE state."""
    if output_console:
        output_console.print("[yellow]Waiting for file processing...[/yellow]")
    for name in (file.name for file in files):
        file = client.files.get(name=name)
        while file.state.name != "ACTIVE":
            if file.state.name == "FAILED":
                raise Exception(f"File {file.name} failed to process")
            if output_console:
                output_console.print(".", end="", style="yellow")
            time.sleep(10)
            file = client.files.get(name=name)
        if output_console:
            output_console.print()  # New line after dots
    if output_console:
        output_console.print("[green]...all files ready[/green]")
        output_console.print()


def upload_or_get(
    client: Any,
    uri: str,
    mime: str,
    sha256hash: str,
    max_retries: int,
    wait_time: float,
    output_console: Any | None = None,
) -> Tuple[bool, List[types.File]]:
    """Try to reuse an existing uploaded file if hash/size matches; otherwise upload.

    Returns (upload_success, files)
    """
    upload_success = False
    files: List[types.File] = []

    for upload_attempt in range(max_retries):
        try:
            if output_console:
                output_console.print()
                output_console.print(f"[blue]checking files for:[/blue] {uri}")
            try:
                file = client.files.get(name=sanitize_filename(Path(uri).name))
                if output_console:
                    output_console.print(file)
                # Prefer checking sha256 hash; fallback to size match
                try:
                    if base64.b64decode(file.sha256_hash).decode("utf-8") == sha256hash or file.size_bytes == Path(uri).stat().st_size:
                        if output_console:
                            output_console.print()
                            output_console.print(f"[cyan]File {file.name} is already at {file.uri}[/cyan]")
                        files = [file]
                        wait_for_files_active(client, files, output_console)
                        if output_console:
                            output_console.print()
                            output_console.print(f"[green]File {file.name} is already active at {file.uri}[/green]")
                        upload_success = True
                        break
                    else:
                        if output_console:
                            output_console.print(
                                f"[yellow]File {file.name} exists but hash/size mismatch. Deleting and re-uploading...[/yellow]"
                            )
                        client.files.delete(name=sanitize_filename(Path(uri).name))
                        # Proceed to upload in outer except branch
                        raise Exception("Delete same name file and retry")
                except Exception:
                    # If file.sha256_hash is not present or decode fails, fallback to upload path
                    raise

            except Exception:
                if output_console:
                    output_console.print()
                    output_console.print(f"[yellow]File {Path(uri).name} is not exist[/yellow]")
                    output_console.print(f"[blue]uploading files for:[/blue] {uri}")
                try:
                    files = [upload_to_gemini(client, uri, mime_type=mime)]
                except Exception:
                    # Name collision fallback with timestamp suffix
                    files = [
                        upload_to_gemini(
                            client,
                            uri,
                            mime_type=mime,
                            name=f"{Path(uri).name}_{int(time.time())}",
                        )
                    ]
                wait_for_files_active(client, files, output_console)
                upload_success = True
                break

        except Exception as e:
            if output_console:
                output_console.print(
                    f"[yellow]Upload attempt {upload_attempt + 1}/{max_retries} failed: {e}[/yellow]"
                )
            if upload_attempt < max_retries - 1:
                time.sleep(wait_time * 2)
            else:
                if output_console:
                    output_console.print("[red]All upload attempts failed[/red]")
                return False, []

    if not upload_success:
        if output_console:
            output_console.print("[red]Failed to upload file[/red]")
        return False, []

    return True, files
