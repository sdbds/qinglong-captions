from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from module.music_export import atomic_output_path
from utils.file_hash import sha256_file

from .options import BatchOptions

SCHEMA_VERSION = 2
KNOWN_OUTPUT_NAMES = frozenset(
    {
        "events.json",
        "events.jsonl",
        "preview.wav",
        "preview.mp3",
    }
)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with atomic_output_path(Path(path)) as temporary:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def publish_staged_outputs(
    staging_dir: Path,
    item_dir: Path,
    *,
    managed_names: Iterable[str],
    metadata_name: str,
) -> None:
    """Publish one output generation, restoring the old generation on I/O failure."""
    names = set(managed_names) | {metadata_name}
    if any(Path(name).name != name or name in {"", ".", ".."} for name in names):
        raise ValueError("Output names must be leaf names")
    item_dir = Path(item_dir)
    if not (Path(staging_dir) / metadata_name).is_file():
        raise ValueError("Staged output metadata is required")
    if any((item_dir / name).exists() and not (item_dir / name).is_file() for name in names):
        raise ValueError("An output destination is not a regular file")
    backup = Path(tempfile.mkdtemp(prefix=".previous-", dir=item_dir))
    moved: list[str] = []
    installed: list[str] = []
    try:
        # Invalidate the completion marker before changing any generation's artifacts.
        if (item_dir / metadata_name).exists():
            os.replace(item_dir / metadata_name, backup / metadata_name)
            moved.append(metadata_name)
        for name in sorted(names - {metadata_name}):
            old = item_dir / name
            if old.exists():
                os.replace(old, backup / name)
                moved.append(name)
            new = Path(staging_dir) / name
            if new.is_file():
                os.replace(new, old)
                installed.append(name)
        os.replace(Path(staging_dir) / metadata_name, item_dir / metadata_name)
        installed.append(metadata_name)
    except BaseException:
        try:
            for name in reversed(installed):
                (item_dir / name).unlink(missing_ok=True)
            for name in reversed(moved):
                os.replace(backup / name, item_dir / name)
        except OSError as exc:
            raise RuntimeError(f"Output rollback failed; previous outputs are retained in {backup}") from exc
        shutil.rmtree(backup, ignore_errors=True)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def run_signature(
    item: Any,
    options: BatchOptions,
    *,
    package_version: str,
    resolved_device: str,
    renderer_id: str = "muscriptor-0.2.1:SF2_URL",
) -> str:
    stat = Path(item.source_path).stat()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_relative_path": Path(item.relative_path).as_posix(),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "source_sha256": sha256_file(item.source_path),
        "muscriptor_version": package_version,
        "model_variant": options.transcription.model.value,
        "requested_device": options.transcription.device,
        "resolved_device": resolved_device,
        "instruments": list(options.transcription.instruments),
        "decode_mode": options.transcription.decode_mode.value,
        "temperature": options.transcription.temperature,
        "cfg_coef": options.transcription.cfg_coef,
        "batch_size": options.transcription.batch_size,
        "strict_eos": options.transcription.strict_eos,
        "beam_size": options.transcription.beam_size,
        "output_formats": sorted(item.value for item in options.output_formats),
        "preview": options.preview.as_dict() if options.preview else None,
        "renderer_id": renderer_id if options.preview else None,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def is_item_complete(
    metadata_path: Path,
    *,
    signature: str,
    requested_names: Iterable[str],
) -> bool:
    metadata_path = Path(metadata_path)
    payload = read_json(metadata_path)
    if payload is None or payload.get("schema_version") != SCHEMA_VERSION:
        return False
    if payload.get("status") != "ok" or payload.get("run_signature") != signature:
        return False
    item_dir = metadata_path.parent
    return all((item_dir / name).is_file() for name in requested_names)


def _known_output_names(output_stem: str | None = None) -> set[str]:
    names = set(KNOWN_OUTPUT_NAMES)
    if output_stem:
        names.add(f"{output_stem}.mid")
        # Preview files are named after the source item; include both formats
        # so disabling/changing preview mode removes stale artifacts.
        names.update({f"{output_stem}_preview.wav", f"{output_stem}_preview.mp3"})
    return names


def prune_known_outputs(
    item_dir: Path,
    *,
    requested_names: set[str],
    output_stem: str | None = None,
) -> None:
    item_dir = Path(item_dir)
    for name in _known_output_names(output_stem) - set(requested_names):
        (item_dir / name).unlink(missing_ok=True)


def cleanup_temporary_outputs(item_dir: Path, *, output_stem: str | None = None) -> None:
    item_dir = Path(item_dir)
    if not item_dir.is_dir():
        return
    known_stems = {Path(name).stem for name in _known_output_names(output_stem)} | {
        "metadata",
        "manifest",
    }
    temporary_name = re.compile(
        rf"^(?:{'|'.join(re.escape(stem) for stem in sorted(known_stems))})\.\d+\.[0-9a-fA-F]{{32}}\.part(?:\.[^.]+)?$"
    )
    for path in item_dir.iterdir():
        if path.is_file() and temporary_name.fullmatch(path.name):
            path.unlink(missing_ok=True)
