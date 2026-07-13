from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .options import BatchOptions
from .outputs import atomic_output_path

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
