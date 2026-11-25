"""Transcribe original mixes to instrument and drum MIDI tracks with Dettato."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock, Timeout

from module.audio_batch import discover_inputs, item_output_dir
from module.music_export.service import publish_staged_outputs
from module.music_export.validation import validate_midi_file
from utils.file_hash import sha256_file

BACKEND = "kobimusic/dettato"
PIPELINE_REVISION = "51ade0f09f2272f4674166026582185d26baaeaf"
METADATA = "metadata.json"


def load_model(device: str):
    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "huggingface"))
    from dettato import Dettato

    print(f"Loading Dettato on {device} (first use downloads four model stages)...", flush=True)
    return Dettato.from_pretrained(None if device == "auto" else device)


def _check_output_directory(target: Path) -> None:
    if target.resolve() != target:
        raise ValueError(f"Refusing a redirected Dettato output directory: {target}")
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(f"Expected a Dettato output directory, found a file: {target}")


def _owned_metadata(target: Path, source: Path) -> dict | None:
    _check_output_directory(target)
    midi, marker = target / f"{source.stem}.mid", target / METADATA
    if any(p.is_symlink() or (p.exists() and not p.is_file()) for p in (midi, marker)):
        raise ValueError(f"Refusing redirected or non-file Dettato outputs: {target}")
    if not marker.exists():
        if midi.exists():
            raise ValueError(f"Refusing to overwrite a MIDI without Dettato metadata: {midi}")
        return None
    previous = json.loads(marker.read_text(encoding="utf-8"))
    if not (
        isinstance(previous, dict)
        and previous.get("backend") == BACKEND
        and isinstance(previous.get("source"), str)
        and Path(previous["source"]) == source
    ):
        raise ValueError(f"Existing outputs do not belong to this Dettato input: {target}")
    return previous


def _should_skip(target: Path, source: Path, signature: dict, *, skip_completed: bool) -> bool:
    previous = _owned_metadata(target, source)
    if not skip_completed or previous is None or previous.get("signature") != signature:
        return False
    try:
        validate_midi_file(target / f"{source.stem}.mid")
    except Exception:
        # Mido's decoders do not share one parse-error type. An unreadable cache entry is not complete.
        return False
    return True


def _export(model, source: Path, target: Path, signature: dict) -> None:
    # Pass the original path: upstream uses the recording's native rate to decide whether to run drums.
    result = model.transcribe(str(source))
    _check_output_directory(target)
    target.mkdir(parents=True, exist_ok=True)
    midi_name = f"{source.stem}.mid"
    with TemporaryDirectory(prefix=".dettato-", dir=target) as temporary:
        stage = Path(temporary)
        result.save_midi(str(stage / midi_name))
        validate_midi_file(stage / midi_name)
        metadata = {
            **result.summary(),
            "source": str(source),
            "backend": BACKEND,
            "signature": signature,
            "device": str(model.device),
        }
        (stage / METADATA).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Batches with different output roots can still publish to the same song directory.
        with FileLock(target / ".dettato-publish.lock", timeout=0):
            _owned_metadata(target, source)
            publish_staged_outputs(stage, target, managed_names=[midi_name], metadata_name=METADATA)
    if isinstance(metadata.get("drums"), str):
        print(f"  Drums: {metadata['drums']}", flush=True)
    print(f"  MIDI: {target / midi_name}", flush=True)


def run_batch(source: Path, output_dir: Path | None, device: str, skip_completed: bool) -> int:
    source = source.expanduser().resolve()
    parent = source if source.is_dir() else source.parent
    output = (output_dir or parent / "dettato_output").expanduser().resolve()
    items = discover_inputs(source, output_dir=output, recursive=True, skip_separation_outputs=True)
    if not items:
        raise ValueError(f"No supported audio files found: {source}")
    weights = os.environ.get("DETTATO_WEIGHTS")
    settings = {
        "revision": PIPELINE_REVISION,
        "device": device,
        "weights": str(Path(weights).expanduser().resolve()) if weights else None,
    }
    model = None
    completed = skipped = failed = 0
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / ".dettato.lock", timeout=0):
        for index, item in enumerate(items, 1):
            path = item.source_path
            target = item_output_dir(output, item)
            label = f"[{index}/{len(items)}] {item.relative_path}"
            try:
                signature = {**settings, "sha256": sha256_file(path)}
                if _should_skip(target, path, signature, skip_completed=skip_completed):
                    print(f"{label}: skipping completed output", flush=True)
                    skipped += 1
                    continue
            except (OSError, ValueError) as exc:
                print(f"{label}: {exc}", file=sys.stderr, flush=True)
                failed += 1
                continue
            if model is None:
                try:
                    model = load_model(device)
                except Exception as exc:
                    raise RuntimeError(f"Could not load Dettato on {device}: {exc}") from exc
            print(f"{label}: transcribing on {model.device}", flush=True)
            try:
                _export(model, path, target, signature)
                completed += 1
            except Exception as exc:
                print(f"{label}: {exc}", file=sys.stderr, flush=True)
                failed += 1
    print(f"Dettato: {completed} completed, {skipped} skipped, {failed} failed.", flush=True)
    return int(bool(failed))


def _device_arg(value: str) -> str:
    if not re.fullmatch(r"auto|cpu|mps|cuda(?::[0-9]+)?", value):
        raise argparse.ArgumentTypeError("device must be auto, cpu, mps, cuda, or cuda:N")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="An audio file or a directory (scanned recursively)")
    parser.add_argument("--output-dir", type=Path, help="Defaults to dettato_output next to the input audio")
    parser.add_argument("--device", type=_device_arg, default="auto", help="auto, cpu, cuda, cuda:N, or mps")
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    try:
        return run_batch(args.input, args.output_dir, args.device, args.skip_completed)
    except Timeout as exc:
        print(f"Dettato output directory is busy: {exc.lock_file}", file=sys.stderr)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"Dettato: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
