# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "torch==2.8.0+cu128; sys_platform in 'win32,linux' and platform_machine in 'AMD64,x86_64'",
#     "torchaudio==2.8.0+cu128; sys_platform in 'win32,linux' and platform_machine in 'AMD64,x86_64'",
#     "torch==2.8.0; sys_platform not in 'win32,linux' or platform_machine not in 'AMD64,x86_64'",
#     "torchaudio==2.8.0; sys_platform not in 'win32,linux' or platform_machine not in 'AMD64,x86_64'",
#     "transformers==4.45.2",
#     "huggingface-hub==0.36.0",
#     "safetensors==0.5.3",
#     "numpy==1.24.3",
#     "scipy==1.13.1",
#     "mir-eval==0.8.2",
#     "pretty-midi==0.2.10",
#     "mido==1.3.3",
#     "setuptools==78.1.1",
#     "filelock>=3.16,<4",
# ]
#
# [tool.uv.sources]
# torch = { index = "cu128", marker = "sys_platform in 'win32,linux' and platform_machine in 'AMD64,x86_64'" }
# torchaudio = { index = "cu128", marker = "sys_platform in 'win32,linux' and platform_machine in 'AMD64,x86_64'" }
#
# [[tool.uv.index]]
# name = "cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///

"""Transcribe audio with the official SheetSage2 model in an isolated uv script."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

MODEL_ID = "m-a-p/SheetSage2"
MODEL_REVISION = "b83c3d79a6d98bcffaf8f9ed112c90aa887887bc"
METADATA_NAME = "metadata.json"
REQUIRED_OUTPUTS = {"transcription.mid", "events.json"}
COMPLETE_OUTPUTS = REQUIRED_OUTPUTS | {"score.abc"}
NATIVE_OUTPUTS = COMPLETE_OUTPUTS | {
    "result.json",
    "playback.json",
    "events.tsv",
    "melody.mid",
    "melody_vocal.mid",
    "melody_instrumental.mid",
    "chords.mid",
    "melody_full.lab",
    "melody_vocal.lab",
    "melody_instrumental.lab",
    "chord.lab",
    "key.lab",
    "structure.lab",
    "rhythm_events.lab",
    "beat.lab",
    "downbeat.lab",
}


@dataclass(frozen=True)
class _BatchJob:
    source: Path
    output_dir: Path
    relative_path: Path


def load_model(requested_device: str):
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required to read audio files; install it and add it to PATH.")
    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "huggingface"))

    import torch
    from transformers import AutoModel

    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    dtype = "fp32"
    if device.type == "cuda":
        if not torch.cuda.is_available() or (device.index is not None and device.index >= torch.cuda.device_count()):
            raise RuntimeError(f"CUDA device {device} is not available; try --device=cpu.")
        with torch.cuda.device(device):
            if torch.cuda.is_bf16_supported(including_emulation=False):
                dtype = "bf16"
    torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f"Loading {MODEL_ID} on {device} ({dtype}); first run downloads SheetSage2 and MERT-v2-FullSong.", flush=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        code_revision=MODEL_REVISION,
        use_safetensors=True,
        trust_remote_code=True,
    ).eval().to(device)
    return model, dtype


def _owned_native_outputs(target: Path, source: Path) -> frozenset[str]:
    if target.resolve() != target:
        raise ValueError(f"SheetSage2 output directory redirects elsewhere: {target}")
    if not target.exists():
        return frozenset()
    if not target.is_dir():
        raise NotADirectoryError(target)
    # Probe the publication paths themselves so case aliases are recognized on Windows and macOS.
    entries = {
        name: path for name in NATIVE_OUTPUTS | {METADATA_NAME}
        if (path := target / name).exists() or path.is_symlink()
    }
    if not entries:
        return frozenset()
    marker = entries.pop(METADATA_NAME, None)
    if marker is None or marker.is_symlink() or not marker.is_file():
        raise ValueError(f"Refusing to change unowned SheetSage2 outputs without regular metadata: {target}")
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid SheetSage2 metadata: {marker}")
    signature = metadata.get("signature")
    files = metadata.get("files")
    if metadata.get("status") == "silent":
        signature, files = metadata, {}
    if not (
        isinstance(signature, dict) and signature.get("model") == MODEL_ID
        and isinstance(signature.get("source"), str) and Path(signature["source"]) == source
        and isinstance(files, dict) and files.keys() <= NATIVE_OUTPUTS
    ):
        raise ValueError(f"Existing SheetSage2 outputs do not belong to {source}: {target}")
    if not entries.keys() <= files.keys() or any(p.is_symlink() or not p.is_file() for p in entries.values()):
        raise ValueError(f"Refusing to change unowned or redirected native outputs: {target}")
    return frozenset(files)


def _has_complete_result(target: Path, signature: dict, *, require_abc: bool) -> bool:
    try:
        metadata = json.loads((target / METADATA_NAME).read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("signature") != signature:
            return False
        if require_abc and metadata.get("abc_error"):
            return False
        required = COMPLETE_OUTPUTS if require_abc else REQUIRED_OUTPUTS
        files = metadata.get("files")
        if not isinstance(files, dict) or not required <= files.keys() <= NATIVE_OUTPUTS:
            return False
        # Edited scores and MIDI are still complete exports.
        return all((target / name).is_file() for name in files) and all(
            (target / name).stat().st_size > 0 for name in required
        )
    except (OSError, ValueError):
        return False


def _transcribe_item(
    model, source: Path, target: Path, *, dtype: str, melody_only: bool, signature: dict, stem_mode: bool,
) -> tuple[Path, str | None]:
    from module.music_export.service import publish_staged_outputs

    def progress(event):
        if event["stage"] == "encoding":
            print(f"  Window {event['window']}/{event['windows']}", flush=True)

    target.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".sheetsage2-", dir=target) as temporary:
        stage = Path(temporary)
        try:
            result = model.transcribe(source, output_dir=stage, dtype=dtype, melody_only=melody_only, progress=progress)
        except RuntimeError as exc:
            # Upstream may complete MIDI and annotations before raising for a missing melody-only ABC.
            result = getattr(exc, "result", None)
            if not (melody_only and isinstance(result, dict) and not result.get("abc") and result.get("abc_error")):
                raise
        files = {name: (stage / name).stat().st_size for name in sorted(NATIVE_OUTPUTS) if (stage / name).is_file()}
        if any(not files.get(name) for name in REQUIRED_OUTPUTS):
            raise RuntimeError("SheetSage2 did not produce both MIDI and events.json.")
        abc_error = None
        if not files.get("score.abc"):
            abc_error = result.get("abc_error") or "SheetSage2 did not produce an ABC score."
            (stage / "score.abc").unlink(missing_ok=True)
            files.pop("score.abc", None)
        published = target
        owned = _owned_native_outputs(target, source)
        previous_score = bool(abc_error and (target / "score.abc").is_file())
        if previous_score and not stem_mode:
            published = target / "partial"
            owned = _owned_native_outputs(published, source)
            published.mkdir(exist_ok=True)
        (stage / METADATA_NAME).write_text(
            json.dumps({"signature": signature, "files": files, "abc_error": abc_error}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        archived = publish_staged_outputs(
            stage, published, managed_names=owned | files.keys(), metadata_name=METADATA_NAME,
            preserve_previous=previous_score and stem_mode,
        )
        if archived:
            print(f"Previous SheetSage2 outputs archived at {archived}", flush=True)
    return published, abc_error


def _run_jobs(
    jobs: list[_BatchJob],
    *,
    device: str,
    melody_only: bool,
    skip_completed: bool,
    stem_mode: bool = False,
) -> int:
    from filelock import FileLock, Timeout

    from utils.file_hash import sha256_file

    model = None
    completed = skipped = failed = 0
    for index, job in enumerate(jobs, start=1):
        output = job.output_dir
        target = output / job.relative_path
        label = f"[{index}/{len(jobs)}] {job.relative_path}"
        lock = FileLock(output / ".sheetsage2.lock", timeout=0) if stem_mode else nullcontext()
        try:
            with lock:
                try:
                    if not target.resolve().is_relative_to(output):
                        raise ValueError(f"Output path escapes the output directory: {target}")
                    _owned_native_outputs(target, job.source)
                    signature = {
                        "version": 1,
                        "model": MODEL_ID,
                        "revision": MODEL_REVISION,
                        "source": str(job.source),
                        "sha256": sha256_file(job.source),
                        "device": device,
                        "melody_only": melody_only,
                    }
                    if skip_completed and _has_complete_result(target, signature, require_abc=not stem_mode):
                        print(f"{label}: skipping completed output", flush=True)
                        skipped += 1
                        continue
                except (OSError, ValueError) as exc:
                    print(f"{label}: {exc}", file=sys.stderr, flush=True)
                    failed += 1
                    continue

                if model is None:
                    model, dtype = load_model(device)
                print(f"{label}: transcribing", flush=True)
                try:
                    published, abc_error = _transcribe_item(
                        model, job.source, target, dtype=dtype, melody_only=melody_only, signature=signature,
                        stem_mode=stem_mode,
                    )
                except Exception as exc:
                    print(f"{label}: {exc}", file=sys.stderr, flush=True)
                    failed += 1
                else:
                    if abc_error:
                        print(f"{label}: ABC unavailable ({abc_error}); MIDI and annotations kept.", file=sys.stderr)
                    if abc_error and not stem_mode:
                        failed += 1
                    else:
                        completed += 1
                    print(f"Saved to {published}", flush=True)
        except Timeout:
            print(f"{label}: output directory is busy: {output}", file=sys.stderr, flush=True)
            failed += 1
    print(f"SheetSage2: {completed} completed, {skipped} skipped, {failed} incomplete or failed.", flush=True)
    return 1 if failed else 0


def run_batch(
    source: Path,
    *,
    output_dir: Path | None,
    device: str,
    melody_only: bool,
    skip_completed: bool,
) -> int:
    from filelock import FileLock, Timeout

    from module.audio_batch import discover_inputs

    source = source.expanduser().resolve()
    parent = source if source.is_dir() else source.parent
    output = (output_dir or parent / "sheetsage2_output").expanduser().resolve()
    items = discover_inputs(source, output_dir=output, recursive=True, skip_separation_outputs=True)
    if not items:
        raise ValueError(f"No supported audio files found: {source}")
    output.mkdir(parents=True, exist_ok=True)
    jobs = [_BatchJob(item.source_path, output, item.relative_path) for item in items]
    try:
        with FileLock(output / ".sheetsage2.lock", timeout=0):
            return _run_jobs(jobs, device=device, melody_only=melody_only, skip_completed=skip_completed)
    except Timeout as exc:
        raise RuntimeError(f"SheetSage2 output directory is busy: {output}") from exc


def _stem_output_dir(source: Path) -> Path:
    from module.audio_batch import STEM_MIDI_OUTPUT_DIRNAME
    from utils.path_safety import safe_child_path

    midi_dir = safe_child_path(source.parent, STEM_MIDI_OUTPUT_DIRNAME)
    return safe_child_path(midi_dir, "sheetsage2")


def archive_silent_stem_outputs(source: Path) -> tuple[Path, ...]:
    from filelock import FileLock, Timeout

    from module.music_export.service import publish_staged_outputs

    source = source.resolve()
    output = _stem_output_dir(source)
    target = output / source.name
    if not target.exists() and not target.is_symlink():
        return ()
    archived = []
    try:
        with FileLock(output / ".sheetsage2.lock", timeout=0):
            generations = [(current, _owned_native_outputs(current, source)) for current in (target, target / "partial")]
            for current, files in generations:
                if not files:
                    continue
                with TemporaryDirectory(prefix=".silent-", dir=current) as temporary:
                    stage = Path(temporary)
                    (stage / METADATA_NAME).write_text(
                        json.dumps({"status": "silent", "source": str(source), "model": MODEL_ID}, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    previous = publish_staged_outputs(
                        stage, current, managed_names=files,
                        metadata_name=METADATA_NAME, preserve_previous=True,
                    )
                    if previous:
                        archived.append(previous)
    except Timeout as exc:
        raise RuntimeError(f"SheetSage2 output directory is busy: {output}") from exc
    return tuple(archived)


def run_stem_batch(manifest: Path, *, device: str, melody_only: bool, skip_completed: bool) -> int:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    paths = payload.get("stems") if isinstance(payload, dict) else None
    if not isinstance(paths, list) or not paths or not all(isinstance(path, str) and path for path in paths):
        raise ValueError("Invalid SheetSage2 stem manifest")
    jobs: dict[Path, _BatchJob] = {}
    failed = False
    for path in paths:
        try:
            source = Path(path).resolve()
            if source in jobs:
                continue
            output = _stem_output_dir(source)
            output.mkdir(parents=True, exist_ok=True)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"{path}: {exc}", file=sys.stderr, flush=True)
            failed = True
            continue
        jobs[source] = _BatchJob(source, output, Path(source.name))
    if not jobs:
        return 1
    code = _run_jobs(list(jobs.values()), device=device, melody_only=melody_only, skip_completed=skip_completed, stem_mode=True)
    return code or int(failed)


def _device_arg(value: str) -> str:
    if not re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", value):
        raise argparse.ArgumentTypeError("device must be auto, cpu, cuda, or cuda:N")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe audio to ABC, MIDI and timed annotations with SheetSage2.")
    parser.add_argument("input", nargs="?", type=Path, help="An audio file or a directory (scanned recursively)")
    parser.add_argument("--output-dir", type=Path, help="Defaults to sheetsage2_output next to the input audio")
    parser.add_argument("--stem-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", type=_device_arg, default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--melody-only", action="store_true", help="Omit chord accompaniment from the score and combined MIDI")
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.stem_manifest is not None and (args.input is not None or args.output_dir is not None):
        parser.error("--stem-manifest cannot be combined with input or --output-dir")
    if args.input is None and args.stem_manifest is None:
        parser.error("an input audio file or directory is required")
    try:
        if args.stem_manifest is not None:
            return run_stem_batch(
                args.stem_manifest, device=args.device, melody_only=args.melody_only, skip_completed=args.skip_completed,
            )
        return run_batch(
            args.input, output_dir=args.output_dir, device=args.device, melody_only=args.melody_only,
            skip_completed=args.skip_completed,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"SheetSage2: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
