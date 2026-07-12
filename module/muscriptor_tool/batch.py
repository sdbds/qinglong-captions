from __future__ import annotations

import importlib.metadata
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from filelock import FileLock, Timeout

from .manifest import (
    SCHEMA_VERSION,
    atomic_write_json,
    cleanup_temporary_outputs,
    is_item_complete,
    prune_known_outputs,
    run_signature,
)
from .options import BatchOptions, OutputFormat
from .outputs import OutputTargets, TranscriptionResult, transcribe_once
from .runtime import load_model, resolve_device

SUPPORTED_AUDIO_EXTENSIONS = frozenset({".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"})
DEFAULT_OUTPUT_DIRNAME = "muscriptor_output"


class OutputDirectoryBusyError(RuntimeError):
    """Raised when another batch owns the output directory lock."""


def default_output_dir(input_path: Path) -> Path:
    """Place batch artifacts beside a file or inside an input directory."""
    source = Path(input_path).expanduser().resolve()
    parent = source if source.is_dir() else source.parent
    return parent / DEFAULT_OUTPUT_DIRNAME


@dataclass(frozen=True)
class BatchItem:
    source_path: Path
    relative_path: Path


@dataclass(frozen=True)
class BatchItemResult:
    source: str
    status: str
    metadata_path: str | None = None
    error: str | None = None


@dataclass
class BatchSummary:
    discovered: int = 0
    processed: int = 0
    skipped: int = 0
    partial: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0
    items: list[BatchItemResult] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return 1 if self.partial or self.failed else 0


def _is_supported_audio(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


def discover_inputs(
    input_path: Path,
    *,
    output_dir: Path,
    recursive: bool,
) -> tuple[BatchItem, ...]:
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if input_path.is_file():
        if not _is_supported_audio(input_path):
            raise ValueError(f"Unsupported audio file: {input_path}")
        return (BatchItem(source_path=input_path, relative_path=Path(input_path.name)),)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if input_path == output_dir:
        raise ValueError("input directory and output directory must be different")

    found: list[BatchItem] = []
    if recursive:
        for directory, directory_names, file_names in os.walk(input_path, followlinks=False):
            directory_path = Path(directory)
            retained_directories: list[str] = []
            for name in sorted(directory_names, key=str.casefold):
                candidate = directory_path / name
                if candidate.is_symlink() or candidate.resolve() == output_dir:
                    continue
                retained_directories.append(name)
            directory_names[:] = retained_directories
            for name in sorted(file_names, key=str.casefold):
                candidate = directory_path / name
                if candidate.is_symlink() or not _is_supported_audio(candidate):
                    continue
                found.append(BatchItem(candidate.resolve(), candidate.resolve().relative_to(input_path)))
    else:
        for candidate in sorted(input_path.iterdir(), key=lambda path: path.name.casefold()):
            if candidate.is_symlink() or not _is_supported_audio(candidate):
                continue
            found.append(BatchItem(candidate.resolve(), Path(candidate.name)))

    return tuple(sorted(found, key=lambda item: item.relative_path.as_posix().casefold()))


def item_output_dir(output_dir: Path, item: BatchItem) -> Path:
    return Path(output_dir) / item.relative_path.parent / item.relative_path.name


def _requested_names(options: BatchOptions) -> set[str]:
    names = {
        OutputFormat.MIDI: "transcription.mid",
        OutputFormat.JSON: "events.json",
        OutputFormat.JSONL: "events.jsonl",
    }
    requested = {names[item] for item in options.output_formats}
    if options.preview is not None:
        requested.add(f"preview.{options.preview.format.value}")
    return requested


def _default_package_version() -> str:
    try:
        return importlib.metadata.version("muscriptor")
    except importlib.metadata.PackageNotFoundError:
        return "0.2.1"


def _default_resolved_device(requested: str) -> str:
    import torch

    return resolve_device(requested, torch)


def _relative_outputs(result: TranscriptionResult, item_dir: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for key, value in result.outputs.items():
        path = Path(value)
        try:
            outputs[key] = path.resolve().relative_to(item_dir.resolve()).as_posix()
        except ValueError:
            outputs[key] = str(path)
    return outputs


def _existing_outputs(
    targets: OutputTargets,
    item_dir: Path,
    preview_target: Path | None = None,
) -> dict[str, str]:
    outputs = {
        key: path.resolve().relative_to(item_dir.resolve()).as_posix()
        for key, path in targets.requested_paths().items()
        if path.is_file()
    }
    if preview_target is not None and preview_target.is_file():
        outputs["preview"] = preview_target.name
    return outputs


def _metadata_payload(
    *,
    item: BatchItem,
    options: BatchOptions,
    signature: str,
    package_version: str,
    requested_device: str,
    resolved_device: str,
    status: str,
    elapsed_seconds: float,
    result: TranscriptionResult | None,
    outputs: dict[str, str],
    error: BaseException | None,
    preview_runtime: Any | None,
) -> dict[str, Any]:
    stat = item.source_path.stat()
    preview_payload = options.preview.as_dict() if options.preview else None
    if preview_payload is not None:
        soundfont_path = getattr(preview_runtime, "soundfont_path", None)
        soundfont = {
            "source": "default",
            "signature_id": str(
                getattr(preview_runtime, "renderer_id", "muscriptor-0.2.1:SF2_URL")
            ),
        }
        if soundfont_path is not None:
            resolved_soundfont = Path(soundfont_path).expanduser().resolve()
            soundfont_stat = resolved_soundfont.stat()
            soundfont.update(
                {
                    "resolved_path": str(resolved_soundfont),
                    "size": soundfont_stat.st_size,
                    "mtime_ns": soundfont_stat.st_mtime_ns,
                }
            )
        preview_payload["soundfont"] = soundfont
    return {
        "schema_version": SCHEMA_VERSION,
        "source_path": item.relative_path.as_posix(),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "status": status,
        "run_signature": signature,
        "muscriptor_version": package_version,
        "model_variant": options.transcription.model.value,
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "instruments": list(options.transcription.instruments),
        "options": {
            **options.transcription.as_dict(),
            "output_formats": [item.value for item in options.output_formats],
            "preview": preview_payload,
        },
        "representation": {
            "fields": ["onset", "offset", "pitch", "instrument"],
            "velocity": "not_transcribed",
            "same_pitch_overlap": "not_representable",
            "drums": "onset_only_with_minimum_duration",
        },
        "note_count": result.note_count if result else 0,
        "event_count": result.event_count if result else 0,
        "chunk_count": result.chunk_count if result else 0,
        "outputs": outputs,
        "warnings": list(result.warnings) if result else [],
        "elapsed_seconds": round(elapsed_seconds, 6),
        "error": (
            {
                "type": type(error).__name__,
                "stage": "transcription_or_output",
                "message": str(error),
            }
            if error is not None
            else None
        ),
    }


def _manifest_payload(
    *,
    input_path: Path,
    output_dir: Path,
    options: BatchOptions,
    summary: BatchSummary,
    started_at: datetime,
    package_version: str | None,
    resolved_device: str | None,
    run_error: BaseException | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(Path(input_path)),
        "output_dir": str(Path(output_dir)),
        "muscriptor_version": package_version,
        "model_variant": options.transcription.model.value,
        "requested_device": options.transcription.device,
        "resolved_device": resolved_device,
        "options": options.as_dict(),
        "counts": {
            "discovered": summary.discovered,
            "processed": summary.processed,
            "skipped": summary.skipped,
            "partial": summary.partial,
            "failed": summary.failed,
        },
        "items": [item.__dict__ for item in summary.items],
        "elapsed_seconds": round(summary.elapsed_seconds, 6),
        "error": (
            {
                "type": type(run_error).__name__,
                "stage": "batch_setup_or_model",
                "message": str(run_error),
            }
            if run_error is not None
            else None
        ),
    }


def run_batch(
    input_path: Path,
    output_dir: Path | None = None,
    options: BatchOptions | None = None,
    *,
    model_loader: Callable[[Any], Any] | None = None,
    transcriber: Callable[..., TranscriptionResult] | None = None,
    preview_preflight: Callable[..., Any] | None = None,
    package_version: str | None = None,
    resolved_device: str | None = None,
    log_callback: Callable[[str], None] | None = None,
    chunk_progress_callback: Callable[[str, int, int], None] | None = None,
) -> BatchSummary:
    options = options or BatchOptions()
    model_loader = model_loader or load_model
    transcriber = transcriber or transcribe_once
    input_path = Path(input_path).expanduser()
    output_dir = (
        default_output_dir(input_path)
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(output_dir / ".muscriptor.lock"), timeout=0)
    started_at = datetime.now(timezone.utc)
    start_clock = time.perf_counter()
    summary = BatchSummary()
    run_error: BaseException | None = None

    try:
        lock.acquire()
    except Timeout as exc:
        raise OutputDirectoryBusyError(f"Output directory is already in use: {output_dir}") from exc

    try:
        items = discover_inputs(input_path, output_dir=output_dir, recursive=options.recursive)
        if not items:
            raise ValueError(f"No supported audio files found: {input_path}")
        summary.discovered = len(items)
        if log_callback is not None:
            log_callback(f"Discovered {summary.discovered} supported audio file(s)")
        package_version = package_version or _default_package_version()
        resolved_device = resolved_device or _default_resolved_device(options.transcription.device)
        requested_names = _requested_names(options)
        pending: list[tuple[BatchItem, str]] = []

        for item in items:
            item_dir = item_output_dir(output_dir, item)
            signature = run_signature(
                item,
                options,
                package_version=package_version,
                resolved_device=resolved_device,
            )
            if (
                options.skip_completed
                and is_item_complete(
                    item_dir / "metadata.json",
                    signature=signature,
                    requested_names=requested_names,
                )
            ):
                summary.skipped += 1
                if log_callback is not None:
                    log_callback(f"Skipping completed input: {item.relative_path.as_posix()}")
                summary.items.append(
                    BatchItemResult(
                        source=item.relative_path.as_posix(),
                        status="skipped",
                        metadata_path=str(item_dir / "metadata.json"),
                    )
                )
                continue
            pending.append((item, signature))

        if not pending:
            if log_callback is not None:
                log_callback("All discovered inputs are already complete")
            summary.elapsed_seconds = time.perf_counter() - start_clock
            return summary

        preview_runtime = None
        if options.preview is not None:
            if preview_preflight is None:
                from .auralization import preflight_preview

                preview_preflight = preflight_preview
            if log_callback is not None:
                log_callback(
                    f"Checking {options.preview.content.value} {options.preview.format.value} preview runtime"
                )
            preview_runtime = preview_preflight(options.preview)

        try:
            if log_callback is not None:
                log_callback(
                    f"Loading official MuScriptor {options.transcription.model.value} model on {resolved_device}"
                )
            loaded = model_loader(options.transcription)
        except Exception as exc:
            run_error = exc
            summary.failed = len(pending)
            if log_callback is not None:
                log_callback(f"Model load failed: {type(exc).__name__}: {exc}")
            raise
        if log_callback is not None:
            log_callback(f"Resolved device: {getattr(loaded, 'resolved_device', resolved_device)}")

        for item, signature in pending:
            item_started = time.perf_counter()
            item_dir = item_output_dir(output_dir, item)
            item_dir.mkdir(parents=True, exist_ok=True)
            cleanup_temporary_outputs(item_dir)
            prune_known_outputs(item_dir, requested_names=set())
            targets = OutputTargets.for_directory(item_dir, options.output_formats)
            preview_target = (
                item_dir / f"preview.{options.preview.format.value}"
                if options.preview is not None
                else None
            )
            result: TranscriptionResult | None = None
            error: BaseException | None = None
            status = "ok"
            if log_callback is not None:
                log_callback(f"Processing input: {item.relative_path.as_posix()}")
            try:
                kwargs = (
                    {"preview_runtime": preview_runtime, "preview_target": preview_target}
                    if preview_runtime is not None
                    else {}
                )
                relative_path = item.relative_path.as_posix()
                if chunk_progress_callback is not None:
                    kwargs["progress_callback"] = (
                        lambda completed, total, relative=relative_path: chunk_progress_callback(
                            relative,
                            completed,
                            total,
                        )
                    )
                elif log_callback is not None:
                    kwargs["progress_callback"] = (
                        lambda completed, total, relative=relative_path: log_callback(
                            f"Processing {relative} chunk {completed}/{total}"
                        )
                    )
                result = transcriber(loaded, item.source_path, options.transcription, targets, **kwargs)
                missing = [name for name in requested_names if not (item_dir / name).is_file()]
                if missing:
                    raise RuntimeError(f"Requested outputs were not created: {', '.join(sorted(missing))}")
                prune_known_outputs(item_dir, requested_names=requested_names)
                outputs = _relative_outputs(result, item_dir)
                summary.processed += 1
                if log_callback is not None:
                    log_callback(
                        f"Completed {item.relative_path.as_posix()}: {', '.join(sorted(outputs.values()))}"
                    )
            except Exception as exc:
                error = exc
                partial_result = getattr(exc, "result", None)
                if isinstance(partial_result, TranscriptionResult):
                    result = partial_result
                outputs = _existing_outputs(targets, item_dir, preview_target)
                if outputs:
                    status = "partial"
                    summary.partial += 1
                else:
                    status = "failed"
                    summary.failed += 1
                if log_callback is not None:
                    log_callback(
                        f"Failed {item.relative_path.as_posix()} during transcription_or_output: "
                        f"{type(exc).__name__}: {exc}"
                    )

            metadata_path = item_dir / "metadata.json"
            atomic_write_json(
                metadata_path,
                _metadata_payload(
                    item=item,
                    options=options,
                    signature=signature,
                    package_version=package_version,
                    requested_device=options.transcription.device,
                    resolved_device=resolved_device,
                    status=status,
                    elapsed_seconds=time.perf_counter() - item_started,
                    result=result,
                    outputs=outputs,
                    error=error,
                    preview_runtime=preview_runtime,
                ),
            )
            summary.items.append(
                BatchItemResult(
                    source=item.relative_path.as_posix(),
                    status=status,
                    metadata_path=str(metadata_path),
                    error=str(error) if error else None,
                )
            )
            if error is not None and options.fail_fast:
                break

        summary.elapsed_seconds = time.perf_counter() - start_clock
        return summary
    except BaseException as exc:
        if run_error is None:
            run_error = exc
        raise
    finally:
        summary.elapsed_seconds = time.perf_counter() - start_clock
        try:
            atomic_write_json(
                output_dir / "manifest.json",
                _manifest_payload(
                    input_path=input_path,
                    output_dir=output_dir,
                    options=options,
                    summary=summary,
                    started_at=started_at,
                    package_version=package_version,
                    resolved_device=resolved_device,
                    run_error=run_error,
                ),
            )
        finally:
            lock.release()
