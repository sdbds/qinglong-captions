from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from utils.path_safety import safe_child_path, safe_leaf_name

from .catalog import OFFICIAL_INSTRUMENT_NAMES
from .manifest import atomic_write_json, read_json
from .options import PreviewRequest, TranscriptionOptions

STEM_MIDI_OUTPUT_DIRNAME = "04_stem_midi"
STEM_MIDI_SCHEMA_VERSION = 1

# Stem labels identify an instrument family, not always one exact MIDI timbre.
# Keep ambiguous families constrained while allowing MuScriptor to choose the subtype.
STEM_INSTRUMENT_FAMILIES: dict[str, tuple[str, ...]] = {
    "vocals": ("voice",),
    "drums": ("drums",),
    "bass": ("acoustic_bass", "electric_bass"),
    "guitar": (
        "acoustic_guitar",
        "clean_electric_guitar",
        "distorted_electric_guitar",
    ),
    "piano": ("acoustic_piano", "electric_piano"),
}


@dataclass(frozen=True)
class StemMidiCandidate:
    source_path: Path
    song_output_dir: Path
    stem_name: str
    input_path: Path


@dataclass(frozen=True)
class StemMidiOutputPaths:
    midi: Path
    metadata: Path
    preview: Path | None = None


@dataclass(frozen=True)
class StemMidiItemResult:
    candidate: StemMidiCandidate
    status: str
    midi_path: Path
    metadata_path: Path
    error: str | None = None


@dataclass
class StemMidiSummary:
    processed: int = 0
    skipped: int = 0
    partial: int = 0
    failed: int = 0
    setup_error: str | None = None
    items: list[StemMidiItemResult] = field(default_factory=list)


def stem_family(stem_name: str) -> str:
    normalized = str(stem_name).strip().lower()
    checks = (
        ("vocals", ("vocal", "voice")),
        ("drums", ("drum", "percussion")),
        ("bass", ("bass",)),
        ("guitar", ("guitar",)),
        ("piano", ("piano", "keyboard", "keys")),
    )
    for family, tokens in checks:
        if any(token in normalized for token in tokens):
            return family
    return "other"


def _canonical_instruments(values: Iterable[str]) -> tuple[str, ...]:
    allowed = set(OFFICIAL_INSTRUMENT_NAMES)
    canonical: list[str] = []
    unknown: list[str] = []
    for value in values:
        name = str(value).strip().lower()
        if not name or name in canonical:
            continue
        if name not in allowed:
            unknown.append(name)
            continue
        canonical.append(name)
    if unknown:
        raise ValueError(f"Unknown MuScriptor instrument(s): {', '.join(unknown)}")
    return tuple(canonical)


def instruments_for_stem(
    stem_name: str,
    *,
    other_instruments: Iterable[str] = (),
) -> tuple[str, ...]:
    family = stem_family(stem_name)
    if family == "other":
        return _canonical_instruments(other_instruments)
    return STEM_INSTRUMENT_FAMILIES[family]


def build_stem_midi_output_dir(song_output_dir: Path) -> Path:
    return safe_child_path(
        song_output_dir,
        STEM_MIDI_OUTPUT_DIRNAME,
        default_name=STEM_MIDI_OUTPUT_DIRNAME,
    )


def build_stem_midi_output_paths(
    candidate: StemMidiCandidate,
    preview: PreviewRequest | None = None,
) -> StemMidiOutputPaths:
    output_dir = build_stem_midi_output_dir(candidate.song_output_dir)
    source_stem = safe_leaf_name(candidate.source_path.stem, default_name="song")
    stem_name = safe_leaf_name(str(candidate.stem_name).strip().lower(), default_name="other")
    output_stem = safe_leaf_name(f"{source_stem}_({stem_name})", default_name="stem")
    return StemMidiOutputPaths(
        midi=safe_child_path(output_dir, f"{output_stem}.mid", default_name="stem.mid"),
        metadata=safe_child_path(
            output_dir,
            f"{output_stem}.metadata.json",
            default_name="stem.metadata.json",
        ),
        preview=(
            safe_child_path(
                output_dir,
                f"{output_stem}.preview.{preview.format.value}",
                default_name=f"stem.preview.{preview.format.value}",
            )
            if preview is not None
            else None
        ),
    )


def _cleanup_stem_outputs(paths: StemMidiOutputPaths) -> None:
    paths.midi.unlink(missing_ok=True)
    paths.midi.with_suffix(".preview.wav").unlink(missing_ok=True)
    paths.midi.with_suffix(".preview.mp3").unlink(missing_ok=True)


def _default_device_resolver(requested: str) -> str:
    import torch

    from .runtime import resolve_device

    return resolve_device(requested, torch)


def _default_model_loader(options: TranscriptionOptions):
    from .runtime import load_model

    return load_model(options)


def _default_transcriber(*args, **kwargs):
    from .outputs import transcribe_once

    return transcribe_once(*args, **kwargs)


def _run_signature(
    candidate: StemMidiCandidate,
    options: TranscriptionOptions,
    *,
    resolved_device: str,
    preview: PreviewRequest | None,
) -> str:
    stat = candidate.input_path.stat()
    payload = {
        "schema_version": STEM_MIDI_SCHEMA_VERSION,
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "stem_name": candidate.stem_name,
        "resolved_device": resolved_device,
        "options": options.as_dict(),
        "preview": preview.as_dict() if preview is not None else None,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _is_complete(paths: StemMidiOutputPaths, signature: str) -> bool:
    metadata = read_json(paths.metadata)
    return bool(
        paths.midi.is_file()
        and metadata
        and metadata.get("schema_version") == STEM_MIDI_SCHEMA_VERSION
        and metadata.get("status") == "ok"
        and metadata.get("run_signature") == signature
        and (paths.preview is None or paths.preview.is_file())
    )


def _metadata_payload(
    candidate: StemMidiCandidate,
    paths: StemMidiOutputPaths,
    options: TranscriptionOptions,
    *,
    signature: str,
    resolved_device: str,
    preview: PreviewRequest | None,
    status: str,
    result: Any | None,
    error: BaseException | None,
) -> dict[str, Any]:
    outputs = {"midi": paths.midi.name} if paths.midi.is_file() else {}
    if paths.preview is not None and paths.preview.is_file():
        outputs["preview"] = paths.preview.name
    return {
        "schema_version": STEM_MIDI_SCHEMA_VERSION,
        "status": status,
        "run_signature": signature,
        "source_path": str(candidate.source_path),
        "stem_audio_path": str(candidate.input_path),
        "stem_name": candidate.stem_name,
        "stem_family": stem_family(candidate.stem_name),
        "model_variant": options.model.value,
        "requested_device": options.device,
        "resolved_device": resolved_device,
        "instruments": list(options.instruments),
        "detected_instruments": list(getattr(result, "detected_instruments", ()) if result else ()),
        "options": options.as_dict(),
        "preview": preview.as_dict() if preview is not None else None,
        "outputs": outputs,
        "warnings": list(getattr(result, "warnings", ()) if result else ()),
        "error": ({"type": type(error).__name__, "message": str(error)} if error is not None else None),
    }


def transcribe_stem_candidates(
    candidates: Sequence[StemMidiCandidate],
    base_options: TranscriptionOptions,
    *,
    other_instruments: Iterable[str] = (),
    preview: PreviewRequest | None = None,
    overwrite: bool = False,
    model_loader: Callable[[TranscriptionOptions], Any] | None = None,
    transcriber: Callable[..., Any] | None = None,
    preview_preflight: Callable[[PreviewRequest], Any] | None = None,
    device_resolver: Callable[[str], str] | None = None,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[StemMidiCandidate, str], None] | None = None,
) -> StemMidiSummary:
    candidates = tuple(candidates)
    summary = StemMidiSummary()
    if not candidates:
        return summary

    other_instruments = _canonical_instruments(other_instruments)
    device_resolver = device_resolver or _default_device_resolver
    model_loader = model_loader or _default_model_loader
    transcriber = transcriber or _default_transcriber
    resolved_device = device_resolver(base_options.device)

    pending: list[tuple[StemMidiCandidate, StemMidiOutputPaths, TranscriptionOptions, str]] = []
    for candidate in candidates:
        instruments = instruments_for_stem(
            candidate.stem_name,
            other_instruments=other_instruments,
        )
        options = replace(base_options, instruments=instruments)
        paths = build_stem_midi_output_paths(candidate, preview)
        signature = _run_signature(
            candidate,
            options,
            resolved_device=resolved_device,
            preview=preview,
        )
        if not overwrite and _is_complete(paths, signature):
            summary.skipped += 1
            summary.items.append(StemMidiItemResult(candidate, "skipped", paths.midi, paths.metadata))
            if log_callback is not None:
                log_callback(f"Skipping completed stem MIDI: {paths.midi}")
            if progress_callback is not None:
                progress_callback(candidate, "skipped")
            continue
        pending.append((candidate, paths, options, signature))

    if not pending:
        return summary

    preview_runtime = None
    try:
        if preview is not None:
            if preview_preflight is None:
                from .auralization import preflight_preview

                preview_preflight = preflight_preview
            if log_callback is not None:
                log_callback(
                    f"Checking {preview.content.value} {preview.format.value} stem preview runtime"
                )
            preview_runtime = preview_preflight(preview)
        loaded = model_loader(base_options)
    except Exception as exc:
        summary.setup_error = f"{type(exc).__name__}: {exc}"
        for candidate, paths, options, signature in pending:
            paths.midi.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                paths.metadata,
                _metadata_payload(
                    candidate,
                    paths,
                    options,
                    signature=signature,
                    resolved_device=resolved_device,
                    preview=preview,
                    status="failed",
                    result=None,
                    error=exc,
                ),
            )
            summary.failed += 1
            summary.items.append(
                StemMidiItemResult(
                    candidate,
                    "failed",
                    paths.midi,
                    paths.metadata,
                    str(exc),
                )
            )
            if progress_callback is not None:
                progress_callback(candidate, "failed")
        if log_callback is not None:
            log_callback(f"Failed MuScriptor stem preview/model setup: {type(exc).__name__}: {exc}")
        return summary

    for candidate, paths, options, signature in pending:
        paths.midi.parent.mkdir(parents=True, exist_ok=True)
        _cleanup_stem_outputs(paths)
        result = None
        error: BaseException | None = None
        status = "ok"
        if log_callback is not None:
            instruments = ", ".join(options.instruments) or "auto"
            log_callback(f"Transcribing stem MIDI: {candidate.input_path} | instruments={instruments}")
        try:
            from .outputs import OutputTargets

            transcribe_kwargs = {}
            if preview_runtime is not None:
                transcribe_kwargs = {
                    "preview_runtime": preview_runtime,
                    "preview_target": paths.preview,
                }
            result = transcriber(
                loaded,
                candidate.input_path,
                options,
                OutputTargets(midi=paths.midi),
                **transcribe_kwargs,
            )
            if not paths.midi.is_file():
                raise RuntimeError(f"MuScriptor did not create the requested MIDI: {paths.midi}")
            if paths.preview is not None and not paths.preview.is_file():
                raise RuntimeError(f"MuScriptor did not create the requested preview: {paths.preview}")
            summary.processed += 1
        except Exception as exc:
            error = exc
            partial_result = getattr(exc, "result", None)
            if partial_result is not None and paths.midi.is_file():
                status = "partial"
                result = partial_result
                if paths.preview is not None:
                    paths.preview.unlink(missing_ok=True)
                summary.processed += 1
                summary.partial += 1
            else:
                status = "failed"
                _cleanup_stem_outputs(paths)
                summary.failed += 1
            if log_callback is not None:
                log_callback(f"Failed stem MIDI for {candidate.input_path}: {type(exc).__name__}: {exc}")
        atomic_write_json(
            paths.metadata,
            _metadata_payload(
                candidate,
                paths,
                options,
                signature=signature,
                resolved_device=resolved_device,
                preview=preview,
                status=status,
                result=result,
                error=error,
            ),
        )
        summary.items.append(
            StemMidiItemResult(
                candidate,
                status,
                paths.midi,
                paths.metadata,
                str(error) if error is not None else None,
            )
        )
        if progress_callback is not None:
            progress_callback(candidate, status)

    return summary
