from __future__ import annotations

import importlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .options import PreviewContent, PreviewFormat, PreviewRequest
from .outputs import atomic_output_path

SAMPLE_RATE = 44100
RENDERER_ID = "muscriptor-0.2.1:SF2_URL"


class PreviewUnavailable(RuntimeError):
    """Raised when the official preview renderer cannot run."""


@dataclass(frozen=True)
class PreviewRuntime:
    request: PreviewRequest
    soundfont_path: Path
    renderer_id: str = RENDERER_ID


def _probe_codec(soundfile_module: Any, request: PreviewRequest) -> None:
    import numpy as np

    channels = 1 if request.content is PreviewContent.MIDI else 2
    shape = (32,) if channels == 1 else (32, channels)
    silence = np.zeros(shape, dtype=np.float32)
    try:
        with tempfile.TemporaryDirectory(prefix="muscriptor-codec-") as temp_dir:
            probe_path = Path(temp_dir) / f"probe.{request.format.value}"
            soundfile_module.write(str(probe_path), silence, SAMPLE_RATE)
            decoded, sample_rate = soundfile_module.read(
                str(probe_path),
                dtype="float32",
                always_2d=True,
            )
            if int(sample_rate) != SAMPLE_RATE or decoded.shape[1] != channels:
                raise RuntimeError("channel or sample-rate mismatch")
    except Exception as exc:
        raise PreviewUnavailable(
            f"Current soundfile/libsndfile runtime cannot write {request.format.value} preview audio"
        ) from exc


def _resolve_official_default_sf2() -> Path:
    module = importlib.import_module("muscriptor.utils.auralization")
    resolver = getattr(module, "_resolve_soundfont")
    return Path(resolver(None))


def preflight_preview(
    request: PreviewRequest,
    *,
    soundfile_module: Any | None = None,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., Any] = subprocess.run,
    resolve_default_sf2: Callable[[], Path] | None = None,
) -> PreviewRuntime:
    soundfile_module = soundfile_module or importlib.import_module("soundfile")
    _probe_codec(soundfile_module, request)

    executable = which("fluidsynth")
    if not executable:
        raise PreviewUnavailable("FluidSynth was not found on PATH")
    try:
        probe = run(
            [executable, "--version"],
            capture_output=True,
            check=False,
            timeout=10,
        )
    except Exception as exc:
        raise PreviewUnavailable("FluidSynth could not start") from exc
    if int(getattr(probe, "returncode", 1)) != 0:
        raise PreviewUnavailable("FluidSynth could not start")

    resolver = resolve_default_sf2 or _resolve_official_default_sf2
    try:
        soundfont_path = Path(resolver()).expanduser().resolve()
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        raise PreviewUnavailable(
            f"MuScriptor's official default SoundFont is unavailable ({detail})"
        ) from exc
    if not soundfont_path.is_file():
        raise PreviewUnavailable(f"MuScriptor's official default SoundFont is missing: {soundfont_path}")
    return PreviewRuntime(request=request, soundfont_path=soundfont_path)


def _upstream_renderers() -> tuple[Callable[..., Any], Callable[..., Any]]:
    module = importlib.import_module("muscriptor.utils.auralization")
    return getattr(module, "synthesize"), getattr(module, "auralize")


def render_preview(
    runtime: PreviewRuntime,
    *,
    midi_bytes: bytes,
    original_audio_path: Path,
    output_path: Path,
    synthesize_func: Callable[..., Any] | None = None,
    auralize_func: Callable[..., Any] | None = None,
) -> None:
    output_path = Path(output_path)
    expected_suffix = f".{runtime.request.format.value}"
    if output_path.suffix.lower() != expected_suffix:
        raise ValueError(
            f"Preview output extension must be {expected_suffix}, got {output_path.suffix or '<none>'}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if synthesize_func is None or auralize_func is None:
        upstream_synthesize, upstream_auralize = _upstream_renderers()
        synthesize_func = synthesize_func or upstream_synthesize
        auralize_func = auralize_func or upstream_auralize

    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".mid",
        prefix="muscriptor-",
        dir=output_path.parent,
        delete=False,
    ) as midi_file:
        midi_file.write(midi_bytes)
        midi_path = Path(midi_file.name)

    try:
        with atomic_output_path(output_path) as temporary_output:
            if runtime.request.content is PreviewContent.MIDI:
                synthesize_func(
                    midi_path,
                    temporary_output,
                    soundfont_path=runtime.soundfont_path,
                )
            else:
                auralize_func(
                    midi_path,
                    Path(original_audio_path),
                    temporary_output,
                    soundfont_path=runtime.soundfont_path,
                )
            if not temporary_output.is_file():
                raise RuntimeError("Preview renderer did not create an output file")
    finally:
        midi_path.unlink(missing_ok=True)
