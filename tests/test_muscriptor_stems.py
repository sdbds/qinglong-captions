from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from module.muscriptor_tool.options import (
    PreviewContent,
    PreviewFormat,
    PreviewRequest,
    TranscriptionOptions,
)
from module.muscriptor_tool.stems import (
    STEM_INSTRUMENT_FAMILIES,
    StemMidiCandidate,
    build_stem_midi_output_paths,
    instruments_for_stem,
    stem_family,
    transcribe_stem_candidates,
)


def test_stem_instrument_hints_constrain_families_without_forcing_subtypes():
    assert instruments_for_stem("vocals") == ("voice",)
    assert instruments_for_stem("drums") == ("drums",)
    assert instruments_for_stem("bass") == ("acoustic_bass", "electric_bass")
    assert instruments_for_stem("guitar") == (
        "acoustic_guitar",
        "clean_electric_guitar",
        "distorted_electric_guitar",
    )
    assert instruments_for_stem("piano") == ("acoustic_piano", "electric_piano")
    assert instruments_for_stem("other") == ()
    assert instruments_for_stem(
        "other",
        other_instruments=("violin", "flutes"),
    ) == ("violin", "flutes")
    assert set(STEM_INSTRUMENT_FAMILIES) == {"vocals", "drums", "bass", "guitar", "piano"}


@pytest.mark.parametrize(
    ("label", "expected"),
    (
        ("Lead Vocals", "vocals"),
        ("DRUM KIT", "drums"),
        ("electric-bass", "bass"),
        ("Guitars", "guitar"),
        ("Keys", "piano"),
        ("other", "other"),
    ),
)
def test_stem_family_accepts_separator_label_variants(label: str, expected: str):
    assert stem_family(label) == expected


def _candidates(tmp_path: Path) -> list[StemMidiCandidate]:
    source = tmp_path / "song.wav"
    source.write_bytes(b"source")
    song_output = tmp_path / "song"
    song_output.mkdir()
    candidates = []
    for stem in ("bass", "drums", "other", "vocals", "guitar", "piano"):
        input_path = song_output / f"song_({stem})_primary.wav"
        input_path.write_bytes(stem.encode("ascii"))
        candidates.append(StemMidiCandidate(source, song_output, stem, input_path))
    return candidates


def test_transcribe_stem_candidates_loads_once_and_uses_per_stem_instruments(tmp_path: Path):
    candidates = _candidates(tmp_path)
    calls: dict[str, object] = {"loads": 0, "instruments": []}

    def load_model(_options):
        calls["loads"] = int(calls["loads"]) + 1
        return object()

    def transcribe(_loaded, _source, options, targets):
        calls["instruments"].append(options.instruments)
        targets.midi.write_bytes(b"MThd")
        detected = options.instruments[:1] or ("violin",)
        return SimpleNamespace(warnings=(), detected_instruments=detected)

    summary = transcribe_stem_candidates(
        candidates,
        TranscriptionOptions(device="cpu"),
        other_instruments=("violin", "flutes"),
        model_loader=load_model,
        transcriber=transcribe,
        device_resolver=lambda _requested: "cpu",
    )

    assert summary.processed == 6
    assert summary.skipped == 0
    assert summary.partial == 0
    assert summary.failed == 0
    assert calls["loads"] == 1
    assert calls["instruments"] == [
        ("acoustic_bass", "electric_bass"),
        ("drums",),
        ("violin", "flutes"),
        ("voice",),
        ("acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar"),
        ("acoustic_piano", "electric_piano"),
    ]

    other_paths = build_stem_midi_output_paths(candidates[2])
    metadata = json.loads(other_paths.metadata.read_text(encoding="utf-8"))
    assert other_paths.midi.read_bytes() == b"MThd"
    assert metadata["stem_family"] == "other"
    assert metadata["instruments"] == ["violin", "flutes"]
    assert metadata["detected_instruments"] == ["violin"]


def test_stem_preview_preflights_once_and_reuses_the_same_transcription(tmp_path: Path):
    candidates = _candidates(tmp_path)[:2]
    preview = PreviewRequest(PreviewContent.COMPARISON, PreviewFormat.MP3)
    runtime = object()
    calls = {"preflight": 0, "transcribe": 0}

    def preflight(request):
        assert request == preview
        calls["preflight"] += 1
        return runtime

    def transcribe(_loaded, source, _options, targets, **kwargs):
        calls["transcribe"] += 1
        assert kwargs["preview_runtime"] is runtime
        assert kwargs["preview_target"].suffix == ".mp3"
        targets.midi.write_bytes(b"MThd")
        kwargs["preview_target"].write_bytes(b"preview")
        return SimpleNamespace(warnings=(), detected_instruments=("drums",))

    summary = transcribe_stem_candidates(
        candidates,
        TranscriptionOptions(device="cpu"),
        preview=preview,
        model_loader=lambda _options: object(),
        transcriber=transcribe,
        preview_preflight=preflight,
        device_resolver=lambda _requested: "cpu",
    )

    assert summary.processed == 2
    assert summary.partial == 0
    assert summary.failed == 0
    assert calls == {"preflight": 1, "transcribe": 2}
    for candidate in candidates:
        paths = build_stem_midi_output_paths(candidate, preview)
        metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
        assert paths.midi.is_file()
        assert paths.preview.read_bytes() == b"preview"
        assert metadata["preview"] == {"content": "comparison", "format": "mp3"}
        assert metadata["outputs"]["preview"] == paths.preview.name


def test_preview_failure_keeps_usable_midi_and_marks_partial(tmp_path: Path):
    candidate = _candidates(tmp_path)[0]
    preview = PreviewRequest(PreviewContent.MIDI, PreviewFormat.WAV)
    partial_result = SimpleNamespace(
        warnings=("preview failed",),
        detected_instruments=("electric_bass",),
    )

    class PreviewFailure(RuntimeError):
        result = partial_result

    def transcribe(_loaded, _source, _options, targets, **_kwargs):
        targets.midi.write_bytes(b"MThd")
        raise PreviewFailure("FluidSynth failed")

    summary = transcribe_stem_candidates(
        [candidate],
        TranscriptionOptions(device="cpu"),
        preview=preview,
        model_loader=lambda _options: object(),
        transcriber=transcribe,
        preview_preflight=lambda _request: object(),
        device_resolver=lambda _requested: "cpu",
    )

    paths = build_stem_midi_output_paths(candidate, preview)
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    assert summary.processed == 1
    assert summary.partial == 1
    assert summary.failed == 0
    assert paths.midi.read_bytes() == b"MThd"
    assert not paths.preview.exists()
    assert metadata["status"] == "partial"
    assert metadata["outputs"] == {"midi": paths.midi.name}


def test_preview_preflight_failure_preserves_existing_midi(tmp_path: Path):
    candidate = _candidates(tmp_path)[0]
    preview = PreviewRequest(PreviewContent.MIDI, PreviewFormat.MP3)
    paths = build_stem_midi_output_paths(candidate, preview)
    paths.midi.parent.mkdir(parents=True, exist_ok=True)
    paths.midi.write_bytes(b"existing-midi")

    summary = transcribe_stem_candidates(
        [candidate],
        TranscriptionOptions(device="cpu"),
        preview=preview,
        model_loader=lambda _options: (_ for _ in ()).throw(
            AssertionError("model should not load after preview preflight fails")
        ),
        preview_preflight=lambda _request: (_ for _ in ()).throw(
            RuntimeError("FluidSynth unavailable")
        ),
        device_resolver=lambda _requested: "cpu",
    )

    metadata = json.loads(summary.items[0].metadata_path.read_text(encoding="utf-8"))
    assert summary.processed == 0
    assert summary.partial == 0
    assert summary.failed == 1
    assert paths.midi.read_bytes() == b"existing-midi"
    assert metadata["status"] == "failed"
    assert metadata["outputs"] == {}


def test_completed_stem_midi_skips_without_loading_model(tmp_path: Path):
    candidates = _candidates(tmp_path)[:1]
    options = TranscriptionOptions(device="cpu")

    def transcribe(_loaded, _source, _options, targets):
        targets.midi.write_bytes(b"MThd")
        return SimpleNamespace(warnings=(), detected_instruments=("electric_bass",))

    first = transcribe_stem_candidates(
        candidates,
        options,
        model_loader=lambda _options: object(),
        transcriber=transcribe,
        device_resolver=lambda _requested: "cpu",
    )
    second = transcribe_stem_candidates(
        candidates,
        options,
        model_loader=lambda _options: (_ for _ in ()).throw(AssertionError("model loaded")),
        transcriber=transcribe,
        device_resolver=lambda _requested: "cpu",
    )

    assert first.processed == 1
    assert second.processed == 0
    assert second.skipped == 1


def test_same_size_same_mtime_stem_change_reprocesses_then_skips(tmp_path: Path):
    candidate = _candidates(tmp_path)[0]
    options = TranscriptionOptions(device="cpu")
    original_stat = candidate.input_path.stat()
    calls: list[bytes] = []

    def transcribe(_loaded, source_path, _options, targets):
        source_bytes = Path(source_path).read_bytes()
        calls.append(source_bytes)
        targets.midi.write_bytes(source_bytes)
        return SimpleNamespace(warnings=(), detected_instruments=("electric_bass",))

    kwargs = {
        "model_loader": lambda _options: object(),
        "transcriber": transcribe,
        "device_resolver": lambda _requested: "cpu",
    }
    first = transcribe_stem_candidates([candidate], options, **kwargs)
    candidate.input_path.write_bytes(b"gass")
    os.utime(
        candidate.input_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    second = transcribe_stem_candidates([candidate], options, **kwargs)
    third = transcribe_stem_candidates(
        [candidate],
        options,
        model_loader=lambda _options: (_ for _ in ()).throw(AssertionError("model loaded")),
        transcriber=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("transcribed")),
        device_resolver=lambda _requested: "cpu",
    )

    paths = build_stem_midi_output_paths(candidate)
    assert first.processed == 1
    assert second.processed == 1
    assert third.skipped == 1
    assert calls == [b"bass", b"gass"]
    assert paths.midi.read_bytes() == b"gass"


def test_model_load_failure_marks_every_pending_stem_failed(tmp_path: Path):
    candidates = _candidates(tmp_path)[:2]

    summary = transcribe_stem_candidates(
        candidates,
        TranscriptionOptions(device="cpu"),
        model_loader=lambda _options: (_ for _ in ()).throw(RuntimeError("weights unavailable")),
        device_resolver=lambda _requested: "cpu",
    )

    assert summary.processed == 0
    assert summary.failed == 2
    assert summary.setup_error == "RuntimeError: weights unavailable"
    assert all(item.status == "failed" for item in summary.items)
    for item in summary.items:
        metadata = json.loads(item.metadata_path.read_text(encoding="utf-8"))
        assert metadata["status"] == "failed"
        assert metadata["error"]["message"] == "weights unavailable"


def test_other_stem_rejects_unknown_manual_instrument(tmp_path: Path):
    candidate = _candidates(tmp_path)[2]

    with pytest.raises(ValueError, match="Unknown MuScriptor instrument"):
        transcribe_stem_candidates(
            [candidate],
            TranscriptionOptions(device="cpu"),
            other_instruments=("kazoo",),
            model_loader=lambda _options: object(),
            device_resolver=lambda _requested: "cpu",
        )
