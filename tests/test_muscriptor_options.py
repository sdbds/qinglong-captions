from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from module.muscriptor_tool.catalog import MUSCRIPTOR_INSTRUMENT_CATALOG_VERSION
from module.muscriptor_tool.events import EventStats, event_to_dict
from module.muscriptor_tool.options import (
    DEFAULT_MODEL,
    BatchOptions,
    DecodingMode,
    ModelVariant,
    OutputFormat,
    PreviewContent,
    PreviewFormat,
    PreviewRequest,
    TranscriptionOptions,
)


@pytest.mark.parametrize(
    "value",
    (
        "model.safetensors",
        "Org/repo",
        "hf://Org/repo/model.safetensors",
        "https://example.com/model.safetensors",
    ),
)
def test_model_variant_rejects_non_official_sources(value: str):
    with pytest.raises(ValueError):
        ModelVariant(value)


def test_model_variant_contains_only_official_sizes():
    assert tuple(item.value for item in ModelVariant) == ("small", "medium", "large")
    assert DEFAULT_MODEL is ModelVariant.LARGE


def test_instrument_catalog_version_matches_dependency_pin():
    project_text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert f'"muscriptor=={MUSCRIPTOR_INSTRUMENT_CATALOG_VERSION}"' in project_text


def test_preview_is_none_or_one_immutable_request():
    preview = PreviewRequest(content=PreviewContent.COMPARISON, format=PreviewFormat.WAV)
    options = BatchOptions(output_formats=(OutputFormat.MIDI,), preview=preview)

    assert options.preview == preview
    with pytest.raises(FrozenInstanceError):
        preview.format = PreviewFormat.MP3  # type: ignore[misc]


def test_preview_defaults_to_mp3():
    assert PreviewRequest(content=PreviewContent.MIDI).format is PreviewFormat.MP3


def test_batch_requires_symbolic_output():
    with pytest.raises(ValueError, match="symbolic output"):
        BatchOptions(output_formats=())


def test_batch_deduplicates_output_formats_without_reordering():
    options = BatchOptions(
        output_formats=(OutputFormat.JSON, OutputFormat.MIDI, OutputFormat.JSON),
    )

    assert options.output_formats == (OutputFormat.JSON, OutputFormat.MIDI)


def test_single_cli_sampling_and_beam_cannot_coexist():
    with pytest.raises(ValueError, match="sampling.*beam"):
        TranscriptionOptions.from_single_cli(sampling=True, beam_size=2)


def test_single_cli_beam_width_selects_beam_mode():
    options = TranscriptionOptions.from_single_cli(beam_size=3)

    assert options.decode_mode is DecodingMode.BEAM
    assert options.upstream_kwargs()["beam_size"] == 3
    assert options.upstream_kwargs()["use_sampling"] is False


def test_sampling_maps_all_upstream_arguments():
    options = TranscriptionOptions.from_single_cli(
        model="small",
        device="cuda:2",
        batch_size=5,
        sampling=True,
        temperature=0.7,
        cfg_coef=1.5,
        strict_eos=True,
        instruments=("piano", "drums"),
        print_notes=True,
    )

    assert options.model is ModelVariant.SMALL
    assert options.device == "cuda:2"
    assert options.print_notes is True
    assert options.upstream_kwargs() == {
        "use_sampling": True,
        "temperature": 0.7,
        "cfg_coef": 1.5,
        "instruments": ["piano", "drums"],
        "batch_size": 5,
        "no_eos_is_ok": False,
        "beam_size": 1,
    }


@pytest.mark.parametrize("device", ("gpu", "cuda:-1", "cuda:x", "mps", ""))
def test_invalid_device_is_rejected(device: str):
    with pytest.raises(ValueError, match="device"):
        TranscriptionOptions(device=device)


def test_non_sampling_temperature_is_not_silently_ignored():
    with pytest.raises(ValueError, match="temperature"):
        TranscriptionOptions(temperature=0.8)


def test_event_to_dict_uses_stable_start_and_end_schema():
    start = SimpleNamespace(pitch=60, start_time=1.25, index=7, instrument="piano")
    end = SimpleNamespace(end_time=2.5, start_event=start, start_event_index=7)

    assert event_to_dict(start) == {
        "type": "start",
        "pitch": 60,
        "start_time": 1.25,
        "index": 7,
        "instrument": "piano",
    }
    assert event_to_dict(end) == {
        "type": "end",
        "end_time": 2.5,
        "start_event_index": 7,
    }


def test_event_to_dict_rejects_unknown_shape():
    with pytest.raises(TypeError, match="Unsupported MuScriptor event"):
        event_to_dict(SimpleNamespace(value="unknown"))


def test_event_stats_counts_notes_events_and_progress():
    stats = EventStats()
    stats.observe_event({"type": "start"})
    stats.observe_event({"type": "end"})
    stats.observe_progress(completed=2, total=4)

    assert stats.note_count == 1
    assert stats.event_count == 2
    assert stats.chunk_count == 4
    assert stats.completed_chunks == 2
