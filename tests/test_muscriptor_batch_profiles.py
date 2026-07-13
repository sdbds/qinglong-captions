from __future__ import annotations

from types import SimpleNamespace

import pytest

from gui.wizard import step6_tools
from module.muscriptor_tool.batch_profiles import (
    InsufficientVRAMError,
    load_batch_profile_catalog,
)
from module.muscriptor_tool.options import ModelVariant

BENCHMARK_VRAM_BYTES = 24_146_083_840


def _gpu_probe(total_vram_bytes: int = BENCHMARK_VRAM_BYTES):
    return SimpleNamespace(
        devices=(
            SimpleNamespace(
                index=0,
                name="NVIDIA GeForce RTX 4090",
                total_vram_bytes=total_vram_bytes,
            ),
        )
    )


def test_recorded_profiles_preserve_exact_bs1_bs2_measurements():
    catalog = load_batch_profile_catalog()

    assert catalog.reserve_bytes == 2 * 1024**3
    assert catalog.low_vram_max_bytes == 16 * 1024**3
    assert catalog.low_vram_reserve_bytes == 1024**3
    assert catalog.benchmark["muscriptor_version"] == "0.2.1"
    assert catalog.benchmark["attention"] == "optimized_sdpa"
    assert {
        variant.value: (
            profile.peak_bs1_bytes,
            profile.bs2_minus_bs1_bytes,
            profile.peak_bs2_bytes,
            profile.minimum_vram_bytes,
            profile.validated_batch_size,
            profile.validation_peak_reserved_bytes,
        )
        for variant, profile in catalog.profiles.items()
    } == {
        "small": (
            933_232_640,
            224_395_264,
            1_157_627_904,
            962_592_768,
            46,
            21_472_739_328,
        ),
        "medium": (
            2_401_239_040,
            543_162_368,
            2_944_401_408,
            2_413_821_952,
            36,
            21_541_945_344,
        ),
        "large": (
            9_950_986_240,
            1_530_920_960,
            11_481_907_200,
            9_963_569_152,
            8,
            21_338_521_600,
        ),
    }


@pytest.mark.parametrize(
    ("model", "expected"),
    (("small", 46), ("medium", 36), ("large", 8)),
)
def test_profile_recommendations_reproduce_the_benchmark_validation(model: str, expected: int):
    catalog = load_batch_profile_catalog()

    assert catalog.recommend(ModelVariant(model), BENCHMARK_VRAM_BYTES) == expected


def test_profile_recommendation_uses_each_users_total_vram():
    catalog = load_batch_profile_catalog()

    assert catalog.recommend("large", 16 * 1024**3) == 4
    assert catalog.recommend("medium", 12 * 1024**3) == 18
    assert catalog.recommend("small", 8 * 1024**3) == 14


def test_profile_uses_one_gib_reserve_through_sixteen_gib():
    catalog = load_batch_profile_catalog()

    assert catalog.reserve_for(8 * 1024**3) == 1024**3
    assert catalog.reserve_for(16 * 1024**3) == 1024**3
    assert catalog.reserve_for(16 * 1024**3 + 1) == 2 * 1024**3


def test_profile_rejects_gpu_below_model_minimum_plus_reserve():
    catalog = load_batch_profile_catalog()
    minimum_total = (
        catalog.profiles[ModelVariant.LARGE].minimum_vram_bytes
        + catalog.low_vram_reserve_bytes
    )

    with pytest.raises(InsufficientVRAMError) as captured:
        catalog.recommend("large", minimum_total - 1)

    assert captured.value.required_bytes == minimum_total


def test_gui_applies_profile_defaults_for_both_muscriptor_tools():
    step = step6_tools.ToolsStep()
    step.gpu_probe = _gpu_probe()

    step._apply_muscriptor_batch_defaults()

    assert step.config["music_transcription_batch_size"] == 8
    assert step.config["audio_separator_muscriptor_batch_size"] == 8


def test_gui_switches_between_detected_gpus_and_recomputes_batch():
    step = step6_tools.ToolsStep()
    step.gpu_probe = SimpleNamespace(
        devices=(
            SimpleNamespace(index=0, name="8 GiB GPU", total_vram_bytes=8 * 1024**3),
            SimpleNamespace(index=1, name="24 GiB GPU", total_vram_bytes=24 * 1024**3),
        )
    )

    options = step._music_transcription_device_options()
    assert options["cuda:0"] == "CUDA 0 - 8 GiB GPU (8.0 GB)"
    assert options["cuda:1"] == "CUDA 1 - 24 GiB GPU (24.0 GB)"

    step._on_music_transcription_device_change("cuda:1")
    assert step.config["music_transcription_batch_size"] == 8

    step._on_music_transcription_device_change("cuda:0")
    assert step.config["music_transcription_batch_size"] == 1


def test_music_transcription_page_opens_with_recommended_slider_value():
    step = step6_tools.ToolsStep()
    step.gpu_probe = _gpu_probe()

    step._render_music_transcription_tool()

    assert step.music_transcription_batch_size_slider.value == 8
    assert step.config["music_transcription_batch_size"] == 8
    assert step._music_transcription_batch_user_edited is False


def test_gui_recomputes_on_model_change_until_user_edits_batch():
    step = step6_tools.ToolsStep()
    step.gpu_probe = _gpu_probe()
    step._apply_muscriptor_batch_defaults()

    step._on_music_transcription_model_change("medium")
    assert step.config["music_transcription_batch_size"] == 36

    step._on_music_transcription_batch_change(12)
    step._on_music_transcription_model_change("small")
    assert step.config["music_transcription_batch_size"] == 12


def test_gui_uses_bs1_for_cpu_without_overriding_manual_value():
    step = step6_tools.ToolsStep()
    step.gpu_probe = _gpu_probe()
    step._on_audio_separator_muscriptor_device_change("cpu")

    assert step.config["audio_separator_muscriptor_batch_size"] == 1

    step._on_audio_separator_muscriptor_batch_change(3)
    step._on_audio_separator_muscriptor_device_change("cuda:0")
    assert step.config["audio_separator_muscriptor_batch_size"] == 3


def test_gui_auto_device_falls_back_to_cpu_below_model_minimum():
    step = step6_tools.ToolsStep()
    step.gpu_probe = _gpu_probe(8 * 1024**3)

    step._apply_muscriptor_batch_default("transcription")

    assert step.config["music_transcription_device"] == "auto"
    assert step.config["music_transcription_batch_size"] == 1
    assert step._resolved_muscriptor_device(
        "music_transcription_model",
        "music_transcription_device",
    ) == "cpu"

    step._on_music_transcription_model_change("small")
    assert step.config["music_transcription_device"] == "auto"
    assert step.config["music_transcription_batch_size"] == 14
    assert step._resolved_muscriptor_device(
        "music_transcription_model",
        "music_transcription_device",
    ) == "auto"


def test_gui_blocks_explicit_cuda_below_model_minimum():
    step = step6_tools.ToolsStep()
    step.gpu_probe = _gpu_probe(8 * 1024**3)
    step.config["music_transcription_device"] = "cuda:0"

    with pytest.raises(ValueError, match="10.28 GiB"):
        step._validate_muscriptor_vram(
            "music_transcription_model",
            "music_transcription_device",
        )
