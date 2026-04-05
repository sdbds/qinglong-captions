# -*- coding: utf-8 -*-

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


def test_captioner_parser_leaves_segment_time_unset_by_default():
    from module.captioner import setup_parser

    args = setup_parser().parse_args(["datasets"])

    assert args.segment_time is None


def test_captioner_parser_leaves_alm_language_unset_by_default():
    from module.captioner import setup_parser

    args = setup_parser().parse_args(["datasets"])

    assert args.alm_language is None


def test_captioner_parser_leaves_gemma4_audio_contract_unset_by_default():
    from module.captioner import setup_parser

    args = setup_parser().parse_args(["datasets"])

    assert args.audio_task == ""
    assert args.gemma4_model_id == ""


def test_captioner_parser_rejects_legacy_config_flag():
    from module.captioner import setup_parser

    with pytest.raises(SystemExit):
        setup_parser().parse_args(["datasets", "--config", "config/config.toml"])


def test_music_flamingo_uses_provider_specific_default_segment_time():
    from providers.catalog import normalize_runtime_args

    args = SimpleNamespace(
        segment_time=None,
        alm_model="music_flamingo_local",
        ocr_model="",
        vlm_image_model="",
    )

    normalize_runtime_args(args)

    assert args.segment_time_explicit is False
    assert args.effective_segment_time == 1200
    assert args.segment_time == 1200


def test_non_alm_routes_keep_legacy_default_segment_time():
    from providers.catalog import normalize_runtime_args

    args = SimpleNamespace(
        segment_time=None,
        alm_model="",
        ocr_model="",
        vlm_image_model="qwen_vl_local",
    )

    normalize_runtime_args(args)

    assert args.segment_time_explicit is False
    assert args.effective_segment_time == 600
    assert args.segment_time == 600


def test_eureka_audio_keeps_generic_default_segment_time():
    from providers.catalog import normalize_runtime_args

    args = SimpleNamespace(
        segment_time=None,
        alm_model="eureka_audio_local",
        ocr_model="",
        vlm_image_model="",
    )

    normalize_runtime_args(args)

    assert args.segment_time_explicit is False
    assert args.effective_segment_time == 600
    assert args.segment_time == 600


def test_acestep_transcriber_keeps_generic_default_segment_time():
    from providers.catalog import normalize_runtime_args

    args = SimpleNamespace(
        segment_time=None,
        alm_model="acestep_transcriber_local",
        ocr_model="",
        vlm_image_model="",
    )

    normalize_runtime_args(args)

    assert args.segment_time_explicit is False
    assert args.effective_segment_time == 600
    assert args.segment_time == 600


def test_cohere_transcribe_keeps_generic_default_segment_time():
    from providers.catalog import normalize_runtime_args

    args = SimpleNamespace(
        segment_time=None,
        alm_model="cohere_transcribe_local",
        ocr_model="",
        vlm_image_model="",
    )

    normalize_runtime_args(args)

    assert args.segment_time_explicit is False
    assert args.effective_segment_time is None
    assert args.segment_time is None


def test_explicit_segment_time_overrides_music_flamingo_default():
    from providers.catalog import normalize_runtime_args

    args = SimpleNamespace(
        segment_time=90,
        alm_model="music_flamingo_local",
        ocr_model="",
        vlm_image_model="",
    )

    normalize_runtime_args(args)

    assert args.segment_time_explicit is True
    assert args.effective_segment_time == 90
    assert args.segment_time == 90
