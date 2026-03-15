# -*- coding: utf-8 -*-

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


def test_catalog_normalizes_mistral_aliases():
    from providers.catalog import canonicalize_provider_name, canonicalize_route_value, route_provider_name

    assert canonicalize_provider_name("pixtral") == "mistral_ocr"
    assert canonicalize_provider_name("pixtral_ocr") == "mistral_ocr"
    assert canonicalize_route_value("ocr_model", "pixtral_ocr") == "mistral_ocr"
    assert route_provider_name("ocr_model", "pixtral") == "mistral_ocr"


def test_catalog_exposes_canonical_route_choices_only_by_default():
    from providers.catalog import route_choices

    ocr_choices = route_choices("ocr_model")
    vlm_choices = route_choices("vlm_image_model")
    assert "mistral_ocr" in ocr_choices
    assert "lighton_ocr" in ocr_choices
    assert "pixtral_ocr" not in ocr_choices
    assert "pixtral" not in ocr_choices
    assert "lfm_vl_local" in vlm_choices


def test_catalog_keeps_legacy_alias_attrs_in_sync():
    from providers.catalog import normalize_runtime_args

    args = SimpleNamespace(
        mistral_api_key="new-key",
        pixtral_api_key="",
        mistral_model_path="mistral-large-latest",
        pixtral_model_path="",
        ocr_model="pixtral_ocr",
        vlm_image_model="qwen_vl_local",
    )

    normalize_runtime_args(args)

    assert args.mistral_api_key == "new-key"
    assert args.pixtral_api_key == "new-key"
    assert args.mistral_model_path == "mistral-large-latest"
    assert args.pixtral_model_path == "mistral-large-latest"
    assert args.ocr_model == "mistral_ocr"


def test_catalog_config_section_candidates_cover_local_provider_compat():
    from providers.catalog import provider_config_sections

    assert provider_config_sections("qwen_vl_local") == ("qwen_vl_local", "qwen")
    assert provider_config_sections("penguin_vl_local") == ("penguin_vl_local", "penguin")
    assert provider_config_sections("step_vl_local") == ("step_vl_local", "stepfun_local")
    assert provider_config_sections("reka_edge_local") == ("reka_edge_local", "reka_edge")
    assert provider_config_sections("lfm_vl_local") == ("lfm_vl_local", "lfm_vl")


def test_catalog_marks_only_remote_routes_as_needing_api_config():
    from providers.catalog import route_requires_remote_config

    assert route_requires_remote_config("ocr_model", "mistral_ocr") is True
    assert route_requires_remote_config("ocr_model", "pixtral") is True
    assert route_requires_remote_config("ocr_model", "deepseek_ocr") is False
    assert route_requires_remote_config("ocr_model", "lighton_ocr") is False
    assert route_requires_remote_config("vlm_image_model", "qwen_vl_local") is False
    assert route_requires_remote_config("vlm_image_model", "reka_edge_local") is False
    assert route_requires_remote_config("vlm_image_model", "lfm_vl_local") is False
