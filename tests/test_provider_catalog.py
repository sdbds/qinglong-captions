# -*- coding: utf-8 -*-

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.compat

ROOT = Path(__file__).resolve().parent.parent


def test_catalog_normalizes_mistral_aliases():
    from module.providers.catalog import canonicalize_provider_name, canonicalize_route_value, route_provider_name

    assert canonicalize_provider_name("pixtral") == "mistral_ocr"
    assert canonicalize_provider_name("pixtral_ocr") == "mistral_ocr"
    assert canonicalize_route_value("ocr_model", "pixtral_ocr") == "mistral_ocr"
    assert route_provider_name("ocr_model", "pixtral") == "mistral_ocr"


def test_catalog_exposes_canonical_route_choices_only_by_default():
    from module.providers.catalog import route_choices

    ocr_choices = route_choices("ocr_model")
    vlm_choices = route_choices("vlm_image_model")
    alm_choices = route_choices("alm_model")
    assert "mistral_ocr" in ocr_choices
    assert "infinity_parser2_ocr" in ocr_choices
    assert "unlimited_ocr" in ocr_choices
    assert "logics_ocr" in ocr_choices
    assert "dots_ocr" in ocr_choices
    assert "qianfan_ocr" in ocr_choices
    assert "lighton_ocr" in ocr_choices
    assert "pixtral_ocr" not in ocr_choices
    assert "pixtral" not in ocr_choices
    assert "lfm_vl_local" in vlm_choices
    assert "gemma4_local" in vlm_choices
    assert "marlin_2b_local" in vlm_choices
    assert "music_flamingo_local" in alm_choices
    assert "eureka_audio_local" in alm_choices
    assert "acestep_transcriber_local" in alm_choices
    assert "cohere_transcribe_local" in alm_choices
    assert "mega_asr_local" in alm_choices
    assert "gemma4_local" in alm_choices


def test_catalog_keeps_legacy_alias_attrs_in_sync():
    from module.providers.catalog import normalize_runtime_args

    args = SimpleNamespace(
        mistral_api_key="new-key",
        pixtral_api_key="",
        mistral_model_path="mistral-large-latest",
        pixtral_model_path="",
        ocr_model="pixtral_ocr",
        vlm_image_model="qwen_vl_local",
        alm_model="music_flamingo_local",
    )

    normalize_runtime_args(args)

    assert args.mistral_api_key == "new-key"
    assert args.pixtral_api_key == "new-key"
    assert args.mistral_model_path == "mistral-large-latest"
    assert args.pixtral_model_path == "mistral-large-latest"
    assert args.ocr_model == "mistral_ocr"
    assert args.alm_model == "music_flamingo_local"


def test_catalog_keeps_marlin_2b_out_of_global_segment_time_defaults():
    from module.providers.catalog import normalize_runtime_args

    args = SimpleNamespace(vlm_image_model="marlin_2b_local", alm_model="", segment_time=None)

    normalize_runtime_args(args)

    assert args.segment_time == 600
    assert args.effective_segment_time == 600


def test_catalog_preserves_cohere_no_split_default_when_marlin_2b_is_also_selected():
    from module.providers.catalog import normalize_runtime_args

    args = SimpleNamespace(vlm_image_model="marlin_2b_local", alm_model="cohere_transcribe_local", segment_time=None)

    normalize_runtime_args(args)

    assert args.segment_time is None
    assert args.effective_segment_time is None


def test_catalog_config_section_candidates_cover_local_provider_compat():
    from module.providers.catalog import provider_config_sections

    assert provider_config_sections("qwen_vl_local") == ("qwen_vl_local", "qwen")
    assert provider_config_sections("penguin_vl_local") == ("penguin_vl_local", "penguin")
    assert provider_config_sections("step_vl_local") == ("step_vl_local", "stepfun_local")
    assert provider_config_sections("reka_edge_local") == ("reka_edge_local", "reka_edge")
    assert provider_config_sections("lfm_vl_local") == ("lfm_vl_local", "lfm_vl")
    assert provider_config_sections("gemma4_local") == ("gemma4_local",)
    assert provider_config_sections("marlin_2b_local") == ("marlin_2b_local", "marlin")
    assert provider_config_sections("eureka_audio_local") == ("eureka_audio_local", "eureka_audio")
    assert provider_config_sections("acestep_transcriber_local") == ("acestep_transcriber_local",)
    assert provider_config_sections("cohere_transcribe_local") == ("cohere_transcribe_local",)
    assert provider_config_sections("mega_asr_local") == ("mega_asr_local", "mega_asr")


def test_catalog_marks_only_remote_routes_as_needing_api_config():
    from module.providers.catalog import route_requires_remote_config

    assert route_requires_remote_config("ocr_model", "mistral_ocr") is True
    assert route_requires_remote_config("ocr_model", "pixtral") is True
    assert route_requires_remote_config("ocr_model", "infinity_parser2_ocr") is False
    assert route_requires_remote_config("ocr_model", "unlimited_ocr") is False
    assert route_requires_remote_config("ocr_model", "deepseek_ocr") is False
    assert route_requires_remote_config("ocr_model", "logics_ocr") is False
    assert route_requires_remote_config("ocr_model", "dots_ocr") is False
    assert route_requires_remote_config("ocr_model", "lighton_ocr") is False
    assert route_requires_remote_config("vlm_image_model", "qwen_vl_local") is False
    assert route_requires_remote_config("vlm_image_model", "reka_edge_local") is False
    assert route_requires_remote_config("vlm_image_model", "lfm_vl_local") is False
    assert route_requires_remote_config("vlm_image_model", "gemma4_local") is False
    assert route_requires_remote_config("vlm_image_model", "marlin_2b_local") is False
    assert route_requires_remote_config("alm_model", "music_flamingo_local") is False
    assert route_requires_remote_config("alm_model", "eureka_audio_local") is False
    assert route_requires_remote_config("alm_model", "acestep_transcriber_local") is False
    assert route_requires_remote_config("alm_model", "cohere_transcribe_local") is False
    assert route_requires_remote_config("alm_model", "mega_asr_local") is False
    assert route_requires_remote_config("alm_model", "gemma4_local") is False


def test_provider_declarations_drive_registry_and_catalog_metadata():
    import module.providers.registry as registry_module
    from module.providers.catalog import (
        get_provider_declaration,
        provider_declared_capabilities,
        provider_module_path,
        provider_priority_order,
    )

    assert not hasattr(registry_module, "_PROVIDER_MODULES")

    declaration = get_provider_declaration("pixtral")
    assert declaration.name == "mistral_ocr"
    assert provider_module_path("mistral_ocr") == "module.providers.vision_api.pixtral"
    assert provider_declared_capabilities("mistral_ocr").supports_cloud_concurrency

    order = provider_priority_order()
    assert "kimi_code" in order
    assert order.index("kimi_code") < order.index("kimi_vl")
    assert not hasattr(registry_module.get_registry(), "_priority_order")


def test_declared_prompt_fallback_keys_replace_resolver_name_branches():
    from module.providers.base import Provider
    from module.providers.catalog import provider_prompt_fallback_keys

    class KimiPolicyProvider(Provider):
        name = "kimi_code"

        @classmethod
        def can_handle(cls, args, mime):
            return False

        def prepare_media(self, uri, mime, args):
            raise NotImplementedError

        def attempt(self, media, prompts):
            raise NotImplementedError

    assert provider_prompt_fallback_keys("stepfun", "video/mp4", "user") == ("step_video_prompt",)
    assert provider_prompt_fallback_keys("minimax_code", "image/png", "user") == ("minimax_api_image_prompt",)
    assert KimiPolicyProvider.prompt_fallback_keys("image/png", "system") == ("kimi_image_system_prompt",)


def test_declared_segmentation_policy_replaces_orchestrator_name_branches():
    from module.providers.base import Provider

    class GemmaPolicyProvider(Provider):
        name = "gemma4_local"

        @classmethod
        def can_handle(cls, args, mime):
            return False

        def prepare_media(self, uri, mime, args):
            raise NotImplementedError

        def attempt(self, media, prompts):
            raise NotImplementedError

    args = SimpleNamespace(segment_time=600)
    gemma_policy = GemmaPolicyProvider.segmentation_policy(args, "audio/wav", {})
    assert gemma_policy.bypass_segmentation is True

    marlin_args = SimpleNamespace(segment_time=600)
    marlin_policy = Provider.segmentation_policy.__func__(
        type("MarlinPolicyProvider", (Provider,), {"name": "marlin_2b_local"}),
        marlin_args,
        "video/mp4",
        {"marlin": {"video_max_seconds": 90}},
    )
    assert marlin_policy.segment_time == 89
    assert marlin_policy.direct_duration_limit_ms == 90_000


def test_register_provider_metadata_refreshes_catalog_indexes():
    import module.providers.catalog as catalog
    import module.providers.declarations as declarations
    import module.providers.registry as registry_module
    from module.providers.capabilities import ProviderCapabilities
    from module.providers.declarations import route

    original_declarations = dict(declarations._DECLARATIONS_BY_NAME)
    original_pending = list(registry_module._pending_registrations)

    try:
        @registry_module.register_provider(
            "dynamic_policy_provider",
            module_path="module.providers.dynamic_policy",
            priority=5,
            routes=(route("vlm_image_model", aliases=("dynamic_route_alias",), order=1),),
            aliases=("dynamic_provider_alias",),
            config_sections=("dynamic_config",),
            prompt_prefixes=("dynamic_prompt",),
            capabilities=ProviderCapabilities(supports_images=True),
        )
        class DynamicPolicyProvider:
            pass

        assert DynamicPolicyProvider.name == "dynamic_policy_provider"
        assert catalog.canonicalize_provider_name("dynamic_provider_alias") == "dynamic_policy_provider"
        assert catalog.canonicalize_route_value("vlm_image_model", "dynamic_route_alias") == "dynamic_policy_provider"
        assert catalog.route_provider_name("vlm_image_model", "dynamic_route_alias") == "dynamic_policy_provider"
        assert catalog.provider_config_sections("dynamic_provider_alias") == ("dynamic_policy_provider", "dynamic_config")
        assert catalog.provider_declared_capabilities("dynamic_provider_alias").supports_images
    finally:
        declarations._DECLARATIONS_BY_NAME.clear()
        declarations._DECLARATIONS_BY_NAME.update(original_declarations)
        registry_module._pending_registrations = original_pending
        catalog._refresh_catalog_indexes()
