"""Provider metadata declarations.

This module is the source of truth for provider identity, discovery paths,
route exposure, compatibility aliases, and declared capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

from .capabilities import ProviderCapabilities


@dataclass(frozen=True)
class ProviderRouteDeclaration:
    route_name: str
    value: str = ""
    aliases: Tuple[str, ...] = ()
    requires_remote_config: bool = False
    order: int = 0


@dataclass(frozen=True)
class PromptFallbackDeclaration:
    mime_prefix: str
    field: str
    keys: Tuple[str, ...]


@dataclass(frozen=True)
class ProviderDeclaration:
    name: str
    module_path: str
    priority: int
    routes: Tuple[ProviderRouteDeclaration, ...] = ()
    aliases: Tuple[str, ...] = ()
    config_sections: Tuple[str, ...] = ()
    prompt_prefixes: Tuple[str, ...] = ()
    prompt_fallbacks: Tuple[PromptFallbackDeclaration, ...] = ()
    arg_alias_groups: Tuple[Tuple[str, ...], ...] = ()
    capabilities: ProviderCapabilities = field(default_factory=ProviderCapabilities)
    segment_time_default: int | None = 600
    segmentation_bypass_mime_prefixes: Tuple[str, ...] = ()
    video_max_seconds_config_sections: Tuple[str, ...] = ()
    video_max_seconds_default: float = 120.0


def route(
    route_name: str,
    *,
    value: str = "",
    aliases: Iterable[str] = (),
    requires_remote_config: bool = False,
    order: int = 0,
) -> ProviderRouteDeclaration:
    return ProviderRouteDeclaration(
        route_name=route_name,
        value=value,
        aliases=tuple(aliases),
        requires_remote_config=requires_remote_config,
        order=order,
    )


def capabilities(**overrides) -> ProviderCapabilities:
    return ProviderCapabilities(**overrides)


def prompt_fallback(mime_prefix: str, field: str, *keys: str) -> PromptFallbackDeclaration:
    return PromptFallbackDeclaration(mime_prefix=mime_prefix, field=field, keys=tuple(keys))


PROVIDER_DECLARATIONS: Tuple[ProviderDeclaration, ...] = (
    ProviderDeclaration(
        name="codex_subscription",
        module_path="module.providers.cloud_vlm.codex_subscription",
        priority=10,
        capabilities=capabilities(supports_images=True, supports_video=True, supports_cloud_concurrency=True),
    ),
    ProviderDeclaration(
        name="grok_build_subscription",
        module_path="module.providers.cloud_vlm.grok_build_subscription",
        priority=15,
        capabilities=capabilities(supports_images=True, supports_structured_output=True),
    ),
    ProviderDeclaration(
        name="openai_compatible",
        module_path="module.providers.cloud_vlm.openai_compatible",
        priority=20,
        capabilities=capabilities(supports_images=True, supports_video=True, supports_cloud_concurrency=True),
    ),
    ProviderDeclaration(
        name="stepfun",
        module_path="module.providers.cloud_vlm.stepfun",
        priority=30,
        prompt_fallbacks=(
            prompt_fallback("video", "system", "step_video_system_prompt"),
            prompt_fallback("video", "user", "step_video_prompt"),
        ),
        capabilities=capabilities(supports_images=True, supports_video=True, supports_cloud_concurrency=True),
    ),
    ProviderDeclaration(
        name="ark",
        module_path="module.providers.cloud_vlm.ark",
        priority=40,
        capabilities=capabilities(supports_images=True, supports_video=True, supports_cloud_concurrency=True),
    ),
    ProviderDeclaration(
        name="qwenvl",
        module_path="module.providers.cloud_vlm.qwenvl",
        priority=50,
        capabilities=capabilities(supports_images=True, supports_video=True, supports_cloud_concurrency=True),
    ),
    ProviderDeclaration(
        name="glm",
        module_path="module.providers.cloud_vlm.glm",
        priority=60,
        capabilities=capabilities(supports_images=True, supports_video=True, supports_cloud_concurrency=True),
    ),
    ProviderDeclaration(
        name="kimi_code",
        module_path="module.providers.cloud_vlm.kimi_code",
        priority=70,
        prompt_fallbacks=(
            prompt_fallback("image", "system", "kimi_image_system_prompt"),
            prompt_fallback("image", "user", "kimi_image_prompt"),
        ),
        capabilities=capabilities(supports_images=True, supports_video=True, supports_cloud_concurrency=True),
    ),
    ProviderDeclaration(
        name="kimi_vl",
        module_path="module.providers.cloud_vlm.kimi_vl",
        priority=80,
        prompt_fallbacks=(
            prompt_fallback("image", "system", "kimi_image_system_prompt"),
            prompt_fallback("image", "user", "kimi_image_prompt"),
        ),
        capabilities=capabilities(supports_images=True, supports_video=True, supports_cloud_concurrency=True),
    ),
    ProviderDeclaration(
        name="mimo",
        module_path="module.providers.cloud_vlm.mimo",
        priority=90,
        config_sections=("mimo",),
        prompt_prefixes=("kimi",),
        capabilities=capabilities(supports_images=True, supports_video=True, supports_cloud_concurrency=True),
    ),
    ProviderDeclaration(
        name="minimax_code",
        module_path="module.providers.cloud_vlm.minimax_code",
        priority=100,
        prompt_fallbacks=(prompt_fallback("image", "user", "minimax_api_image_prompt"),),
        capabilities=capabilities(supports_images=True, supports_video=True, supports_cloud_concurrency=True),
    ),
    ProviderDeclaration(
        name="minimax_api",
        module_path="module.providers.cloud_vlm.minimax_api",
        priority=110,
        capabilities=capabilities(supports_images=True, supports_video=True, supports_cloud_concurrency=True),
    ),
    ProviderDeclaration(
        name="infinity_parser2_ocr",
        module_path="module.providers.ocr.infinity_parser2",
        priority=120,
        routes=(route("ocr_model", order=10),),
        config_sections=("infinity_parser2_ocr",),
        prompt_prefixes=("infinity_parser2",),
        capabilities=capabilities(supports_images=True, supports_documents=True),
    ),
    ProviderDeclaration(
        name="ovis_ocr2",
        module_path="module.providers.ocr.ovis_ocr2",
        priority=122,
        routes=(route("ocr_model", order=15),),
        config_sections=("ovis_ocr2",),
        prompt_prefixes=("ovis_ocr2",),
        capabilities=capabilities(supports_images=True, supports_documents=True),
    ),
    ProviderDeclaration(
        name="unlimited_ocr",
        module_path="module.providers.ocr.unlimited",
        priority=125,
        routes=(route("ocr_model", order=85),),
        config_sections=("unlimited_ocr",),
        prompt_prefixes=("unlimited_ocr",),
        capabilities=capabilities(supports_images=True, supports_documents=True),
    ),
    ProviderDeclaration(
        name="deepseek_ocr",
        module_path="module.providers.ocr.deepseek",
        priority=130,
        routes=(route("ocr_model", order=80),),
        capabilities=capabilities(supports_images=True, supports_documents=True),
    ),
    ProviderDeclaration(
        name="logics_ocr",
        module_path="module.providers.ocr.logics",
        priority=140,
        routes=(route("ocr_model", order=120),),
        capabilities=capabilities(supports_images=True, supports_documents=True),
    ),
    ProviderDeclaration(
        name="dots_ocr",
        module_path="module.providers.ocr.dots",
        priority=150,
        routes=(route("ocr_model", order=30),),
        capabilities=capabilities(supports_images=True, supports_documents=True),
    ),
    ProviderDeclaration(
        name="qianfan_ocr",
        module_path="module.providers.ocr.qianfan",
        priority=160,
        routes=(route("ocr_model", order=70),),
        capabilities=capabilities(supports_images=True, supports_documents=True),
    ),
    ProviderDeclaration(
        name="lighton_ocr",
        module_path="module.providers.ocr.lighton",
        priority=170,
        routes=(route("ocr_model", order=40),),
        capabilities=capabilities(supports_images=True, supports_documents=True),
    ),
    ProviderDeclaration(
        name="hunyuan_ocr",
        module_path="module.providers.ocr.hunyuan",
        priority=180,
        routes=(route("ocr_model", order=140),),
        capabilities=capabilities(supports_images=True, supports_documents=True),
    ),
    ProviderDeclaration(
        name="glm_ocr",
        module_path="module.providers.ocr.glm",
        priority=190,
        routes=(route("ocr_model", order=90),),
        capabilities=capabilities(supports_images=True, supports_documents=True),
    ),
    ProviderDeclaration(
        name="chandra_ocr",
        module_path="module.providers.ocr.chandra",
        priority=200,
        routes=(route("ocr_model", order=20),),
        capabilities=capabilities(supports_images=True, supports_documents=True),
    ),
    ProviderDeclaration(
        name="olmocr",
        module_path="module.providers.ocr.olmocr",
        priority=210,
        routes=(route("ocr_model", order=50),),
        capabilities=capabilities(supports_images=True, supports_documents=True),
    ),
    ProviderDeclaration(
        name="paddle_ocr",
        module_path="module.providers.ocr.paddle",
        priority=220,
        routes=(route("ocr_model", order=60),),
        capabilities=capabilities(supports_images=True, supports_documents=True),
    ),
    ProviderDeclaration(
        name="nanonets_ocr",
        module_path="module.providers.ocr.nanonets",
        priority=230,
        routes=(route("ocr_model", order=110),),
        capabilities=capabilities(supports_images=True, supports_documents=True),
    ),
    ProviderDeclaration(
        name="firered_ocr",
        module_path="module.providers.ocr.firered",
        priority=240,
        routes=(route("ocr_model", order=100),),
        capabilities=capabilities(supports_images=True, supports_documents=True),
    ),
    ProviderDeclaration(
        name="moondream",
        module_path="module.providers.local_vlm.moondream",
        priority=250,
        routes=(route("ocr_model", order=150), route("vlm_image_model", order=10)),
        capabilities=capabilities(supports_images=True),
    ),
    ProviderDeclaration(
        name="qwen_vl_local",
        module_path="module.providers.local_vlm.qwen_vl_local",
        priority=260,
        routes=(route("vlm_image_model", order=20),),
        config_sections=("qwen_vl_local", "qwen"),
        capabilities=capabilities(supports_images=True),
    ),
    ProviderDeclaration(
        name="step_vl_local",
        module_path="module.providers.local_vlm.step_vl_local",
        priority=270,
        routes=(route("vlm_image_model", order=30),),
        config_sections=("step_vl_local", "stepfun_local"),
        prompt_fallbacks=(
            prompt_fallback("video", "system", "step_video_system_prompt"),
            prompt_fallback("video", "user", "step_video_prompt"),
        ),
        capabilities=capabilities(supports_images=True),
    ),
    ProviderDeclaration(
        name="penguin_vl_local",
        module_path="module.providers.local_vlm.penguin_vl_local",
        priority=280,
        routes=(route("vlm_image_model", order=40),),
        config_sections=("penguin_vl_local", "penguin"),
        capabilities=capabilities(supports_images=True),
    ),
    ProviderDeclaration(
        name="reka_edge_local",
        module_path="module.providers.local_vlm.reka_edge_local",
        priority=290,
        routes=(route("vlm_image_model", order=50),),
        config_sections=("reka_edge_local", "reka_edge"),
        prompt_prefixes=("reka_edge",),
        capabilities=capabilities(supports_images=True, supports_video=True),
    ),
    ProviderDeclaration(
        name="lfm_vl_local",
        module_path="module.providers.local_vlm.lfm_vl_local",
        priority=300,
        routes=(route("vlm_image_model", order=60),),
        config_sections=("lfm_vl_local", "lfm_vl"),
        capabilities=capabilities(supports_images=True),
    ),
    ProviderDeclaration(
        name="gemma4_local",
        module_path="module.providers.local_vlm.gemma4_local",
        priority=310,
        routes=(route("vlm_image_model", order=70), route("alm_model", order=60)),
        config_sections=("gemma4_local",),
        prompt_prefixes=("gemma4",),
        capabilities=capabilities(supports_images=True, supports_video=True, supports_audio=True),
        segmentation_bypass_mime_prefixes=("video", "audio"),
    ),
    ProviderDeclaration(
        name="marlin_2b_local",
        module_path="module.providers.local_vlm.marlin_2b_local",
        priority=320,
        routes=(route("vlm_image_model", order=80),),
        config_sections=("marlin_2b_local", "marlin"),
        prompt_prefixes=("marlin",),
        capabilities=capabilities(supports_images=True, supports_video=True),
        video_max_seconds_config_sections=("marlin_2b_local", "marlin"),
    ),
    ProviderDeclaration(
        name="music_flamingo_local",
        module_path="module.providers.local_alm.music_flamingo_local",
        priority=330,
        routes=(route("alm_model", order=10),),
        config_sections=("music_flamingo_local", "music_flamingo"),
        prompt_prefixes=("music_flamingo",),
        capabilities=capabilities(supports_audio=True),
        segment_time_default=1200,
    ),
    ProviderDeclaration(
        name="eureka_audio_local",
        module_path="module.providers.local_alm.eureka_audio_local",
        priority=340,
        routes=(route("alm_model", order=20),),
        config_sections=("eureka_audio_local", "eureka_audio"),
        prompt_prefixes=("eureka",),
        capabilities=capabilities(supports_audio=True),
    ),
    ProviderDeclaration(
        name="acestep_transcriber_local",
        module_path="module.providers.local_alm.acestep_transcriber_local",
        priority=350,
        routes=(route("alm_model", order=30),),
        config_sections=("acestep_transcriber_local",),
        prompt_prefixes=("acestep_transcriber",),
        capabilities=capabilities(supports_audio=True),
    ),
    ProviderDeclaration(
        name="cohere_transcribe_local",
        module_path="module.providers.local_alm.cohere_transcribe_local",
        priority=360,
        routes=(route("alm_model", order=40),),
        config_sections=("cohere_transcribe_local",),
        capabilities=capabilities(supports_audio=True),
        segment_time_default=None,
    ),
    ProviderDeclaration(
        name="mega_asr_local",
        module_path="module.providers.local_alm.mega_asr_local",
        priority=370,
        routes=(route("alm_model", order=50),),
        config_sections=("mega_asr_local", "mega_asr"),
        capabilities=capabilities(supports_audio=True),
    ),
    ProviderDeclaration(
        name="mistral_ocr",
        module_path="module.providers.vision_api.pixtral",
        priority=380,
        routes=(route("ocr_model", aliases=("pixtral_ocr", "pixtral"), requires_remote_config=True, order=130),),
        aliases=("pixtral", "pixtral_ocr"),
        config_sections=("mistral_ocr", "pixtral"),
        prompt_prefixes=("mistral_ocr", "pixtral"),
        arg_alias_groups=(("mistral_api_key", "pixtral_api_key"), ("mistral_model_path", "pixtral_model_path")),
        prompt_fallbacks=(
            prompt_fallback("image", "system", "mistral_ocr_image_system_prompt", "pixtral_image_system_prompt"),
            prompt_fallback("image", "user", "mistral_ocr_image_prompt", "pixtral_image_prompt"),
        ),
        capabilities=capabilities(supports_images=True, supports_documents=True, supports_cloud_concurrency=True),
    ),
    ProviderDeclaration(
        name="gemini",
        module_path="module.providers.vision_api.gemini",
        priority=390,
        capabilities=capabilities(supports_images=True, supports_documents=True, supports_cloud_concurrency=True),
    ),
)

_DECLARATIONS_BY_NAME: Dict[str, ProviderDeclaration] = {declaration.name: declaration for declaration in PROVIDER_DECLARATIONS}


def provider_declarations() -> Tuple[ProviderDeclaration, ...]:
    return tuple(sorted(_DECLARATIONS_BY_NAME.values(), key=lambda declaration: declaration.priority))


def provider_declaration(name: str) -> ProviderDeclaration | None:
    return _DECLARATIONS_BY_NAME.get(name)


def register_provider_declaration(declaration: ProviderDeclaration) -> None:
    _DECLARATIONS_BY_NAME[declaration.name] = declaration
