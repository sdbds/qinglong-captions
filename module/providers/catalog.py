"""Central provider naming catalog.

This module is the single source of truth for:
- canonical provider names
- backward-compatible aliases
- config section lookup candidates
- route option names exposed by CLI / GUI
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple


@dataclass(frozen=True)
class ProviderSpec:
    canonical_name: str
    aliases: Tuple[str, ...] = ()
    config_sections: Tuple[str, ...] = ()
    prompt_prefixes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteSpec:
    value: str
    provider: str
    aliases: Tuple[str, ...] = ()
    requires_remote_config: bool = False


def _unique(values: Iterable[str]) -> Tuple[str, ...]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


PROVIDER_SPECS: Dict[str, ProviderSpec] = {
    "mistral_ocr": ProviderSpec(
        canonical_name="mistral_ocr",
        aliases=("pixtral", "pixtral_ocr"),
        config_sections=("mistral_ocr", "pixtral"),
        prompt_prefixes=("mistral_ocr", "pixtral"),
    ),
    "qwen_vl_local": ProviderSpec(
        canonical_name="qwen_vl_local",
        config_sections=("qwen_vl_local", "qwen"),
    ),
    "step_vl_local": ProviderSpec(
        canonical_name="step_vl_local",
        config_sections=("step_vl_local", "stepfun_local"),
    ),
    "penguin_vl_local": ProviderSpec(
        canonical_name="penguin_vl_local",
        config_sections=("penguin_vl_local", "penguin"),
    ),
    "reka_edge_local": ProviderSpec(
        canonical_name="reka_edge_local",
        config_sections=("reka_edge_local", "reka_edge"),
        prompt_prefixes=("reka_edge",),
    ),
    "lfm_vl_local": ProviderSpec(
        canonical_name="lfm_vl_local",
        config_sections=("lfm_vl_local", "lfm_vl"),
    ),
    "music_flamingo_local": ProviderSpec(
        canonical_name="music_flamingo_local",
        config_sections=("music_flamingo_local", "music_flamingo"),
        prompt_prefixes=("music_flamingo",),
    ),
}

ROUTE_SPECS: Dict[str, Tuple[RouteSpec, ...]] = {
    "ocr_model": (
        RouteSpec("chandra_ocr", "chandra_ocr"),
        RouteSpec("dots_ocr", "dots_ocr"),
        RouteSpec("lighton_ocr", "lighton_ocr"),
        RouteSpec("logics_ocr", "logics_ocr"),
        RouteSpec("olmocr", "olmocr"),
        RouteSpec("paddle_ocr", "paddle_ocr"),
        RouteSpec("qianfan_ocr", "qianfan_ocr"),
        RouteSpec("deepseek_ocr", "deepseek_ocr"),
        RouteSpec("glm_ocr", "glm_ocr"),
        RouteSpec("firered_ocr", "firered_ocr"),
        RouteSpec("nanonets_ocr", "nanonets_ocr"),
        RouteSpec("mistral_ocr", "mistral_ocr", aliases=("pixtral_ocr", "pixtral"), requires_remote_config=True),
        RouteSpec("hunyuan_ocr", "hunyuan_ocr"),
        RouteSpec("moondream", "moondream"),
    ),
    "vlm_image_model": (
        RouteSpec("moondream", "moondream"),
        RouteSpec("qwen_vl_local", "qwen_vl_local"),
        RouteSpec("step_vl_local", "step_vl_local"),
        RouteSpec("penguin_vl_local", "penguin_vl_local"),
        RouteSpec("reka_edge_local", "reka_edge_local"),
        RouteSpec("lfm_vl_local", "lfm_vl_local"),
    ),
    "alm_model": (
        RouteSpec("music_flamingo_local", "music_flamingo_local"),
    ),
}

_PROVIDER_ALIAS_MAP: Dict[str, str] = {}
for canonical_name, spec in PROVIDER_SPECS.items():
    for alias in (canonical_name, *spec.aliases):
        _PROVIDER_ALIAS_MAP[alias] = canonical_name

_ROUTE_ALIAS_MAP: Dict[str, Dict[str, str]] = {}
_ROUTE_PROVIDER_MAP: Dict[str, Dict[str, str]] = {}
_ROUTE_REMOTE_CONFIG_MAP: Dict[str, Dict[str, bool]] = {}
for route_name, specs in ROUTE_SPECS.items():
    alias_map: Dict[str, str] = {}
    provider_map: Dict[str, str] = {}
    remote_config_map: Dict[str, bool] = {}
    for spec in specs:
        provider_map[spec.value] = spec.provider
        remote_config_map[spec.value] = spec.requires_remote_config
        for alias in (spec.value, *spec.aliases):
            alias_map[alias] = spec.value
    _ROUTE_ALIAS_MAP[route_name] = alias_map
    _ROUTE_PROVIDER_MAP[route_name] = provider_map
    _ROUTE_REMOTE_CONFIG_MAP[route_name] = remote_config_map


def canonicalize_provider_name(name: str) -> str:
    if not name:
        return name
    return _PROVIDER_ALIAS_MAP.get(name, name)


def provider_aliases(name: str) -> Tuple[str, ...]:
    canonical_name = canonicalize_provider_name(name)
    spec = PROVIDER_SPECS.get(canonical_name)
    if not spec:
        return (canonical_name,)
    return _unique((canonical_name, *spec.aliases))


def provider_config_sections(name: str) -> Tuple[str, ...]:
    canonical_name = canonicalize_provider_name(name)
    spec = PROVIDER_SPECS.get(canonical_name)
    if not spec:
        return (canonical_name,)
    return _unique((canonical_name, *spec.config_sections))


def provider_prompt_prefixes(name: str) -> Tuple[str, ...]:
    canonical_name = canonicalize_provider_name(name)
    spec = PROVIDER_SPECS.get(canonical_name)
    if not spec:
        return (canonical_name,)
    fallback_prefixes = spec.prompt_prefixes or spec.aliases
    return _unique((canonical_name, *fallback_prefixes))


def route_choices(route_name: str, *, include_aliases: bool = False, include_empty: bool = True) -> Tuple[str, ...]:
    values = [""] if include_empty else []
    for spec in ROUTE_SPECS.get(route_name, ()):
        values.append(spec.value)
        if include_aliases:
            values.extend(spec.aliases)
    return _unique(values)


def canonicalize_route_value(route_name: str, value: str) -> str:
    if not value:
        return value
    return _ROUTE_ALIAS_MAP.get(route_name, {}).get(value, value)


def route_provider_name(route_name: str, value: str) -> str:
    canonical_value = canonicalize_route_value(route_name, value)
    if not canonical_value:
        return ""
    provider_name = _ROUTE_PROVIDER_MAP.get(route_name, {}).get(canonical_value, canonical_value)
    return canonicalize_provider_name(provider_name)


def route_requires_remote_config(route_name: str, value: str) -> bool:
    canonical_value = canonicalize_route_value(route_name, value)
    if not canonical_value:
        return False
    return _ROUTE_REMOTE_CONFIG_MAP.get(route_name, {}).get(canonical_value, False)


def route_matches_provider(route_name: str, value: str, provider_name: str) -> bool:
    return route_provider_name(route_name, value) == canonicalize_provider_name(provider_name)


def get_first_attr(obj: Any, *names: str, default: Any = "") -> Any:
    for name in names:
        value = getattr(obj, name, None)
        if value not in (None, ""):
            return value
    return default


def _resolve_effective_segment_time(args: Any) -> int:
    raw_segment_time = getattr(args, "segment_time", None)
    explicit = raw_segment_time not in (None, "")
    setattr(args, "segment_time_explicit", explicit)

    if explicit:
        effective = int(raw_segment_time)
    elif getattr(args, "alm_model", "") == "music_flamingo_local":
        effective = 1200
    else:
        effective = 600

    setattr(args, "effective_segment_time", effective)
    if hasattr(args, "segment_time"):
        args.segment_time = effective
    return effective


def normalize_runtime_args(args: Any) -> Any:
    """Normalize runtime args in-place while preserving old aliases."""
    mistral_api_key = get_first_attr(args, "mistral_api_key", "pixtral_api_key", default="")
    mistral_model_path = get_first_attr(args, "mistral_model_path", "pixtral_model_path", default="")

    setattr(args, "mistral_api_key", mistral_api_key)
    setattr(args, "pixtral_api_key", mistral_api_key)
    setattr(args, "mistral_model_path", mistral_model_path)
    setattr(args, "pixtral_model_path", mistral_model_path)

    if hasattr(args, "ocr_model"):
        args.ocr_model = canonicalize_route_value("ocr_model", getattr(args, "ocr_model", ""))
    if hasattr(args, "vlm_image_model"):
        args.vlm_image_model = canonicalize_route_value("vlm_image_model", getattr(args, "vlm_image_model", ""))
    if hasattr(args, "alm_model"):
        args.alm_model = canonicalize_route_value("alm_model", getattr(args, "alm_model", ""))
    if hasattr(args, "segment_time"):
        _resolve_effective_segment_time(args)

    return args
