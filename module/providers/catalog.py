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

from .capabilities import ProviderCapabilities
from .declarations import (
    ProviderDeclaration,
    ProviderRouteDeclaration,
    provider_declaration as _provider_declaration,
    provider_declarations,
    register_provider_declaration as _register_provider_declaration,
)
from .policies import SegmentationPolicy


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


def _build_provider_specs() -> Dict[str, ProviderSpec]:
    return {
        declaration.name: ProviderSpec(
            canonical_name=declaration.name,
            aliases=declaration.aliases,
            config_sections=declaration.config_sections,
            prompt_prefixes=declaration.prompt_prefixes,
        )
        for declaration in provider_declarations()
    }


def _route_spec(declaration: ProviderDeclaration, route_declaration: ProviderRouteDeclaration) -> RouteSpec:
    value = route_declaration.value or declaration.name
    return RouteSpec(
        value=value,
        provider=declaration.name,
        aliases=route_declaration.aliases,
        requires_remote_config=route_declaration.requires_remote_config,
    )


def _build_route_specs() -> Dict[str, Tuple[RouteSpec, ...]]:
    grouped: Dict[str, list[tuple[int, RouteSpec]]] = {}
    for declaration in provider_declarations():
        for route_declaration in declaration.routes:
            grouped.setdefault(route_declaration.route_name, []).append(
                (route_declaration.order, _route_spec(declaration, route_declaration))
            )
    return {
        route_name: tuple(spec for _order, spec in sorted(specs, key=lambda item: item[0]))
        for route_name, specs in grouped.items()
    }


PROVIDER_SPECS: Dict[str, ProviderSpec] = {}
ROUTE_SPECS: Dict[str, Tuple[RouteSpec, ...]] = {}
_PROVIDER_ALIAS_MAP: Dict[str, str] = {}
_ROUTE_ALIAS_MAP: Dict[str, Dict[str, str]] = {}
_ROUTE_PROVIDER_MAP: Dict[str, Dict[str, str]] = {}
_ROUTE_REMOTE_CONFIG_MAP: Dict[str, Dict[str, bool]] = {}


def _refresh_catalog_indexes() -> None:
    global PROVIDER_SPECS, ROUTE_SPECS, _PROVIDER_ALIAS_MAP, _ROUTE_ALIAS_MAP, _ROUTE_PROVIDER_MAP, _ROUTE_REMOTE_CONFIG_MAP

    PROVIDER_SPECS = _build_provider_specs()
    ROUTE_SPECS = _build_route_specs()

    provider_alias_map: Dict[str, str] = {}
    for canonical_name, spec in PROVIDER_SPECS.items():
        for alias in (canonical_name, *spec.aliases):
            provider_alias_map[alias] = canonical_name

    route_alias_map: Dict[str, Dict[str, str]] = {}
    route_provider_map: Dict[str, Dict[str, str]] = {}
    route_remote_config_map: Dict[str, Dict[str, bool]] = {}
    for route_name, specs in ROUTE_SPECS.items():
        alias_map: Dict[str, str] = {}
        provider_map: Dict[str, str] = {}
        remote_config_map: Dict[str, bool] = {}
        for spec in specs:
            provider_map[spec.value] = spec.provider
            remote_config_map[spec.value] = spec.requires_remote_config
            for alias in (spec.value, *spec.aliases):
                alias_map[alias] = spec.value
        route_alias_map[route_name] = alias_map
        route_provider_map[route_name] = provider_map
        route_remote_config_map[route_name] = remote_config_map

    _PROVIDER_ALIAS_MAP = provider_alias_map
    _ROUTE_ALIAS_MAP = route_alias_map
    _ROUTE_PROVIDER_MAP = route_provider_map
    _ROUTE_REMOTE_CONFIG_MAP = route_remote_config_map


def register_provider_declaration(declaration: ProviderDeclaration) -> None:
    _register_provider_declaration(declaration)
    _refresh_catalog_indexes()


_refresh_catalog_indexes()


def canonicalize_provider_name(name: str) -> str:
    if not name:
        return name
    return _PROVIDER_ALIAS_MAP.get(name, name)


def get_provider_declaration(name: str) -> ProviderDeclaration | None:
    return _provider_declaration(canonicalize_provider_name(name))


def provider_module_paths() -> Dict[str, str]:
    return {declaration.name: declaration.module_path for declaration in provider_declarations()}


def provider_module_path(name: str) -> str:
    declaration = get_provider_declaration(name)
    return declaration.module_path if declaration is not None else ""


def provider_priority_order() -> Tuple[str, ...]:
    return tuple(declaration.name for declaration in provider_declarations())


def provider_declared_capabilities(name: str) -> ProviderCapabilities:
    declaration = get_provider_declaration(name)
    return declaration.capabilities if declaration is not None else ProviderCapabilities()


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


def provider_prompt_fallback_keys(name: str, mime: str, field: str) -> Tuple[str, ...]:
    declaration = get_provider_declaration(name)
    if declaration is None:
        return tuple()
    return tuple(
        key
        for fallback in declaration.prompt_fallbacks
        if mime.startswith(fallback.mime_prefix) and fallback.field == field
        for key in fallback.keys
    )


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


def _sync_declared_arg_aliases(args: Any) -> None:
    for declaration in provider_declarations():
        for alias_group in declaration.arg_alias_groups:
            value = get_first_attr(args, *alias_group, default="")
            for name in alias_group:
                setattr(args, name, value)


def _coerce_float(value: Any, default: float) -> float:
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _section(config: Any, names: Tuple[str, ...]) -> dict:
    if not config or not hasattr(config, "get"):
        return {}
    for name in names:
        value = config.get(name)
        if (isinstance(value, dict) or hasattr(value, "get")) and value:
            return value
    return {}


def provider_segment_time_default(name: str) -> int | None:
    declaration = get_provider_declaration(name)
    return declaration.segment_time_default if declaration is not None else 600


def provider_segmentation_policy(name: str, args: Any, mime: str, config: Any) -> SegmentationPolicy:
    declaration = get_provider_declaration(name)
    segment_time = getattr(args, "segment_time", None)
    if declaration is None:
        return SegmentationPolicy(segment_time=segment_time)

    bypass = any(str(mime).startswith(prefix) for prefix in declaration.segmentation_bypass_mime_prefixes)
    direct_limit_ms = None

    if str(mime).startswith("video") and declaration.video_max_seconds_config_sections:
        section = _section(config, declaration.video_max_seconds_config_sections)
        max_seconds = _coerce_float(section.get("video_max_seconds", declaration.video_max_seconds_default), declaration.video_max_seconds_default)
        if max_seconds > 0:
            direct_limit_ms = int(max_seconds * 1000)
            safe_segment_time = max(1, int(max_seconds) - 1)
            if segment_time in (None, ""):
                segment_time = safe_segment_time
            else:
                try:
                    requested_segment_time = int(segment_time)
                except (TypeError, ValueError):
                    requested_segment_time = safe_segment_time
                segment_time = safe_segment_time if requested_segment_time <= 0 else min(requested_segment_time, safe_segment_time)

    return SegmentationPolicy(
        segment_time=segment_time,
        bypass_segmentation=bypass,
        direct_duration_limit_ms=direct_limit_ms,
    )


def _resolve_effective_segment_time(args: Any) -> int | None:
    raw_segment_time = getattr(args, "segment_time", None)
    explicit = raw_segment_time not in (None, "")
    setattr(args, "segment_time_explicit", explicit)

    if explicit:
        effective = int(raw_segment_time)
    else:
        provider_name = route_provider_name("alm_model", getattr(args, "alm_model", ""))
        effective = provider_segment_time_default(provider_name)

    setattr(args, "effective_segment_time", effective)
    if hasattr(args, "segment_time"):
        args.segment_time = effective
    return effective


def normalize_runtime_args(args: Any) -> Any:
    """Normalize runtime args in-place while preserving old aliases."""
    _sync_declared_arg_aliases(args)

    if hasattr(args, "ocr_model"):
        args.ocr_model = canonicalize_route_value("ocr_model", getattr(args, "ocr_model", ""))
    if hasattr(args, "vlm_image_model"):
        args.vlm_image_model = canonicalize_route_value("vlm_image_model", getattr(args, "vlm_image_model", ""))
    if hasattr(args, "alm_model"):
        args.alm_model = canonicalize_route_value("alm_model", getattr(args, "alm_model", ""))
    if hasattr(args, "segment_time"):
        _resolve_effective_segment_time(args)

    return args
