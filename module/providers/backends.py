"""Runtime backend helpers for local providers.

This module separates provider semantics from execution transport:
- direct: in-process transformers / model-specific Python APIs
- openai: OpenAI-compatible HTTP server for local self-hosted models
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class RuntimeBackendConfig:
    """Resolved backend configuration for a provider/runtime."""

    mode: str = "direct"
    base_url: str = ""
    api_key: str = ""
    model_id: str = ""
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    json_mode: bool = False

    @property
    def is_openai(self) -> bool:
        return self.mode == "openai"


def make_runtime_backend_config(
    *,
    mode: str = "direct",
    base_url: str = "",
    api_key: str = "",
    model_id: str = "",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 2048,
    json_mode: bool = False,
) -> RuntimeBackendConfig:
    normalized_mode = (mode or "direct").strip().lower()
    if normalized_mode not in {"direct", "openai"}:
        normalized_mode = "direct"
    return RuntimeBackendConfig(
        mode=normalized_mode,
        base_url=(base_url or "").strip(),
        api_key=(api_key or "").strip(),
        model_id=(model_id or "").strip(),
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_tokens),
        json_mode=bool(json_mode),
    )


def _get_attr(obj: Any, name: str, default: Any = "") -> Any:
    if obj is None:
        return default
    value = getattr(obj, name, default)
    return default if value is None else value


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_float(value: Any, default: float) -> float:
    if value in (None, ""):
        return float(default)
    return float(value)


def _coerce_int(value: Any, default: int) -> int:
    if value in (None, ""):
        return int(default)
    return int(value)


def _normalize_model_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def find_model_config_section(
    config: Optional[Mapping[str, Any]],
    model_name: str,
    *,
    preferred_sections: tuple[str, ...] = (),
) -> Mapping[str, Any]:
    if not config or not model_name:
        return {}

    candidates: list[tuple[str, Mapping[str, Any]]] = []
    seen_names: set[str] = set()

    def add_candidate(name: str, section: Any) -> None:
        if name in seen_names or not isinstance(section, Mapping):
            return
        seen_names.add(name)
        candidates.append((name, section))

    for section_name in preferred_sections:
        add_candidate(section_name, config.get(section_name))

    for section_name, section in config.items():
        add_candidate(str(section_name), section)

    target = str(model_name).strip().lower()
    normalized_target = _normalize_model_key(model_name)

    for _, section in candidates:
        section_model = str(section.get("runtime_model_id") or section.get("model_id") or "").strip().lower()
        if section_model and section_model == target:
            return section

    for section_name, section in candidates:
        if _normalize_model_key(section_name) == normalized_target:
            return section

    return {}


def resolve_runtime_backend(
    args: Any,
    section: Optional[Mapping[str, Any]],
    *,
    arg_prefix: str,
    shared_prefix: str = "",
    default_model_id: str = "",
    default_temperature: float = 0.0,
    default_top_p: float = 1.0,
    default_max_tokens: int = 2048,
    default_json_mode: bool = False,
) -> RuntimeBackendConfig:
    """Resolve runtime backend from CLI args first, config section second."""

    section = section or {}

    explicit_mode = _get_attr(args, f"{arg_prefix}_backend", "")
    local_base_url = _get_attr(args, f"{arg_prefix}_base_url", "")
    shared_base_url = _get_attr(args, f"{shared_prefix}_base_url", "") if shared_prefix else ""
    shared_api_key = _get_attr(args, f"{shared_prefix}_api_key", "") if shared_prefix else ""
    shared_model_id = _get_attr(args, f"{shared_prefix}_model_name", "") if shared_prefix else ""

    base_url = (
        local_base_url
        or shared_base_url
        or section.get("runtime_base_url", "")
    )
    mode = explicit_mode or ("openai" if base_url else section.get("runtime_backend", "direct"))
    api_key = (
        _get_attr(args, f"{arg_prefix}_api_key", "")
        or shared_api_key
        or section.get("runtime_api_key", "")
    )
    model_id = (
        _get_attr(args, f"{arg_prefix}_model_id", "")
        or shared_model_id
        or section.get("runtime_model_id", "")
        or default_model_id
    )
    temperature = _coerce_float(
        _get_attr(args, f"{arg_prefix}_temperature", None)
        if _get_attr(args, f"{arg_prefix}_temperature", None) is not None
        else section.get("runtime_temperature", default_temperature),
        default_temperature,
    )
    top_p = _coerce_float(
        _get_attr(args, f"{arg_prefix}_top_p", None)
        if _get_attr(args, f"{arg_prefix}_top_p", None) is not None
        else section.get("runtime_top_p", default_top_p),
        default_top_p,
    )
    max_tokens = _coerce_int(
        _get_attr(args, f"{arg_prefix}_max_tokens", None)
        if _get_attr(args, f"{arg_prefix}_max_tokens", None) is not None
        else section.get("runtime_max_tokens", default_max_tokens),
        default_max_tokens,
    )
    json_mode = _coerce_bool(
        _get_attr(args, f"{arg_prefix}_json_mode", None)
        if _get_attr(args, f"{arg_prefix}_json_mode", None) is not None
        else section.get("runtime_json_mode", default_json_mode),
        default_json_mode,
    )

    return make_runtime_backend_config(
        mode=str(mode),
        base_url=str(base_url),
        api_key=str(api_key),
        model_id=str(model_id),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        json_mode=json_mode,
    )


def extract_message_text(content: Any) -> str:
    """Normalize OpenAI SDK message content to a plain string."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    chunks.append(str(text))
                continue
            text = getattr(item, "text", "")
            if text:
                chunks.append(str(text))
        return "".join(chunks)
    return str(content)


class OpenAIChatRuntime:
    """Thin wrapper around OpenAI-compatible chat completions."""

    def __init__(self, config: RuntimeBackendConfig):
        self.config = config

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict[str, Any]] = None,
        stop: Optional[list[str]] = None,
    ) -> str:
        if not self.config.base_url:
            raise RuntimeError("OPENAI_RUNTIME_BASE_URL_MISSING")
        if not self.config.model_id:
            raise RuntimeError("OPENAI_RUNTIME_MODEL_ID_MISSING")

        from openai import OpenAI

        client = OpenAI(
            api_key=self.config.api_key or "sk-no-key-required",
            base_url=self.config.base_url,
        )
        request = {
            "model": self.config.model_id,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else float(temperature),
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens if max_tokens is None else int(max_tokens),
        }
        if response_format is not None:
            request["response_format"] = response_format
        if stop:
            request["stop"] = stop

        completion = client.chat.completions.create(**request)
        return extract_message_text(completion.choices[0].message.content)
