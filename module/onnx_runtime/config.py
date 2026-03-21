"""Configuration objects for shared ONNX runtime helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

_RUNTIME_FIELDS = {
    "execution_provider",
    "model_cache_dir",
    "session_cache_dir",
    "force_download",
    "graph_optimization_level",
    "enable_mem_pattern",
    "enable_mem_reuse",
    "execution_mode",
    "inter_op_num_threads",
    "intra_op_num_threads",
}
_SESSION_FIELDS = {
    "graph_optimization_level",
    "enable_mem_pattern",
    "enable_mem_reuse",
    "execution_mode",
    "inter_op_num_threads",
    "intra_op_num_threads",
}
_PROVIDER_SECTIONS = ("cuda", "tensorrt", "nvtensorrtrtx", "openvino")


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = {key: value for key, value in base.items()}
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(_as_mapping(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            frozen[key] = _freeze_mapping(item)
        else:
            frozen[key] = item
    return MappingProxyType(frozen)


def _thaw_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    thawed: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            thawed[key] = _thaw_mapping(item)
        else:
            thawed[key] = item
    return thawed


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _coerce_int(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _default_provider_options() -> dict[str, dict[str, Any]]:
    return {
        "cuda": {
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
            "cudnn_conv_use_max_workspace": "1",
            "tunable_op_enable": True,
            "tunable_op_tuning_enable": True,
        },
        "tensorrt": {
            "engine_cache_enable": True,
            "timing_cache_enable": True,
            "fp16_enable": True,
            "builder_optimization_level": 3,
            "max_partition_iterations": 1000,
            "engine_hw_compatible": True,
            "force_sequential_engine_build": False,
            "context_memory_sharing_enable": True,
            "sparsity_enable": True,
            "min_subgraph_size": 7,
        },
        "nvtensorrtrtx": {
            "nv_dump_subgraphs": False,
            "nv_detailed_build_log": True,
            "enable_cuda_graph": True,
            "nv_multi_profile_enable": False,
            "nv_use_external_data_initializer": False,
        },
        "openvino": {
            "device_type": "GPU_FP32",
        },
    }


@dataclass(frozen=True)
class OnnxRuntimeConfig:
    execution_provider: str = "auto"
    model_cache_dir: str = ""
    session_cache_dir: str = ""
    force_download: bool = False
    graph_optimization_level: str = "ORT_ENABLE_ALL"
    enable_mem_pattern: bool = True
    enable_mem_reuse: bool = True
    execution_mode: str = ""
    inter_op_num_threads: int = 0
    intra_op_num_threads: int = 0
    provider_options: Mapping[str, Mapping[str, Any]] = field(
        default_factory=lambda: _freeze_mapping(_default_provider_options())
    )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> "OnnxRuntimeConfig":
        data = data or {}
        session = _as_mapping(data.get("session"))
        provider_options = _default_provider_options()

        explicit_provider_options = _as_mapping(data.get("provider_options"))
        for provider_name in _PROVIDER_SECTIONS:
            merged_provider = provider_options.get(provider_name, {}).copy()
            merged_provider.update(_as_mapping(data.get(provider_name)))
            merged_provider.update(_as_mapping(explicit_provider_options.get(provider_name)))
            provider_options[provider_name] = merged_provider

        return cls(
            execution_provider=str(data.get("execution_provider", "auto") or "auto"),
            model_cache_dir=str(data.get("model_cache_dir", "") or ""),
            session_cache_dir=str(data.get("session_cache_dir", "") or ""),
            force_download=_coerce_bool(data.get("force_download"), False),
            graph_optimization_level=str(
                data.get("graph_optimization_level", session.get("graph_optimization_level", "ORT_ENABLE_ALL"))
                or "ORT_ENABLE_ALL"
            ),
            enable_mem_pattern=_coerce_bool(
                data.get("enable_mem_pattern", session.get("enable_mem_pattern")),
                True,
            ),
            enable_mem_reuse=_coerce_bool(
                data.get("enable_mem_reuse", session.get("enable_mem_reuse")),
                True,
            ),
            execution_mode=str(data.get("execution_mode", session.get("execution_mode", "")) or ""),
            inter_op_num_threads=_coerce_int(
                data.get("inter_op_num_threads", session.get("inter_op_num_threads")),
                0,
            ),
            intra_op_num_threads=_coerce_int(
                data.get("intra_op_num_threads", session.get("intra_op_num_threads")),
                0,
            ),
            provider_options=_freeze_mapping(provider_options),
        )

    @classmethod
    def from_runtime_sections(
        cls,
        *,
        defaults: Mapping[str, Any] | None = None,
        legacy: Mapping[str, Any] | None = None,
        override: Mapping[str, Any] | None = None,
        cli_override: Mapping[str, Any] | None = None,
    ) -> "OnnxRuntimeConfig":
        merged: dict[str, Any] = {}
        for section in (defaults, legacy, override, cli_override):
            if section:
                merged = _deep_merge(merged, section)
        return cls.from_mapping(merged)

    def resolve_model_cache_dir(self, model_id: str) -> Path:
        if self.model_cache_dir:
            return Path(self.model_cache_dir)
        return Path("huggingface") / model_id.replace("/", "_")

    def runtime_fingerprint_payload(self) -> dict[str, Any]:
        return {
            "execution_provider": self.execution_provider,
            "session_cache_dir": self.session_cache_dir,
            "graph_optimization_level": self.graph_optimization_level,
            "enable_mem_pattern": self.enable_mem_pattern,
            "enable_mem_reuse": self.enable_mem_reuse,
            "execution_mode": self.execution_mode,
            "inter_op_num_threads": self.inter_op_num_threads,
            "intra_op_num_threads": self.intra_op_num_threads,
            "provider_options": _thaw_mapping(self.provider_options),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_provider": self.execution_provider,
            "model_cache_dir": self.model_cache_dir,
            "session_cache_dir": self.session_cache_dir,
            "force_download": self.force_download,
            "graph_optimization_level": self.graph_optimization_level,
            "enable_mem_pattern": self.enable_mem_pattern,
            "enable_mem_reuse": self.enable_mem_reuse,
            "execution_mode": self.execution_mode,
            "inter_op_num_threads": self.inter_op_num_threads,
            "intra_op_num_threads": self.intra_op_num_threads,
            "provider_options": _thaw_mapping(self.provider_options),
        }


def legacy_runtime_fields(data: Mapping[str, Any] | None = None) -> dict[str, Any]:
    section = data or {}
    legacy: dict[str, Any] = {}
    for field_name in _RUNTIME_FIELDS:
        if field_name in section:
            legacy[field_name] = section[field_name]
    for provider_name in _PROVIDER_SECTIONS:
        provider_section = _as_mapping(section.get(provider_name))
        if provider_section:
            legacy[provider_name] = dict(provider_section)
    return legacy


def resolve_tool_runtime_config(
    config: Mapping[str, Any] | None,
    *,
    tool_name: str,
    legacy: Mapping[str, Any] | None = None,
    cli_override: Mapping[str, Any] | None = None,
) -> OnnxRuntimeConfig:
    merged_config = config or {}
    onnx_runtime = _as_mapping(merged_config.get("onnx_runtime"))
    return OnnxRuntimeConfig.from_runtime_sections(
        defaults=_as_mapping(onnx_runtime.get("defaults")),
        legacy=legacy_runtime_fields(legacy),
        override=_as_mapping(onnx_runtime.get(tool_name)),
        cli_override=cli_override,
    )
