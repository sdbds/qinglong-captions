"""Shared ONNX Runtime session creation and caching."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .config import OnnxRuntimeConfig

_SESSION_BUNDLE_CACHE: dict[tuple[Any, ...], "OnnxSessionBundle"] = {}
_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class OnnxSessionBundle:
    sessions: dict[str, Any]
    providers: tuple[Any, ...]


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize_for_json(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(item) for item in value]
    return value


def _stable_digest(payload: Any) -> str:
    normalized = _normalize_for_json(payload)
    serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _provider_name(provider: Any) -> str:
    return provider[0] if isinstance(provider, tuple) else str(provider)


def _provider_signature(provider: Any) -> tuple[str, str]:
    if isinstance(provider, tuple):
        name, options = provider
        return (str(name), _stable_digest(options))
    return (str(provider), "")


def _runtime_cache_dir(runtime: OnnxRuntimeConfig) -> str:
    return str(Path(runtime.session_cache_dir)) if runtime.session_cache_dir else ""


def _cuda_provider(runtime: OnnxRuntimeConfig) -> tuple[str, dict[str, Any]]:
    return ("CUDAExecutionProvider", dict(runtime.provider_options.get("cuda", {})))


def _tensorrt_provider(runtime: OnnxRuntimeConfig) -> tuple[str, dict[str, Any]]:
    raw = dict(runtime.provider_options.get("tensorrt", {}))
    cache_dir = _runtime_cache_dir(runtime)
    options = {
        "trt_engine_cache_enable": raw.get("engine_cache_enable", True),
        "trt_timing_cache_enable": raw.get("timing_cache_enable", True),
        "trt_fp16_enable": raw.get("fp16_enable", True),
    }
    if "builder_optimization_level" in raw:
        options["trt_builder_optimization_level"] = raw["builder_optimization_level"]
    if cache_dir:
        options["trt_engine_cache_path"] = raw.get("engine_cache_path", cache_dir)
        options["trt_timing_cache_path"] = raw.get("timing_cache_path", cache_dir)
    return ("TensorrtExecutionProvider", options)


def _nvtensorrtrtx_provider(runtime: OnnxRuntimeConfig) -> tuple[str, dict[str, Any]]:
    raw = dict(runtime.provider_options.get("nvtensorrtrtx", {}))
    cache_dir = _runtime_cache_dir(runtime)
    options: dict[str, Any] = {}
    if cache_dir:
        options["nv_runtime_cache_path"] = raw.get("runtime_cache_path", cache_dir)
    for key, value in raw.items():
        if key == "runtime_cache_path":
            continue
        options[key] = value
    return ("NvTensorRtRtxExecutionProvider", options)


def _openvino_provider(runtime: OnnxRuntimeConfig) -> tuple[str, dict[str, Any]]:
    return ("OpenVINOExecutionProvider", dict(runtime.provider_options.get("openvino", {})))


def build_execution_providers(
    runtime_config: OnnxRuntimeConfig,
    *,
    available_providers: list[str] | tuple[str, ...] | None = None,
) -> list[Any]:
    if available_providers is None:
        import onnxruntime as ort

        available = list(ort.get_available_providers())
    else:
        available = list(available_providers)

    normalized = str(runtime_config.execution_provider or "auto").strip().lower()
    explicit_map = {
        "cpu": "CPUExecutionProvider",
        "cpuexecutionprovider": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "cudaexecutionprovider": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
        "tensorrtexecutionprovider": "TensorrtExecutionProvider",
        "nvtensorrtrtx": "NvTensorRtRtxExecutionProvider",
        "nvtensorrtrtxexecutionprovider": "NvTensorRtRtxExecutionProvider",
        "rocm": "ROCMExecutionProvider",
        "rocmexecutionprovider": "ROCMExecutionProvider",
        "openvino": "OpenVINOExecutionProvider",
        "openvinoexecutionprovider": "OpenVINOExecutionProvider",
    }
    explicit_name = explicit_map.get(normalized, "")

    def with_cpu_fallback(primary: Any, *extra: Any) -> list[Any]:
        providers = [primary, *extra]
        if "CPUExecutionProvider" in available and all(_provider_name(item) != "CPUExecutionProvider" for item in providers):
            providers.append("CPUExecutionProvider")
        return providers

    if explicit_name == "CPUExecutionProvider":
        return ["CPUExecutionProvider"]
    if explicit_name == "CUDAExecutionProvider" and "CUDAExecutionProvider" in available:
        return with_cpu_fallback(_cuda_provider(runtime_config))
    if explicit_name == "TensorrtExecutionProvider" and "TensorrtExecutionProvider" in available:
        extras: list[Any] = []
        if "CUDAExecutionProvider" in available:
            extras.append(_cuda_provider(runtime_config))
        return with_cpu_fallback(_tensorrt_provider(runtime_config), *extras)
    if explicit_name == "NvTensorRtRtxExecutionProvider" and "NvTensorRtRtxExecutionProvider" in available:
        extras = []
        if "CUDAExecutionProvider" in available:
            extras.append(_cuda_provider(runtime_config))
        return with_cpu_fallback(_nvtensorrtrtx_provider(runtime_config), *extras)
    if explicit_name == "ROCMExecutionProvider" and "ROCMExecutionProvider" in available:
        return with_cpu_fallback("ROCMExecutionProvider")
    if explicit_name == "OpenVINOExecutionProvider" and "OpenVINOExecutionProvider" in available:
        return with_cpu_fallback(_openvino_provider(runtime_config))

    if "NvTensorRtRtxExecutionProvider" in available:
        extras = []
        if "CUDAExecutionProvider" in available:
            extras.append(_cuda_provider(runtime_config))
        return with_cpu_fallback(_nvtensorrtrtx_provider(runtime_config), *extras)

    if "TensorrtExecutionProvider" in available:
        extras = []
        if "CUDAExecutionProvider" in available:
            extras.append(_cuda_provider(runtime_config))
        return with_cpu_fallback(_tensorrt_provider(runtime_config), *extras)

    if "CUDAExecutionProvider" in available:
        return with_cpu_fallback(_cuda_provider(runtime_config))

    if "ROCMExecutionProvider" in available:
        return with_cpu_fallback("ROCMExecutionProvider")

    if "OpenVINOExecutionProvider" in available:
        return with_cpu_fallback(_openvino_provider(runtime_config))

    return ["CPUExecutionProvider"]


def select_execution_providers(
    preference: str = "auto",
    *,
    available_providers: list[str] | tuple[str, ...] | None = None,
    session_cache_dir: str | Path | None = None,
) -> list[Any]:
    runtime = OnnxRuntimeConfig(
        execution_provider=preference,
        session_cache_dir=str(Path(session_cache_dir)) if session_cache_dir else "",
    )
    return build_execution_providers(runtime, available_providers=available_providers)


def build_session_options(runtime_config: OnnxRuntimeConfig):
    import onnxruntime as ort

    sess_options = ort.SessionOptions()

    graph_level_name = str(runtime_config.graph_optimization_level or "ORT_ENABLE_ALL")
    graph_level = getattr(ort.GraphOptimizationLevel, graph_level_name, None)
    if graph_level is not None:
        sess_options.graph_optimization_level = graph_level

    sess_options.enable_mem_pattern = bool(runtime_config.enable_mem_pattern)
    sess_options.enable_mem_reuse = bool(runtime_config.enable_mem_reuse)

    execution_mode_name = str(runtime_config.execution_mode or "").strip().upper()
    if execution_mode_name:
        if not execution_mode_name.startswith("ORT_"):
            execution_mode_name = f"ORT_{execution_mode_name}"
        execution_mode = getattr(ort.ExecutionMode, execution_mode_name, None)
        if execution_mode is not None:
            sess_options.execution_mode = execution_mode

    if runtime_config.inter_op_num_threads > 0:
        sess_options.inter_op_num_threads = int(runtime_config.inter_op_num_threads)
    if runtime_config.intra_op_num_threads > 0:
        sess_options.intra_op_num_threads = int(runtime_config.intra_op_num_threads)

    return sess_options


def make_runtime_fingerprint(runtime_config: OnnxRuntimeConfig, providers: list[Any]) -> str:
    payload = {
        "runtime": runtime_config.runtime_fingerprint_payload(),
        "providers": [_provider_signature(provider) for provider in providers],
    }
    return _stable_digest(payload)


def _make_cache_key(
    bundle_key: str,
    session_paths: Mapping[str, str | Path],
    providers: list[Any],
    runtime_fingerprint: str,
) -> tuple[Any, ...]:
    normalized_paths = tuple(sorted((name, str(Path(path))) for name, path in session_paths.items()))
    provider_descriptors = tuple(_provider_signature(provider) for provider in providers)
    return (bundle_key, normalized_paths, provider_descriptors, runtime_fingerprint)


def load_session_bundle(
    *,
    bundle_key: str,
    session_paths: Mapping[str, str | Path],
    runtime_config: OnnxRuntimeConfig | None = None,
    available_providers: list[str] | tuple[str, ...] | None = None,
    session_factory: Callable[..., Any] | None = None,
    session_options_factory: Callable[[], Any] | None = None,
) -> OnnxSessionBundle:
    runtime = runtime_config or OnnxRuntimeConfig()
    providers = build_execution_providers(runtime, available_providers=available_providers)
    runtime_fingerprint = make_runtime_fingerprint(runtime, providers)
    cache_key = _make_cache_key(bundle_key, session_paths, providers, runtime_fingerprint)

    with _CACHE_LOCK:
        cached = _SESSION_BUNDLE_CACHE.get(cache_key)
        if cached is not None:
            return cached

    if session_factory is None:
        import onnxruntime as ort

        session_factory = ort.InferenceSession
    if session_options_factory is None:
        session_options_factory = lambda: build_session_options(runtime)

    sessions = {
        name: session_factory(str(Path(path)), sess_options=session_options_factory(), providers=providers)
        for name, path in session_paths.items()
    }
    bundle = OnnxSessionBundle(sessions=sessions, providers=tuple(providers))

    with _CACHE_LOCK:
        _SESSION_BUNDLE_CACHE[cache_key] = bundle

    return bundle


def clear_session_bundle_cache(bundle_key_prefix: str | None = None) -> None:
    with _CACHE_LOCK:
        if not bundle_key_prefix:
            _SESSION_BUNDLE_CACHE.clear()
            return

        keys = [key for key in _SESSION_BUNDLE_CACHE if str(key[0]).startswith(bundle_key_prefix)]
        for key in keys:
            _SESSION_BUNDLE_CACHE.pop(key, None)
