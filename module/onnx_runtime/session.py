"""Shared ONNX Runtime session creation and caching."""

from __future__ import annotations

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


def _provider_name(provider: Any) -> str:
    return provider[0] if isinstance(provider, tuple) else str(provider)


def select_execution_providers(
    preference: str = "auto",
    *,
    available_providers: list[str] | tuple[str, ...] | None = None,
    session_cache_dir: str | Path | None = None,
) -> list[Any]:
    if available_providers is None:
        import onnxruntime as ort

        available = list(ort.get_available_providers())
    else:
        available = list(available_providers)

    normalized = str(preference or "auto").strip().lower()
    cache_dir = str(Path(session_cache_dir)) if session_cache_dir else ""

    if normalized in {"cpu", "cpuexecutionprovider"}:
        return ["CPUExecutionProvider"]

    explicit_map = {
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
        "nvtensorrtrtx": "NvTensorRtRtxExecutionProvider",
        "rocm": "ROCMExecutionProvider",
        "openvino": "OpenVINOExecutionProvider",
    }
    explicit_name = explicit_map.get(normalized, preference)
    if normalized not in {"", "auto"} and explicit_name in available:
        result: list[Any] = [explicit_name]
        if explicit_name != "CPUExecutionProvider" and "CPUExecutionProvider" in available:
            result.append("CPUExecutionProvider")
        return result

    if "NvTensorRtRtxExecutionProvider" in available:
        return [
            (
                "NvTensorRtRtxExecutionProvider",
                {
                    "nv_runtime_cache_path": cache_dir,
                },
            ),
            "CUDAExecutionProvider" if "CUDAExecutionProvider" in available else "CPUExecutionProvider",
            "CPUExecutionProvider",
        ]

    if "TensorrtExecutionProvider" in available:
        return [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": cache_dir,
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": cache_dir,
                },
            ),
            "CUDAExecutionProvider" if "CUDAExecutionProvider" in available else "CPUExecutionProvider",
            "CPUExecutionProvider",
        ]

    if "CUDAExecutionProvider" in available:
        return [
            (
                "CUDAExecutionProvider",
                {
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            ),
            "CPUExecutionProvider",
        ]

    if "ROCMExecutionProvider" in available:
        return ["ROCMExecutionProvider", "CPUExecutionProvider" if "CPUExecutionProvider" in available else "ROCMExecutionProvider"]

    if "OpenVINOExecutionProvider" in available:
        return [("OpenVINOExecutionProvider", {"device_type": "GPU_FP32"}), "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


def _default_session_options_factory():
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_mem_pattern = True
    sess_options.enable_mem_reuse = True
    return sess_options


def _make_cache_key(bundle_key: str, session_paths: Mapping[str, str | Path], providers: list[Any]) -> tuple[Any, ...]:
    normalized_paths = tuple(sorted((name, str(Path(path))) for name, path in session_paths.items()))
    provider_names = tuple(_provider_name(provider) for provider in providers)
    return (bundle_key, normalized_paths, provider_names)


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
    providers = select_execution_providers(
        runtime.execution_provider,
        available_providers=available_providers,
        session_cache_dir=runtime.session_cache_dir,
    )
    cache_key = _make_cache_key(bundle_key, session_paths, providers)

    with _CACHE_LOCK:
        cached = _SESSION_BUNDLE_CACHE.get(cache_key)
        if cached is not None:
            return cached

    if session_factory is None:
        import onnxruntime as ort

        session_factory = ort.InferenceSession
    if session_options_factory is None:
        session_options_factory = _default_session_options_factory

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
