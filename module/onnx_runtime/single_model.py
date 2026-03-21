"""Single-model ONNX artifact and session helpers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .artifacts import download_onnx_artifact
from .config import OnnxRuntimeConfig
from .session import OnnxSessionBundle, load_session_bundle


@dataclass(frozen=True)
class OnnxModelSpec:
    repo_id: str
    onnx_filename: str
    local_dir: str | Path
    bundle_key: str


@dataclass(frozen=True)
class SingleModelOnnxBundle:
    model_path: Path
    session: Any
    providers: tuple[Any, ...]
    input_metas: tuple[Any, ...]
    runtime_config: OnnxRuntimeConfig
    session_bundle: OnnxSessionBundle


def _supports_keyword_argument(func: Callable[..., Any], name: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True

    if name in signature.parameters:
        return True
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())


def load_single_model_bundle(
    *,
    spec: OnnxModelSpec,
    runtime_config: OnnxRuntimeConfig | None = None,
    artifact_loader: Callable[..., Path] | None = None,
    session_bundle_loader: Callable[..., OnnxSessionBundle] | None = None,
    logger: Callable[..., Any] | None = None,
) -> SingleModelOnnxBundle:
    runtime = runtime_config or OnnxRuntimeConfig()
    artifact_loader = artifact_loader or download_onnx_artifact
    session_bundle_loader = session_bundle_loader or load_session_bundle

    artifact_kwargs = {
        "local_dir": spec.local_dir,
        "force_download": runtime.force_download,
    }
    if logger is not None and _supports_keyword_argument(artifact_loader, "logger"):
        artifact_kwargs["logger"] = logger

    model_path = Path(
        artifact_loader(
            spec.repo_id,
            spec.onnx_filename,
            **artifact_kwargs,
        )
    )
    session_bundle = session_bundle_loader(
        bundle_key=spec.bundle_key,
        session_paths={"model": model_path},
        runtime_config=runtime,
    )
    session = session_bundle.sessions["model"]

    return SingleModelOnnxBundle(
        model_path=model_path,
        session=session,
        providers=tuple(session_bundle.providers),
        input_metas=tuple(session.get_inputs()),
        runtime_config=runtime,
        session_bundle=session_bundle,
    )
