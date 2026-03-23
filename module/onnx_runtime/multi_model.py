"""Multi-model ONNX artifact and session helpers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .artifacts import download_onnx_artifact_set, download_repo_file_set
from .config import OnnxRuntimeConfig
from .session import OnnxSessionBundle, load_session_bundle


@dataclass(frozen=True)
class OnnxMultiModelSpec:
    repo_id: str
    artifacts: Mapping[str, str]
    local_dir: str | Path
    bundle_key: str
    support_files: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MultiModelOnnxBundle:
    artifact_paths: dict[str, Path]
    support_paths: dict[str, Path]
    sessions: dict[str, Any]
    providers: tuple[Any, ...]
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


def load_multi_model_bundle(
    *,
    spec: OnnxMultiModelSpec,
    runtime_config: OnnxRuntimeConfig | None = None,
    artifact_loader: Callable[..., dict[str, Path]] | None = None,
    support_file_loader: Callable[..., dict[str, Path]] | None = None,
    session_bundle_loader: Callable[..., OnnxSessionBundle] | None = None,
    logger: Callable[..., Any] | None = None,
) -> MultiModelOnnxBundle:
    runtime = runtime_config or OnnxRuntimeConfig()
    artifact_loader = artifact_loader or download_onnx_artifact_set
    support_file_loader = support_file_loader or download_repo_file_set
    session_bundle_loader = session_bundle_loader or load_session_bundle

    artifact_kwargs = {
        "local_dir": spec.local_dir,
        "force_download": runtime.force_download,
    }
    if logger is not None and _supports_keyword_argument(artifact_loader, "logger"):
        artifact_kwargs["logger"] = logger

    artifact_paths = {
        name: Path(path)
        for name, path in artifact_loader(
            spec.repo_id,
            dict(spec.artifacts),
            **artifact_kwargs,
        ).items()
    }

    support_paths: dict[str, Path] = {}
    if spec.support_files:
        support_kwargs = {
            "local_dir": spec.local_dir,
            "force_download": runtime.force_download,
        }
        if logger is not None and _supports_keyword_argument(support_file_loader, "logger"):
            support_kwargs["logger"] = logger
        support_paths = {
            name: Path(path)
            for name, path in support_file_loader(
                spec.repo_id,
                dict(spec.support_files),
                **support_kwargs,
            ).items()
        }

    session_bundle = session_bundle_loader(
        bundle_key=spec.bundle_key,
        session_paths=artifact_paths,
        runtime_config=runtime,
    )
    return MultiModelOnnxBundle(
        artifact_paths=artifact_paths,
        support_paths=support_paths,
        sessions=dict(session_bundle.sessions),
        providers=tuple(session_bundle.providers),
        runtime_config=runtime,
        session_bundle=session_bundle,
    )
