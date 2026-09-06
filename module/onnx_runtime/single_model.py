"""Single-model ONNX artifact and session helpers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .artifacts import download_onnx_artifact, download_repo_file_set
from .config import OnnxRuntimeConfig
from .session import OnnxSessionBundle, load_session_bundle, refresh_session_artifacts


@dataclass(frozen=True)
class OnnxModelSpec:
    repo_id: str
    onnx_filename: str
    local_dir: str | Path
    bundle_key: str
    support_files: Mapping[str, str] = field(default_factory=dict)
    revision: str | None = None


@dataclass(frozen=True)
class SingleModelOnnxBundle:
    model_path: Path
    support_paths: dict[str, Path]
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
    support_file_loader: Callable[..., dict[str, Path]] | None = None,
    session_bundle_loader: Callable[..., OnnxSessionBundle] | None = None,
    logger: Callable[..., Any] | None = None,
) -> SingleModelOnnxBundle:
    runtime = runtime_config or OnnxRuntimeConfig()
    artifact_loader = artifact_loader or download_onnx_artifact
    support_file_loader = support_file_loader or download_repo_file_set
    session_bundle_loader = session_bundle_loader or load_session_bundle

    artifact_kwargs = {
        "local_dir": spec.local_dir,
        "force_download": runtime.force_download,
    }
    if spec.revision is not None:
        artifact_kwargs["revision"] = spec.revision
    if logger is not None and _supports_keyword_argument(artifact_loader, "logger"):
        artifact_kwargs["logger"] = logger

    with refresh_session_artifacts(
        {"model": Path(spec.local_dir) / spec.onnx_filename}, enabled=runtime.force_download,
    ):
        model_path = Path(
            artifact_loader(
                spec.repo_id,
                spec.onnx_filename,
                **artifact_kwargs,
            )
        )
        support_paths: dict[str, Path] = {}
        if spec.support_files:
            support_kwargs = {
                "local_dir": spec.local_dir,
                "force_download": runtime.force_download,
            }
            if spec.revision is not None:
                support_kwargs["revision"] = spec.revision
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
    session_kwargs = {}
    if spec.revision is not None and _supports_keyword_argument(session_bundle_loader, "artifact_revision"):
        session_kwargs["artifact_revision"] = spec.revision
    session_bundle = session_bundle_loader(
        bundle_key=spec.bundle_key,
        session_paths={"model": model_path},
        runtime_config=runtime,
        **session_kwargs,
    )
    session = session_bundle.sessions["model"]

    return SingleModelOnnxBundle(
        model_path=model_path,
        support_paths=support_paths,
        session=session,
        providers=tuple(session_bundle.providers),
        input_metas=tuple(session.get_inputs()),
        runtime_config=runtime,
        session_bundle=session_bundle,
    )
