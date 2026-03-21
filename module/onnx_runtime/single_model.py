"""Single-model ONNX artifact and session helpers."""

from __future__ import annotations

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


def load_single_model_bundle(
    *,
    spec: OnnxModelSpec,
    runtime_config: OnnxRuntimeConfig | None = None,
    artifact_loader: Callable[..., Path] | None = None,
    session_bundle_loader: Callable[..., OnnxSessionBundle] | None = None,
) -> SingleModelOnnxBundle:
    runtime = runtime_config or OnnxRuntimeConfig()
    artifact_loader = artifact_loader or download_onnx_artifact
    session_bundle_loader = session_bundle_loader or load_session_bundle

    model_path = Path(
        artifact_loader(
            spec.repo_id,
            spec.onnx_filename,
            local_dir=spec.local_dir,
            force_download=runtime.force_download,
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
