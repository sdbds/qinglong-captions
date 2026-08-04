from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path


def default_pose_model_cache_dir() -> Path:
    configured = os.environ.get("QINGLONG_CAPTIONS_MODEL_CACHE")
    base = Path(configured).expanduser() if configured else Path.home() / ".cache" / "qinglong-captions" / "models"
    return base / "auto-rig" / "pose"


class PoseProviderPool:
    """Own at most one loaded instance of each heavyweight pose backend."""

    def __init__(
        self,
        *,
        model_cache_dir: str | Path | None = None,
        sdpose_bundle_path: str | Path | None = None,
        detrpose_weights_path: str | Path | None = None,
        device: str | None = None,
        prefer_fa2: bool = True,
        provider_loader: Callable[[str], object] | None = None,
    ) -> None:
        self._cache = Path(model_cache_dir if model_cache_dir is not None else default_pose_model_cache_dir()).expanduser()
        self._sdpose_bundle_path = sdpose_bundle_path
        self._detrpose_weights_path = detrpose_weights_path
        self._device = device
        self._prefer_fa2 = prefer_fa2
        self._provider_loader = provider_loader or self._load_builtin
        self._providers: dict[str, object] = {}
        self._closed = False
        self._lock = threading.RLock()

    def _load_builtin(self, provider_key: str) -> object:
        if provider_key == "sdpose":
            from .artifacts import (
                SDPOSE_BODY_SOURCE,
                PoseArtifactError,
                download_pose_source,
            )
            from .bundle import (
                SDPOSE_BODY_BUNDLE_FILENAME,
                build_sdpose_body_bundle,
                inspect_sdpose_body_bundle,
            )
            from .loader import load_sdpose_body_models
            from .sdpose import SDPoseBodyProvider

            bundle = Path(
                self._sdpose_bundle_path
                if self._sdpose_bundle_path is not None
                else self._cache / "sdpose-body17" / SDPOSE_BODY_BUNDLE_FILENAME
            ).expanduser()
            try:
                inspect_sdpose_body_bundle(
                    bundle,
                    source_contract=SDPOSE_BODY_SOURCE,
                )
            except (FileNotFoundError, OSError, PoseArtifactError, ValueError):
                source = download_pose_source(SDPOSE_BODY_SOURCE)
                build_sdpose_body_bundle(
                    source,
                    bundle,
                    source_contract=SDPOSE_BODY_SOURCE,
                )
                inspect_sdpose_body_bundle(
                    bundle,
                    source_contract=SDPOSE_BODY_SOURCE,
                )
            return SDPoseBodyProvider(
                load_sdpose_body_models(
                    bundle,
                    device=self._device,
                    prefer_fa2=self._prefer_fa2,
                    verify_bundle=False,
                )
            )
        if provider_key == "detrpose":
            from .artifacts import DETRPOSE_X_CROWDPOSE_MODEL, download_pose_model
            from .detrpose import DETRPoseXProvider, load_detrpose_x_crowdpose

            explicit = self._detrpose_weights_path is not None
            weights = (
                Path(self._detrpose_weights_path).expanduser() if explicit else download_pose_model(DETRPOSE_X_CROWDPOSE_MODEL)
            )
            return DETRPoseXProvider(
                load_detrpose_x_crowdpose(
                    weights,
                    device=self._device,
                    verify_weights=explicit,
                )
            )
        raise ValueError(f"unknown built-in pose provider: {provider_key}")

    def resolve(self, provider_key: str) -> object:
        with self._lock:
            if self._closed:
                raise RuntimeError("pose provider pool is closed")
            if provider_key not in self._providers:
                self._providers[provider_key] = self._provider_loader(provider_key)
            return self._providers[provider_key]

    def __call__(self, provider_key: str) -> object:
        return self.resolve(provider_key)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            providers = tuple(self._providers.values())
            self._providers.clear()
            self._closed = True
        for provider in providers:
            close = getattr(provider, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "PoseProviderPool":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def make_builtin_pose_resolver(
    *,
    model_cache_dir: str | Path | None = None,
    sdpose_bundle_path: str | Path | None = None,
    detrpose_weights_path: str | Path | None = None,
    device: str | None = None,
    prefer_fa2: bool = True,
) -> PoseProviderPool:
    return PoseProviderPool(
        model_cache_dir=model_cache_dir,
        sdpose_bundle_path=sdpose_bundle_path,
        detrpose_weights_path=detrpose_weights_path,
        device=device,
        prefer_fa2=prefer_fa2,
    )


__all__ = [
    "PoseProviderPool",
    "default_pose_model_cache_dir",
    "make_builtin_pose_resolver",
]
