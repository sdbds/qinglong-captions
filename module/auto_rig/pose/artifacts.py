from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Literal

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PoseArtifactError(RuntimeError):
    """Raised when a pose model artifact violates its pinned contract."""


def _validate_artifact_fields(relative_path: str, size: int, sha256: str) -> None:
    path = PurePosixPath(relative_path)
    if not relative_path or "\\" in relative_path or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("pose artifact path must be normalized and relative")
    if size <= 0:
        raise ValueError("pose artifact size must be positive")
    if _SHA256.fullmatch(sha256) is None:
        raise ValueError("pose artifact SHA-256 must be 64 lowercase hex characters")


@dataclass(frozen=True, slots=True)
class PoseArtifactFile:
    relative_path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        _validate_artifact_fields(self.relative_path, self.size, self.sha256)


@dataclass(frozen=True, slots=True)
class PoseModelContract:
    provider_id: str
    repo_id: str
    revision: str
    relative_path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.provider_id or not self.repo_id or not self.revision:
            raise ValueError("pose model identity must be non-empty")
        _validate_artifact_fields(self.relative_path, self.size, self.sha256)


@dataclass(frozen=True, slots=True)
class PoseSourceContract:
    provider_id: str
    repo_id: str
    revision: str
    files: tuple[PoseArtifactFile, ...]

    def __post_init__(self) -> None:
        if not self.provider_id or not self.repo_id or not self.revision:
            raise ValueError("pose source identity must be non-empty")
        paths = tuple(item.relative_path for item in self.files)
        if not paths or len(paths) != len(set(paths)):
            raise ValueError("pose source files must be non-empty and unique")


SDPOSE_BODY_SOURCE = PoseSourceContract(
    provider_id="sdpose-body17",
    repo_id="teemosliang/SDPose-Body",
    revision="5a34e0c7df4c8ea5fc8774c5f2ae4229e962238c",
    files=(
        PoseArtifactFile(
            "decoder/decoder.safetensors",
            6_986_756,
            "32994dfc90beb84786c8e9296eeef60dad66980d86d7349be7c7ff80f3aaa8a4",
        ),
        PoseArtifactFile(
            "scheduler/scheduler_config.json",
            344,
            "ce14ed1d0a58d10a1e22b2d16786ecde6b14ef52cd0e21f2631eca39b49fac63",
        ),
        PoseArtifactFile(
            "unet/config.json",
            1_872,
            "39e3b8a8550583c3aa15de950526aa6cccc3dc9965fb18ae586d214bd80b1ff4",
        ),
        PoseArtifactFile(
            "unet/diffusion_pytorch_model.safetensors",
            3_470_311_272,
            "a75d358808e58cd5eb305dd3362d0d1457d243d787ac4bf1905b64da71d8934a",
        ),
        PoseArtifactFile(
            "vae/config.json",
            611,
            "d69281aa3f6a0f3c41aaf6778e35464fc6ee8a92e6ac8a8b1eb679f6df6423eb",
        ),
        PoseArtifactFile(
            "vae/diffusion_pytorch_model.safetensors",
            334_643_276,
            "a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815",
        ),
    ),
)

COMFY_SDPOSE_WHOLEBODY_MODEL = PoseModelContract(
    provider_id="sdpose-wholebody-reference",
    repo_id="Comfy-Org/SDPose",
    revision="1c1c71485e57dd0dc87e0a94a2ecfc7e4df07930",
    relative_path="checkpoints/sdpose_wholebody_fp16.safetensors",
    size=1_916_645_792,
    sha256="63d01f9a7494560693b24767f4469d59c9d3266b31ff0a253e74d1e611442721",
)

DETRPOSE_X_CROWDPOSE_MODEL = PoseModelContract(
    provider_id="detrpose-x-crowdpose",
    repo_id="SebasJanampa/DETRPose_X_CROWDPOSE",
    revision="cebc9cb1ad6289f262262412f604fd03a1d4d6a4",
    relative_path="model.safetensors",
    size=298_505_628,
    sha256="563431b5f20434a1954ba2998f2010d1e960a672996b4a07f2f32ab694e125ee",
)


def classify_comfy_sdpose_repository_file(
    relative_path: str,
) -> Literal["pose_estimator", "person_detector", "other"]:
    normalized = str(relative_path).replace("\\", "/")
    if normalized == COMFY_SDPOSE_WHOLEBODY_MODEL.relative_path:
        return "pose_estimator"
    if normalized.startswith("diffusion_models/rt_detr_v4-") and normalized.endswith(".safetensors"):
        return "person_detector"
    return "other"


def _verify_file(path: Path, *, size: int, sha256: str, label: str) -> Path:
    actual_size = path.stat().st_size
    if actual_size != size:
        raise PoseArtifactError(f"pose model size mismatch for {label}: expected {size}, got {actual_size}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != sha256:
        raise PoseArtifactError(f"pose model SHA-256 mismatch for {label}: {actual_sha256}")
    return path


def verify_pose_model_file(
    path: str | Path,
    contract: PoseModelContract,
) -> Path:
    candidate = Path(path).expanduser().resolve(strict=True)
    return _verify_file(
        candidate,
        size=contract.size,
        sha256=contract.sha256,
        label=contract.provider_id,
    )


def verify_pose_source_snapshot(
    snapshot: str | Path,
    contract: PoseSourceContract,
) -> Path:
    root = Path(snapshot).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise PoseArtifactError(f"pose source snapshot is not a directory: {root}")
    for artifact in contract.files:
        path = root / Path(*artifact.relative_path.split("/"))
        if not path.is_file():
            raise PoseArtifactError(f"pose source omitted required file: {artifact.relative_path}")
        _verify_file(
            path,
            size=artifact.size,
            sha256=artifact.sha256,
            label=f"{contract.provider_id}/{artifact.relative_path}",
        )
    return root


def _snapshot_downloader(
    downloader: Callable[..., str] | None,
) -> Callable[..., str]:
    if downloader is not None:
        return downloader
    from utils.transformer_loader import snapshot_download_with_reporting

    return snapshot_download_with_reporting


def download_pose_model(
    contract: PoseModelContract,
    *,
    downloader: Callable[..., str] | None = None,
    verify: bool = True,
    console: object | None = None,
) -> Path:
    kwargs: dict[str, object] = {
        "revision": contract.revision,
        "allow_patterns": [contract.relative_path],
    }
    if console is not None:
        kwargs["console"] = console
    snapshot = Path(_snapshot_downloader(downloader)(contract.repo_id, **kwargs))
    model_path = snapshot / Path(*contract.relative_path.split("/"))
    if not model_path.is_file():
        raise PoseArtifactError(f"pose model download omitted required file: {contract.relative_path}")
    return verify_pose_model_file(model_path, contract) if verify else model_path


def download_pose_source(
    contract: PoseSourceContract,
    *,
    downloader: Callable[..., str] | None = None,
    verify: bool = True,
    console: object | None = None,
) -> Path:
    allow_patterns = sorted(item.relative_path for item in contract.files)
    kwargs: dict[str, object] = {
        "revision": contract.revision,
        "allow_patterns": allow_patterns,
    }
    if console is not None:
        kwargs["console"] = console
    snapshot = Path(_snapshot_downloader(downloader)(contract.repo_id, **kwargs))
    missing = [item.relative_path for item in contract.files if not (snapshot / Path(*item.relative_path.split("/"))).is_file()]
    if missing:
        raise PoseArtifactError(f"pose source omitted required file: {missing[0]}")
    return verify_pose_source_snapshot(snapshot, contract) if verify else snapshot


__all__ = [
    "COMFY_SDPOSE_WHOLEBODY_MODEL",
    "DETRPOSE_X_CROWDPOSE_MODEL",
    "SDPOSE_BODY_SOURCE",
    "PoseArtifactError",
    "PoseArtifactFile",
    "PoseModelContract",
    "PoseSourceContract",
    "classify_comfy_sdpose_repository_file",
    "download_pose_model",
    "download_pose_source",
    "verify_pose_model_file",
    "verify_pose_source_snapshot",
]
