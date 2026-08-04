from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .artifacts import (
    DETRPOSE_X_CROWDPOSE_MODEL,
    PoseArtifactError,
    PoseModelContract,
    verify_pose_model_file,
)
from .contracts import RawPoseResult, ScoredKeypoint
from .preprocess import PoseCropTransform, preprocess_detrpose_image

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
DETRPOSE_INFERENCE_CODE_REVISION = "da12e0cd29864b1d3a289c562c34d423371a4157"


@dataclass(frozen=True, slots=True)
class DETRPoseLoadedModel:
    model: Any
    device: torch.device
    weights_sha256: str
    code_revision: str

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.weights_sha256) is None:
            raise ValueError("DETRPose weights SHA-256 must be lowercase hexadecimal")
        if _GIT_REVISION.fullmatch(self.code_revision) is None:
            raise ValueError("DETRPose code revision must be a full Git commit")


@dataclass(frozen=True, slots=True)
class DETRPoseRuntimeReport:
    backend: str
    device: str
    dtype: str
    weights_sha256: str
    code_revision: str
    input_size: tuple[int, int]
    bbox_padding: float
    instance_score_threshold: float


def load_detrpose_state_dict(model: Any, path: str | Path) -> None:
    from safetensors.torch import load_file

    candidate = Path(path).expanduser().resolve(strict=True)
    try:
        state = load_file(str(candidate), device="cpu")
        model.load_state_dict(state, strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise PoseArtifactError("DETRPose checkpoint inventory does not match its pinned architecture") from exc


def _build_detrpose_x_model() -> Any:
    from detrpose import __file__ as detrpose_init
    from detrpose.core import LazyConfig
    from detrpose.core.lazy import _visit_dict_config
    from detrpose.core.utils import _convert_target_to_string
    from detrpose.engine.hf_model import HFModel
    from omegaconf import OmegaConf

    if detrpose_init is None:
        raise PoseArtifactError("DETRPose package location is unavailable")
    config_path = Path(detrpose_init).resolve().parent / "configs" / "detrpose" / "detrpose_hgnetv2_x_crowdpose.py"
    if not config_path.is_file():
        raise PoseArtifactError("pinned DETRPose-X CrowdPose config is unavailable")
    try:
        config = LazyConfig.load(str(config_path))
        config_copy = deepcopy(config)

        def replace_target(value: Any) -> None:
            if "_target_" in value and callable(value._target_):
                value._target_ = _convert_target_to_string(value._target_)

        _visit_dict_config(config_copy, replace_target)
        hf_config = {
            "model": OmegaConf.to_container(config_copy.model, resolve=True),
            "postprocessor": OmegaConf.to_container(
                config_copy.postprocessor,
                resolve=True,
            ),
        }
        return HFModel(hf_config)
    except PoseArtifactError:
        raise
    except Exception as exc:
        raise PoseArtifactError("failed to instantiate the pinned DETRPose-X CrowdPose architecture") from exc


def load_detrpose_x_crowdpose(
    weights_path: str | Path,
    *,
    contract: PoseModelContract = DETRPOSE_X_CROWDPOSE_MODEL,
    device: str | torch.device | None = None,
    verify_weights: bool = True,
) -> DETRPoseLoadedModel:
    path = Path(weights_path).expanduser().resolve(strict=True)
    if verify_weights:
        verify_pose_model_file(path, contract)
    resolved_device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    if resolved_device.type not in {"cpu", "cuda"}:
        raise ValueError("DETRPose v1 supports only CPU and CUDA devices")
    model = _build_detrpose_x_model()
    load_detrpose_state_dict(model, path)
    try:
        model = model.deploy().eval().to(resolved_device)
    except Exception as exc:
        raise PoseArtifactError("failed to deploy the pinned DETRPose model") from exc
    return DETRPoseLoadedModel(
        model=model,
        device=resolved_device,
        weights_sha256=contract.sha256,
        code_revision=DETRPOSE_INFERENCE_CODE_REVISION,
    )


def _validated_predictions(
    scores: np.ndarray,
    keypoints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    score_values = np.asarray(scores, dtype=np.float32)
    point_values = np.asarray(keypoints, dtype=np.float32)
    if score_values.ndim != 1:
        raise ValueError("DETRPose scores must have shape (N,)")
    if point_values.shape != (score_values.shape[0], 14, 2):
        raise ValueError("DETRPose keypoints must have shape (N,14,2)")
    if not len(score_values):
        raise ValueError("DETRPose returned no person predictions")
    if not np.isfinite(score_values).all() or np.any((score_values < 0.0) | (score_values > 1.0)):
        raise ValueError("DETRPose scores must be finite probabilities")
    return score_values, point_values


def select_detrpose_instance(
    scores: np.ndarray,
    keypoints: np.ndarray,
    *,
    person_bbox: tuple[float, float, float, float],
) -> int:
    score_values, point_values = _validated_predictions(scores, keypoints)
    if len(person_bbox) != 4 or not all(math.isfinite(float(value)) for value in person_bbox):
        raise ValueError("DETRPose person bbox must contain four finite values")
    x1, y1, x2, y2 = (float(value) for value in person_bbox)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("DETRPose person bbox must be non-empty")
    target_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
    candidates: list[tuple[float, float, int]] = []
    for index, points in enumerate(point_values):
        finite = np.isfinite(points).all(axis=1)
        if not np.any(finite):
            continue
        center = np.median(points[finite], axis=0)
        distance = float(np.linalg.norm(center - target_center))
        candidates.append((distance, -float(score_values[index]), index))
    if not candidates:
        raise ValueError("DETRPose keypoints contain no finite person")
    return min(candidates)[2]


def build_detrpose_raw_result(
    *,
    scores: np.ndarray,
    keypoints: np.ndarray,
    person_bbox: tuple[float, float, float, float],
    canvas_size: tuple[int, int],
    provider_version: str,
) -> RawPoseResult:
    score_values, point_values = _validated_predictions(scores, keypoints)
    selected = select_detrpose_instance(
        score_values,
        point_values,
        person_bbox=person_bbox,
    )
    canvas_width, canvas_height = canvas_size
    confidence = float(score_values[selected])
    return RawPoseResult(
        provider_id="detrpose-x-crowdpose",
        provider_version=provider_version,
        layout="crowdpose14",
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        keypoints=tuple(
            ScoredKeypoint(
                x=float(point[0]),
                y=float(point[1]),
                score=confidence,
            )
            for point in point_values[selected]
        ),
    )


def _first_batch(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0 or value.shape[0] != 1:
            raise RuntimeError(f"DETRPose {name} must contain one input batch")
        return value[0].detach().to(device="cpu").numpy()
    if not isinstance(value, (list, tuple)) or len(value) != 1:
        raise RuntimeError(f"DETRPose {name} must contain one input batch")
    item = value[0]
    if isinstance(item, torch.Tensor):
        return item.detach().to(device="cpu").numpy()
    return np.asarray(item)


class DETRPoseXProvider:
    def __init__(
        self,
        loaded: DETRPoseLoadedModel,
        *,
        bbox_padding: float = 1.25,
        input_size: tuple[int, int] = (640, 640),
        instance_score_threshold: float = 0.3,
    ) -> None:
        if not math.isfinite(bbox_padding) or bbox_padding < 1.0:
            raise ValueError("DETRPose bbox padding must be finite and >= 1")
        if (
            len(input_size) != 2
            or any(type(value) is not int or value <= 0 for value in input_size)
            or any(value % 32 for value in input_size)
        ):
            raise ValueError("DETRPose input dimensions must be positive multiples of 32")
        if not math.isfinite(instance_score_threshold) or not (0.0 <= instance_score_threshold <= 1.0):
            raise ValueError("DETRPose instance threshold must be in [0,1]")
        self.loaded = loaded
        self.bbox_padding = float(bbox_padding)
        self.input_size = input_size
        self.instance_score_threshold = float(instance_score_threshold)

    @property
    def runtime_report(self) -> DETRPoseRuntimeReport:
        backend = "torch-cuda" if self.loaded.device.type == "cuda" else "torch-cpu"
        return DETRPoseRuntimeReport(
            backend=backend,
            device=str(self.loaded.device),
            dtype="float32",
            weights_sha256=self.loaded.weights_sha256,
            code_revision=self.loaded.code_revision,
            input_size=self.input_size,
            bbox_padding=self.bbox_padding,
            instance_score_threshold=self.instance_score_threshold,
        )

    @torch.inference_mode()
    def infer(
        self,
        image: Image.Image,
        *,
        person_bbox: tuple[float, float, float, float],
    ) -> RawPoseResult:
        transform = PoseCropTransform.from_bbox(
            canvas_size=image.size,
            person_bbox=person_bbox,
            padding=self.bbox_padding,
            input_size=self.input_size,
        )
        input_tensor = preprocess_detrpose_image(image, transform).to(device=self.loaded.device)
        input_width, input_height = self.input_size
        target_size = torch.tensor(
            [[input_width, input_height]],
            device=self.loaded.device,
            dtype=torch.int64,
        )
        wrapper = self.loaded.model
        output = wrapper.model(input_tensor)
        results = wrapper.postprocessor(output, target_size)
        if not isinstance(results, (tuple, list)) or len(results) != 3:
            raise RuntimeError("DETRPose postprocessor returned an invalid result")
        scores = _first_batch(results[0], name="scores").astype(np.float32)
        labels = _first_batch(results[1], name="labels")
        keypoints = _first_batch(results[2], name="keypoints").astype(np.float32)
        if labels.shape != scores.shape:
            raise RuntimeError("DETRPose labels and scores have different shapes")
        keep = scores >= self.instance_score_threshold
        if not bool(np.any(keep)):
            raise RuntimeError("DETRPose returned no person above the confidence gate")
        canvas_keypoints = transform.model_to_canvas(keypoints[keep])
        return build_detrpose_raw_result(
            scores=scores[keep],
            keypoints=canvas_keypoints,
            person_bbox=person_bbox,
            canvas_size=image.size,
            provider_version=("detrpose-x-crowdpose-v1:" + self.loaded.weights_sha256[:12] + ":" + self.loaded.code_revision[:12]),
        )


__all__ = [
    "DETRPOSE_INFERENCE_CODE_REVISION",
    "DETRPoseLoadedModel",
    "DETRPoseRuntimeReport",
    "DETRPoseXProvider",
    "build_detrpose_raw_result",
    "load_detrpose_state_dict",
    "load_detrpose_x_crowdpose",
    "select_detrpose_instance",
]
