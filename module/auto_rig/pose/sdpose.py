from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image

from .attention import FlashAttentionBackendError
from .contracts import RawPoseResult, ScoredKeypoint
from .heatmap import decode_udp_heatmaps
from .preprocess import PoseCropTransform, preprocess_sdpose_image


@dataclass(frozen=True, slots=True)
class SDPoseLoadedModels:
    vae: Any
    unet: Any
    decoder: Any
    empty_prompt: torch.Tensor
    device: torch.device
    dtype: torch.dtype
    backend: Literal["torch-fa2", "torch-sdpa", "torch-cpu"]
    bundle_sha256: str
    flash_processor: Any | None = None
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if tuple(self.empty_prompt.shape) != (1, 2, 1024):
            raise ValueError("SDPose empty prompt must have shape (1,2,1024)")
        if len(self.bundle_sha256) != 64:
            raise ValueError("SDPose bundle SHA-256 must contain 64 hex characters")


@dataclass(frozen=True, slots=True)
class SDPoseRuntimeReport:
    backend: str
    device: str
    dtype: str
    bundle_sha256: str
    latent_seed: int
    input_size: tuple[int, int]
    bbox_padding: float
    fallback_reason: str | None


def sdpose_task_embedding(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("SDPose task embedding batch size must be positive")
    label = torch.tensor([[1.0, 0.0]], device=device, dtype=torch.float32)
    embedding = torch.cat((torch.sin(label), torch.cos(label)), dim=-1)
    return embedding.to(dtype=dtype).repeat(batch_size, 1)


def _capture_final_up_block_features(
    unet: Any,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    up_blocks = getattr(unet, "up_blocks", None)
    if up_blocks is None or not len(up_blocks):
        raise RuntimeError("SDPose UNet exposes no up blocks")
    captured: list[torch.Tensor] = []

    def capture(_: Any, __: Any, output: Any) -> None:
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("SDPose final up block returned a non-tensor")
        captured.append(value)

    handle = up_blocks[-1].register_forward_hook(capture)
    try:
        unet(*args, **kwargs)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("SDPose final up-block feature capture was not unique")
    return captured[0]


class SDPoseBodyProvider:
    def __init__(
        self,
        models: SDPoseLoadedModels,
        *,
        latent_seed: int = 0,
        bbox_padding: float = 1.25,
    ) -> None:
        if latent_seed < 0:
            raise ValueError("SDPose latent seed must be non-negative")
        if not math.isfinite(bbox_padding) or bbox_padding < 1.0:
            raise ValueError("SDPose bbox padding must be finite and >= 1")
        self.models = models
        self.latent_seed = latent_seed
        self.bbox_padding = bbox_padding
        self._backend = models.backend
        self._fallback_reason = models.fallback_reason

    @property
    def runtime_report(self) -> SDPoseRuntimeReport:
        return SDPoseRuntimeReport(
            backend=self._backend,
            device=str(self.models.device),
            dtype=str(self.models.dtype).removeprefix("torch."),
            bundle_sha256=self.models.bundle_sha256,
            latent_seed=self.latent_seed,
            input_size=(768, 1024),
            bbox_padding=self.bbox_padding,
            fallback_reason=self._fallback_reason,
        )

    def _fall_back_to_sdpa(self, reason: str) -> None:
        from diffusers.models.attention_processor import AttnProcessor2_0

        self.models.unet.set_attn_processor(AttnProcessor2_0())
        self._backend = "torch-sdpa"
        self._fallback_reason = reason

    @torch.inference_mode()
    def _infer_heatmaps(self, image_tensor: torch.Tensor) -> torch.Tensor:
        image_tensor = image_tensor.to(
            device=self.models.device,
            dtype=self.models.dtype,
        )
        posterior = self.models.vae.encode(image_tensor).latent_dist
        generator = torch.Generator(device=self.models.device).manual_seed(self.latent_seed)
        latent = posterior.sample(generator=generator) * 0.18215
        batch_size = latent.shape[0]
        context = self.models.empty_prompt.to(
            device=self.models.device,
            dtype=self.models.dtype,
        ).repeat(batch_size, 1, 1)
        class_labels = sdpose_task_embedding(
            batch_size,
            device=self.models.device,
            dtype=self.models.dtype,
        )
        timestep = torch.tensor(999, device=self.models.device, dtype=torch.long)
        unet_kwargs = {
            "encoder_hidden_states": context,
            "class_labels": class_labels,
            "return_dict": False,
        }
        try:
            features = _capture_final_up_block_features(
                self.models.unet,
                latent,
                timestep,
                **unet_kwargs,
            )
            processor = self.models.flash_processor
            if self._backend == "torch-fa2" and (
                processor is None or not processor.saw_self_attention or not processor.saw_cross_attention
            ):
                raise FlashAttentionBackendError("SDPose FA2 forward did not exercise self and cross attention")
        except FlashAttentionBackendError as exc:
            if self._backend != "torch-fa2":
                raise
            self._fall_back_to_sdpa(str(exc))
            features = _capture_final_up_block_features(
                self.models.unet,
                latent,
                timestep,
                **unet_kwargs,
            )
        heatmaps = self.models.decoder(features)
        if tuple(heatmaps.shape[:2]) != (batch_size, 17) or heatmaps.ndim != 4:
            raise RuntimeError("SDPose Body decoder returned the wrong heatmap shape")
        if not bool(torch.isfinite(heatmaps).all()):
            raise RuntimeError("SDPose Body decoder returned non-finite heatmaps")
        return heatmaps

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
        )
        input_tensor = preprocess_sdpose_image(image, transform)
        heatmaps = self._infer_heatmaps(input_tensor)
        model_keypoints, scores = decode_udp_heatmaps(heatmaps[0])
        canvas_keypoints = transform.model_to_canvas(model_keypoints)
        keypoints = tuple(
            ScoredKeypoint(
                x=float(point[0]),
                y=float(point[1]),
                score=float(np.clip(score, 0.0, 1.0)),
            )
            for point, score in zip(canvas_keypoints, scores, strict=True)
        )
        return RawPoseResult(
            provider_id="sdpose-body17",
            provider_version=("sdpose-body17-bundle-v1:" + self.models.bundle_sha256[:12]),
            layout="coco17",
            canvas_width=image.width,
            canvas_height=image.height,
            keypoints=keypoints,
        )


__all__ = [
    "SDPoseBodyProvider",
    "SDPoseLoadedModels",
    "SDPoseRuntimeReport",
    "sdpose_task_embedding",
]
