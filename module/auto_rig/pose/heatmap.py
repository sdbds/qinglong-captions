from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SDPoseBodyHeatmapHead(nn.Module):
    """The exact learned-module layout used by the official Body17 decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(
                320,
                320,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False,
            ),
            nn.InstanceNorm2d(320),
            nn.SiLU(inplace=True),
        )
        self.conv_layers = nn.Sequential(
            nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(320),
            nn.SiLU(inplace=True),
        )
        self.final_layer = nn.Conv2d(320, 17, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.final_layer(self.conv_layers(self.deconv_layers(features)))


def _gaussian_blur_preserving_maximum(
    heatmaps: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    import cv2

    if kernel_size <= 0 or kernel_size % 2 != 1:
        raise ValueError("blur kernel size must be a positive odd integer")
    border = (kernel_size - 1) // 2
    blurred = np.empty_like(heatmaps, dtype=np.float32)
    for index, heatmap in enumerate(heatmaps):
        original_maximum = float(np.max(heatmap))
        if border:
            padded = np.zeros(
                (heatmap.shape[0] + 2 * border, heatmap.shape[1] + 2 * border),
                dtype=np.float32,
            )
            padded[border:-border, border:-border] = heatmap
            filtered = cv2.GaussianBlur(
                padded,
                (kernel_size, kernel_size),
                0,
            )[border:-border, border:-border]
        else:
            filtered = heatmap.copy()
        filtered_maximum = float(np.max(filtered))
        if original_maximum > 0.0 and filtered_maximum > 0.0:
            filtered *= original_maximum / filtered_maximum
        blurred[index] = filtered
    return blurred


def _dark_udp_refine(
    keypoints: np.ndarray,
    heatmaps: np.ndarray,
    *,
    blur_kernel_size: int,
) -> np.ndarray:
    keypoint_count, height, width = heatmaps.shape
    log_heatmaps = _gaussian_blur_preserving_maximum(
        heatmaps,
        blur_kernel_size,
    )
    np.clip(log_heatmaps, 1e-3, 50.0, out=log_heatmaps)
    np.log(log_heatmaps, out=log_heatmaps)
    padded = np.pad(log_heatmaps, ((0, 0), (1, 1), (1, 1)), mode="edge")

    refined = keypoints.copy()
    epsilon = np.finfo(np.float32).eps
    for index in range(keypoint_count):
        x, y = refined[index].astype(np.int64)
        if x < 0 or y < 0:
            continue
        x += 1
        y += 1
        center = padded[index, y, x]
        right = padded[index, y, x + 1]
        left = padded[index, y, x - 1]
        down = padded[index, y + 1, x]
        up = padded[index, y - 1, x]
        down_right = padded[index, y + 1, x + 1]
        up_left = padded[index, y - 1, x - 1]

        derivative = np.array(
            [0.5 * (right - left), 0.5 * (down - up)],
            dtype=np.float64,
        )
        hessian = np.array(
            [
                [
                    right - 2.0 * center + left,
                    0.5 * (down_right - right - down + 2.0 * center - left - up + up_left),
                ],
                [
                    0.5 * (down_right - right - down + 2.0 * center - left - up + up_left),
                    down - 2.0 * center + up,
                ],
            ],
            dtype=np.float64,
        )
        offset = np.linalg.solve(hessian + epsilon * np.eye(2), derivative)
        refined[index] -= offset.astype(np.float32)
    return refined


def decode_udp_heatmaps(
    heatmaps: np.ndarray | torch.Tensor,
    *,
    input_size: tuple[int, int] = (768, 1024),
    blur_kernel_size: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().float().cpu().numpy()
    values = np.asarray(heatmaps, dtype=np.float32)
    if values.ndim != 3 or not values.shape[0] or min(values.shape[1:]) < 2:
        raise ValueError("heatmaps must have shape (K,H,W) with H,W >= 2")
    if len(input_size) != 2 or min(input_size) <= 0:
        raise ValueError("input size must be positive (width,height)")
    keypoint_count, height, width = values.shape
    flattened = values.reshape(keypoint_count, -1)
    flat_indices = np.argmax(flattened, axis=1)
    scores = np.max(flattened, axis=1).astype(np.float32)
    keypoints = np.stack(
        (flat_indices % width, flat_indices // width),
        axis=-1,
    ).astype(np.float32)
    keypoints[scores <= 0.0] = -1.0
    keypoints = _dark_udp_refine(
        keypoints,
        values.copy(),
        blur_kernel_size=blur_kernel_size,
    )
    keypoints *= np.array(
        [input_size[0] / (width - 1), input_size[1] / (height - 1)],
        dtype=np.float32,
    )
    return keypoints, scores


__all__ = ["SDPoseBodyHeatmapHead", "decode_udp_heatmaps"]
