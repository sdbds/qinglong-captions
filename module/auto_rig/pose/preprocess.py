from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True, slots=True)
class PoseCropTransform:
    canvas_width: int
    canvas_height: int
    crop_x1: int
    crop_y1: int
    crop_x2: int
    crop_y2: int
    input_width: int = 768
    input_height: int = 1024

    def __post_init__(self) -> None:
        if (
            min(
                self.canvas_width,
                self.canvas_height,
                self.input_width,
                self.input_height,
            )
            <= 0
        ):
            raise ValueError("pose crop dimensions must be positive")
        if not (0 <= self.crop_x1 < self.crop_x2 <= self.canvas_width and 0 <= self.crop_y1 < self.crop_y2 <= self.canvas_height):
            raise ValueError("pose crop box must be a non-empty canvas subregion")

    @property
    def crop_box(self) -> tuple[int, int, int, int]:
        return self.crop_x1, self.crop_y1, self.crop_x2, self.crop_y2

    @property
    def scale_x(self) -> float:
        return self.input_width / (self.crop_x2 - self.crop_x1)

    @property
    def scale_y(self) -> float:
        return self.input_height / (self.crop_y2 - self.crop_y1)

    @classmethod
    def from_bbox(
        cls,
        *,
        canvas_size: tuple[int, int],
        person_bbox: tuple[float, float, float, float],
        padding: float = 1.25,
        input_size: tuple[int, int] = (768, 1024),
    ) -> PoseCropTransform:
        canvas_width, canvas_height = canvas_size
        input_width, input_height = input_size
        if len(person_bbox) != 4 or not all(math.isfinite(float(value)) for value in person_bbox):
            raise ValueError("person bbox must contain four finite coordinates")
        if padding < 1.0 or not math.isfinite(float(padding)):
            raise ValueError("pose bbox padding must be finite and >= 1")
        x1, y1, x2, y2 = (float(value) for value in person_bbox)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("person bbox must have positive width and height")
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        half_width = (x2 - x1) * padding / 2.0
        half_height = (y2 - y1) * padding / 2.0
        crop_x1 = max(0, math.floor(center_x - half_width))
        crop_y1 = max(0, math.floor(center_y - half_height))
        crop_x2 = min(canvas_width, math.ceil(center_x + half_width))
        crop_y2 = min(canvas_height, math.ceil(center_y + half_height))
        return cls(
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            crop_x1=crop_x1,
            crop_y1=crop_y1,
            crop_x2=crop_x2,
            crop_y2=crop_y2,
            input_width=input_width,
            input_height=input_height,
        )

    def canvas_to_model(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float32)
        if values.shape[-1:] != (2,):
            raise ValueError("pose points must end with an xy dimension")
        result = values.copy()
        result[..., 0] = (result[..., 0] - self.crop_x1) * self.scale_x
        result[..., 1] = (result[..., 1] - self.crop_y1) * self.scale_y
        return result

    def model_to_canvas(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float32)
        if values.shape[-1:] != (2,):
            raise ValueError("pose points must end with an xy dimension")
        result = values.copy()
        result[..., 0] = result[..., 0] / self.scale_x + self.crop_x1
        result[..., 1] = result[..., 1] / self.scale_y + self.crop_y1
        return result


def _rgb_on_white(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def preprocess_sdpose_image(
    image: Image.Image,
    transform: PoseCropTransform,
) -> torch.Tensor:
    if image.size != (transform.canvas_width, transform.canvas_height):
        raise ValueError("pose image size differs from the crop transform canvas")
    cropped = _rgb_on_white(image).crop(transform.crop_box)
    resized = cropped.resize(
        (transform.input_width, transform.input_height),
        Image.Resampling.BILINEAR,
    )
    pixels = np.asarray(resized, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).contiguous()


def preprocess_detrpose_image(
    image: Image.Image,
    transform: PoseCropTransform,
) -> torch.Tensor:
    if image.size != (transform.canvas_width, transform.canvas_height):
        raise ValueError("pose image size differs from the crop transform canvas")
    cropped = _rgb_on_white(image).crop(transform.crop_box)
    resized = cropped.resize(
        (transform.input_width, transform.input_height),
        Image.Resampling.BILINEAR,
    )
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).contiguous()


__all__ = [
    "PoseCropTransform",
    "preprocess_detrpose_image",
    "preprocess_sdpose_image",
]
