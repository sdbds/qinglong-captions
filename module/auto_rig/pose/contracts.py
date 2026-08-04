from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

PoseLayout = Literal["coco17", "crowdpose14"]
PoseMode = Literal["disabled", "auto", "sdpose", "detrpose", "compare"]


@dataclass(frozen=True, slots=True)
class ScoredKeypoint:
    x: float
    y: float
    score: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(float(value)) for value in (self.x, self.y, self.score)):
            raise ValueError("pose keypoint values must be finite")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("pose keypoint score must be in [0,1]")


@dataclass(frozen=True, slots=True)
class RawPoseResult:
    provider_id: str
    provider_version: str
    layout: PoseLayout
    canvas_width: int
    canvas_height: int
    keypoints: tuple[ScoredKeypoint, ...]

    def __post_init__(self) -> None:
        expected = {"coco17": 17, "crowdpose14": 14}.get(self.layout)
        if expected is None or len(self.keypoints) != expected:
            raise ValueError("pose keypoint count differs from its declared layout")
        if not self.provider_id or not self.provider_version:
            raise ValueError("pose provider identity must be non-empty")
        if self.canvas_width <= 0 or self.canvas_height <= 0:
            raise ValueError("pose canvas must be positive")


__all__ = ["PoseLayout", "PoseMode", "RawPoseResult", "ScoredKeypoint"]
