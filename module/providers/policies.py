"""Shared provider policy value objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class SegmentationPolicy:
    segment_time: Any = None
    bypass_segmentation: bool = False
    direct_duration_limit_ms: Optional[int] = None
