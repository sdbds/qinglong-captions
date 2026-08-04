from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class PoseOnnxContract:
    provider_id: str
    graph_sha256: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]

    def is_complete(self) -> bool:
        return bool(
            self.provider_id
            and _SHA256.fullmatch(self.graph_sha256)
            and self.input_names
            and self.output_names
            and len(self.input_names) == len(set(self.input_names))
            and len(self.output_names) == len(set(self.output_names))
        )


@dataclass(frozen=True, slots=True)
class PoseRuntimeSelection:
    backend: Literal["onnx", "torch-fa2", "torch-sdpa", "torch-cpu"]
    fallback_reason: str | None


def select_pose_runtime(
    *,
    onnx_path: str | Path | None,
    onnx_contract: PoseOnnxContract | None,
    cuda_available: bool,
    flash_attn_available: bool,
) -> PoseRuntimeSelection:
    if onnx_path is not None and Path(onnx_path).is_file() and onnx_contract is not None and onnx_contract.is_complete():
        return PoseRuntimeSelection("onnx", None)
    if cuda_available and flash_attn_available:
        reason = "onnx_contract_unavailable" if onnx_path is not None else None
        return PoseRuntimeSelection("torch-fa2", reason)
    if cuda_available:
        return PoseRuntimeSelection("torch-sdpa", "flash_attn_unavailable")
    return PoseRuntimeSelection("torch-cpu", "cuda_unavailable")


__all__ = ["PoseOnnxContract", "PoseRuntimeSelection", "select_pose_runtime"]
