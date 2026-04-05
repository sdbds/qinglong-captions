from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING, Any


@dataclass(frozen=True)
class GPUDeviceInfo:
    index: int
    name: str
    capability: tuple[int, int] | None
    capability_label: str
    sm: str | None
    total_vram_bytes: int
    total_vram_gb: float
    bf16_supported: bool


@dataclass(frozen=True)
class GPUProbeResult:
    torch_available: bool
    cuda_available: bool
    cuda_version: str | None
    device_count: int
    current_device_index: int | None
    devices: tuple[GPUDeviceInfo, ...]
    tier: str
    tier_label: str

    @property
    def primary_device(self) -> GPUDeviceInfo | None:
        if not self.devices:
            return None
        if self.current_device_index is None:
            return self.devices[0]
        for device in self.devices:
            if device.index == self.current_device_index:
                return device
        return self.devices[0]


if TYPE_CHECKING:
    from module.see_through.see_through_profile import SeeThroughRecommendation


def classify_vram_tier(total_vram_bytes: int | None, *, cuda_available: bool) -> str:
    if not cuda_available or not total_vram_bytes or total_vram_bytes <= 0:
        return "cpu_only"

    total_vram_gb = total_vram_bytes / 1024**3
    if total_vram_gb < 8:
        return "lt8gb"
    if total_vram_gb < 12:
        return "8_to_12gb"
    if total_vram_gb <= 16:
        return "12_to_16gb"
    return "gt16gb"


def tier_label(tier: str) -> str:
    return {
        "cpu_only": "CPU only",
        "lt8gb": "<8 GB",
        "8_to_12gb": "8-12 GB",
        "12_to_16gb": "12-16 GB",
        "gt16gb": ">16 GB",
    }.get(tier, tier)


def _cuda_device_label(index: int) -> str:
    return "cuda" if int(index) <= 0 else f"cuda:{int(index)}"


_GPU_PROBE_SCRIPT = textwrap.dedent(
    """
    import json

    def main():
        try:
            import torch
        except Exception as exc:
            print(json.dumps({
                "torch_available": False,
                "cuda_available": False,
                "cuda_version": None,
                "device_count": 0,
                "current_device_index": None,
                "devices": [],
                "error": f"{type(exc).__name__}: {exc}",
            }))
            return

        cuda_module = getattr(torch, "cuda", None)
        if cuda_module is None or not hasattr(cuda_module, "is_available") or not cuda_module.is_available():
            print(json.dumps({
                "torch_available": True,
                "cuda_available": False,
                "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
                "device_count": 0,
                "current_device_index": None,
                "devices": [],
            }))
            return

        try:
            device_count = int(cuda_module.device_count())
        except Exception:
            device_count = 0

        try:
            current_device_index = int(cuda_module.current_device()) if device_count > 0 else None
        except Exception:
            current_device_index = 0 if device_count > 0 else None

        bf16_supported = False
        if hasattr(cuda_module, "is_bf16_supported"):
            try:
                bf16_supported = bool(cuda_module.is_bf16_supported())
            except Exception:
                bf16_supported = False

        devices = []
        for index in range(device_count):
            try:
                props = cuda_module.get_device_properties(index)
            except Exception:
                continue

            major = getattr(props, "major", None)
            minor = getattr(props, "minor", None)
            capability = None
            if major is not None and minor is not None:
                capability = [int(major), int(minor)]

            total_vram_bytes = int(getattr(props, "total_memory", 0) or 0)
            devices.append({
                "index": index,
                "name": str(cuda_module.get_device_name(index)),
                "capability": capability,
                "capability_label": f"{capability[0]}.{capability[1]}" if capability else "unknown",
                "sm": f"sm{capability[0]}{capability[1]}" if capability else None,
                "total_vram_bytes": total_vram_bytes,
                "total_vram_gb": round(total_vram_bytes / 1024**3, 2),
                "bf16_supported": bool(bf16_supported if current_device_index == index else False),
            })

        print(json.dumps({
            "torch_available": True,
            "cuda_available": bool(devices),
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
            "device_count": len(devices),
            "current_device_index": current_device_index if devices else None,
            "devices": devices,
        }))

    if __name__ == "__main__":
        main()
    """
).strip()


def _probe_result_from_payload(payload: dict[str, Any]) -> GPUProbeResult:
    devices = tuple(
        GPUDeviceInfo(
            index=int(device["index"]),
            name=str(device["name"]),
            capability=tuple(device["capability"]) if device.get("capability") is not None else None,
            capability_label=str(device.get("capability_label") or "unknown"),
            sm=device.get("sm"),
            total_vram_bytes=int(device.get("total_vram_bytes", 0) or 0),
            total_vram_gb=float(device.get("total_vram_gb", 0.0) or 0.0),
            bf16_supported=bool(device.get("bf16_supported", False)),
        )
        for device in payload.get("devices", [])
    )
    probe = GPUProbeResult(
        torch_available=bool(payload.get("torch_available", False)),
        cuda_available=bool(payload.get("cuda_available", False)),
        cuda_version=payload.get("cuda_version"),
        device_count=int(payload.get("device_count", len(devices)) or 0),
        current_device_index=payload.get("current_device_index"),
        devices=devices,
        tier="cpu_only",
        tier_label=tier_label("cpu_only"),
    )
    primary_device = probe.primary_device
    tier = classify_vram_tier(
        primary_device.total_vram_bytes if primary_device is not None else None,
        cuda_available=probe.cuda_available,
    )
    return GPUProbeResult(
        torch_available=probe.torch_available,
        cuda_available=probe.cuda_available,
        cuda_version=probe.cuda_version,
        device_count=probe.device_count,
        current_device_index=probe.current_device_index,
        devices=probe.devices,
        tier=tier,
        tier_label=tier_label(tier),
    )


def _probe_gpu_environment_subprocess() -> GPUProbeResult:
    try:
        result = subprocess.run(
            [sys.executable, "-c", _GPU_PROBE_SCRIPT],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=20,
        )
    except Exception:
        return GPUProbeResult(
            torch_available=False,
            cuda_available=False,
            cuda_version=None,
            device_count=0,
            current_device_index=None,
            devices=(),
            tier="cpu_only",
            tier_label=tier_label("cpu_only"),
        )

    stdout = (result.stdout or "").strip()
    if result.returncode != 0 or not stdout:
        return GPUProbeResult(
            torch_available=False,
            cuda_available=False,
            cuda_version=None,
            device_count=0,
            current_device_index=None,
            devices=(),
            tier="cpu_only",
            tier_label=tier_label("cpu_only"),
        )

    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError:
        return GPUProbeResult(
            torch_available=False,
            cuda_available=False,
            cuda_version=None,
            device_count=0,
            current_device_index=None,
            devices=(),
            tier="cpu_only",
            tier_label=tier_label("cpu_only"),
        )

    return _probe_result_from_payload(payload)


def probe_gpu_environment(*, torch_module: Any | None = None) -> GPUProbeResult:
    if torch_module is None:
        return _probe_gpu_environment_subprocess()

    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is None or not hasattr(cuda_module, "is_available") or not cuda_module.is_available():
        return GPUProbeResult(
            torch_available=True,
            cuda_available=False,
            cuda_version=getattr(getattr(torch_module, "version", None), "cuda", None),
            device_count=0,
            current_device_index=None,
            devices=(),
            tier="cpu_only",
            tier_label=tier_label("cpu_only"),
        )

    try:
        device_count = int(cuda_module.device_count())
    except Exception:
        device_count = 0

    try:
        current_device_index = int(cuda_module.current_device()) if device_count > 0 else None
    except Exception:
        current_device_index = 0 if device_count > 0 else None

    bf16_supported = False
    if hasattr(cuda_module, "is_bf16_supported"):
        try:
            bf16_supported = bool(cuda_module.is_bf16_supported())
        except Exception:
            bf16_supported = False

    devices: list[GPUDeviceInfo] = []
    for index in range(device_count):
        try:
            props = cuda_module.get_device_properties(index)
        except Exception:
            continue

        capability: tuple[int, int] | None = None
        major = getattr(props, "major", None)
        minor = getattr(props, "minor", None)
        if major is not None and minor is not None:
            capability = (int(major), int(minor))

        capability_label = f"{capability[0]}.{capability[1]}" if capability else "unknown"
        sm = f"sm{capability[0]}{capability[1]}" if capability else None
        total_vram_bytes = int(getattr(props, "total_memory", 0) or 0)
        device_bf16_supported = bf16_supported if current_device_index == index else False

        devices.append(
            GPUDeviceInfo(
                index=index,
                name=str(cuda_module.get_device_name(index)),
                capability=capability,
                capability_label=capability_label,
                sm=sm,
                total_vram_bytes=total_vram_bytes,
                total_vram_gb=round(total_vram_bytes / 1024**3, 2),
                bf16_supported=device_bf16_supported,
            )
        )

    return _probe_result_from_payload(
        {
            "torch_available": True,
            "cuda_available": bool(devices),
            "cuda_version": getattr(getattr(torch_module, "version", None), "cuda", None),
            "device_count": len(devices),
            "current_device_index": current_device_index if devices else None,
            "devices": [
                {
                    "index": device.index,
                    "name": device.name,
                    "capability": list(device.capability) if device.capability is not None else None,
                    "capability_label": device.capability_label,
                    "sm": device.sm,
                    "total_vram_bytes": device.total_vram_bytes,
                    "total_vram_gb": device.total_vram_gb,
                    "bf16_supported": device.bf16_supported,
                }
                for device in devices
            ],
        }
    )


@lru_cache(maxsize=1)
def get_cached_gpu_probe() -> GPUProbeResult:
    return probe_gpu_environment()


def recommend_see_through_config(probe: GPUProbeResult) -> "SeeThroughRecommendation":
    from module.see_through.see_through_profile import recommend_see_through_config as _recommend_see_through_config

    return _recommend_see_through_config(probe)


def format_gpu_device_lines(probe: GPUProbeResult) -> tuple[str, ...]:
    if not probe.devices:
        return ()

    lines: list[str] = []
    for device in probe.devices:
        parts = [_cuda_device_label(device.index), device.name]
        if device.sm:
            parts.append(device.sm)
        parts.append(f"{device.total_vram_gb:.1f} GB")
        if probe.cuda_version:
            parts.append(f"CUDA {probe.cuda_version}")
        if probe.current_device_index == device.index and probe.device_count > 1:
            parts.append("active")
        lines.append(" | ".join(parts))
    return tuple(lines)


def format_gpu_summary(probe: GPUProbeResult) -> str:
    primary_device = probe.primary_device
    if primary_device is None:
        return "CPU only"

    parts: list[str] = []
    if probe.current_device_index is not None:
        parts.append(_cuda_device_label(probe.current_device_index))
    parts.append(primary_device.name)
    if primary_device.sm:
        parts.append(primary_device.sm)
    parts.append(f"{primary_device.total_vram_gb:.1f} GB")
    if probe.cuda_version:
        parts.append(f"CUDA {probe.cuda_version}")
    if probe.device_count > 1:
        parts.append(f"{probe.device_count} GPUs")
    return " | ".join(parts)
