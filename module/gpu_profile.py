from __future__ import annotations

import csv
from dataclasses import dataclass
import subprocess
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

    @property
    def primary_device(self) -> GPUDeviceInfo | None:
        if not self.devices:
            return None
        return max(self.devices, key=lambda device: (device.total_vram_bytes, -device.index))

    @property
    def available_vram_bytes(self) -> int:
        primary_device = self.primary_device
        if primary_device is None:
            return 0
        return primary_device.total_vram_bytes

    @property
    def available_vram_gb(self) -> float:
        primary_device = self.primary_device
        if primary_device is None:
            return 0.0
        return primary_device.total_vram_gb


if TYPE_CHECKING:
    from module.see_through.see_through_profile import SeeThroughRecommendation


_NVIDIA_SMI_QUERY = (
    "nvidia-smi",
    "--query-gpu=index,name,memory.total",
    "--format=csv,noheader,nounits",
)

_GPU_PROBE_CACHE: GPUProbeResult | None = None


def _cpu_only_probe() -> GPUProbeResult:
    return GPUProbeResult(
        torch_available=False,
        cuda_available=False,
        cuda_version=None,
        device_count=0,
        current_device_index=None,
        devices=(),
    )


def _build_probe_result(devices: tuple[GPUDeviceInfo, ...]) -> GPUProbeResult:
    return GPUProbeResult(
        torch_available=False,
        cuda_available=bool(devices),
        cuda_version=None,
        device_count=len(devices),
        current_device_index=None,
        devices=devices,
    )


def _parse_nvidia_smi_output(stdout: str) -> tuple[GPUDeviceInfo, ...]:
    devices: list[GPUDeviceInfo] = []
    reader = csv.reader(line for line in stdout.splitlines() if line.strip())
    for row in reader:
        if len(row) < 3:
            continue

        try:
            index = int(str(row[0]).strip())
            total_vram_mib = int(float(str(row[2]).strip()))
        except (TypeError, ValueError):
            continue

        total_vram_bytes = total_vram_mib * 1024**2
        devices.append(
            GPUDeviceInfo(
                index=index,
                name=str(row[1]).strip() or f"GPU {index}",
                capability=None,
                capability_label="unknown",
                sm=None,
                total_vram_bytes=total_vram_bytes,
                total_vram_gb=round(total_vram_bytes / 1024**3, 2),
                bf16_supported=False,
            )
        )

    devices.sort(key=lambda device: device.index)
    return tuple(devices)


def probe_gpu_environment(
    *,
    torch_module: Any | None = None,
    python_executable: str | None = None,
    env: dict[str, str] | None = None,
) -> GPUProbeResult:
    del torch_module, python_executable

    try:
        result = subprocess.run(
            list(_NVIDIA_SMI_QUERY),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            check=False,
            timeout=5,
        )
    except Exception:
        return _cpu_only_probe()

    stdout = (result.stdout or "").strip()
    if result.returncode != 0 or not stdout:
        return _cpu_only_probe()

    devices = _parse_nvidia_smi_output(stdout)
    if not devices:
        return _cpu_only_probe()
    return _build_probe_result(devices)


def clear_gpu_probe_cache() -> None:
    global _GPU_PROBE_CACHE
    _GPU_PROBE_CACHE = None


def get_cached_gpu_probe(*, refresh: bool = False) -> GPUProbeResult:
    global _GPU_PROBE_CACHE

    if refresh:
        _GPU_PROBE_CACHE = None

    if _GPU_PROBE_CACHE is None:
        _GPU_PROBE_CACHE = probe_gpu_environment()

    return _GPU_PROBE_CACHE


def recommend_see_through_config(probe: GPUProbeResult) -> "SeeThroughRecommendation":
    from module.see_through.see_through_profile import recommend_see_through_config as _recommend_see_through_config

    return _recommend_see_through_config(probe)


def format_gpu_device_lines(probe: GPUProbeResult) -> tuple[str, ...]:
    if not probe.devices:
        return ()

    lines: list[str] = []
    for device in probe.devices:
        parts = [f"GPU {device.index}", device.name]
        if device.sm:
            parts.append(device.sm)
        parts.append(f"{device.total_vram_gb:.1f} GB")
        lines.append(" | ".join(parts))
    return tuple(lines)


def format_gpu_summary(probe: GPUProbeResult) -> str:
    primary_device = probe.primary_device
    if primary_device is None:
        return "CPU only"

    parts = [primary_device.name]
    if primary_device.sm:
        parts.append(primary_device.sm)
    parts.append(f"{primary_device.total_vram_gb:.1f} GB")
    if probe.device_count > 1:
        parts.append(f"{probe.device_count} GPUs")
    return " | ".join(parts)
