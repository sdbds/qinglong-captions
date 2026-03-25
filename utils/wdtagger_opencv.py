from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Mapping

DEFAULT_OPENCV_REQUIREMENT = "opencv-contrib-python"
CUDA_OPENCV_REQUIREMENTS = {
    "cu128": "opencv-contrib-python @ https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.11.0.20250124/opencv_contrib_python_rolling-4.12.0.86-cp37-abi3-win_amd64.whl",
    "cu129": "opencv-contrib-python @ https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.12.0.88/opencv_contrib_python-4.12.0.88-cp37-abi3-win_amd64.whl",
    "cu130": "opencv-contrib-python @ https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.13.0.20250811/opencv_contrib_python_rolling-4.13.0.20250812-cp37-abi3-win_amd64.whl",
}
_NVCC_RELEASE_RE = re.compile(r"release\s+(\d+)\.(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class WdtaggerOpenCvSelection:
    package_spec: str
    cuda_tag: str | None
    source: str
    detail: str


def parse_nvcc_cuda_version_tag(output: str) -> str | None:
    match = _NVCC_RELEASE_RE.search(output or "")
    if not match:
        return None
    major, minor = match.groups()
    return f"cu{major}{minor}"


def detect_nvcc_cuda_version_tag(env: Mapping[str, str] | None = None) -> str | None:
    search_path = None if env is None else env.get("PATH")
    nvcc_path = shutil.which("nvcc", path=search_path)
    if not nvcc_path:
        return None

    try:
        result = subprocess.run(
            [nvcc_path, "-V"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=None if env is None else dict(env),
            check=False,
        )
    except OSError:
        return None

    combined_output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    return parse_nvcc_cuda_version_tag(combined_output)


def resolve_wdtagger_windows_opencv_requirement(
    env: Mapping[str, str] | None = None,
    *,
    platform: str | None = None,
) -> WdtaggerOpenCvSelection:
    normalized_platform = (platform or sys.platform).lower()
    if normalized_platform != "win32":
        return WdtaggerOpenCvSelection(
            package_spec=DEFAULT_OPENCV_REQUIREMENT,
            cuda_tag=None,
            source="default",
            detail="non-Windows platform uses default opencv-contrib-python",
        )

    cuda_tag = detect_nvcc_cuda_version_tag(env)
    wheel_spec = CUDA_OPENCV_REQUIREMENTS.get(cuda_tag or "")
    if wheel_spec:
        return WdtaggerOpenCvSelection(
            package_spec=wheel_spec,
            cuda_tag=cuda_tag,
            source="cuda-wheel",
            detail=f"detected CUDA toolkit {cuda_tag}",
        )

    if cuda_tag:
        detail = f"unsupported CUDA toolkit {cuda_tag}; fallback to default opencv-contrib-python"
    else:
        detail = "nvcc not detected; fallback to default opencv-contrib-python"

    return WdtaggerOpenCvSelection(
        package_spec=DEFAULT_OPENCV_REQUIREMENT,
        cuda_tag=cuda_tag,
        source="default",
        detail=detail,
    )
