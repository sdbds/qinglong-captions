from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Mapping

DEFAULT_OPENCV_REQUIREMENT = "opencv-contrib-python"
DEFAULT_OPENCV_PACKAGE_NAME = "opencv-contrib-python"
CUDA_OPENCV_REQUIREMENTS = {
    "cu128": "opencv-contrib-python-rolling @ https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.11.0.20250124/opencv_contrib_python_rolling-4.12.0.86-cp37-abi3-win_amd64.whl",
    "cu129": "opencv-contrib-python @ https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.12.0.88/opencv_contrib_python-4.12.0.88-cp37-abi3-win_amd64.whl",
    "cu130": "opencv-contrib-python-rolling @ https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.13.0.20250811/opencv_contrib_python_rolling-4.13.0.20250812-cp37-abi3-win_amd64.whl",
}
OPENCV_DISTRIBUTION_PACKAGES = (
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
    "opencv-contrib-python-rolling",
)
_NVCC_RELEASE_RE = re.compile(r"release\s+(\d+)\.(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class WdtaggerOpenCvSelection:
    package_name: str
    package_spec: str
    cuda_tag: str | None
    source: str
    detail: str


@dataclass(frozen=True)
class WdtaggerOpenCvInstallPlan:
    cleanup_packages: tuple[str, ...]
    attempts: tuple[WdtaggerOpenCvSelection, ...]


def _package_name_from_requirement(requirement: str) -> str:
    return requirement.split("@", 1)[0].strip()


def build_default_opencv_selection(detail: str, *, source: str = "default") -> WdtaggerOpenCvSelection:
    return WdtaggerOpenCvSelection(
        package_name=DEFAULT_OPENCV_PACKAGE_NAME,
        package_spec=DEFAULT_OPENCV_REQUIREMENT,
        cuda_tag=None,
        source=source,
        detail=detail,
    )


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
        return build_default_opencv_selection(
            "non-Windows platform uses default opencv-contrib-python",
        )

    cuda_tag = detect_nvcc_cuda_version_tag(env)
    wheel_spec = CUDA_OPENCV_REQUIREMENTS.get(cuda_tag or "")
    if wheel_spec:
        return WdtaggerOpenCvSelection(
            package_name=_package_name_from_requirement(wheel_spec),
            package_spec=wheel_spec,
            cuda_tag=cuda_tag,
            source="cuda-wheel",
            detail=f"detected CUDA toolkit {cuda_tag}",
        )

    if cuda_tag:
        detail = f"unsupported CUDA toolkit {cuda_tag}; fallback to default opencv-contrib-python"
    else:
        detail = "nvcc not detected; fallback to default opencv-contrib-python"

    selection = build_default_opencv_selection(detail)
    return WdtaggerOpenCvSelection(
        package_name=selection.package_name,
        package_spec=selection.package_spec,
        cuda_tag=cuda_tag,
        source=selection.source,
        detail=selection.detail,
    )


def build_wdtagger_opencv_install_plan(
    env: Mapping[str, str] | None = None,
    *,
    platform: str | None = None,
) -> WdtaggerOpenCvInstallPlan:
    selection = resolve_wdtagger_windows_opencv_requirement(env=env, platform=platform)
    attempts = [selection]
    if selection.source == "cuda-wheel":
        attempts.append(
            build_default_opencv_selection(
                "GPU OpenCV import probe failed; fallback to default opencv-contrib-python",
                source="cpu-fallback",
            )
        )
    return WdtaggerOpenCvInstallPlan(
        cleanup_packages=OPENCV_DISTRIBUTION_PACKAGES,
        attempts=tuple(attempts),
    )


def probe_cv2_runtime() -> tuple[dict[str, object], int]:
    try:
        import cv2
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 1

    payload: dict[str, object] = {
        "ok": True,
        "version": getattr(cv2, "__version__", None),
        "file": getattr(cv2, "__file__", None),
    }
    cuda = getattr(cv2, "cuda", None)
    if cuda is not None and hasattr(cuda, "getCudaEnabledDeviceCount"):
        try:
            payload["cuda_count"] = cuda.getCudaEnabledDeviceCount()
        except Exception as exc:
            payload["cuda_probe_error"] = f"{type(exc).__name__}: {exc}"
    return payload, 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--probe-cv2"]:
        payload, exit_code = probe_cv2_runtime()
        print(json.dumps(payload, ensure_ascii=False))
        return exit_code
    if args[:1] == ["--plan-install"]:
        platform = None
        if len(args) == 3 and args[1] == "--platform":
            platform = args[2]
        elif len(args) != 1:
            print("Usage: python utils/wdtagger_opencv.py --plan-install [--platform win32]", file=sys.stderr)
            return 2
        plan = build_wdtagger_opencv_install_plan(env=None, platform=platform)
        print(json.dumps(asdict(plan), ensure_ascii=False))
        return 0
    print("Usage: python utils/wdtagger_opencv.py --probe-cv2 | --plan-install [--platform win32]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
