from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class CubismRendererError(RuntimeError):
    """Raised when the opt-in official SDK render harness rejects a request."""


LIVE2D_E0_VALIDATOR_PROTOCOL_DIGEST = (
    "sha256:51e77ee76d08072db76e1ccef0638e8c706ccba7bae5ea2ae8e19b71269283a3"
)


@dataclass(frozen=True, slots=True)
class CubismRenderEvidence:
    width: int
    height: int
    rgba: bytes
    rgba_sha256: str
    nonzero_alpha_pixels: int
    alpha_bbox: tuple[int, int, int, int] | None
    driver_type: str
    premultiplied_alpha_input: bool
    validator_protocol_digest: str
    parameter_values: Mapping[str, float]


_PARAMETER_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")


def _validated_path(value: str | Path, *, field: str) -> Path:
    try:
        path = Path(value).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CubismRendererError(f"{field} does not resolve to an existing file") from exc
    if not path.is_file():
        raise CubismRendererError(f"{field} must be a file")
    return path


def _validate_dimension(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 8192:
        raise CubismRendererError(f"{field} must be an integer in [1, 8192]")
    return value


def _validate_parameter_id(parameter_id: object) -> str:
    if not isinstance(parameter_id, str) or _PARAMETER_ID_PATTERN.fullmatch(parameter_id) is None:
        raise CubismRendererError(
            "parameter IDs must be non-empty stable ASCII identifiers"
        )
    return parameter_id


def _validate_parameters(parameter_values: Mapping[str, float] | None) -> tuple[str, ...]:
    if parameter_values is None:
        return ()
    if not isinstance(parameter_values, Mapping):
        raise CubismRendererError("parameter_values must be a mapping")
    arguments: list[str] = []
    for parameter_id in sorted(parameter_values):
        value = parameter_values[parameter_id]
        _validate_parameter_id(parameter_id)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise CubismRendererError("parameter values must be finite JSON numbers")
        arguments.extend(("--parameter", f"{parameter_id}={float(value):.17g}"))
    return tuple(arguments)


def _validate_texture_paths(
    value: str | Path | Sequence[str | Path],
) -> tuple[Path, ...]:
    raw_values: tuple[str | Path, ...]
    if isinstance(value, (str, Path)):
        raw_values = (value,)
    elif isinstance(value, Sequence):
        raw_values = tuple(value)
    else:
        raise CubismRendererError("texture_path must be a path or ordered path sequence")
    if not 1 <= len(raw_values) <= 4:
        raise CubismRendererError("texture_path must contain one to four ordered pages")
    paths = tuple(
        _validated_path(item, field=f"texture_path[{index}]")
        for index, item in enumerate(raw_values)
    )
    if len(set(paths)) != len(paths):
        raise CubismRendererError("texture_path must not contain duplicate pages")
    return paths


def _validate_observed_parameters(parameter_ids: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(parameter_ids, tuple):
        raise CubismRendererError("observe_parameter_ids must be a tuple")
    normalized = tuple(sorted(_validate_parameter_id(parameter_id) for parameter_id in parameter_ids))
    if len(set(normalized)) != len(normalized):
        raise CubismRendererError("observe_parameter_ids must not contain duplicates")
    arguments: list[str] = []
    for parameter_id in normalized:
        arguments.extend(("--observe-parameter", parameter_id))
    return tuple(arguments)


def _alpha_summary(rgba: bytes, width: int, height: int) -> tuple[int, tuple[int, int, int, int] | None]:
    nonzero = 0
    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    for index in range(width * height):
        if rgba[index * 4 + 3] == 0:
            continue
        nonzero += 1
        x = index % width
        y = index // width
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
    if nonzero == 0:
        return 0, None
    return nonzero, (min_x, min_y, max_x + 1, max_y + 1)


def render_moc_with_offscreen_harness(
    executable: str | Path,
    moc_path: str | Path,
    texture_path: str | Path | Sequence[str | Path],
    *,
    width: int = 512,
    height: int = 512,
    parameter_values: Mapping[str, float] | None = None,
    motion_path: str | Path | None = None,
    expression_path: str | Path | None = None,
    evaluation_time: float = 0.0,
    observe_parameter_ids: tuple[str, ...] = (),
    timeout_seconds: float = 60.0,
) -> CubismRenderEvidence:
    executable_file = _validated_path(executable, field="executable")
    moc_file = _validated_path(moc_path, field="moc_path")
    texture_files = _validate_texture_paths(texture_path)
    width = _validate_dimension(width, field="width")
    height = _validate_dimension(height, field="height")
    motion_file = None if motion_path is None else _validated_path(motion_path, field="motion_path")
    expression_file = (
        None
        if expression_path is None
        else _validated_path(expression_path, field="expression_path")
    )
    if isinstance(evaluation_time, bool) or not isinstance(evaluation_time, (int, float)):
        raise CubismRendererError("evaluation_time must be a non-negative finite number")
    evaluation_time = float(evaluation_time)
    if not math.isfinite(evaluation_time) or evaluation_time < 0.0:
        raise CubismRendererError("evaluation_time must be a non-negative finite number")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise CubismRendererError("timeout_seconds must be a positive finite number")
    timeout_seconds = float(timeout_seconds)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise CubismRendererError("timeout_seconds must be a positive finite number")
    parameter_arguments = _validate_parameters(parameter_values)
    observed_parameter_arguments = _validate_observed_parameters(observe_parameter_ids)

    with tempfile.TemporaryDirectory(prefix="auto-rig-live2d-render-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        rgba_path = temporary_root / "frame.rgba"
        report_path = temporary_root / "report.json"
        command = (
            str(executable_file),
            "--moc",
            str(moc_file),
            *(
                argument
                for texture_file in texture_files
                for argument in ("--texture", str(texture_file))
            ),
            "--output",
            str(rgba_path),
            "--report",
            str(report_path),
            "--width",
            str(width),
            "--height",
            str(height),
            *parameter_arguments,
            *(("--motion", str(motion_file)) if motion_file is not None else ()),
            *(("--expression", str(expression_file)) if expression_file is not None else ()),
            "--time",
            f"{evaluation_time:.17g}",
            *observed_parameter_arguments,
        )
        try:
            result = subprocess.run(
                command,
                cwd=executable_file.parent,
                capture_output=True,
                check=False,
                text=True,
                timeout=timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CubismRendererError("official SDK render harness could not complete") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise CubismRendererError(
                f"official SDK render harness failed with exit code {result.returncode}: {detail}"
            )
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            rgba = rgba_path.read_bytes()
        except (OSError, json.JSONDecodeError) as exc:
            raise CubismRendererError("official SDK render harness omitted valid evidence") from exc

    expected_fields = {
        "alpha_bbox",
        "driver_type",
        "height",
        "nonzero_alpha_pixels",
        "parameter_values",
        "premultiplied_alpha_input",
        "schema_version",
        "validator_protocol_digest",
        "width",
    }
    if type(report) is not dict or set(report) != expected_fields:
        raise CubismRendererError("official SDK render report fields do not match the contract")
    if report["schema_version"] != "auto-rig-live2d-render-v1":
        raise CubismRendererError("official SDK render report schema is unsupported")
    if report["validator_protocol_digest"] != LIVE2D_E0_VALIDATOR_PROTOCOL_DIGEST:
        raise CubismRendererError(
            "official SDK render harness protocol is not attested"
        )
    if report["width"] != width or report["height"] != height:
        raise CubismRendererError("official SDK render dimensions do not match the request")
    if len(rgba) != width * height * 4:
        raise CubismRendererError("official SDK raw RGBA byte length is invalid")
    nonzero_alpha_pixels, alpha_bbox = _alpha_summary(rgba, width, height)
    reported_bbox = report["alpha_bbox"]
    normalized_reported_bbox = None if reported_bbox is None else tuple(reported_bbox)
    if (
        report["nonzero_alpha_pixels"] != nonzero_alpha_pixels
        or normalized_reported_bbox != alpha_bbox
    ):
        raise CubismRendererError("official SDK render summary does not match the raw RGBA evidence")
    if report["driver_type"] != "d3d11-warp":
        raise CubismRendererError("official SDK render harness did not use the D3D11 WARP driver")
    if type(report["premultiplied_alpha_input"]) is not bool:
        raise CubismRendererError("official SDK render alpha mode is invalid")
    reported_parameter_values = report["parameter_values"]
    if type(reported_parameter_values) is not dict or set(reported_parameter_values) != set(
        observe_parameter_ids
    ):
        raise CubismRendererError("official SDK render observed parameter set is invalid")
    normalized_parameter_values: dict[str, float] = {}
    for parameter_id, value in reported_parameter_values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CubismRendererError("official SDK render observed parameter value is invalid")
        normalized_value = float(value)
        if not math.isfinite(normalized_value):
            raise CubismRendererError("official SDK render observed parameter value is invalid")
        normalized_parameter_values[parameter_id] = normalized_value
    return CubismRenderEvidence(
        width=width,
        height=height,
        rgba=rgba,
        rgba_sha256=f"sha256:{hashlib.sha256(rgba).hexdigest()}",
        nonzero_alpha_pixels=nonzero_alpha_pixels,
        alpha_bbox=alpha_bbox,
        driver_type=report["driver_type"],
        premultiplied_alpha_input=report["premultiplied_alpha_input"],
        validator_protocol_digest=report["validator_protocol_digest"],
        parameter_values=MappingProxyType(normalized_parameter_values),
    )


__all__ = [
    "CubismRenderEvidence",
    "CubismRendererError",
    "LIVE2D_E0_VALIDATOR_PROTOCOL_DIGEST",
    "render_moc_with_offscreen_harness",
]
