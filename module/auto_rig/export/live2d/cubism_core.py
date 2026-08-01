from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

CONSISTENCY_EXPORT = "csmHasMocConsistency"
MIN_SAFE_CONSISTENCY_VERSION_RAW = 0x04020004

E0_CORE_EXPORTS = (
    "csmGetDrawableCount",
    "csmGetDrawableDrawOrders",
    "csmGetDrawableIds",
    "csmGetDrawableOpacities",
    "csmGetDrawableTextureIndices",
    "csmGetDrawableVertexCounts",
    "csmGetDrawableVertexPositions",
    "csmGetDrawableVertexUvs",
    "csmGetLatestMocVersion",
    "csmGetMocVersion",
    "csmGetParameterCount",
    "csmGetParameterDefaultValues",
    "csmGetParameterIds",
    "csmGetParameterMaximumValues",
    "csmGetParameterMinimumValues",
    "csmGetParameterValues",
    "csmGetPartCount",
    "csmGetPartIds",
    "csmGetPartOpacities",
    "csmGetSizeofModel",
    "csmGetVersion",
    "csmInitializeModelInPlace",
    "csmReadCanvasInfo",
    "csmReviveMocInPlace",
    "csmUpdateModel",
)


class CubismCoreError(RuntimeError):
    """Raised when the opt-in Cubism Core probe cannot produce trusted evidence."""


@dataclass(frozen=True)
class CubismCoreVersion:
    raw: int
    major: int
    minor: int
    revision: int

    @property
    def label(self) -> str:
        return f"{self.major:02d}.{self.minor:02d}.{self.revision:04d}"


@dataclass(frozen=True)
class CubismCoreCapabilities:
    e0_runtime_api_available: bool
    consistency_api_available: bool
    consistency_floor_met: bool
    attestation_candidate: bool
    missing_e0_exports: tuple[str, ...]
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class CubismCoreProbe:
    path: str
    file_size: int
    sha256: str
    version: CubismCoreVersion
    latest_moc_version: int
    available_exports: tuple[str, ...]
    capabilities: CubismCoreCapabilities


@dataclass(frozen=True)
class CubismParameterState:
    id: str
    minimum_value: float
    maximum_value: float
    default_value: float
    value: float


@dataclass(frozen=True)
class CubismPartState:
    id: str
    opacity: float


@dataclass(frozen=True)
class CubismDrawableState:
    id: str
    texture_index: int
    draw_order: int
    opacity: float
    vertex_positions: tuple[tuple[float, float], ...]
    vertex_uvs: tuple[tuple[float, float], ...]


def _drawable_point_digest(drawables: tuple[CubismDrawableState, ...], field: str) -> str:
    digest = hashlib.sha256()
    for drawable in drawables:
        encoded_id = drawable.id.encode("utf-8")
        digest.update(struct.pack("<I", len(encoded_id)))
        digest.update(encoded_id)
        points = getattr(drawable, field)
        digest.update(struct.pack("<I", len(points)))
        for x, y in points:
            digest.update(struct.pack("<2f", x, y))
    return digest.hexdigest()


@dataclass(frozen=True)
class CubismModelState:
    parameters: tuple[CubismParameterState, ...]
    parts: tuple[CubismPartState, ...]
    drawables: tuple[CubismDrawableState, ...]

    @property
    def vertex_position_sha256(self) -> str:
        return _drawable_point_digest(self.drawables, "vertex_positions")

    @property
    def vertex_uv_sha256(self) -> str:
        return _drawable_point_digest(self.drawables, "vertex_uvs")


@dataclass(frozen=True)
class CubismMocRuntimeResult:
    core_path: str
    core_sha256: str
    moc_path: str
    moc_sha256: str
    moc_file_size: int
    detected_moc_version: int
    consistency: bool | None
    model_size: int
    canvas_size: tuple[float, float]
    canvas_origin: tuple[float, float]
    pixels_per_unit: float
    parameter_count: int
    part_count: int
    drawable_count: int
    total_vertex_count: int
    nonzero_drawable_count: int
    finite_vertices: bool
    vertex_bounds: tuple[float, float, float, float] | None
    model_state: CubismModelState | None


def decode_core_version(raw: int) -> CubismCoreVersion:
    if type(raw) is not int or raw < 0 or raw > 0xFFFFFFFF:
        raise CubismCoreError("Cubism Core version must be an unsigned 32-bit integer")
    return CubismCoreVersion(
        raw=raw,
        major=(raw >> 24) & 0xFF,
        minor=(raw >> 16) & 0xFF,
        revision=raw & 0xFFFF,
    )


def assess_core_capabilities(version_raw: int, available_exports: Iterable[str]) -> CubismCoreCapabilities:
    version = decode_core_version(version_raw)
    exports = frozenset(available_exports)
    if any(type(name) is not str or not name for name in exports):
        raise CubismCoreError("Cubism Core export names must be non-empty strings")

    missing_e0_exports = tuple(sorted(set(E0_CORE_EXPORTS) - exports))
    consistency_api_available = CONSISTENCY_EXPORT in exports
    consistency_floor_met = version.raw >= MIN_SAFE_CONSISTENCY_VERSION_RAW
    blockers: list[str] = []
    if missing_e0_exports:
        blockers.append(f"missing_e0_exports:{','.join(missing_e0_exports)}")
    if not consistency_api_available:
        blockers.append(f"missing_export:{CONSISTENCY_EXPORT}")
    if not consistency_floor_met:
        floor = decode_core_version(MIN_SAFE_CONSISTENCY_VERSION_RAW)
        blockers.append(f"core_version_below_consistency_floor:{floor.label}")

    return CubismCoreCapabilities(
        e0_runtime_api_available=not missing_e0_exports,
        consistency_api_available=consistency_api_available,
        consistency_floor_met=consistency_floor_met,
        attestation_candidate=not blockers,
        missing_e0_exports=missing_e0_exports,
        blockers=tuple(blockers),
    )


def _resolve_file(path: str | os.PathLike[str], *, field: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise CubismCoreError(f"{field} is not a file: {resolved}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_cubism_core(core_path: str | os.PathLike[str]) -> CubismCoreProbe:
    """Load a trusted local Core DLL and inspect the exact API needed by E0."""

    if os.name != "nt" or not hasattr(ctypes, "WinDLL"):
        raise CubismCoreError("the native Cubism Core probe currently supports Windows only")
    path = _resolve_file(core_path, field="Cubism Core path")
    try:
        library = ctypes.WinDLL(str(path))
    except OSError as exc:
        raise CubismCoreError(f"failed to load Cubism Core: {path}") from exc

    known_exports = (*E0_CORE_EXPORTS, CONSISTENCY_EXPORT)
    available_exports = tuple(sorted(name for name in known_exports if hasattr(library, name)))
    for required in ("csmGetVersion", "csmGetLatestMocVersion"):
        if required not in available_exports:
            raise CubismCoreError(f"Cubism Core is missing probe export: {required}")

    get_version = library.csmGetVersion
    get_version.argtypes = []
    get_version.restype = ctypes.c_uint32
    get_latest_moc_version = library.csmGetLatestMocVersion
    get_latest_moc_version.argtypes = []
    get_latest_moc_version.restype = ctypes.c_uint32
    version_raw = int(get_version())

    return CubismCoreProbe(
        path=str(path),
        file_size=path.stat().st_size,
        sha256=_sha256_file(path),
        version=decode_core_version(version_raw),
        latest_moc_version=int(get_latest_moc_version()),
        available_exports=available_exports,
        capabilities=assess_core_capabilities(version_raw, available_exports),
    )


def _require_worker_payload(payload: object) -> dict[str, object]:
    if type(payload) is not dict:
        raise CubismCoreError("Cubism Core worker returned a non-object payload")
    result = payload
    if result.get("status") != "ok":
        message = result.get("message")
        raise CubismCoreError(f"Cubism Core worker failed: {message if type(message) is str else 'unknown error'}")
    return result


def _required_value(payload: dict[str, object], field: str, expected_type: type) -> object:
    value = payload.get(field)
    if expected_type is int:
        valid = type(value) is int
    elif expected_type is float:
        valid = type(value) in (int, float)
    else:
        valid = type(value) is expected_type
    if not valid:
        raise CubismCoreError(f"Cubism Core worker returned invalid {field}")
    return value


def _float_pair(payload: dict[str, object], field: str) -> tuple[float, float]:
    value = payload.get(field)
    if type(value) is not list or len(value) != 2 or any(type(item) not in (int, float) for item in value):
        raise CubismCoreError(f"Cubism Core worker returned invalid {field}")
    return (float(value[0]), float(value[1]))


def _optional_bounds(payload: dict[str, object]) -> tuple[float, float, float, float] | None:
    value = payload.get("vertex_bounds")
    if value is None:
        return None
    if type(value) is not list or len(value) != 4 or any(type(item) not in (int, float) for item in value):
        raise CubismCoreError("Cubism Core worker returned invalid vertex_bounds")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _normalize_parameter_values(parameter_values: Mapping[str, float] | None) -> tuple[tuple[str, float], ...]:
    if parameter_values is None:
        return ()
    if not isinstance(parameter_values, Mapping):
        raise CubismCoreError("parameter_values must be a mapping")
    normalized: list[tuple[str, float]] = []
    for parameter_id, value in parameter_values.items():
        if type(parameter_id) is not str or not parameter_id:
            raise CubismCoreError("parameter IDs must be non-empty strings")
        if type(value) not in (int, float) or not math.isfinite(value):
            raise CubismCoreError(f"parameter {parameter_id!r} must have a finite numeric value")
        normalized.append((parameter_id, float(value)))
    normalized.sort(key=lambda item: item[0])
    return tuple(normalized)


def _finite_float(value: object, *, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise CubismCoreError(f"Cubism Core worker returned invalid {field}")
    return float(value)


def _state_id(value: object, *, field: str) -> str:
    if type(value) is not str or not value:
        raise CubismCoreError(f"Cubism Core worker returned invalid {field}")
    return value


def _state_list(value: object, *, field: str) -> list[object]:
    if type(value) is not list:
        raise CubismCoreError(f"Cubism Core worker returned invalid {field}")
    return value


def _state_points(value: object, *, field: str) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = []
    for index, point in enumerate(_state_list(value, field=field)):
        if type(point) is not list or len(point) != 2:
            raise CubismCoreError(f"Cubism Core worker returned invalid {field}[{index}]")
        points.append(
            (
                _finite_float(point[0], field=f"{field}[{index}].x"),
                _finite_float(point[1], field=f"{field}[{index}].y"),
            )
        )
    return tuple(points)


def _parse_model_state(value: object) -> CubismModelState | None:
    if value is None:
        return None
    if type(value) is not dict or set(value) != {"parameters", "parts", "drawables"}:
        raise CubismCoreError("Cubism Core worker returned invalid model_state")

    parameters: list[CubismParameterState] = []
    for index, record in enumerate(_state_list(value["parameters"], field="model_state.parameters")):
        expected = {"id", "minimum_value", "maximum_value", "default_value", "value"}
        if type(record) is not dict or set(record) != expected:
            raise CubismCoreError(f"Cubism Core worker returned invalid model_state.parameters[{index}]")
        parameters.append(
            CubismParameterState(
                id=_state_id(record["id"], field=f"model_state.parameters[{index}].id"),
                minimum_value=_finite_float(
                    record["minimum_value"], field=f"model_state.parameters[{index}].minimum_value"
                ),
                maximum_value=_finite_float(
                    record["maximum_value"], field=f"model_state.parameters[{index}].maximum_value"
                ),
                default_value=_finite_float(
                    record["default_value"], field=f"model_state.parameters[{index}].default_value"
                ),
                value=_finite_float(record["value"], field=f"model_state.parameters[{index}].value"),
            )
        )

    parts: list[CubismPartState] = []
    for index, record in enumerate(_state_list(value["parts"], field="model_state.parts")):
        if type(record) is not dict or set(record) != {"id", "opacity"}:
            raise CubismCoreError(f"Cubism Core worker returned invalid model_state.parts[{index}]")
        parts.append(
            CubismPartState(
                id=_state_id(record["id"], field=f"model_state.parts[{index}].id"),
                opacity=_finite_float(record["opacity"], field=f"model_state.parts[{index}].opacity"),
            )
        )

    drawables: list[CubismDrawableState] = []
    for index, record in enumerate(_state_list(value["drawables"], field="model_state.drawables")):
        expected = {"id", "texture_index", "draw_order", "opacity", "vertex_positions", "vertex_uvs"}
        if type(record) is not dict or set(record) != expected:
            raise CubismCoreError(f"Cubism Core worker returned invalid model_state.drawables[{index}]")
        texture_index = record["texture_index"]
        draw_order = record["draw_order"]
        if type(texture_index) is not int or type(draw_order) is not int:
            raise CubismCoreError(f"Cubism Core worker returned invalid model_state.drawables[{index}] indices")
        positions = _state_points(
            record["vertex_positions"], field=f"model_state.drawables[{index}].vertex_positions"
        )
        uvs = _state_points(record["vertex_uvs"], field=f"model_state.drawables[{index}].vertex_uvs")
        if len(positions) != len(uvs):
            raise CubismCoreError(f"Cubism Core worker returned mismatched model_state.drawables[{index}] vertices")
        drawables.append(
            CubismDrawableState(
                id=_state_id(record["id"], field=f"model_state.drawables[{index}].id"),
                texture_index=texture_index,
                draw_order=draw_order,
                opacity=_finite_float(record["opacity"], field=f"model_state.drawables[{index}].opacity"),
                vertex_positions=positions,
                vertex_uvs=uvs,
            )
        )

    for field, records in (("parameter", parameters), ("part", parts), ("drawable", drawables)):
        ids = [record.id for record in records]
        if len(ids) != len(set(ids)):
            raise CubismCoreError(f"Cubism Core worker returned duplicate {field} IDs")
    return CubismModelState(parameters=tuple(parameters), parts=tuple(parts), drawables=tuple(drawables))


def exercise_moc_with_core(
    core_path: str | os.PathLike[str],
    moc_path: str | os.PathLike[str],
    *,
    parameter_values: Mapping[str, float] | None = None,
    capture_model_state: bool = False,
    timeout_seconds: float = 30.0,
) -> CubismMocRuntimeResult:
    """Load and update a MOC in a subprocess so a native crash cannot kill the batch worker."""

    normalized_parameter_values = _normalize_parameter_values(parameter_values)
    if type(capture_model_state) is not bool:
        raise CubismCoreError("capture_model_state must be a bool")
    if timeout_seconds <= 0:
        raise CubismCoreError("Cubism Core worker timeout must be positive")
    core = _resolve_file(core_path, field="Cubism Core path")
    moc = _resolve_file(moc_path, field="MOC path")
    probe = probe_cubism_core(core)
    if not probe.capabilities.e0_runtime_api_available:
        missing = ",".join(probe.capabilities.missing_e0_exports)
        raise CubismCoreError(f"Cubism Core lacks the E0 runtime API: {missing}")

    with tempfile.TemporaryDirectory(prefix="auto-rig-cubism-core-") as temp_dir:
        output_path = Path(temp_dir) / "result.json"
        request_path = Path(temp_dir) / "request.json"
        request_path.write_text(
            json.dumps(
                {
                    "capture_model_state": capture_model_state,
                    "parameter_values": [
                        {"id": parameter_id, "value": value}
                        for parameter_id, value in normalized_parameter_values
                    ],
                },
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="ascii",
        )
        command = (
            sys.executable,
            "-m",
            "module.auto_rig.export.live2d._cubism_core_worker",
            "--core",
            str(core),
            "--moc",
            str(moc),
            "--output",
            str(output_path),
            "--request",
            str(request_path),
        )
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise CubismCoreError(f"Cubism Core worker timed out after {timeout_seconds:g}s") from exc

        payload: object = None
        if output_path.is_file():
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CubismCoreError("Cubism Core worker result is unreadable") from exc
        if completed.returncode != 0:
            if type(payload) is dict and type(payload.get("message")) is str:
                detail = payload["message"]
            else:
                detail = completed.stderr.strip() or completed.stdout.strip() or "native process terminated"
            raise CubismCoreError(f"Cubism Core worker exited with {completed.returncode}: {detail}")

        result = _require_worker_payload(payload)
        consistency = result.get("consistency")
        if consistency is not None and type(consistency) is not bool:
            raise CubismCoreError("Cubism Core worker returned invalid consistency")
        finite_vertices = _required_value(result, "finite_vertices", bool)

        model_state = _parse_model_state(result.get("model_state"))
        parameter_count = int(_required_value(result, "parameter_count", int))
        part_count = int(_required_value(result, "part_count", int))
        drawable_count = int(_required_value(result, "drawable_count", int))
        if model_state is not None and (
            len(model_state.parameters) != parameter_count
            or len(model_state.parts) != part_count
            or len(model_state.drawables) != drawable_count
        ):
            raise CubismCoreError("Cubism Core worker model_state counts do not match its summary")

        return CubismMocRuntimeResult(
            core_path=str(_required_value(result, "core_path", str)),
            core_sha256=str(_required_value(result, "core_sha256", str)),
            moc_path=str(_required_value(result, "moc_path", str)),
            moc_sha256=str(_required_value(result, "moc_sha256", str)),
            moc_file_size=int(_required_value(result, "moc_file_size", int)),
            detected_moc_version=int(_required_value(result, "detected_moc_version", int)),
            consistency=consistency,
            model_size=int(_required_value(result, "model_size", int)),
            canvas_size=_float_pair(result, "canvas_size"),
            canvas_origin=_float_pair(result, "canvas_origin"),
            pixels_per_unit=float(_required_value(result, "pixels_per_unit", float)),
            parameter_count=parameter_count,
            part_count=part_count,
            drawable_count=drawable_count,
            total_vertex_count=int(_required_value(result, "total_vertex_count", int)),
            nonzero_drawable_count=int(_required_value(result, "nonzero_drawable_count", int)),
            finite_vertices=bool(finite_vertices),
            vertex_bounds=_optional_bounds(result),
            model_state=model_state,
        )


__all__ = [
    "CONSISTENCY_EXPORT",
    "E0_CORE_EXPORTS",
    "MIN_SAFE_CONSISTENCY_VERSION_RAW",
    "CubismCoreCapabilities",
    "CubismCoreError",
    "CubismCoreProbe",
    "CubismCoreVersion",
    "CubismDrawableState",
    "CubismModelState",
    "CubismMocRuntimeResult",
    "CubismParameterState",
    "CubismPartState",
    "assess_core_capabilities",
    "decode_core_version",
    "exercise_moc_with_core",
    "probe_cubism_core",
]
