from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path

from .cubism_core import CONSISTENCY_EXPORT, MIN_SAFE_CONSISTENCY_VERSION_RAW

MOC_ALIGNMENT = 64
MODEL_ALIGNMENT = 16


class _Vector2(ctypes.Structure):
    _fields_ = (("x", ctypes.c_float), ("y", ctypes.c_float))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aligned_copy(payload: bytes, alignment: int) -> tuple[ctypes.Array[ctypes.c_char], ctypes.c_void_p]:
    owner = ctypes.create_string_buffer(len(payload) + alignment - 1)
    address = (ctypes.addressof(owner) + alignment - 1) & ~(alignment - 1)
    ctypes.memmove(address, payload, len(payload))
    return owner, ctypes.c_void_p(address)


def _aligned_zero(size: int, alignment: int) -> tuple[ctypes.Array[ctypes.c_char], ctypes.c_void_p]:
    owner = ctypes.create_string_buffer(size + alignment - 1)
    address = (ctypes.addressof(owner) + alignment - 1) & ~(alignment - 1)
    ctypes.memset(address, 0, size)
    return owner, ctypes.c_void_p(address)


def _bind(library: ctypes.WinDLL, name: str, restype: object, *argtypes: object) -> object:
    function = getattr(library, name)
    function.restype = restype
    function.argtypes = list(argtypes)
    return function


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate request key: {key}")
        result[key] = value
    return result


def _load_request(request_path: Path) -> tuple[bool, tuple[tuple[str, float], ...]]:
    try:
        payload = json.loads(
            request_path.read_text(encoding="ascii"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Cubism Core request is unreadable") from exc
    if type(payload) is not dict or set(payload) != {"capture_model_state", "parameter_values"}:
        raise RuntimeError("Cubism Core request has an invalid schema")
    capture_model_state = payload["capture_model_state"]
    records = payload["parameter_values"]
    if type(capture_model_state) is not bool or type(records) is not list:
        raise RuntimeError("Cubism Core request has invalid field types")
    values: list[tuple[str, float]] = []
    for index, record in enumerate(records):
        if type(record) is not dict or set(record) != {"id", "value"}:
            raise RuntimeError(f"Cubism Core parameter request {index} is invalid")
        parameter_id = record["id"]
        value = record["value"]
        if type(parameter_id) is not str or not parameter_id:
            raise RuntimeError(f"Cubism Core parameter request {index} has an invalid ID")
        if type(value) not in (int, float) or not math.isfinite(value):
            raise RuntimeError(f"Cubism Core parameter request {parameter_id!r} has an invalid value")
        values.append((parameter_id, float(value)))
    if values != sorted(values, key=lambda item: item[0]) or len({item[0] for item in values}) != len(values):
        raise RuntimeError("Cubism Core parameter requests must have unique sorted IDs")
    return capture_model_state, tuple(values)


def _decode_ids(pointer: object, count: int, *, field: str) -> list[str]:
    result: list[str] = []
    for index in range(count):
        raw = pointer[index]
        if raw is None:
            raise RuntimeError(f"Core returned a null {field} ID")
        try:
            value = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"Core returned a non-UTF-8 {field} ID") from exc
        if not value:
            raise RuntimeError(f"Core returned an empty {field} ID")
        result.append(value)
    if len(result) != len(set(result)):
        raise RuntimeError(f"Core returned duplicate {field} IDs")
    return result


def _exercise(
    core_path: Path,
    moc_path: Path,
    *,
    capture_model_state: bool,
    requested_parameter_values: tuple[tuple[str, float], ...],
) -> dict[str, object]:
    if os.name != "nt" or not hasattr(ctypes, "WinDLL"):
        raise RuntimeError("the native Cubism Core worker currently supports Windows only")
    core_path = core_path.resolve(strict=True)
    moc_path = moc_path.resolve(strict=True)
    moc_payload = moc_path.read_bytes()
    if not moc_payload:
        raise RuntimeError("MOC payload is empty")
    if len(moc_payload) > 0xFFFFFFFF:
        raise RuntimeError("MOC payload exceeds the Core uint32 size limit")

    library = ctypes.WinDLL(str(core_path))
    get_version = _bind(library, "csmGetVersion", ctypes.c_uint32)
    get_latest_moc_version = _bind(library, "csmGetLatestMocVersion", ctypes.c_uint32)
    get_moc_version = _bind(library, "csmGetMocVersion", ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32)
    revive_moc = _bind(library, "csmReviveMocInPlace", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32)
    get_model_size = _bind(library, "csmGetSizeofModel", ctypes.c_uint32, ctypes.c_void_p)
    initialize_model = _bind(
        library,
        "csmInitializeModelInPlace",
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
    )
    update_model = _bind(library, "csmUpdateModel", None, ctypes.c_void_p)
    read_canvas = _bind(
        library,
        "csmReadCanvasInfo",
        None,
        ctypes.c_void_p,
        ctypes.POINTER(_Vector2),
        ctypes.POINTER(_Vector2),
        ctypes.POINTER(ctypes.c_float),
    )
    get_parameter_count = _bind(library, "csmGetParameterCount", ctypes.c_int32, ctypes.c_void_p)
    get_parameter_ids = _bind(
        library,
        "csmGetParameterIds",
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_void_p,
    )
    get_parameter_minimum_values = _bind(
        library,
        "csmGetParameterMinimumValues",
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
    )
    get_parameter_maximum_values = _bind(
        library,
        "csmGetParameterMaximumValues",
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
    )
    get_parameter_default_values = _bind(
        library,
        "csmGetParameterDefaultValues",
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
    )
    get_parameter_values = _bind(
        library,
        "csmGetParameterValues",
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
    )
    get_part_count = _bind(library, "csmGetPartCount", ctypes.c_int32, ctypes.c_void_p)
    get_part_ids = _bind(library, "csmGetPartIds", ctypes.POINTER(ctypes.c_char_p), ctypes.c_void_p)
    get_part_opacities = _bind(
        library,
        "csmGetPartOpacities",
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
    )
    get_drawable_count = _bind(library, "csmGetDrawableCount", ctypes.c_int32, ctypes.c_void_p)
    get_drawable_ids = _bind(
        library,
        "csmGetDrawableIds",
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_void_p,
    )
    get_drawable_texture_indices = _bind(
        library,
        "csmGetDrawableTextureIndices",
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_void_p,
    )
    get_drawable_draw_orders = _bind(
        library,
        "csmGetDrawableDrawOrders",
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_void_p,
    )
    get_drawable_opacities = _bind(
        library,
        "csmGetDrawableOpacities",
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
    )
    get_vertex_counts = _bind(library, "csmGetDrawableVertexCounts", ctypes.POINTER(ctypes.c_int32), ctypes.c_void_p)
    get_vertex_positions = _bind(
        library,
        "csmGetDrawableVertexPositions",
        ctypes.POINTER(ctypes.POINTER(_Vector2)),
        ctypes.c_void_p,
    )
    get_vertex_uvs = _bind(
        library,
        "csmGetDrawableVertexUvs",
        ctypes.POINTER(ctypes.POINTER(_Vector2)),
        ctypes.c_void_p,
    )

    version_raw = int(get_version())
    latest_moc_version = int(get_latest_moc_version())
    moc_owner, moc_pointer = _aligned_copy(moc_payload, MOC_ALIGNMENT)
    if int(moc_pointer.value or 0) % MOC_ALIGNMENT:
        raise RuntimeError("failed to align MOC memory")

    consistency: bool | None = None
    if version_raw >= MIN_SAFE_CONSISTENCY_VERSION_RAW and hasattr(library, CONSISTENCY_EXPORT):
        has_consistency = _bind(
            library,
            CONSISTENCY_EXPORT,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_uint32,
        )
        consistency = bool(has_consistency(moc_pointer, len(moc_payload)))
        if not consistency:
            raise RuntimeError("csmHasMocConsistency rejected the MOC")

    detected_moc_version = int(get_moc_version(moc_pointer, len(moc_payload)))
    if detected_moc_version <= 0 or detected_moc_version > latest_moc_version:
        raise RuntimeError(f"Core cannot load MOC version {detected_moc_version}; latest supported is {latest_moc_version}")
    revived = revive_moc(moc_pointer, len(moc_payload))
    if not revived:
        raise RuntimeError("csmReviveMocInPlace rejected the MOC")

    model_size = int(get_model_size(ctypes.c_void_p(revived)))
    if model_size <= 0:
        raise RuntimeError("csmGetSizeofModel returned an invalid size")
    model_owner, model_pointer = _aligned_zero(model_size, MODEL_ALIGNMENT)
    if int(model_pointer.value or 0) % MODEL_ALIGNMENT:
        raise RuntimeError("failed to align model memory")
    model = initialize_model(ctypes.c_void_p(revived), model_pointer, model_size)
    if not model:
        raise RuntimeError("csmInitializeModelInPlace rejected the revived MOC")

    canvas_size = _Vector2()
    canvas_origin = _Vector2()
    pixels_per_unit = ctypes.c_float()
    read_canvas(ctypes.c_void_p(model), ctypes.byref(canvas_size), ctypes.byref(canvas_origin), ctypes.byref(pixels_per_unit))
    parameter_count = int(get_parameter_count(ctypes.c_void_p(model)))
    part_count = int(get_part_count(ctypes.c_void_p(model)))
    drawable_count = int(get_drawable_count(ctypes.c_void_p(model)))
    if min(parameter_count, part_count, drawable_count) < 0 or drawable_count > 10_000_000:
        raise RuntimeError("Core returned an invalid object count")

    parameter_ids_pointer = get_parameter_ids(ctypes.c_void_p(model))
    parameter_ids = _decode_ids(parameter_ids_pointer, parameter_count, field="parameter")
    parameter_minimum_values = get_parameter_minimum_values(ctypes.c_void_p(model))
    parameter_maximum_values = get_parameter_maximum_values(ctypes.c_void_p(model))
    parameter_default_values = get_parameter_default_values(ctypes.c_void_p(model))
    parameter_values = get_parameter_values(ctypes.c_void_p(model))
    parameter_indices = {parameter_id: index for index, parameter_id in enumerate(parameter_ids)}
    for parameter_id, value in requested_parameter_values:
        parameter_index = parameter_indices.get(parameter_id)
        if parameter_index is None:
            raise RuntimeError(f"unknown parameter: {parameter_id}")
        minimum = float(parameter_minimum_values[parameter_index])
        maximum = float(parameter_maximum_values[parameter_index])
        if value < minimum or value > maximum:
            raise RuntimeError(
                f"parameter {parameter_id} value {value:g} is outside the model range [{minimum:g}, {maximum:g}]"
            )
        parameter_values[parameter_index] = value

    update_model(ctypes.c_void_p(model))
    vertex_counts = get_vertex_counts(ctypes.c_void_p(model))
    vertex_positions = get_vertex_positions(ctypes.c_void_p(model))
    vertex_uvs = get_vertex_uvs(ctypes.c_void_p(model))
    total_vertex_count = 0
    nonzero_drawable_count = 0
    finite_vertices = True
    min_x = min_y = math.inf
    max_x = max_y = -math.inf
    for drawable_index in range(drawable_count):
        vertex_count = int(vertex_counts[drawable_index])
        if vertex_count < 0 or vertex_count > 10_000_000:
            raise RuntimeError("Core returned an invalid drawable vertex count")
        total_vertex_count += vertex_count
        if total_vertex_count > 100_000_000:
            raise RuntimeError("Core returned an unreasonable total vertex count")
        if vertex_count:
            nonzero_drawable_count += 1
        for vertex_index in range(vertex_count):
            point = vertex_positions[drawable_index][vertex_index]
            finite_vertices = finite_vertices and math.isfinite(point.x) and math.isfinite(point.y)
            min_x = min(min_x, point.x)
            min_y = min(min_y, point.y)
            max_x = max(max_x, point.x)
            max_y = max(max_y, point.y)

    vertex_bounds = None
    if total_vertex_count:
        vertex_bounds = [min_x, min_y, max_x, max_y]

    model_state: dict[str, object] | None = None
    if capture_model_state:
        part_ids = _decode_ids(get_part_ids(ctypes.c_void_p(model)), part_count, field="part")
        part_opacities = get_part_opacities(ctypes.c_void_p(model))
        drawable_ids = _decode_ids(get_drawable_ids(ctypes.c_void_p(model)), drawable_count, field="drawable")
        drawable_texture_indices = get_drawable_texture_indices(ctypes.c_void_p(model))
        drawable_draw_orders = get_drawable_draw_orders(ctypes.c_void_p(model))
        drawable_opacities = get_drawable_opacities(ctypes.c_void_p(model))
        model_state = {
            "parameters": [
                {
                    "id": parameter_ids[index],
                    "minimum_value": float(parameter_minimum_values[index]),
                    "maximum_value": float(parameter_maximum_values[index]),
                    "default_value": float(parameter_default_values[index]),
                    "value": float(parameter_values[index]),
                }
                for index in range(parameter_count)
            ],
            "parts": [
                {"id": part_ids[index], "opacity": float(part_opacities[index])}
                for index in range(part_count)
            ],
            "drawables": [
                {
                    "id": drawable_ids[drawable_index],
                    "texture_index": int(drawable_texture_indices[drawable_index]),
                    "draw_order": int(drawable_draw_orders[drawable_index]),
                    "opacity": float(drawable_opacities[drawable_index]),
                    "vertex_positions": [
                        [
                            float(vertex_positions[drawable_index][vertex_index].x),
                            float(vertex_positions[drawable_index][vertex_index].y),
                        ]
                        for vertex_index in range(int(vertex_counts[drawable_index]))
                    ],
                    "vertex_uvs": [
                        [
                            float(vertex_uvs[drawable_index][vertex_index].x),
                            float(vertex_uvs[drawable_index][vertex_index].y),
                        ]
                        for vertex_index in range(int(vertex_counts[drawable_index]))
                    ],
                }
                for drawable_index in range(drawable_count)
            ],
        }
    del model_owner, moc_owner
    return {
        "status": "ok",
        "core_path": str(core_path),
        "core_sha256": _sha256_file(core_path),
        "core_version_raw": version_raw,
        "latest_moc_version": latest_moc_version,
        "moc_path": str(moc_path),
        "moc_sha256": hashlib.sha256(moc_payload).hexdigest(),
        "moc_file_size": len(moc_payload),
        "detected_moc_version": detected_moc_version,
        "consistency": consistency,
        "model_size": model_size,
        "canvas_size": [canvas_size.x, canvas_size.y],
        "canvas_origin": [canvas_origin.x, canvas_origin.y],
        "pixels_per_unit": pixels_per_unit.value,
        "parameter_count": parameter_count,
        "part_count": part_count,
        "drawable_count": drawable_count,
        "total_vertex_count": total_vertex_count,
        "nonzero_drawable_count": nonzero_drawable_count,
        "finite_vertices": finite_vertices,
        "vertex_bounds": vertex_bounds,
        "model_state": model_state,
    }


def _write_result(output_path: Path, payload: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--moc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    try:
        capture_model_state, requested_parameter_values = _load_request(args.request)
        payload = _exercise(
            args.core,
            args.moc,
            capture_model_state=capture_model_state,
            requested_parameter_values=requested_parameter_values,
        )
    except BaseException as exc:
        _write_result(
            args.output,
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        return 2
    _write_result(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
