from __future__ import annotations

import hashlib
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping

from ...jcs import jcs_bytes, jcs_sha256
from . import frame_kernel, moc3_layout_kernel, moc3_sections_kernel, uv_kernel
from .attestation import (
    ATTESTATION_SCHEMA_VERSION,
    COORDINATE_SCHEMA_VERSION,
    KERNEL_SOURCE_DIGEST_VERSION,
    kernel_source_sha256,
    load_live2d_frame_attestation,
    validate_live2d_frame_attestation,
)
from .cubism_core import CubismModelState, exercise_moc_with_core, probe_cubism_core
from .cubism_renderer import (
    CubismRenderEvidence,
    probe_offscreen_harness,
    render_moc_with_offscreen_harness,
)
from .e0_assets import (
    E0_BASE_PARAMETER_VALUES,
    E0_EXPECTED_PARAMETER_VALUES,
    E0_RUNTIME_ASSET_VERSION,
    build_e0_expression_json_bytes,
    build_e0_motion_json_bytes,
    build_e0_orientation_rgba,
    write_e0_orientation_texture,
)
from .e0_fixture import (
    DEFORMER_E0_CANVAS_SIZE,
    STATIC_E0_CORE_UVS,
    STATIC_E0_ROOT_VERTICES,
    build_deformer_e0_missing_default_moc,
    build_deformer_e0_moc,
    build_static_e0_moc,
    evaluate_deformer_e0_canvas,
    evaluate_deformer_e0_reversed_stack_canvas,
)
from .moc3 import moc3_v400_layout_descriptor
from .moc3_codec import moc3_v400_sections_descriptor
from .runtime_toolchain import LIVE2D_VALIDATOR_PROTOCOL_DIGEST, detect_live2d_runtime_platform

E0_ATTESTATION_GENERATOR_VERSION = "live2d-e0-attestation-generator-v3"
E0_VALIDATOR_PROTOCOL_VERSION = "live2d-e0-validator-semantics-v4"
EXPECTED_CORE_VERSION = "06.00.0001"
EXPECTED_CORE_SHA256 = "d883c00d114fdf6cef61f439feb23e02d000fdf683e092803010470b80dfaf09"
EXPECTED_CORE_SHA256_BY_PLATFORM = {
    "linux-x86_64": "f741b043ae01a2821824412e3aa498b96564904fbf38cb40a9036f8372f3d77f",
    "windows-x86_64": EXPECTED_CORE_SHA256,
}
EXPECTED_BACKEND_BY_PLATFORM = {
    "linux-x86_64": "opengl-egl-headless",
    "windows-x86_64": "d3d11-warp",
}

_GEOMETRY_CASES = (
    {},
    {"ParamBreath": -1.0},
    {"ParamBreath": 1.0},
    {"ParamOuter": -20.0},
    {"ParamOuter": 20.0},
    {"ParamInner": -30.0},
    {"ParamInner": 30.0},
    {"ParamBreath": 0.375, "ParamOuter": -7.5, "ParamInner": 12.5},
    {"ParamBreath": -0.625, "ParamOuter": 13.25, "ParamInner": -21.75},
)


class E0AttestationGenerationError(RuntimeError):
    """Raised when fresh official runtime evidence cannot sign the frame contract."""


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _require_model_state(state: CubismModelState | None, *, fixture: str) -> CubismModelState:
    if state is None:
        raise E0AttestationGenerationError(f"{fixture} omitted the requested Core model state")
    return state


def _root_to_canvas(point: tuple[float, float]) -> tuple[float, float]:
    width, height = DEFORMER_E0_CANVAS_SIZE
    ppu = max(width, height)
    return (point[0] * ppu + width / 2.0, height / 2.0 - point[1] * ppu)


def _geometry_residual(
    state: CubismModelState,
    expected_by_id: Mapping[str, tuple[tuple[float, float], ...]],
) -> float:
    residual = 0.0
    if tuple(drawable.id for drawable in state.drawables) != tuple(expected_by_id):
        raise E0AttestationGenerationError("Core drawable order differs from the E0 geometry contract")
    for drawable in state.drawables:
        actual = tuple(_root_to_canvas(point) for point in drawable.vertex_positions)
        expected = expected_by_id[drawable.id]
        if len(actual) != len(expected):
            raise E0AttestationGenerationError("Core drawable vertex count differs from E0")
        for actual_point, expected_point in zip(actual, expected):
            residual = max(
                residual,
                abs(actual_point[0] - expected_point[0]),
                abs(actual_point[1] - expected_point[1]),
            )
    return residual


def _negative_default_residual(positive: CubismModelState, negative: CubismModelState) -> float:
    positive_nested = next(drawable for drawable in positive.drawables if drawable.id == "ArtMeshNested")
    negative_nested = next(drawable for drawable in negative.drawables if drawable.id == "ArtMeshNested")
    ppu = max(DEFORMER_E0_CANVAS_SIZE)
    return max(
        max(abs(good[0] - bad[0]), abs(good[1] - bad[1])) * ppu
        for good, bad in zip(positive_nested.vertex_positions, negative_nested.vertex_positions)
    )


def _sample_near_static_vertex(
    evidence: CubismRenderEvidence,
    vertex: tuple[float, float],
) -> tuple[int, int, int, int]:
    centroid = (
        sum(point[0] for point in STATIC_E0_ROOT_VERTICES) / len(STATIC_E0_ROOT_VERTICES),
        sum(point[1] for point in STATIC_E0_ROOT_VERTICES) / len(STATIC_E0_ROOT_VERTICES),
    )
    point = (
        vertex[0] * 0.9 + centroid[0] * 0.1,
        vertex[1] * 0.9 + centroid[1] * 0.1,
    )
    x = round((point[0] + 1.0) * 0.5 * (evidence.width - 1))
    y = round((1.0 - point[1]) * 0.5 * (evidence.height - 1))
    offset = (y * evidence.width + x) * 4
    return tuple(evidence.rgba[offset : offset + 4])  # type: ignore[return-value]


def _orientation_passed(evidence: CubismRenderEvidence) -> bool:
    samples = tuple(_sample_near_static_vertex(evidence, vertex) for vertex in STATIC_E0_ROOT_VERTICES)
    return (
        samples[0][0] > 200
        and samples[0][1] < 30
        and samples[0][2] < 30
        and samples[1][1] > 200
        and samples[1][0] < 30
        and samples[1][2] < 30
        and samples[2][0] > 200
        and samples[2][1] > 200
        and samples[2][2] < 30
        and samples[3][2] > 200
        and samples[3][0] < 30
        and samples[3][1] < 30
    )


def _protocol() -> dict[str, Any]:
    return {
        "protocol_version": E0_VALIDATOR_PROTOCOL_VERSION,
        "core_call_order": [
            "csmHasMocConsistency",
            "csmReviveMocInPlace",
            "csmInitializeModelInPlace",
            "set-parameter-defaults-or-requested-values",
            "csmUpdateModel",
            "capture-parameter-part-drawable-state",
        ],
        "runtime_application_order": [
            "load-model",
            "set-base-parameters",
            "start-motion-at-zero",
            "evaluate-motion-at-one-second-with-serialized-zero-fade",
            "apply-zero-fade-expression-at-full-weight",
            "update-model",
            "render-platform-offscreen-backend",
        ],
        "thresholds": {
            "default_rest_max_px": 0.1,
            "geometry_max_px": 0.1,
            "negative_default_min_px_exclusive": 0.1,
            "noncommuting_stack_min_px_exclusive": 0.1,
            "parameter_abs": 1e-6,
        },
        "runtime_json_encoding": {
            "encoding": "ascii",
            "indent": 2,
            "sort_keys": True,
            "terminal_newline": True,
            "reason": "Framework-5-r.5-numeric-parser-requires-delimiter-before-closing-token",
        },
    }


def _pure_vectors() -> list[dict[str, Any]]:
    return [
        {
            "vector_id": "canvas-root-round-trip",
            "kernel_id": "frame-kernel-v1",
            "operation": "canvas-root-round-trip-v1",
            "payload": {
                "input": {"canvas": [1024.0, 768.0], "point": [137.0, 611.0]},
                "expected": {
                    "canvas": [137.0, 611.0],
                    "root": [-0.3662109375, -0.2216796875],
                },
                "tolerance": 1e-12,
            },
        },
        {
            "vector_id": "cubism-v400-uv-path",
            "kernel_id": "uv-kernel-v1",
            "operation": "cubism-v400-uv-path-v1",
            "payload": {
                "input": {"canonical_top_left_uv": [0.125, 0.75]},
                "expected": {
                    "canonical_from_core": [0.125, 0.75],
                    "canonical_from_moc": [0.125, 0.75],
                    "core_api_uv": [0.125, 0.25],
                    "d3d11_sample_uv": [0.125, 0.75],
                    "moc_uv": [0.125, 0.75],
                },
                "tolerance": 1e-12,
            },
        },
        {
            "vector_id": "rotation-stack-rank-conflict",
            "kernel_id": "frame-kernel-v1",
            "operation": "rotation-stack-rank-conflict-v1",
            "payload": {
                "input": {
                    "entries": [
                        {"angle_degrees": 0.0, "origin": [0.0, 0.0], "rank": 10, "scale": 1.0},
                        {"angle_degrees": 5.0, "origin": [1.0, 2.0], "rank": 10, "scale": 1.0},
                    ]
                },
                "expected": {"conflict": True},
                "tolerance": 0.0,
            },
        },
        {
            "vector_id": "rotation-stack-round-trip",
            "kernel_id": "frame-kernel-v1",
            "operation": "rotation-stack-round-trip-v1",
            "payload": {
                "input": {
                    "entries": [
                        {"angle_degrees": 90.0, "origin": [10.0, 0.0], "rank": 10, "scale": 1.0},
                        {"angle_degrees": 0.0, "origin": [1.0, 0.0], "rank": 20, "scale": 1.0},
                    ],
                    "point": [0.0, 0.0],
                },
                "expected": {"local": [0.0, 0.0], "parent": [10.0, 1.0]},
                "tolerance": 1e-12,
            },
        },
        {
            "vector_id": "similarity-round-trip",
            "kernel_id": "frame-kernel-v1",
            "operation": "similarity-round-trip-v1",
            "payload": {
                "input": {
                    "angle_degrees": 90.0,
                    "origin": [10.0, 20.0],
                    "point": [2.0, 0.0],
                    "scale": 2.0,
                },
                "expected": {"local": [2.0, 0.0], "parent": [10.0, 24.0]},
                "tolerance": 1e-12,
            },
        },
    ]


def _kernel_sources() -> dict[str, bytes]:
    return {
        "frame-kernel-v1": Path(frame_kernel.__file__).read_bytes(),
        "moc3-layout-kernel-v1": Path(moc3_layout_kernel.__file__).read_bytes(),
        "moc3-sections-kernel-v1": Path(moc3_sections_kernel.__file__).read_bytes(),
        "uv-kernel-v1": Path(uv_kernel.__file__).read_bytes(),
    }


def _semantic_kernels(sources: Mapping[str, bytes]) -> list[dict[str, Any]]:
    return [
        {
            "kernel_id": "frame-kernel-v1",
            "kernel_version": frame_kernel.FRAME_KERNEL_VERSION,
            "semantics": {
                "canvas_y_axis": "down",
                "root_y_axis": "up",
                "rotation_order": "lower-rank-outer",
                "similarity_order": "T-origin-then-R-angle-then-S-scale",
            },
            "source_digest_version": KERNEL_SOURCE_DIGEST_VERSION,
            "source_sha256": kernel_source_sha256(sources["frame-kernel-v1"]),
        },
        {
            "kernel_id": "moc3-layout-kernel-v1",
            "kernel_version": moc3_layout_kernel.MOC3_LAYOUT_KERNEL_VERSION,
            "semantics": {"scope": "v400-header-sot-count-canvas-envelope"},
            "source_digest_version": KERNEL_SOURCE_DIGEST_VERSION,
            "source_sha256": kernel_source_sha256(sources["moc3-layout-kernel-v1"]),
        },
        {
            "kernel_id": "moc3-sections-kernel-v1",
            "kernel_version": moc3_sections_kernel.MOC3_SECTION_KERNEL_VERSION,
            "semantics": {"scope": "v400-typed-sections-and-writer-alignment"},
            "source_digest_version": KERNEL_SOURCE_DIGEST_VERSION,
            "source_sha256": kernel_source_sha256(sources["moc3-sections-kernel-v1"]),
        },
        {
            "kernel_id": "uv-kernel-v1",
            "kernel_version": uv_kernel.CUBISM_V400_UV_KERNEL_VERSION,
            "semantics": {
                "canonical_origin": "top-left",
                "core_api_v": "one-minus-canonical-v",
                "d3d11_shader_v": "one-minus-core-api-v",
                "moc_v": "canonical-v",
            },
            "source_digest_version": KERNEL_SOURCE_DIGEST_VERSION,
            "source_sha256": kernel_source_sha256(sources["uv-kernel-v1"]),
        },
    ]


def generate_live2d_e0_attestation(
    core_path: str | Path,
    renderer_path: str | Path,
    *,
    sdk_release: str,
    license_policy_acknowledged: bool,
    platform_id: str | None = None,
    backend_id: str | None = None,
    existing_attestation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if sdk_release != "5-r.5":
        raise E0AttestationGenerationError("this generator is pinned to Cubism SDK 5-r.5")
    if license_policy_acknowledged is not True:
        raise E0AttestationGenerationError("license policy acknowledgement is required")
    selected_platform = platform_id or detect_live2d_runtime_platform()
    expected_backend = EXPECTED_BACKEND_BY_PLATFORM.get(selected_platform)
    expected_core_sha256 = EXPECTED_CORE_SHA256_BY_PLATFORM.get(selected_platform)
    if expected_backend is None or expected_core_sha256 is None:
        raise E0AttestationGenerationError("the requested runtime platform is not supported")
    selected_backend = backend_id or expected_backend
    if selected_backend != expected_backend:
        raise E0AttestationGenerationError("runtime backend does not match the selected platform")
    renderer_probe = probe_offscreen_harness(
        renderer_path,
        expected_backend=selected_backend,
    )
    if renderer_probe.protocol_digest != LIVE2D_VALIDATOR_PROTOCOL_DIGEST:
        raise E0AttestationGenerationError("validator protocol does not match the reviewed protocol")

    core_probe = probe_cubism_core(core_path)
    if core_probe.version.label != EXPECTED_CORE_VERSION or core_probe.sha256 != expected_core_sha256:
        raise E0AttestationGenerationError("Cubism Core identity does not match the reviewed SDK binary")
    if not core_probe.capabilities.attestation_candidate:
        raise E0AttestationGenerationError("Cubism Core lacks the complete E0 API/consistency gate")

    static_moc = build_static_e0_moc()
    deformer_moc = build_deformer_e0_moc()
    negative_moc = build_deformer_e0_missing_default_moc()
    motion_bytes = build_e0_motion_json_bytes()
    expression_bytes = build_e0_expression_json_bytes()
    with tempfile.TemporaryDirectory(prefix="auto-rig-live2d-e0-attestation-") as directory:
        root = Path(directory)
        static_path = root / "static.moc3"
        deformer_path = root / "deformer.moc3"
        negative_path = root / "negative-default.moc3"
        texture_path = root / "orientation.png"
        motion_path = root / "e0.motion3.json"
        expression_path = root / "e0.exp3.json"
        static_path.write_bytes(static_moc)
        deformer_path.write_bytes(deformer_moc)
        negative_path.write_bytes(negative_moc)
        motion_path.write_bytes(motion_bytes)
        expression_path.write_bytes(expression_bytes)
        write_e0_orientation_texture(texture_path)

        static_core = exercise_moc_with_core(
            core_path,
            static_path,
            capture_model_state=True,
        )
        positive_default = exercise_moc_with_core(
            core_path,
            deformer_path,
            capture_model_state=True,
        )
        negative_default = exercise_moc_with_core(
            core_path,
            negative_path,
            capture_model_state=True,
        )
        for result in (static_core, positive_default, negative_default):
            if result.consistency is not True:
                raise E0AttestationGenerationError("an E0 fixture failed csmHasMocConsistency")
        static_state = _require_model_state(static_core.model_state, fixture="static E0")
        positive_state = _require_model_state(positive_default.model_state, fixture="deformer E0")
        negative_state = _require_model_state(negative_default.model_state, fixture="negative E0")

        geometry_matrix: list[dict[str, Any]] = []
        max_geometry_residual = 0.0
        for parameters in _GEOMETRY_CASES:
            result = exercise_moc_with_core(
                core_path,
                deformer_path,
                parameter_values=parameters,
                capture_model_state=True,
            )
            if result.consistency is not True:
                raise E0AttestationGenerationError("a parameterized E0 case failed consistency")
            state = _require_model_state(result.model_state, fixture="parameterized deformer E0")
            residual = _geometry_residual(state, evaluate_deformer_e0_canvas(parameters))
            max_geometry_residual = max(max_geometry_residual, residual)
            geometry_matrix.append(
                {
                    "parameter_values": dict(sorted(parameters.items())),
                    "residual_px": round(residual, 9),
                    "vertex_position_sha256": f"sha256:{state.vertex_position_sha256}",
                }
            )
        if max_geometry_residual > 0.1:
            raise E0AttestationGenerationError("Core geometry residual exceeds 0.1 px")
        default_residual = _geometry_residual(positive_state, evaluate_deformer_e0_canvas())
        if default_residual > 0.1:
            raise E0AttestationGenerationError("explicit non-midpoint default does not preserve rest")
        negative_residual = _negative_default_residual(positive_state, negative_state)
        if negative_residual <= 0.1:
            raise E0AttestationGenerationError("negative default fixture did not expose the missing rest key")
        forward = evaluate_deformer_e0_canvas({"ParamOuter": 20.0, "ParamInner": 30.0})["ArtMeshNested"]
        reversed_stack = evaluate_deformer_e0_reversed_stack_canvas({"ParamOuter": 20.0, "ParamInner": 30.0})
        noncommuting_delta = max(max(abs(good[0] - bad[0]), abs(good[1] - bad[1])) for good, bad in zip(forward, reversed_stack))
        if noncommuting_delta <= 0.1:
            raise E0AttestationGenerationError("rotation stack fixture is not observably non-commutative")

        static_drawable = static_state.drawables[0]
        if any(
            max(abs(actual[0] - expected[0]), abs(actual[1] - expected[1])) > 1e-6
            for actual, expected in zip(static_drawable.vertex_uvs, STATIC_E0_CORE_UVS)
        ):
            raise E0AttestationGenerationError("Core UV representation differs from the signed adapter")
        static_render = render_moc_with_offscreen_harness(
            renderer_path,
            static_path,
            texture_path,
            expected_backend=selected_backend,
        )
        orientation_passed = _orientation_passed(static_render)
        straight_alpha_edge_passed = any(0 < alpha < 255 for alpha in static_render.rgba[3::4])
        if not orientation_passed or not straight_alpha_edge_passed:
            raise E0AttestationGenerationError("static UV/straight-alpha render evidence failed")

        baseline_render = render_moc_with_offscreen_harness(
            renderer_path,
            deformer_path,
            texture_path,
            parameter_values=E0_BASE_PARAMETER_VALUES,
            expected_backend=selected_backend,
        )
        effect_render = render_moc_with_offscreen_harness(
            renderer_path,
            deformer_path,
            texture_path,
            parameter_values=E0_BASE_PARAMETER_VALUES,
            motion_path=motion_path,
            expression_path=expression_path,
            evaluation_time=1.0,
            observe_parameter_ids=tuple(E0_EXPECTED_PARAMETER_VALUES),
            expected_backend=selected_backend,
        )
        for parameter_id, expected in E0_EXPECTED_PARAMETER_VALUES.items():
            actual = effect_render.parameter_values[parameter_id]
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6):
                raise E0AttestationGenerationError("motion/expression parameter formula mismatch")
        motion_expression_pixel_changed = baseline_render.rgba_sha256 != effect_render.rgba_sha256
        if not motion_expression_pixel_changed:
            raise E0AttestationGenerationError("motion/expression did not change rendered pixels")

    protocol = _protocol()
    sources = _kernel_sources()
    repository_root = Path(__file__).resolve().parents[4]
    validator_source_root = repository_root / "tools" / "auto_rig_live2d_e0"
    platform_source = (
        "windows/main_d3d11.cpp"
        if selected_platform == "windows-x86_64"
        else "linux/main_egl.cpp"
    )
    validator_source_files = (
        "CMakeLists.txt",
        "common/validator_common.cpp",
        "common/validator_common.hpp",
        platform_source,
    )
    validator_source_inventory = [
        {
            "path": relative_path,
            "sha256": _sha256_file(validator_source_root / Path(*relative_path.split("/"))),
        }
        for relative_path in validator_source_files
    ]
    runtime_fixtures = [
        {
            "fixture_id": "deformer-runtime-v1",
            "payload": {
                "baseline_rgba_sha256": baseline_render.rgba_sha256,
                "core_default_vertex_sha256": f"sha256:{positive_state.vertex_position_sha256}",
                "deformer_moc_sha256": _sha256_bytes(deformer_moc),
                "effect_rgba_sha256": effect_render.rgba_sha256,
                "expression_sha256": _sha256_bytes(expression_bytes),
                "geometry_case_count": len(geometry_matrix),
                "geometry_matrix_sha256": jcs_sha256(geometry_matrix),
                "max_geometry_residual_px": round(max_geometry_residual, 9),
                "missing_default_moc_sha256": _sha256_bytes(negative_moc),
                "motion_expression_pixel_changed": motion_expression_pixel_changed,
                "motion_sha256": _sha256_bytes(motion_bytes),
                "negative_default_residual_px": round(negative_residual, 9),
                "noncommuting_stack_delta_px": round(noncommuting_delta, 9),
                "observed_parameter_values": dict(effect_render.parameter_values),
                "runtime_asset_version": E0_RUNTIME_ASSET_VERSION,
            },
        },
        {
            "fixture_id": "static-uv-alpha-v1",
            "payload": {
                "alpha_bbox": list(static_render.alpha_bbox) if static_render.alpha_bbox else None,
                "canonical_texture_rgba_sha256": _sha256_bytes(build_e0_orientation_rgba()),
                "core_uv_sha256": f"sha256:{static_state.vertex_uv_sha256}",
                "driver_type": static_render.driver_type,
                "nonzero_alpha_pixels": static_render.nonzero_alpha_pixels,
                "orientation_passed": orientation_passed,
                "premultiplied_alpha_input": static_render.premultiplied_alpha_input,
                "render_rgba_sha256": static_render.rgba_sha256,
                "static_moc_sha256": _sha256_bytes(static_moc),
                "straight_alpha_edge_passed": straight_alpha_edge_passed,
            },
        },
    ]
    descriptor: dict[str, Any] = {
        "coordinate_schema_version": COORDINATE_SCHEMA_VERSION,
        "frame_kinds": [
            {
                "frame_kind_id": "CANVAS_PIXEL",
                "ordinal": 0,
                "semantics": {"units": "layerdiff-canvas-pixel", "y_axis": "down"},
            },
            {
                "frame_kind_id": "ROOT_MODEL",
                "ordinal": 1,
                "semantics": {"origin": "canvas-center", "units": "max-edge-ppu", "y_axis": "up"},
            },
            {
                "frame_kind_id": "WARP_LOCAL",
                "ordinal": 2,
                "semantics": {
                    "domain": "unit-square",
                    "mapping": "axis-aligned-rest-grid",
                    "quad_transforms": True,
                    "v1_scope": "rectangular-structural-warp-only",
                },
            },
            {
                "frame_kind_id": "ROTATION_LOCAL",
                "ordinal": 3,
                "semantics": {
                    "origin": "parent-deformer-frame",
                    "rigid_vector_scale": "canvas-ppu-not-parent-warp-stretch",
                    "transform": "T-origin-then-R-angle-then-S-scale",
                },
            },
            {
                "frame_kind_id": "ARTMESH_PARENT_LOCAL",
                "ordinal": 4,
                "semantics": {"parent": "direct-parent-input-frame"},
            },
        ],
        "semantic_kernels": _semantic_kernels(sources),
        "layout_descriptors": [
            moc3_v400_layout_descriptor(),
            moc3_v400_sections_descriptor(),
        ],
        "pure_vectors": _pure_vectors(),
        "invariants": [
            {
                "invariant_id": "default-rest",
                "payload": {"max_canvas_residual_px": 0.1, "negative_control_required": True},
            },
            {"invariant_id": "e0-validator-protocol", "payload": protocol},
            {
                "invariant_id": "lower-rank-outer",
                "payload": {"minimum_noncommuting_delta_px_exclusive": 0.1},
            },
            {
                "invariant_id": "runtime-json-compatibility",
                "payload": protocol["runtime_json_encoding"],
            },
            {
                "invariant_id": "uv-alpha-path",
                "payload": {
                    "canonical_alpha": "straight",
                    "canonical_color_space": "sRGB",
                    "canonical_uv_origin": "top-left",
                },
            },
        ],
    }
    runtime_record = {
        "backend_id": selected_backend,
        "core_sha256": f"sha256:{core_probe.sha256}",
        "core_version": core_probe.version.label,
        "e0_fixtures": runtime_fixtures,
        "platform_id": selected_platform,
        "runtime_provenance": {
            "core_latest_moc_version": core_probe.latest_moc_version,
            "generator_version": E0_ATTESTATION_GENERATOR_VERSION,
            "renderer_sha256": _sha256_file(Path(renderer_path).resolve()),
            "sdk_release": sdk_release,
            "validator_source_files": validator_source_inventory,
        },
        "validator_protocol_digest": renderer_probe.protocol_digest,
    }
    runtime_records: list[dict[str, Any]] = []
    if existing_attestation is not None:
        validate_live2d_frame_attestation(existing_attestation, kernel_sources=sources)
        if existing_attestation["contract_descriptor"] != descriptor:
            raise E0AttestationGenerationError(
                "existing attestation uses a different shared frame contract"
            )
        raw_records = existing_attestation["runtime_attestations"]
        if not isinstance(raw_records, list):
            raise E0AttestationGenerationError("existing runtime attestations are malformed")
        runtime_records.extend(
            dict(record)
            for record in raw_records
            if isinstance(record, dict)
            and (record.get("platform_id"), record.get("backend_id"))
            != (selected_platform, selected_backend)
        )
    runtime_records.append(runtime_record)
    runtime_records.sort(
        key=lambda record: (
            record["platform_id"],
            record["backend_id"],
            record["core_sha256"],
            record["validator_protocol_digest"],
        )
    )
    payload = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "contract_descriptor": descriptor,
        "live2d_frame_contract_digest": jcs_sha256(descriptor),
        "provenance": {
            "generator_version": E0_ATTESTATION_GENERATOR_VERSION,
            "license_policy": {
                "acknowledged": True,
                "core_or_sdk_redistributed_by_package": False,
                "organizational_release_license": "external-gate-not-asserted-by-code",
            },
            "sdk_release": sdk_release,
            "spec_revision": 45,
        },
        "runtime_attestations": runtime_records,
    }
    validate_live2d_frame_attestation(payload, kernel_sources=sources)
    return payload


def write_live2d_e0_attestation(
    output_path: str | Path,
    core_path: str | Path,
    renderer_path: str | Path,
    *,
    sdk_release: str,
    license_policy_acknowledged: bool,
    platform_id: str | None = None,
    backend_id: str | None = None,
) -> None:
    output = Path(output_path)
    existing: Mapping[str, Any] | None = None
    if output.is_file():
        loaded = load_live2d_frame_attestation(output.read_bytes())
        if loaded.get("schema_version") == ATTESTATION_SCHEMA_VERSION:
            existing = loaded
    payload = generate_live2d_e0_attestation(
        core_path,
        renderer_path,
        sdk_release=sdk_release,
        license_policy_acknowledged=license_policy_acknowledged,
        platform_id=platform_id,
        backend_id=backend_id,
        existing_attestation=existing,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(jcs_bytes(payload))


__all__ = [
    "E0_ATTESTATION_GENERATOR_VERSION",
    "E0_VALIDATOR_PROTOCOL_VERSION",
    "E0AttestationGenerationError",
    "EXPECTED_CORE_SHA256",
    "EXPECTED_CORE_SHA256_BY_PLATFORM",
    "EXPECTED_CORE_VERSION",
    "generate_live2d_e0_attestation",
    "write_live2d_e0_attestation",
]
