from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from module.auto_rig.export.live2d.attestation import (
    Live2DAttestationError,
    kernel_source_sha256,
    load_live2d_frame_attestation,
    load_packaged_live2d_frame_attestation,
    select_runtime_attestation,
    validate_live2d_frame_attestation,
)
from module.auto_rig.export.live2d.moc3 import moc3_v400_layout_descriptor
from module.auto_rig.export.live2d.runtime_toolchain import (
    LIVE2D_VALIDATOR_PROTOCOL_DIGEST,
    live2d_validator_source_sha256,
)
from module.auto_rig.jcs import jcs_bytes, jcs_sha256


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('ascii')).hexdigest()}"


def _kernel_sources() -> dict[str, bytes]:
    return {
        "frame-kernel-v1": b"frame\nsemantic\n",
        "moc3-layout-kernel-v1": b"layout\nsemantic\n",
        "uv-kernel-v1": b"uv\nsemantic\n",
    }


def _pure_vectors() -> list[dict[str, object]]:
    return [
        {
            "vector_id": "canvas-root-round-trip",
            "kernel_id": "frame-kernel-v1",
            "operation": "canvas-root-round-trip-v1",
            "payload": {
                "input": {"canvas": [1024.0, 512.0], "point": [0.0, 0.0]},
                "expected": {"canvas": [0.0, 0.0], "root": [-0.5, 0.25]},
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


def _attestation() -> tuple[dict[str, object], dict[str, bytes]]:
    sources = _kernel_sources()
    layout = moc3_v400_layout_descriptor()
    descriptor: dict[str, object] = {
        "coordinate_schema_version": "live2d-frames-v1",
        "frame_kinds": [
            {"frame_kind_id": "CANVAS_PIXEL", "ordinal": 0, "semantics": {"units": "layerdiff-canvas-pixel", "y_axis": "down"}},
            {"frame_kind_id": "ROOT_MODEL", "ordinal": 1, "semantics": {"units": "ppu-normalized", "y_axis": "up"}},
            {"frame_kind_id": "WARP_LOCAL", "ordinal": 2, "semantics": {"domain": "unit-square", "mapping": "rest-grid-inverse"}},
            {"frame_kind_id": "ROTATION_LOCAL", "ordinal": 3, "semantics": {"origin": "parent-frame-pivot", "units": "parent-dependent"}},
            {"frame_kind_id": "ARTMESH_PARENT_LOCAL", "ordinal": 4, "semantics": {"parent": "direct-parent-input-frame"}},
        ],
        "semantic_kernels": [
            {
                "kernel_id": "frame-kernel-v1",
                "kernel_version": "live2d-frame-kernel-v1",
                "semantics": {"rotation_order": "lower-rank-outer"},
                "source_digest_version": "kernel-source-digest-v1",
                "source_sha256": kernel_source_sha256(sources["frame-kernel-v1"]),
            },
            {
                "kernel_id": "moc3-layout-kernel-v1",
                "kernel_version": "moc3-v400-envelope-layout-v2",
                "semantics": {"scope": "header-sot-count-canvas-envelope-only"},
                "source_digest_version": "kernel-source-digest-v1",
                "source_sha256": kernel_source_sha256(sources["moc3-layout-kernel-v1"]),
            },
            {
                "kernel_id": "uv-kernel-v1",
                "kernel_version": "cubism-v400-uv-kernel-v1",
                "semantics": {"canonical_origin": "top-left", "moc_v": "identity"},
                "source_digest_version": "kernel-source-digest-v1",
                "source_sha256": kernel_source_sha256(sources["uv-kernel-v1"]),
            },
        ],
        "layout_descriptors": [layout],
        "pure_vectors": _pure_vectors(),
        "invariants": [
            {"invariant_id": "default-rest", "payload": {"max_canvas_residual_px": 0.1}},
            {"invariant_id": "e0-validator-protocol", "payload": {"protocol_version": "test-e0-protocol-v1", "steps": ["parse", "compare"]}},
            {"invariant_id": "lower-rank-outer", "payload": {"minimum_noncommuting_delta_px": 0.1}},
        ],
    }
    runtime_attestations: list[dict[str, object]] = [
        {
            "backend_id": "d3d11-warp",
            "core_sha256": _digest("windows-test-core"),
            "core_version": "test-only",
            "e0_fixtures": [
                {"fixture_id": "deformer-runtime-v1", "payload": {"render_rgba_sha256": _digest("windows-deformer")}},
                {"fixture_id": "static-uv-alpha-v1", "payload": {"render_rgba_sha256": _digest("windows-static")}},
            ],
            "platform_id": "windows-x86_64",
            "runtime_provenance": {"generator": "test-only"},
            "validator_protocol_digest": _digest("test-validator-protocol"),
            "validator_source_sha256": _digest("windows-validator-source"),
        },
    ]
    payload: dict[str, object] = {
        "schema_version": "live2d-frame-attestation-v3",
        "contract_descriptor": descriptor,
        "live2d_frame_contract_digest": jcs_sha256(descriptor),
        "provenance": {
            "compiler_version": "test-only",
            "source_date_epoch": "1785542400",
        },
        "runtime_attestations": runtime_attestations,
    }
    return payload, sources


def test_validate_and_select_live2d_frame_attestation(tmp_path: Path) -> None:
    payload, sources = _attestation()
    path = tmp_path / "live2d-frames-v1.json"
    path.write_bytes(jcs_bytes(payload))

    loaded = load_live2d_frame_attestation(path.read_bytes())
    validate_live2d_frame_attestation(loaded, kernel_sources=sources)
    selected = select_runtime_attestation(
        loaded,
        platform_id="windows-x86_64",
        backend_id="d3d11-warp",
        core_sha256=_digest("windows-test-core"),
        validator_protocol_digest=_digest("test-validator-protocol"),
        validator_source_sha256=_digest("windows-validator-source"),
    )

    assert selected.backend_id == "d3d11-warp"


def test_validate_attestation_rejects_mismatched_descriptor_or_vector(tmp_path: Path) -> None:
    payload, sources = _attestation()
    tampered = copy.deepcopy(payload)
    tampered["live2d_frame_contract_digest"] = _digest("wrong")
    path = tmp_path / "bad.json"
    path.write_bytes(jcs_bytes(tampered))

    with pytest.raises(Live2DAttestationError, match="digest"):
        validate_live2d_frame_attestation(load_live2d_frame_attestation(path.read_bytes()), kernel_sources=sources)


def test_packaged_live2d_frame_attestation_is_valid() -> None:
    validated = load_packaged_live2d_frame_attestation()

    assert validated["contract_descriptor"]["coordinate_schema_version"] == "live2d-frames-v1"
    for record in validated["runtime_attestations"]:
        assert record["validator_protocol_digest"] == LIVE2D_VALIDATOR_PROTOCOL_DIGEST
        assert record["validator_source_sha256"] == live2d_validator_source_sha256(
            Path("tools/auto_rig_live2d_e0"),
            platform_id=record["platform_id"],
        )
