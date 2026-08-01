from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from module.auto_rig.export.live2d import (
    frame_kernel,
    moc3_layout_kernel,
    moc3_sections_kernel,
    uv_kernel,
)
from module.auto_rig.export.live2d.attestation import (
    Live2DAttestationError,
    kernel_source_sha256,
    load_live2d_frame_attestation,
    validate_live2d_frame_attestation,
)
from module.auto_rig.export.live2d.e0_attestation import generate_live2d_e0_attestation
from module.auto_rig.export.live2d.moc3 import moc3_v400_layout_descriptor
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
    assert isinstance(layout["payload"], dict)
    protocol = {"protocol_version": "test-e0-protocol-v1", "steps": ["parse", "compare"]}
    descriptor: dict[str, object] = {
        "coordinate_schema_version": "live2d-frames-v1",
        "frame_kinds": [
            {
                "frame_kind_id": "CANVAS_PIXEL",
                "ordinal": 0,
                "semantics": {"units": "layerdiff-canvas-pixel", "y_axis": "down"},
            },
            {
                "frame_kind_id": "ROOT_MODEL",
                "ordinal": 1,
                "semantics": {"units": "ppu-normalized", "y_axis": "up"},
            },
            {
                "frame_kind_id": "WARP_LOCAL",
                "ordinal": 2,
                "semantics": {"domain": "unit-square", "mapping": "rest-grid-inverse"},
            },
            {
                "frame_kind_id": "ROTATION_LOCAL",
                "ordinal": 3,
                "semantics": {"origin": "parent-frame-pivot", "units": "parent-dependent"},
            },
            {
                "frame_kind_id": "ARTMESH_PARENT_LOCAL",
                "ordinal": 4,
                "semantics": {"parent": "direct-parent-input-frame"},
            },
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
        "e0_fixtures": [
            {
                "fixture_id": "synthetic-structural-v1",
                "payload": {
                    "fixture_sha256": _digest("synthetic-structural-v1"),
                    "scope": "parser-and-pure-vectors-only",
                },
            }
        ],
        "invariants": [
            {
                "invariant_id": "default-rest",
                "payload": {"max_canvas_residual_px": 0.1},
            },
            {
                "invariant_id": "e0-validator-protocol",
                "payload": protocol,
            },
            {
                "invariant_id": "lower-rank-outer",
                "payload": {"minimum_noncommuting_delta_px": 0.1},
            },
        ],
        "approved_core_binaries": [
            {
                "arch": "x86_64",
                "core_sha256": _digest("test-core"),
                "core_version": "test-only",
                "platform": "windows",
            }
        ],
        "e0_validator_protocol_digest": jcs_sha256(protocol),
    }
    payload: dict[str, object] = {
        "schema_version": "live2d-frame-attestation-v1",
        "contract_descriptor": descriptor,
        "live2d_frame_contract_digest": jcs_sha256(descriptor),
        "provenance": {
            "compiler_version": "test-only",
            "source_date_epoch": "1785542400",
        },
    }
    return payload, sources


def test_validate_live2d_frame_attestation_runs_structural_gate() -> None:
    payload, sources = _attestation()

    result = validate_live2d_frame_attestation(payload, kernel_sources=sources)

    assert result.schema_version == "live2d-frame-attestation-v1"
    assert result.coordinate_schema_version == "live2d-frames-v1"
    assert result.contract_digest == payload["live2d_frame_contract_digest"]
    assert result.kernel_ids == ("frame-kernel-v1", "moc3-layout-kernel-v1", "uv-kernel-v1")
    assert result.vector_ids == (
        "canvas-root-round-trip",
        "cubism-v400-uv-path",
        "rotation-stack-rank-conflict",
        "rotation-stack-round-trip",
        "similarity-round-trip",
    )
    assert result.approved_core_keys == (("windows", "x86_64", _digest("test-core")),)


def test_packaged_live2d_frame_attestation_is_current_and_release_attested() -> None:
    attestation_path = (
        Path(__file__).parents[1]
        / "module"
        / "auto_rig"
        / "export"
        / "live2d"
        / "attestations"
        / "live2d-frames-v1.json"
    )
    payload = load_live2d_frame_attestation(attestation_path.read_bytes())
    sources = {
        "frame-kernel-v1": Path(frame_kernel.__file__).read_bytes(),
        "moc3-layout-kernel-v1": Path(moc3_layout_kernel.__file__).read_bytes(),
        "moc3-sections-kernel-v1": Path(moc3_sections_kernel.__file__).read_bytes(),
        "uv-kernel-v1": Path(uv_kernel.__file__).read_bytes(),
    }

    result = validate_live2d_frame_attestation(payload, kernel_sources=sources)

    assert result.approved_core_keys == (
        (
            "windows",
            "x86_64",
            "sha256:d883c00d114fdf6cef61f439feb23e02d000fdf683e092803010470b80dfaf09",
        ),
    )
    descriptor = payload["contract_descriptor"]
    assert isinstance(descriptor, dict)
    fixtures = {
        fixture["fixture_id"]: fixture["payload"] for fixture in descriptor["e0_fixtures"]
    }
    assert fixtures["deformer-runtime-v1"]["max_geometry_residual_px"] <= 0.1
    assert fixtures["deformer-runtime-v1"]["negative_default_residual_px"] > 0.1
    assert fixtures["deformer-runtime-v1"]["motion_expression_pixel_changed"] is True
    assert fixtures["static-uv-alpha-v1"]["orientation_passed"] is True
    assert fixtures["static-uv-alpha-v1"]["straight_alpha_edge_passed"] is True
    provenance = payload["provenance"]
    assert isinstance(provenance, dict)
    assert provenance["sdk_release"] == "5-r.5"
    assert provenance["core_version"] == "06.00.0001"


@pytest.mark.optional_runtime
def test_packaged_attestation_reproduces_from_fresh_official_e0_evidence() -> None:
    core_path = os.environ.get("LIVE2D_CUBISM_CORE_PATH")
    renderer_path = os.environ.get("LIVE2D_E0_RENDERER_PATH")
    if not core_path or not renderer_path:
        pytest.skip("LIVE2D_CUBISM_CORE_PATH and LIVE2D_E0_RENDERER_PATH are required")
    attestation_path = (
        Path(__file__).parents[1]
        / "module"
        / "auto_rig"
        / "export"
        / "live2d"
        / "attestations"
        / "live2d-frames-v1.json"
    )

    regenerated = generate_live2d_e0_attestation(
        core_path,
        renderer_path,
        sdk_release="5-r.5",
        license_policy_acknowledged=True,
    )

    assert jcs_bytes(regenerated) == attestation_path.read_bytes()


def test_attestation_digest_excludes_provenance() -> None:
    payload, sources = _attestation()
    changed = copy.deepcopy(payload)
    changed["provenance"] = {"compiler_version": "different-provenance"}

    first = validate_live2d_frame_attestation(payload, kernel_sources=sources)
    second = validate_live2d_frame_attestation(changed, kernel_sources=sources)

    assert first.contract_digest == second.contract_digest


def test_kernel_source_digest_normalizes_only_bom_and_newlines() -> None:
    assert kernel_source_sha256(b"\xef\xbb\xbfalpha\r\nbeta\r") == kernel_source_sha256("alpha\nbeta\n")
    assert kernel_source_sha256("alpha\n") != kernel_source_sha256("alpha")
    assert kernel_source_sha256("alpha  \n") != kernel_source_sha256("alpha\n")


def test_kernel_source_digest_rejects_invalid_utf8() -> None:
    with pytest.raises(Live2DAttestationError):
        kernel_source_sha256(b"\xff")


def test_load_attestation_accepts_only_canonical_jcs_without_duplicate_keys() -> None:
    payload, _ = _attestation()

    assert load_live2d_frame_attestation(jcs_bytes(payload)) == payload

    with pytest.raises(Live2DAttestationError):
        load_live2d_frame_attestation(json.dumps(payload, indent=2, ensure_ascii=False))
    with pytest.raises(Live2DAttestationError):
        load_live2d_frame_attestation(b'{"schema_version":"a","schema_version":"b"}')


@pytest.mark.parametrize("source_mutation", ["missing", "extra", "mismatch"])
def test_validate_attestation_requires_exact_kernel_source_set(source_mutation: str) -> None:
    payload, sources = _attestation()
    if source_mutation == "missing":
        del sources["frame-kernel-v1"]
    elif source_mutation == "extra":
        sources["unattested-kernel"] = b"extra\n"
    else:
        sources["frame-kernel-v1"] = b"changed\n"

    with pytest.raises(Live2DAttestationError):
        validate_live2d_frame_attestation(payload, kernel_sources=sources)


def test_validate_attestation_rejects_descriptor_digest_mismatch() -> None:
    payload, sources = _attestation()
    payload["live2d_frame_contract_digest"] = _digest("wrong")

    with pytest.raises(Live2DAttestationError):
        validate_live2d_frame_attestation(payload, kernel_sources=sources)


def test_validate_attestation_rejects_validator_protocol_digest_mismatch() -> None:
    payload, sources = _attestation()
    descriptor = payload["contract_descriptor"]
    assert isinstance(descriptor, dict)
    descriptor["e0_validator_protocol_digest"] = _digest("wrong-protocol")
    payload["live2d_frame_contract_digest"] = jcs_sha256(descriptor)

    with pytest.raises(Live2DAttestationError, match="validator protocol digest mismatch"):
        validate_live2d_frame_attestation(payload, kernel_sources=sources)


def test_validate_attestation_rejects_unsorted_or_duplicate_records() -> None:
    payload, sources = _attestation()
    descriptor = payload["contract_descriptor"]
    assert isinstance(descriptor, dict)
    vectors = descriptor["pure_vectors"]
    assert isinstance(vectors, list)
    descriptor["pure_vectors"] = list(reversed(vectors))
    payload["live2d_frame_contract_digest"] = jcs_sha256(descriptor)

    with pytest.raises(Live2DAttestationError):
        validate_live2d_frame_attestation(payload, kernel_sources=sources)

    payload, sources = _attestation()
    descriptor = payload["contract_descriptor"]
    assert isinstance(descriptor, dict)
    fixtures = descriptor["e0_fixtures"]
    assert isinstance(fixtures, list)
    fixtures.append(copy.deepcopy(fixtures[0]))
    payload["live2d_frame_contract_digest"] = jcs_sha256(descriptor)

    with pytest.raises(Live2DAttestationError):
        validate_live2d_frame_attestation(payload, kernel_sources=sources)


def test_validate_attestation_rejects_json_text_instead_of_nested_payload() -> None:
    payload, sources = _attestation()
    descriptor = payload["contract_descriptor"]
    assert isinstance(descriptor, dict)
    layouts = descriptor["layout_descriptors"]
    assert isinstance(layouts, list)
    assert isinstance(layouts[0], dict)
    layouts[0]["payload"] = '{"hidden":true}'
    payload["live2d_frame_contract_digest"] = jcs_sha256(descriptor)

    with pytest.raises(Live2DAttestationError):
        validate_live2d_frame_attestation(payload, kernel_sources=sources)


def test_validate_attestation_rejects_production_registry_rows_in_frame_contract() -> None:
    payload, sources = _attestation()
    descriptor = payload["contract_descriptor"]
    assert isinstance(descriptor, dict)
    descriptor["rigid_driver_registry"] = [{"parameter_id": "ParamAngleX", "rank": 300}]
    payload["live2d_frame_contract_digest"] = jcs_sha256(descriptor)

    with pytest.raises(Live2DAttestationError):
        validate_live2d_frame_attestation(payload, kernel_sources=sources)


def test_validate_attestation_fails_when_a_pure_vector_changes() -> None:
    payload, sources = _attestation()
    descriptor = payload["contract_descriptor"]
    assert isinstance(descriptor, dict)
    vectors = descriptor["pure_vectors"]
    assert isinstance(vectors, list)
    vector = vectors[0]
    assert isinstance(vector, dict)
    vector_payload = vector["payload"]
    assert isinstance(vector_payload, dict)
    expected = vector_payload["expected"]
    assert isinstance(expected, dict)
    expected["root"] = [0.0, 0.0]
    payload["live2d_frame_contract_digest"] = jcs_sha256(descriptor)

    with pytest.raises(Live2DAttestationError):
        validate_live2d_frame_attestation(payload, kernel_sources=sources)


def test_validate_attestation_rejects_unknown_vector_operation() -> None:
    payload, sources = _attestation()
    descriptor = payload["contract_descriptor"]
    assert isinstance(descriptor, dict)
    vectors = descriptor["pure_vectors"]
    assert isinstance(vectors, list)
    assert isinstance(vectors[0], dict)
    vectors[0]["operation"] = "not-attested-v1"
    payload["live2d_frame_contract_digest"] = jcs_sha256(descriptor)

    with pytest.raises(Live2DAttestationError):
        validate_live2d_frame_attestation(payload, kernel_sources=sources)


def test_validate_attestation_checks_descriptor_identity_before_executing_vectors() -> None:
    payload, sources = _attestation()
    descriptor = payload["contract_descriptor"]
    assert isinstance(descriptor, dict)
    vectors = descriptor["pure_vectors"]
    assert isinstance(vectors, list)
    assert isinstance(vectors[0], dict)
    vectors[0]["operation"] = "not-attested-v1"
    payload["live2d_frame_contract_digest"] = _digest("wrong")

    with pytest.raises(Live2DAttestationError, match="contract digest mismatch"):
        validate_live2d_frame_attestation(payload, kernel_sources=sources)
