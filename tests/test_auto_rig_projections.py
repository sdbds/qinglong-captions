from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from module.auto_rig.projections import (
    MOTION_MANIFEST_PROJECTOR_VERSION,
    RIG_REPORT_PROJECTOR_VERSION,
    MotionManifestProjection,
    ProjectionError,
    motion_manifest_bytes,
    project_motion_manifest,
    project_rig_report,
    rig_report_bytes,
    validate_motion_manifest_projection,
    validate_rig_report_projection,
)
from tests.test_auto_rig_rig_document import _build


def test_motion_manifest_is_a_read_only_projection_of_rig_decisions(
    tmp_path: Path,
) -> None:
    *_inputs, rig = _build(tmp_path)

    projection = project_motion_manifest(rig)
    payload = projection.to_dict()

    assert projection.projector_version == MOTION_MANIFEST_PROJECTOR_VERSION
    assert validate_motion_manifest_projection(projection, rig) is projection
    assert payload["rig_json_sha256"] == rig.document_sha256
    assert payload["global_symbol_table_sha256"] == rig.to_dict()["export_symbols"][
        "table_sha256"
    ]
    assert payload["profile"] == "dual_runtime_core_v1"
    assert payload["default_clip"] == "idle"
    assert payload["runtime_application"] == rig.to_dict()["runtime_application"]

    by_id = {preset["preset_id"]: preset for preset in payload["presets"]}
    assert by_id["idle"]["required"] is True
    assert by_id["idle"]["supported_formats"] == [
        "live2d_moc3_v4_00",
        "spine_4_2",
    ]
    assert by_id["wave.xmin"]["supported_formats"] == ["spine_4_2"]
    assert by_id["wave.xmin"]["formats"]["live2d_moc3_v4_00"] == {
        "status": "omitted",
        "reason": "live2d_joint_bend_requires_glue",
        "incompatible_with": [],
        "artifact": None,
    }
    assert by_id["happy"]["formats"]["live2d_moc3_v4_00"]["reason"] == (
        "live2d_parameter_conflict"
    )
    assert by_id["talk"]["formats"]["spine_4_2"]["artifact"].startswith(
        "spine/skeleton.json#animations/"
    )
    assert by_id["talk"]["formats"]["live2d_moc3_v4_00"]["artifact"].endswith(
        ".motion3.json"
    )


def test_report_projection_contains_only_diagnostics_and_frozen_summaries(
    tmp_path: Path,
) -> None:
    *_inputs, rig = _build(tmp_path)

    report = project_rig_report(rig)
    payload = report.to_dict()

    assert report.projector_version == RIG_REPORT_PROJECTOR_VERSION
    assert validate_rig_report_projection(report, rig) is report
    assert payload["rig_json_sha256"] == rig.document_sha256
    assert payload["counts"] == {
        "parts": len(rig.parts),
        "joints": len(rig.joints),
        "bones": len(rig.bones),
        "meshes": len(rig.meshes),
        "texture_pages": len(rig.to_dict()["texture_pages"]),
    }
    assert set(payload["format_status"]) == {
        "live2d_moc3_v4_00",
        "spine_4_2",
    }
    assert b"primitive_candidates" not in rig_report_bytes(report)


def test_projection_validators_reject_hand_edited_but_valid_json(tmp_path: Path) -> None:
    *_inputs, rig = _build(tmp_path)
    projection = project_motion_manifest(rig)
    changed = projection.to_dict()
    changed["default_clip"] = "head_nod"
    tampered = MotionManifestProjection.from_payload(changed)

    with pytest.raises(ProjectionError) as exc_info:
        validate_motion_manifest_projection(tampered, rig)

    assert exc_info.value.code == "input_contract_mismatch"
    assert motion_manifest_bytes(projection) != motion_manifest_bytes(tampered)
