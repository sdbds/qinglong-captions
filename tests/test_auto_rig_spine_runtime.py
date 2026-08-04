from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from module.auto_rig.export.spine.runtime_validator import (
    SPINE_RUNTIME_VALIDATOR_PROTOCOL_DIGEST,
    SpineRuntimeValidationError,
    validate_spine_runtime_bundle,
)
from tests.test_auto_rig_stage_d import _run_stage_d


def _write_bundle(root: Path) -> tuple[Path, Path, Path]:
    executable = root / "spine-runtime.exe"
    executable.write_bytes(b"runtime")
    skeleton = root / "skeleton.json"
    skeleton.write_text(
        json.dumps(
            {
                "skeleton": {"spine": "4.2"},
                "bones": [{"name": "root"}],
                "slots": [{"name": "slot", "bone": "root"}],
                "skins": [{"name": "default", "attachments": {}}],
                "animations": {"idle": {}},
            }
        ),
        encoding="utf-8",
    )
    atlas = root / "skeleton.atlas"
    atlas.write_text("page.png\nsize: 16,16\n\nregion\nbounds: 0,0,1,1\n", encoding="utf-8")
    (root / "page.png").write_bytes(b"png")
    return executable, skeleton, atlas


def _harness_report(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "auto-rig-spine-runtime-v3",
        "validator_protocol_digest": SPINE_RUNTIME_VALIDATOR_PROTOCOL_DIGEST,
        "runtime_version": "4.2",
        "skeleton_version": "4.2",
        "sample_rate_hz": 60,
        "bone_count": 1,
        "slot_count": 1,
        "skin_count": 1,
        "setup_attachment_count": 1,
        "animation_count": 1,
        "atlas_page_count": 1,
        "atlas_region_count": 1,
        "animations": [
            {
                "name": "idle",
                "duration": 1.0,
                "sample_count": 61,
                "finite": True,
                "visible_change": True,
                "maximum_numeric_state_delta": 10.0,
                "maximum_world_vertex_displacement": 8.0,
                "maximum_world_vertex_displacement_ratio": 0.02,
                "maximum_slot_alpha_delta": 1.0,
                "maximum_setup_restore_residual": 0.0,
            }
        ],
    }
    payload.update(updates)
    return payload


def _patch_harness(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
    *,
    returncode: int = 0,
) -> None:
    def run(command, **_kwargs):
        report_path = Path(command[command.index("--report") + 1])
        report_path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(
            command,
            returncode,
            stdout="",
            stderr="runtime rejected input" if returncode else "",
        )

    monkeypatch.setattr(subprocess, "run", run)


def test_runtime_validator_accepts_exact_official_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable, skeleton, atlas = _write_bundle(tmp_path)
    _patch_harness(monkeypatch, _harness_report())

    report = validate_spine_runtime_bundle(
        executable,
        skeleton,
        atlas,
        expected_animation_names=("idle",),
    )

    assert report.validated is True
    assert report.runtime_version == "4.2"
    assert report.skeleton_version == "4.2"
    assert report.animation_count == 1
    assert report.animations[0].visible_change is True
    assert report.animations[0].maximum_world_vertex_displacement == 8.0
    assert report.report_sha256.startswith("sha256:")
    assert "executable_path" not in report.to_dict()


def test_runtime_validator_accepts_scale_normalized_breath_below_legacy_pixel_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable, skeleton, atlas = _write_bundle(tmp_path)
    report = _harness_report()
    animation = dict(report["animations"][0])
    animation.update(
        {
            "name": "breath",
            "maximum_world_vertex_displacement": 3.75,
            "maximum_world_vertex_displacement_ratio": 0.015,
            "maximum_slot_alpha_delta": 0.0,
        }
    )
    report["animations"] = [animation]
    _patch_harness(monkeypatch, report)

    validated = validate_spine_runtime_bundle(
        executable,
        skeleton,
        atlas,
        expected_animation_names=("breath",),
    )

    assert validated.animations[0].maximum_world_vertex_displacement == 3.75
    assert validated.animations[0].maximum_world_vertex_displacement_ratio == 0.015


def test_runtime_validator_accepts_opacity_driven_talk_without_geometry_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable, skeleton, atlas = _write_bundle(tmp_path)
    report = _harness_report()
    animation = dict(report["animations"][0])
    animation.update(
        {
            "name": "talk",
            "maximum_world_vertex_displacement": 0.0,
            "maximum_world_vertex_displacement_ratio": 0.0,
            "maximum_slot_alpha_delta": 1.0,
        }
    )
    report["animations"] = [animation]
    _patch_harness(monkeypatch, report)

    validated = validate_spine_runtime_bundle(
        executable,
        skeleton,
        atlas,
        expected_animation_names=("talk",),
    )

    assert validated.animations[0].maximum_slot_alpha_delta == 1.0


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"runtime_version": "4.3"}, "version"),
        (
            {
                "animations": [
                    {
                        "name": "idle",
                        "duration": 1.0,
                        "sample_count": 61,
                        "finite": True,
                        "visible_change": False,
                        "maximum_numeric_state_delta": 0.0,
                        "maximum_world_vertex_displacement": 0.0,
                        "maximum_world_vertex_displacement_ratio": 0.0,
                        "maximum_slot_alpha_delta": 0.0,
                        "maximum_setup_restore_residual": 0.0,
                    }
                ]
            },
            "visible",
        ),
        ({"animations": []}, "animation"),
    ],
)
def test_runtime_validator_rejects_false_positive_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, object],
    message: str,
) -> None:
    executable, skeleton, atlas = _write_bundle(tmp_path)
    _patch_harness(monkeypatch, _harness_report(**updates))

    with pytest.raises(SpineRuntimeValidationError, match=message):
        validate_spine_runtime_bundle(
            executable,
            skeleton,
            atlas,
            expected_animation_names=("idle",),
        )


def test_runtime_validator_surfaces_native_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable, skeleton, atlas = _write_bundle(tmp_path)
    _patch_harness(monkeypatch, _harness_report(), returncode=1)

    with pytest.raises(SpineRuntimeValidationError, match="runtime rejected input"):
        validate_spine_runtime_bundle(
            executable,
            skeleton,
            atlas,
            expected_animation_names=("idle",),
        )


@pytest.mark.parametrize(
    (
        "name",
        "maximum_world_vertex_displacement_ratio",
        "maximum_slot_alpha_delta",
    ),
    (
        ("breath", 0.00499, 0.0),
        ("talk", 0.00199, 0.79),
    ),
)
def test_runtime_validator_rejects_imperceptible_semantic_motion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    maximum_world_vertex_displacement_ratio: float,
    maximum_slot_alpha_delta: float,
) -> None:
    executable, skeleton, atlas = _write_bundle(tmp_path)
    report = _harness_report()
    animation = dict(report["animations"][0])
    animation.update(
        {
            "name": name,
            "maximum_world_vertex_displacement_ratio": (maximum_world_vertex_displacement_ratio),
            "maximum_slot_alpha_delta": maximum_slot_alpha_delta,
        }
    )
    report["animations"] = [animation]
    _patch_harness(monkeypatch, report)

    with pytest.raises(SpineRuntimeValidationError, match="semantic motion threshold"):
        validate_spine_runtime_bundle(
            executable,
            skeleton,
            atlas,
            expected_animation_names=(name,),
        )


@pytest.mark.optional_runtime
def test_official_spine_42_runtime_executes_every_exported_animation(
    tmp_path: Path,
) -> None:
    executable = os.environ.get("SPINE_RUNTIME_VALIDATOR_PATH")
    if not executable:
        pytest.skip("SPINE_RUNTIME_VALIDATOR_PATH is not configured")
    _stage_c, stage_d = _run_stage_d(tmp_path)
    bundle = tmp_path / "rig" / "spine"

    report = validate_spine_runtime_bundle(
        executable,
        bundle / "skeleton.json",
        bundle / "skeleton.atlas",
        expected_animation_names=tuple(record.artifact_export_name for record in stage_d.animation_plan.records),
    )

    assert report.validated is True
    assert {record.name for record in report.animations} == {
        record.artifact_export_name for record in stage_d.animation_plan.records
    }
    assert all(record.finite for record in report.animations)
    assert all(record.visible_change for record in report.animations)
