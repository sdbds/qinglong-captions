from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from module.auto_rig.pipeline import AutoRigPipelineError, run_auto_rig_item
from module.auto_rig.pose.contracts import RawPoseResult, ScoredKeypoint
from tests.test_auto_rig_contracts import _json_bytes, _write_item, _write_psd_payload


def _write_pipeline_item(root: Path) -> None:
    _write_item(
        root,
        edge=768,
        tags=("face", "neck", "topwear"),
        mode="psd",
    )
    parts = {
        "face": {
            "tag": "face",
            "xyxy": [100, 40, 140, 92],
            "depth_median": 0.2,
        },
        "neck": {
            "tag": "neck",
            "xyxy": [115, 88, 125, 106],
            "depth_median": 0.4,
        },
        "topwear": {
            "tag": "topwear",
            "xyxy": [90, 102, 150, 172],
            "depth_median": 0.6,
        },
    }
    (root / "optimized" / "info.json").write_bytes(_json_bytes({"frame_size": [768, 768], "parts": parts}))
    _write_psd_payload(root, 768, parts)


def _write_open_mouth_pipeline_item(root: Path) -> None:
    parts = _write_item(
        root,
        edge=768,
        tags=("face", "mouth", "neck", "topwear"),
        mode="png",
    )
    parts.update(
        {
            "face": {"tag": "face", "xyxy": [100, 40, 140, 92], "depth_median": 0.2},
            "mouth": {"tag": "mouth", "xyxy": [108, 70, 132, 82], "depth_median": 0.1},
            "neck": {"tag": "neck", "xyxy": [115, 88, 125, 106], "depth_median": 0.4},
            "topwear": {"tag": "topwear", "xyxy": [90, 102, 150, 172], "depth_median": 0.6},
        }
    )
    (root / "optimized" / "info.json").write_bytes(_json_bytes({"frame_size": [768, 768], "parts": parts}))
    for tag, part in parts.items():
        x1, y1, x2, y2 = part["xyxy"]
        size = (x2 - x1, y2 - y1)
        image = Image.new("RGBA", size, (180, 120, 120, 255))
        if tag == "mouth":
            image = Image.new("RGBA", size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.ellipse((2, 1, size[0] - 3, size[1] - 2), fill=(105, 30, 45, 255))
            draw.ellipse((6, 3, size[0] - 7, size[1] - 4), fill=(0, 0, 0, 0))
        image.save(root / "optimized" / f"{tag}.png")
        Image.new("L", size, 128).save(root / "optimized" / f"{tag}_depth.png")


def _write_merged_limb_pipeline_item(root: Path) -> None:
    _write_item(
        root,
        edge=768,
        tags=("face", "neck", "topwear", "handwear"),
        mode="psd",
    )
    parts = {
        "face": {"tag": "face", "xyxy": [100, 40, 140, 92], "depth_median": 0.2},
        "neck": {"tag": "neck", "xyxy": [115, 88, 125, 106], "depth_median": 0.4},
        "topwear": {"tag": "topwear", "xyxy": [90, 102, 150, 172], "depth_median": 0.6},
        "handwear": {"tag": "handwear", "xyxy": [50, 100, 190, 180], "depth_median": 0.5},
    }
    (root / "optimized" / "info.json").write_bytes(_json_bytes({"frame_size": [768, 768], "parts": parts}))
    _write_psd_payload(root, 768, parts)


class _PipelinePoseProvider:
    runtime_report = SimpleNamespace(
        backend="torch-sdpa",
        device="cuda:0",
        dtype="float16",
        bundle_sha256="a" * 64,
        latent_seed=0,
        input_size=(768, 1024),
        bbox_padding=1.25,
        fallback_reason=None,
    )

    def __init__(self) -> None:
        self.calls = 0

    def infer(self, image, *, person_bbox):
        self.calls += 1
        points = [ScoredKeypoint(120, 120, 0.9) for _ in range(17)]
        points[5] = ScoredKeypoint(150, 110, 0.9)
        points[6] = ScoredKeypoint(90, 110, 0.9)
        points[7] = ScoredKeypoint(170, 140, 0.9)
        points[8] = ScoredKeypoint(70, 140, 0.9)
        points[9] = ScoredKeypoint(185, 170, 0.9)
        points[10] = ScoredKeypoint(55, 170, 0.9)
        return RawPoseResult(
            provider_id="sdpose-body17",
            provider_version="sdpose-pipeline-fixture-v1",
            layout="coco17",
            canvas_width=image.width,
            canvas_height=image.height,
            keypoints=tuple(points),
        )


def test_run_auto_rig_item_executes_real_psd_through_structural_stage_e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_pipeline_item(tmp_path)
    runtime = tmp_path / "spine-runtime.exe"
    runtime.write_bytes(b"official-runtime-probe")
    evidence = {
        "validated": True,
        "validator_version": "spine-runtime-validator-v3",
        "runtime_version": "4.2",
        "animation_count": 1,
        "report_sha256": "sha256:" + ("2" * 64),
    }
    monkeypatch.setattr(
        "module.auto_rig.stage_d.validate_spine_runtime_bundle",
        lambda *_args, **_kwargs: SimpleNamespace(to_dict=lambda: evidence),
    )

    result = run_auto_rig_item(
        tmp_path,
        profile_id="dual_runtime_core_v1",
        validation_tier="structural",
        spine_runtime_path=runtime,
        finalize=False,
    )

    assert tuple(result.stage_manifests) == ("A", "B", "C", "D", "E")
    assert result.terminal is None
    assert (tmp_path / "rig" / "rig.json").is_file()
    assert (tmp_path / "rig" / "spine" / "skeleton.json").is_file()
    assert (tmp_path / "rig" / "live2d" / "model.moc3").is_file()
    report = json.loads((tmp_path / "rig" / "report.json").read_text(encoding="utf-8"))
    assert report["profile"] == "dual_runtime_core_v1"
    assert result.stage_d.report["official_spine_runtime_gate"] == {
        "status": "passed",
        "validation": evidence,
    }


def test_spine_dev_pipeline_stops_at_d_without_live2d_or_external_runtime(
    tmp_path: Path,
) -> None:
    _write_pipeline_item(tmp_path)
    stale_live2d = tmp_path / "rig" / "live2d" / "stale.bin"
    stale_live2d.parent.mkdir(parents=True, exist_ok=True)
    stale_live2d.write_bytes(b"stale")

    result = run_auto_rig_item(
        tmp_path,
        profile_id="spine_4_2_dev",
        validation_tier="structural",
        finalize=False,
        pose_mode="disabled",
    )

    assert tuple(result.stage_manifests) == ("A", "B", "C", "D")
    assert result.stage_d is not None
    assert result.stage_d.report["official_spine_runtime_gate"]["status"] == "not_run"
    assert result.stage_e is None
    assert (tmp_path / "rig" / "spine" / "skeleton.json").is_file()
    assert not stale_live2d.exists()
    assert not any((tmp_path / "rig" / "live2d").rglob("*"))
    assert result.terminal is None


def test_open_mouth_crossfade_keeps_a_geometric_opening_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_open_mouth_pipeline_item(tmp_path)
    runtime = tmp_path / "spine-runtime.exe"
    runtime.write_bytes(b"official-runtime-probe")
    monkeypatch.setattr(
        "module.auto_rig.stage_d.validate_spine_runtime_bundle",
        lambda *_args, **_kwargs: SimpleNamespace(
            to_dict=lambda: {
                "validated": True,
                "validator_version": "spine-runtime-validator-v3",
                "runtime_version": "4.2",
                "animation_count": 1,
                "report_sha256": "sha256:" + ("2" * 64),
            }
        ),
    )

    run_auto_rig_item(
        tmp_path,
        profile_id="dual_runtime_core_v1",
        validation_tier="structural",
        spine_runtime_path=runtime,
        finalize=False,
    )

    skeleton = json.loads((tmp_path / "rig" / "spine" / "skeleton.json").read_text(encoding="utf-8"))
    talk = skeleton["animations"]["talk"]
    assert "slots" in talk
    assert "attachments" in talk


def test_pipeline_injected_pose_provider_changes_stage_a_and_emits_limb_joints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_merged_limb_pipeline_item(tmp_path)
    runtime = tmp_path / "spine-runtime.exe"
    runtime.write_bytes(b"official-runtime-probe")
    monkeypatch.setattr(
        "module.auto_rig.stage_d.validate_spine_runtime_bundle",
        lambda *_args, **_kwargs: SimpleNamespace(
            to_dict=lambda: {
                "validated": True,
                "validator_version": "spine-runtime-validator-v3",
                "runtime_version": "4.2",
                "animation_count": 1,
                "report_sha256": "sha256:" + ("2" * 64),
            }
        ),
    )

    disabled = run_auto_rig_item(
        tmp_path,
        validation_tier="structural",
        spine_runtime_path=runtime,
        finalize=False,
        pose_mode="disabled",
    )
    disabled_a = disabled.stage_manifests["A"].stage_fingerprint
    enabled = run_auto_rig_item(
        tmp_path,
        validation_tier="structural",
        spine_runtime_path=runtime,
        finalize=False,
        pose_mode="sdpose",
        pose_providers={"sdpose": _PipelinePoseProvider()},
    )

    assert enabled.stage_manifests["A"].stage_fingerprint != disabled_a
    by_joint = {item.joint_id: item for item in enabled.geometry_cache.joint_plan.joints.resolutions}
    assert by_joint["joint/elbow.xmin"].source == "pose"
    pose_report = json.loads((tmp_path / "rig" / "cache" / "A" / "pose_report.json").read_text(encoding="utf-8"))
    assert pose_report["selected_provider_id"] == "sdpose-body17"


def test_pipeline_reuses_validated_a_to_e_without_reloading_pose_or_rewriting_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_merged_limb_pipeline_item(tmp_path)
    runtime = tmp_path / "spine-runtime.exe"
    runtime.write_bytes(b"official-runtime-probe")
    monkeypatch.setattr(
        "module.auto_rig.stage_d.validate_spine_runtime_bundle",
        lambda *_args, **_kwargs: SimpleNamespace(
            to_dict=lambda: {
                "validated": True,
                "validator_version": "spine-runtime-validator-v3",
                "runtime_version": "4.2",
                "animation_count": 1,
                "report_sha256": "sha256:" + ("2" * 64),
            }
        ),
    )
    provider = _PipelinePoseProvider()
    first = run_auto_rig_item(
        tmp_path,
        validation_tier="structural",
        spine_runtime_path=runtime,
        finalize=False,
        pose_mode="sdpose",
        pose_providers={"sdpose": provider},
    )
    output_mtimes = {
        record.path: (tmp_path / Path(*record.path.split("/"))).stat().st_mtime_ns
        for stage in ("A", "B", "C", "D", "E")
        for record in first.stage_manifests[stage].output_file_sha256
    }

    second = run_auto_rig_item(
        tmp_path,
        validation_tier="structural",
        spine_runtime_path=runtime,
        finalize=False,
        pose_mode="sdpose",
        pose_providers={"sdpose": provider},
    )

    assert provider.calls == 1
    assert second.reused_stages == ("A", "B", "C", "D", "E")
    assert {path: (tmp_path / Path(*path.split("/"))).stat().st_mtime_ns for path in output_mtimes} == output_mtimes


def test_pipeline_does_not_hide_a_cache_loss_by_rematerializing_before_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_merged_limb_pipeline_item(tmp_path)
    runtime = tmp_path / "spine-runtime.exe"
    runtime.write_bytes(b"official-runtime-probe")
    monkeypatch.setattr(
        "module.auto_rig.stage_d.validate_spine_runtime_bundle",
        lambda *_args, **_kwargs: SimpleNamespace(
            to_dict=lambda: {
                "validated": True,
                "validator_version": "spine-runtime-validator-v3",
                "runtime_version": "4.2",
                "animation_count": 1,
                "report_sha256": "sha256:" + ("2" * 64),
            }
        ),
    )
    provider = _PipelinePoseProvider()
    first = run_auto_rig_item(
        tmp_path,
        validation_tier="structural",
        spine_runtime_path=runtime,
        finalize=False,
        pose_mode="sdpose",
        pose_providers={"sdpose": provider},
    )
    qcl = next(record for record in first.stage_manifests["A"].output_file_sha256 if record.path.endswith(".qcl"))
    (tmp_path / Path(*qcl.path.split("/"))).unlink()

    second = run_auto_rig_item(
        tmp_path,
        validation_tier="structural",
        spine_runtime_path=runtime,
        finalize=False,
        pose_mode="sdpose",
        pose_providers={"sdpose": provider},
    )

    assert provider.calls == 2
    assert "A" not in second.reused_stages


def test_pipeline_failure_publishes_one_public_terminal_error(
    tmp_path: Path,
) -> None:
    _write_merged_limb_pipeline_item(tmp_path)

    class FailingPoseProvider(_PipelinePoseProvider):
        def infer(self, image, *, person_bbox):
            self.calls += 1
            raise RuntimeError("fixture pose failure")

    with pytest.raises(RuntimeError, match="fixture pose failure"):
        run_auto_rig_item(
            tmp_path,
            validation_tier="structural",
            finalize=False,
            pose_mode="sdpose",
            pose_providers={"sdpose": FailingPoseProvider()},
        )

    error = json.loads((tmp_path / "rig" / "error.json").read_text(encoding="utf-8"))
    assert error["terminal_state"] == "failed"
    assert error["failed_stages"] == ["A"]
    assert (tmp_path / "rig" / "cache" / "A" / "failure.json").is_file()
    assert (tmp_path / "rig" / "cache" / "G" / "manifest.json").is_file()
    assert not (tmp_path / "rig" / "export_manifest.json").exists()
    assert not (tmp_path / "rig" / "cache" / "A" / "manifest.json").exists()


def test_invalid_pipeline_configuration_fails_before_item_failure_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_pipeline_item(tmp_path)
    publication_attempted = False

    def fail_if_called(*_args, **_kwargs) -> None:
        nonlocal publication_attempted
        publication_attempted = True

    monkeypatch.setattr(
        "module.auto_rig.pipeline._publish_pipeline_failure",
        fail_if_called,
    )

    with pytest.raises(AutoRigPipelineError, match="validation_tier"):
        run_auto_rig_item(
            tmp_path,
            validation_tier="invalid",
            finalize=False,
        )
    with pytest.raises(AutoRigPipelineError, match="terminal dual-runtime"):
        run_auto_rig_item(
            tmp_path,
            profile_id="spine_4_2_dev",
            validation_tier="release",
            finalize=True,
        )

    assert publication_attempted is False
    assert not (tmp_path / "rig" / "error.json").exists()


def test_pipeline_does_not_swallow_terminal_failure_publication_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = RuntimeError("fixture item failure")

    def fail_item(*_args, **_kwargs):
        raise original

    def fail_terminal(*_args, **_kwargs):
        raise OSError("fixture terminal failure")

    monkeypatch.setattr(
        "module.auto_rig.pipeline._run_auto_rig_item_impl",
        fail_item,
    )
    monkeypatch.setattr(
        "module.auto_rig.pipeline._publish_pipeline_failure",
        fail_terminal,
    )

    with pytest.raises(AutoRigPipelineError, match="could not be published") as exc_info:
        run_auto_rig_item(
            tmp_path,
            validation_tier="structural",
            finalize=False,
        )

    assert exc_info.value.__cause__ is original
