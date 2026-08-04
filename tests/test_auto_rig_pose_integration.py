from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from module.auto_rig.joint_pipeline import build_stage_a_joint_plan
from module.auto_rig.overrides import load_rig_override_source, validate_rig_override_source
from module.auto_rig.pose.contracts import RawPoseResult, ScoredKeypoint
from module.auto_rig.pose.integration import execute_pose_observation
from tests.test_auto_rig_anatomy import _part
from tests.test_auto_rig_limb_geometry import _anatomy, _bent_arm_points, _torso
from tests.test_auto_rig_overrides import _target


class _FakeBodyProvider:
    def __init__(self, backend: str = "torch-sdpa") -> None:
        self.calls = 0
        self.runtime_report = SimpleNamespace(
            backend=backend,
            device="cuda:0",
            dtype="float16",
            bundle_sha256="a" * 64,
            latent_seed=0,
            input_size=(768, 1024),
            bbox_padding=1.25,
            fallback_reason=None,
        )

    def infer(self, image: Image.Image, *, person_bbox):
        self.calls += 1
        points = [ScoredKeypoint(75, 150, 0.9) for _ in range(17)]
        points[5] = ScoredKeypoint(90, 115, 0.9)
        points[6] = ScoredKeypoint(60, 115, 0.9)
        points[7] = ScoredKeypoint(90, 140, 0.9)
        points[8] = ScoredKeypoint(60, 140, 0.9)
        points[9] = ScoredKeypoint(75, 165, 0.9)
        points[10] = ScoredKeypoint(75, 165, 0.9)
        return RawPoseResult(
            provider_id="sdpose-body17",
            provider_version="sdpose-body17-test",
            layout="coco17",
            canvas_width=image.width,
            canvas_height=image.height,
            keypoints=tuple(points),
        )


class _FailingBodyProvider:
    def __init__(self, message: str) -> None:
        self.message = message

    def infer(self, image: Image.Image, *, person_bbox):
        raise FileNotFoundError(self.message)


def _inputs(tmp_path, *, merged: bool):
    target = _target(tmp_path)
    overrides = validate_rig_override_source(load_rig_override_source(tmp_path), target)
    anatomy = _anatomy(
        tmp_path,
        _torso(),
        _part(
            "handwear",
            side=None if merged else "xmin",
            xyxy=(50, 100, 102, 180),
            points=_bent_arm_points(),
        ),
    )
    geometry = build_stage_a_joint_plan(
        anatomy,
        target=target,
        overrides=overrides,
    )
    return target, overrides, anatomy, geometry


def test_auto_pose_runs_only_for_merged_unresolved_limbs_and_fuses_valid_points(
    tmp_path,
) -> None:
    target, overrides, anatomy, geometry = _inputs(tmp_path, merged=True)
    provider = _FakeBodyProvider()

    execution = execute_pose_observation(
        mode="auto",
        anatomy=anatomy,
        geometry_plan=geometry,
        target=target,
        overrides=overrides,
        image=Image.new("RGB", (768, 768), "white"),
        providers={"sdpose": provider},
    )

    wrist = next(item for item in execution.joint_plan.joints.resolutions if item.joint_id == "joint/wrist.xmin")
    assert provider.calls == 1
    assert execution.selected_provider_id == "sdpose-body17"
    assert wrist.status == "resolved"
    assert wrist.source == "pose"
    assert execution.report["trigger"]["reason"] == "merged_limb_unresolved_hinge"


def test_auto_pose_does_not_call_the_provider_when_geometry_is_sufficient(tmp_path) -> None:
    target, overrides, anatomy, geometry = _inputs(tmp_path, merged=False)
    provider = _FakeBodyProvider()

    execution = execute_pose_observation(
        mode="auto",
        anatomy=anatomy,
        geometry_plan=geometry,
        target=target,
        overrides=overrides,
        image=Image.new("RGB", (768, 768), "white"),
        providers={"sdpose": provider},
    )

    assert provider.calls == 0
    assert execution.joint_plan is geometry
    assert execution.selected_provider_id is None


def test_exercised_runtime_backend_changes_the_pose_and_stage_a_identity(tmp_path) -> None:
    target, overrides, anatomy, geometry = _inputs(tmp_path, merged=True)
    common = dict(
        mode="sdpose",
        anatomy=anatomy,
        geometry_plan=geometry,
        target=target,
        overrides=overrides,
        image=Image.new("RGB", (768, 768), "white"),
    )

    sdpa = execute_pose_observation(
        **common,
        providers={"sdpose": _FakeBodyProvider("torch-sdpa")},
    )
    fa2 = execute_pose_observation(
        **common,
        providers={"sdpose": _FakeBodyProvider("torch-fa2")},
    )

    assert sdpa.joint_plan.pose.provider_fingerprint != fa2.joint_plan.pose.provider_fingerprint
    assert sdpa.joint_plan.plan_sha256 != fa2.joint_plan.plan_sha256


def test_auto_provider_failure_report_does_not_persist_environment_specific_paths(
    tmp_path,
) -> None:
    target, overrides, anatomy, geometry = _inputs(tmp_path, merged=True)
    common = dict(
        mode="auto",
        anatomy=anatomy,
        geometry_plan=geometry,
        target=target,
        overrides=overrides,
        image=Image.new("RGB", (768, 768), "white"),
    )

    first = execute_pose_observation(
        **common,
        providers={"sdpose": _FailingBodyProvider(r"C:\first\model.safetensors")},
    )
    second = execute_pose_observation(
        **common,
        providers={"sdpose": _FailingBodyProvider(r"D:\second\model.safetensors")},
    )

    assert first.report == second.report
    assert first.report["provider_runs"] == [
        {
            "provider_key": "sdpose",
            "status": "failed",
            "failure_code": "pose_provider_failed",
            "error_type": "FileNotFoundError",
            "selected": False,
        }
    ]
