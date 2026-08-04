from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file
from torch import nn

from module.auto_rig.pose.artifacts import PoseArtifactError
from module.auto_rig.pose.detrpose import (
    DETRPoseLoadedModel,
    DETRPoseXProvider,
    build_detrpose_raw_result,
    load_detrpose_state_dict,
    select_detrpose_instance,
)


def _person(center_x: float, center_y: float) -> np.ndarray:
    offsets = np.linspace(-6.0, 6.0, 14, dtype=np.float32)
    return np.stack((center_x + offsets, center_y + offsets), axis=-1)


def test_detrpose_person_selection_prefers_the_target_character_not_raw_score() -> None:
    keypoints = np.stack((_person(900, 900), _person(110, 130)))
    scores = np.array([0.98, 0.72], dtype=np.float32)

    selected = select_detrpose_instance(
        scores,
        keypoints,
        person_bbox=(50, 40, 180, 220),
    )

    assert selected == 1


def test_detrpose_selection_uses_score_then_stable_index_for_equal_geometry() -> None:
    same = _person(100, 100)
    keypoints = np.stack((same, same, same))

    assert (
        select_detrpose_instance(
            np.array([0.4, 0.9, 0.9], dtype=np.float32),
            keypoints,
            person_bbox=(50, 50, 150, 150),
        )
        == 1
    )


def test_detrpose_raw_result_broadcasts_instance_confidence_without_inventing_joint_scores() -> None:
    keypoints = np.stack((_person(100, 120), _person(400, 500)))
    result = build_detrpose_raw_result(
        scores=np.array([0.8, 0.2], dtype=np.float32),
        keypoints=keypoints,
        person_bbox=(50, 50, 180, 220),
        canvas_size=(768, 768),
        provider_version="detrpose-x-crowdpose-v1:test",
    )

    assert result.provider_id == "detrpose-x-crowdpose"
    assert result.layout == "crowdpose14"
    assert len(result.keypoints) == 14
    assert [point.score for point in result.keypoints] == pytest.approx([0.8] * 14)
    assert result.keypoints[13].x == pytest.approx(keypoints[0, 13, 0])


def test_detrpose_selection_rejects_malformed_or_nonfinite_predictions() -> None:
    with pytest.raises(ValueError, match="keypoints"):
        select_detrpose_instance(
            np.array([0.9], dtype=np.float32),
            np.full((1, 14, 2), np.nan, dtype=np.float32),
            person_bbox=(0, 0, 100, 100),
        )


class _FakeDETRCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input: torch.Tensor | None = None

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        self.input = image.detach().clone()
        return {"token": torch.tensor(1, device=image.device)}


class _FakeDETRPostprocessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.orig_target_sizes: torch.Tensor | None = None

    def forward(
        self,
        output: dict[str, torch.Tensor],
        orig_target_sizes: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        assert int(output["token"]) == 1
        self.orig_target_sizes = orig_target_sizes.detach().clone()
        scores = torch.tensor([[0.9, 0.2]], device=orig_target_sizes.device)
        labels = torch.tensor([[0, 0]], device=orig_target_sizes.device)
        first = torch.tensor(_person(320, 320), device=orig_target_sizes.device)
        second = torch.tensor(_person(40, 40), device=orig_target_sizes.device)
        return [scores[0]], [labels[0]], [torch.stack((first, second))]


class _FakeDETRModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeDETRCore()
        self.postprocessor = _FakeDETRPostprocessor()


def test_detrpose_provider_uses_the_shared_crop_and_restores_canvas_coordinates() -> None:
    model = _FakeDETRModel()
    provider = DETRPoseXProvider(
        DETRPoseLoadedModel(
            model=model,
            device=torch.device("cpu"),
            weights_sha256="a" * 64,
            code_revision="b" * 40,
        ),
        bbox_padding=1.0,
        input_size=(640, 640),
        instance_score_threshold=0.3,
    )

    result = provider.infer(
        Image.new("RGBA", (200, 100), (0, 0, 0, 0)),
        person_bbox=(20, 10, 180, 90),
    )

    assert result.layout == "crowdpose14"
    assert (result.canvas_width, result.canvas_height) == (200, 100)
    assert result.keypoints[0].x == pytest.approx(98.5, abs=0.01)
    assert result.keypoints[0].y == pytest.approx(49.25, abs=0.01)
    assert model.model.input is not None
    assert model.model.input.shape == (1, 3, 640, 640)
    assert model.model.input[0, :, 0, 0].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert model.postprocessor.orig_target_sizes is not None
    assert model.postprocessor.orig_target_sizes.tolist() == [[640, 640]]
    assert provider.runtime_report.backend == "torch-cpu"
    assert provider.runtime_report.device == "cpu"
    assert provider.runtime_report.dtype == "float32"


def test_detrpose_provider_rejects_all_predictions_below_the_instance_gate() -> None:
    class _LowScorePostprocessor(_FakeDETRPostprocessor):
        def forward(self, output, orig_target_sizes):
            scores, labels, keypoints = super().forward(output, orig_target_sizes)
            return [torch.tensor([0.2, 0.1])], labels, keypoints

    model = _FakeDETRModel()
    model.postprocessor = _LowScorePostprocessor()
    provider = DETRPoseXProvider(
        DETRPoseLoadedModel(
            model=model,
            device=torch.device("cpu"),
            weights_sha256="a" * 64,
            code_revision="b" * 40,
        ),
        instance_score_threshold=0.3,
    )

    with pytest.raises(RuntimeError, match="confidence"):
        provider.infer(
            Image.new("RGB", (100, 100), "white"),
            person_bbox=(10, 10, 90, 90),
        )


def test_detrpose_safetensors_are_loaded_strictly_without_the_upstream_resume_bug(
    tmp_path,
) -> None:
    module = nn.Linear(2, 1, bias=False)
    path = tmp_path / "model.safetensors"
    save_file({"weight": torch.tensor([[2.0, 3.0]])}, str(path))

    load_detrpose_state_dict(module, path)

    assert module.weight.detach().tolist() == [[2.0, 3.0]]


def test_detrpose_strict_loader_rejects_an_incomplete_checkpoint(tmp_path) -> None:
    module = nn.Linear(2, 1, bias=True)
    path = tmp_path / "model.safetensors"
    save_file({"weight": torch.tensor([[2.0, 3.0]])}, str(path))

    with pytest.raises(PoseArtifactError, match="inventory"):
        load_detrpose_state_dict(module, path)
