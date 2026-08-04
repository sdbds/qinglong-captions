from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from module.auto_rig.pose.attention import FlashAttentionBackendError
from module.auto_rig.pose.heatmap import (
    SDPoseBodyHeatmapHead,
    decode_udp_heatmaps,
)
from module.auto_rig.pose.preprocess import (
    PoseCropTransform,
    preprocess_sdpose_image,
)
from module.auto_rig.pose.sdpose import (
    SDPoseBodyProvider,
    SDPoseLoadedModels,
    sdpose_task_embedding,
)


def test_body_heatmap_head_matches_the_pinned_decoder_tensor_contract() -> None:
    head = SDPoseBodyHeatmapHead()

    assert {name: tuple(value.shape) for name, value in head.state_dict().items()} == {
        "conv_layers.0.bias": (320,),
        "conv_layers.0.weight": (320, 320, 1, 1),
        "deconv_layers.0.weight": (320, 320, 4, 4),
        "final_layer.bias": (17,),
        "final_layer.weight": (17, 320, 1, 1),
    }
    output = head(torch.zeros((1, 320, 128, 96), dtype=torch.float32))
    assert output.shape == (1, 17, 256, 192)


def test_udp_decoder_uses_the_unbiased_endpoint_scale_and_preserves_scores() -> None:
    heatmaps = np.zeros((2, 5, 4), dtype=np.float32)
    heatmaps[0, 3, 2] = 0.75
    heatmaps[1, 1, 1] = 0.4

    keypoints, scores = decode_udp_heatmaps(
        heatmaps,
        input_size=(12, 20),
        blur_kernel_size=3,
    )

    assert keypoints.shape == (2, 2)
    assert scores.tolist() == pytest.approx([0.75, 0.4])
    assert keypoints[0].tolist() == pytest.approx([8.0, 15.0], abs=1e-4)
    assert keypoints[1].tolist() == pytest.approx([4.0, 5.0], abs=1e-4)


def test_udp_dark_refinement_recovers_a_subpixel_gaussian_peak() -> None:
    height, width = 32, 24
    center_x, center_y = 10.35, 18.65
    yy, xx = np.mgrid[:height, :width]
    heatmap = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * 2.0**2)).astype(np.float32)[None]

    keypoints, scores = decode_udp_heatmaps(
        heatmap,
        input_size=(width - 1, height - 1),
        blur_kernel_size=3,
    )

    assert scores[0] > 0.95
    assert keypoints[0].tolist() == pytest.approx([center_x, center_y], abs=0.12)


def test_udp_decoder_rejects_wrong_body_heatmap_shape() -> None:
    with pytest.raises(ValueError, match="heatmaps"):
        decode_udp_heatmaps(np.zeros((1, 17, 256, 192), dtype=np.float32))


def test_pose_crop_padding_clipping_and_coordinate_round_trip_are_explicit() -> None:
    transform = PoseCropTransform.from_bbox(
        canvas_size=(100, 80),
        person_bbox=(20, 10, 60, 50),
        padding=1.25,
    )

    assert transform.crop_box == (15, 5, 65, 55)
    assert transform.scale_x == pytest.approx(768 / 50)
    assert transform.scale_y == pytest.approx(1024 / 50)
    canvas_points = np.array([[15.0, 5.0], [40.25, 30.5], [65.0, 55.0]])
    restored = transform.model_to_canvas(transform.canvas_to_model(canvas_points))
    assert restored == pytest.approx(canvas_points, abs=1e-5)

    clipped = PoseCropTransform.from_bbox(
        canvas_size=(100, 80),
        person_bbox=(-10, -5, 20, 30),
        padding=1.25,
    )
    assert clipped.crop_box[0:2] == (0, 0)


def test_preprocess_composites_rgba_on_white_and_directly_stretches_the_crop() -> None:
    rgba = np.array(
        [
            [[0, 0, 0, 0], [255, 0, 0, 255]],
            [[0, 255, 0, 255], [0, 0, 255, 255]],
        ],
        dtype=np.uint8,
    )
    transform = PoseCropTransform.from_bbox(
        canvas_size=(2, 2),
        person_bbox=(0, 0, 2, 2),
        padding=1.0,
        input_size=(2, 2),
    )

    tensor = preprocess_sdpose_image(Image.fromarray(rgba, mode="RGBA"), transform)

    assert tensor.shape == (1, 3, 2, 2)
    assert tensor.dtype == torch.float32
    assert tensor[0, :, 0, 0].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert tensor[0, :, 0, 1].tolist() == pytest.approx([1.0, -1.0, -1.0])


def test_sdpose_task_embedding_matches_the_training_annotation_label() -> None:
    embedding = sdpose_task_embedding(2, device=torch.device("cpu"))

    assert embedding.shape == (2, 4)
    assert embedding[0].tolist() == pytest.approx([np.sin(1.0), 0.0, np.cos(1.0), 1.0])
    assert embedding[1].equal(embedding[0])


class _FakeLatentDistribution:
    def __init__(self, batch_size: int, owner: "_FakeVAE") -> None:
        self.batch_size = batch_size
        self.owner = owner

    def sample(self, *, generator: torch.Generator) -> torch.Tensor:
        self.owner.seed = generator.initial_seed()
        return torch.zeros((self.batch_size, 4, 2, 2), dtype=torch.float32)


class _FakeVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.input: torch.Tensor | None = None
        self.seed: int | None = None

    def encode(self, value: torch.Tensor):
        self.input = value.detach().clone()
        return type(
            "Encoded",
            (),
            {"latent_dist": _FakeLatentDistribution(value.shape[0], self)},
        )()


class _FeatureBlock(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _FakeUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.up_blocks = nn.ModuleList([_FeatureBlock(), _FeatureBlock()])
        self.call: dict[str, torch.Tensor | bool] | None = None
        self.call_count = 0
        self.processor = None

    def set_attn_processor(self, processor) -> None:
        self.processor = processor

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        class_labels: torch.Tensor,
        return_dict: bool,
    ):
        self.call_count += 1
        self.call = {
            "sample": sample.detach().clone(),
            "timestep": timestep.detach().clone(),
            "encoder_hidden_states": encoder_hidden_states.detach().clone(),
            "class_labels": class_labels.detach().clone(),
            "return_dict": return_dict,
        }
        features = self.up_blocks[-1](torch.zeros((sample.shape[0], 320, 2, 2), dtype=sample.dtype))
        return (features[:, :4],)


class _FakeDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.features: torch.Tensor | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.features = features.detach().clone()
        heatmaps = torch.zeros(
            (features.shape[0], 17, 4, 3),
            dtype=features.dtype,
        )
        heatmaps[:, :, 2, 1] = 0.9
        return heatmaps


def test_sdpose_provider_runs_fixed_seed_x0_inference_and_restores_canvas_coordinates() -> None:
    vae = _FakeVAE()
    unet = _FakeUNet()
    decoder = _FakeDecoder()
    provider = SDPoseBodyProvider(
        SDPoseLoadedModels(
            vae=vae,
            unet=unet,
            decoder=decoder,
            empty_prompt=torch.zeros((1, 2, 1024), dtype=torch.float32),
            device=torch.device("cpu"),
            dtype=torch.float32,
            backend="torch-cpu",
            bundle_sha256="a" * 64,
        ),
        latent_seed=123,
    )

    result = provider.infer(
        Image.new("RGB", (100, 80), "white"),
        person_bbox=(20, 10, 60, 70),
    )

    assert result.provider_id == "sdpose-body17"
    assert result.layout == "coco17"
    assert (result.canvas_width, result.canvas_height) == (100, 80)
    assert len(result.keypoints) == 17
    assert result.keypoints[0].score == pytest.approx(0.9)
    assert (result.keypoints[0].x, result.keypoints[0].y) == pytest.approx(
        (40.0, 52.6667),
        abs=0.02,
    )
    assert vae.input is not None and vae.input.shape == (1, 3, 1024, 768)
    assert vae.seed == 123
    assert unet.call is not None
    assert int(unet.call["timestep"]) == 999
    assert unet.call["encoder_hidden_states"].shape == (1, 2, 1024)
    assert unet.call["class_labels"][0].tolist() == pytest.approx([np.sin(1.0), 0.0, np.cos(1.0), 1.0])
    assert decoder.features is not None
    assert decoder.features.shape == (1, 320, 2, 2)
    assert provider.runtime_report.device == "cpu"
    assert provider.runtime_report.dtype == "float32"
    assert provider.runtime_report.backend == "torch-cpu"


class _FailOnceUNet(_FakeUNet):
    def forward(self, *args, **kwargs):
        if self.call_count == 0:
            self.call_count += 1
            raise FlashAttentionBackendError("synthetic FA2 kernel failure")
        return super().forward(*args, **kwargs)


class _FlashCoverage:
    saw_self_attention = False
    saw_cross_attention = False


def test_sdpose_provider_retries_with_sdpa_only_for_a_typed_fa2_failure() -> None:
    unet = _FailOnceUNet()
    provider = SDPoseBodyProvider(
        SDPoseLoadedModels(
            vae=_FakeVAE(),
            unet=unet,
            decoder=_FakeDecoder(),
            empty_prompt=torch.zeros((1, 2, 1024), dtype=torch.float32),
            device=torch.device("cpu"),
            dtype=torch.float32,
            backend="torch-fa2",
            bundle_sha256="b" * 64,
            flash_processor=_FlashCoverage(),
        )
    )

    provider.infer(
        Image.new("RGB", (100, 80), "white"),
        person_bbox=(20, 10, 60, 70),
    )

    assert unet.call_count == 2
    assert unet.processor.__class__.__name__ == "AttnProcessor2_0"
    assert provider.runtime_report.backend == "torch-sdpa"
    assert provider.runtime_report.fallback_reason == "synthetic FA2 kernel failure"
