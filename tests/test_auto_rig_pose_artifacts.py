from __future__ import annotations

from pathlib import Path

import pytest

from module.auto_rig.pose.artifacts import (
    COMFY_SDPOSE_WHOLEBODY_MODEL,
    DETRPOSE_X_CROWDPOSE_MODEL,
    SDPOSE_BODY_SOURCE,
    PoseArtifactError,
    classify_comfy_sdpose_repository_file,
    download_pose_model,
    download_pose_source,
    verify_pose_model_file,
    verify_pose_source_snapshot,
)
from module.auto_rig.pose.runtime import PoseOnnxContract, select_pose_runtime


def test_sdpose_body_source_contract_covers_every_runtime_component() -> None:
    assert SDPOSE_BODY_SOURCE.repo_id == "teemosliang/SDPose-Body"
    assert SDPOSE_BODY_SOURCE.revision == "5a34e0c7df4c8ea5fc8774c5f2ae4229e962238c"
    assert {item.relative_path: (item.size, item.sha256) for item in SDPOSE_BODY_SOURCE.files} == {
        "unet/diffusion_pytorch_model.safetensors": (
            3_470_311_272,
            "a75d358808e58cd5eb305dd3362d0d1457d243d787ac4bf1905b64da71d8934a",
        ),
        "vae/diffusion_pytorch_model.safetensors": (
            334_643_276,
            "a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815",
        ),
        "decoder/decoder.safetensors": (
            6_986_756,
            "32994dfc90beb84786c8e9296eeef60dad66980d86d7349be7c7ff80f3aaa8a4",
        ),
        "unet/config.json": (
            1_872,
            "39e3b8a8550583c3aa15de950526aa6cccc3dc9965fb18ae586d214bd80b1ff4",
        ),
        "vae/config.json": (
            611,
            "d69281aa3f6a0f3c41aaf6778e35464fc6ee8a92e6ac8a8b1eb679f6df6423eb",
        ),
        "scheduler/scheduler_config.json": (
            344,
            "ce14ed1d0a58d10a1e22b2d16786ecde6b14ef52cd0e21f2631eca39b49fac63",
        ),
    }


def test_pinned_comparison_model_contract_matches_the_verified_hugging_face_file() -> None:
    assert DETRPOSE_X_CROWDPOSE_MODEL.repo_id == "SebasJanampa/DETRPose_X_CROWDPOSE"
    assert DETRPOSE_X_CROWDPOSE_MODEL.revision == "cebc9cb1ad6289f262262412f604fd03a1d4d6a4"
    assert DETRPOSE_X_CROWDPOSE_MODEL.relative_path == "model.safetensors"
    assert DETRPOSE_X_CROWDPOSE_MODEL.size == 298_505_628
    assert DETRPOSE_X_CROWDPOSE_MODEL.sha256 == "563431b5f20434a1954ba2998f2010d1e960a672996b4a07f2f32ab694e125ee"


def test_comfy_rt_detr_weights_are_classified_as_detectors_not_pose_models() -> None:
    assert classify_comfy_sdpose_repository_file("checkpoints/sdpose_wholebody_fp16.safetensors") == "pose_estimator"
    assert classify_comfy_sdpose_repository_file("diffusion_models/rt_detr_v4-x-hgnet_fp16.safetensors") == "person_detector"


def test_pose_runtime_prefers_only_a_complete_registered_onnx_contract(tmp_path: Path) -> None:
    onnx = tmp_path / "sdpose.onnx"
    onnx.write_bytes(b"onnx")

    assert (
        select_pose_runtime(
            onnx_path=onnx,
            onnx_contract=None,
            cuda_available=True,
            flash_attn_available=True,
        ).backend
        == "torch-fa2"
    )

    contract = PoseOnnxContract(
        provider_id="sdpose-body17",
        graph_sha256="sha256:" + "1" * 64,
        input_names=("image",),
        output_names=("keypoints", "scores"),
    )
    assert (
        select_pose_runtime(
            onnx_path=onnx,
            onnx_contract=contract,
            cuda_available=True,
            flash_attn_available=True,
        ).backend
        == "onnx"
    )
    assert (
        select_pose_runtime(
            onnx_path=None,
            onnx_contract=None,
            cuda_available=True,
            flash_attn_available=False,
        ).backend
        == "torch-sdpa"
    )


def test_pose_model_download_reuses_the_shared_snapshot_downloader(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    model_path = tmp_path / Path(*DETRPOSE_X_CROWDPOSE_MODEL.relative_path.split("/"))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"placeholder")

    def fake_download(repo_id: str, **kwargs: object) -> str:
        calls.append((repo_id, kwargs))
        return str(tmp_path)

    resolved = download_pose_model(
        DETRPOSE_X_CROWDPOSE_MODEL,
        downloader=fake_download,
        verify=False,
    )

    assert resolved == model_path
    assert calls == [
        (
            "SebasJanampa/DETRPose_X_CROWDPOSE",
            {
                "revision": DETRPOSE_X_CROWDPOSE_MODEL.revision,
                "allow_patterns": [DETRPOSE_X_CROWDPOSE_MODEL.relative_path],
            },
        )
    ]


def test_sdpose_source_download_requests_and_verifies_the_complete_coherence_group(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    for artifact in SDPOSE_BODY_SOURCE.files:
        path = tmp_path / Path(*artifact.relative_path.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")

    def fake_download(repo_id: str, **kwargs: object) -> str:
        calls.append((repo_id, kwargs))
        return str(tmp_path)

    resolved = download_pose_source(
        SDPOSE_BODY_SOURCE,
        downloader=fake_download,
        verify=False,
    )

    assert resolved == tmp_path
    assert calls == [
        (
            "teemosliang/SDPose-Body",
            {
                "revision": SDPOSE_BODY_SOURCE.revision,
                "allow_patterns": [
                    "decoder/decoder.safetensors",
                    "scheduler/scheduler_config.json",
                    "unet/config.json",
                    "unet/diffusion_pytorch_model.safetensors",
                    "vae/config.json",
                    "vae/diffusion_pytorch_model.safetensors",
                ],
            },
        )
    ]


def test_pose_model_verifier_rejects_a_wrong_size_before_loading(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"wrong")

    with pytest.raises(PoseArtifactError, match="size"):
        verify_pose_model_file(path, DETRPOSE_X_CROWDPOSE_MODEL)


def test_sdpose_source_verifier_rejects_an_incomplete_snapshot(tmp_path: Path) -> None:
    with pytest.raises(PoseArtifactError, match="omitted required file"):
        verify_pose_source_snapshot(tmp_path, SDPOSE_BODY_SOURCE)


def test_comfy_wholebody_is_reference_only_not_the_sdpose_body_source() -> None:
    assert COMFY_SDPOSE_WHOLEBODY_MODEL.provider_id == "sdpose-wholebody-reference"
    assert COMFY_SDPOSE_WHOLEBODY_MODEL.repo_id == "Comfy-Org/SDPose"
    assert COMFY_SDPOSE_WHOLEBODY_MODEL.relative_path == ("checkpoints/sdpose_wholebody_fp16.safetensors")
