from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from module.auto_rig.pose.artifacts import (
    PoseArtifactError,
    PoseArtifactFile,
    PoseSourceContract,
)
from module.auto_rig.pose.bundle import (
    SDPOSE_BODY_BUNDLE_SCHEMA,
    build_sdpose_body_bundle,
    inspect_sdpose_body_bundle,
)
from module.auto_rig.pose.conditioning import (
    sdpose_empty_prompt,
    sdpose_empty_prompt_provenance,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_fixture(tmp_path: Path) -> tuple[Path, PoseSourceContract]:
    files = {
        "unet/diffusion_pytorch_model.safetensors": {
            "conv.weight": torch.tensor([[1.25, -2.5]], dtype=torch.float32),
        },
        "vae/diffusion_pytorch_model.safetensors": {
            "encoder.weight": torch.tensor([3.0], dtype=torch.float32),
        },
        "decoder/decoder.safetensors": {
            "final_layer.weight": torch.tensor([4.0, 5.0], dtype=torch.float32),
            "step": torch.tensor([7], dtype=torch.int64),
        },
    }
    configs = {
        "unet/config.json": {"_class_name": "UNet2DConditionModel", "in_channels": 4},
        "vae/config.json": {"_class_name": "AutoencoderKL", "latent_channels": 4},
        "scheduler/scheduler_config.json": {
            "_class_name": "DDPMScheduler",
            "prediction_type": "sample",
        },
    }
    for relative_path, tensors in files.items():
        path = tmp_path / Path(*relative_path.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(path))
    for relative_path, payload in configs.items():
        path = tmp_path / Path(*relative_path.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    artifacts = tuple(
        PoseArtifactFile(
            relative_path=relative_path,
            size=path.stat().st_size,
            sha256=_sha256(path),
        )
        for relative_path in sorted((*files, *configs))
        for path in [tmp_path / Path(*relative_path.split("/"))]
    )
    return tmp_path, PoseSourceContract(
        provider_id="sdpose-body17-test",
        repo_id="test/sdpose-body",
        revision="1" * 40,
        files=artifacts,
    )


def test_body_bundle_prefixes_components_and_converts_learned_floats_to_fp16(
    tmp_path: Path,
) -> None:
    source, contract = _source_fixture(tmp_path / "source")
    output = tmp_path / "sdpose_body17_fp16.safetensors"

    info = build_sdpose_body_bundle(
        source,
        output,
        source_contract=contract,
    )

    with safe_open(output, framework="pt", device="cpu") as bundle:
        assert set(bundle.keys()) == {
            "conditioning.empty_prompt",
            "decoder.final_layer.weight",
            "decoder.step",
            "unet.conv.weight",
            "vae.encoder.weight",
        }
        assert bundle.get_tensor("unet.conv.weight").dtype == torch.float16
        assert bundle.get_tensor("decoder.final_layer.weight").dtype == torch.float16
        assert bundle.get_tensor("decoder.step").dtype == torch.int64
        assert bundle.get_tensor("conditioning.empty_prompt").equal(sdpose_empty_prompt())
        metadata = bundle.metadata()
        assert metadata["bundle_schema"] == SDPOSE_BODY_BUNDLE_SCHEMA
        assert json.loads(metadata["unet_config_jcs"])["in_channels"] == 4
        assert json.loads(metadata["scheduler_config_jcs"])["prediction_type"] == "sample"
        assert json.loads(metadata["conditioning_provenance_jcs"]) == (sdpose_empty_prompt_provenance())

    assert info.path == output.resolve()
    assert info.tensor_count == 5
    assert info.file_sha256 == _sha256(output)
    assert inspect_sdpose_body_bundle(output, source_contract=contract) == info


def test_body_bundle_is_byte_deterministic_for_the_same_source_and_conditioning(
    tmp_path: Path,
) -> None:
    source, contract = _source_fixture(tmp_path / "source")
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"

    build_sdpose_body_bundle(
        source,
        first,
        source_contract=contract,
    )
    build_sdpose_body_bundle(
        source,
        second,
        source_contract=contract,
    )

    assert first.read_bytes() == second.read_bytes()


def test_bundle_inspection_detects_payload_tampering_without_source_files(
    tmp_path: Path,
) -> None:
    source, contract = _source_fixture(tmp_path / "source")
    output = tmp_path / "bundle.safetensors"
    build_sdpose_body_bundle(
        source,
        output,
        source_contract=contract,
    )
    raw = bytearray(output.read_bytes())
    raw[-1] ^= 0x01
    output.write_bytes(raw)

    with pytest.raises(PoseArtifactError, match="tensor payload"):
        inspect_sdpose_body_bundle(output, source_contract=contract)


def test_failed_rebuild_does_not_replace_a_previous_valid_bundle(tmp_path: Path) -> None:
    source, contract = _source_fixture(tmp_path / "source")
    output = tmp_path / "bundle.safetensors"
    build_sdpose_body_bundle(
        source,
        output,
        source_contract=contract,
    )
    original = output.read_bytes()
    (source / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"broken")

    with pytest.raises(PoseArtifactError, match="size"):
        build_sdpose_body_bundle(
            source,
            output,
            source_contract=contract,
        )

    assert output.read_bytes() == original
