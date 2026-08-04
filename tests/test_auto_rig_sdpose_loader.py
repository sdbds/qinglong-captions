from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from module.auto_rig.pose.artifacts import PoseArtifactError
from module.auto_rig.pose.loader import (
    load_prefixed_state_dict,
    normalize_sdpose_vae_state_dict,
)


def test_prefixed_state_loader_returns_only_the_requested_component(tmp_path: Path) -> None:
    path = tmp_path / "bundle.safetensors"
    save_file(
        {
            "conditioning.empty_prompt": torch.zeros((1, 2, 1024)),
            "decoder.final_layer.bias": torch.tensor([1.0, 2.0]),
            "decoder.final_layer.weight": torch.tensor([[3.0, 4.0]]),
            "unet.conv.weight": torch.tensor([5.0]),
        },
        str(path),
    )

    state = load_prefixed_state_dict(path, "decoder")

    assert set(state) == {"final_layer.bias", "final_layer.weight"}
    assert state["final_layer.bias"].tolist() == [1.0, 2.0]


def test_prefixed_state_loader_rejects_a_missing_component(tmp_path: Path) -> None:
    path = tmp_path / "bundle.safetensors"
    save_file({"unet.weight": torch.ones(1)}, str(path))

    with pytest.raises(PoseArtifactError, match="component"):
        load_prefixed_state_dict(path, "decoder")


def test_legacy_sdpose_vae_attention_names_are_mapped_to_current_diffusers() -> None:
    state = {
        "encoder.mid_block.attentions.0.query.weight": torch.ones(1),
        "encoder.mid_block.attentions.0.key.bias": torch.ones(1) * 2,
        "decoder.mid_block.attentions.0.value.weight": torch.ones(1) * 3,
        "decoder.mid_block.attentions.0.proj_attn.bias": torch.ones(1) * 4,
        "encoder.conv_in.weight": torch.ones(1) * 5,
    }

    normalized = normalize_sdpose_vae_state_dict(state)

    assert set(normalized) == {
        "encoder.mid_block.attentions.0.to_q.weight",
        "encoder.mid_block.attentions.0.to_k.bias",
        "decoder.mid_block.attentions.0.to_v.weight",
        "decoder.mid_block.attentions.0.to_out.0.bias",
        "encoder.conv_in.weight",
    }
    assert normalized["decoder.mid_block.attentions.0.to_out.0.bias"].item() == 4


def test_vae_name_mapping_rejects_a_collision_instead_of_overwriting_weights() -> None:
    with pytest.raises(PoseArtifactError, match="collision"):
        normalize_sdpose_vae_state_dict(
            {
                "encoder.mid_block.attentions.0.query.weight": torch.ones(1),
                "encoder.mid_block.attentions.0.to_q.weight": torch.ones(1),
            }
        )
