from pathlib import Path
from types import SimpleNamespace

import torch

from utils import onnx_export


def test_parse_args_keeps_legacy_roformer_default(tmp_path):
    args = onnx_export.parse_args(["--model-dir", str(tmp_path), "--print-only"])

    assert args.target == "roformer"
    assert args.model_dir == tmp_path
    assert args.print_only is True


def test_parse_args_accepts_musvit_subcommand(tmp_path):
    output_path = tmp_path / "musvit" / "model.onnx"

    args = onnx_export.parse_args(["musvit", "--output-path", str(output_path), "--device", "cpu"])

    assert args.target == "musvit"
    assert args.repo_id == "PRAIG/musvit"
    assert args.output_path == output_path
    assert args.device == "cpu"


def test_build_musvit_export_metadata_records_embedding_contract(tmp_path):
    model = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=768,
            image_size=1024,
            patch_size=16,
            num_channels=3,
            _commit_hash="abc123",
        )
    )
    output_path = tmp_path / "model.onnx"

    metadata = onnx_export.build_musvit_export_metadata(
        model=model,
        repo_id="PRAIG/musvit",
        revision=None,
        output_path=output_path,
        image_size=1024,
        preprocess_mode="page_resize",
    )

    assert metadata["model_type"] == "musvit"
    assert metadata["repo_id"] == "PRAIG/musvit"
    assert metadata["revision"] == "abc123"
    assert metadata["input_name"] == "pixel_values"
    assert metadata["output_name"] == "last_hidden_state"
    assert metadata["image_size"] == [1024, 1024]
    assert metadata["patch_size"] == 16
    assert metadata["patch_grid"] == [64, 64]
    assert metadata["hidden_size"] == 768
    assert metadata["contains_cls_token"] is True
    assert metadata["preprocess"]["mode"] == "page_resize"
    assert metadata["export_notes"]["uses_vit_model_encoder"] is True
    assert metadata["export_notes"]["does_not_export_mae_decoder"] is True


def test_build_musvit_load_kwargs_disables_unused_pooler():
    kwargs = onnx_export.build_musvit_load_kwargs(revision="abc123")

    assert kwargs["trust_remote_code"] is True
    assert kwargs["add_pooling_layer"] is False
    assert kwargs["revision"] == "abc123"


def test_build_musvit_preprocessor_config_matches_model_card_processor():
    config = onnx_export.build_musvit_preprocessor_config(image_size=1024, preprocess_mode="page_resize")

    assert config["image_processor_type"] == "ViTImageProcessor"
    assert config["do_resize"] is True
    assert config["size"] == {"height": 1024, "width": 1024}
    assert config["do_rescale"] is True
    assert config["rescale_factor"] == 1 / 255
    assert config["do_normalize"] is False
    assert config["preprocess_mode"] == "page_resize"


def test_write_musvit_preprocessor_config_defaults_next_to_onnx(tmp_path):
    output_path = tmp_path / "model.onnx"

    config_path = onnx_export.write_musvit_preprocessor_config(
        {"image_processor_type": "ViTImageProcessor"},
        output_path,
    )

    assert config_path == tmp_path / "preprocessor_config.json"
    assert config_path.read_text(encoding="utf-8").strip().startswith("{")


def test_musvit_encoder_wrapper_returns_last_hidden_state():
    class FakeModel(torch.nn.Module):
        def forward(self, pixel_values):
            batch = pixel_values.shape[0]
            return SimpleNamespace(last_hidden_state=torch.ones(batch, 4097, 768))

    wrapper = onnx_export.MuSViTEncoderWrapper(FakeModel())

    output = wrapper(torch.zeros(2, 3, 1024, 1024))

    assert tuple(output.shape) == (2, 4097, 768)
