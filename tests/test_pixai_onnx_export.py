import importlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from utils import onnx_export


def pixai_export():
    assert importlib.util.find_spec("utils.onnx_export_pixai") is not None, "PixAI ONNX exporter is missing"
    return importlib.import_module("utils.onnx_export_pixai")


def test_pixai_export_cli_resolves_a_separate_target(tmp_path):
    args = onnx_export.parse_args(["pixai-tagger", "--model-dir", str(tmp_path), "--verify", "--device", "cpu"])
    assert args.target == "pixai-tagger"
    assert args.model_dir == tmp_path
    assert args.verify is True
    assert args.device == "cpu"


def test_real_rotary_export_matches_complex_reference():
    exporter = pixai_export()
    generator = torch.Generator().manual_seed(42)
    q = torch.randn(2, 3, 9, 8, generator=generator)
    k = torch.randn(2, 3, 9, 8, generator=generator)
    angles = torch.randn(9, 4, generator=generator)
    cis = torch.polar(torch.ones_like(angles), angles)
    attention = torch.nn.Module()
    attention.use_rope = True
    attention.register_buffer("freqs_cis", cis, persistent=False)
    model = torch.nn.Sequential(attention)
    expected = [torch.view_as_real(torch.view_as_complex(x.reshape(2, 3, 9, 4, 2)) * cis).flatten(3) for x in (q, k)]

    assert exporter.prepare_pixai_for_onnx(model) == 1
    actual = attention._apply_rope(q, k)

    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed, reference, rtol=1e-6, atol=1e-6)
    assert all(not value.is_complex() for value in model.buffers())


def test_real_rotary_graph_runs_with_dynamic_batch(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    exporter = pixai_export()

    class RotaryModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.use_rope = True
            angles = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 13
            self.register_buffer("freqs_cis", torch.polar(torch.ones_like(angles), angles), persistent=False)

        def forward(self, x):
            return self._apply_rope(x, x)[0]

    model = RotaryModel().eval()
    exporter.prepare_pixai_for_onnx(model)
    output = tmp_path / "rotary.onnx"
    torch.onnx.export(
        model,
        (torch.ones(1, 2, 6, 8),),
        str(output),
        dynamo=False,
        opset_version=20,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
    )
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    x = torch.arange(192, dtype=torch.float32).reshape(2, 2, 6, 8) / 100
    np.testing.assert_allclose(session.run(None, {"x": x.numpy()})[0], model(x).numpy(), rtol=1e-6, atol=1e-6)


def test_numeric_verification_rejects_bad_or_nonfinite_probabilities():
    exporter = pixai_export()
    reference = np.array([[0.0, 1.0, -2.0]], dtype=np.float32)
    report = exporter.compare_logits(reference, reference + 1e-6)
    assert report["max_probability_error"] < 1e-5
    with pytest.raises(ValueError, match="probability"):
        exporter.compare_logits(reference, reference + 1.0)
    with pytest.raises(ValueError, match="finite"):
        exporter.compare_logits(reference, np.full_like(reference, np.nan))


def test_export_plan_does_not_load_or_download_a_model(tmp_path, capsys):
    args = onnx_export.parse_args(["pixai-tagger", "--model-dir", str(tmp_path), "--print-only"])
    assert pixai_export().run_pixai_export(args) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["input_name"] == "pixel_values"
    assert plan["repo_id"] == "pixai-labs/pixai-tagger-v1.0"
    assert plan["revision"] == "main"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("filename", ["model.safetensors", "config.json", "preprocessor_config.json"])
def test_export_cannot_overwrite_source_artifacts(tmp_path, filename):
    args = onnx_export.parse_args(["pixai-tagger", "--model-dir", str(tmp_path), "--output-path", str(tmp_path / filename)])
    with pytest.raises(ValueError, match="output"):
        pixai_export().run_pixai_export(args)


@pytest.mark.parametrize("stage", ["export", "verify", "publish", "cancel"])
def test_failed_export_removes_only_its_temporary_file(tmp_path, monkeypatch, stage):
    from transformers import AutoImageProcessor, AutoModel

    exporter = pixai_export()
    (tmp_path / "config.json").write_text(json.dumps({"img_size": 4, "depth": 0}), encoding="utf-8")
    output = tmp_path / "model.onnx"
    output.write_bytes(b"existing model")
    unrelated = tmp_path / ".other.exporting.onnx"
    unrelated.write_bytes(b"another job")
    created = []

    class TinyModel(torch.nn.Module):
        def forward(self, pixels):
            return pixels.mean(dim=(1, 2, 3)).unsqueeze(1)

    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *args, **kwargs: TinyModel())
    monkeypatch.setattr(
        AutoImageProcessor,
        "from_pretrained",
        lambda *args, **kwargs: lambda images: {"pixel_values": torch.from_numpy(exporter.preprocess_pixai_images(images, 4))},
    )

    def export(*args, **kwargs):
        path = Path(args[2])
        path.write_bytes(b"partial export")
        created.append(path)
        if stage == "export":
            raise OSError("export failed")
        if stage == "cancel":
            raise KeyboardInterrupt()

    monkeypatch.setattr(torch.onnx, "export", export)
    if stage == "verify":
        onnx = pytest.importorskip("onnx")

        def reject_graph(*args, **kwargs):
            raise ValueError("invalid graph")

        monkeypatch.setattr(onnx.checker, "check_model", reject_graph)
    if stage == "publish":
        original_replace = os.replace

        def reject_publish(path, target):
            if Path(target) == output:
                raise PermissionError("cannot publish")
            return original_replace(path, target)

        monkeypatch.setattr(os, "replace", reject_publish)

    args = onnx_export.parse_args(
        ["pixai-tagger", "--model-dir", str(tmp_path), "--device=cpu", "--overwrite", *(["--verify"] if stage == "verify" else [])]
    )
    expected_error = {"export": OSError, "verify": ValueError, "publish": PermissionError, "cancel": KeyboardInterrupt}[stage]
    with pytest.raises(expected_error):
        exporter.run_pixai_export(args)
    assert created and not created[0].exists()
    assert output.read_bytes() == b"existing model"
    assert unrelated.read_bytes() == b"another job"
