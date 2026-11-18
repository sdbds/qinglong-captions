import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from utils import onnx_export_defense as exporter


class _Tiles(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input1 = torch.nn.Linear(3, 1)

    def forward(self, tiles, candidates, events, prefix, event_valid, unknown_red):
        return {"score": tiles[:, 0, 0, 0]}


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "metadata.json").write_text('{"architecture":"tiny", "head_schema":{}}', encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"source checkpoint")
    model = torch.nn.Module()
    model.tiles = _Tiles()
    monkeypatch.setattr(exporter, "load_defense_model", lambda *args: model)
    return model_dir


def _tiny_export(monkeypatch, *, external=True, weight=1.0):
    onnx = pytest.importorskip("onnx")

    def write_graph(wrapper, path, device, opset):
        path.parent.mkdir(parents=True, exist_ok=True)
        scalar = onnx.numpy_helper.from_array(np.array([weight], dtype=np.float32), name="w")
        inputs = [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])]
        outputs = [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])]
        graph = onnx.helper.make_graph([onnx.helper.make_node("Add", ["x", "w"], ["y"])], "tiny", inputs, outputs, [scalar])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", opset)])
        onnx.save_model(model, path, save_as_external_data=external, location=path.name + ".data", size_threshold=0)
        return path

    monkeypatch.setattr(exporter, "export_to_onnx", write_graph)


def _run(model_dir, output, *args):
    return exporter.main(["--model-dir", str(model_dir), "--output-path", str(output), "--device=cpu", *args])


def test_source_imports_are_isolated_between_model_directories(tmp_path):
    for name in ("first", "second"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "heads.py").write_text(f"LABEL = {name!r}\n", encoding="utf-8")
        (directory / "model.py").write_text("from .heads import LABEL\n", encoding="utf-8")
        assert exporter.import_model_source(directory).LABEL == name


@pytest.mark.parametrize("name", ["model.safetensors", "metadata.onnx"])
def test_output_cannot_overwrite_source_inputs(model_dir, name, monkeypatch):
    _tiny_export(monkeypatch)
    before = {p.name: p.read_bytes() for p in model_dir.iterdir()}
    with pytest.raises(ValueError):
        _run(model_dir, model_dir / name)
    assert {p.name: p.read_bytes() for p in model_dir.iterdir()} == before


@pytest.mark.parametrize("name", ["reader.onnx", "reader.json", "reader.onnx.data"])
@pytest.mark.parametrize("flags", [(), ("--overwrite",)])
def test_unowned_graph_metadata_and_external_data_are_never_replaced(model_dir, tmp_path, name, flags, monkeypatch):
    _tiny_export(monkeypatch)
    (tmp_path / name).write_bytes(b"user file")
    with pytest.raises(FileExistsError):
        _run(model_dir, tmp_path / "reader.onnx", *flags)
    assert (tmp_path / name).read_bytes() == b"user file"
    assert {p.name for p in tmp_path.glob("reader*")} == {name}


def test_verification_failure_preserves_previous_external_graph(model_dir, tmp_path, monkeypatch):
    output = tmp_path / "reader.onnx"
    _tiny_export(monkeypatch)
    assert _run(model_dir, output) == 0
    before = {p.name: p.read_bytes() for p in tmp_path.glob("reader*")}
    assert set(before) == {"reader.onnx", "reader.json", "reader.onnx.data"}

    def reject_graph(wrapper, path, device):
        assert path != output and path.is_file()
        assert (path.parent / "reader.onnx.data").is_file()
        assert (tmp_path / "reader.onnx.data").read_bytes() == before["reader.onnx.data"]
        raise ValueError("verification failed")

    _tiny_export(monkeypatch, weight=2.0)
    monkeypatch.setattr(exporter, "verify_export", reject_graph)
    with pytest.raises(ValueError, match="verification failed"):
        _run(model_dir, output, "--verify", "--overwrite")
    assert {p.name: p.read_bytes() for p in tmp_path.glob("reader*")} == before


def test_successful_replacement_retires_only_owned_external_data(model_dir, tmp_path, monkeypatch):
    output = tmp_path / "reader.onnx"
    _tiny_export(monkeypatch)
    assert _run(model_dir, output) == 0
    _tiny_export(monkeypatch, external=False)
    assert _run(model_dir, output, "--overwrite") == 0
    assert not (tmp_path / "reader.onnx.data").exists()
    assert json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))["onnx_path"] == str(output)
    (tmp_path / "reader.onnx.data").write_bytes(b"unrelated data")
    _tiny_export(monkeypatch)
    with pytest.raises(FileExistsError):
        _run(model_dir, output, "--overwrite")
    assert (tmp_path / "reader.onnx.data").read_bytes() == b"unrelated data"


def test_edited_previous_graph_is_not_owned_for_replacement(model_dir, tmp_path, monkeypatch):
    output = tmp_path / "reader.onnx"
    _tiny_export(monkeypatch)
    assert _run(model_dir, output) == 0
    output.write_bytes(b"different graph")
    with pytest.raises(FileExistsError):
        _run(model_dir, output, "--overwrite")
    assert output.read_bytes() == b"different graph"


def test_invalid_graph_without_optional_parity_check_is_never_published(model_dir, tmp_path, monkeypatch):
    onnx = pytest.importorskip("onnx")
    output = tmp_path / "reader.onnx"
    _tiny_export(monkeypatch)
    assert _run(model_dir, output) == 0
    before = {p.name: p.read_bytes() for p in tmp_path.glob("reader*")}

    def invalid_graph(wrapper, path, device, opset):
        path.write_bytes(b"not an ONNX model")

    monkeypatch.setattr(exporter, "export_to_onnx", invalid_graph)
    with pytest.raises(onnx.checker.ValidationError):
        _run(model_dir, output, "--overwrite")
    assert {p.name: p.read_bytes() for p in tmp_path.glob("reader*")} == before


def test_existing_nonempty_lock_filename_is_not_clobbered(model_dir, tmp_path, monkeypatch):
    _tiny_export(monkeypatch)
    lock = tmp_path / ".reader.json.defense-export.lock"
    lock.write_bytes(b"another program's file")
    with pytest.raises(FileExistsError):
        _run(model_dir, tmp_path / "reader.onnx")
    assert lock.read_bytes() == b"another program's file"
    assert not (tmp_path / "reader.onnx").exists()


def test_failed_publication_restores_graph_json_and_data(model_dir, tmp_path, monkeypatch):
    output = tmp_path / "reader.onnx"
    _tiny_export(monkeypatch)
    assert _run(model_dir, output) == 0
    before = {p.name: p.read_bytes() for p in tmp_path.glob("reader*")}
    replace = os.replace
    denied = False

    def fail_once(src, dst):
        nonlocal denied
        if not denied and Path(dst) == output.with_suffix(".json"):
            denied = True
            raise PermissionError("metadata locked")
        return replace(src, dst)

    _tiny_export(monkeypatch, external=False)
    monkeypatch.setattr(os, "replace", fail_once)
    with pytest.raises(PermissionError, match="locked"):
        _run(model_dir, output, "--overwrite")
    assert {p.name: p.read_bytes() for p in tmp_path.glob("reader*")} == before
