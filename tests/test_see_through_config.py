from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from config.loader import load_config


ROOT = Path(__file__).resolve().parent.parent


def test_model_toml_contains_see_through_section():
    parsed = tomllib.loads((ROOT / "config" / "model.toml").read_text(encoding="utf-8"))

    assert "see_through" in parsed
    assert parsed["see_through"]["output_dir"] == "workspace/see_through_output"
    assert parsed["see_through"]["resolution_depth"] == 720
    assert parsed["see_through"]["inference_steps_depth"] == -1
    assert parsed["see_through"]["seed"] == 1026
    assert parsed["see_through"]["quant_mode"] == "none"
    assert parsed["see_through"]["group_offload"] is False


def test_model_toml_contains_musvit_section():
    parsed = tomllib.loads((ROOT / "config" / "model.toml").read_text(encoding="utf-8"))

    assert parsed["musvit"]["repo_id"] == "bdsqlsz/qinglong-musvit-1.0"
    assert parsed["musvit"]["revision"] == "6f47cefe0e736fbdd0eab9e8bc8d4602e81939fe"
    assert parsed["musvit"]["model_dir"] == "huggingface"
    assert parsed["musvit"]["output_format"] == "musicxml"
    assert "output_dir" not in parsed["musvit"]
    assert parsed["musvit"]["pdf_dpi"] == 144
    assert parsed["musvit"]["recursive"] is True
    assert parsed["musvit"]["skip_completed"] is True
    assert "batch_size" not in parsed["musvit"]
    assert "preprocess_mode" not in parsed["musvit"]
    assert "max_tokens" not in parsed["musvit"]


def test_onnx_toml_contains_musvit_runtime_section():
    parsed = tomllib.loads((ROOT / "config" / "onnx.toml").read_text(encoding="utf-8"))

    musvit = parsed["onnx_runtime"]["musvit"]
    assert musvit["execution_provider"] == "cuda"
    assert musvit["cuda"]["arena_extend_strategy"] == "kNextPowerOfTwo"
    assert musvit["cuda"]["use_tf32"] == 0
    assert musvit["cuda"]["tunable_op_enable"] is False
    assert musvit["cuda"]["tunable_op_tuning_enable"] is False


def test_split_loader_reads_see_through_file(tmp_path):
    (tmp_path / "model.toml").write_text("[see_through]\nresolution = 1024\n", encoding="utf-8")

    config = load_config(str(tmp_path))

    assert config["see_through"]["resolution"] == 1024
