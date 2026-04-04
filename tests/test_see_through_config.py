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
    assert parsed["see_through"]["seed"] == 42
    assert parsed["see_through"]["quant_mode"] == "none"
    assert parsed["see_through"]["group_offload"] is False


def test_split_loader_reads_see_through_file(tmp_path):
    (tmp_path / "model.toml").write_text("[see_through]\nresolution = 1024\n", encoding="utf-8")

    config = load_config(str(tmp_path))

    assert config["see_through"]["resolution"] == 1024
