import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module"))


def test_runtime_config_loads_split_files(tmp_path):
    from config.runtime_config import load_runtime_config

    (tmp_path / "prompts.toml").write_text("[prompts]\nsystem_prompt = 'sys'\n", encoding="utf-8")
    (tmp_path / "general.toml").write_text("[colors]\nimage = 'green'\n", encoding="utf-8")
    (tmp_path / "model.toml").write_text("[gemini]\nmodel_id = 'gemini-test'\n", encoding="utf-8")

    config = load_runtime_config(str(tmp_path))

    assert config["gemini"]["model_id"] == "gemini-test"
    assert config.prompts["system_prompt"] == "sys"
    assert config.colors["image"] == "green"


def test_runtime_config_loads_onnx_split_file(tmp_path):
    from config.runtime_config import load_runtime_config

    (tmp_path / "onnx.toml").write_text("[onnx_runtime.defaults]\nexecution_provider = 'cpu'\n", encoding="utf-8")

    config = load_runtime_config(str(tmp_path))

    assert config["onnx_runtime"]["defaults"]["execution_provider"] == "cpu"


def test_runtime_config_is_read_only():
    from config.runtime_config import coerce_runtime_config

    config = coerce_runtime_config({"prompts": {"system_prompt": "sys"}})

    with pytest.raises(TypeError):
        config["prompts"]["system_prompt"] = "changed"


def test_legacy_load_config_returns_read_only_snapshot(tmp_path):
    from config import config as legacy_config

    (tmp_path / "prompts.toml").write_text("[prompts]\nsystem_prompt = 'sys'\n", encoding="utf-8")
    (tmp_path / "general.toml").write_text("[colors]\nimage = 'green'\n", encoding="utf-8")
    (tmp_path / "model.toml").write_text("[gemini]\nmodel_id = 'gemini-test'\n", encoding="utf-8")

    snapshot = legacy_config.load_config(str(tmp_path))

    assert snapshot.prompts["system_prompt"] == "sys"
    assert isinstance(legacy_config.DATASET_SCHEMA, tuple)

    with pytest.raises(TypeError):
        legacy_config.CONSOLE_COLORS["image"] = "changed"


def test_legacy_load_config_exposes_onnx_runtime_from_config_toml(tmp_path):
    from config import config as legacy_config

    (tmp_path / "config.toml").write_text(
        "[onnx_runtime.defaults]\nexecution_provider = 'cpu'\n",
        encoding="utf-8",
    )

    snapshot = legacy_config.load_config(str(tmp_path))

    assert snapshot["onnx_runtime"]["defaults"]["execution_provider"] == "cpu"
