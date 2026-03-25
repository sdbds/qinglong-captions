from pathlib import Path

from gui.utils import env_config


def test_uv_extra_index_default_covers_onnx_cuda13_and_torch_cu130():
    defaults = {item["key"]: item["default"] for item in env_config.ENV_VAR_DEFINITIONS}

    assert defaults["UV_EXTRA_INDEX_URL"] == env_config.UV_EXTRA_INDEX_URL_DEFAULT
    assert env_config.UV_ONNX_CUDA13_EXTRA_INDEX_URL in defaults["UV_EXTRA_INDEX_URL"]
    assert env_config.UV_PYTORCH_EXTRA_INDEX_URL in defaults["UV_EXTRA_INDEX_URL"]


def test_load_env_config_migrates_legacy_single_pytorch_extra_index(tmp_path, monkeypatch):
    config_path = tmp_path / "env_vars.json"
    config_path.write_text(
        '{"UV_EXTRA_INDEX_URL": "https://download.pytorch.org/whl/cu130"}',
        encoding="utf-8",
    )

    monkeypatch.setattr(env_config, "CONFIG_PATH", Path(config_path))
    monkeypatch.setattr(env_config, "_detect_system_language", lambda: "en")

    loaded = env_config.load_env_config()

    assert loaded["UV_EXTRA_INDEX_URL"] == env_config.UV_EXTRA_INDEX_URL_DEFAULT


def test_load_env_config_preserves_custom_extra_index(tmp_path, monkeypatch):
    custom_value = "https://example.invalid/simple/"
    config_path = tmp_path / "env_vars.json"
    config_path.write_text(
        f'{{"UV_EXTRA_INDEX_URL": "{custom_value}"}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(env_config, "CONFIG_PATH", Path(config_path))
    monkeypatch.setattr(env_config, "_detect_system_language", lambda: "en")

    loaded = env_config.load_env_config()

    assert loaded["UV_EXTRA_INDEX_URL"] == custom_value
