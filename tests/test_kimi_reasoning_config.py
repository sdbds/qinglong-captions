from pathlib import Path

import pytest
import toml


def _write_config(
    path: Path,
    *,
    kimi_vl_effort: str | None,
    kimi_code_effort: str | None,
    kimi_vl_thinking: str = "enabled",
    kimi_code_thinking: str = "enabled",
) -> None:
    kimi_vl_line = f'reasoning_effort = "{kimi_vl_effort}"  # keep kimi-vl comment\n' if kimi_vl_effort is not None else ""
    kimi_code_line = f'reasoning_effort = "{kimi_code_effort}"  # keep kimi-code comment\n' if kimi_code_effort is not None else ""
    path.write_text(
        (
            "# keep file comment\n"
            "[kimi_vl]\n"
            f'thinking = "{kimi_vl_thinking}"\n'
            f"{kimi_vl_line}\n"
            "[kimi_code]\n"
            f'thinking = "{kimi_code_thinking}"\n'
            f"{kimi_code_line}\n"
            "[unrelated]\n"
            "value = 7\n"
        ),
        encoding="utf-8",
    )


def test_load_kimi_reasoning_effort_reads_section_and_defaults_missing_value(tmp_path):
    from gui.utils.kimi_reasoning_config import load_kimi_reasoning_effort

    config_path = tmp_path / "model.toml"
    _write_config(config_path, kimi_vl_effort="high", kimi_code_effort=None)

    assert load_kimi_reasoning_effort("kimi_vl", config_path=config_path) == "high"
    assert load_kimi_reasoning_effort("kimi_code", config_path=config_path) == "high"


def test_load_kimi_thinking_reads_each_legacy_toggle(tmp_path):
    from gui.utils.kimi_reasoning_config import load_kimi_thinking

    config_path = tmp_path / "model.toml"
    _write_config(
        config_path,
        kimi_vl_effort="high",
        kimi_code_effort="high",
        kimi_vl_thinking="disabled",
        kimi_code_thinking="enabled",
    )

    assert load_kimi_thinking("kimi_vl", config_path=config_path) == "disabled"
    assert load_kimi_thinking("kimi_code", config_path=config_path) == "enabled"


def test_save_kimi_reasoning_effort_updates_only_requested_section_in_both_files(tmp_path):
    from gui.utils.kimi_reasoning_config import save_kimi_reasoning_effort

    model_path = tmp_path / "model.toml"
    legacy_path = tmp_path / "config.toml"
    _write_config(model_path, kimi_vl_effort="low", kimi_code_effort="max")
    _write_config(legacy_path, kimi_vl_effort="low", kimi_code_effort="max")

    saved = save_kimi_reasoning_effort(
        "kimi_vl",
        " HIGH ",
        model_config_path=model_path,
        legacy_config_path=legacy_path,
    )

    assert saved == "high"
    for path in (model_path, legacy_path):
        parsed = toml.load(path)
        assert parsed["kimi_vl"]["reasoning_effort"] == "high"
        assert parsed["kimi_code"]["reasoning_effort"] == "max"
        assert parsed["unrelated"]["value"] == 7
        text = path.read_text(encoding="utf-8")
        assert "# keep file comment" in text
        assert "# keep kimi-vl comment" in text
        assert "# keep kimi-code comment" in text


def test_save_kimi_thinking_updates_only_requested_section_in_both_files(tmp_path):
    from gui.utils.kimi_reasoning_config import save_kimi_thinking

    model_path = tmp_path / "model.toml"
    legacy_path = tmp_path / "config.toml"
    _write_config(
        model_path,
        kimi_vl_effort="high",
        kimi_code_effort="high",
        kimi_vl_thinking="enabled",
        kimi_code_thinking="enabled",
    )
    _write_config(
        legacy_path,
        kimi_vl_effort="high",
        kimi_code_effort="high",
        kimi_vl_thinking="enabled",
        kimi_code_thinking="enabled",
    )

    saved = save_kimi_thinking(
        "kimi_code",
        " DISABLED ",
        model_config_path=model_path,
        legacy_config_path=legacy_path,
    )

    assert saved == "disabled"
    for path in (model_path, legacy_path):
        parsed = toml.load(path)
        assert parsed["kimi_vl"]["thinking"] == "enabled"
        assert parsed["kimi_code"]["thinking"] == "disabled"
        assert parsed["kimi_code"]["reasoning_effort"] == "high"


def test_invalid_kimi_reasoning_effort_leaves_both_files_unchanged(tmp_path):
    from gui.utils.kimi_reasoning_config import save_kimi_reasoning_effort

    model_path = tmp_path / "model.toml"
    legacy_path = tmp_path / "config.toml"
    _write_config(model_path, kimi_vl_effort="low", kimi_code_effort="max")
    _write_config(legacy_path, kimi_vl_effort="low", kimi_code_effort="max")
    before = {
        model_path: model_path.read_text(encoding="utf-8"),
        legacy_path: legacy_path.read_text(encoding="utf-8"),
    }

    with pytest.raises(ValueError, match="Unsupported K3 reasoning effort"):
        save_kimi_reasoning_effort(
            "kimi_code",
            "medium",
            model_config_path=model_path,
            legacy_config_path=legacy_path,
        )

    assert model_path.read_text(encoding="utf-8") == before[model_path]
    assert legacy_path.read_text(encoding="utf-8") == before[legacy_path]


def test_second_config_replace_failure_rolls_back_first_file(tmp_path, monkeypatch):
    import gui.utils.kimi_reasoning_config as config_module

    model_path = tmp_path / "model.toml"
    legacy_path = tmp_path / "config.toml"
    _write_config(model_path, kimi_vl_effort="low", kimi_code_effort="max")
    _write_config(legacy_path, kimi_vl_effort="low", kimi_code_effort="max")
    before = {
        model_path: model_path.read_text(encoding="utf-8"),
        legacy_path: legacy_path.read_text(encoding="utf-8"),
    }
    real_replace = config_module._replace_file
    failed_once = False

    def fail_legacy_once(source, destination):
        nonlocal failed_once
        if Path(destination) == legacy_path and not failed_once:
            failed_once = True
            raise OSError("legacy config is locked")
        real_replace(source, destination)

    monkeypatch.setattr(config_module, "_replace_file", fail_legacy_once)

    with pytest.raises(OSError, match="legacy config is locked"):
        config_module.save_kimi_reasoning_effort(
            "kimi_vl",
            "high",
            model_config_path=model_path,
            legacy_config_path=legacy_path,
        )

    assert model_path.read_text(encoding="utf-8") == before[model_path]
    assert legacy_path.read_text(encoding="utf-8") == before[legacy_path]
    assert not list(tmp_path.glob(".*.tmp"))


def test_save_kimi_reasoning_effort_requires_existing_provider_section(tmp_path):
    from gui.utils.kimi_reasoning_config import save_kimi_reasoning_effort

    model_path = tmp_path / "model.toml"
    legacy_path = tmp_path / "config.toml"
    model_path.write_text("[unrelated]\nvalue = 1\n", encoding="utf-8")
    legacy_path.write_text("[unrelated]\nvalue = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Missing \[kimi_vl\] section"):
        save_kimi_reasoning_effort(
            "kimi_vl",
            "low",
            model_config_path=model_path,
            legacy_config_path=legacy_path,
        )
