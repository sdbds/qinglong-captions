import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui.utils.process_runner import ProcessRunner


def test_normalize_console_color_system_accepts_aliases():
    assert ProcessRunner._normalize_console_color_system("truecolor") == "truecolor"
    assert ProcessRunner._normalize_console_color_system("full") == "truecolor"
    assert ProcessRunner._normalize_console_color_system("24-bit") == "truecolor"
    assert ProcessRunner._normalize_console_color_system("256color") == "256"
    assert ProcessRunner._normalize_console_color_system("AUTO") == "auto"


def test_build_native_wrapper_env_injects_requested_color_system():
    base_env = {"EXISTING": "1"}

    wrapper_env = ProcessRunner._build_native_wrapper_env(base_env, "full")

    assert wrapper_env["EXISTING"] == "1"
    assert wrapper_env["_QINGLONG_RICH_COLOR_SYSTEM"] == "truecolor"
    assert "_QINGLONG_RICH_COLOR_SYSTEM" not in base_env


def test_build_native_wrapper_env_drops_color_override_when_none():
    wrapper_env = ProcessRunner._build_native_wrapper_env(
        {"_QINGLONG_RICH_COLOR_SYSTEM": "256"},
        None,
    )

    assert "_QINGLONG_RICH_COLOR_SYSTEM" not in wrapper_env


def test_normalize_console_color_system_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported console_color_system"):
        ProcessRunner._normalize_console_color_system("neon")
