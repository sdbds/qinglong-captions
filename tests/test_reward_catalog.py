from __future__ import annotations

import json
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from gui.utils import reward_catalog


def _create_project_python(project_root: Path, name: str = ".venv") -> Path:
    relative = Path(name) / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    python_path = project_root / relative
    python_path.parent.mkdir(parents=True)
    python_path.touch()
    return python_path


@pytest.mark.parametrize("venv_name", [".venv", "venv"])
def test_load_reward_catalog_uses_project_venv_and_parses_api_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    venv_name: str,
):
    python_path = _create_project_python(tmp_path, venv_name)
    captured = {}
    payload = {
        "scorers": ["alpha", "beta"],
        "checkpoints": {
            "alpha": [
                {
                    "identifier": "owner/default",
                    "is_default": True,
                    "tracks_updates": False,
                }
            ],
            "beta": [],
        },
    }

    def fake_run(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(reward_catalog.subprocess, "run", fake_run)

    catalog = reward_catalog.load_reward_catalog(project_root=tmp_path)

    assert captured["command"] == [
        str(python_path),
        "-m",
        "gui.utils.reward_catalog",
    ]
    assert captured["kwargs"]["cwd"] == tmp_path
    assert captured["kwargs"]["env"]["VIRTUAL_ENV"] == str(tmp_path / venv_name)
    assert catalog.scorers == ("alpha", "beta")
    assert catalog.checkpoints["alpha"] == (
        reward_catalog.RewardCheckpoint(
            identifier="owner/default",
            is_default=True,
            tracks_updates=False,
        ),
    )


def test_load_reward_catalog_does_not_fall_back_to_gui_python(tmp_path: Path):
    with pytest.raises(reward_catalog.RewardCatalogError, match=r"\.venv"):
        reward_catalog.load_reward_catalog(project_root=tmp_path)


def test_catalog_payload_uses_qinglong_score_public_api(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    @dataclass(frozen=True, slots=True)
    class Checkpoint:
        identifier: str
        is_default: bool
        tracks_updates: bool

    module = types.ModuleType("qinglong_score")
    module.list_scorers = lambda: ("alpha", "beta")

    def list_checkpoints(scorer):
        calls.append(scorer)
        return (Checkpoint(f"owner/{scorer}", scorer == "alpha", scorer == "beta"),)

    module.list_checkpoints = list_checkpoints
    monkeypatch.setitem(sys.modules, "qinglong_score", module)

    payload = reward_catalog._build_catalog_payload()

    assert calls == ["alpha", "beta"]
    assert payload == {
        "scorers": ["alpha", "beta"],
        "checkpoints": {
            "alpha": [
                {
                    "identifier": "owner/alpha",
                    "is_default": True,
                    "tracks_updates": False,
                }
            ],
            "beta": [
                {
                    "identifier": "owner/beta",
                    "is_default": False,
                    "tracks_updates": True,
                }
            ],
        },
    }
