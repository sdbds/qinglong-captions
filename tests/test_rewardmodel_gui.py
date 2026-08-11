from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from gui.utils.i18n import TRANSLATIONS
from gui.utils.reward_catalog import RewardCatalog, RewardCheckpoint
from gui.wizard import step6_tools
from module.reward_policy import RewardPolicy, Threshold

ROOT = Path(__file__).resolve().parent.parent


def test_tools_module_import_does_not_load_optional_reward_runtime():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import gui.wizard.step6_tools; "
                "assert 'qinglong_score' not in sys.modules; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("checkpoint", "expected_checkpoint_arg"),
    [(None, None), ("owner/checkpoint", "--checkpoint=owner/checkpoint")],
)
def test_start_reward_builds_scorer_checkpoint_arguments(
    monkeypatch, tmp_path: Path, checkpoint, expected_checkpoint_arg
):
    step = step6_tools.ToolsStep()
    step.reward_input = types.SimpleNamespace(value=str(tmp_path))
    step.reward_scorer = types.SimpleNamespace(value="aesthetic_predictor_v2_5")
    step.reward_checkpoint = types.SimpleNamespace(value=checkpoint)
    step.reward_device = types.SimpleNamespace(value="cpu")
    step.reward_dtype = types.SimpleNamespace(value="float32")
    step.reward_active_scorer = "aesthetic_predictor_v2_5"
    step.reward_threshold_dirty = False
    captured = {}

    class FakePanel:
        async def run_job(self, script, args, **kwargs):
            captured.update(script=script, args=args, kwargs=kwargs)

    monkeypatch.setattr(step, "_ensure_execution_panel", lambda: FakePanel())

    asyncio.run(step._start_reward())

    assert captured["script"] == "module.rewardmodel"
    args = captured["args"]
    assert "--scorer=aesthetic_predictor_v2_5" in args
    assert "--batch_size=1" in args
    assert "--device=cpu" in args
    assert "--dtype=float32" in args
    assert not any(argument.startswith("--repo_id") for argument in args)
    if expected_checkpoint_arg is None:
        assert not any(argument.startswith("--checkpoint") for argument in args)
    else:
        assert expected_checkpoint_arg in args


def test_start_reward_blocks_only_dirty_selected_profile(monkeypatch, tmp_path: Path):
    step = step6_tools.ToolsStep()
    step.reward_input = types.SimpleNamespace(value=str(tmp_path))
    step.reward_scorer = types.SimpleNamespace(value="selected")
    step.reward_checkpoint = types.SimpleNamespace(value=None)
    step.reward_device = types.SimpleNamespace(value="cpu")
    step.reward_dtype = types.SimpleNamespace(value="auto")
    step.reward_active_scorer = "selected"
    step.reward_threshold_dirty = True
    notifications = []
    started = []

    class FakePanel:
        async def run_job(self, *_args, **_kwargs):
            started.append(True)

    monkeypatch.setattr(step6_tools.ui, "notify", lambda message, **_kwargs: notifications.append(message))
    monkeypatch.setattr(step, "_ensure_execution_panel", lambda: FakePanel())

    asyncio.run(step._start_reward())
    assert started == []
    assert notifications

    step.reward_scorer.value = "other"
    asyncio.run(step._start_reward())
    assert started == [True]


def test_save_threshold_draft_persists_only_active_scorer(monkeypatch):
    step = step6_tools.ToolsStep()
    step.reward_active_scorer = "alpha"
    step.reward_threshold_draft = [
        {"name": "high", "max_score": 9.0, "color": "bold #00ff00"},
        {"name": "low", "max_score": 2.0, "color": "bold #ff0000"},
    ]
    step.reward_threshold_dirty = True
    calls = []

    def fake_save(path, scorer, rows):
        calls.append((path, scorer, rows))
        return (
            Threshold("low", 2.0, "bold #ff0000"),
            Threshold("high", 9.0, "bold #00ff00"),
        )

    monkeypatch.setattr(step6_tools, "save_threshold_profile", fake_save)

    assert step._save_reward_threshold_draft() is True
    assert calls[0][0] == step6_tools.REWARD_MODEL_TOML
    assert calls[0][1] == "alpha"
    assert [row["name"] for row in calls[0][2]] == ["high", "low"]
    assert [row["name"] for row in step.reward_threshold_draft] == ["low", "high"]
    assert step.reward_threshold_dirty is False


def test_discard_threshold_draft_does_not_rerender_during_tab_transition(
    monkeypatch,
):
    step = step6_tools.ToolsStep()
    step.reward_active_scorer = "alpha"
    step.reward_threshold_dirty = True
    renders = []
    policy = RewardPolicy(
        default_scorer="alpha",
        profiles={"alpha": (Threshold("saved", 1.0, "red"),)},
    )
    monkeypatch.setattr(step6_tools, "load_reward_policy", lambda _path: policy)
    monkeypatch.setattr(step, "_render_reward_threshold_grid", lambda: renders.append(True))
    monkeypatch.setattr(step, "_ensure_tool_panel_rendered", lambda _tab: None)
    monkeypatch.setattr(step, "_sync_execution_action", lambda: None)

    step._discard_reward_threshold_draft()

    assert renders == []
    assert step.reward_threshold_draft == [
        {"name": "saved", "max_score": 1.0, "color": "red"}
    ]
    assert step.reward_threshold_dirty is False
    assert step._reward_threshold_render_pending is True

    step._handle_tool_tab_change("preprocess")
    assert renders == []

    step._handle_tool_tab_change("reward")
    assert renders == [True]
    assert step._reward_threshold_render_pending is False


@pytest.mark.parametrize(
    ("action", "switched", "save_calls", "discard_calls"),
    [
        ("save", True, 1, 0),
        ("discard", True, 0, 1),
        ("cancel", False, 0, 0),
    ],
)
def test_dirty_scorer_switch_supports_save_discard_cancel(
    monkeypatch, action, switched, save_calls, discard_calls
):
    step = step6_tools.ToolsStep()
    step.reward_active_scorer = "alpha"
    step.reward_threshold_dirty = True
    step.reward_scorer = types.SimpleNamespace(value="beta")
    calls = {"save": 0, "discard": 0, "load": []}

    async def fake_action():
        return action

    def fake_save():
        calls["save"] += 1
        step.reward_threshold_dirty = False
        return True

    def fake_discard():
        calls["discard"] += 1
        step.reward_threshold_dirty = False

    def fake_load(scorer):
        calls["load"].append(scorer)
        step.reward_active_scorer = scorer
        step.reward_threshold_dirty = False

    monkeypatch.setattr(step, "_ask_reward_draft_action", fake_action)
    monkeypatch.setattr(step, "_save_reward_threshold_draft", fake_save)
    monkeypatch.setattr(step, "_discard_reward_threshold_draft", fake_discard)
    monkeypatch.setattr(step, "_load_reward_threshold_draft", fake_load)
    monkeypatch.setattr(step, "_refresh_reward_checkpoints", lambda _scorer: None)

    result = asyncio.run(step._switch_reward_scorer("beta"))

    assert result is switched
    assert calls["save"] == save_calls
    assert calls["discard"] == discard_calls
    assert calls["load"] == (["beta"] if switched else [])
    assert step.reward_scorer.value == ("beta" if switched else "alpha")


def test_reward_discovery_uses_public_names_and_checkpoint_metadata(monkeypatch):
    step = step6_tools.ToolsStep()
    step.reward_policy = RewardPolicy(
        default_scorer="configured_default",
        profiles={"configured_profile": ()},
    )
    scorer_updates = []
    checkpoint_updates = []

    class FakeControl:
        def __init__(self, value):
            self.value = value

        def set_options(self, options, value=None):
            self.value = value
            (scorer_updates if self is step.reward_scorer else checkpoint_updates).append(
                (options, value)
            )

    step.reward_scorer = FakeControl("configured_default")
    step.reward_checkpoint = FakeControl(None)

    catalog = RewardCatalog(
        scorers=("zeta", "configured_default", "alpha"),
        checkpoints={
            "configured_default": (
                RewardCheckpoint("owner/default", True, False),
                RewardCheckpoint("owner/tracking", False, True),
            )
        },
    )
    monkeypatch.setattr(step6_tools, "load_reward_catalog", lambda: catalog)

    asyncio.run(step._refresh_reward_discovery_async())

    scorer_options, selected_scorer = scorer_updates[-1]
    assert list(scorer_options) == [
        "configured_default",
        "alpha",
        "configured_profile",
        "zeta",
    ]
    assert selected_scorer == "configured_default"
    checkpoint_options, selected_checkpoint = checkpoint_updates[-1]
    assert list(checkpoint_options)[0] == ""
    assert step6_tools.t("reward_checkpoint_default_tag") in checkpoint_options[
        "owner/default"
    ]
    assert step6_tools.t("reward_checkpoint_tracking_tag") in checkpoint_options[
        "owner/tracking"
    ]
    assert selected_checkpoint == ""


def test_reward_gui_translation_keys_exist_in_every_language():
    keys = {
        "reward_scorer",
        "reward_checkpoint",
        "reward_checkpoint_default",
        "reward_checkpoint_default_tag",
        "reward_checkpoint_tracking_tag",
        "reward_discovery_refresh",
        "reward_thresholds",
        "reward_threshold_name",
        "reward_threshold_max_score",
        "reward_threshold_color",
        "reward_threshold_add",
        "reward_threshold_delete",
        "reward_threshold_save",
        "reward_threshold_dirty",
        "reward_threshold_dirty_message",
        "reward_threshold_discard",
        "reward_threshold_saved",
        "reward_threshold_invalid",
    }

    for language, mapping in TRANSLATIONS.items():
        for key in keys:
            assert isinstance(mapping.get(key), str), (language, key)
            assert mapping[key].strip(), (language, key)
