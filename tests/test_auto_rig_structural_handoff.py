from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

import module.auto_rig.cli as cli
import module.auto_rig.pipeline as pipeline
from module.see_through.runner import ExecutionItem, _run_auto_rig_phase


@pytest.fixture
def structural_runtime(monkeypatch):
    def no_inference_required(root, **kwargs):
        assert kwargs["validation_tier"] == "structural"
        assert kwargs["finalize"] is False
        (root / "stage-validated.txt").write_text("structural", encoding="utf-8")
        return SimpleNamespace(item_root=root, terminal=None, reused_stages=())

    monkeypatch.setattr(pipeline, "_run_auto_rig_item_impl", no_inference_required)


def test_see_through_structural_tier_does_not_request_formal_finalization(tmp_path, structural_runtime):
    item = ExecutionItem(tmp_path / "a.png", Path("a.png"), tmp_path / "item", "completed")
    successful, failed = _run_auto_rig_phase(
        config=SimpleNamespace(auto_rig_validation_tier="structural", auto_rig_pose_mode="disabled"),
        items=[item],
        console_obj=Console(file=io.StringIO(), force_terminal=False, color_system=None),
    )
    assert failed == 0
    assert successful == [item]
    assert (item.item_dir / "stage-validated.txt").is_file()
    assert not (item.item_dir / "error.json").exists()


def test_standalone_cli_structural_tier_remains_stage_validated(monkeypatch, tmp_path, capsys, structural_runtime):
    monkeypatch.setenv("AUTO_RIG_VALIDATION_TIER", "structural")
    monkeypatch.setenv("AUTO_RIG_PROFILE", "dual_runtime_core_v1")
    assert cli.main([str(tmp_path)]) == 0
    assert '"status": "stage_validated"' in capsys.readouterr().out
    assert (tmp_path / "stage-validated.txt").is_file()
