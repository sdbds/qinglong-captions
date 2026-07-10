import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent

from gui.utils.process_runner import SCRIPT_REGISTRY
from gui.wizard import step6_tools


def test_process_runner_registers_sheet_music_musvit():
    assert SCRIPT_REGISTRY["module.sheet_music_musvit"] == ("./module/sheet_music_musvit.py", "musvit-onnx")


def test_tools_step_exposes_sheet_music_tab_and_action():
    step = step6_tools.ToolsStep()

    assert ("sheet_music", "sheet_music", "library_music") in step.TOOL_TABS
    label_key, callback = step._tool_action_for_tab("sheet_music")
    assert label_key == "start_sheet_music"
    assert callback == step._start_sheet_music


def test_tools_step_sheet_music_maps_args(monkeypatch, tmp_path):
    step = step6_tools.ToolsStep()
    input_dir = tmp_path / "scores"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    captured = {}

    async def fake_run_job(script_key, args, name, **kwargs):
        captured["script_key"] = script_key
        captured["args"] = list(args)
        captured["name"] = name
        captured["kwargs"] = kwargs
        return SimpleNamespace(status="ok")

    notifications = []
    monkeypatch.setattr(step6_tools.ui, "notify", lambda message, **kwargs: notifications.append((message, kwargs)))

    step.sheet_music_input = SimpleNamespace(value=str(input_dir))
    step.sheet_music_output = SimpleNamespace(value=str(output_dir))
    step.sheet_music_repo_id = SimpleNamespace(value="bdsqlsz/musvit-onnx")
    step.sheet_music_model_dir = SimpleNamespace(value="huggingface")
    step.sheet_music_preprocess_mode = SimpleNamespace(value="pad_square")
    step.panel = SimpleNamespace(run_job=fake_run_job)
    step.config["sheet_music_batch_size"] = 3
    step.config["sheet_music_recursive"] = False
    step.config["sheet_music_skip_completed"] = False
    step.config["sheet_music_overwrite"] = True
    step.config["sheet_music_force_download"] = True
    step.config["sheet_music_pdf_dpi"] = 180

    asyncio.run(step._start_sheet_music())

    assert notifications == []
    assert captured["script_key"] == "module.sheet_music_musvit"
    assert captured["name"] == step6_tools.t("job_name_sheet_music")
    assert captured["args"][0] == str(input_dir)
    assert f"--output_dir={output_dir}" in captured["args"]
    assert "--repo_id=bdsqlsz/musvit-onnx" in captured["args"]
    assert "--model_dir=huggingface" in captured["args"]
    assert "--batch_size=3" in captured["args"]
    assert "--preprocess_mode=pad_square" in captured["args"]
    assert "--pdf_dpi=180" in captured["args"]
    assert "--no-recursive" in captured["args"]
    assert "--no-skip_completed" in captured["args"]
    assert "--overwrite" in captured["args"]
    assert "--force_download" in captured["args"]
