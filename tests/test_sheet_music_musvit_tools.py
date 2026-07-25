import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_sheet_music_output_selector_starts_blank():
    step = step6_tools.ToolsStep()

    step._render_sheet_music_tool()

    assert step.sheet_music_output.value in {"", None}


@pytest.mark.parametrize(
    ("overwrite", "overwrite_arg", "opposite_arg"),
    (
        (True, "--overwrite", "--no-overwrite"),
        (False, "--no-overwrite", "--overwrite"),
    ),
)
def test_tools_step_sheet_music_maps_args(
    monkeypatch,
    tmp_path,
    overwrite,
    overwrite_arg,
    opposite_arg,
):
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
    step.sheet_music_output_format = SimpleNamespace(value="both")
    step.panel = SimpleNamespace(run_job=fake_run_job)
    step.config["sheet_music_recursive"] = False
    step.config["sheet_music_skip_completed"] = False
    step.config["sheet_music_overwrite"] = overwrite
    step.config["sheet_music_pdf_dpi"] = 180

    asyncio.run(step._start_sheet_music())

    assert notifications == []
    assert captured["script_key"] == "module.sheet_music_musvit"
    assert captured["name"] == step6_tools.t("job_name_sheet_music")
    assert captured["args"][0] == str(input_dir)
    assert f"--output_dir={output_dir}" in captured["args"]
    assert "--output_format=both" in captured["args"]
    assert "--pdf_dpi=180" in captured["args"]
    assert "--no-recursive" in captured["args"]
    assert "--no-skip_completed" in captured["args"]
    assert overwrite_arg in captured["args"]
    assert opposite_arg not in captured["args"]
    assert not any(arg.startswith("--repo_id") for arg in captured["args"])
    assert not any(arg.startswith("--model_dir") for arg in captured["args"])
    assert not any(arg.startswith("--batch_size") for arg in captured["args"])
    assert not any(arg.startswith("--preprocess_mode") for arg in captured["args"])
    assert "--force_download" not in captured["args"]


def test_tools_step_sheet_music_omits_blank_output_directory(
    monkeypatch,
    tmp_path,
):
    step = step6_tools.ToolsStep()
    input_path = tmp_path / "score.pdf"
    input_path.write_bytes(b"%PDF")
    captured = {}

    async def fake_run_job(script_key, args, name, **kwargs):
        captured["args"] = list(args)
        return SimpleNamespace(status="ok")

    monkeypatch.setattr(step6_tools.ui, "notify", lambda *args, **kwargs: None)
    step.sheet_music_input = SimpleNamespace(value=str(input_path))
    step.sheet_music_output = SimpleNamespace(value="")
    step.sheet_music_output_format = SimpleNamespace(value="musicxml")
    step.panel = SimpleNamespace(run_job=fake_run_job)

    asyncio.run(step._start_sheet_music())

    assert not any(
        argument.startswith("--output_dir=")
        for argument in captured["args"]
    )
