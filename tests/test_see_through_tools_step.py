import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "gui"))
sys.path.insert(2, str(ROOT / "gui" / "wizard"))

from gui.wizard import step6_tools

ToolsStep = step6_tools.ToolsStep


def test_tools_step_see_through_maps_args(monkeypatch, tmp_path):
    step = ToolsStep()
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "outputs"
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

    step.see_through_input = SimpleNamespace(value=str(input_dir))
    step.see_through_output = SimpleNamespace(value=str(output_dir))
    step.see_through_repo_id_layerdiff = SimpleNamespace(value="layerdiff/repo")
    step.see_through_repo_id_depth = SimpleNamespace(value="marigold/repo")
    step.panel = SimpleNamespace(run_job=fake_run_job)
    step.config["see_through_resolution_depth"] = 720
    step.config["see_through_inference_steps_depth"] = 9
    step.config["see_through_seed"] = 123
    step.config["see_through_quant_mode"] = "nf4"
    step.config["see_through_group_offload"] = True
    step.config["see_through_skip_completed"] = True
    step.config["see_through_continue_on_error"] = True
    step.config["see_through_save_to_psd"] = True
    step.config["see_through_limit_images"] = 123
    step.config["see_through_force_eager_attention"] = True

    asyncio.run(step._start_see_through())

    assert notifications == []
    assert captured["script_key"] == "module.see_through.cli"
    assert captured["name"] == "See-through"
    assert f"--input_dir={input_dir}" in captured["args"]
    assert f"--output_dir={output_dir}" in captured["args"]
    assert "--repo_id_layerdiff=layerdiff/repo" in captured["args"]
    assert "--repo_id_depth=marigold/repo" in captured["args"]
    assert "--resolution_depth=720" in captured["args"]
    assert "--inference_steps_depth=9" in captured["args"]
    assert "--seed=123" in captured["args"]
    assert "--quant_mode=nf4" in captured["args"]
    assert "--group_offload" in captured["args"]
    assert "--skip_completed" in captured["args"]
    assert not any(arg.startswith("--limit_images=") for arg in captured["args"])
    assert "--continue_on_error" not in captured["args"]
    assert "--no-continue_on_error" not in captured["args"]
    assert "--force_eager_attention" not in captured["args"]
    assert "--no-force_eager_attention" not in captured["args"]


def test_tools_step_see_through_requires_existing_input(monkeypatch, tmp_path):
    step = ToolsStep()
    notifications = []
    run_calls = []

    async def fake_run_job(*args, **kwargs):
        run_calls.append((args, kwargs))
        return SimpleNamespace(status="ok")

    monkeypatch.setattr(step6_tools.ui, "notify", lambda message, **kwargs: notifications.append((message, kwargs)))

    step.see_through_input = SimpleNamespace(value=str(tmp_path / "missing"))
    step.see_through_output = SimpleNamespace(value=str(tmp_path / "outputs"))
    step.see_through_repo_id_layerdiff = SimpleNamespace(value="layerdiff/repo")
    step.see_through_repo_id_depth = SimpleNamespace(value="marigold/repo")
    step.panel = SimpleNamespace(run_job=fake_run_job)
    step.config["see_through_quant_mode"] = "none"
    step.config["see_through_group_offload"] = False

    asyncio.run(step._start_see_through())

    assert run_calls == []
    assert notifications
    assert notifications[-1][1]["type"] == "warning"


def test_tools_step_see_through_uses_input_dir_when_output_blank(monkeypatch, tmp_path):
    step = ToolsStep()
    input_dir = tmp_path / "images"
    input_dir.mkdir()

    captured = {}

    async def fake_run_job(script_key, args, name, **kwargs):
        captured["script_key"] = script_key
        captured["args"] = list(args)
        captured["name"] = name
        return SimpleNamespace(status="ok")

    notifications = []
    monkeypatch.setattr(step6_tools.ui, "notify", lambda message, **kwargs: notifications.append((message, kwargs)))

    step.see_through_input = SimpleNamespace(value=str(input_dir))
    step.see_through_output = SimpleNamespace(value="")
    step.see_through_repo_id_layerdiff = SimpleNamespace(value="layerdiff/repo")
    step.see_through_repo_id_depth = SimpleNamespace(value="marigold/repo")
    step.panel = SimpleNamespace(run_job=fake_run_job)
    step.config["see_through_quant_mode"] = "none"
    step.config["see_through_group_offload"] = False

    asyncio.run(step._start_see_through())

    assert notifications == []
    assert captured["script_key"] == "module.see_through.cli"
    assert f"--output_dir={input_dir}" in captured["args"]
    assert "--no-group_offload" in captured["args"]
