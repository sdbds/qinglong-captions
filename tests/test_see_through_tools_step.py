import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent

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


def test_tools_step_see_through_maps_auto_rig_followup(monkeypatch, tmp_path):
    step = ToolsStep()
    input_dir = tmp_path / "images"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    sdk_root = tmp_path / "CubismSdkForNative-5-r.5"
    sdk_root.mkdir()
    spine_path = tmp_path / "auto_rig_spine_runtime.exe"
    spine_path.write_bytes(b"runtime")

    captured = {}

    async def fake_run_job(script_key, args, name, **kwargs):
        captured["script_key"] = script_key
        captured["args"] = list(args)
        captured["kwargs"] = kwargs
        return SimpleNamespace(status="ok")

    monkeypatch.setattr(step6_tools.ui, "notify", lambda *args, **kwargs: None)
    step.see_through_input = SimpleNamespace(value=str(input_dir))
    step.see_through_output = SimpleNamespace(value=str(output_dir))
    step.see_through_repo_id_layerdiff = SimpleNamespace(value="layerdiff/repo")
    step.see_through_repo_id_depth = SimpleNamespace(value="marigold/repo")
    step.see_through_auto_rig_profile = SimpleNamespace(value="dual_runtime_avatar_v1")
    step.see_through_auto_rig_pose_mode = SimpleNamespace(value="compare")
    step.see_through_auto_rig_pose_device = SimpleNamespace(value="cuda")
    step.see_through_auto_rig_sdk_root = SimpleNamespace(value=str(sdk_root))
    step.see_through_auto_rig_spine_runtime_path = SimpleNamespace(value=str(spine_path))
    step.panel = SimpleNamespace(run_job=fake_run_job)
    step.config.update(
        {
            "see_through_auto_rig": True,
            "see_through_auto_rig_profile": "dual_runtime_avatar_v1",
            "see_through_auto_rig_pose_mode": "compare",
            "see_through_auto_rig_pose_device": "cuda",
            "see_through_auto_rig_pose_fa2": False,
            "see_through_save_to_psd": True,
        }
    )

    asyncio.run(step._start_see_through())

    assert captured["script_key"] == "module.see_through.cli"
    assert "--auto_rig" in captured["args"]
    assert "--auto_rig_profile=dual_runtime_avatar_v1" in captured["args"]
    assert "--auto_rig_pose_mode=compare" in captured["args"]
    assert "--auto_rig_pose_device=cuda" in captured["args"]
    assert "--no-auto_rig_pose_fa2" in captured["args"]
    assert f"--auto_rig_sdk_root={sdk_root}" in captured["args"]
    assert not any("renderer_path" in argument for argument in captured["args"])
    assert "see_through_auto_rig_renderer_path" not in step.config
    assert f"--auto_rig_spine_runtime_path={spine_path}" in captured["args"]
    assert captured["kwargs"]["runner_kwargs"] == {
        "uv_extra_args": ["--extra", "auto-rig-pose"]
    }


def test_tools_step_auto_rig_toggle_reveals_options_and_enables_psd():
    class DummyContainer:
        visible = None

        def set_visibility(self, visible):
            self.visible = visible

    class DummyToggle:
        value = None

        def set_toggle_value(self, value):
            self.value = value

    step = ToolsStep()
    step._see_through_auto_rig_container = DummyContainer()
    step.see_through_save_to_psd_toggle = DummyToggle()
    step.config["see_through_save_to_psd"] = False

    step._on_see_through_auto_rig_toggle(True)

    assert step._see_through_auto_rig_container.visible is True
    assert step.config["see_through_save_to_psd"] is True
    assert step.see_through_save_to_psd_toggle.value is True


def test_tools_step_missing_runtime_does_not_block_spine_export(monkeypatch, tmp_path):
    step = ToolsStep()
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    captured = {}

    async def fake_run_job(script_key, args, name, **kwargs):
        captured["args"] = list(args)
        return SimpleNamespace(status="ok")

    notifications = []
    monkeypatch.setattr(step6_tools.ui, "notify", lambda message, **kwargs: notifications.append((message, kwargs)))
    step.see_through_input = SimpleNamespace(value=str(input_dir))
    step.see_through_output = SimpleNamespace(value=str(tmp_path / "outputs"))
    step.see_through_repo_id_layerdiff = SimpleNamespace(value="layerdiff/repo")
    step.see_through_repo_id_depth = SimpleNamespace(value="marigold/repo")
    step.see_through_auto_rig_profile = SimpleNamespace(value="dual_runtime_core_v1")
    step.see_through_auto_rig_pose_mode = SimpleNamespace(value="auto")
    step.see_through_auto_rig_pose_device = SimpleNamespace(value="auto")
    step.see_through_auto_rig_sdk_root = SimpleNamespace(value="")
    step.see_through_auto_rig_spine_runtime_path = SimpleNamespace(value=str(tmp_path / "missing.exe"))
    step.panel = SimpleNamespace(run_job=fake_run_job)
    step.config["see_through_auto_rig"] = True
    step.config["see_through_save_to_psd"] = True

    asyncio.run(step._start_see_through())

    assert "--auto_rig" in captured["args"]
    assert not any(arg.startswith("--auto_rig_sdk_root=") for arg in captured["args"])
    assert not any("renderer_path" in arg for arg in captured["args"])
    assert not any(arg.startswith("--auto_rig_spine_runtime_path=") for arg in captured["args"])
    assert [item[1]["type"] for item in notifications].count("warning") == 2
