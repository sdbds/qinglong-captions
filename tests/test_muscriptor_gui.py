from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from gui.wizard import step6_tools

ROOT = Path(__file__).resolve().parents[1]


def test_tools_step_exposes_music_transcription_between_audio_and_sheet_music():
    tabs = list(step6_tools.ToolsStep.TOOL_TABS)
    music = ("music_transcription", "music_transcription", "piano")

    assert music in tabs
    assert tabs.index(music) == tabs.index(("audio_separator", "audio_separator", "graphic_eq")) + 1


def test_music_transcription_defaults_have_no_custom_sources():
    step = step6_tools.ToolsStep()

    assert step.config["music_transcription_model"] == "medium"
    assert step.config["music_transcription_device"] == "auto"
    assert step.config["music_transcription_batch_size"] == 0
    assert step.config["music_transcription_output_formats"] == ["midi"]
    assert step.config["music_transcription_preview_mode"] == "none"
    assert not any("soundfont" in key or "model_source" in key for key in step.config)


def test_music_transcription_tab_maps_renderer_and_action():
    step = step6_tools.ToolsStep()

    assert step._get_tool_renderer("music_transcription") == step._render_music_transcription_tool
    assert step._tool_action_for_tab("music_transcription") == (
        "start_music_transcription",
        step._start_music_transcription,
    )


def test_music_transcription_tool_renders_complete_controls():
    step = step6_tools.ToolsStep()

    step._render_music_transcription_tool()

    assert step.music_transcription_input.selection_type == "dir"
    assert step.music_transcription_model.value == "medium"
    assert step.music_transcription_output_formats.value == ["midi"]


def _configured_step(tmp_path: Path, *, preview_mode: str = "comparison"):
    step = step6_tools.ToolsStep()
    input_dir = tmp_path / "audio"
    input_dir.mkdir()
    step.music_transcription_input = SimpleNamespace(value=str(input_dir))
    step.music_transcription_output = SimpleNamespace(value=str(tmp_path / "out"))
    step.config.update(
        {
            "music_transcription_model": "large",
            "music_transcription_device": "cuda:0",
            "music_transcription_batch_size": 3,
            "music_transcription_instrument_mode": "specify",
            "music_transcription_instruments": ["piano", "drums"],
            "music_transcription_decode_mode": "sampling",
            "music_transcription_temperature": 0.8,
            "music_transcription_cfg_coef": 1.5,
            "music_transcription_strict_eos": True,
            "music_transcription_beam_size": 4,
            "music_transcription_output_formats": ["midi", "jsonl"],
            "music_transcription_preview_mode": preview_mode,
            "music_transcription_preview_format": "mp3",
            "music_transcription_recursive": False,
            "music_transcription_skip_completed": False,
            "music_transcription_overwrite": True,
            "music_transcription_fail_fast": True,
            "music_transcription_notes": True,
        }
    )
    return step


def test_music_transcription_maps_complete_batch_args(monkeypatch, tmp_path: Path):
    step = _configured_step(tmp_path)
    captured = {}

    async def fake_run_job(script_key, args, name, **kwargs):
        captured.update(script_key=script_key, args=list(args), name=name, kwargs=kwargs)
        return SimpleNamespace(status="ok")

    step.panel = SimpleNamespace(run_job=fake_run_job)
    notifications = []
    monkeypatch.setattr(step6_tools.ui, "notify", lambda message, **kwargs: notifications.append((message, kwargs)))

    asyncio.run(step._start_music_transcription())

    assert notifications == []
    assert captured["script_key"] == "module.muscriptor_tool.cli"
    assert captured["name"] == step6_tools.t("job_name_music_transcription")
    assert captured["args"][0] == "batch"
    for argument in (
        "--model=large",
        "--device=cuda:0",
        "--batch-size=3",
        "--instruments=piano,drums",
        "--decode-mode=sampling",
        "--temperature=0.8",
        "--cfg-coef=1.5",
        "--strict-eos",
        "--format=midi",
        "--format=jsonl",
        "--preview-mode=comparison",
        "--preview-format=mp3",
        "--no-recursive",
        "--no-skip-completed",
        "--overwrite",
        "--fail-fast",
        "--notes",
    ):
        assert argument in captured["args"]
    assert not any("soundfont" in argument for argument in captured["args"])
    assert not any(argument.startswith("--beam-size") for argument in captured["args"])


def test_preview_off_and_auto_batch_size_omit_inactive_arguments(monkeypatch, tmp_path: Path):
    step = _configured_step(tmp_path, preview_mode="none")
    step.config["music_transcription_batch_size"] = 0
    step.config["music_transcription_instrument_mode"] = "auto"
    captured = {}

    async def fake_run_job(_script_key, args, name, **_kwargs):
        captured["args"] = list(args)
        return SimpleNamespace(status="ok")

    step.panel = SimpleNamespace(run_job=fake_run_job)
    monkeypatch.setattr(step6_tools.ui, "notify", lambda *_args, **_kwargs: None)

    asyncio.run(step._start_music_transcription())

    assert "--preview-mode=none" not in captured["args"]
    assert not any(argument.startswith("--preview-format") for argument in captured["args"])
    assert not any(argument.startswith("--batch-size") for argument in captured["args"])
    assert not any(argument.startswith("--instruments") for argument in captured["args"])


def test_beam_mode_rejects_width_one_before_job_submission(monkeypatch, tmp_path: Path):
    step = _configured_step(tmp_path)
    step.config["music_transcription_decode_mode"] = "beam"
    step.config["music_transcription_beam_size"] = 1
    step.panel = SimpleNamespace(run_job=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("submitted")))
    notifications = []
    monkeypatch.setattr(step6_tools.ui, "notify", lambda message, **kwargs: notifications.append((message, kwargs)))

    asyncio.run(step._start_music_transcription())

    assert notifications
    assert notifications[-1][1]["type"] == "warning"


def test_instrument_catalog_parser_caches_by_package_version():
    step6_tools._MUSCRIPTOR_INSTRUMENT_CACHE.clear()
    payload = json.dumps(
        {
            "schema_version": 1,
            "package_version": "0.2.1",
            "instruments": ["piano", "drums"],
        }
    )

    names = step6_tools._parse_music_instrument_catalog(["install log", payload])

    assert names == ("piano", "drums")
    assert step6_tools._MUSCRIPTOR_INSTRUMENT_CACHE["0.2.1"] == names


def test_lazy_instrument_probe_uses_current_task_runtime(monkeypatch, tmp_path: Path):
    from gui.utils.process_runner import ProcessResult, ProcessRunner, ProcessStatus

    step6_tools._MUSCRIPTOR_INSTRUMENT_CACHE.clear()
    captured = {}

    class FakeTabs:
        async def ensure_active_tab_runtime_ready(self):
            return True

        def runner_kwargs(self):
            return {
                "tab_id": "tab-2",
                "tab_name": "Task 2",
                "python_path": str(tmp_path / "python"),
                "venv_path": str(tmp_path / "venv"),
            }

    async def fake_run(self, script_key, args, **kwargs):
        captured.update(script_key=script_key, args=list(args), kwargs=kwargs)
        self._log_buffer.push(
            json.dumps(
                {
                    "schema_version": 1,
                    "package_version": "0.2.1",
                    "instruments": ["piano", "drums"],
                }
            )
        )
        return ProcessResult(ProcessStatus.SUCCESS, 0, "ok")

    monkeypatch.setattr(ProcessRunner, "run_python_script", fake_run)
    step = step6_tools.ToolsStep()
    step.panel = SimpleNamespace(execution_tabs=FakeTabs())

    names = asyncio.run(step._probe_music_instruments())

    assert names == ("piano", "drums")
    assert captured["script_key"] == "module.muscriptor_tool.cli"
    assert captured["args"] == ["list-instruments", "--format", "json"]
    assert captured["kwargs"]["python_path"] == str(tmp_path / "python")
    assert captured["kwargs"]["venv_path"] == str(tmp_path / "venv")
    assert captured["kwargs"]["native_console"] is False


def test_importing_tools_step_does_not_import_torch_or_muscriptor():
    code = (
        "import json,sys; "
        "before=set(sys.modules); "
        "import gui.wizard.step6_tools; "
        "added=set(sys.modules)-before; "
        "print(json.dumps(sorted(name for name in added if name == 'torch' or name.startswith('muscriptor'))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )

    assert json.loads(result.stdout) == []
