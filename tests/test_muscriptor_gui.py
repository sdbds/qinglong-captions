from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from gui.utils.i18n import I18n, get_i18n, set_language
from gui.wizard import step6_tools
from module.muscriptor_tool.catalog import (
    MUSCRIPTOR_INSTRUMENT_CATALOG_VERSION,
    OFFICIAL_INSTRUMENT_NAMES,
)

ROOT = Path(__file__).resolve().parents[1]


def test_tools_step_exposes_music_transcription_between_audio_and_sheet_music():
    tabs = list(step6_tools.ToolsStep.TOOL_TABS)
    music = ("music_transcription", "music_transcription", "piano")

    assert music in tabs
    assert tabs.index(music) == tabs.index(("audio_separator", "audio_separator", "graphic_eq")) + 1


def test_music_transcription_defaults_have_no_custom_sources():
    step = step6_tools.ToolsStep()

    assert step.config["music_transcription_model"] == "large"
    assert step.config["music_transcription_device"] == "auto"
    assert step.config["music_transcription_batch_size"] == 0
    assert step.config["music_transcription_output_formats"] == ["midi"]
    assert step.config["music_transcription_preview_mode"] == "none"
    assert step.config["music_transcription_preview_format"] == "mp3"
    assert "music_transcription_input_mode" not in step.config
    assert "music_transcription_recursive" not in step.config
    assert "music_transcription_overwrite" not in step.config
    assert "music_transcription_fail_fast" not in step.config
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

    assert step.music_transcription_input.selection_type == "file_or_dir"
    assert step.music_transcription_input.placeholder == step6_tools.t(
        "music_transcription_input_placeholder"
    )
    assert step.music_transcription_model.value == "large"
    assert step.music_transcription_model.options == {
        "small": "MuScriptor/muscriptor-small",
        "medium": "MuScriptor/muscriptor-medium",
        "large": "MuScriptor/muscriptor-large",
    }
    assert step.music_transcription_output_formats.value == ["midi"]
    assert step.music_transcription_preview_format.value == "mp3"
    assert step._music_transcription_gpu_summary_label is not None
    assert step._music_transcription_gpu_summary_meta_container is not None
    assert step._music_transcription_gpu_details_container is not None
    assert tuple(step.music_transcription_instruments.options) == OFFICIAL_INSTRUMENT_NAMES
    assert "option" in step.music_transcription_instruments.slots
    assert "props.opt.value" in step.music_transcription_instruments.slots["option"].template
    assert not hasattr(step, "music_transcription_output")


def _configured_step(tmp_path: Path, *, preview_mode: str = "comparison"):
    step = step6_tools.ToolsStep()
    input_dir = tmp_path / "audio"
    input_dir.mkdir()
    step.music_transcription_input = SimpleNamespace(value=str(input_dir))
    step.config.update(
        {
            "music_transcription_model": "large",
            "music_transcription_device": "cuda:0",
            "music_transcription_batch_size": 3,
            "music_transcription_instrument_mode": "specify",
            "music_transcription_instruments": ["acoustic_piano", "drums"],
            "music_transcription_decode_mode": "sampling",
            "music_transcription_temperature": 0.8,
            "music_transcription_cfg_coef": 1.5,
            "music_transcription_strict_eos": True,
            "music_transcription_beam_size": 4,
            "music_transcription_output_formats": ["midi", "jsonl"],
            "music_transcription_preview_mode": preview_mode,
            "music_transcription_preview_format": "mp3",
            "music_transcription_skip_completed": False,
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
        "--instruments=acoustic_piano,drums",
        "--decode-mode=sampling",
        "--temperature=0.8",
        "--cfg-coef=1.5",
        "--strict-eos",
        "--format=midi",
        "--format=jsonl",
        "--preview-mode=comparison",
        "--preview-format=mp3",
        "--no-skip-completed",
        "--notes",
    ):
        assert argument in captured["args"]
    assert not any("soundfont" in argument for argument in captured["args"])
    assert not any(argument.startswith("--output-dir") for argument in captured["args"])
    assert not any("recursive" in argument for argument in captured["args"])
    assert "--overwrite" not in captured["args"]
    assert "--fail-fast" not in captured["args"]
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


def test_pinned_instrument_catalog_is_immediately_available_without_runtime_probe():
    step = step6_tools.ToolsStep()

    class FakeContainer:
        visible = None

        def set_visibility(self, visible):
            self.visible = visible

    container = FakeContainer()
    step._music_transcription_instrument_container = container
    step._on_music_instrument_mode_change("specify")

    assert MUSCRIPTOR_INSTRUMENT_CATALOG_VERSION == "0.2.1"
    assert len(OFFICIAL_INSTRUMENT_NAMES) == 35
    assert len(set(OFFICIAL_INSTRUMENT_NAMES)) == 35
    assert tuple(step._music_transcription_instrument_options()) == OFFICIAL_INSTRUMENT_NAMES
    assert tuple(step6_tools.MUSCRIPTOR_INSTRUMENT_ICONS) == OFFICIAL_INSTRUMENT_NAMES
    assert step.config["music_transcription_instrument_mode"] == "specify"
    assert container.visible is True
    assert not hasattr(step, "_music_transcription_instrument_loading")
    assert not hasattr(step, "_ensure_music_instruments")


def test_music_transcription_instruments_are_localized_in_all_gui_languages():
    acoustic_piano_labels = {
        "en": "Acoustic Piano",
        "zh": "原声钢琴",
        "ja": "アコースティックピアノ",
        "ko": "어쿠스틱 피아노",
    }
    audio_terms = {"en": "audio", "zh": "音频", "ja": "音声", "ko": "오디오"}

    original_language = get_i18n().lang
    try:
        for lang in ("en", "zh", "ja", "ko"):
            i18n = I18n(lang)
            names = i18n.t("music_transcription_instrument_names")
            set_language(lang)

            assert step6_tools.ToolsStep._music_transcription_instrument_options() == names
            assert tuple(names) == OFFICIAL_INSTRUMENT_NAMES
            assert all(names[name] != name for name in OFFICIAL_INSTRUMENT_NAMES)
            assert names["acoustic_piano"] == acoustic_piano_labels[lang]
            assert audio_terms[lang] in i18n.t("music_transcription_input_placeholder").lower()
    finally:
        set_language(original_language)


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


def test_audio_separator_muscriptor_defaults_keep_other_stem_on_auto():
    step = step6_tools.ToolsStep()

    assert step.config["audio_separator_muscriptor_midi"] is False
    assert step.config["audio_separator_muscriptor_model"] == "large"
    assert step.config["audio_separator_muscriptor_device"] == "auto"
    assert step.config["audio_separator_muscriptor_batch_size"] == 0
    assert step.config["audio_separator_muscriptor_other_mode"] == "auto"
    assert step.config["audio_separator_muscriptor_other_instruments"] == []
    assert step.config["audio_separator_muscriptor_preview_mode"] == "none"
    assert step.config["audio_separator_muscriptor_preview_format"] == "mp3"


def test_audio_separator_maps_muscriptor_stem_midi_args_and_dependency(monkeypatch, tmp_path: Path):
    step = step6_tools.ToolsStep()
    input_dir = tmp_path / "audio"
    input_dir.mkdir()
    step.audio_separator_input = SimpleNamespace(value=str(input_dir))
    step.audio_separator_output_format = SimpleNamespace(value="flac")
    step.config.update(
        {
            "audio_separator_muscriptor_midi": True,
            "audio_separator_muscriptor_model": "medium",
            "audio_separator_muscriptor_device": "cuda:0",
            "audio_separator_muscriptor_batch_size": 3,
            "audio_separator_muscriptor_other_mode": "specify",
            "audio_separator_muscriptor_other_instruments": ["violin", "flutes"],
            "audio_separator_muscriptor_preview_mode": "comparison",
            "audio_separator_muscriptor_preview_format": "wav",
        }
    )
    captured = {}

    async def fake_run_job(script_key, args, name, **kwargs):
        captured.update(script_key=script_key, args=list(args), name=name, kwargs=kwargs)
        return SimpleNamespace(status="ok")

    step.panel = SimpleNamespace(run_job=fake_run_job)
    notifications = []
    monkeypatch.setattr(step6_tools.ui, "notify", lambda message, **kwargs: notifications.append((message, kwargs)))

    asyncio.run(step._start_audio_separator())

    assert notifications == []
    assert captured["script_key"] == "module.audio_separator"
    assert "--muscriptor_midi" in captured["args"]
    assert "--muscriptor_model=medium" in captured["args"]
    assert "--muscriptor_device=cuda:0" in captured["args"]
    assert "--muscriptor_batch_size=3" in captured["args"]
    assert "--muscriptor_other_instruments=violin,flutes" in captured["args"]
    assert "--muscriptor_preview_mode=comparison" in captured["args"]
    assert "--muscriptor_preview_format=wav" in captured["args"]
    assert captured["kwargs"]["runner_kwargs"] == {
        "uv_extra_args": ["--extra", "muscriptor-local"]
    }


def test_audio_separator_muscriptor_auto_other_omits_instrument_constraint(monkeypatch, tmp_path: Path):
    step = step6_tools.ToolsStep()
    input_dir = tmp_path / "audio"
    input_dir.mkdir()
    step.audio_separator_input = SimpleNamespace(value=str(input_dir))
    step.audio_separator_output_format = SimpleNamespace(value="wav")
    step.config["audio_separator_muscriptor_midi"] = True
    captured = {}

    async def fake_run_job(_script_key, args, name, **kwargs):
        captured.update(args=list(args), kwargs=kwargs)
        return SimpleNamespace(status="ok")

    step.panel = SimpleNamespace(run_job=fake_run_job)
    monkeypatch.setattr(step6_tools.ui, "notify", lambda *_args, **_kwargs: None)

    asyncio.run(step._start_audio_separator())

    assert not any(arg.startswith("--muscriptor_other_instruments") for arg in captured["args"])
    assert not any(arg.startswith("--muscriptor_preview") for arg in captured["args"])
    assert captured["kwargs"]["runner_kwargs"]["uv_extra_args"] == [
        "--extra",
        "muscriptor-local",
    ]
