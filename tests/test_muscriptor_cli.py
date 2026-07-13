from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
from click import unstyle
from typer.testing import CliRunner

from module.muscriptor_tool import cli

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[1]


def test_chunk_progress_reuses_one_task_for_sequential_batch_files(monkeypatch):
    calls = {"start": 0, "stop": 0, "add": [], "update": []}

    class FakeProgress:
        def start(self):
            calls["start"] += 1

        def stop(self):
            calls["stop"] += 1

        def add_task(self, description, **kwargs):
            calls["add"].append((description, kwargs))
            return 7

        def update(self, task_id, **kwargs):
            calls["update"].append((task_id, kwargs))

    monkeypatch.setattr(cli, "create_caption_progress", lambda *_args, **_kwargs: FakeProgress())

    with cli._ChunkProgressReporter(SimpleNamespace()) as reporter:
        reporter.update("disc1/song.wav", 0, 4)
        reporter.update("disc1/song.wav", 4, 4)
        reporter.update("disc2/song.wav", 0, 8)

    assert calls["start"] == 1
    assert calls["stop"] == 1
    assert len(calls["add"]) == 1
    assert len(calls["update"]) == 2
    assert calls["update"][-1][0] == 7
    assert "disc2/song.wav" in calls["update"][-1][1]["description"]


def test_transcribe_help_exposes_model_capabilities_without_custom_sources():
    result = runner.invoke(cli.app, ["transcribe", "--help"])

    assert result.exit_code == 0, result.stdout
    for option in (
        "--model",
        "--device",
        "--sampling",
        "--temperature",
        "--cfg-coef",
        "--batch-size",
        "--strict-eos",
        "--beam-size",
        "--preview",
        "--preview-mode",
        "--instruments",
    ):
        assert option in result.stdout
    assert "--soundfont" not in result.stdout
    assert "PATH|URL" not in result.stdout


def test_batch_help_has_complete_batch_surface():
    result = runner.invoke(cli.app, ["batch", "--help"])

    assert result.exit_code == 0, result.stdout
    for option in (
        "--output-dir",
        "--format",
        "--preview-mode",
        "--preview-format",
        "--decode-mode",
        "--recursive",
        "--skip-completed",
        "--fail-fast",
        "--notes",
    ):
        assert option in result.stdout
    assert "--soundfont" not in result.stdout
    assert "--overwrite" not in result.stdout
    assert "5-second audio" in result.stdout
    assert "chunks per" in result.stdout


def _install_single_backend(monkeypatch, calls: list[str]) -> None:
    def resolve(values):
        calls.append("resolve")
        return tuple(values)

    def preflight(request):
        calls.append("preflight")
        return SimpleNamespace(request=request)

    def load(options, *, console=None):
        calls.append("load")
        assert isinstance(console, cli.Console)
        return SimpleNamespace(
            package_version="0.2.1",
            requested_device=options.device,
            resolved_device="cpu",
        )

    def transcribe(_loaded, _source, _options, targets, **kwargs):
        calls.append("transcribe")
        progress_callback = kwargs.get("progress_callback")
        if progress_callback is not None:
            progress_callback(1, 2)
        for name, target in targets.requested_paths().items():
            target.parent.mkdir(parents=True, exist_ok=True)
            if name == "midi":
                target.write_bytes(b"MThd")
            elif name == "json":
                target.write_text('[{"type":"start"}]\n', encoding="utf-8")
            else:
                target.write_text('{"type":"start"}\n', encoding="utf-8")
        if targets.jsonl_stream is not None:
            targets.jsonl_stream.write('{"type":"start"}\n')
            targets.jsonl_stream.flush()
        if targets.midi_stream is not None:
            targets.midi_stream.write(b"MThd")
            targets.midi_stream.flush()
        preview_target = kwargs.get("preview_target")
        if preview_target is not None:
            preview_target.write_bytes(b"preview")
        return SimpleNamespace(warnings=(), outputs={})

    monkeypatch.setattr(cli, "resolve_instruments", resolve)
    monkeypatch.setattr(cli, "preflight_preview", preflight)
    monkeypatch.setattr(cli, "load_model", load)
    monkeypatch.setattr(cli, "transcribe_once", transcribe)


def test_jsonl_stdout_stays_machine_readable_with_preview(monkeypatch, tmp_path: Path):
    calls: list[str] = []
    _install_single_backend(monkeypatch, calls)
    source = tmp_path / "song.wav"
    source.write_bytes(b"audio")
    preview = tmp_path / "preview.wav"

    result = runner.invoke(
        cli.app,
        ["transcribe", str(source), "-f", "jsonl", "-o", "-", "--preview", str(preview)],
    )

    assert result.exit_code == 0, result.stdout
    assert [json.loads(line) for line in result.stdout.splitlines()] == [{"type": "start"}]
    assert "Loading model" not in result.stdout
    stderr = unstyle(result.stderr)
    assert "Transcribing song.wav" in stderr
    assert "1/2" in stderr
    assert preview.read_bytes() == b"preview"
    assert calls == ["resolve", "preflight", "load", "transcribe"]


def test_input_main_output_and_preview_must_be_distinct(tmp_path: Path):
    source = tmp_path / "song.wav"
    source.write_bytes(b"audio")

    same_output = runner.invoke(cli.app, ["transcribe", str(source), "-o", str(source)])
    same_preview = runner.invoke(
        cli.app,
        ["transcribe", str(source), "-o", str(tmp_path / "song.mid"), "--preview", str(source)],
    )

    assert same_output.exit_code == 2
    assert same_preview.exit_code == 2


def test_default_output_is_written_next_to_input_with_selected_extension(monkeypatch, tmp_path: Path):
    calls: list[str] = []
    _install_single_backend(monkeypatch, calls)
    source = tmp_path / "song.wav"
    source.write_bytes(b"audio")

    result = runner.invoke(cli.app, ["transcribe", str(source), "--format", "json"])

    assert result.exit_code == 0, result.stderr
    assert json.loads((tmp_path / "song.json").read_text(encoding="utf-8")) == [{"type": "start"}]


def test_midi_stdout_is_binary_and_contains_no_logs(monkeypatch, tmp_path: Path):
    calls: list[str] = []
    _install_single_backend(monkeypatch, calls)
    source = tmp_path / "song.wav"
    source.write_bytes(b"audio")

    result = runner.invoke(cli.app, ["transcribe", str(source), "--output", "-"])

    assert result.exit_code == 0
    assert result.stdout_bytes == b"MThd"
    assert b"Loading" not in result.stdout_bytes
    assert "Loading official MuScriptor" in result.stderr


def test_preview_mode_without_preview_and_preview_format_while_off_are_parameter_errors(tmp_path: Path):
    source = tmp_path / "song.wav"
    source.write_bytes(b"audio")

    single = runner.invoke(cli.app, ["transcribe", str(source), "--preview-mode", "midi"])
    batch = runner.invoke(cli.app, ["batch", str(source), "--preview-format", "wav"])

    assert single.exit_code == 2
    assert batch.exit_code == 2


def test_single_rejects_non_official_model_and_invalid_preview_extension(tmp_path: Path):
    source = tmp_path / "song.wav"
    source.write_bytes(b"audio")

    custom_model = runner.invoke(cli.app, ["transcribe", str(source), "--model", "Org/custom"])
    bad_preview = runner.invoke(
        cli.app,
        ["transcribe", str(source), "--preview", str(tmp_path / "preview.ogg")],
    )

    assert custom_model.exit_code == 2
    assert bad_preview.exit_code == 2


def test_invalid_instrument_is_a_parameter_error_before_model_loading(monkeypatch, tmp_path: Path):
    source = tmp_path / "song.wav"
    source.write_bytes(b"audio")
    monkeypatch.setattr(cli, "resolve_instruments", lambda _values: (_ for _ in ()).throw(ValueError("unknown instrument")))
    monkeypatch.setattr(cli, "load_model", lambda _options: (_ for _ in ()).throw(AssertionError("model loaded")))

    result = runner.invoke(cli.app, ["transcribe", str(source), "--instruments", "unknown"])

    assert result.exit_code == 2
    assert "unknown instrument" in result.stderr


def test_batch_normalizes_repeatable_formats_and_omits_auto_batch_size(monkeypatch, tmp_path: Path):
    captured = {}
    source = tmp_path / "inputs"
    source.mkdir()
    (source / "song.wav").write_bytes(b"audio")

    monkeypatch.setattr(cli, "resolve_instruments", lambda values: tuple(values))

    def fake_load(_options, *, console=None):
        captured["model_console"] = console
        return SimpleNamespace(resolved_device="cpu")

    monkeypatch.setattr(cli, "load_model", fake_load)

    def fake_run(input_path, output_dir, options, **kwargs):
        captured.update(input_path=input_path, output_dir=output_dir, options=options, kwargs=kwargs)
        kwargs["model_loader"](options.transcription)
        kwargs["log_callback"]("Processing input: song.wav")
        kwargs["chunk_progress_callback"]("song.wav", 1, 1)
        return SimpleNamespace(
            discovered=1,
            processed=1,
            skipped=0,
            partial=0,
            failed=0,
            elapsed_seconds=0.1,
            exit_code=0,
        )

    monkeypatch.setattr(cli, "run_batch", fake_run)
    result = runner.invoke(
        cli.app,
        [
            "batch",
            str(source),
            "--output-dir",
            str(tmp_path / "out"),
            "--format",
            "json",
            "--format",
            "jsonl",
            "--batch-size",
            "0",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["options"].transcription.batch_size is None
    assert [item.value for item in captured["options"].output_formats] == ["json", "jsonl"]
    assert isinstance(captured["model_console"], cli.Console)
    stderr = unstyle(result.stderr)
    assert "processed=1" in stderr
    assert "Transcribing song.wav" in stderr
    assert "1/1" in stderr


def test_batch_accepts_explicit_preview_none_without_preflight(monkeypatch, tmp_path: Path):
    source = tmp_path / "song.wav"
    source.write_bytes(b"audio")
    captured = {}
    monkeypatch.setattr(cli, "resolve_instruments", lambda values: tuple(values))

    def fake_run(input_path, output_dir, options, **_kwargs):
        captured["input_path"] = input_path
        captured["output_dir"] = output_dir
        captured["model"] = options.transcription.model.value
        captured["preview"] = options.preview
        return SimpleNamespace(
            discovered=1,
            processed=1,
            skipped=0,
            partial=0,
            failed=0,
            elapsed_seconds=0.0,
            exit_code=0,
        )

    monkeypatch.setattr(cli, "run_batch", fake_run)
    result = runner.invoke(cli.app, ["batch", str(source), "--preview-mode", "none"])

    assert result.exit_code == 0, result.stdout
    assert captured["preview"] is None
    assert captured["model"] == "large"
    assert captured["output_dir"] == tmp_path / "muscriptor_output"


def test_batch_preview_defaults_to_mp3(monkeypatch, tmp_path: Path):
    source = tmp_path / "song.wav"
    source.write_bytes(b"audio")
    captured = {}
    monkeypatch.setattr(cli, "resolve_instruments", lambda values: tuple(values))

    def fake_run(_input_path, _output_dir, options, **_kwargs):
        captured["preview"] = options.preview
        return SimpleNamespace(
            discovered=1,
            processed=1,
            skipped=0,
            partial=0,
            failed=0,
            elapsed_seconds=0.0,
            exit_code=0,
        )

    monkeypatch.setattr(cli, "run_batch", fake_run)

    result = runner.invoke(cli.app, ["batch", str(source), "--preview-mode", "midi"])

    assert result.exit_code == 0, result.stdout
    assert captured["preview"].format.value == "mp3"


def test_list_instruments_json_matches_text_order(monkeypatch):
    monkeypatch.setattr(cli, "list_instruments", lambda: ("piano", "drums"))
    monkeypatch.setattr(cli, "muscriptor_version", lambda: "0.2.1")

    text_result = runner.invoke(cli.app, ["list-instruments"])
    json_result = runner.invoke(cli.app, ["list-instruments", "--format", "json"])

    assert text_result.stdout.splitlines() == ["piano", "drums"]
    assert json.loads(json_result.stdout) == {
        "schema_version": 1,
        "package_version": "0.2.1",
        "instruments": ["piano", "drums"],
    }


def test_runtime_failure_exits_one_and_interrupt_exits_130(monkeypatch, tmp_path: Path):
    source = tmp_path / "song.wav"
    source.write_bytes(b"audio")
    monkeypatch.setattr(cli, "resolve_instruments", lambda _values: ())
    monkeypatch.setattr(
        cli,
        "load_model",
        lambda _options, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    failed = runner.invoke(cli.app, ["transcribe", str(source)])
    monkeypatch.setattr(
        cli,
        "load_model",
        lambda _options, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    interrupted = runner.invoke(cli.app, ["transcribe", str(source)])

    assert failed.exit_code == 1
    assert "boom" in failed.stderr
    assert interrupted.exit_code == 130


def _real_smoke_python() -> str:
    python_path = os.getenv("MUSCRIPTOR_SMOKE_PYTHON", sys.executable)
    probe = subprocess.run(
        [
            python_path,
            "-c",
            "import importlib.metadata as m; print(m.version('muscriptor'))",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if probe.returncode != 0 or probe.stdout.strip() != "0.2.1":
        pytest.fail(
            "MUSCRIPTOR_SMOKE requires a muscriptor-local Python. Set "
            "MUSCRIPTOR_SMOKE_PYTHON after installing the profile."
        )
    return python_path


@pytest.fixture
def muscriptor_sample_audio(tmp_path: Path) -> Path:
    sample_rate = 16_000
    source = tmp_path / "sample.wav"
    frames = bytearray()
    for index in range(sample_rate * 2):
        sample = int(0.15 * 32767 * math.sin(2 * math.pi * 440 * index / sample_rate))
        frames.extend(struct.pack("<h", sample))
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(frames)
    return source


@pytest.mark.optional_runtime
@pytest.mark.network
@pytest.mark.skipif(os.getenv("MUSCRIPTOR_SMOKE") != "1", reason="requires gated official weights")
def test_real_small_cpu_midi_smoke(muscriptor_sample_audio: Path, tmp_path: Path):
    output = tmp_path / "transcription.mid"
    result = subprocess.run(
        [
            _real_smoke_python(),
            "-m",
            "module.muscriptor_tool.cli",
            "transcribe",
            str(muscriptor_sample_audio),
            "--model",
            "small",
            "--device",
            "cpu",
            "--format",
            "midi",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_bytes().startswith(b"MThd")


@pytest.mark.optional_runtime
@pytest.mark.network
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("MUSCRIPTOR_SMOKE_CUDA") != "1", reason="requires CUDA and gated weights")
def test_real_cuda_batch_smoke(muscriptor_sample_audio: Path, tmp_path: Path):
    output_dir = tmp_path / "batch"
    result = subprocess.run(
        [
            _real_smoke_python(),
            "-m",
            "module.muscriptor_tool.cli",
            "batch",
            str(muscriptor_sample_audio),
            "--output-dir",
            str(output_dir),
            "--model",
            "small",
            "--device",
            "cuda",
            "--format",
            "midi",
            "--format",
            "jsonl",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "sample.wav" / "sample.mid").is_file()
    assert (output_dir / "sample.wav" / "events.jsonl").is_file()


@pytest.mark.optional_runtime
@pytest.mark.network
@pytest.mark.skipif(
    os.getenv("MUSCRIPTOR_SMOKE_PREVIEW") != "1" or shutil.which("fluidsynth") is None,
    reason="requires FluidSynth, gated weights, and the official SoundFont",
)
@pytest.mark.parametrize("preview_mode", ("midi", "comparison"))
def test_real_wav_preview_smoke(muscriptor_sample_audio: Path, tmp_path: Path, preview_mode: str):
    midi_output = tmp_path / "transcription.mid"
    preview_output = tmp_path / f"preview-{preview_mode}.wav"
    result = subprocess.run(
        [
            _real_smoke_python(),
            "-m",
            "module.muscriptor_tool.cli",
            "transcribe",
            str(muscriptor_sample_audio),
            "--model",
            "small",
            "--device",
            "cpu",
            "--output",
            str(midi_output),
            "--preview",
            str(preview_output),
            "--preview-mode",
            preview_mode,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert preview_output.read_bytes().startswith(b"RIFF")


@pytest.mark.optional_runtime
@pytest.mark.network
@pytest.mark.skipif(
    os.getenv("MUSCRIPTOR_SMOKE_MP3") != "1" or shutil.which("fluidsynth") is None,
    reason="requires a writable MP3 codec, FluidSynth, gated weights, and the official SoundFont",
)
def test_real_midi_mp3_preview_smoke(muscriptor_sample_audio: Path, tmp_path: Path):
    preview_output = tmp_path / "preview.mp3"
    result = subprocess.run(
        [
            _real_smoke_python(),
            "-m",
            "module.muscriptor_tool.cli",
            "transcribe",
            str(muscriptor_sample_audio),
            "--model",
            "small",
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "transcription.mid"),
            "--preview",
            str(preview_output),
            "--preview-mode",
            "midi",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert preview_output.stat().st_size > 0
