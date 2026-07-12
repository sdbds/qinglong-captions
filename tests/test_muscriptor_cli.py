from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from module.muscriptor_tool import cli

runner = CliRunner()


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
        "--overwrite",
        "--fail-fast",
        "--notes",
    ):
        assert option in result.stdout
    assert "--soundfont" not in result.stdout


def _install_single_backend(monkeypatch, calls: list[str]) -> None:
    def resolve(values):
        calls.append("resolve")
        return tuple(values)

    def preflight(request):
        calls.append("preflight")
        return SimpleNamespace(request=request)

    def load(options):
        calls.append("load")
        return SimpleNamespace(
            package_version="0.2.1",
            requested_device=options.device,
            resolved_device="cpu",
        )

    def transcribe(_loaded, _source, _options, targets, **kwargs):
        calls.append("transcribe")
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


def test_batch_normalizes_repeatable_formats_and_omits_auto_batch_size(monkeypatch, tmp_path: Path):
    captured = {}
    source = tmp_path / "inputs"
    source.mkdir()
    (source / "song.wav").write_bytes(b"audio")

    monkeypatch.setattr(cli, "resolve_instruments", lambda values: tuple(values))

    def fake_run(input_path, output_dir, options):
        captured.update(input_path=input_path, output_dir=output_dir, options=options)
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
    assert "processed=1" in result.stderr


def test_batch_accepts_explicit_preview_none_without_preflight(monkeypatch, tmp_path: Path):
    source = tmp_path / "song.wav"
    source.write_bytes(b"audio")
    captured = {}
    monkeypatch.setattr(cli, "resolve_instruments", lambda values: tuple(values))

    def fake_run(_input_path, _output_dir, options):
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
    monkeypatch.setattr(cli, "load_model", lambda _options: (_ for _ in ()).throw(RuntimeError("boom")))

    failed = runner.invoke(cli.app, ["transcribe", str(source)])
    monkeypatch.setattr(cli, "load_model", lambda _options: (_ for _ in ()).throw(KeyboardInterrupt()))
    interrupted = runner.invoke(cli.app, ["transcribe", str(source)])

    assert failed.exit_code == 1
    assert "boom" in failed.stderr
    assert interrupted.exit_code == 130
