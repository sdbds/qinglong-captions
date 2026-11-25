from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from filelock import FileLock


class FakeModel:
    def __init__(self):
        self.calls = []
        self.failures = set()
        self.missing_scores = set()

    def transcribe(self, source, *, output_dir, dtype, melody_only, progress):
        self.calls.append((source, dtype, melody_only))
        (output_dir / "events.json").write_text('{"events": []}', encoding="utf-8")
        if source.name in self.failures:
            raise RuntimeError("inference failed after writing an annotation")
        (output_dir / "transcription.mid").write_bytes(
            b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0MTrk\x00\x00\x00\x04\x00\xff\x2f\x00"
        )
        (output_dir / "chord.lab").write_text("0.0\t1.0\tC:maj\n", encoding="utf-8")
        if source.name in self.missing_scores:
            result = {"abc": None, "abc_error": "No key was decoded"}
            if melody_only:
                error = RuntimeError("Melody-only ABC unavailable")
                error.result = result
                raise error
            return result
        score = "X:1\nM:4/4\nK:C\nC D E F|\n"
        (output_dir / "score.abc").write_text(score, encoding="utf-8")
        return {"abc": score, "abc_error": None}


def _audio(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test audio")
    return path


def test_batch_keeps_native_score_annotations_and_distinct_song_paths(monkeypatch, tmp_path):
    from module import sheetsage2_tool

    album = tmp_path / "album"
    first = _audio(album / "disc 1" / "Track 01.wav")
    second = _audio(album / "disc 1" / "Track 01.flac")
    output = album / "sheetsage2_output"
    _audio(output / "ignore.mp3")
    model = FakeModel()
    loads = []

    def load(device):
        loads.append(device)
        return model, "bf16"

    monkeypatch.setattr(sheetsage2_tool, "load_model", load)

    assert sheetsage2_tool.main([str(album), "--device=cuda:1", "--melody-only"]) == 0
    assert loads == ["cuda:1"]
    assert model.calls == [(second, "bf16", True), (first, "bf16", True)]
    for source in (first, second):
        target = output / "disc 1" / source.name
        assert (target / "score.abc").read_text(encoding="utf-8").endswith("C D E F|\n")
        assert (target / "transcription.mid").read_bytes().startswith(b"MThd")
        assert json.loads((target / "events.json").read_text(encoding="utf-8")) == {"events": []}
        assert (target / "chord.lab").read_text(encoding="utf-8") == "0.0\t1.0\tC:maj\n"


def test_batch_skips_complete_outputs_but_rebuilds_missing_artifacts_or_changed_options(monkeypatch, tmp_path):
    from module import sheetsage2_tool

    source = _audio(tmp_path / "track.wav")
    model = FakeModel()
    monkeypatch.setattr(sheetsage2_tool, "load_model", lambda device: (model, "fp32"))

    assert sheetsage2_tool.main([str(source)]) == 0
    assert sheetsage2_tool.main([str(source)]) == 0
    assert len(model.calls) == 1

    (tmp_path / "sheetsage2_output" / "track.wav" / "score.abc").unlink()
    assert sheetsage2_tool.main([str(source)]) == 0
    assert sheetsage2_tool.main([str(source), "--melody-only"]) == 0
    assert len(model.calls) == 3


def test_default_skip_preserves_an_edited_abc_score(monkeypatch, tmp_path):
    from module import sheetsage2_tool

    source = _audio(tmp_path / "track.wav")
    model = FakeModel()
    monkeypatch.setattr(sheetsage2_tool, "load_model", lambda device: (model, "fp32"))
    assert sheetsage2_tool.main([str(source)]) == 0
    score = tmp_path / "sheetsage2_output" / source.name / "score.abc"
    edited = "X:1\nT:Edited\nM:4/4\nK:C\nD E F G|\n"
    score.write_text(edited, encoding="utf-8")

    assert sheetsage2_tool.main([str(source)]) == 0
    assert score.read_text(encoding="utf-8") == edited
    assert len(model.calls) == 1


def test_shared_output_refuses_another_source_with_the_same_filename(monkeypatch, tmp_path):
    from module import sheetsage2_tool

    first = _audio(tmp_path / "album-a" / "song.wav")
    second = _audio(tmp_path / "album-b" / "song.wav")
    output = tmp_path / "archive"
    model = FakeModel()
    monkeypatch.setattr(sheetsage2_tool, "load_model", lambda device: (model, "fp32"))
    assert sheetsage2_tool.main([str(first), "--output-dir", str(output)]) == 0
    target = output / first.name
    (target / "score.abc").write_text("Edited score", encoding="utf-8")
    metadata = (target / "metadata.json").read_bytes()

    assert sheetsage2_tool.main([str(second), "--output-dir", str(output)]) == 1
    assert (target / "score.abc").read_text(encoding="utf-8") == "Edited score"
    assert (target / "metadata.json").read_bytes() == metadata
    assert len(model.calls) == 1


@pytest.mark.parametrize("stem_mode", [False, True])
def test_transcription_does_not_claim_unowned_native_filenames(monkeypatch, tmp_path, stem_mode):
    from module import sheetsage2_tool

    source = _audio(tmp_path / "song.wav")
    if stem_mode:
        manifest = tmp_path / "stems.json"
        manifest.write_text(json.dumps({"stems": [str(source)]}), encoding="utf-8")
        args = ["--stem-manifest", str(manifest)]
        target = source.parent / "04_stem_midi" / "sheetsage2" / source.name
    else:
        args = [str(source)]
        target = source.parent / "sheetsage2_output" / source.name
    target.mkdir(parents=True)
    (target / "events.json").write_text("my events", encoding="utf-8")
    model = FakeModel()
    monkeypatch.setattr(sheetsage2_tool, "load_model", lambda device: (model, "fp32"))

    assert sheetsage2_tool.main(args) == 1
    assert (target / "events.json").read_text(encoding="utf-8") == "my events"
    assert not (target / "metadata.json").exists()
    assert not model.calls


def test_owned_result_does_not_claim_a_native_file_missing_from_its_manifest(monkeypatch, tmp_path):
    from module import sheetsage2_tool

    source = _audio(tmp_path / "song.wav")
    model = FakeModel()
    monkeypatch.setattr(sheetsage2_tool, "load_model", lambda device: (model, "fp32"))
    assert sheetsage2_tool.main([str(source)]) == 0
    unrelated = source.parent / "sheetsage2_output" / source.name / "melody.mid"
    unrelated.write_bytes(b"hand-crafted MIDI")

    assert sheetsage2_tool.main([str(source), "--no-skip-completed"]) == 1
    assert unrelated.read_bytes() == b"hand-crafted MIDI"
    assert len(model.calls) == 1


def test_case_alias_of_unowned_score_is_not_overwritten(monkeypatch, tmp_path):
    from module import sheetsage2_tool

    source = _audio(tmp_path / "song.wav")
    target = source.parent / "sheetsage2_output" / source.name
    target.mkdir(parents=True)
    original = target / "Score.ABC"
    original.write_text("My score", encoding="utf-8")
    if not (target / "score.abc").exists():
        pytest.skip("Filesystem distinguishes case")
    model = FakeModel()
    monkeypatch.setattr(sheetsage2_tool, "load_model", lambda device: (model, "fp32"))

    assert sheetsage2_tool.main([str(source)]) == 1
    assert original.read_text(encoding="utf-8") == "My score"


def test_partial_result_does_not_replace_unowned_native_outputs(monkeypatch, tmp_path):
    from module import sheetsage2_tool

    source = _audio(tmp_path / "song.wav")
    model = FakeModel()
    monkeypatch.setattr(sheetsage2_tool, "load_model", lambda device: (model, "fp32"))
    assert sheetsage2_tool.main([str(source)]) == 0
    partial = source.parent / "sheetsage2_output" / source.name / "partial"
    partial.mkdir()
    (partial / "events.json").write_text("my partial events", encoding="utf-8")
    model.missing_scores.add(source.name)

    assert sheetsage2_tool.main([str(source), "--melody-only"]) == 1
    assert (partial / "events.json").read_text(encoding="utf-8") == "my partial events"
    assert not (partial / "metadata.json").exists()


def test_same_source_can_transcribe_again_after_silent_retirement(monkeypatch, tmp_path):
    from module import sheetsage2_tool

    source = _audio(tmp_path / "song.wav")
    manifest = tmp_path / "stems.json"
    manifest.write_text(json.dumps({"stems": [str(source)]}), encoding="utf-8")
    args = ["--stem-manifest", str(manifest)]
    model = FakeModel()
    monkeypatch.setattr(sheetsage2_tool, "load_model", lambda device: (model, "fp32"))
    assert sheetsage2_tool.main(args) == 0
    assert sheetsage2_tool.archive_silent_stem_outputs(source)
    assert sheetsage2_tool.main(args) == 0
    assert len(model.calls) == 2


def test_failed_song_preserves_previous_generation_and_next_song_runs(monkeypatch, tmp_path):
    from module import sheetsage2_tool

    source = _audio(tmp_path / "album" / "a.wav")
    model = FakeModel()
    monkeypatch.setattr(sheetsage2_tool, "load_model", lambda device: (model, "fp32"))
    assert sheetsage2_tool.main([str(source)]) == 0
    target = source.parent / "sheetsage2_output" / source.name
    (target / "events.json").write_text('{"events": ["old"]}', encoding="utf-8")
    _audio(source.parent / "b.wav")
    model.failures.add("a.wav")

    assert sheetsage2_tool.main([str(source.parent), "--no-skip-completed"]) == 1
    assert json.loads((target / "events.json").read_text(encoding="utf-8")) == {"events": ["old"]}
    assert (target.parent / "b.wav" / "score.abc").is_file()


@pytest.mark.parametrize("previous_score", [False, True])
def test_missing_abc_publishes_partial_results_without_replacing_an_existing_score(
    monkeypatch, tmp_path, capsys, previous_score,
):
    from module import sheetsage2_tool

    source = _audio(tmp_path / "track.wav")
    model = FakeModel()
    monkeypatch.setattr(sheetsage2_tool, "load_model", lambda device: (model, "fp32"))
    target = tmp_path / "sheetsage2_output" / source.name
    if previous_score:
        assert sheetsage2_tool.main([str(source)]) == 0
        score = (target / "score.abc").read_text(encoding="utf-8")
        (target / "notes.txt").write_text("user notes", encoding="utf-8")
    model.missing_scores.add(source.name)

    assert sheetsage2_tool.main([str(source), "--melody-only"]) == 1
    assert "No key was decoded" in capsys.readouterr().err
    partial = target / "partial" if previous_score else target
    assert not (partial / "score.abc").exists()
    assert (partial / "transcription.mid").is_file()
    assert (partial / "events.json").is_file()
    if previous_score:
        assert (target / "score.abc").read_text(encoding="utf-8") == score
        assert (target / "notes.txt").read_text(encoding="utf-8") == "user notes"
        assert sheetsage2_tool.main([str(source)]) == 0
        assert len(model.calls) == 2


def test_busy_output_directory_reports_error_without_loading_model(monkeypatch, tmp_path, capsys):
    from module import sheetsage2_tool

    source = _audio(tmp_path / "track.wav")
    output = tmp_path / "out"
    output.mkdir()

    def unexpected_load(device):
        raise AssertionError("must not load while the output directory is locked")

    monkeypatch.setattr(sheetsage2_tool, "load_model", unexpected_load)
    with FileLock(str(output / ".sheetsage2.lock")):
        assert sheetsage2_tool.main([str(source), "--output-dir", str(output)]) == 1
    assert "busy" in capsys.readouterr().err


def test_stem_batch_loads_once_and_reuses_midi_without_abc(monkeypatch, tmp_path):
    from module import sheetsage2_tool

    vocal = _audio(tmp_path / "a.wav.stems" / "vocal.wav")
    drums = _audio(tmp_path / "b.wav.stems" / "drums.wav")
    manifest = tmp_path / "batch.json"
    manifest.write_text(json.dumps({"stems": [str(vocal), str(drums)]}), encoding="utf-8")
    model = FakeModel()
    model.missing_scores.add(drums.name)
    loads = []

    def load(device):
        loads.append(device)
        return model, "fp32"

    monkeypatch.setattr(sheetsage2_tool, "load_model", load)
    args = ["--stem-manifest", str(manifest), "--device=cpu", "--melody-only"]

    assert sheetsage2_tool.main(args) == 0
    assert sheetsage2_tool.main(args) == 0
    assert loads == ["cpu"]
    assert model.calls == [(vocal, "fp32", True), (drums, "fp32", True)]
    vocal_output = vocal.parent / "04_stem_midi" / "sheetsage2" / vocal.name
    drum_output = drums.parent / "04_stem_midi" / "sheetsage2" / drums.name
    assert (vocal_output / "score.abc").is_file()
    assert (drum_output / "transcription.mid").is_file()
    assert (drum_output / "events.json").is_file()
    assert not (drum_output / "score.abc").exists()


def test_stem_batch_keeps_new_midi_current_and_archives_the_previous_abc(monkeypatch, tmp_path):
    from module import sheetsage2_tool

    source = _audio(tmp_path / "song.wav.stems" / "vocals.wav")
    manifest = tmp_path / "batch.json"
    manifest.write_text(json.dumps({"stems": [str(source)]}), encoding="utf-8")
    model = FakeModel()
    monkeypatch.setattr(sheetsage2_tool, "load_model", lambda device: (model, "fp32"))
    args = ["--stem-manifest", str(manifest)]
    target = source.parent / "04_stem_midi" / "sheetsage2" / source.name
    assert sheetsage2_tool.main(args) == 0
    (target / "score.abc").write_text("Edited score", encoding="utf-8")
    model.missing_scores.add(source.name)

    assert sheetsage2_tool.main([*args, "--melody-only"]) == 0
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["signature"]["melody_only"] is True
    assert (target / "transcription.mid").is_file()
    assert not (target / "score.abc").is_file()
    assert [path.read_text(encoding="utf-8") for path in target.rglob("score.abc")] == ["Edited score"]


def test_stem_batch_continues_other_songs_when_one_output_cannot_be_created(monkeypatch, tmp_path):
    from module import sheetsage2_tool

    blocked = _audio(tmp_path / "blocked.wav.stems" / "vocals.wav")
    valid = _audio(tmp_path / "valid.wav.stems" / "vocals.wav")
    (blocked.parent / "04_stem_midi").write_text("Existing file", encoding="utf-8")
    manifest = tmp_path / "batch.json"
    manifest.write_text(json.dumps({"stems": [str(blocked), str(valid)]}), encoding="utf-8")
    model = FakeModel()
    monkeypatch.setattr(sheetsage2_tool, "load_model", lambda device: (model, "fp32"))

    assert sheetsage2_tool.main(["--stem-manifest", str(manifest)]) == 1
    assert model.calls == [(valid, "fp32", False)]
    assert (valid.parent / "04_stem_midi" / "sheetsage2" / valid.name / "transcription.mid").is_file()


def test_mix_batch_does_not_rediscover_generated_stems(monkeypatch, tmp_path):
    from module import sheetsage2_tool

    source = _audio(tmp_path / "a.wav")
    vocal = _audio(tmp_path / "a.wav.stems" / "vocal.wav")
    (vocal.parent / "separation_metadata.json").write_text("{}", encoding="utf-8")
    model = FakeModel()
    monkeypatch.setattr(sheetsage2_tool, "load_model", lambda device: (model, "fp32"))

    assert sheetsage2_tool.main([str(tmp_path)]) == 0
    assert model.calls == [(source, "fp32", False)]
    assert sheetsage2_tool.main([str(vocal.parent)]) == 0
    assert model.calls[-1] == (vocal, "fp32", False)


@pytest.mark.skipif(not shutil.which("pwsh"), reason="PowerShell is not installed")
def test_powershell_keeps_positional_audio_and_accepts_named_backend():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "pwsh", "-NoProfile", "-Command",
            r"""
$ErrorActionPreference = 'Stop'
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    (Resolve-Path './2.7.music_transcription.ps1').Path, [ref]$tokens, [ref]$errors
)
if ($errors) { throw ($errors | Out-String) }
$parameters = $ast.Extent.Text.Substring(0, $ast.ParamBlock.Extent.EndOffset)
$bind = [scriptblock]::Create($parameters + [Environment]::NewLine + '[pscustomobject]@{Backend=$Backend; Arguments=$Arguments}')
@(
    & $bind './my album' '--format' 'midi'
    & $bind -Backend sheetsage2 './my song.wav' '--melody-only'
    & $bind -Backend dettato './my mix.wav' '--device=cpu'
) | ConvertTo-Json -Compress
""",
        ],
        cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        {"Backend": "muscriptor", "Arguments": ["./my album", "--format", "midi"]},
        {"Backend": "sheetsage2", "Arguments": ["./my song.wav", "--melody-only"]},
        {"Backend": "dettato", "Arguments": ["./my mix.wav", "--device=cpu"]},
    ]
