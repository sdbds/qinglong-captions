from __future__ import annotations

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import mido
import pytest
from filelock import FileLock
from packaging.requirements import Requirement


def test_optional_profile_pins_the_pipeline_used_for_resume_without_requiring_the_shared_torch_build():
    from module.dettato_tool import PIPELINE_REVISION

    with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    deps = [Requirement(spec) for spec in project["optional-dependencies"]["dettato-local"]]
    [sdk] = [dep for dep in deps if dep.name == "dettato"]
    assert sdk.url == f"git+https://github.com/kobimusic/dettato.git@{PIPELINE_REVISION}"
    assert all("torch-base" not in dep.extras and dep.name != "torchvision" for dep in deps)
    assert all(Requirement(spec).name != "dettato" for spec in project["dependencies"])


def _audio(path, data=b"input"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class RecordingModel:
    device = "cpu"

    def __init__(self):
        self.sources = []

    def transcribe(self, path):
        path = Path(path)
        self.sources.append(path)

        def save_midi(target):
            if path.read_bytes() == b"bad":
                Path(target).write_bytes(b"not a MIDI")
                return
            midi = mido.MidiFile()
            for channel, pitch in ((0, 64), (9, 38)):
                midi.add_track().append(mido.Message("note_on", channel=channel, note=pitch, velocity=80))
            midi.save(target)

        return SimpleNamespace(
            save_midi=save_midi,
            summary=lambda: {
                "source": str(path),
                "seconds": 1.0,
                "tuning_cents": 0,
                "drums": {"hits": 1},
                "tracks": [
                    {"name": "Piano", "program": 0, "is_drum": False, "notes": 1},
                    {"name": "Drums", "program": None, "is_drum": True, "notes": 1},
                ],
            },
        )


@pytest.fixture
def worker(monkeypatch):
    from module import dettato_tool

    model = RecordingModel()
    devices = []

    def load(device):
        devices.append(device)
        return model

    monkeypatch.setattr(dettato_tool, "load_model", load)
    return dettato_tool.main, model.sources, devices


def test_batch_preserves_source_paths_native_tracks_and_reuses_model(tmp_path, worker):
    run, sources, devices = worker
    album = tmp_path / "album"
    flac = _audio(album / "disc 1" / "song.flac")
    wav = _audio(album / "disc 1" / "song.wav")
    stem = _audio(album / "song.stems" / "vocals.wav")
    (stem.parent / "separation_metadata.json").write_text("{}")
    output = album / "dettato_output"
    _audio(output / "previous.wav")

    assert run([str(album), "--device=cuda:1"]) == 0
    assert sources == [flac, wav]
    assert devices == ["cuda:1"]
    for name in ("song.flac", "song.wav"):
        target = output / "disc 1" / name
        midi = mido.MidiFile(target / "song.mid")
        assert [msg.channel for track in midi.tracks for msg in track if msg.type == "note_on"] == [0, 9]
        metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["tracks"][-1] == {"name": "Drums", "program": None, "is_drum": True, "notes": 1}
        assert metadata["drums"] == {"hits": 1}


def test_resume_skips_without_loading_and_reruns_on_input_or_output_change(tmp_path, worker):
    run, sources, devices = worker
    song = _audio(tmp_path / "song.wav")
    args = [str(song)]
    assert run(args) == 0
    assert run(args) == 0
    assert sources == [song] and devices == ["auto"]

    midi = tmp_path / "dettato_output" / song.name / "song.mid"
    midi.unlink()
    assert run(args) == 0
    song.write_bytes(b"new audio")
    assert run(args) == 0
    assert run([*args, "--no-skip-completed"]) == 0
    assert sources == [song] * 4


def test_failed_regeneration_keeps_last_pair_and_processes_next_song(tmp_path, worker):
    run, _, _ = worker
    song = _audio(tmp_path / "a.wav")
    assert run([str(song)]) == 0
    target = tmp_path / "dettato_output" / song.name
    previous = {p.name: p.read_bytes() for p in target.iterdir()}

    song.write_bytes(b"bad")
    _audio(tmp_path / "b.wav")
    assert run([str(tmp_path)]) == 1
    assert {p.name: p.read_bytes() for p in target.iterdir()} == previous
    assert (target.parent / "b.wav" / "b.mid").is_file()


@pytest.mark.parametrize(
    "track",
    [
        pytest.param("00ff5100 00ff2f00", id="empty-tempo"),
        pytest.param("00ff59020002 00ff2f00", id="invalid-key-signature"),
    ],
)
def test_malformed_midi_does_not_abort_resume_or_later_songs(tmp_path, worker, track):
    run, sources, _ = worker
    first = _audio(tmp_path / "a.wav")
    assert run([str(first)]) == 0
    midi = tmp_path / "dettato_output" / "a.wav" / "a.mid"
    events = bytes.fromhex(track)
    midi.write_bytes(bytes.fromhex("4d54686400000006000000010060 4d54726b") + len(events).to_bytes(4, "big") + events)
    second = _audio(tmp_path / "b.wav")
    assert run([str(tmp_path)]) == 0
    assert sources == [first, first, second]
    assert len(mido.MidiFile(midi).tracks) == 2


def test_explicit_output_root_cannot_overwrite_another_song_or_unowned_midi(tmp_path, worker):
    run, sources, _ = worker
    a = _audio(tmp_path / "a" / "mix.wav")
    b = _audio(tmp_path / "b" / "mix.wav")
    root = tmp_path / "output"
    args = ["--output-dir", str(root), "--no-skip-completed"]
    assert run([str(a), *args]) == 0
    target = root / "mix.wav"
    before = (target / "mix.mid").read_bytes()
    assert run([str(b), *args]) == 1
    (target / "metadata.json").unlink()
    assert run([str(a), *args]) == 1
    assert (target / "mix.mid").read_bytes() == before
    assert sources == [a]


def test_overlapping_output_roots_cannot_replace_another_source_during_inference(tmp_path, monkeypatch):
    from module import dettato_tool

    album = tmp_path / "album"
    first = _audio(album / "disc" / "mix.wav")
    second = _audio(tmp_path / "other" / "mix.wav")
    output = tmp_path / "out"

    class InterleavedModel(RecordingModel):
        def transcribe(self, path):
            if Path(path) == first:
                assert dettato_tool.main([str(second), "--output-dir", str(output / "disc")]) == 0
            return super().transcribe(path)

    monkeypatch.setattr(dettato_tool, "load_model", lambda device: InterleavedModel())
    assert dettato_tool.main([str(album), "--output-dir", str(output)]) == 1
    metadata = json.loads((output / "disc" / "mix.wav" / "metadata.json").read_text(encoding="utf-8"))
    assert Path(metadata["source"]) == second


def test_empty_directory_or_busy_output_never_loads_model(tmp_path, worker):
    run, _, devices = worker
    assert run([str(tmp_path)]) == 1
    song = _audio(tmp_path / "song.wav")
    output = tmp_path / "out"
    output.mkdir()
    with FileLock(output / ".dettato.lock"):
        assert run([str(song), "--output-dir", str(output)]) == 1
    assert not devices


def test_one_model_load_failure_reports_an_actionable_error_without_writing_results(tmp_path, monkeypatch, capsys):
    from module import dettato_tool

    _audio(tmp_path / "a.wav")
    _audio(tmp_path / "b.wav")
    loads = []

    def unavailable(device):
        loads.append(device)
        raise LookupError("missing drum weights")

    monkeypatch.setattr(dettato_tool, "load_model", unavailable)
    assert dettato_tool.main([str(tmp_path)]) == 1
    assert loads == ["auto"]
    assert "missing drum weights" in capsys.readouterr().err
    assert not list(tmp_path.rglob("metadata.json"))


def test_output_cannot_be_the_input_file_even_on_a_forced_run(tmp_path, worker):
    run, sources, devices = worker
    song = _audio(tmp_path / "song.wav")
    assert run([str(song), "--output-dir", str(tmp_path), "--no-skip-completed"]) == 1
    assert not sources and not devices
    assert song.read_bytes() == b"input"
