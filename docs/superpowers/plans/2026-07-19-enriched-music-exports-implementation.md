# Enriched Music Exports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Replace fixed 120 BPM exports with shared audio-derived timing, add relative MuScriptor velocity and MusicXML, reuse timing in GAME vocal MIDI, and provide an on-demand project-private FluidSynth runtime.

**Architecture:** A dependency-light module/music_analysis package owns timing facts and seconds/beats/ticks conversion. MuScriptor builds one canonical AnalyzedScore, then pure MIDI/JSON/JSONL/MusicXML exporters consume it. Preview and WebUI use project-owned render/server adapters so private FluidSynth and enriched MIDI cannot be bypassed.

**Tech Stack:** Python 3.10-3.12, pytest, torch, beat-this 1.1.0 small0, mido, music21 9.9.1/10.5.0, FastAPI, soundfile, filelock.

## Global Constraints

- Preserve Python requires-python >=3.10,<3.13.
- Pin beat-this==1.1.0 and use checkpoint small0 on CPU by default.
- Pin music21==9.9.1 on Python 3.10 and music21==10.5.0 on Python 3.11/3.12.
- No test may download Beat This weights or FluidSynth unless an existing explicit real-smoke environment gate is enabled.
- Manual BPM and meter together must bypass Beat This.
- Automatic timing failure must emit source=fallback, 120 BPM, 4/4, and an actionable warning.
- MuScriptor velocity is relative only, normally 24-120, fallback 80; GAME vocal velocity remains 64.
- Dynamics uses 30-second core chunks with 250 ms overlap and never retains a whole-song spectrogram.
- Existing system FluidSynth wins; managed Windows x64 cache is second; no permanent or process-wide PATH mutation.
- QINGLONG_CAPTIONS_DISABLE_FLUIDSYNTH_DOWNLOAD=1 disables managed FluidSynth download.
- Symbolic exports never probe or download FluidSynth unless preview is requested.
- All owned file outputs use temporary files plus atomic replacement.
- Preserve MuScriptor 0.2.1 canonical validation and overlap-trimming behavior before enrichment.
- Do not edit upstream compiled web_dist assets.
- Design source of truth: docs/superpowers/specs/2026-07-19-music-analysis-enriched-midi-musicxml-design.md.

---

## File Structure

New common timing files:

- module/music_analysis/__init__.py: stable public exports.
- module/music_analysis/types.py: validated immutable timing types and manual override parsing.
- module/music_analysis/postprocess.py: beat interval cleanup, tempo map, meter and quality calculation.
- module/music_analysis/midi_timing.py: piecewise seconds/beats/ticks conversion and conductor events.
- module/music_analysis/beat_this_runtime.py: lazy small0 adapter with injected test seam.
- module/music_analysis/cache.py: timing signatures and per-source analysis cache.

New MuScriptor files:

- module/muscriptor_tool/score.py: canonical event pairing/cleanup and AnalyzedScore.
- module/muscriptor_tool/dynamics.py: fixed-memory chunked relative intensity.
- module/muscriptor_tool/midi_export.py: Type 1 conductor/instrument MIDI.
- module/muscriptor_tool/structured_export.py: JSON v2 and JSONL records.
- module/muscriptor_tool/musicxml_export.py: music21 score construction and readback validation.
- module/muscriptor_tool/fluidsynth_runtime.py: resolver and managed Windows installer.
- module/muscriptor_tool/web_server.py: local /transcribe and /auralize routes over the upstream app.

Modified integration files:

- module/muscriptor_tool/options.py: timing overrides and MusicXML format.
- module/muscriptor_tool/outputs.py: one canonical score feeding all exporters.
- module/muscriptor_tool/auralization.py: local renderer with absolute executable.
- module/muscriptor_tool/cli.py: timing flags and MusicXML targets.
- module/muscriptor_tool/batch.py: timing signature/cache and partial output accounting.
- module/muscriptor_tool/manifest.py: schema bump, MusicXML and sidecar ownership.
- module/muscriptor_tool/stems.py: per-song shared timing and one dynamics pass.
- module/muscriptor_tool/webui.py: project web app factory.
- module/vocal_midi.py: common timing, conductor track, velocity 64 and sidecar.
- module/audio_separator.py: shared timing controls and per-song reuse.
- gui/wizard/step6_tools.py: MusicXML plus Auto/Manual timing controls.
- pyproject.toml: conditional dependencies and lock metadata.

New focused tests:

- tests/test_music_analysis_types.py
- tests/test_music_analysis_postprocess.py
- tests/test_music_analysis_midi_timing.py
- tests/test_music_analysis_runtime.py
- tests/test_muscriptor_score.py
- tests/test_muscriptor_dynamics.py
- tests/test_muscriptor_musicxml.py
- tests/test_muscriptor_fluidsynth.py

Existing integration tests are extended in place.

---

### Task 1: Timing Types And Manual Overrides

**Files:**
- Create: module/music_analysis/__init__.py
- Create: module/music_analysis/types.py
- Create: tests/test_music_analysis_types.py

**Interfaces:**
- Produces: TempoPoint, MusicTiming, TimingOverrides, parse_time_signature(), fallback_timing().
- Consumes: no project runtime dependencies.

- [ ] **Step 1: Write failing validation tests**

~~~python
from module.music_analysis.types import (
    MusicTiming,
    TempoPoint,
    TimingOverrides,
    fallback_timing,
    parse_time_signature,
)


def test_manual_overrides_require_finite_bpm_in_supported_range():
    assert TimingOverrides(bpm=98.5, time_signature=(3, 4)).complete
    with pytest.raises(ValueError, match="30 and 300"):
        TimingOverrides(bpm=float("nan"))


def test_parse_time_signature_rejects_non_binary_denominator():
    assert parse_time_signature("6/8") == (6, 8)
    with pytest.raises(ValueError, match="denominator"):
        parse_time_signature("4/3")


def test_fallback_is_explicit_and_never_empty():
    timing = fallback_timing(12.0, warnings=("BEAT_ANALYSIS_FAILED: decode",))
    assert timing.global_bpm == 120.0
    assert timing.time_signature == (4, 4)
    assert timing.tempo_source == "fallback"
    assert timing.meter_source == "fallback"
    assert timing.tempo_map == (TempoPoint(0.0, 0.0, 120.0),)
~~~

- [ ] **Step 2: Run tests and verify RED**

Run: python -m pytest tests/test_music_analysis_types.py -q

Expected: collection fails because module.music_analysis does not exist.

- [ ] **Step 3: Implement immutable validated types**

~~~python
@dataclass(frozen=True)
class TempoPoint:
    time_seconds: float
    beat_position: float
    bpm: float


@dataclass(frozen=True)
class TimingOverrides:
    bpm: float | None = None
    time_signature: tuple[int, int] | None = None

    @property
    def complete(self) -> bool:
        return self.bpm is not None and self.time_signature is not None


@dataclass(frozen=True)
class MusicTiming:
    duration_seconds: float
    beats_seconds: tuple[float, ...]
    downbeats_seconds: tuple[float, ...]
    global_bpm: float
    tempo_map: tuple[TempoPoint, ...]
    numerator: int
    denominator: int
    tempo_source: str
    meter_source: str
    quality_score: float
    usable: bool
    warnings: tuple[str, ...]
    algorithm: str
    algorithm_version: str
    checkpoint: str | None

    @property
    def time_signature(self) -> tuple[int, int]:
        return self.numerator, self.denominator
~~~

Implement __post_init__ checks exactly from the design and export the public names from __init__.py.

- [ ] **Step 4: Run tests and verify GREEN**

Run: python -m pytest tests/test_music_analysis_types.py -q

Expected: all tests pass.

- [ ] **Step 5: Commit**

~~~powershell
git add module/music_analysis/__init__.py module/music_analysis/types.py tests/test_music_analysis_types.py
git commit -m "feat: add shared music timing types"
~~~

### Task 2: Beat And Meter Postprocessing

**Files:**
- Create: module/music_analysis/postprocess.py
- Create: tests/test_music_analysis_postprocess.py

**Interfaces:**
- Consumes: MusicTiming, TempoPoint, TimingOverrides.
- Produces: build_music_timing(duration_seconds, beats_seconds, downbeats_seconds, overrides) -> MusicTiming.

- [ ] **Step 1: Write failing stable, variable and fallback tests**

~~~python
def test_stable_clicks_produce_single_tempo():
    timing = build_music_timing(
        duration_seconds=3.0,
        beats_seconds=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
        downbeats_seconds=(0.0, 2.0),
        overrides=TimingOverrides(time_signature=(4, 4)),
    )
    assert timing.global_bpm == pytest.approx(120.0)
    assert len(timing.tempo_map) == 1
    assert timing.tempo_source == "detected"


def test_variable_clicks_keep_a_tempo_map_without_half_time_folding():
    beats = (0.0, 0.60, 1.16, 1.68, 2.16, 2.60, 3.00, 3.36)
    timing = build_music_timing(3.5, beats, (), TimingOverrides(time_signature=(4, 4)))
    assert len(timing.tempo_map) > 1
    assert timing.tempo_map[-1].bpm > timing.tempo_map[0].bpm


def test_ambiguous_compound_meter_falls_back_but_keeps_detected_tempo():
    timing = build_music_timing(
        8.0,
        tuple(index * 0.5 for index in range(16)),
        (0.0, 3.0, 6.0),
        TimingOverrides(),
    )
    assert timing.tempo_source == "detected"
    assert timing.meter_source == "fallback"
    assert timing.time_signature == (4, 4)
    assert any("METER" in warning for warning in timing.warnings)
~~~

Also cover local Hampel rejection, fewer than four beats, less than 60% valid intervals, 2/4, 3/4, 4/4, partial manual override and the exact quality formula.

- [ ] **Step 2: Run tests and verify RED**

Run: python -m pytest tests/test_music_analysis_postprocess.py -q

Expected: import failure for module.music_analysis.postprocess.

- [ ] **Step 3: Implement pure deterministic postprocessing**

~~~python
def build_music_timing(
    duration_seconds: float,
    beats_seconds: Sequence[float],
    downbeats_seconds: Sequence[float],
    overrides: TimingOverrides = TimingOverrides(),
) -> MusicTiming:
    cleaned = clean_beat_intervals(beats_seconds)
    detected_tempo = _tempo_from_intervals(duration_seconds, cleaned)
    meter = _meter_from_downbeats(beats_seconds, downbeats_seconds)
    return _merge_detection_and_overrides(
        duration_seconds=duration_seconds,
        beats_seconds=beats_seconds,
        downbeats_seconds=downbeats_seconds,
        detected_tempo=detected_tempo,
        detected_meter=meter,
        overrides=overrides,
    )
~~~

Keep thresholds in one frozen PostprocessConfig with algorithm_version="music-timing-v1".

- [ ] **Step 4: Run tests and verify GREEN**

Run: python -m pytest tests/test_music_analysis_postprocess.py -q

Expected: all tests pass.

- [ ] **Step 5: Commit**

~~~powershell
git add module/music_analysis/postprocess.py tests/test_music_analysis_postprocess.py
git commit -m "feat: derive tempo maps and meter from beats"
~~~

### Task 3: Seconds, Beats And MIDI Ticks

**Files:**
- Create: module/music_analysis/midi_timing.py
- Create: tests/test_music_analysis_midi_timing.py

**Interfaces:**
- Consumes: MusicTiming.
- Produces: TimingMap, build_conductor_track().

- [ ] **Step 1: Write failing round-trip and conductor tests**

~~~python
def test_variable_tempo_seconds_ticks_round_trip():
    timing = variable_timing_fixture()
    mapping = TimingMap(timing, ticks_per_beat=480)
    for seconds in (0.0, 0.2, 1.0, 2.7, 4.0):
        ticks = mapping.seconds_to_ticks(seconds)
        assert mapping.ticks_to_seconds(ticks) == pytest.approx(seconds, abs=0.002)


def test_conductor_contains_tempo_and_meter_at_tick_zero():
    track = build_conductor_track(stable_timing_fixture(), ticks_per_beat=480)
    assert track[0].type == "set_tempo"
    assert any(message.type == "time_signature" for message in track)
~~~

- [ ] **Step 2: Run tests and verify RED**

Run: python -m pytest tests/test_music_analysis_midi_timing.py -q

Expected: import failure for midi_timing.

- [ ] **Step 3: Implement piecewise integration**

~~~python
class TimingMap:
    def __init__(self, timing: MusicTiming, *, ticks_per_beat: int = 480):
        self.timing = timing
        self.ticks_per_beat = ticks_per_beat
        self._times = tuple(point.time_seconds for point in timing.tempo_map)
        self._beats = tuple(point.beat_position for point in timing.tempo_map)
        self._bpms = tuple(point.bpm for point in timing.tempo_map)

    def seconds_to_beats(self, seconds: float) -> float:
        index = max(0, bisect_right(self._times, max(0.0, seconds)) - 1)
        return self._beats[index] + (max(0.0, seconds) - self._times[index]) * self._bpms[index] / 60.0

    def beats_to_seconds(self, beats: float) -> float:
        index = max(0, bisect_right(self._beats, max(0.0, beats)) - 1)
        return self._times[index] + (max(0.0, beats) - self._beats[index]) * 60.0 / self._bpms[index]

    def seconds_to_ticks(self, seconds: float) -> int:
        return round(self.seconds_to_beats(seconds) * self.ticks_per_beat)

    def ticks_to_seconds(self, ticks: int) -> float:
        return self.beats_to_seconds(max(0, ticks) / self.ticks_per_beat)


def build_conductor_track(timing: MusicTiming, *, ticks_per_beat: int):
    import mido
    track = mido.MidiTrack()
    mapping = TimingMap(timing, ticks_per_beat=ticks_per_beat)
    absolute = [
        (
            mapping.seconds_to_ticks(point.time_seconds),
            0,
            mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(point.bpm), time=0),
        )
        for point in timing.tempo_map
    ]
    absolute.append(
        (
            0,
            1,
            mido.MetaMessage(
                "time_signature",
                numerator=timing.numerator,
                denominator=timing.denominator,
                time=0,
            ),
        )
    )
    previous = 0
    for tick, _, message in sorted(absolute, key=lambda item: (item[0], item[1])):
        message.time = tick - previous
        track.append(message)
        previous = tick
    return track
~~~

Use bisect over precomputed segment starts; clamp negative seconds to zero and preserve monotonic integer ticks.

- [ ] **Step 4: Run tests and verify GREEN**

Run: python -m pytest tests/test_music_analysis_midi_timing.py -q

Expected: all tests pass.

- [ ] **Step 5: Commit**

~~~powershell
git add module/music_analysis/midi_timing.py tests/test_music_analysis_midi_timing.py
git commit -m "feat: map audio time through MIDI tempo maps"
~~~

### Task 4: Beat This Runtime And Cache

**Files:**
- Create: module/music_analysis/beat_this_runtime.py
- Create: module/music_analysis/cache.py
- Create: tests/test_music_analysis_runtime.py
- Modify: pyproject.toml
- Modify: tests/test_muscriptor_dependencies.py
- Modify: tests/test_audio_separator_dependency_profiles.py

**Interfaces:**
- Produces: BeatThisAnalyzer.analyze(path, overrides) -> MusicTiming.
- Produces: TimingCache.get_or_analyze(source, overrides, analyzer) -> MusicTiming.
- Consumes: build_music_timing and fallback_timing.

- [ ] **Step 1: Write failing lazy-load and fallback tests**

~~~python
def test_complete_manual_override_never_loads_model(tmp_path):
    calls = []
    analyzer = BeatThisAnalyzer(model_loader=lambda: calls.append("load"))
    timing = analyzer.analyze(
        tmp_path / "song.wav",
        TimingOverrides(bpm=90.0, time_signature=(3, 4)),
        duration_seconds=4.0,
    )
    assert calls == []
    assert timing.tempo_source == timing.meter_source == "manual"


def test_runtime_failure_returns_explicit_fallback(tmp_path):
    analyzer = BeatThisAnalyzer(model_loader=lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    timing = analyzer.analyze(tmp_path / "song.wav", duration_seconds=4.0)
    assert timing.global_bpm == 120.0
    assert "offline" in timing.warnings[0]
~~~

Add tests proving one model load across files, injected Audio2Frames output, official local-peak postprocessor behavior, and cache invalidation on source stat/checkpoint/override/schema changes.

- [ ] **Step 2: Add dependency contract tests and verify RED**

Assert pyproject markers are exactly:

~~~toml
"beat-this==1.1.0"
"music21==9.9.1; python_version == '3.10'"
"music21==10.5.0; python_version >= '3.11' and python_version < '3.13'"
~~~

Run: python -m pytest tests/test_music_analysis_runtime.py tests/test_muscriptor_dependencies.py tests/test_audio_separator_dependency_profiles.py -q

Expected: new runtime imports and dependency assertions fail.

- [ ] **Step 3: Implement lazy adapter, fallback and cache**

~~~python
class BeatThisAnalyzer:
    def __init__(self, *, checkpoint="small0", device="cpu", model_loader=None, audio_loader=None):
        self.checkpoint = checkpoint
        self.device = device
        self._model_loader = model_loader or _load_default_model
        self._audio_loader = audio_loader or _load_default_audio
        self._model = None

    def analyze(
        self,
        source: Path,
        overrides: TimingOverrides = TimingOverrides(),
        *,
        duration_seconds: float | None = None,
    ) -> MusicTiming:
        if overrides.complete:
            return manual_timing(duration_seconds or probe_duration(source), overrides)
        try:
            audio, sample_rate = self._audio_loader(source)
            duration = duration_seconds or len(audio) / sample_rate
            if self._model is None:
                self._model = self._model_loader()
            beat_logits, downbeat_logits = infer_frame_logits(
                self._model,
                audio,
                sample_rate,
                device=self.device,
            )
            beats, downbeats = local_peak_times(beat_logits, downbeat_logits)
            return build_music_timing(duration, beats, downbeats, overrides)
        except Exception as exc:
            return fallback_with_overrides(
                duration_seconds or safe_probe_duration(source),
                overrides,
                warning=f"BEAT_ANALYSIS_FAILED: {type(exc).__name__}: {exc}",
            )


class TimingCache:
    def __init__(self):
        self._items: dict[str, MusicTiming] = {}

    def get_or_analyze(
        self,
        source: Path,
        overrides: TimingOverrides,
        analyzer: BeatThisAnalyzer,
    ) -> MusicTiming:
        key = timing_cache_key(
            source,
            overrides,
            checkpoint=analyzer.checkpoint,
            algorithm_version=POSTPROCESS_CONFIG.algorithm_version,
        )
        if key not in self._items:
            self._items[key] = analyzer.analyze(source, overrides)
        return self._items[key]
~~~

Import beat_this inside the default loader only. Use Audio2Frames and official local peaks, never label raw logits as confidence.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: python -m pytest tests/test_music_analysis_runtime.py tests/test_muscriptor_dependencies.py tests/test_audio_separator_dependency_profiles.py -q

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

~~~powershell
git add pyproject.toml module/music_analysis/beat_this_runtime.py module/music_analysis/cache.py tests/test_music_analysis_runtime.py tests/test_muscriptor_dependencies.py tests/test_audio_separator_dependency_profiles.py
git commit -m "feat: add lazy Beat This timing analysis"
~~~

### Task 5: GAME Vocal MIDI Uses Shared Timing

**Files:**
- Modify: module/vocal_midi.py
- Modify: tests/test_audio_separator_onnx.py
- Create: tests/test_vocal_midi_timing.py

**Interfaces:**
- Consumes: MusicTiming, TimingOverrides, TimingMap, build_conductor_track.
- Produces: atomic .mid plus .timing.json sidecar.

- [ ] **Step 1: Write failing MIDI timing tests**

~~~python
def test_vocal_midi_uses_supplied_tempo_map_and_explicit_velocity(tmp_path):
    target = tmp_path / "voice.mid"
    transcriber = object.__new__(GameOnnxTranscriber)
    transcriber._save_midi_file(
        target,
        [NoteInfo(0.5, 1.0, 60.0)],
        timing=stable_timing_fixture(90.0),
    )
    midi = mido.MidiFile(target)
    assert any(message.type == "set_tempo" and message.tempo == mido.bpm2tempo(90) for message in midi.tracks[0])
    note_on = next(message for track in midi.tracks for message in track if message.type == "note_on")
    assert note_on.velocity == 64


def test_vocal_sidecar_is_required_for_skip_completion(tmp_path):
    mid_path = tmp_path / "voice.mid"
    sidecar_path = tmp_path / "voice.timing.json"
    mid_path.write_bytes(b"MThd")
    assert vocal_midi_outputs_complete(mid_path, sidecar_path, "signature-v2") is False
    atomic_write_json(sidecar_path, {"schema_version": 2, "signature": "signature-v2"})
    assert vocal_midi_outputs_complete(mid_path, sidecar_path, "signature-v2") is True
    assert vocal_midi_outputs_complete(mid_path, sidecar_path, expected_signature) is False
~~~

Also assert source seconds survive round-trip through ticks and no onset * 120 * 8 formula remains.

- [ ] **Step 2: Run tests and verify RED**

Run: python -m pytest tests/test_vocal_midi_timing.py tests/test_audio_separator_onnx.py -q

Expected: _save_midi_file rejects timing and sidecar helper is missing.

- [ ] **Step 3: Implement Type 1 vocal MIDI and sidecar**

~~~python
def _save_midi_file(
    self,
    output_path: Path,
    notes: Sequence[NoteInfo],
    *,
    timing: MusicTiming,
) -> Path:
    midi_file = mido.MidiFile(type=1, ticks_per_beat=480, charset="utf8")
    midi_file.tracks.append(build_conductor_track(timing, ticks_per_beat=480))
    midi_file.tracks.append(_vocal_note_track(notes, TimingMap(timing, ticks_per_beat=480), velocity=64))
    with atomic_output_path(output_path) as temporary:
        midi_file.save(temporary)
    return output_path
~~~

Add bpm/time_signature/timing parameters to transcribe_file and CLI parser. Write the sidecar after all requested outputs using the same atomic helper.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: python -m pytest tests/test_vocal_midi_timing.py tests/test_audio_separator_onnx.py -q

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

~~~powershell
git add module/vocal_midi.py tests/test_vocal_midi_timing.py tests/test_audio_separator_onnx.py
git commit -m "feat: apply detected timing to vocal MIDI"
~~~

### Task 6: Canonical MuScriptor Score

**Files:**
- Create: module/muscriptor_tool/score.py
- Create: tests/test_muscriptor_score.py
- Modify: module/muscriptor_tool/runtime.py

**Interfaces:**
- Produces: CanonicalNote, EnrichedNote, AnalyzedScore, build_canonical_notes(), enrich_note_positions().
- Consumes: raw MuScriptor events and MusicTiming.

- [ ] **Step 1: Write failing cleanup parity tests**

~~~python
def test_canonical_builder_fixes_duration_trims_overlap_and_sorts():
    events = [
        start(0, pitch=60, instrument="piano", time=0.0),
        end(0, 1.0),
        start(1, pitch=60, instrument="piano", time=0.5),
        end(1, 0.5),
    ]
    notes, warnings = build_canonical_notes(events, program_resolver=lambda _: 0)
    assert [(note.start_seconds, note.end_seconds) for note in notes] == [(0.0, 0.5), (0.5, 0.51)]


def test_unclosed_start_is_dropped_with_warning():
    notes, warnings = build_canonical_notes([start(3)], program_resolver=lambda _: 0)
    assert notes == ()
    assert warnings == ("UNCLOSED_NOTE: index=3",)
~~~

Add a parity fixture against MuScriptor 0.2.1 validate_notes plus trim_overlapping_notes when the optional runtime is installed.

- [ ] **Step 2: Run tests and verify RED**

Run: python -m pytest tests/test_muscriptor_score.py -q

Expected: module.muscriptor_tool.score is missing.

- [ ] **Step 3: Implement canonical score and public program resolver**

~~~python
@dataclass(frozen=True)
class CanonicalNote:
    index: int
    pitch: int
    instrument: str
    gm_program: int | None
    is_drum: bool
    start_seconds: float
    end_seconds: float


def build_canonical_notes(
    events: Iterable[Any],
    *,
    program_resolver: Callable[[str], int],
) -> tuple[tuple[CanonicalNote, ...], tuple[str, ...]]:
    opened: dict[int, CanonicalNote] = {}
    closed: list[CanonicalNote] = []
    for event in events:
        if hasattr(event, "start_time"):
            is_drum = str(event.instrument) == "drums"
            opened[int(event.index)] = CanonicalNote(
                index=int(event.index),
                pitch=int(event.pitch),
                instrument=str(event.instrument),
                gm_program=128 if is_drum else int(program_resolver(str(event.instrument))),
                is_drum=is_drum,
                start_seconds=float(event.start_time),
                end_seconds=float(event.start_time),
            )
            continue
        index = int(event.start_event_index)
        note = opened.pop(index)
        minimum_end = note.start_seconds + 0.01
        requested_end = float(event.end_time)
        fixed_end = max(requested_end, minimum_end) if (not note.is_drum or requested_end < note.start_seconds) else requested_end
        closed.append(replace(note, end_seconds=fixed_end))
    warnings = tuple(f"UNCLOSED_NOTE: index={index}" for index in sorted(opened))
    grouped: dict[tuple[int | None, int, bool], list[CanonicalNote]] = defaultdict(list)
    for note in closed:
        grouped[(note.gm_program, note.pitch, note.is_drum)].append(note)
    trimmed: list[CanonicalNote] = []
    for notes in grouped.values():
        notes.sort(key=lambda item: item.start_seconds)
        for position in range(len(notes) - 1):
            current = notes[position]
            following = notes[position + 1]
            if current.end_seconds > following.start_seconds:
                notes[position] = replace(current, end_seconds=following.start_seconds)
        trimmed.extend(note for note in notes if note.start_seconds < note.end_seconds)
    trimmed.sort(key=lambda note: (note.start_seconds, note.is_drum, note.gm_program, note.pitch, note.end_seconds))
    return tuple(trimmed), warnings
~~~

Expose LoadedModel.program_for_instrument(name) instead of calling the upstream private method from exporters.

- [ ] **Step 4: Run tests and verify GREEN**

Run: python -m pytest tests/test_muscriptor_score.py tests/test_muscriptor_runtime.py -q

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

~~~powershell
git add module/muscriptor_tool/score.py module/muscriptor_tool/runtime.py tests/test_muscriptor_score.py tests/test_muscriptor_runtime.py
git commit -m "feat: build canonical MuScriptor scores"
~~~

### Task 7: Fixed-Memory Relative Dynamics

**Files:**
- Create: module/muscriptor_tool/dynamics.py
- Create: tests/test_muscriptor_dynamics.py

**Interfaces:**
- Consumes: one or more candidate note collections plus full-mix audio reader.
- Produces: velocity and velocity_source per canonical note.

- [ ] **Step 1: Write failing intensity and chunk-boundary tests**

~~~python
def test_louder_same_pitch_maps_to_higher_velocity():
    waveform = harmonic_onsets(amplitudes=(0.1, 0.8), pitch=69)
    notes = notes_at((1.0, 2.0), pitch=69, instrument="piano")
    enriched, warnings = estimate_relative_velocities_from_waveform(waveform, 44100, notes)
    assert enriched[0].velocity < enriched[1].velocity
    assert all(24 <= note.velocity <= 120 for note in enriched)


def test_chunk_boundary_matches_single_chunk_reference():
    onset = 30.0
    waveform = harmonic_onsets(amplitudes=(0.2, 0.8, 0.4, 0.6), pitch=69, duration=61.0)
    notes = notes_at((29.0, onset, 31.0, 40.0), pitch=69, instrument="piano")
    chunked, _ = estimate_relative_velocities_from_waveform(
        waveform,
        44100,
        notes,
        config=DynamicsConfig(core_seconds=30.0),
    )
    reference, _ = estimate_relative_velocities_from_waveform(
        waveform,
        44100,
        notes,
        config=DynamicsConfig(core_seconds=120.0),
    )
    assert chunked[1].velocity == reference[1].velocity


def test_low_information_group_falls_back_to_80():
    enriched, warnings = estimate_relative_velocities_from_waveform(
        np.zeros(44100, dtype=np.float32), 44100, notes_at((0.2, 0.4, 0.6), instrument="piano")
    )
    assert {note.velocity for note in enriched} == {80}
~~~

Add drum spectral-flux, per-instrument normalization, unrelated-band interference, failed STFT and one-hour fake-reader live-chunk tests.

- [ ] **Step 2: Run tests and verify RED**

Run: python -m pytest tests/test_muscriptor_dynamics.py -q

Expected: dynamics module is missing.

- [ ] **Step 3: Implement chunk scanner and robust mapping**

~~~python
@dataclass(frozen=True)
class DynamicsConfig:
    core_seconds: float = 30.0
    overlap_seconds: float = 0.250
    window_seconds: float = 0.093
    hop_seconds: float = 0.012
    context_seconds: float = 0.080
    low_velocity: int = 24
    high_velocity: int = 120
    fallback_velocity: int = 80


def estimate_relative_velocities(
    source_audio: Path,
    note_groups: Sequence[Sequence[CanonicalNote]],
    *,
    audio_reader: ChunkAudioReader | None = None,
    config: DynamicsConfig = DynamicsConfig(),
) -> tuple[tuple[tuple[VelocityEstimate, ...], ...], tuple[str, ...]]:
    reader = audio_reader or FfmpegChunkAudioReader(source_audio)
    indexed = index_notes_by_core_chunk(note_groups, config.core_seconds)
    raw: dict[tuple[int, int], float] = {}
    warnings: list[str] = []
    for core_index, note_refs in indexed.items():
        chunk = reader.read_with_overlap(
            core_index * config.core_seconds,
            config.core_seconds,
            config.overlap_seconds,
        )
        try:
            power = chunk_power_spectrogram(chunk.samples, chunk.sample_rate, config)
            raw.update(extract_chunk_intensities(power, chunk, note_refs, config))
        except Exception as exc:
            warnings.append(f"DYNAMICS_CHUNK_FAILED: chunk={core_index}: {type(exc).__name__}: {exc}")
        finally:
            del chunk
    estimates, normalization_warnings = normalize_instrument_velocities(note_groups, raw, config)
    return estimates, tuple([*warnings, *normalization_warnings])
~~~

Assign each onset to one core chunk, compute torch.stft only for that chunk plus overlap, retain raw intensity scalars, then apply 5th-95th percentile mapping per instrument.

- [ ] **Step 4: Run tests and verify GREEN**

Run: python -m pytest tests/test_muscriptor_dynamics.py -q

Expected: all focused tests pass and the fake-reader assertion observes one live chunk.

- [ ] **Step 5: Commit**

~~~powershell
git add module/muscriptor_tool/dynamics.py tests/test_muscriptor_dynamics.py
git commit -m "feat: estimate chunked relative note dynamics"
~~~

### Task 8: Pure MIDI And Structured Exporters

**Files:**
- Create: module/muscriptor_tool/midi_export.py
- Create: module/muscriptor_tool/structured_export.py
- Modify: module/muscriptor_tool/events.py
- Modify: module/muscriptor_tool/outputs.py
- Modify: tests/test_muscriptor_events_outputs.py

**Interfaces:**
- Consumes: AnalyzedScore.
- Produces: score_to_midi_bytes(), score_to_json_payload(), iter_jsonl_records().

- [ ] **Step 1: Replace fake upstream-MIDI expectations with failing enriched contracts**

~~~python
def test_four_symbolic_views_share_timing_and_velocity(tmp_path):
    result = transcribe_once(fake_loaded, source, options, targets, timing_analyzer=fake_analyzer)
    midi = mido.MidiFile(file=io.BytesIO((tmp_path / "song.mid").read_bytes()))
    json_payload = json.loads((tmp_path / "events.json").read_text())
    jsonl = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert json_payload["schema_version"] == 2
    assert jsonl[0]["record_type"] == "metadata"
    assert json_payload["analysis"]["global_bpm"] == 90.0
    assert first_midi_velocity(midi) == first_json_velocity(json_payload)
~~~

Add Type 1 conductor/program/drum channel, end-before-repeated-on ordering, empty transcription, JSON v1 reader compatibility, JSON/JSONL equivalence and per-format partial failure tests.

- [ ] **Step 2: Run tests and verify RED**

Run: python -m pytest tests/test_muscriptor_events_outputs.py -q

Expected: JSON is still a v1 array and MIDI still calls loaded.midi_bytes.

- [ ] **Step 3: Implement pure exporters**

~~~python
def score_to_midi_bytes(score: AnalyzedScore, *, ticks_per_beat: int = 480) -> bytes:
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    midi.tracks.append(build_conductor_track(score.timing, ticks_per_beat=ticks_per_beat))
    for instrument, notes in group_notes_for_midi(score.notes):
        midi.tracks.append(build_instrument_track(instrument, notes, TimingMap(score.timing, ticks_per_beat=ticks_per_beat)))
    output = io.BytesIO()
    midi.save(file=output)
    return output.getvalue()

def score_to_json_payload(score: AnalyzedScore) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "analysis": analysis_to_dict(score.timing),
        "events": list(score_events(score)),
    }

def iter_jsonl_records(score: AnalyzedScore) -> Iterator[dict[str, Any]]:
    yield {"record_type": "metadata", "schema_version": 2, "analysis": analysis_to_dict(score.timing)}
    for event in score_events(score):
        yield {"record_type": "event", **event}
~~~

Refactor transcribe_once into collection, canonical-score construction, then independent exporter calls. JSONL may spool canonical event records but must write final metadata and enriched events atomically.

- [ ] **Step 4: Run tests and verify GREEN**

Run: python -m pytest tests/test_muscriptor_events_outputs.py tests/test_muscriptor_score.py -q

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

~~~powershell
git add module/muscriptor_tool/midi_export.py module/muscriptor_tool/structured_export.py module/muscriptor_tool/events.py module/muscriptor_tool/outputs.py tests/test_muscriptor_events_outputs.py
git commit -m "feat: export enriched MIDI and structured events"
~~~

### Task 9: MusicXML Export

**Files:**
- Create: module/muscriptor_tool/musicxml_export.py
- Create: tests/test_muscriptor_musicxml.py
- Modify: module/muscriptor_tool/options.py
- Modify: module/muscriptor_tool/outputs.py
- Modify: module/muscriptor_tool/cli.py
- Modify: tests/test_muscriptor_options.py
- Modify: tests/test_muscriptor_cli.py

**Interfaces:**
- Consumes: AnalyzedScore.
- Produces: score_to_musicxml_bytes() and OutputFormat.MUSICXML.

- [ ] **Step 1: Write failing notation and CLI tests**

~~~python
def test_musicxml_round_trip_preserves_parts_tempo_ties_and_voices(tmp_path):
    payload = score_to_musicxml_bytes(polyphonic_score_fixture())
    path = tmp_path / "score.musicxml"
    path.write_bytes(payload)
    parsed = converter.parse(path)
    assert len(parsed.parts) == 2
    assert list(parsed.recurse().getElementsByClass(tempo.MetronomeMark))
    assert any(note.tie is not None for note in parsed.recurse().notes)
    assert parsed.isWellFormedNotation()


def test_empty_score_exports_one_rest_measure():
    parsed = parse_bytes(score_to_musicxml_bytes(empty_score_fixture()))
    assert len(parsed.parts[0].getElementsByClass(stream.Measure)) == 1
    assert list(parsed.recurse().getElementsByClass(note.Rest))
~~~

Add pickup measure 0, sixteenth versus triplet quantization, chords, percussion clef/GM mapping, persistent dynamics, CLI file extension and stdout tests.

- [ ] **Step 2: Run tests and verify RED**

Run: python -m pytest tests/test_muscriptor_musicxml.py tests/test_muscriptor_options.py tests/test_muscriptor_cli.py -q

Expected: exporter and OutputFormat.MUSICXML are missing.

- [ ] **Step 3: Implement music21 builder and readback validation**

~~~python
def score_to_musicxml_bytes(score: AnalyzedScore) -> bytes:
    music21_score = build_music21_score(score)
    if not music21_score.isWellFormedNotation():
        raise MusicXmlExportError("music21 rejected the generated notation")
    with tempfile.TemporaryDirectory(prefix="muscriptor-musicxml-") as directory:
        path = Path(directory) / "score.musicxml"
        music21_score.write("musicxml", fp=path)
        _validate_musicxml_readback(path, expected=score)
        return path.read_bytes()
~~~

Use makeVoices, makeTies, makeRests and makeBeams after quantization. Add a legal rest-only part when notes are empty.

- [ ] **Step 4: Run tests and verify GREEN**

Run: python -m pytest tests/test_muscriptor_musicxml.py tests/test_muscriptor_options.py tests/test_muscriptor_cli.py -q

Expected: all focused tests pass under the installed marker-selected music21 version.

- [ ] **Step 5: Commit**

~~~powershell
git add module/muscriptor_tool/musicxml_export.py module/muscriptor_tool/options.py module/muscriptor_tool/outputs.py module/muscriptor_tool/cli.py tests/test_muscriptor_musicxml.py tests/test_muscriptor_options.py tests/test_muscriptor_cli.py
git commit -m "feat: export editable MusicXML scores"
~~~

### Task 10: Managed FluidSynth And Local Rendering

**Files:**
- Create: module/muscriptor_tool/fluidsynth_runtime.py
- Create: tests/test_muscriptor_fluidsynth.py
- Modify: module/muscriptor_tool/auralization.py
- Modify: tests/test_muscriptor_preview.py

**Interfaces:**
- Produces: resolve_fluidsynth() -> Path.
- PreviewRuntime gains fluidsynth_executable: Path.
- render_preview executes the exact absolute path.

- [ ] **Step 1: Write failing resolver security tests**

~~~python
def test_system_fluidsynth_wins_without_download(tmp_path):
    downloads = []
    resolved = resolve_fluidsynth(
        which=lambda _: str(tmp_path / "system.exe"),
        run=successful_version_probe,
        downloader=lambda *_: downloads.append("download"),
    )
    assert resolved == (tmp_path / "system.exe").resolve()
    assert downloads == []


def test_hash_mismatch_never_promotes_staging(tmp_path):
    with pytest.raises(FluidSynthUnavailable, match="SHA256"):
        install_managed_fluidsynth(cache_root=tmp_path, downloader=bad_archive_downloader)
    assert not (tmp_path / "2.5.6").exists()


def test_zip_traversal_is_rejected(tmp_path):
    with pytest.raises(FluidSynthUnavailable, match="unsafe ZIP"):
        install_from_archive(traversal_zip(), cache_root=tmp_path, expected_sha256=sha256(traversal_zip()))
~~~

Add lock double-check, interrupted staging, opt-out, non-Windows instructions, codec-before-resolver and no-preview/no-probe tests.

- [ ] **Step 2: Run tests and verify RED**

Run: python -m pytest tests/test_muscriptor_fluidsynth.py tests/test_muscriptor_preview.py -q

Expected: fluidsynth_runtime module and executable field are missing.

- [ ] **Step 3: Implement pinned Windows installer**

~~~python
FLUIDSYNTH_VERSION = "2.5.6"
WINDOWS_X64_ASSET = "fluidsynth-v2.5.6-win10-x64-cpp11.zip"
WINDOWS_X64_URL = "https://github.com/FluidSynth/fluidsynth/releases/download/v2.5.6/" + WINDOWS_X64_ASSET
WINDOWS_X64_SHA256 = "a4b8bd4f133b7b6770537f6c18b2b2b93579338d51e26f777d025e40e15a7e81"


def resolve_fluidsynth(
    *,
    allow_download: bool = True,
    cache_root: Path | None = None,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    downloader: Callable[[str, Path], None] = download_file,
    platform_name: str = sys.platform,
    machine: str = platform.machine(),
) -> Path:
    system = which("fluidsynth")
    if system and probe_fluidsynth(Path(system), run=run):
        return Path(system).resolve()
    managed = managed_executable(cache_root or default_cache_root())
    if managed.is_file() and probe_fluidsynth(managed, run=run):
        return managed.resolve()
    if not allow_download or auto_install_disabled():
        raise FluidSynthUnavailable("FluidSynth is missing and automatic installation is disabled")
    if platform_name != "win32" or machine.lower() not in {"amd64", "x86_64"}:
        raise FluidSynthUnavailable(platform_install_message(platform_name))
    return install_managed_fluidsynth(
        cache_root=cache_root or default_cache_root(),
        downloader=downloader,
        run=run,
    )
~~~

Use filelock, .part, SHA256, safe member validation, staging, manifest, atomic rename and --version.

- [ ] **Step 4: Replace upstream renderer with local absolute-path subprocess**

~~~python
@dataclass(frozen=True)
class PreviewRuntime:
    request: PreviewRequest
    soundfont_path: Path
    fluidsynth_executable: Path
    renderer_id: str = RENDERER_ID


def _synthesize_midi(runtime: PreviewRuntime, midi_path: Path) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        temp_wav = Path(handle.name)
    try:
        command = [
            str(runtime.fluidsynth_executable), "-ni", "-F", str(temp_wav),
            "-r", str(SAMPLE_RATE), str(runtime.soundfont_path), str(midi_path),
        ]
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode != 0 or not temp_wav.is_file():
            detail = result.stderr.decode(errors="replace")
            raise PreviewUnavailable(f"FluidSynth rendering failed: {detail}")
        audio, sample_rate = soundfile.read(temp_wav, dtype="float32")
        if int(sample_rate) != SAMPLE_RATE:
            raise PreviewUnavailable(f"FluidSynth returned unexpected sample rate: {sample_rate}")
        return np.asarray(audio, dtype=np.float32)
    finally:
        temp_wav.unlink(missing_ok=True)
~~~

Keep existing synth-only and comparison output behavior, but remove imports of upstream synthesize/auralize.

- [ ] **Step 5: Run tests and verify GREEN**

Run: python -m pytest tests/test_muscriptor_fluidsynth.py tests/test_muscriptor_preview.py -q

Expected: all focused tests pass without network.

- [ ] **Step 6: Commit**

~~~powershell
git add module/muscriptor_tool/fluidsynth_runtime.py module/muscriptor_tool/auralization.py tests/test_muscriptor_fluidsynth.py tests/test_muscriptor_preview.py
git commit -m "feat: manage a private FluidSynth runtime"
~~~

### Task 11: Batch, Stem And Manifest Integration

**Files:**
- Modify: module/muscriptor_tool/batch.py
- Modify: module/muscriptor_tool/manifest.py
- Modify: module/muscriptor_tool/stems.py
- Modify: tests/test_muscriptor_batch.py
- Modify: tests/test_muscriptor_stems.py

**Interfaces:**
- Consumes: TimingCache, AnalyzedScore builders and multi-group dynamics.
- Produces: schema-upgraded completion signatures and partial results.

- [ ] **Step 1: Write failing completion and reuse tests**

~~~python
def test_old_output_without_timing_signature_is_reprocessed(tmp_path):
    write_legacy_success_metadata(tmp_path)
    source = tmp_path / "song.wav"
    source.write_bytes(b"audio")
    summary = run_batch(
        source,
        output_dir=tmp_path / "out",
        options=BatchOptions(skip_completed=True),
        model_loader=loader_with_calls({}),
        transcriber=successful_transcriber({}),
        timing_analyzer=fake_analyzer,
    )
    assert summary.processed == 1


def test_stems_share_one_song_timing_and_one_dynamics_scan(tmp_path):
    calls = Counter()
    summary = transcribe_stem_candidates(
        two_stems_same_source(tmp_path),
        options,
        timing_analyzer=lambda source, overrides: calls.update(timing=1) or timing_fixture(),
        dynamics_estimator=lambda source, groups: calls.update(dynamics=1) or estimates(groups),
    )
    assert calls == Counter(timing=1, dynamics=1)
~~~

Add MusicXML requested names, cleanup ownership, unknown-file preservation, source stat invalidation, override/checkpoint/schema signature changes and single-format partial failure.

- [ ] **Step 2: Run tests and verify RED**

Run: python -m pytest tests/test_muscriptor_batch.py tests/test_muscriptor_stems.py -q

Expected: old metadata still skips and per-song analysis injection is unavailable.

- [ ] **Step 3: Implement schema/signature and two-phase stem processing**

~~~python
def transcribe_stem_candidates(
    candidates: Sequence[StemMidiCandidate],
    base_options: TranscriptionOptions,
    *,
    timing_overrides: TimingOverrides = TimingOverrides(),
    timing_analyzer=None,
    dynamics_estimator=None,
    other_instruments: Iterable[str] = (),
    preview: PreviewRequest | None = None,
    overwrite: bool = False,
    model_loader: Callable[[TranscriptionOptions], Any] | None = None,
    transcriber: Callable[..., Any] | None = None,
    preview_preflight: Callable[[PreviewRequest], Any] | None = None,
    device_resolver: Callable[[str], str] | None = None,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[StemMidiCandidate, str], None] | None = None,
) -> StemMidiSummary:
    grouped = group_candidates_by_source(candidates)
    summary = StemMidiSummary()
    for source_path, song_candidates in grouped.items():
        timing = (timing_analyzer or default_timing_analyzer)(source_path, timing_overrides)
        pending = transcribe_candidates_to_canonical_notes(song_candidates, base_options)
        velocity_groups, dynamics_warnings = (dynamics_estimator or estimate_relative_velocities)(
            source_path,
            [item.notes for item in pending],
        )
        for item, velocities in zip(pending, velocity_groups, strict=True):
            score = item.to_analyzed_score(timing, velocities, dynamics_warnings)
            summary.add(export_stem_score(item, score, preview=preview, overwrite=overwrite))
    return summary
~~~

Manifest known names include source-stem.musicxml and timing sidecars. Temporary cleanup remains nonce-pattern constrained.

- [ ] **Step 4: Run tests and verify GREEN**

Run: python -m pytest tests/test_muscriptor_batch.py tests/test_muscriptor_stems.py -q

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

~~~powershell
git add module/muscriptor_tool/batch.py module/muscriptor_tool/manifest.py module/muscriptor_tool/stems.py tests/test_muscriptor_batch.py tests/test_muscriptor_stems.py
git commit -m "feat: share enriched analysis across music batches"
~~~

### Task 12: Audio Separator Shared Timing

**Files:**
- Modify: module/audio_separator.py
- Modify: tests/test_audio_separator_onnx.py
- Modify: tests/test_audio_separator_dependency_profiles.py

**Interfaces:**
- Consumes: one TimingCache per run and TimingOverrides from CLI.
- Produces: the same MusicTiming instance for vocal and all stems of one source.

- [ ] **Step 1: Write failing parser and one-analysis tests**

~~~python
def test_audio_separator_exposes_shared_manual_timing_flags():
    args = build_parser().parse_args(["song.wav", "--vocal_midi", "--music_bpm", "88", "--music_time_signature", "3/4"])
    assert args.music_bpm == 88.0
    assert args.music_time_signature == "3/4"


def test_audio_separator_analyzes_source_once_for_vocal_and_stems(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(audio_separator, "analyze_music_timing", lambda source, **_: calls.append(source) or timing_fixture())
    run_audio_separator(args_with_vocal_and_stems(tmp_path))
    assert calls == [source_audio_path(tmp_path)]
~~~

- [ ] **Step 2: Run tests and verify RED**

Run: python -m pytest tests/test_audio_separator_onnx.py tests/test_audio_separator_dependency_profiles.py -q

Expected: parser rejects the new flags and analysis is not shared.

- [ ] **Step 3: Wire one per-song timing map**

~~~python
timing_overrides = TimingOverrides(
    bpm=args.music_bpm,
    time_signature=parse_time_signature(args.music_time_signature) if args.music_time_signature else None,
)
timing_by_source = {
    candidate.source_path: timing_cache.get_or_analyze(candidate.source_path, timing_overrides, analyzer)
    for candidate in unique_song_candidates
}
~~~

Pass timing into GameOnnxTranscriber.transcribe_file and transcribe_stem_candidates. Include sidecars in skip checks.

- [ ] **Step 4: Run tests and verify GREEN**

Run: python -m pytest tests/test_audio_separator_onnx.py tests/test_audio_separator_dependency_profiles.py -q

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

~~~powershell
git add module/audio_separator.py tests/test_audio_separator_onnx.py tests/test_audio_separator_dependency_profiles.py
git commit -m "feat: share song timing across separated MIDI"
~~~

### Task 13: Project-Owned MuScriptor Web Routes

**Files:**
- Create: module/muscriptor_tool/web_server.py
- Modify: module/muscriptor_tool/webui.py
- Modify: tests/test_muscriptor_webui.py

**Interfaces:**
- Produces: create_project_app(loaded, web_dir, timing_analyzer, preview_factory) -> FastAPI.
- Consumes: canonical/enriched exporters and local renderer.

- [ ] **Step 1: Write failing route contract tests**

~~~python
def test_project_transcribe_stream_replaces_final_upstream_midi(tmp_path):
    app = create_project_app(fake_loaded(), web_dir=None, timing_analyzer=lambda _: timing_fixture(90))
    with TestClient(app) as client:
        with client.stream("POST", "/transcribe", files={"file": ("song.wav", wav_bytes(), "audio/wav")}) as response:
            records = parse_sse(response.iter_lines())
    assert [record["type"] for record in records[:-1]] == ["start", "end"]
    assert records[-1]["type"] == "midi"
    assert midi_bpm(base64.b64decode(records[-1]["data"])) == pytest.approx(90)


def test_project_auralize_uses_local_preview_runtime(monkeypatch):
    calls = {}
    runtime = preview_runtime_fixture(Path("C:/managed/fluidsynth.exe"))
    def fake_renderer(current, *, output_path, **kwargs):
        calls["fluidsynth"] = current.fluidsynth_executable
        Path(output_path).write_bytes(wav_bytes())

    app = create_project_app(
        fake_loaded(),
        web_dir=None,
        preview_preflight=lambda request: runtime,
        preview_renderer=fake_renderer,
    )
    response = TestClient(app).post(
        "/auralize",
        files={"midi": ("song.mid", minimal_midi_bytes(), "audio/midi")},
        data={"mode": "synth"},
    )
    assert response.headers["content-type"].startswith("audio/wav")
    assert calls["fluidsynth"].is_absolute()
~~~

Also preserve health/instruments/static mounting, lock, cancellation, cache headers, multipart validation, mode validation and temporary cleanup.

- [ ] **Step 2: Run tests and verify RED**

Run: python -m pytest tests/test_muscriptor_webui.py -q

Expected: web_server module is missing and webui still calls upstream create_app directly.

- [ ] **Step 3: Implement parent app with local route priority**

~~~python
def create_project_app(
    loaded: LoadedModel,
    *,
    web_dir: Path | None,
    timing_analyzer: BeatThisAnalyzer | None = None,
    preview_preflight=preflight_preview,
) -> FastAPI:
    parent = FastAPI(title="muscriptor-project")
    _register_transcribe_route(parent, loaded, timing_analyzer or BeatThisAnalyzer())
    _register_auralize_route(parent, preview_preflight)
    parent.mount("/", upstream_create_app(loaded.model, web_dir=web_dir))
    return parent
~~~

Implement the pinned 0.2.1 SSE record contract locally and keep the upstream app mounted only after local routes.

- [ ] **Step 4: Run tests and verify GREEN**

Run: python -m pytest tests/test_muscriptor_webui.py -q

Expected: all route and adaptive-batch tests pass.

- [ ] **Step 5: Commit**

~~~powershell
git add module/muscriptor_tool/web_server.py module/muscriptor_tool/webui.py tests/test_muscriptor_webui.py
git commit -m "feat: serve enriched MIDI from MuScriptor WebUI"
~~~

### Task 14: CLI And NiceGUI Controls

**Files:**
- Modify: module/muscriptor_tool/cli.py
- Modify: gui/wizard/step6_tools.py
- Modify: tests/test_muscriptor_cli.py
- Modify: tests/test_muscriptor_gui.py
- Modify: tests/test_audio_separator_onnx.py

**Interfaces:**
- Consumes: OutputFormat.MUSICXML and TimingOverrides.
- Produces: Auto/Manual settings consistently passed to single, batch, vocal and separator paths.

- [ ] **Step 1: Write failing CLI and GUI mapping tests**

~~~python
def test_single_cli_passes_manual_timing_without_loading_beat_model(monkeypatch, tmp_path):
    result = runner.invoke(app, ["transcribe", str(source), "--bpm", "96", "--time-signature", "3/4", "-f", "musicxml"])
    assert result.exit_code == 0
    assert captured["timing_overrides"] == TimingOverrides(96.0, (3, 4))


def test_music_gui_maps_musicxml_and_manual_timing(monkeypatch, tmp_path):
    step = configured_music_step(tmp_path)
    step.music_output_formats = ["midi", "musicxml"]
    step.music_timing_mode = "manual"
    step.music_bpm = 96.0
    step.music_time_signature = "3/4"
    step.run_music_transcription()
    assert "--format" in command and "musicxml" in command
    assert command[-4:] == ["--bpm", "96.0", "--time-signature", "3/4"]
~~~

Add Auto mode omission, partial override, invalid values before model loading, batch repeatable formats and separator shared controls.

- [ ] **Step 2: Run tests and verify RED**

Run: python -m pytest tests/test_muscriptor_cli.py tests/test_muscriptor_gui.py tests/test_audio_separator_onnx.py -q

Expected: new options/controls are absent.

- [ ] **Step 3: Implement ergonomic controls and argument mapping**

Use a mode selector for Auto/Manual, numeric BPM input, time-signature select/input, MusicXML checkbox, and existing job submission patterns. Binary settings remain checkbox/toggle controls; no explanatory landing content is added.

~~~python
timing_args: list[str] = []
if self.music_timing_mode == "manual":
    timing_args.extend(["--bpm", str(self.music_bpm), "--time-signature", self.music_time_signature])
~~~

- [ ] **Step 4: Run tests and verify GREEN**

Run: python -m pytest tests/test_muscriptor_cli.py tests/test_muscriptor_gui.py tests/test_audio_separator_onnx.py -q

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

~~~powershell
git add module/muscriptor_tool/cli.py gui/wizard/step6_tools.py tests/test_muscriptor_cli.py tests/test_muscriptor_gui.py tests/test_audio_separator_onnx.py
git commit -m "feat: expose timing and MusicXML controls"
~~~

### Task 15: Full Verification And User Documentation

**Files:**
- Modify: docs/configuration.md
- Modify: docs/configuration.en.md
- Modify: docs/troubleshooting.md
- Modify: docs/troubleshooting.en.md
- Modify: tests/test_muscriptor_dependencies.py

**Interfaces:**
- Documents: model download, explicit fallback, MusicXML versions and private FluidSynth cache/opt-out.

- [ ] **Step 1: Add documentation contract assertions**

~~~python
def test_music_export_docs_name_fallback_and_private_runtime():
    text = Path("docs/configuration.md").read_text(encoding="utf-8")
    assert "120 BPM" in text
    assert "MusicXML" in text
    assert "small0" in text
    assert "FluidSynth" in text
~~~

- [ ] **Step 2: Run contract test and verify RED**

Run: python -m pytest tests/test_muscriptor_dependencies.py -q

Expected: documentation assertions fail.

- [ ] **Step 3: Update bilingual configuration and troubleshooting**

Document:

- Auto timing and manual --bpm/--time-signature.
- 120/4 fallback is explicit, not detected.
- Beat This small0 first-use download and offline manual bypass.
- Relative velocity limitations.
- MusicXML target and supported Python/music21 pins.
- Windows managed FluidSynth cache, hash validation, opt-out and Linux/macOS system install guidance.

- [ ] **Step 4: Run all focused music suites**

Run:

~~~powershell
python -m pytest tests/test_music_analysis_types.py tests/test_music_analysis_postprocess.py tests/test_music_analysis_midi_timing.py tests/test_music_analysis_runtime.py tests/test_vocal_midi_timing.py tests/test_muscriptor_score.py tests/test_muscriptor_dynamics.py tests/test_muscriptor_events_outputs.py tests/test_muscriptor_musicxml.py tests/test_muscriptor_fluidsynth.py tests/test_muscriptor_preview.py tests/test_muscriptor_batch.py tests/test_muscriptor_stems.py tests/test_muscriptor_webui.py tests/test_muscriptor_cli.py tests/test_muscriptor_gui.py tests/test_audio_separator_onnx.py tests/test_muscriptor_dependencies.py tests/test_audio_separator_dependency_profiles.py -q
~~~

Expected: zero failures and zero errors.

- [ ] **Step 5: Run repository correctness gates**

Run:

~~~powershell
python -m compileall -q module gui utils config tests
python -m ruff check module gui utils config tests --select F821,F823
python -m pytest tests -q --strict-markers
~~~

Expected: every command exits 0. Record any pre-existing failure separately; do not claim completion if a new failure remains.

- [ ] **Step 6: Run optional installed-runtime smoke only when explicitly enabled**

Run:

~~~powershell
$env:RUN_MUSCRIPTOR_REAL_SMOKE='1'
python -m pytest tests/test_muscriptor_cli.py -m real_muscriptor -q
Remove-Item Env:RUN_MUSCRIPTOR_REAL_SMOKE
~~~

Expected: skipped by default; when enabled on a prepared machine, stable/variable timing output parses and preview uses an absolute FluidSynth path.

- [ ] **Step 7: Commit documentation and final verification fixtures**

~~~powershell
git add docs/configuration.md docs/configuration.en.md docs/troubleshooting.md docs/troubleshooting.en.md tests/test_muscriptor_dependencies.py
git commit -m "docs: explain enriched music export runtime"
~~~

---

## Execution Notes

- Implement tasks in order because later exporter and integration contracts consume earlier interfaces.
- At every RED step, confirm the failure is caused by the missing behavior rather than an import typo.
- Do not combine task commits with unrelated dirty files from the original checkout.
- If the installed Beat This or music21 API differs from the pinned package, inspect the pinned wheel/source and update the adapter only; keep project-facing interfaces stable.
- If an existing unrelated baseline test fails, stop execution and report the exact failure before attributing it to this feature.

## Self-Review

Spec coverage:

- Common BPM/meter analysis, manual bypass, fallback and quality: Tasks 1-4.
- Shared seconds/beats/ticks and conductor tracks: Task 3.
- GAME vocal MIDI and metadata upgrade: Tasks 5 and 12.
- Canonical MuScriptor behavior parity: Task 6.
- Fixed-memory relative velocity: Task 7.
- MIDI, JSON v2 and JSONL equivalence: Task 8.
- MusicXML notation and Python compatibility: Tasks 4 and 9.
- Managed FluidSynth and preview isolation: Task 10.
- Batch signatures, atomicity, cleanup and per-song reuse: Tasks 11 and 12.
- Project-owned WebUI routes: Task 13.
- CLI/NiceGUI controls: Task 14.
- Bilingual documentation and full verification: Task 15.

No design requirement is intentionally deferred. Valid tuple type annotations use the Python Ellipsis token as syntax; no executable step contains an omitted function body.
