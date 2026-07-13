# MuScriptor Batch Transcription Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the complete MuScriptor v0.2.1 official-model backend, single-file and batch CLIs, and a dedicated GUI batch transcription tool with optional official-preview WAV/MP3 export.

**Architecture:** A lightweight `module.muscriptor_tool` package owns closed option types, lazy upstream loading, one-pass event fan-out, batch recovery metadata, and official preview rendering. The GUI imports only lightweight constants and launches the batch CLI in the existing task-tab process environment; torch, MuScriptor, model weights, FluidSynth, and the default SoundFont remain outside the GUI process.

**Tech Stack:** Python 3.10-3.12, Typer, MuScriptor 0.2.1, PyTorch, filelock, soundfile/libsndfile, FluidSynth, NiceGUI, pytest, uv.

## Global Constraints

- Only `small`, `medium`, and `large` official models are accepted by project CLI, GUI, and runtime.
- Use `muscriptor==0.2.1`; do not accept local weights, custom Hugging Face repositories, or URLs.
- Preview always uses MuScriptor's official default `MuseScore_General.sf2`; expose neither system synths nor custom SoundFonts.
- MIDI, JSON, and JSONL must work without FluidSynth or the default SoundFont.
- Each batch loads the model at most once and calls `transcribe()` exactly once per processed input.
- A preview is additional to at least one symbolic output and is either `midi` or `comparison`, never both.
- MP3 is the default preview format and is enabled only after a real runtime write/read probe; WAV remains the explicit fallback.
- GUI import and render must not import torch or MuScriptor and must not load or download model weights.
- Preserve all existing GAME vocal MIDI, audio separator, MuSViT, Provider V2, task-tab, and ProcessRunner behavior.
- Do not implement the upstream WebUI, piano roll, live playback, or the later single-file demo page.

---

## File Map

**Create**

- `module/muscriptor_tool/__init__.py`: dependency-free public constants and package marker.
- `module/muscriptor_tool/catalog.py`: pinned dependency-free instrument names for immediate GUI rendering.
- `module/muscriptor_tool/options.py`: enums, immutable option records, defaults, and validation.
- `module/muscriptor_tool/events.py`: stable JSON event conversion and counters.
- `module/muscriptor_tool/runtime.py`: lazy torch/MuScriptor import, device validation, official model loading, and instrument resolution.
- `module/muscriptor_tool/outputs.py`: atomic files and one-pass MIDI/JSON/JSONL fan-out.
- `module/muscriptor_tool/auralization.py`: official default SF2 preflight and WAV/MP3 preview rendering.
- `module/muscriptor_tool/manifest.py`: run signatures, metadata, manifest, and completion checks.
- `module/muscriptor_tool/batch.py`: discovery, input-local output layout, locking, skip/re-run, and file-level isolation.
- `module/muscriptor_tool/cli.py`: Typer `transcribe`, `batch`, and `list-instruments` commands.
- `tests/test_muscriptor_options.py`: closed option and decoding validation.
- `tests/test_muscriptor_events_outputs.py`: stable event schema and single-pass output fan-out.
- `tests/test_muscriptor_runtime.py`: lazy imports, device behavior, official model mapping, and gated errors.
- `tests/test_muscriptor_preview.py`: official SF2 and codec preflight without system/custom sources.
- `tests/test_muscriptor_batch.py`: discovery, one-load/one-inference, recovery, locking, and manifests.
- `tests/test_muscriptor_cli.py`: command surface, stdout/stderr, aliases, and exit codes.
- `tests/test_muscriptor_gui.py`: tab/action registration and GUI-to-CLI argument mapping.
- `tests/test_muscriptor_dependencies.py`: extra, registry, wrapper, and config contracts.
- `2.7.music_transcription.ps1`: uv wrapper for the batch command.

**Modify**

- `pyproject.toml`: add `muscriptor-local` extra and required uv conflicts.
- `config/model.toml`: add `[muscriptor]` defaults without a SoundFont field.
- `gui/utils/process_runner.py`: register module-mode CLI with `muscriptor-local`.
- `gui/wizard/step6_tools.py`: add the dedicated batch tool and command construction.
- `gui/utils/i18n.py`: add matching English, Chinese, Japanese, and Korean keys.
- `README.md`, `README.en.md`: document CLI, models, gated access, outputs, and preview prerequisites.
- `gui/README.md`, `gui/PARAMETERS.md`: document the new GUI tab and persisted settings.

---

### Task 1: Closed Options And Stable Event Schema

**Files:**
- Create: `module/muscriptor_tool/__init__.py`
- Create: `module/muscriptor_tool/options.py`
- Create: `module/muscriptor_tool/events.py`
- Create: `tests/test_muscriptor_options.py`

**Interfaces:**
- Produces: `ModelVariant`, `DecodingMode`, `OutputFormat`, `PreviewContent`, `PreviewFormat`, `PreviewRequest`, `TranscriptionOptions`, `BatchOptions`, `event_to_dict()`, and `EventStats`.
- Consumes: no torch or MuScriptor import.

- [ ] **Step 1: Write failing closed-model and preview-state tests**

```python
from dataclasses import FrozenInstanceError

import pytest

from module.muscriptor_tool.options import (
    BatchOptions,
    DecodingMode,
    ModelVariant,
    OutputFormat,
    PreviewContent,
    PreviewFormat,
    PreviewRequest,
    TranscriptionOptions,
)


def test_model_variant_rejects_paths_repos_and_urls():
    for value in ("model.safetensors", "Org/repo", "hf://Org/repo/model.safetensors", "https://example/model"):
        with pytest.raises(ValueError):
            ModelVariant(value)


def test_preview_is_none_or_one_complete_request():
    preview = PreviewRequest(content=PreviewContent.COMPARISON, format=PreviewFormat.WAV)
    options = BatchOptions(output_formats=(OutputFormat.MIDI,), preview=preview)
    assert options.preview == preview
    with pytest.raises(FrozenInstanceError):
        preview.format = PreviewFormat.MP3


def test_sampling_and_beam_cannot_coexist():
    with pytest.raises(ValueError, match="sampling.*beam"):
        TranscriptionOptions.from_single_cli(sampling=True, beam_size=2)


def test_batch_requires_a_symbolic_output():
    with pytest.raises(ValueError, match="symbolic output"):
        BatchOptions(output_formats=())
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run: `python -m pytest tests/test_muscriptor_options.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'module.muscriptor_tool'`.

- [ ] **Step 3: Add immutable option types and centralized validation**

```python
class ModelVariant(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class DecodingMode(str, Enum):
    GREEDY = "greedy"
    SAMPLING = "sampling"
    BEAM = "beam"


class OutputFormat(str, Enum):
    MIDI = "midi"
    JSON = "json"
    JSONL = "jsonl"


class PreviewContent(str, Enum):
    MIDI = "midi"
    COMPARISON = "comparison"


class PreviewFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"


@dataclass(frozen=True)
class PreviewRequest:
    content: PreviewContent
    format: PreviewFormat


@dataclass(frozen=True)
class TranscriptionOptions:
    model: ModelVariant = ModelVariant.MEDIUM
    device: str = "auto"
    batch_size: int | None = None
    decode_mode: DecodingMode = DecodingMode.GREEDY
    temperature: float = 1.0
    cfg_coef: float = 1.0
    strict_eos: bool = False
    beam_size: int = 1
    instruments: tuple[str, ...] = ()
    print_notes: bool = False

    def upstream_kwargs(self) -> dict[str, object]:
        return {
            "use_sampling": self.decode_mode is DecodingMode.SAMPLING,
            "temperature": self.temperature,
            "cfg_coef": self.cfg_coef,
            "instruments": list(self.instruments) or None,
            "batch_size": self.batch_size,
            "no_eos_is_ok": not self.strict_eos,
            "beam_size": self.beam_size,
        }


@dataclass(frozen=True)
class BatchOptions:
    transcription: TranscriptionOptions = field(default_factory=TranscriptionOptions)
    output_formats: tuple[OutputFormat, ...] = (OutputFormat.MIDI,)
    preview: PreviewRequest | None = None
    recursive: bool = True
    skip_completed: bool = True
    fail_fast: bool = False
```

Implement `__post_init__` and `from_single_cli()` so batch size is positive or `None`, device matches `auto|cpu|cuda|cuda:N`, temperature is finite and positive, CFG is finite, greedy/sampling use beam size 1, beam mode uses width at least 2, and non-sampling modes reject a non-default temperature.

- [ ] **Step 4: Add stable event conversion tests and implementation**

```python
def test_event_to_dict_uses_stable_end_reference():
    start = SimpleNamespace(pitch=60, start_time=1.25, index=7, instrument="piano")
    end = SimpleNamespace(end_time=2.5, start_event=start, start_event_index=7)
    assert event_to_dict(start) == {
        "type": "start", "pitch": 60, "start_time": 1.25, "index": 7, "instrument": "piano"
    }
    assert event_to_dict(end) == {"type": "end", "end_time": 2.5, "start_event_index": 7}
```

Implement attribute-based conversion without importing MuScriptor. Reject unknown event shapes with `TypeError` instead of serializing arbitrary objects.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest tests/test_muscriptor_options.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add module/muscriptor_tool/__init__.py module/muscriptor_tool/options.py module/muscriptor_tool/events.py tests/test_muscriptor_options.py
git commit -m "feat: define MuScriptor transcription contracts"
```

---

### Task 2: Lazy Official Runtime And One-Pass Symbolic Outputs

**Files:**
- Create: `module/muscriptor_tool/runtime.py`
- Create: `module/muscriptor_tool/outputs.py`
- Create: `tests/test_muscriptor_runtime.py`
- Create: `tests/test_muscriptor_events_outputs.py`

**Interfaces:**
- Consumes: `TranscriptionOptions`, `OutputFormat`, and `event_to_dict()` from Task 1.
- Produces: `LoadedModel`, `load_model()`, `list_instruments()`, `resolve_instruments()`, `OutputTargets`, `TranscriptionResult`, and `transcribe_once()`.

- [ ] **Step 1: Write failing lazy-import and official-model tests**

```python
def test_importing_runtime_does_not_import_torch_or_muscriptor():
    sys.modules.pop("module.muscriptor_tool.runtime", None)
    before = set(sys.modules)
    importlib.import_module("module.muscriptor_tool.runtime")
    added = set(sys.modules) - before
    assert "torch" not in added
    assert "muscriptor" not in added


def test_load_model_passes_only_official_variant(fake_upstream):
    loaded = load_model(TranscriptionOptions(model=ModelVariant.SMALL, device="cpu"), upstream=fake_upstream)
    assert fake_upstream.model_cls.calls == [{"weights_path": "small", "device": "cpu"}]
    assert loaded.resolved_device == "cpu"
```

Use injected fake upstream objects in tests; do not install or load real weights.

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_muscriptor_runtime.py -q`

Expected: failures report missing `load_model` and `LoadedModel`.

- [ ] **Step 3: Implement lazy runtime boundaries**

```python
@dataclass(frozen=True)
class UpstreamBindings:
    torch: Any
    model_cls: type
    progress_event_type: type
    version: str


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    package_version: str
    requested_device: str
    resolved_device: str
    progress_event_type: type

    def transcribe(self, source: Path, options: TranscriptionOptions):
        return self.model.transcribe(audio=source, **options.upstream_kwargs())

    def midi_bytes(self, events: Iterable[Any]) -> bytes:
        return self.model.events_to_midi_bytes(iter(events))


def load_model(options: TranscriptionOptions, *, upstream: UpstreamBindings | None = None) -> LoadedModel:
    bindings = upstream or _import_upstream()
    device = resolve_device(options.device, bindings.torch)
    model = bindings.model_cls.load_model(weights_path=options.model.value, device=device)
    return LoadedModel(
        model=model,
        package_version=bindings.version,
        requested_device=options.device,
        resolved_device=str(model._device),
        progress_event_type=bindings.progress_event_type,
    )
```

`_import_upstream()` performs all torch/MuScriptor imports inside the function. Map `ModelDownloadError`, `GatedRepoError`, and 401/403 failures to one actionable error that names the official model page and `hf auth login`. Explicit CUDA failures never fall back to CPU.

- [ ] **Step 4: Write the failing one-pass fan-out test**

```python
def test_transcribe_once_fans_one_event_stream_to_all_formats(tmp_path):
    loaded = FakeLoadedModel(events=[progress(0, 1), start(60, 0.0, 0), end(1.0, 0), progress(1, 1)])
    targets = OutputTargets.for_directory(tmp_path, (OutputFormat.MIDI, OutputFormat.JSON, OutputFormat.JSONL))
    result = transcribe_once(loaded, Path("song.wav"), TranscriptionOptions(), targets)
    assert loaded.transcribe_calls == 1
    assert loaded.midi_calls == 1
    assert json.loads((tmp_path / "events.json").read_text())[-1]["type"] == "end"
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 2
    assert (tmp_path / "song.mid").read_bytes() == b"MThd..."
    assert result.note_count == 1
    assert result.event_count == 2
```

- [ ] **Step 5: Run and verify RED**

Run: `python -m pytest tests/test_muscriptor_events_outputs.py::test_transcribe_once_fans_one_event_stream_to_all_formats -q`

Expected: failure reports missing `OutputTargets` or `transcribe_once`.

- [ ] **Step 6: Implement atomic one-pass fan-out**

`OutputTargets.for_directory()` maps MIDI to the source stem plus `.mid`, JSON to `events.json`, and JSONL to `events.jsonl`. `transcribe_once()` must:

```python
events: list[Any] = []
stats = EventStats()
with atomic_text_writer(targets.jsonl) if targets.jsonl else nullcontext(None) as jsonl:
    for event in loaded.transcribe(source, options):
        if isinstance(event, loaded.progress_event_type):
            stats.observe_progress(event.completed, event.total)
            continue
        payload = event_to_dict(event)
        stats.observe(payload)
        if jsonl is not None:
            jsonl.write(json.dumps(payload, ensure_ascii=False) + "\n")
            jsonl.flush()
        if targets.needs_event_collection:
            events.append(event)
        if options.print_notes:
            print(event, file=stderr)
```

After iteration, serialize requested JSON, call `loaded.midi_bytes(events)` once when MIDI or preview needs it, and atomically replace unique same-directory temporary files. Media temporary names retain `.wav` or `.mp3` as the final suffix. Empty streams produce an empty MIDI, `[]`, and an empty JSONL file plus `EMPTY_TRANSCRIPTION` warning.

- [ ] **Step 7: Run runtime and output tests**

Run: `python -m pytest tests/test_muscriptor_runtime.py tests/test_muscriptor_events_outputs.py -q`

Expected: all tests pass.

- [ ] **Step 8: Commit**

```powershell
git add module/muscriptor_tool/runtime.py module/muscriptor_tool/outputs.py tests/test_muscriptor_runtime.py tests/test_muscriptor_events_outputs.py
git commit -m "feat: add one-pass MuScriptor runtime outputs"
```

---

### Task 3: Batch Discovery, Recovery Metadata, And Locking

**Files:**
- Create: `module/muscriptor_tool/manifest.py`
- Create: `module/muscriptor_tool/batch.py`
- Create: `tests/test_muscriptor_batch.py`

**Interfaces:**
- Consumes: `BatchOptions`, `load_model()`, `OutputTargets`, and `transcribe_once()`.
- Produces: `discover_inputs()`, `item_output_dir()`, `run_signature()`, `is_item_complete()`, `BatchSummary`, and `run_batch()`.

- [ ] **Step 1: Write failing discovery and layout tests**

```python
def test_discovery_preserves_relative_paths_and_prunes_output_tree(tmp_path):
    source = tmp_path / "album"
    output = source / "generated"
    (source / "disc1").mkdir(parents=True)
    output.mkdir()
    (source / "disc1" / "song.wav").write_bytes(b"audio")
    (output / "old.mp3").write_bytes(b"preview")
    found = discover_inputs(source, output_dir=output, recursive=True)
    assert [item.relative_path.as_posix() for item in found] == ["disc1/song.wav"]
    assert item_output_dir(output, found[0]) == output / "disc1" / "song.wav"


def test_same_input_and_output_directory_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="output directory"):
        discover_inputs(tmp_path, output_dir=tmp_path, recursive=True)
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_muscriptor_batch.py -q`

Expected: collection fails because `module.muscriptor_tool.batch` does not exist.

- [ ] **Step 3: Implement deterministic discovery and known output paths**

Use the project's supported audio suffixes plus `.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg`, and `.aac`; explicitly exclude `.mid` and `.midi`. Resolve the output root before traversal, prune its entire subtree, do not follow directory symlinks, and sort normalized relative paths.

- [ ] **Step 4: Write failing signature, skip, and one-load tests**

```python
def test_batch_loads_once_and_transcribes_each_pending_file_once(tmp_path, monkeypatch):
    inputs = make_audio_tree(tmp_path, ["a.wav", "sub/b.flac"])
    calls = {"loads": 0, "files": []}
    monkeypatch.setattr(batch, "load_model", fake_loader(calls))
    monkeypatch.setattr(batch, "transcribe_once", fake_transcriber(calls))
    summary = run_batch(inputs, tmp_path / "out", BatchOptions())
    assert calls == {"loads": 1, "files": ["a.wav", "b.flac"]}
    assert summary.processed == 2


def test_all_complete_items_skip_without_model_or_preview_preflight(tmp_path, monkeypatch):
    seed_complete_metadata(tmp_path)
    monkeypatch.setattr(batch, "load_model", fail_if_called)
    monkeypatch.setattr(batch, "preflight_preview", fail_if_called)
    summary = run_batch(tmp_path / "inputs", tmp_path / "out", BatchOptions())
    assert summary.skipped == 1
```

- [ ] **Step 5: Implement signatures, metadata, manifests, and file lock**

`run_signature()` hashes canonical UTF-8 JSON containing source relative path/size/`mtime_ns`, MuScriptor version, official model variant, resolved device, instruments, normalized decoding options, symbolic formats, preview content/format, and fixed renderer id `muscriptor-0.2.1:SF2_URL`. `is_item_complete()` requires parseable schema-2 metadata with status `ok`, matching signature, and every requested output present. Item metadata keeps requested instruments separate from the actual instruments summarized from note-start events.

`run_batch()` acquires `<output>/.muscriptor.lock` before loading a model. It discovers and checks completion first, loads once only when pending work exists, processes files serially, records `ok|partial|failed`, continues unless `fail_fast`, writes metadata last, and always atomically writes a run manifest for normal, partial, and fail-fast completion.

- [ ] **Step 6: Add stale-output and partial-result tests**

```python
def test_switching_preview_format_removes_only_old_known_preview(tmp_path):
    item_dir = tmp_path / "out" / "song.wav"
    item_dir.mkdir(parents=True)
    (item_dir / "preview.wav").write_bytes(b"old")
    (item_dir / "keep.txt").write_text("user")
    prune_known_outputs(item_dir, requested={"preview.mp3", "song.mid"}, output_stem="song")
    assert not (item_dir / "preview.wav").exists()
    assert (item_dir / "keep.txt").read_text() == "user"


def test_preview_failure_keeps_symbolic_outputs_and_marks_partial(tmp_path):
    summary = run_with_preview_failure(tmp_path)
    metadata = json.loads(next(tmp_path.rglob("metadata.json")).read_text())
    assert summary.partial == 1
    assert metadata["status"] == "partial"
    assert (next(tmp_path.rglob("song.mid"))).exists()
```

- [ ] **Step 7: Run batch tests**

Run: `python -m pytest tests/test_muscriptor_batch.py -q`

Expected: all tests pass.

- [ ] **Step 8: Commit**

```powershell
git add module/muscriptor_tool/manifest.py module/muscriptor_tool/batch.py tests/test_muscriptor_batch.py
git commit -m "feat: add recoverable MuScriptor batch processing"
```

---

### Task 4: Official Preview Rendering

**Files:**
- Create: `module/muscriptor_tool/auralization.py`
- Create: `tests/test_muscriptor_preview.py`
- Modify: `module/muscriptor_tool/outputs.py`
- Modify: `module/muscriptor_tool/batch.py`

**Interfaces:**
- Consumes: `PreviewRequest` and temporary MIDI bytes from Tasks 1-3.
- Produces: `PreviewRuntime`, `preflight_preview()`, and `render_preview()`.

- [ ] **Step 1: Write failing codec and official-source tests**

```python
def test_preview_api_has_no_soundfont_or_system_synth_parameter():
    assert "soundfont_path" not in inspect.signature(preflight_preview).parameters
    assert "soundfont_path" not in inspect.signature(render_preview).parameters
    assert "system_synth" not in inspect.signature(render_preview).parameters


def test_mp3_probe_writes_and_reads_requested_channel_shape(tmp_path):
    fake_sf = RecordingSoundFile(tmp_path)
    runtime = preflight_preview(
        PreviewRequest(PreviewContent.COMPARISON, PreviewFormat.MP3),
        soundfile_module=fake_sf,
        which=lambda name: "fluidsynth",
        run=fake_successful_fluidsynth,
        resolve_default_sf2=lambda: tmp_path / "MuseScore_General.sf2",
    )
    assert fake_sf.written_shape[1] == 2
    assert runtime.soundfont_path.name == "MuseScore_General.sf2"
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_muscriptor_preview.py -q`

Expected: missing module/function failures.

- [ ] **Step 3: Implement cheap-to-expensive preflight**

```python
@dataclass(frozen=True)
class PreviewRuntime:
    request: PreviewRequest
    soundfont_path: Path
    renderer_id: str = "muscriptor-0.2.1:SF2_URL"


def preflight_preview(request: PreviewRequest, *, soundfile_module=None, which=shutil.which, run=subprocess.run,
                      resolve_default_sf2=None) -> PreviewRuntime:
    sf = soundfile_module or importlib.import_module("soundfile")
    _probe_codec(sf, request.format, channels=1 if request.content is PreviewContent.MIDI else 2)
    executable = which("fluidsynth")
    if not executable:
        raise PreviewUnavailable("FluidSynth was not found on PATH")
    probe = run([executable, "--version"], capture_output=True, check=False)
    if probe.returncode != 0:
        raise PreviewUnavailable("FluidSynth could not start")
    resolver = resolve_default_sf2 or _official_soundfont_resolver()
    return PreviewRuntime(request=request, soundfont_path=Path(resolver()))
```

The default resolver lazily imports MuScriptor's pinned auralization module and resolves only its official default. There is no path argument, environment override, `gm.dls`, Microsoft GS synth, or custom `.sf2` branch.

- [ ] **Step 4: Implement pure and comparison rendering through upstream functions**

`render_preview()` writes MIDI bytes to a unique temporary `.mid`, invokes upstream `synthesize()` for `midi` or `auralize()` for `comparison`, passes the internally resolved default SF2, writes to a unique `.part.wav`/`.part.mp3`, and atomically replaces the target. It always removes temporary MIDI/media files in `finally`.

- [ ] **Step 5: Connect preview to one-pass output and batch preflight ordering**

Only collect events for preview when MIDI/JSON did not already require collection. `run_batch()` performs completion scanning first; if every item skips it never calls `preflight_preview()`. With pending work, it preflights once before model loading and reuses the `PreviewRuntime` for every item.

- [ ] **Step 6: Run preview plus output tests**

Run: `python -m pytest tests/test_muscriptor_preview.py tests/test_muscriptor_events_outputs.py tests/test_muscriptor_batch.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add module/muscriptor_tool/auralization.py module/muscriptor_tool/outputs.py module/muscriptor_tool/batch.py tests/test_muscriptor_preview.py tests/test_muscriptor_events_outputs.py tests/test_muscriptor_batch.py
git commit -m "feat: render official MuScriptor previews"
```

---

### Task 5: Typer CLI Contracts

**Files:**
- Create: `module/muscriptor_tool/cli.py`
- Create: `tests/test_muscriptor_cli.py`

**Interfaces:**
- Consumes: all backend interfaces from Tasks 1-4.
- Produces: module entry point with `transcribe`, `batch`, and `list-instruments`.

- [ ] **Step 1: Write failing help and forbidden-surface tests**

```python
runner = CliRunner()


def test_transcribe_help_exposes_model_capabilities_without_custom_sources():
    result = runner.invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0
    for option in ("--model", "--device", "--sampling", "--temperature", "--cfg-coef", "--batch-size",
                   "--strict-eos", "--beam-size", "--preview", "--preview-mode", "--instruments"):
        assert option in result.stdout
    assert "--soundfont" not in result.stdout
    assert "PATH|URL" not in result.stdout


def test_batch_help_has_complete_batch_surface():
    result = runner.invoke(app, ["batch", "--help"])
    for option in ("--output-dir", "--format", "--preview-mode", "--preview-format", "--decode-mode",
                   "--recursive", "--skip-completed", "--fail-fast", "--notes"):
        assert option in result.stdout
    assert "--overwrite" not in result.stdout
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_muscriptor_cli.py -q`

Expected: import fails because `cli.py` does not exist.

- [ ] **Step 3: Implement the Typer command surface**

Create `app = typer.Typer(add_completion=False)` and the exact commands from the spec. `transcribe` keeps upstream-compatible raw flags and normalizes them into `TranscriptionOptions`; `batch` uses the explicit `--decode-mode` enum. `--preview` and `--auralize` are aliases for the same single-file destination. Batch `--format` is repeatable and defaults to MIDI only when absent.

- [ ] **Step 4: Add stdout/stderr and path-collision tests**

```python
def test_jsonl_stdout_stays_machine_readable_with_preview(monkeypatch, tmp_path):
    install_fake_backend(monkeypatch)
    preview = tmp_path / "preview.wav"
    result = runner.invoke(app, ["transcribe", "song.wav", "-f", "jsonl", "-o", "-", "--preview", str(preview)])
    assert result.exit_code == 0
    assert all(json.loads(line) for line in result.stdout.splitlines())
    assert "Loading model" not in result.stdout


def test_input_main_output_and_preview_must_be_distinct(tmp_path):
    source = tmp_path / "song.wav"
    source.write_bytes(b"audio")
    result = runner.invoke(app, ["transcribe", str(source), "-o", str(source)])
    assert result.exit_code == 2
```

MIDI stdout uses `sys.stdout.buffer`; JSON is one newline-terminated array; JSONL flushes every object; all logging and notes use stderr. Parameter errors exit 2, runtime/output failures exit 1, and interrupts exit 130.

- [ ] **Step 5: Add list-instruments text/JSON tests**

```python
def test_list_instruments_json_matches_text_order(monkeypatch):
    monkeypatch.setattr(cli, "list_instruments", lambda: ("piano", "drums"))
    text_result = runner.invoke(app, ["list-instruments"])
    json_result = runner.invoke(app, ["list-instruments", "--format", "json"])
    assert text_result.stdout.splitlines() == ["piano", "drums"]
    assert json.loads(json_result.stdout) == {"schema_version": 1, "instruments": ["piano", "drums"]}
```

- [ ] **Step 6: Run CLI tests**

Run: `python -m pytest tests/test_muscriptor_cli.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add module/muscriptor_tool/cli.py tests/test_muscriptor_cli.py
git commit -m "feat: expose MuScriptor CLI commands"
```

---

### Task 6: Dependency Profile, Registry, Config, And Wrapper

**Files:**
- Create: `tests/test_muscriptor_dependencies.py`
- Create: `2.7.music_transcription.ps1`
- Modify: `pyproject.toml`
- Modify: `config/model.toml`
- Modify: `gui/utils/process_runner.py`

**Interfaces:**
- Produces: the `muscriptor-local` environment and ProcessRunner module target.

- [ ] **Step 1: Write failing dependency and registry tests**

```python
def test_muscriptor_extra_is_pinned_and_uses_torch_base():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = project["project"]["optional-dependencies"]["muscriptor-local"]
    assert "qinglong-captions[torch-base]" in deps
    assert "muscriptor==0.2.1" in deps
    assert any(dep.startswith("filelock") for dep in deps)


def test_registry_runs_cli_in_module_mode():
    assert SCRIPT_REGISTRY["module.muscriptor_tool.cli"] == ("-m:module.muscriptor_tool.cli", "muscriptor-local")


def test_config_has_no_model_source_or_soundfont():
    cfg = tomllib.loads((ROOT / "config/model.toml").read_text(encoding="utf-8"))["muscriptor"]
    assert cfg["model"] == "large"
    assert cfg["output_dir"] == ""
    assert cfg["preview_mode"] == "none"
    assert "model_source" not in cfg
    assert "soundfont_path" not in cfg
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_muscriptor_dependencies.py -q`

Expected: failures report missing extra, registry entry, config section, and wrapper.

- [ ] **Step 3: Add dependency and config entries**

```toml
muscriptor-local = [
    "qinglong-captions[torch-base]",
    "muscriptor==0.2.1",
    "filelock>=3.16",
]
```

```toml
[muscriptor]
model = "large"
device = "auto"
batch_size = 0
instruments = []
decode_mode = "greedy"
temperature = 1.0
cfg_coef = 1.0
strict_eos = false
beam_size = 1
output_dir = ""
output_formats = ["midi"]
preview_mode = "none"
preview_format = "mp3"
recursive = true
skip_completed = true
fail_fast = false
print_notes = false
```

Add required uv conflict entries following existing torch-extra patterns. Do not add `uv.lock`: this repository intentionally uses incremental task profiles without a checked global lock.

- [ ] **Step 4: Register ProcessRunner and add PowerShell wrapper**

Registry entry:

```python
"module.muscriptor_tool.cli": ("-m:module.muscriptor_tool.cli", "muscriptor-local"),
```

The wrapper accepts remaining arguments, locates `uv` and the selected project Python, incrementally installs the profile, and executes:

```powershell
& $uv pip install --python $PythonExe -r pyproject.toml --extra muscriptor-local
& $PythonExe -m module.muscriptor_tool.cli batch @Arguments
exit $LASTEXITCODE
```

It must not recreate Python validation or silently change stdout/stderr.

- [ ] **Step 5: Resolve and verify dependencies without creating a lock**

Run: `uv pip install --dry-run --python <task-python> -r pyproject.toml --extra muscriptor-local`

Expected: exit 0 and the plan includes `muscriptor==0.2.1` without creating `uv.lock`.

- [ ] **Step 6: Run dependency tests**

Run: `python -m pytest tests/test_muscriptor_dependencies.py tests/test_pyproject_uv_conflicts.py tests/test_pyproject_uv_build_deps.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add pyproject.toml config/model.toml gui/utils/process_runner.py 2.7.music_transcription.ps1 tests/test_muscriptor_dependencies.py
git commit -m "build: add MuScriptor runtime profile"
```

---

### Task 7: Dedicated GUI Batch Tool

**Files:**
- Create: `tests/test_muscriptor_gui.py`
- Modify: `gui/wizard/step6_tools.py`
- Modify: `gui/utils/i18n.py`

**Interfaces:**
- Consumes: lightweight constants from `module.muscriptor_tool.options` and ProcessRunner registry from Task 6.
- Produces: `music_transcription` tool tab and `_start_music_transcription()`.

- [ ] **Step 1: Write failing tab/action/default tests**

```python
def test_tools_step_exposes_music_transcription_between_audio_and_sheet_music():
    tabs = list(ToolsStep.TOOL_TABS)
    music = ("music_transcription", "music_transcription", "piano")
    assert music in tabs
    assert tabs.index(music) == tabs.index(("audio_separator", "audio_separator", "graphic_eq")) + 1


def test_music_transcription_defaults_have_no_custom_sources():
    step = ToolsStep()
    assert step.config["music_transcription_model"] == "large"
    assert step.config["music_transcription_preview_mode"] == "none"
    assert not any("soundfont" in key or "model_source" in key for key in step.config)
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_muscriptor_gui.py -q`

Expected: assertions fail because the tab and defaults are absent.

- [ ] **Step 3: Add tab, renderer, state containers, and mode callbacks**

Add the tab between audio separator and sheet music, map it in `_get_tool_renderer()` and `_tool_action_for_tab()`, and initialize exact keys for model, device, inference batch size, instrument mode/list, decode mode, temperature, CFG, strict EOS, output formats, preview mode/format, skip, and notes. Input type comes from the selected path; output location, recursive discovery, and continue-on-error use backend defaults.

Render compact full-width groups using existing `styled_select`, `editable_slider`, `toggle_switch`, `create_path_selector`, and `ui.toggle`. Use `styled_select(multiple=True)` for instruments and output formats so labels do not overlap the dropdown. Do not nest a new card inside the tool card. Preview controls expose only Off, Pure MIDI, Comparison, WAV, and MP3; no SoundFont field exists.

- [ ] **Step 4: Add the pinned lightweight instrument catalog**

Add a dependency-free snapshot of the official `MT3_FULL_PLUS_GROUP_NAMES` for the pinned `muscriptor==0.2.1` release. Render it immediately in the GUI without a child process or runtime installation. Keep the CLI `list-instruments` command dynamic, and require a catalog/test update whenever the package pin changes. Never import MuScriptor or torch in `step6_tools.py`.

- [ ] **Step 5: Write failing GUI-to-CLI mapping tests**

```python
def test_music_transcription_maps_complete_batch_args(monkeypatch, tmp_path):
    step = configured_music_step(tmp_path)
    captured = capture_run_job(step)
    asyncio.run(step._start_music_transcription())
    assert captured["script_key"] == "module.muscriptor_tool.cli"
    assert captured["args"][0] == "batch"
    for arg in (
        "--model=large", "--device=cuda:0", "--batch-size=3", "--decode-mode=sampling",
        "--temperature=0.8", "--cfg-coef=1.5", "--format=midi", "--format=jsonl",
        "--preview-mode=comparison", "--preview-format=mp3",
        "--no-skip-completed", "--notes",
    ):
        assert arg in captured["args"]
    assert not any("soundfont" in arg for arg in captured["args"])
    assert not any(arg.startswith("--output-dir") or "recursive" in arg for arg in captured["args"])


def test_preview_off_omits_preview_format(tmp_path):
    step = configured_music_step(tmp_path, preview_mode="none")
    captured = capture_run_job(step)
    asyncio.run(step._start_music_transcription())
    assert "--preview-mode=none" not in captured["args"]
    assert not any(arg.startswith("--preview-format") for arg in captured["args"])
```

- [ ] **Step 6: Implement command construction and validation**

Validate that the input path exists and is a file or directory, plus non-empty output format selection, beam width, and active-mode parameters before submitting. Build arguments from current controls in one function; hidden sampling, beam, preview, and instrument controls never leak stale values. Submit through the existing ExecutionPanel with script key `module.muscriptor_tool.cli`, first positional argument `batch`, translated job name, and existing Start/Stop/log behavior.

- [ ] **Step 7: Add four-language keys and parity assertions**

Add translations for the tab, description, MuScriptor model/device, instrument mode, decoding modes, CFG, strict EOS, output formats, preview modes/formats, skip option, start/success/failure/log/job labels. Extend `test_gui_i18n.py`'s recent-key list with `music_transcription`, `music_transcription_preview`, and `job_name_music_transcription`.

- [ ] **Step 8: Run GUI tests**

Run: `python -m pytest tests/test_muscriptor_gui.py tests/test_gui_i18n.py tests/test_sheet_music_musvit_tools.py tests/test_execution_panel.py -q`

Expected: all tests pass and importing `step6_tools` does not add torch or MuScriptor to `sys.modules`.

- [ ] **Step 9: Commit**

```powershell
git add gui/wizard/step6_tools.py gui/utils/i18n.py tests/test_muscriptor_gui.py tests/test_gui_i18n.py
git commit -m "feat: add MuScriptor batch GUI tool"
```

---

### Task 8: Documentation, Real Smoke Gate, And Full Verification

**Files:**
- Modify: `README.md`
- Modify: `README.en.md`
- Modify: `gui/README.md`
- Modify: `gui/PARAMETERS.md`
- Modify: `tests/test_muscriptor_cli.py`

**Interfaces:**
- Consumes: completed backend, CLI, dependency, and GUI surfaces.
- Produces: user documentation and gated real-model checks.

- [ ] **Step 1: Add documentation contract tests**

```python
def test_readmes_document_official_models_and_preview_boundary():
    for relative in ("README.md", "README.en.md", "gui/README.md", "gui/PARAMETERS.md"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "MuScriptor" in text
        assert "small" in text and "medium" in text and "large" in text
        assert "MIDI" in text and "JSONL" in text
        assert "SoundFont" in text
```

The prose must say that the official default SoundFont is automatic and not configurable, not imply that a SoundFont control exists.

- [ ] **Step 2: Document installation, authentication, CLI, GUI, outputs, and licenses**

Include the repository's `uv pip install --python <task-python> -r pyproject.toml --extra muscriptor-local` workflow, Hugging Face gate acceptance plus `hf auth login`, the three official repos, single and batch examples, output tree, official preview modes, FluidSynth installation/detection, runtime-dependent MP3, WAV fallback, CC BY-NC 4.0 weight restrictions, and the explicit absence of system/custom SoundFonts and WebUI features.

- [ ] **Step 3: Add environment-gated real smoke tests**

```python
@pytest.mark.skipif(os.getenv("MUSCRIPTOR_SMOKE") != "1", reason="requires gated official weights")
def test_real_small_cpu_midi_smoke(sample_audio):
    result = subprocess.run(
        [sys.executable, "-m", "module.muscriptor_tool.cli", "transcribe", str(sample_audio),
         "--model", "small", "--device", "cpu", "--format", "midi"],
        check=False,
    )
    assert result.returncode == 0
```

Add gated batch CUDA and preview smoke cases only when the environment exposes CUDA/FluidSynth/MP3 support. Default CI never downloads weights.

- [ ] **Step 4: Run all focused MuScriptor tests**

Run:

```powershell
python -m pytest tests/test_muscriptor_options.py tests/test_muscriptor_runtime.py tests/test_muscriptor_events_outputs.py tests/test_muscriptor_preview.py tests/test_muscriptor_batch.py tests/test_muscriptor_cli.py tests/test_muscriptor_dependencies.py tests/test_muscriptor_gui.py -q
```

Expected: all tests pass, with real smoke tests skipped unless `MUSCRIPTOR_SMOKE=1`.

- [ ] **Step 5: Run lint on changed Python files**

Run:

```powershell
python -m ruff check module/muscriptor_tool gui/wizard/step6_tools.py gui/utils/process_runner.py gui/utils/i18n.py tests/test_muscriptor_*.py
```

Expected: exit 0 with no lint errors.

- [ ] **Step 6: Run dependency and regression verification**

Run:

```powershell
uv pip install --dry-run --python <task-python> -r pyproject.toml --extra muscriptor-local
python -m pytest tests/test_audio_separator_dependency_profiles.py tests/test_audio_separator_onnx.py tests/test_sheet_music_musvit.py tests/test_sheet_music_musvit_tools.py tests/test_process_runner_native_resources.py tests/test_job_manager_task_tabs.py tests/test_gui_main_lazy_import.py -q
```

Expected: exit 0 and all selected regressions pass.

- [ ] **Step 7: Run the complete test suite**

Run: `python -m pytest -q`

Expected: exit 0 with no failures. Record pre-existing skips separately from failures.

- [ ] **Step 8: Verify CLI help without model loading**

Run:

```powershell
python -m module.muscriptor_tool.cli --help
python -m module.muscriptor_tool.cli transcribe --help
python -m module.muscriptor_tool.cli batch --help
```

Expected: all exit 0, expose no `--soundfont` or custom model source, and do not download weights.

- [ ] **Step 9: Commit documentation and final verification changes**

```powershell
git add README.md README.en.md gui/README.md gui/PARAMETERS.md tests/test_muscriptor_cli.py
git commit -m "docs: document MuScriptor batch transcription"
```

---

## Self-Review Checklist

- [ ] Every spec target maps to a task and test above.
- [ ] The project surfaces accept only official model variants.
- [ ] Neither CLI nor GUI accepts a SoundFont or system synth.
- [ ] Preview remains optional and cannot be the only output.
- [ ] All heavy imports occur only in runtime child processes.
- [ ] One batch model load and one per-file transcription are asserted.
- [ ] JSONL streaming and stdout purity are asserted.
- [ ] Metadata, manifest, locking, stale cleanup, and partial outputs are asserted.
- [ ] GUI ordering, arguments, task tabs, Stop, and i18n are asserted.
- [ ] Default tests never download gated weights or the default SF2.
- [ ] Focused, regression, lint, lock, help, and full-suite commands are recorded.
