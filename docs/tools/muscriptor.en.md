# MuScriptor Audio To MIDI

The project pins [MuScriptor 0.4.0](https://github.com/muscriptor/muscriptor/releases/tag/v0.4.0). The application has changed, not the three official small, medium, and large transcription checkpoints. Cached weights can be reused; the default is still large.

Accept the Hugging Face model terms and run `hf auth login` before first use. Model weights are CC BY-NC 4.0 and carry additional input-rights requirements.

```powershell
.\2.7.music_transcription.ps1 .\audio --format midi --format jsonl
.\2.7.music_transcription.ps1 .\audio --format midi --format quantized-midi --format sheets
uv pip install --python .\.venv\Scripts\python.exe -r pyproject.toml --extra muscriptor-local
python -m module.muscriptor_tool.cli transcribe song.wav --format midi
python -m module.muscriptor_tool.cli transcribe song.wav --quantize --grid 4 --detect-tempo false --bpm 96
python -m module.muscriptor_tool.cli batch .\album --model large --device cuda:0 --format midi --format json --format jsonl
```

To try the official MuScriptor WebUI, run:

```powershell
.\2.7.1.muscriptor_webui.ps1
.\2.7.1.muscriptor_webui.ps1 -Model small -Device cuda:0 -BatchSize 4 -Port 8222
.\2.7.1.muscriptor_webui.ps1 -NoBrowser
```

The launcher installs `muscriptor-local` into the shared project `.venv`, loads the model with upstream's native optimized SDPA while retaining the official WebUI and HTTP app, and opens the browser after the health check succeeds. Pass `-NoBrowser` to disable automatic opening. `-Model` (alias: `-ModelSize`) accepts `small`, `medium`, or `large` and defaults to `large`; the device defaults to `auto`. `-BatchSize` defaults to `0`: startup reads the recorded model-memory profile and selects an even batch from total VRAM without rerunning BS1/BS2 calibration. OOM retries use a smaller batch; on Windows, detected growth in per-process shared GPU memory reduces later batches by two. After every completed or cancelled request, transient tensors and idle CUDA cache are released while model weights remain resident. CPU defaults to BS1. A positive integer bypasses profile selection but keeps the memory guards. BS1 keeps upstream's cross-chunk prelude forcing; larger batches turn it off for throughput, which can change notes at chunk boundaries. Avoid `uvx muscriptor serve`: `uvx` resolves a separate tool environment and may install another Torch/CUDA stack instead of reusing the project environment.

The CLI, official-WebUI launcher, batch transcription, and audio-separator paths share model-memory profiles in `config/muscriptor_batch_profiles.toml` and the same adaptive CUDA runtime. Before weights load, insufficient `auto` devices fall back to CPU and explicit CUDA choices are rejected. The CUDA allocator budget subtracts both the tiered reserve and memory already used outside PyTorch on the selected device. When either GUI page opens, the selected user's GPU total VRAM is also substituted into the BS1/BS2 and validated marginal-memory formula to choose an even batch; the recorded GPU and version (0.2.1) describe the original benchmark, not a new measurement for 0.4.0. With the 1 GiB reserve used through 16 GiB, the minimum total VRAM is approximately 1.90 GiB for small, 3.25 GiB for medium, and 10.28 GiB for large. Model/device changes recompute the recommendation until the user edits the batch control.

Each recording is transcribed once. Select multiple formats in the GUI or repeat `--format` with the `batch` subcommand. Use `batch song.wav` for multiple formats from a single recording; `transcribe` selects one format. Batch output defaults to `muscriptor_output` and preserves input-relative paths:

| Format | Per-recording file | Contents |
| --- | --- | --- |
| `midi` (default) | `<name>.mid` | Performance timing, with a tempo map when the detected beats do not fit a constant tempo. |
| `quantized-midi` | `<name>_quantized.mid` | Notes snapped to the beat grid for editing. `--grid 4` means four subdivisions per beat. |
| `sheets` | `<name>_sheets.zip` | `score.mid`, `score.musicxml`, and PDFs of the full score and parts, engraved from the quantized MIDI. |
| `json` / `jsonl` | `events.json` / `events.jsonl` | Original note events in seconds. Event-only output without preview does not run the beat tracker. |

In the GUI, expand **MIDI timing** to follow the beats, use a fixed tempo, or enter a manual BPM and steps per beat. Manual mode works without the beat tracker. A preview plays the performance MIDI when selected, or the quantized version when only quantized MIDI or sheets are requested.

The separate Beat This! `final0` checkpoint downloads on first use. The default `--detect-tempo best-effort` falls back to a placeholder 120 BPM if the beats cannot be detected. Notes are left unsnapped when there is no usable grid; check the warnings in `metadata.json`. Use `true` to require successful detection, or `false` to skip it; for offline quantization supply `--bpm` and `--grid`. Use `--tempo-map` to follow the tracked beats or `--fixed-tempo` to write one tempo. The separator skips per-stem beat detection so its tracks keep a common timeline.

Only `sheets` needs **MuseScore 4 or newer**. Set `MUSCRIPTOR_MUSESCORE` to its executable if it is not on PATH. The batch checks for it before loading model weights. Per-recording `metadata.json` records outputs, warnings and detected instruments; `manifest.json` summarizes the run. Changing formats or timing options invalidates the previous completion signature.

Changing formats only retires old artifacts recorded in that input's `metadata.json`. If a requested name is already in use by an untracked file, the item fails without replacing it. Move the file or choose another `--output-dir`.

Preview audio requires FluidSynth and the official `MuseScore_General.sf2` SoundFont. MIDI, JSON, and JSONL remain available when FluidSynth is missing. Events contain onset, offset, pitch, and instrument; original velocity is not reconstructed.
