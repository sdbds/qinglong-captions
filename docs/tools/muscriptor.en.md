# MuScriptor Audio To MIDI

The project pins `muscriptor==0.2.1` and exports MIDI, JSON, or JSONL events from music audio. It accepts the official small, medium, and large models only and defaults to large.

Accept the Hugging Face model terms and run `hf auth login` before first use. Model weights are CC BY-NC 4.0 and carry additional input-rights requirements.

```powershell
.\2.7.music_transcription.ps1 .\audio --format midi --format jsonl
uv pip install --python .\.venv\Scripts\python.exe -r pyproject.toml --extra muscriptor-local
python -m module.muscriptor_tool.cli transcribe song.wav --format midi
python -m module.muscriptor_tool.cli batch .\album --model large --device cuda:0 --format midi --format json --format jsonl
```

To try the official MuScriptor WebUI, run:

```powershell
.\2.7.1.muscriptor_webui.ps1
.\2.7.1.muscriptor_webui.ps1 -Model small -Device cuda:0 -BatchSize 4 -Port 8222
.\2.7.1.muscriptor_webui.ps1 -NoBrowser
```

The launcher installs `muscriptor-local` into the shared project `.venv`, loads the model through the project's optimized SDPA runtime while retaining the official WebUI and HTTP app, and opens the browser after the health check succeeds. Pass `-NoBrowser` to disable automatic opening. `-Model` (alias: `-ModelSize`) accepts `small`, `medium`, or `large` and defaults to `large`; the device defaults to `auto`. `-BatchSize` defaults to `0`: startup reads the recorded model-memory profile and selects an even batch from total VRAM without rerunning BS1/BS2 calibration. OOM retries use a smaller batch; on Windows, detected growth in per-process shared GPU memory reduces later batches by two. After every completed or cancelled request, transient tensors and idle CUDA cache are released while model weights remain resident. CPU uses BS1. A positive integer bypasses profile selection but keeps the memory guards. Values above 1 improve throughput but delay browser events for later 5-second chunks in the same batch. Avoid `uvx muscriptor serve`: `uvx` resolves a separate tool environment and may install another Torch/CUDA stack instead of reusing the project environment.

The CLI, official-WebUI launcher, batch transcription, and audio-separator paths share model-memory profiles in `config/muscriptor_batch_profiles.toml` and the same adaptive CUDA runtime. Before weights load, insufficient `auto` devices fall back to CPU and explicit CUDA choices are rejected. The CUDA allocator budget subtracts both the tiered reserve and memory already used outside PyTorch on the selected device. When either GUI page opens, the selected user's GPU total VRAM is also substituted into the BS1/BS2 and validated marginal-memory formula to choose an even batch; the recorded GPU only documents measurement provenance. With the 1 GiB reserve used through 16 GiB, the minimum total VRAM is approximately 1.90 GiB for small, 3.25 GiB for medium, and 10.28 GiB for large. Model/device changes recompute the recommendation until the user edits the batch control.

Batch output is rooted at `muscriptor_output` by default and contains a source-stem MIDI file (`<source-stem>.mid`), event files, metadata, and a manifest. Each `metadata.json` distinguishes manually requested `instruments` from the model's actual `detected_instruments`. Completion signatures, atomic writes, and output locks support interrupted-job recovery.

Preview audio requires FluidSynth and the official `MuseScore_General.sf2` SoundFont. MIDI, JSON, and JSONL remain available when FluidSynth is missing. Events contain onset, offset, pitch, and instrument; original velocity is not reconstructed.
