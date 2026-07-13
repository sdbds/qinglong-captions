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
.\2.7.1.muscriptor_webui.ps1 -Model small -Device cuda:0 -Port 8222
```

The launcher installs `muscriptor-local` into the shared project `.venv` and starts the official `muscriptor serve` from that same environment. `-Model` (alias: `-ModelSize`) accepts `small`, `medium`, or `large` and defaults to `large`; the device defaults to `auto`. Avoid `uvx muscriptor serve`: `uvx` resolves a separate tool environment and may install another Torch/CUDA stack instead of reusing the project environment.

Batch output is rooted at `muscriptor_output` by default and contains a source-stem MIDI file (`<source-stem>.mid`), event files, metadata, and a manifest. Each `metadata.json` distinguishes manually requested `instruments` from the model's actual `detected_instruments`. Completion signatures, atomic writes, and output locks support interrupted-job recovery.

Preview audio requires FluidSynth and the official `MuseScore_General.sf2` SoundFont. MIDI, JSON, and JSONL remain available when FluidSynth is missing. Events contain onset, offset, pitch, and instrument; original velocity is not reconstructed.
