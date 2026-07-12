# MuScriptor Audio To MIDI

The project pins `muscriptor==0.2.1` and exports MIDI, JSON, or JSONL events from music audio. It accepts the official small, medium, and large models only and defaults to large.

Accept the Hugging Face model terms and run `hf auth login` before first use. Model weights are CC BY-NC 4.0 and carry additional input-rights requirements.

```powershell
.\2.7.music_transcription.ps1 .\audio --format midi --format jsonl
uv pip install --python .\.venv\Scripts\python.exe -r pyproject.toml --extra muscriptor-local
python -m module.muscriptor_tool.cli transcribe song.wav --format midi
python -m module.muscriptor_tool.cli batch .\album --model large --device cuda:0 --format midi --format json --format jsonl
```

Batch output is rooted at `muscriptor_output` by default and contains transcription files, event files, metadata, and a manifest. Completion signatures, atomic writes, and output locks support interrupted-job recovery.

Preview audio requires FluidSynth and the official `MuseScore_General.sf2` SoundFont. MIDI, JSON, and JSONL remain available when FluidSynth is missing. Events contain onset, offset, pitch, and instrument; original velocity is not reconstructed.
