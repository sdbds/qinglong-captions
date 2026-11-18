# SheetSage2 music transcription

[SheetSage2](https://huggingface.co/m-a-p/SheetSage2) extracts vocal and instrumental melodies, chords, beats, key, and structure from a mix. Its lead sheet is not a note-for-note transcription of every instrument. Use [MuScriptor](muscriptor.en.md) for multi-instrument note events.

In **Tools → Music Transcription**, select **SheetSage2** and choose an audio file or directory. From PowerShell:

```powershell
.\2.7.music_transcription.ps1 -Backend sheetsage2 .\audio --device=cuda:0
.\2.7.music_transcription.ps1 -Backend sheetsage2 .\song.wav --melody-only
```

Run `uv run --script module/sheetsage2_tool.py --help` for the direct CLI. Omitting `-Backend` still selects MuScriptor.

- Install FFmpeg on `PATH`. `uv` creates a separate Python 3.11 environment without modifying the project `.venv`. The first run also downloads SheetSage2 and the MERT-v2-FullSong base model. Both models' weights use CC BY-NC 4.0.
- The default device is `auto` (CUDA when available, otherwise CPU). Use `cpu`, `cuda`, or `cuda:N` to select explicitly. Windows and Linux x64 use the PyTorch CUDA 12.8 wheel and require a compatible NVIDIA driver for GPU inference. Full songs can take longer on CPU.
- Each song has a directory under `sheetsage2_output` matching its relative input path, including the extension. Outputs include `score.abc`, `transcription.mid`, separate melody and chord MIDI, `events.json`, and timed `.lab` annotations. Set `--output-dir` to change the output root.
- `--melody-only` removes chord accompaniment from the ABC and combined MIDI but keeps the raw events and annotations. If the model cannot produce ABC, the tool preserves MIDI and annotations and reports an incomplete song. When a score already exists, the incomplete exports go in `partial/` without replacing the previous results.
- Complete results matching the input, model revision, and options are skipped. Use `--no-skip-completed` to run them again.

## Separate Stems First

Enable **Separate stems first** in **Tools → Music Transcription → SheetSage2** to run the existing ONNX separator, save WAV stems, then transcribe each non-silent primary stem. The same workflow is available by choosing SheetSage2 under **Audio Separator → All Stems to MIDI**. The MIDI, annotations, and any ABC score go to each song's `04_stem_midi/sheetsage2/<stem filename>/`.

Matching separated audio is reused. A stem without an ABC score can still export its current MIDI and annotations. When a stem becomes silent, or a new MIDI replaces a result with an ABC score, previous exports are kept in a `.previous-*` folder. SheetSage2 does not transcribe individual drum hits; use MuScriptor for multi-instrument notes. With the switch off, SheetSage2 detects harmony and beats directly from the original mix.

With the project's `vocal-midi` dependencies installed, run `python -m module.audio_separator ./audio --audio_files_only --sheetsage2_midi` for the same workflow.
