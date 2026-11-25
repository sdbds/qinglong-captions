# Dettato multi-instrument MIDI

In **Tools → Music Transcription → Dettato**, choose the original mix or a directory of recordings. Each output MIDI contains a track per detected instrument and a drum track when hits are found. This is performance MIDI, not separated audio or a quantized score.

```powershell
.\2.7.music_transcription.ps1 -Backend dettato ".\my song.wav"
.\2.7.music_transcription.ps1 -Backend dettato .\album --device=cuda:0 --output-dir .\midi
```

Directories are scanned recursively for WAV, FLAC, MP3, M4A, OGG and AAC. For example, `album/disc1/song.flac` produces `album/dettato_output/disc1/song.flac/song.mid`. The adjacent `metadata.json` records instrument programs, note counts, estimated tuning and drum results. A sibling `song.wav` has its own output directory. Completed matching inputs are skipped; use `--no-skip-completed` to rerun them.

- The launcher installs the optional `dettato-local` dependencies into the shared project environment, reusing PyTorch. Omitting `-Backend` still selects MuScriptor. After installation, run `python -m module.dettato_tool --help` for the direct CLI.
- The first transcription downloads the [four official models](https://huggingface.co/collections/kobimusic/dettato); a batch loads them only once. With dependencies and weights cached, set `UV_OFFLINE=1` and `HF_HUB_OFFLINE=1` to run offline. For a separate local copy, point `DETTATO_WEIGHTS` at the parent of the four model directories. After updating weights, use `--no-skip-completed` to regenerate old results. The device defaults to automatic selection; the CLI also accepts `cpu`, `cuda`, `cuda:N` or `mps`.
- The drum stage requires a source sampled at 40 kHz or higher. Lower-rate recordings still produce pitched tracks; the log and JSON explain why drums were skipped. Prefer the original 44.1/48 kHz audio over upsampling.
- Some compressed formats need both `ffmpeg` and `ffprobe` on PATH. MIDI export itself needs no SoundFont or FluidSynth. In dense mixes, notes can end up on the wrong instrument; see the [upstream limitations](https://github.com/kobimusic/dettato#limits).
