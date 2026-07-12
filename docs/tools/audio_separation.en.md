# Audio Separation

The ONNX separator produces six stems and can optionally run harmony separation or vocal-to-MIDI helpers.

```powershell
.\2.5.audio_separator.ps1
python -m module.audio_separator --help
```

The profile is `vocal-midi`. Inputs may be a file, directory, or Lance dataset; outputs support WAV, FLAC, and MP3. `segment_size`, `overlap`, and `batch_size` trade speed, boundary quality, and memory. Disable harmony and MIDI helpers while diagnosing the base six-stem path.
