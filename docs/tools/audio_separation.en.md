# Audio Separation

The ONNX separator produces six stems and can optionally run harmony separation, GAME vocal-to-MIDI, or MuScriptor transcription for every primary stem.

```powershell
.\2.5.audio_separator.ps1
python -m module.audio_separator --help
```

The base profile is `vocal-midi`; enabling MuScriptor stem MIDI also loads `muscriptor-local`. Inputs may be a file, directory, or Lance dataset; outputs support WAV, FLAC, and MP3. MuScriptor MIDI, optional preview audio, and metadata are written under each song's `04_stem_midi` directory. Preview modes are pure synthesized MIDI or an original-stem-left / synthesized-MIDI-right comparison, in MP3 or WAV.

Stem labels constrain only what they reliably identify: vocals use `voice`, drums use `drums`, while bass, guitar, and piano retain all acoustic/electric subtypes in their respective families. MuScriptor chooses the subtype. The `other` stem uses automatic detection unless its secondary setting supplies a manual instrument allowlist.

Each stem is transcribed once with one shared model load. Preview rendering reuses the same MIDI events and does not run inference again. Its metadata records requested and detected instruments, preview settings, and a cache signature. Existing primary stems are reused without rerunning separation when their model tag and audio format match the current settings.

Preview rendering requires FluidSynth on `PATH` and MuScriptor's official default `MuseScore_General.sf2` SoundFont. The runtime is checked once before model loading, and a failed check does not delete an existing MIDI file. Official MuScriptor weights require accepting the Hugging Face terms and running `hf auth login`. Disable harmony and both MIDI helpers while diagnosing the base six-stem path.
