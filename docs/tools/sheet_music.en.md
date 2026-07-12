# MuSViT Sheet-Music Embeddings

MuSViT extracts ONNX embeddings from scanned score images or PDF pages and writes a manifest. It does not transcribe audio to MIDI; use [MuScriptor](muscriptor.en.md) for that workflow.

```powershell
python -m module.sheet_music_musvit --help
```

The main entrypoint is GUI Tools and the profile is `musvit-onnx`. Inputs may be images, PDFs, or directories. Configure page-resize versus square-padding preprocessing, PDF DPI, recursion, skip-completed behavior, and overwrite behavior.
