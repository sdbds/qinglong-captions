# Image2PSD / See-through

Image2PSD converts a character image or directory into editable layers through LayerDiff decomposition, Marigold depth estimation, and PSD export.

```powershell
.\2.6.image2psd.ps1
python -m module.see_through.cli --help
```

The profile is `see-through`. Seed and inference-step controls affect repeatability, speed, and detail. Model caches and intermediate files require substantial disk space; low-memory systems should use CPU/offload settings. Complex backgrounds and transparent effects often need manual layer cleanup.
