# Video Splitting

Split detects scene boundaries and optionally extracts representative frames for each scene. It is normally the first processing step before video tagging or captioning.

```powershell
.\2.0.video_spliter.ps1
python -m module.videospilter --help
```

The dependency profile is `video-split`. Supported detectors include Content, Adaptive, Hash, Histogram, and Threshold. Increase `threshold` or `min_scene_len` when cuts are too frequent; lower the threshold when cuts are missed. Flashing lights, music videos, and animated effects often need a longer minimum scene length.
