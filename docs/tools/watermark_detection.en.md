# Watermark Detection

WaterDetect assigns watermark probabilities to image directories for cleaning and filtering.

```powershell
.\2.1.image_watermark_detect.ps1
uv run module/waterdetect.py --help
```

The Python file owns a PEP 723 environment and should be run through `uv run`. Configure the model repository/cache, batch size, and threshold. Lower the batch size on memory errors and validate threshold changes with manual samples.
