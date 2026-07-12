# Image Preprocessing And Alignment

This tool resizes, crops, handles transparent borders, and optionally aligns images to references before tagging, scoring, or dataset import.

```powershell
.\2.2.preprocess_images.ps1
python -m utils.preprocess_datasets --help
```

The profile is `image-align`. Controls include long/short edge limits, total pixels, background handling, recursion, workers, and matcher backend. Write results to a separate output directory and inspect several aspect ratios before a full run.
