# Image Tagging

Tagger generates content labels for image directories or Lance datasets with WDTagger or CL Tagger. Tags can be used for filtering and as caption prompt context.

```powershell
.\3.tagger.ps1
python utils/wdtagger.py --help
```

WDTagger uses the `wdtagger` profile; CL Tagger v2 uses `wdtagger-cl-tagger-v2`. Gated repositories require accepted Hugging Face terms and `HF_TOKEN`. Tune `batch_size`, `thresh`, `general_threshold`, and `character_threshold` on a representative sample before processing the full dataset.
