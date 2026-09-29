# Image Tagging

Tagger generates content labels for image directories or Lance datasets with WDTagger, CL Tagger, or PixAI Tagger. Tags can be used for filtering and as caption prompt context.

```powershell
.\3.tagger.ps1
python utils/wdtagger.py --help
```

WDTagger uses the `wdtagger` profile; CL Tagger v2 uses `wdtagger-cl-tagger-v2`. Gated repositories require accepted Hugging Face terms and `HF_TOKEN`. Tune `batch_size`, `thresh`, `general_threshold`, and `character_threshold` on a representative sample before processing the full dataset.

## PixAI Tagger v1.0 ONNX

Use the converted [bdsqlsz/pixai-tagger-v1.0-ONNX](https://huggingface.co/bdsqlsz/pixai-tagger-v1.0-ONNX) directly. Select this repository in the GUI, or run:

```powershell
python utils/wdtagger.py ./datasets --repo_id=bdsqlsz/pixai-tagger-v1.0-ONNX --batch_size=1
```

The first run downloads `model.onnx` and the official-format `config.json` to `wd14_tagger_model/bdsqlsz_pixai-tagger-v1.0-ONNX/`. Tag order, categories, and image size are read from `config.json`. There is no `model.json` dependency, model-file hash computation/validation, or pinned remote commit.

Existing local files are reused. After updating the repository, pass `--force_download` to refresh the model, configuration, and ONNX session. The old `pixai-labs/pixai-tagger-v1.0` ID remains a supported alias with its original cache directory; missing files and forced refreshes use the new ONNX repository.

The GUI and PowerShell launcher select `wdtagger-pixai`; routine inference does not load Transformers or remote model code. The default batch size is 1. Default thresholds come from the downloaded `config.json.category_best_threshold` and follow configuration refreshes. The current release uses general 0.17, character 0.27, style 0.15, copyright 0.24, meta 0.17, rating 0.41. The GUI enables "Model Default Thresholds" initially; turn it off for manual values. CLI overrides use `--general_threshold`, `--character_threshold`, `--style_threshold`, `--copyright_threshold`, `--meta_threshold`, and `--rating_threshold`. An explicit `--thresh` overrides category defaults, while explicit category values take precedence (including zero). Rating output remains controlled by `--use_rating_tags`.

The graph takes RGB float32 `[batch, 3, 1008, 1008]` and returns logits `[batch, 30877]`; the tagger applies sigmoid once. Preprocessing matches the official tensor resize, black padding, normalization, and white alpha compositing. Runtime `auto` selects CUDA/CPU with TF32 disabled by default. Explicit provider settings remain respected; TensorRT/FP16 requires separate validation.

For manual re-export, the converter remains available. It downloads upstream `main` by default, accepts an optional `--revision`, and exports ONNX without a `model.json` or hash manifest:

```powershell
uv pip install --python .venv/Scripts/python.exe -e ".[pixai-onnx-export]"
python -m utils.onnx_export pixai-tagger --download --device cuda --verify
```

Use `--model-dir` for source weights, `--output-path` for the ONNX destination, and `--overwrite` to replace an export. The old file is replaced only after export and validation succeed; failures and cancellation clean up this run's temporary file. Keep the official `config.json` alongside ONNX when transferring it. If Xet transfers stall behind a proxy, retry with `$env:HF_HUB_DISABLE_XET="1"`.
