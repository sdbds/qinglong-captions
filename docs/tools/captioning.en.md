# Captioning And Multimodal Descriptions

Caption is the main project workflow. It reads a media directory or Lance dataset, selects a cloud API, OpenAI-compatible server, local OCR, local VLM, or local ALM, and writes captions or structured descriptions.

```powershell
.\4.captioner.ps1
python -m module.captioner --help
```

Recommended order: split videos, add tags when useful, validate the Provider and prompt on a small batch, run the full caption job, then export the intended dataset version.

Pass a `.lance` directory to open it directly. For a parent directory, Caption opens `dataset.lance` or the only available Lance dataset. It does not guess between multiple alternatives or silently rebuild a snapshot. Only directories with no dataset are imported automatically. Import new media explicitly into a new dataset. Missing originals or a mismatch with a stored SHA-256 stop the batch before inference.

Segmented captions may include cues from a manually supplied SRT without replacing that file. The generated sidecar is tracked separately and is not used as input on reruns. OCR artifacts live under `.qinglong-ocr-artifacts` and are excluded from discovery. Old artifact directories are retained so links in older captions stay valid.

The summary counts persisted, policy-skipped and failed items. Exit code `0` means no failures (including policy-only skips), `3` means some items failed but others were saved, and other nonzero codes indicate failure. Failed or skipped rows are not republished over local sidecars.

Important controls include `dataset_path`, `pair_dir`, `mode`, retry/wait settings, segment length, and optional scene detection. Local routes install matching extras from `config/model.toml`. OpenAI-compatible configuration is documented in [its dedicated guide](../openai_compatible.md).

Set `vlm_image_model = "lfm_vl_local"` to caption a single image or an image pair with [LFM3-VL-2B-ONNX](https://huggingface.co/LiquidAI/LFM3-VL-2B-ONNX). The first run downloads only the selected graph pair and its weights, plus the tokenizer and processor settings. The default is `q4` on CPU and `q4f16` on CUDA; set `decoder_variant` in `config/model.toml` to choose another precision. Native ONNX inference uses greedy decoding with thinking disabled; this route does not accept video. To reproduce an older batch, select **LFM 2.5 VL 1.6B ONNX** and retain the original precision and prompt.

The standard local `deepseek_ocr` route uses built-in DeepSeek OCR 2 on Transformers **5.23.0**, tested on the same GPU against native 5.17.0 and legacy 4.46.3. Existing prompts and configured remote services are unchanged. `deepseek_ocr_legacy` with the isolated `deepseek-ocr-legacy` extra explicitly runs the former backend. See the [migration and reproducible GPU comparison](deepseek_ocr2.md).

API keys may still appear in child-process arguments and task logs. Do not share raw commands, process lists, or unredacted logs.
