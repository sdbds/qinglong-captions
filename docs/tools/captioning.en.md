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

Set `vlm_image_model = "moondream"` to run [Moondream 3.1 9B A2B](https://huggingface.co/moondream/moondream3.1-9B-A2B) locally with Photon. No API key is required; the first run caches the 10.5 GB checkpoint and its tokenizer. For image OCR use `ocr_model = "moondream"` with `document_image = true`. Select **Moondream 3 Preview** to keep using the previous model and its Transformers loader.

Photon's precompiled Windows kernels require CUDA 12. For local 3.1 inference, the default GUI and PowerShell launchers prepare `.runtime_venvs/moondream-photon` with PyTorch 2.11 / CUDA 12.8 instead of downgrading the shared environment or GPU driver. Use that interpreter for manual 3.1 runs as well. Preview keeps its original dependency stack. Caption, query and image OCR through an OpenAI-compatible server do not install Photon; point and detect still require local inference.

With local Photon, images, prompts and outputs stay on the machine. Photon attempts to report the hostname, GPU information and aggregate usage by default; `HF_HUB_OFFLINE=1` controls Hub downloads, not [Photon telemetry](https://github.com/m87-labs/kestrel#telemetry). If you explicitly select a custom GUI environment, use a dedicated one: the launcher installs dependencies into that environment.

Set `vlm_image_model = "lfm_vl_local"` to caption a single image or an image pair with [LFM3-VL-2B-ONNX](https://huggingface.co/LiquidAI/LFM3-VL-2B-ONNX). The first run downloads only the selected graph pair and its weights, plus the tokenizer and processor settings. The default is `q4` on CPU and `q4f16` on CUDA; set `decoder_variant` in `config/model.toml` to choose another precision. Native ONNX inference uses greedy decoding with thinking disabled; this route does not accept video. To reproduce an older batch, select **LFM 2.5 VL 1.6B ONNX** and retain the original precision and prompt.

With the selected LFM3 precision cached, a new process can start offline. Install or update the `lfm-vl-local` extra first. Keep the tokenizer, chat template, processor settings, both graphs, and all weight shards they reference. Set `HF_HUB_OFFLINE=1` to disallow network access. An uncached precision or a forced download requires a connection. Existing `float32`, `base`, and `default` settings still select FP32, not `auto`.

The standard local `deepseek_ocr` route uses built-in DeepSeek OCR 2 on Transformers **5.23.0**, tested on the same GPU against native 5.17.0 and legacy 4.46.3. Existing prompts and configured remote services are unchanged. `deepseek_ocr_legacy` with the isolated `deepseek-ocr-legacy` extra explicitly runs the former backend. See the [migration and reproducible GPU comparison](deepseek_ocr2.md).

API keys may still appear in child-process arguments and task logs. Do not share raw commands, process lists, or unredacted logs.
