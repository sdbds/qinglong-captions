# Configuration Guide

The project has two configuration paths: GUI settings and the `Configuration` block at the top of each PowerShell wrapper. Both eventually pass values to the same Python CLIs. Use the GUI for daily work; use the wrappers when you need a repeatable batch job.

## Configuration files

| File | Purpose |
| --- | --- |
| `config/config.toml` | Provider, generation, and tool defaults |
| `config/model.toml` | Provider routes, model IDs, and runtime defaults |
| `config/general.toml` | Shared paths and general defaults |
| `config/onnx.toml` | ONNX Runtime, execution providers, and caches |
| `config/task_tabs.toml` | GUI task-tab runtime and concurrency settings |
| `config/prompts.toml` | Caption prompts and templates |
| `config/env_vars.json` | GUI-managed environment state; plaintext, never commit |

Relative paths assume the repository root as the working directory. The GUI changes to that directory before starting jobs.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `HF_HOME` | Hugging Face model cache, default `huggingface` |
| `HF_ENDPOINT` | Hugging Face mirror or endpoint |
| `HF_TOKEN` | Gated/private model access |
| `CUDA_VISIBLE_DEVICES` | Limit visible GPUs, for example `0` or `0,1` |
| `UV_INDEX_URL` | Primary uv package index |
| `UV_EXTRA_INDEX_URL` | Additional PyTorch / ONNX indexes |
| `UV_CACHE_DIR` | uv download cache |
| `HTTP_PROXY` / `HTTPS_PROXY` | Network proxy |

`config/env_vars.json` is not encrypted. Revoke and regenerate a token if it is exposed. Prefer injecting secrets through the current shell or a controlled credential store instead of writing them to a script or shared config file.

## Provider settings

Cloud Providers are configured on the GUI `Caption` page. OpenAI-compatible services use `openai_base_url`, `openai_model_name`, and `openai_api_key`; a local server may accept a placeholder key. See [the OpenAI-compatible guide](openai_compatible.md) for server examples.

For local OCR, VLM, and ALM routes, the GUI reads `config/model.toml` and installs the matching `uv extra`. Do not install every optional profile in one environment because OCR, CUDA, and Transformers combinations can conflict.

### OvisOCR2

The `ovis_ocr2` route accepts images and PDFs. Its default Direct Transformers runtime loads `ATH-MaaS/OvisOCR2`:

```powershell
uv sync --extra ovis-ocr2
```

Defaults live in `[ovis_ocr2]` in `config/model.toml`. With `visual_region_mode = "crop"`, matching `images/bbox_L_T_R_B.jpg` regions are cropped into the output directory and their renderable tags are retained. Set it to `"drop"` to remove only those tags. A blank `prompt` uses the official model prompt.

The same route can call an external OpenAI-compatible vLLM server. vLLM does not natively support Windows, so run it on Linux or WSL:

```bash
pip install "vllm==0.22.1" pillow
vllm serve ATH-MaaS/OvisOCR2 --gdn-prefill-backend triton --gpu-memory-utilization 0.8
```

Configure the shared `openai_base_url`, `openai_model_name = ATH-MaaS/OvisOCR2`, and API key afterward. Both runtimes use the same page failure handling, repeated-tail cleanup, bbox processing, and Markdown persistence; generated text is not expected to be byte-identical across inference engines.

## Lance and translation tags

- `data_storage_version` in `lanceImport.ps1` controls newly created dataset format; the wrapper defaults to `2.2`.
- Confirm the intended tag/version before running `lanceExport.ps1`.
- Translation creates `raw.import.*`, `norm.docling.*`, and `tr.<model>.<lang>.*` versions and writes language-suffixed Markdown by default.

## Dependency profiles

| Profile | Main use |
| --- | --- |
| `video-split` | Video scene detection |
| `wdtagger` | WDTagger / CL Tagger |
| `image-align` | Image preprocessing and feature alignment |
| `reward-model` | Image quality scoring |
| `translate` | Text normalization and translation |
| `vocal-midi` | ONNX audio separation and MIDI helpers |
| `psdexport` | PSD layer reading and export |
| `musvit-onnx` | MuSViT ONNX sheet-music scanning |
| `see-through` | Image2PSD / See-through |
| `onnx-base` | ONNX Runtime GPU base dependencies |

Daily runs resolve profiles directly from `pyproject.toml`; a checked-in `uv.lock` is not required by the wrappers.
