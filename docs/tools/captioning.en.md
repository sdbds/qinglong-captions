# Captioning And Multimodal Descriptions

Caption is the main project workflow. It reads a media directory or Lance dataset, selects a cloud API, OpenAI-compatible server, local OCR, local VLM, or local ALM, and writes captions or structured descriptions.

```powershell
.\4.captioner.ps1
python -m module.captioner --help
```

Recommended order: split videos, add tags when useful, validate the Provider and prompt on a small batch, run the full caption job, then export the intended dataset version.

Important controls include `dataset_path`, `pair_dir`, `mode`, retry/wait settings, segment length, and optional scene detection. Local routes install matching extras from `config/model.toml`. OpenAI-compatible configuration is documented in [its dedicated guide](../openai_compatible.md).

API keys may still appear in child-process arguments and task logs. Do not share raw commands, process lists, or unredacted logs.
