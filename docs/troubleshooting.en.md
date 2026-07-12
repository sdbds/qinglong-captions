# Troubleshooting

Run commands from the repository root. Redact API keys, paths, and personal information before sharing logs.

## Installation

### `uv` is not found

1. Restart PowerShell or the terminal so the installer can update PATH.
2. Run `uv --version`.
3. If it is still unavailable, rerun `1.install-uv-qinglong.ps1` and check whether a proxy or execution policy blocked the installer.

### `pwsh` is missing on Linux

The installer filename contains a space and the script uses Bash syntax. The bundled helper currently downloads x86_64 PowerShell only:

```bash
sudo bash "./0.install pwsh.sh"
```

On ARM64 or another architecture, skip the helper and install `pwsh` 7+ using your distribution or the upstream documentation.

Check it with:

```bash
pwsh -NoLogo -NoProfile -Command '$PSVersionTable.PSVersion'
```

### Python version mismatch

The project requires Python `>=3.10,<3.13`. Before deleting `.venv` or `venv`, confirm that it contains no local packages you need. Keep the system Python, project environment, and GUI PEP 723 runtime distinct and intentional.

## GUI

### The GUI does not start

Run:

```powershell
uv run gui/launch.py --help
uv run gui/launch.py --port 7899 --no-browser
```

If `--help` fails, investigate `uv`, network dependencies, or the PEP 723 runtime. If help succeeds but the page fails, check the port, browser, and NiceGUI output.

`start_gui.ps1` changes to the repository root and runs `uv run gui/launch.py`; do not guess the entrypoint from inside `gui/`.

### Port conflicts

The GUI tries subsequent ports and prints the actual URL. You can also choose one explicitly:

```powershell
uv run gui/launch.py --port 7900 --no-browser
```

### Remote access / `--cloud`

`--cloud` binds to `0.0.0.0` but does not add authentication. Use it only on a trusted network or behind a reverse proxy with authentication, TLS, and access control. Return to `127.0.0.1` after debugging.

## Models and dependencies

### A local route is missing dependencies

Do not install every optional profile. Select the route again in `Caption` or `Tools` and let the GUI install the matching profile. If one environment already contains conflicting OCR, CUDA, or Transformers packages, create a fresh project environment for validation.

### Hugging Face 403 / gated model

1. Sign in to Hugging Face and accept the model terms.
2. Set `HF_TOKEN` in the runtime that starts the job.
3. Confirm `HF_HOME` is writable and has enough disk space.

Never put `HF_TOKEN` in committed scripts, screenshots, task logs, or a shared copy of `config/env_vars.json`.

### Out of memory or slow model loading

Lower batch size, resolution, context length, or concurrency. Use CPU/offload/quantized paths where supported, close other GPU processes, and keep the model cache on a disk with enough space.

## Workflow failures

### Import or export fails

- Check that the input exists and that the wrapper runs from the repository root.
- `lanceExport.ps1` must point to a valid `.lance` dataset.
- Confirm that `version` matches the dataset tag you intend to export.
- Stop other writers before exporting a dataset.

### Translation fails

Run `5.translate.ps1` to install the `translate` profile and activate the project environment (`.venv` or `venv`) in the current shell before validating normalization only:

```powershell
python -m module.texttranslate ./datasets --normalize_only
```

Then check `chunk_offsets`, translation dependencies, and output permissions. For OpenAI-compatible translation, validate connectivity with `--runtime_backend openai`, `--openai_base_url`, and `--openai_model_name` before running the full job.

### Audio separation fails

Use a supported audio file, directory, or `.lance` dataset. The first model download needs network access and disk space. Output formats are `wav`, `flac`, and `mp3`. Disable harmony separation while validating the base six-stem path.

### Image2PSD fails

Confirm that the input contains readable images, the output directory is writable, and there is enough space for models and intermediates. Start with a small `limit_images` smoke test before increasing resolution or batch size.

## Logs and credentials

Task logs can contain absolute paths, model IDs, proxy addresses, and error context. Some caption paths also pass API keys as subprocess arguments. Redact logs and process-list screenshots before sharing them. Revoke and regenerate a token if it has appeared in a log or `config/env_vars.json`.
