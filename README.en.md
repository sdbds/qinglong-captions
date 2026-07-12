# Qinglong Captions

An AI media-processing and document-translation toolkit built around Lance datasets. It covers video and image captioning, OCR, tagging, translation, audio separation, and Image2PSD workflows.

Current version: `4.5.0` · [中文 README](README.md) · [Changelog](CHANGELOG.md)

## Start Here

| Goal | Recommended entrypoint | Result |
| --- | --- | --- |
| Batch video / image captions | GUI: `Import -> Split -> Tagger -> Caption -> Export` | Lance dataset plus caption files |
| OCR or local VLM / ALM | GUI `Caption` page | OCR, image descriptions, or audio transcripts |
| Text / document translation | GUI `Tools` page or `5.translate.ps1` | Language-suffixed Markdown such as `*_zh_cn.md` |
| Audio separation | GUI `Tools` page or `2.5.audio_separator.ps1` | Six stems, with optional harmony separation |
| Sheet-music embeddings | GUI `Tools` page or `module.sheet_music_musvit` | MuSViT ONNX embeddings and manifest |
| Single-image layered PSD | GUI `Tools` page or `2.6.image2psd.ps1` | Layered PSD and intermediate outputs |
| Batch automation | The matching `.ps1` wrapper or Python CLI | Scriptable offline workflows |

See the [GUI manual](gui/README.md), [configuration guide](docs/configuration.en.md), and [troubleshooting guide](docs/troubleshooting.en.md) for details.

## Requirements

- Windows or Linux (the bundled Linux PowerShell installer currently downloads x86_64 only; install `pwsh` 7+ manually on ARM64 or other architectures)
- Python `>=3.10,<3.13`; the installer creates Python 3.11 by default
- `uv`; the installer attempts to install it when missing
- PowerShell 5.1+ on Windows; `pwsh` on Linux
- A GPU is optional for the base workflow, but local VLM/OCR/translation and Image2PSD commonly require GPU memory and substantial disk space
- Models are downloaded on first use and stored under `huggingface/` by default

## Installation

### Windows

Run from the repository root:

```powershell
.\1.install-uv-qinglong.ps1
```

The base install uses the default dependencies in `pyproject.toml`. Selecting a local Provider in the GUI installs the required optional profile incrementally.

### Linux

The PowerShell installer filename contains a space, so keep the quotes:

```bash
chmod +x "./0.install pwsh.sh"
sudo bash "./0.install pwsh.sh"
pwsh ./1.install-uv-qinglong.ps1
```

The installer creates or reuses `.venv` or `venv` and resolves dependencies directly from `pyproject.toml`; this repository does not require a checked-in `uv.lock` for daily runs. The bundled helper downloads Linux x86_64 PowerShell; on ARM64 or another architecture, install `pwsh` 7+ through your distribution or the upstream documentation instead. The commands below use `.venv`; replace it with `venv` if that is the environment you kept.

### Verify the installation

```powershell
uv --version
uv run gui/launch.py --help
```

If a newly installed `uv` is not found, restart the shell and try again.

## Usage

### 1. GUI quick start (recommended)

```powershell
.\start_gui.ps1
```

The default browser URL is `http://127.0.0.1:7899`. If the port is busy, the GUI tries subsequent ports and prints the actual URL.

`start_gui.ps1` runs `uv run gui/launch.py`. The Python entrypoint declares a PEP 723 script environment, so this wrapper uses a GUI-scoped runtime rather than promising to reuse the project `.venv`.

Direct alternatives:

```powershell
uv run gui/launch.py --port 7899 --no-browser
uv run gui/launch.py --native --port 7899
```

- `--no-browser` disables automatic browser opening.
- `--native` uses a native window and requires `pywebview`.
- `--cloud` binds to `0.0.0.0` and should only be used on a trusted network or behind an authenticated gateway.

The GUI has no built-in login. Do not expose `--cloud` directly to the public internet.

### 2. Recommended workflow

1. **Setup**: inspect Python, PyTorch, CUDA, model caches, and environment variables.
2. **Import**: import an input directory into a Lance dataset and configure sidecars, tags, and import mode as needed.
3. **Split**: run scene detection on video directories and generate scene images; skip this step for non-video inputs.
4. **Tagger**: run WDTagger / CL Tagger. Accept gated model terms on Hugging Face and set `HF_TOKEN` first when required.
5. **Caption**: select a cloud API, OpenAI-compatible service, local OCR, VLM, or ALM route.
6. **Export**: export media and captions from the Lance dataset, optionally selecting a version tag or suffix.
7. **Tools**: run preprocessing, scoring, WaterDetect, audio separation, MuSViT sheet-music embeddings, translation, or Image2PSD as needed.

Tasks appear in GUI task tabs. Switching pages does not stop a background job; stop it from its task controls.

### 3. Provider and model configuration

- Cloud Providers are configured on the `Caption` page.
- OpenAI-compatible services use `openai_base_url` and `openai_model_name`; a local service may use a placeholder API key.
- Selecting a local OCR / VLM / ALM route lets the GUI install the matching `uv extra` and show memory guidance.
- Gated or private Hugging Face models require `HF_TOKEN`; do not commit tokens in PowerShell files or Markdown.
- See [docs/openai_compatible.md](docs/openai_compatible.md) for server examples.

The GUI stores environment settings in `config/env_vars.json`. It is plaintext local state and may contain tokens or proxy information. **Do not commit, upload, or share it.**

### 4. Script mode

PowerShell wrappers keep editable settings in a `Configuration` block near the top. Edit that block, then run from the repository root:

| Script | Purpose |
| --- | --- |
| `lanceImport.ps1` | Import a media/data directory into Lance |
| `2.0.video_spliter.ps1` | Scene detection and video splitting |
| `3.tagger.ps1` | WDTagger / CL Tagger |
| `4.captioner.ps1` | Batch caption generation |
| `lanceExport.ps1` | Export media and captions from Lance |
| `2.1.image_watermark_detect.ps1` | Watermark detection |
| `2.2.preprocess_images.ps1` | Resize, crop, and optional alignment |
| `2.3.image_reward_model.ps1` | Image quality scoring |
| `2.4.psdexport.ps1` | PSD layer export |
| `2.5.audio_separator.ps1` | ONNX audio separation |
| `2.6.image2psd.ps1` | See-through layered PSD generation |
| `5.translate.ps1` | Text normalization and translation |

Example:

```powershell
.\lanceImport.ps1
.\4.captioner.ps1
.\lanceExport.ps1
.\5.translate.ps1
```

Wrappers install the selected profile incrementally. Avoid manually assembling a long list of `uv sync --extra` commands; see [docs/configuration.en.md](docs/configuration.en.md) when you need precise profile control.

### 5. Python CLI

The installer activates `.venv` only inside its child PowerShell process. Before running Python CLIs directly, activate the project environment in the current shell:

```powershell
# Windows PowerShell
. .\.venv\Scripts\Activate.ps1

# Linux Bash
source .venv/bin/activate
```

Base entrypoints can show help immediately. Optional tools import model dependencies before argparse, so run the matching wrapper first to install its profile. WaterDetect is the exception and uses its own PEP 723 `uv run` environment:

| Entrypoint | Dependency preparation | Help command |
| --- | --- | --- |
| `module.lanceImport` | Base dependencies | `python -m module.lanceImport --help` |
| `module.lanceexport` | Base dependencies | `python -m module.lanceexport --help` |
| `module.captioner` | Base dependencies; install route extras for local models | `python -m module.captioner --help` |
| `module/waterdetect.py` | PEP 723 script dependencies | `uv run module/waterdetect.py --help` |
| `module.texttranslate` | Run `5.translate.ps1` (`translate`) first | `python -m module.texttranslate --help` |
| `utils.psd_dataset_pipeline` | Run `2.4.psdexport.ps1` (`psdexport`) first | `python -m utils.psd_dataset_pipeline --help` |
| `module.audio_separator` | Run `2.5.audio_separator.ps1` (`vocal-midi`) first | `python -m module.audio_separator --help` |
| `module.muscriptor_tool.cli` | Run `2.7.music_transcription.ps1` (`muscriptor-local`) first | `python -m module.muscriptor_tool.cli --help` |
| `module.sheet_music_musvit` | GUI Tools or install `musvit-onnx` first | `python -m module.sheet_music_musvit --help` |
| `module.see_through.cli` | Run `2.6.image2psd.ps1` (`see-through`) first | `python -m module.see_through.cli --help` |
| `utils.preprocess_datasets` | Run `2.2.preprocess_images.ps1` (`image-align`) first | `python -m utils.preprocess_datasets --help` |

Notable behavior:

- `module.lanceImport --data_storage_version` controls the format of newly created datasets.
- `module.lanceexport` supports `--version`, `--caption_suffix`, and `--caption_extension`.
- `module.texttranslate` supports `--normalize_only`, `--skip_normalize`, `--no_export`, and `--runtime_backend openai`.
- `module.audio_separator` outputs six stems by default and supports `--harmony_separation`.
- `module.see_through.cli` is intended for batch image processing and can consume substantial disk space.

## MuScriptor audio-to-MIDI

The integration pins `muscriptor==0.2.1` and accepts only the three official checkpoints:
[small](https://huggingface.co/MuScriptor/muscriptor-small),
[medium](https://huggingface.co/MuScriptor/muscriptor-medium), and
[large](https://huggingface.co/MuScriptor/muscriptor-large). The CLI, GUI, and runtime default to `large`. Local weights, URLs, and custom repositories are intentionally rejected.
Before first use, accept the terms on the Hugging Face model page and run `hf auth login`. The weights are CC BY-NC 4.0 and the model terms also require users to hold the necessary rights to input music; the code and SoundFont retain their separate licenses.

The PowerShell batch wrapper installs the isolated profile incrementally into the selected project Python:

```powershell
.\2.7.music_transcription.ps1 .\audio --format midi --format jsonl
```

For direct CLI use, install the same profile first:

```powershell
uv pip install --python .\.venv\Scripts\python.exe -r pyproject.toml --extra muscriptor-local
python -m module.muscriptor_tool.cli transcribe song.wav --format midi
python -m module.muscriptor_tool.cli batch .\album --model large --device cuda:0 --format midi --format json --format jsonl
python -m module.muscriptor_tool.cli list-instruments --format json
```

Each batch item can emit `transcription.mid`, `events.json`, `events.jsonl`, and `metadata.json`; the output root also receives `manifest.json`. Without `--output-dir`, the root is an input-local `muscriptor_output` directory. One input is inferred once even when MIDI, JSON, and JSONL are all selected. Atomic outputs, completion signatures, and an output-directory lock support recovery. Directory scans are recursive by default; disabling Skip Completed is the single explicit re-run control.

Events contain onset, offset, pitch, and instrument only. The model does not transcribe source velocity; MIDI uses the upstream fixed velocity. Simultaneous overlapping notes with the same instrument and pitch are not representable, and drums use onset-only events with the upstream minimum duration. Dense mixes, rare timbres, heavily processed audio, and some choral material can reduce accuracy; this integration does not invent missing data in post-processing.

Audio preview is optional and additional to symbolic output. `--preview-mode midi` renders synthesized MIDI only; `comparison` renders original audio on the left and synthesized MIDI on the right. MP3 is the default and is available only when the active `soundfile/libsndfile` runtime passes a real write/read probe; WAV is the explicit fallback. Preview requires FluidSynth on PATH and automatically uses MuScriptor's official `MuseScore_General.sf2` SoundFont. There is no system or custom SoundFont setting. The upstream WebUI, piano roll, and single-file demo page are outside this integration.

On Linux, install FluidSynth with the distribution package manager, for example `sudo apt install fluidsynth`. On Windows, install an official build from the [FluidSynth releases](https://github.com/FluidSynth/fluidsynth/releases) and add its executable directory to PATH. Verify either installation with `fluidsynth --version`. MIDI, JSON, and JSONL remain available when this check fails; disable preview, or choose WAV first to separate FluidSynth issues from MP3 codec support.

## Data and security notes

- Lance is the main intermediate format for imports, captions, tags, and translations; select the intended version/tag before export.
- Translation writes language-suffixed Markdown and does not overwrite the source file by default.
- Logs may contain input paths, model names, and error details. Redact logs before sharing them.
- Some caption paths still pass API keys as command-line arguments. Do not publish full commands, process-list screenshots, or raw task logs containing credentials.
- `--cloud` has no built-in authentication and binds the GUI to all interfaces.

## Repository map

| Path | Purpose |
| --- | --- |
| `gui/launch.py` | GUI Python entrypoint |
| `gui/README.md` | GUI pages, startup options, and troubleshooting |
| `module/providers/` | Provider V2 implementations |
| `module/caption_pipeline/` | Caption orchestration and Lance synchronization |
| `config/model.toml` | Provider and local-model defaults |
| `config/config.toml` | Runtime and model settings |
| `config/onnx.toml` | ONNX Runtime and cache settings |
| `config/task_tabs.toml` | GUI task-tab runtime settings |
| `config/env_vars.json` | Local GUI environment state; do not commit |
| `tests/` | Unit, integration, and compatibility tests |

## Troubleshooting

1. **`uv` is not found**: restart the shell, confirm `uv --version`, and rerun the installer.
2. **GUI startup fails**: use `start_gui.ps1` and test `uv run gui/launch.py --help` from the repository root.
3. **Port conflict**: pass `--port` or use the URL printed after automatic port selection.
4. **Missing local dependencies**: reselect the route in `Caption` or `Tools` so the GUI can install the profile.
5. **Hugging Face 403**: accept the model terms and inject `HF_TOKEN` into the current runtime.
6. **Out of memory**: lower batch size, resolution, or concurrency; enable CPU/offload options where available.
7. **Translation or Lance errors**: validate the input path and run `--normalize_only` before full translation.

See [docs/troubleshooting.en.md](docs/troubleshooting.en.md) for the expanded checklist.

## Development and verification

Test tools live in the `test` dependency group and are not part of the base install. From the repository root, install that group into the project environment first:

```powershell
# Windows PowerShell
uv pip install --python .\.venv\Scripts\python.exe --group test

# Linux Bash
uv pip install --python .venv/bin/python --group test
```

Then activate `.venv` in the current shell (`. .\.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` in Bash):

```shell
python -m pytest tests -q --strict-markers -m "not optional_runtime and not gpu and not network"
python -m ruff check module gui utils config tests --select F821,F823
```

Do not stage `config/env_vars.json`, model caches, datasets, logs, or local credentials.

## Upstream and citations

- [See-through](https://github.com/shitagaki-lab/see-through) for Image2PSD
- [GAME](https://github.com/openvpi/GAME) for the `vocal-midi` reference models
- Follow each upstream repository's license and model terms.
- See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

The project uses the root [LICENSE](LICENSE). Third-party directories and downloaded models remain subject to their own licenses and access terms.
