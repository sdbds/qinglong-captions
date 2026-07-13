# Qinglong Captions

An AI media-processing and document-translation toolkit built around Lance datasets. It covers video and image captioning, OCR, tagging, translation, audio separation, and Image2PSD workflows.

Current version: `4.6.0` · [中文说明](README.zh-CN.md) · [Changelog](CHANGELOG.md)

## 4.6.0 Highlights

- Audio Separation can now send all six primary stems to MuScriptor MIDI from a secondary setting, with fixed instrument families for vocals, drums, bass, guitar, and piano plus automatic or manual choices for `other`.
- The same setting can produce a pure-MIDI preview or an original-left / synthesized-MIDI-right comparison in MP3 or WAV without running MuScriptor inference twice.
- The `muscriptor-local` profile includes SOCKS proxy support for the official SoundFont download, and failed preview setup now reports its shared root cause in the final task log.

## Start Here

| Goal | Recommended entrypoint | Result |
| --- | --- | --- |
| Batch video / image captions | GUI: `Import -> Split -> Tagger -> Caption -> Export` | Lance dataset plus caption files |
| Video scene splitting | GUI `Split` or `2.0.video_spliter.ps1` | Scene ranges, representative frames, and reports |
| Image content tagging | GUI `Tagger` or `3.tagger.ps1` | WDTagger / CL Tagger labels |
| OCR, VLM, and ALM captions | GUI `Caption` or `4.captioner.ps1` | Descriptions, OCR, transcripts, or captions |
| Other standalone tools | [Tool documentation index](docs/tools/README.en.md) | Preprocessing, scoring, audio, PSD, translation, and more |

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

## Complete a Job in Five Steps

### 1. Start the GUI

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

### 2. Import media

Open `Import`, select a video, image, or existing data directory, and choose an output Lance path. Keep the default data version and storage options for your first run. Lance is the shared dataset used by splitting, tagging, captioning, and export.

### 3. Split videos

Open `Split` and select the imported dataset. Start with the default detector. Raise the threshold or minimum scene length when cuts are too frequent; lower the threshold when cuts are missed. Skip this step for image-only jobs.

### 4. Generate tags and captions

Use `Tagger` with WDTagger or CL Tagger, then open `Caption`:

1. Choose a cloud API, OpenAI-compatible server, or local model.
2. Enter the model, API key, and Base URL; follow the GUI prompt for local dependencies.
3. Select the prompt and output field, then test a small sample.
4. Run the full job after checking the sample. Jobs continue when you switch pages.

Skip Tagger when you only need captions, or skip Caption when you only need tags.

### 5. Export results

Open `Export`, select the dataset version, caption suffix, and output format, then export media and sidecar captions. Test a small sample before overwriting existing files. See the [Import / Export guide](docs/tools/import_export.en.md).

## Core Workflow

The main workflow is `Import -> Split -> Tagger -> Caption -> Export`. Jobs continue in GUI task tabs when you change pages.

### Video splitting

Split detects scene boundaries in long videos and optionally extracts representative frames. It supports Content, Adaptive, Hash, Histogram, and Threshold detectors. See the [video splitting guide](docs/tools/video_split.en.md) for inputs and tuning.

### Image tagging

Tagger runs WDTagger or CL Tagger against image directories and Lance datasets. General, character, and concept thresholds are configurable; gated models require accepted Hugging Face terms and `HF_TOKEN`. See the [image tagging guide](docs/tools/tagging.en.md).

### Captioning and multimodal descriptions

Caption connects cloud APIs, OpenAI-compatible servers, local OCR, local VLMs, and local ALMs to image, video, audio, and document inputs. Provider routing, segmentation, retries, prompts, and outputs are covered in the [captioning guide](docs/tools/captioning.en.md).

Lance versions and media/caption export are documented in the [Import / Export guide](docs/tools/import_export.en.md).

## Other Tools

| Function | Guide | Entrypoint |
| --- | --- | --- |
| Watermark detection | [WaterDetect](docs/tools/watermark_detection.en.md) | `2.1.image_watermark_detect.ps1` |
| Image preprocessing | [Preprocess](docs/tools/image_preprocessing.en.md) | `2.2.preprocess_images.ps1` |
| Image quality scoring | [Reward Model](docs/tools/image_scoring.en.md) | `2.3.image_reward_model.ps1` |
| PSD layer export | [PSD Export](docs/tools/psd_export.en.md) | `2.4.psdexport.ps1` |
| Audio separation | [Audio Separation](docs/tools/audio_separation.en.md) | `2.5.audio_separator.ps1` |
| Image2PSD | [See-through](docs/tools/image2psd.en.md) | `2.6.image2psd.ps1` |
| Audio to MIDI | [MuScriptor](docs/tools/muscriptor.en.md) | `2.7.music_transcription.ps1` |
| Sheet-music embeddings | [MuSViT](docs/tools/sheet_music.en.md) | GUI Tools |
| Text and document translation | [Translation](docs/tools/text_translation.en.md) | `5.translate.ps1` |

MuScriptor installs through the `muscriptor-local` profile and supports the official `small`, `medium`, and `large` models. Runs that request only MIDI, JSON, or JSONL do not need an audio synthesizer. If preview is enabled, its runtime preflight runs before model inference and stops the batch when FluidSynth or the official SoundFont is unavailable; disable preview to export symbolic outputs without it. The profile includes SOCKS proxy support for first-use SoundFont downloads.

Optional MIDI-only or left-original/right-synthesized preview audio, in either MP3 or WAV format, requires the native [FluidSynth](https://github.com/FluidSynth/fluidsynth/releases) executable on `PATH`. Switching to WAV does not remove this requirement; MP3 additionally requires working `soundfile/libsndfile` MP3 encoding. On Windows, use an x64 build, add its extracted `bin` directory to `PATH`, restart the shell and GUI, and verify it with `fluidsynth --version`. The official `MuseScore_General.sf2` SoundFont is resolved automatically; no system or custom SoundFont is required.

See the [tool documentation index](docs/tools/README.en.md) for every entrypoint.

## Choose a Caption Model

- Cloud Providers are configured on the `Caption` page.
- OpenAI-compatible services use `openai_base_url` and `openai_model_name`; a local service may use a placeholder API key.
- Selecting a local OCR / VLM / ALM route lets the GUI install the matching `uv extra` and show memory guidance.
- Gated or private Hugging Face models require `HF_TOKEN`; do not commit tokens in PowerShell files or Markdown.
- See [docs/openai_compatible.md](docs/openai_compatible.md) for server examples.

The GUI stores environment settings in `config/env_vars.json`. It is plaintext local state and may contain tokens or proxy information. **Do not commit, upload, or share it.**

## Batch Scripts

PowerShell wrappers keep editable settings in a `Configuration` block near the top. Edit that block, then run from the repository root:

| Script | Purpose |
| --- | --- |
| `lanceImport.ps1` | Import a media/data directory into Lance |
| `2.0.video_spliter.ps1` | Scene detection and video splitting |
| `3.tagger.ps1` | WDTagger / CL Tagger |
| `4.captioner.ps1` | Batch caption generation |
| `lanceExport.ps1` | Export media and captions from Lance |

Example:

```powershell
.\lanceImport.ps1
.\4.captioner.ps1
.\lanceExport.ps1
.\5.translate.ps1
```

Wrappers install the selected profile incrementally. Avoid manually assembling a long list of `uv sync --extra` commands; see [docs/configuration.en.md](docs/configuration.en.md) when you need precise profile control.

## Data and security notes

- Lance is the main intermediate format for imports, captions, tags, and translations; select the intended version/tag before export.
- Translation writes language-suffixed Markdown and does not overwrite the source file by default.
- Logs may contain input paths, model names, and error details. Redact logs before sharing them.
- Some caption paths still pass API keys as command-line arguments. Do not publish full commands, process-list screenshots, or raw task logs containing credentials.
- `--cloud` has no built-in authentication and binds the GUI to all interfaces.

## Troubleshooting

1. **`uv` is not found**: restart the shell, confirm `uv --version`, and rerun the installer.
2. **GUI startup fails**: use `start_gui.ps1` and test `uv run gui/launch.py --help` from the repository root.
3. **Port conflict**: pass `--port` or use the URL printed after automatic port selection.
4. **Missing local dependencies**: reselect the route in `Caption` or `Tools` so the GUI can install the profile.
5. **Hugging Face 403**: accept the model terms and inject `HF_TOKEN` into the current runtime.
6. **Out of memory**: lower batch size, resolution, or concurrency; enable CPU/offload options where available.
7. **Translation or Lance errors**: validate the input path and run `--normalize_only` before full translation.

See [docs/troubleshooting.en.md](docs/troubleshooting.en.md) for the expanded checklist.

## Upstream and citations

- `Image2PSD / see-through` integrates [shitagaki-lab/see-through](https://github.com/shitagaki-lab/see-through). The local `module/see_through/` package is a workflow adaptation, not a file-for-file mirror.
- The `vocal-midi` path references [openvpi/GAME](https://github.com/openvpi/GAME). GAME does not currently publish an official BibTeX entry in its README, so the legacy repository-level citation is retained below; prefer a future upstream citation.

```bibtex
@article{lin2026seethrough,
  title={See-through: Single-image Layer Decomposition for Anime Characters},
  author={Lin, Jian and Li, Chengze and Qin, Haoyun and Chan, Kwun Wang and Jin, Yanghua and Liu, Hanyuan and Choy, Stephen Chun Wang and Liu, Xueting},
  journal={arXiv preprint arXiv:2602.03749},
  year={2026}
}
```

```bibtex
@InProceedings{ke2023repurposing,
  title={Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation},
  author={Bingxin Ke and Anton Obukhov and Shengyu Huang and Nando Metzger and Rodrigo Caye Daudt and Konrad Schindler},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

```bibtex
@software{openvpi_game,
  title={GAME: Generative Adaptive MIDI Extractor},
  author={{OpenVPI}},
  url={https://github.com/openvpi/GAME}
}
```

Follow each upstream repository's license and model terms. See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

The project uses the root [LICENSE](LICENSE). Third-party directories and downloaded models remain subject to their own licenses and access terms.
