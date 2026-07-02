# MuSViT ONNX Sheet Music Tool Design

## Background

The request is to add a scanned sheet-music model to the tools page, using MuSViT from Hugging Face, and to base the integration on an ONNX artifact. The implementation order matters:

1. First update the local ONNX export flow.
2. Download the gated Hugging Face weights after access is available.
3. Export MuSViT to ONNX.
4. Upload the ONNX artifact to a public ONNX repo.
5. Build the runtime and GUI around that ONNX contract, not around a live PyTorch/Transformers model.

This spec records the target shape and validation points. The implementation is now approved and uses the uploaded ONNX runtime repo as the default source.

## Model Facts

Source model: `PRAIG/musvit`

Verified from Hugging Face on 2026-07-02:

- The repo is public but gated: file access requires logging in and accepting the model conditions.
- The current repo revision is `42f1eab3cb3e6d02aed4bcee0d10bcc29a85e1c6`.
- Files listed by the Hub API: `.gitattributes`, `README.md`, `config.json`, `model.safetensors`.
- The model card describes MuSViT as a foundation vision encoder for sheet-music pages, pre-trained with MAE on 9.7M IMSLP sheet-music images.
- Model type is MAE / `vit_mae`; advertised architecture is `ViTMAEForPreTraining`.
- The card recommends loading with `ViTModel.from_pretrained("PRAIG/musvit", trust_remote_code=True)` when complete patch embeddings are needed.
- Page input example resizes image to `1024 x 1024`, converts to tensor, and returns `last_hidden_state` with shape `B x 4097 x 768`, including CLS token.
- For non-page/system-level crops, the card recommends either white padding to `1024 x 1024` or positional interpolation. Padding is recommended for zero-shot usage.
- License shown on the model card: `CC BY-NC-SA 4.0`.

Important consequence: this is an embedding/foundation encoder, not an end-to-end OMR parser. The first version must expose embeddings and simple batch extraction outputs. It must not promise MusicXML, MIDI, semantic score reconstruction, or note-level recognition unless a downstream head is added later.

## Runtime ONNX Repo Facts

Runtime repo: `bdsqlsz/musvit-onnx`

Verified from Hugging Face on 2026-07-02:

- Repo URL: `https://huggingface.co/bdsqlsz/musvit-onnx`
- Current Hub API revision: `be7c996d1ae8592ac8170e479a96cd2b81dbd532`
- Tags include `onnx`; page license is `cc-by-sa-4.0`.
- Files listed by the Hub API: `.gitattributes`, `README.md`, `model.onnx`, `preprocessor_config.json`.
- The model page states it is from `PRAIG/musvit`.
- There is currently no `model.json` metadata file in the uploaded ONNX repo.

Runtime consequence: GUI/backend must download `model.onnx` and `preprocessor_config.json` from `bdsqlsz/musvit-onnx`. It must not require `model.json`; any missing metadata must be derived from ONNX session input/output shapes and the preprocessor config.

## Three Questions Before Design

1. Is this a real problem or imaginary complexity?

Yes, but only if the target is "local sheet-music image embedding/extraction". MuSViT is useful as a visual representation backbone. It does not by itself solve full optical music recognition.

2. Is there a simpler way?

Yes. The simplest stable design is not to add a new provider route. This should be a tools-page utility like audio separator, vocal MIDI, and see-through: input path in, artifact outputs out. The input path may be an image file, PDF file, or directory. The backend should use ONNX Runtime through the existing `module.onnx_runtime` helpers.

3. What will this break?

The main risk is changing `utils/onnx_export.py`, which currently exports RoFormer audio separator models. If MuSViT logic is bolted into that file as another special case, it will make the exporter harder to reason about and risks breaking existing audio separator export behavior. The design therefore requires a small internal split inside the exporter while preserving current CLI behavior.

## Goals

1. Add a MuSViT export mode to the local ONNX export flow.
2. Keep the local `PRAIG/musvit` export path available for regenerating the artifact.
3. Use `bdsqlsz/musvit-onnx` as the default runtime/download repo.
4. Add a sheet-music scan tool under GUI step 6 tools page.
5. Run inference through the shared ONNX Runtime helpers.
6. Support batch image/PDF-page processing and save deterministic embedding outputs.
7. Keep the first version small, testable, and honest about output semantics.

## Non-Goals

- No full OMR pipeline in the first version.
- No MusicXML, MIDI, note/event extraction, staff segmentation, or symbol recognition.
- No Provider V2 route unless a later feature consumes embeddings inside caption generation.
- No remote inference provider.
- No default background auto-download on GUI start.
- No change to existing audio separator ONNX export behavior.
- No attempt to bypass Hugging Face gated access.
- No PDF text extraction or OCR-specific layout parsing. PDF support means page rasterization followed by the same visual embedding pipeline used for images.

## Recommended Approach

### Option A: Extend `utils/onnx_export.py` with a clean subcommand split

Add a first-level export target:

- `roformer`
- `musvit`

Keep current behavior compatible by making the default target `roformer` when legacy RoFormer args are used. The MuSViT path then owns only:

- HF model download/load
- preprocessing contract metadata
- PyTorch wrapper
- ONNX export
- ONNX validation
- metadata writing

Recommendation: choose this option. It respects the user's request to modify local `onnx_export.py`, while avoiding a pile of unrelated branches inside one parser path.

### Option B: Create a separate `utils/musvit_onnx_export.py`

This is technically cleaner, but it does not follow the request. It also creates another export entrypoint when the project already has an export utility.

### Option C: Skip ONNX and run Transformers directly

This is the lowest upfront work, but it violates the requested ONNX-base design and would pull gated remote-code model loading into the GUI runtime. Reject for this task.

## Architecture

### 1. Export Layer

Modify:

- `utils/onnx_export.py`

Target shape:

- Existing RoFormer functions remain intact.
- New MuSViT-specific functions are grouped together and do not share RoFormer config parsing.
- `main()` dispatches by explicit `--target` or subcommand.

Suggested CLI:

```powershell
.\.venv\Scripts\python.exe utils\onnx_export.py musvit `
  --repo-id PRAIG/musvit `
  --output-path huggingface\PRAIG_musvit_ONNX\model.onnx `
  --metadata-path huggingface\PRAIG_musvit_ONNX\model.json `
  --device cpu `
  --opset-version 20 `
  --verify
```

Fallback legacy CLI behavior:

```powershell
.\.venv\Scripts\python.exe utils\onnx_export.py --model-dir D:\BS-ROFO-SW-Fixed
```

must keep exporting RoFormer exactly as before.

### 2. MuSViT Export Wrapper

Do not export `ViTMAEForPreTraining` reconstruction logits for the first version. Export the encoder embedding path:

- load with `ViTModel.from_pretrained(repo_id, trust_remote_code=True)`
- input: `pixel_values`, float32, shape `B x 3 x 1024 x 1024`
- output: `last_hidden_state`, float32, shape `B x 4097 x 768`

Reason: the model card explicitly warns that `AutoModel` / MAE masking behavior can shuffle/mask patches unless special care is taken. `ViTModel` is the direct path for complete embeddings.

Wrapper contract:

```text
input_name: pixel_values
input_layout: batch, channels, height, width
input_dtype: float32
input_size: 1024 x 1024
output_name: last_hidden_state
output_layout: batch, tokens, hidden
output_tokens: 4097
patch_tokens: 4096
hidden_size: 768
contains_cls_token: true
patch_grid: 64 x 64
patch_size: 16
```

Dynamic axes:

- batch dimension can be dynamic.
- height/width should stay fixed at `1024 x 1024` for the first version.
- token dimension should stay fixed because the exported graph is built for the fixed page size.

### 3. Metadata

Write `model.json` next to `model.onnx`. This is not optional; the runtime should not infer model semantics from tensor shapes.

Suggested metadata:

```json
{
  "model_type": "musvit",
  "repo_id": "PRAIG/musvit",
  "revision": "42f1eab3cb3e6d02aed4bcee0d10bcc29a85e1c6",
  "onnx_path": "model.onnx",
  "input_name": "pixel_values",
  "output_name": "last_hidden_state",
  "input_layout": ["batch", "channels", "height", "width"],
  "output_layout": ["batch", "tokens", "hidden"],
  "image_size": [1024, 1024],
  "patch_size": 16,
  "patch_grid": [64, 64],
  "hidden_size": 768,
  "contains_cls_token": true,
  "preprocess": {
    "mode": "page_resize",
    "resize": [1024, 1024],
    "color": "RGB",
    "scale": "0_to_1",
    "normalize": null
  },
  "license": "CC BY-NC-SA 4.0",
  "export_notes": {
    "uses_vit_model_encoder": true,
    "does_not_export_mae_decoder": true,
    "fixed_page_size": true,
    "gated_source_model": true
  }
}
```

Implementation must confirm whether the upstream processor/config requires mean/std normalization. The model card examples use only `Resize` + `ToTensor`, so the first spec default is `scale 0_to_1` with no mean/std normalization. If `AutoImageProcessor` metadata contradicts that during implementation, metadata must reflect the real preprocessing.

### 4. Runtime Artifact Layout

Use the public ONNX repo for normal runtime downloads:

```text
repo_id: bdsqlsz/musvit-onnx
model: model.onnx
support: preprocessor_config.json
```

Use a repo-local cache/output directory consistent with current ONNX tools:

```text
huggingface/bdsqlsz_musvit-onnx/
  model.onnx
  preprocessor_config.json
```

For local regenerated artifacts, runtime loading should also support local path mode:

```text
huggingface/PRAIG_musvit_ONNX/
  model.onnx
  preprocessor_config.json
  model.json  # optional local export metadata, not required by runtime
```

### 5. Runtime Backend

Create a new backend module:

- `module/sheet_music_musvit.py`

Responsibilities:

- Load `model.onnx` through `module.onnx_runtime.load_single_model_bundle`.
- Download/load `preprocessor_config.json` as a required support file.
- Treat local `model.json` as optional metadata only.
- Accept image files, PDF files, and directories containing supported inputs.
- Expand PDF files into page images before MuSViT preprocessing, following the existing OCR-style PDF render path.
- Preprocess image/page files.
- Run ONNX Runtime in batches.
- Save embedding outputs and summary metadata.

Do not put image preprocessing into `module.onnx_runtime`. That layer is for artifact/session mechanics, not task semantics.

Do not put PDF rasterization into `module.onnx_runtime` either. PDF handling is input expansion for this task, not a model artifact/session concern.

PDF rendering should reuse the existing OCR utility path unless implementation proves it is unsuitable:

```text
utils.stream_util.pdf_to_images_high_quality(pdf_path, dpi=144, image_format="PNG")
```

This currently uses PyMuPDF/`fitz` and returns one PIL image per page. The MuSViT backend should consume those PIL pages without writing permanent intermediate page images unless debugging or cache options are explicitly added later.

Suggested public class:

```text
MuSViTOnnxEmbedder
  - repo_id or local_model_dir
  - runtime_config
  - batch_size
  - preprocess_page(image)
  - embed_file(path)
  - embed_pdf(path)
  - embed_directory(input_dir, output_dir)
```

### 6. Outputs

For each input image:

```text
<output_dir>/<relative_input_path>/
  embedding.npz
  metadata.json
```

For each PDF page:

```text
<output_dir>/<relative_pdf_path>/
  page_0001/
    embedding.npz
    metadata.json
  page_0002/
    embedding.npz
    metadata.json
```

`embedding.npz` keys:

- `last_hidden_state`: `4097 x 768`, float32
- `cls_embedding`: `768`, float32
- `patch_embeddings`: `4096 x 768`, float32

`metadata.json`:

- source path
- source type: `image` or `pdf_page`
- PDF page index and total page count for PDF-derived pages
- rendered page size for PDF-derived pages
- image size before preprocessing
- preprocessing mode
- ONNX model path
- preprocessor config path
- execution providers
- output tensor shapes
- elapsed time

Optional aggregate file:

```text
<output_dir>/manifest.json
```

with processed/skipped/failed counts and model metadata.

Do not write large JSON arrays for embeddings. Use `.npz` to keep output compact and loadable.

PDF page images are intermediate inputs. Do not persist them by default because the final artifact is the embedding, not a rendered document cache. If later debugging requires rendered pages, add an explicit `--save_rendered_pages` option rather than silently writing images.

### 7. CLI

The backend should have an independent CLI so GUI is only a caller.

Suggested:

- `module/sheet_music_musvit.py`

Arguments:

```text
input_path
--output_dir
--model_dir
--repo_id
--batch_size
--force_download
--overwrite
--skip_completed
--recursive
--preprocess_mode page_resize|pad_square
--pdf_dpi
--save_cls
--save_patches
--save_full
```

First-version defaults:

- `repo_id=bdsqlsz/musvit-onnx`
- `preprocess_mode=page_resize`
- `batch_size=1`
- `pdf_dpi=144`
- `skip_completed=true`
- save all three arrays in `.npz`

`pad_square` should be supported because the model card recommends padding for non-page/system-level images. It can be present as a user option but should not be the default for full pages.

PDF input behavior:

- Single `.pdf` input: render all pages and scan each page.
- Directory input: include supported image extensions and `.pdf` files.
- Recursive mode: recurse into directories for both images and PDFs.
- Batch size applies to rendered PDF pages and image files together after input expansion.
- Skipping/completion checks are per output page, not per PDF file.

### 8. GUI Tools Page

Affected files:

- `gui/wizard/step6_tools.py`
- `gui/utils/i18n.py`
- `gui/utils/process_runner.py`

New tab:

- key: `sheet_music`
- label: "Sheet Music" / "乐谱扫描"
- icon: `library_music` or `queue_music`

Controls:

- input path: image file, PDF file, or directory
- output directory
- model directory
- repo id or local ONNX path
- batch size
- preprocess mode: page resize / pad square
- PDF DPI
- skip completed
- overwrite
- recursive
- execution provider display from ONNX runtime config

Start action:

```text
module.sheet_music_musvit
```

GUI must not download or load the model during tab render. It should only collect parameters. Heavy work begins after the user clicks Start.

### 9. Process Runner

Add script registry entry:

```text
"module.sheet_music_musvit": ("./module/sheet_music_musvit.py", "musvit-onnx")
```

Add a new optional extra only if needed:

- `musvit-onnx`

Dependencies should be minimal because runtime is ONNX:

- `qinglong-captions[onnx-base]`
- `numpy`
- `pillow`
- `PyMuPDF`

`PyMuPDF` is needed only for PDF inputs and should be imported lazily. If it is missing and the user selects a PDF, fail with a direct message telling them to install/run the `musvit-onnx` extra.

Export-time dependencies need Transformers/Torch, but the local `.venv` already has the required stack. Do not force GUI runtime users to install Transformers just to run an already exported ONNX model unless implementation proves metadata download requires it.

### 10. Config

Add to `config/onnx.toml`:

```toml
[onnx_runtime.musvit]
```

Optional tool defaults can go in `config/model.toml`:

```toml
[musvit]
repo_id = "bdsqlsz/musvit-onnx"
model_dir = "huggingface"
batch_size = 1
preprocess_mode = "page_resize"
pdf_dpi = 144
skip_completed = true
recursive = true
```

Keep runtime provider settings in `onnx.toml`; keep model/tool semantic defaults in `model.toml`.

## Environment Plan

Use local `.venv` explicitly:

```powershell
.\.venv\Scripts\python.exe -m pip show transformers torch torchvision onnxruntime-gpu huggingface_hub safetensors
```

Current observed `.venv` state:

- Python `3.11.10`
- `transformers 4.57.1`
- `torch 2.12.1+cu130`
- `torchvision 0.27.1+cu130`
- `onnxruntime-gpu 1.26.0`
- `huggingface_hub 0.36.2`
- `safetensors 0.8.0`
- missing: `onnx`
- missing: `onnxscript`

Export requires at least `onnx` when `--verify` is used. Depending on the PyTorch export path chosen, `onnxscript` may also be needed. The first implementation should prefer the existing `torch.onnx.export(..., dynamo=False)` style used by the RoFormer export to minimize new dependencies. If export fails because the graph needs newer ONNX export tooling, add `onnx` / `onnxscript` as explicit export-time setup notes rather than burying the requirement.

Gated model access:

```powershell
.\.venv\Scripts\python.exe -m huggingface_hub.commands.huggingface_cli login
```

or use `HF_TOKEN` in the environment. Implementation must fail loud if access has not been granted:

```text
PRAIG/musvit is gated. Log in to Hugging Face and accept the model conditions before export.
```

## Error Handling

### Export Errors

- Gated model access missing: fail with a direct message and no retry loop.
- `ViTModel` load fails: include repo id, revision, and local cache path if available.
- ONNX export fails: include opset, device, input shape, and exporter path.
- ONNX checker missing package: tell user to install `onnx` into `.venv`.

### Runtime Errors

- Missing `model.onnx` or `preprocessor_config.json`: fail before scanning inputs.
- ONNX session input/output shape mismatch vs MuSViT expected contract: fail before batch processing.
- Unsupported image/PDF extension: skip with warning in directory mode, fail in single-file mode.
- PDF render dependency missing: fail before running model inference and name the missing dependency.
- Encrypted, corrupt, or zero-page PDF: fail that PDF with a clear message; directory mode should continue to other files if the batch runner supports per-file continuation.
- PDF render memory failure: fail with PDF path, page index if known, and requested DPI.
- Corrupt image: record per-file failure and continue if `continue_on_error` is later added; first version can fail the run unless batch semantics require continuation.
- Empty input directory: fail with a clear message.

## Testing Plan

### 1. Export Parser Tests

Add tests for `utils/onnx_export.py` without downloading real weights:

- legacy RoFormer args still resolve to RoFormer behavior
- `musvit` target parses repo id, output path, metadata path, device, opset
- `musvit` target does not call RoFormer YAML/checkpoint resolution

### 2. Export Metadata Tests

Mock a fake model/config and verify:

- metadata includes fixed `1024 x 1024`
- output shape fields match `4097 x 768`
- `contains_cls_token=true`
- license and gated-source flags are written

### 3. Runtime Preprocess Tests

Use small generated PIL images:

- RGB conversion
- page resize returns `1 x 3 x 1024 x 1024`, float32
- pad square preserves aspect inside white canvas
- invalid mode fails

Add PDF input expansion tests with mocked PDF rendering:

- single PDF path expands to `page_0001`, `page_0002`, ...
- rendered PIL pages feed the same preprocessing path as image files
- directory collection includes `.pdf` alongside image files
- skip/overwrite checks are per PDF page output
- PDF render dependency failure produces a direct error

### 4. Runtime ONNX Wiring Tests

Mock `load_single_model_bundle` and fake session:

- backend feeds `pixel_values`
- output arrays are saved as `.npz`
- `metadata.json` records providers and shapes
- PDF page metadata records source PDF path, page index, page count, and rendered page size
- `skip_completed` avoids rerun
- `overwrite` forces rerun

### 5. GUI Tests

Extend existing step 6 tool tests:

- `ToolsStep.TOOL_TABS` includes sheet music tab
- `_tool_action_for_tab("sheet_music")` maps to start label and callback
- `_start_sheet_music` builds expected CLI args
- PDF DPI control is included in GUI args
- invalid input path shows warning
- process runner receives `script_key="module.sheet_music_musvit"`

### 6. Config Tests

- `config/onnx.toml` has `[onnx_runtime.musvit]`
- optional `model.toml` section is loaded if added
- `process_runner.SCRIPT_REGISTRY` maps `module.sheet_music_musvit`

### 7. Slow Manual Test

After gated access is accepted:

```powershell
.\.venv\Scripts\python.exe utils\onnx_export.py musvit --repo-id PRAIG/musvit --output-path huggingface\PRAIG_musvit_ONNX\model.onnx --verify
.\.venv\Scripts\python.exe module\sheet_music_musvit.py path\to\score.png --output_dir output\musvit_test --repo_id bdsqlsz/musvit-onnx --model_dir huggingface
.\.venv\Scripts\python.exe module\sheet_music_musvit.py path\to\score.pdf --output_dir output\musvit_pdf_test --repo_id bdsqlsz/musvit-onnx --model_dir huggingface --pdf_dpi 144
```

Acceptance:

- ONNX file exists.
- `preprocessor_config.json` exists.
- ONNX Runtime session loads on CPU and CUDA if available.
- One test image produces `embedding.npz`.
- `last_hidden_state.shape == (4097, 768)` for a single page.
- One multi-page PDF produces one `embedding.npz` per rendered page, under `page_0001`, `page_0002`, ...

## Implementation Order

1. Refactor `utils/onnx_export.py` argument dispatch without changing RoFormer behavior.
2. Add MuSViT export mode with mocked tests first.
3. Add metadata writer for MuSViT ONNX artifact.
4. Run real gated download/export from `.venv`.
5. Add `module/sheet_music_musvit.py` backend using `bdsqlsz/musvit-onnx`.
6. Add PDF input expansion using the existing OCR-style PDF renderer.
7. Add focused backend tests with fake ONNX session.
8. Add `config/onnx.toml` musvit runtime section.
9. Add process runner script registry entry and optional extra if needed.
10. Add tools page GUI tab and i18n keys.
11. Run focused tests, then full relevant regression.

## Acceptance Criteria

- Existing RoFormer ONNX export still works with old args.
- `PRAIG/musvit` can still be exported to `huggingface/PRAIG_musvit_ONNX/model.onnx` from local `.venv` after gated access.
- Runtime can download and use `bdsqlsz/musvit-onnx` without gated source-model access.
- Runtime uses `preprocessor_config.json` and ONNX session metadata as the contract.
- Runtime uses ONNX Runtime, not live Transformers.
- Tool page starts a batch embedding job through the shared `ExecutionPanel`.
- A scanned sheet-music image produces `.npz` embeddings and per-file metadata.
- A scanned sheet-music PDF renders pages and produces `.npz` embeddings plus per-page metadata for each page.
- No GUI render path performs heavy model loading.
- Tests cover parser dispatch, metadata, runtime preprocessing, PDF input expansion, ONNX wiring, and GUI arg mapping.

## Risks

### 1. The model is not OMR

MuSViT is a representation model. If the real goal is "scan score and output notes/MIDI/MusicXML", this design is only the first backbone step. It will not satisfy that higher-level goal by itself.

### 2. Gated access can block implementation

The model files are gated. Export cannot proceed until the user accepts the terms and `.venv` has HF credentials or `HF_TOKEN`.

### 3. `utils/onnx_export.py` is already specialized

It currently contains a deep RoFormer export path. Adding MuSViT without dispatch separation will create a bad mixed-purpose script. The implementation must introduce a clean target boundary first.

### 4. ONNX export may expose unsupported ops

ViT encoder export is usually manageable, but the real graph must be tested. If opset 20 with `dynamo=False` fails, implementation may need opset changes or `onnxscript`.

### 5. Preprocessing ambiguity

The model card examples use `Resize` + `ToTensor`, but `AutoImageProcessor` may encode mean/std defaults. The implementation must verify this against the downloaded config and real processor before freezing metadata.

### 6. Storage size

Full patch embeddings are large:

```text
4097 * 768 * 4 bytes ~= 12 MB per page
```

Batch processing large score collections can produce substantial output. GUI should make the output directory explicit and avoid silently writing next to inputs unless requested.

### 7. PDF rendering cost

PDF support can multiply work by page count. Rendering a large PDF at high DPI before ONNX inference can consume significant CPU memory. Keep the default DPI conservative (`144`) and make it explicit in CLI/GUI. Do not render the whole document into permanent image files by default.

## Lower-Cost Alternative

If the user's real goal is searchable/clusterable sheet-music pages, this ONNX embedding tool is a good first step.

If the user's real goal is full OMR, the lower-cost path is to skip MuSViT as a standalone GUI tool and instead integrate an OMR model that directly outputs MusicXML/MIDI, using MuSViT only later as a backbone if fine-tuning is planned. Building a GUI around embeddings alone may be a detour for OMR.

## Handoff

Implementation is approved as of 2026-07-02. Use this document as the contract while adding the backend, GUI tool entry, process-runner registration, and focused tests.
