# OCR Model Generations

The existing `unlimited_ocr`, `logics_ocr`, and `lighton_ocr` routes now default
to Unlimited-OCR-2, Logics-Parsing-V3, and LightOnOCR-3-1B. Select the other
checkpoints in `config/model.toml` or the GUI model configuration. The legacy
Unlimited-OCR, Logics-Parsing-v2, and LightOnOCR-2-1B choices remain available.

## Dependency Profiles

| Checkpoint | Native extra | Transformers |
| --- | --- | --- |
| Unlimited-OCR-2 | `unlimited-ocr2` | `4.52.1`, Hub `0.36.0`, tokenizers `0.21.4` |
| Unlimited-OCR | `unlimited-ocr` | `4.57.1` |
| Logics V3 / V2 | `logics-ocr` | `>=5.6.0,<6` |
| LightOnOCR 3 / 2 | `lighton-ocr` | `>=5.5.4,<6` |

The GUI and `4.captioner.ps1` choose the Unlimited extra from the selected model
and `model_version`. Extras are dependency profiles, not separate permanent
virtual environments: the existing launcher updates the shared `.venv` when
switching profiles. Do not install mutually exclusive extras together or switch
that environment underneath a running model. Use separate environments/servers
for concurrent incompatible model versions.

`model_version = "auto"` recognizes official IDs and supported local checkpoint
metadata. For opaque server aliases or renamed checkpoints, explicitly set
`"1"`/`"2"` for Unlimited or `"2"`/`"3"` for Logics and LightOn. For renamed
LightOn Qwen checkpoints, `architecture = "qwen3_5"` is also available.

## Unlimited OCR 2

- Uses the checkpoint's native SR-SWA implementation and structured `infer`
  return value. All PDF pages stay in one document request. The legacy
  `page_budget` applies only to V1 and does not split V2 cross-page context.
- Converts the native `<PAGE>/<BLOCK>` format to Markdown, merges continuation
  blocks/tables, and extracts figure crops for valid Markdown image links.
- `max_length = 0` selects 131072 for V2 or 32768 for V1. V2
  `max_new_tokens = 0` uses the remaining context budget. `max_page_tokens = 20000`
  is the upstream page guard; 0 disables its per-page cap.
- Token limits and missing pages produce errors, not a successful partial
  document. A page forced to end by its token guard is also rejected.
- Writes `result.md`, `result_raw.txt`, `result.json`, and referenced image crops
  under the document output directory. V1 retains its prior output conventions.

Native inference still has a finite configured context/output budget and retains
page images in memory inside the upstream model. SR-SWA does not make arbitrary
documents fit on every GPU. This adapter remains native-only; it does not install
or launch the separate customized SGLang serving runtime.

## Logics Parsing V3

V3 uses Qwen3.5, thinking disabled, and two-page non-overlapping windows by
default (`window_size = 2`). Each window carries only bounded structural state,
not the entire prior document. The last window remains in long-document mode
even when it contains only one page. State is reset for every new document.

Leave `logics_ocr_prompt` empty. V3 constructs the official single-page or
stateful DSL prompt internally; V2 still selects `QwenVL HTML`. Pixel settings
of 0 select version defaults: V3 65536-4194304; V2 3136-7372800.

Outputs include `result.md`, `<source-stem>.md`, `raw_output.dsl`, `hierarchy.md`
(multi-page inputs with an outline), figure crops, and `result.json`. JSON blocks
retain heading levels, page locations, continuation information, and caption
`refs` linking to `[page_index, x1, y1, x2, y2]` targets.

Both native and the existing OpenAI-compatible runtime use the same window and
output pipeline. The upstream serving reference uses vLLM 0.22.1 with Transformers
5.6.0; allow at least `window_size` images per request on the server.

## LightOnOCR 3

All three sizes are available. The 1B model uses `LightOnOcrForConditionalGeneration`
and a 1540-pixel longest edge. The 0.8B/4B models use Qwen3.5, thinking disabled,
and a 2048-pixel longest edge. `longest_edge = 0` selects these defaults.

- `mode = "auto"`: grounding for V3, plain OCR for legacy models.
- `mode = "plain"`: image-only transcription, without coordinate markers.
- `mode = "grounding"`: V3 only; sends the official `grounding` prompt.

V3 defaults to temperature 0.1; legacy native inference remains greedy. Explicit
temperature/runtime settings take precedence. The default V3 output budget is
8192 tokens per page. Reaching a token limit fails the document instead of
publishing an incomplete transcription.

Markdown contains readable content, not fake `![label](x1,y1,x2,y2)` image links.
Grounding boxes and continuation flags are stored separately in `result.json`,
with per-page JSON/Markdown/raw-text files in `page_NNNN/`. The document JSON's
`pages` array retains blank pages. Coordinates are normalized to 0-1000, page
indices are zero-based, and recorded image dimensions refer to the actual model
input image. Plain V3 mode has empty block arrays; legacy plain mode does not
add a layout JSON file.

The OpenAI-compatible path has the same grounding export. Upstream recommends a
separate vLLM 0.30.0 / Transformers 5.16.1 serving environment; its documented
5.17 incompatibility concerns that server combination, not native Transformers.

## Sources And Verification

- [Unlimited OCR 2 inference](https://github.com/baidu/Unlimited-OCR#inference)
- [Logics V3 reference](https://github.com/alibaba/Logics-Parsing)
- [LightOnOCR models and modes](https://github.com/lightonai/LightOnOCR)
- Vendored parser provenance and licenses: `module/providers/ocr/vendor/README.md`.

The regression tests use real format parsers, generated image fixtures and
controlled inference responses. They do not establish model accuracy, GPU memory
requirements, or successful inference with downloaded model weights.
