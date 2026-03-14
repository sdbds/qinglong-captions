# Intern VL Local Provider Design

Date: 2026-03-15
Status: Approved for implementation

## Goal

Add a new local VLM provider, `intern_vl_local`, for image understanding only.

This integration must:

- support `image/*` inputs only
- support single-image and paired-image workflows
- reuse the existing Provider V2 route and local runtime patterns
- ignore image generation and image editing features for now

This integration must not:

- handle `video/*`
- load or expose generation / editing routes
- require users to run an external OpenAI-compatible server for direct mode

## Chosen Approach

Use a pinned, vendored snapshot of the official InternVL-U VLM code in a VLM-only direct-loading path.

Why this approach:

- it preserves direct local inference
- it avoids loading the unused `generation_decoder` and `vae` weights
- it matches the user request to ignore generation and editing
- it avoids hidden runtime network fetches and self-mutating dependency logic

Rejected alternatives:

1. Load the full `InternVLUPipeline`
   - simpler to wire
   - wastes disk, memory, and startup time on unused generation components

2. Require an OpenAI-compatible server
   - small repo diff
   - not a true local direct provider

## External Dependencies

### Model weights

- Hugging Face model repo: `InternVL-U/InternVL-U`
- processor assets live under `processor/`
- VLM weights live under `vlm/`

### Upstream code

- GitHub repo: `https://github.com/OpenGVLab/InternVL-U.git`
- vendored source ref for this integration:
  - `e814cb053a6025df152abf7628074acef3bbebd2`

The upstream repo is not packaged as an installable Python distribution, so the design uses a pinned source snapshot in this repository instead of runtime fetching.

### Model revision

- Hugging Face model revision:
  - `5141a7aed9208605a69d57034280a30cd12eeb0d`

## Architecture

### Provider name and routing

- provider name: `intern_vl_local`
- route key: `vlm_image_model`
- match rule:
  - `args.vlm_image_model == "intern_vl_local"`
  - `mime.startswith("image")`

### Runtime modes

The provider keeps the existing `LocalVLMProvider` behavior:

- `direct`: default path, loads official InternVL-U VLM code and weights locally
- `openai`: optional reuse of the existing OpenAI-compatible local server backend

`video/*` is intentionally unsupported and must never route to this provider.

### Direct loading strategy

Direct mode loads only the understanding stack:

- vendored Python code snapshot from `third_party/internvl_u`
- `InternVLUProcessor` from the HF `processor/` subdirectory
- `InternVLUChatModel` from the HF `vlm/` subdirectory

Direct mode must not load:

- `generation_decoder/`
- `vae/`
- diffusion pipeline classes

### Vendored source snapshot

The provider imports from a pinned local snapshot under `third_party/internvl_u` instead of cloning code at runtime.

Design requirements:

- the vendored files must be copied from the pinned upstream commit
- the import path must not depend on runtime network access
- upstream sync must be an explicit maintenance action, not provider side effect

## Inference Flow

### Media handling

Reuse existing `LocalVLMProvider.prepare_media`.

That preserves:

- single-image inputs
- optional paired-image lookup through `pair_dir`

### Prompt and image assembly

For direct mode:

- single image: pass one `PIL.Image`
- paired image: pass `list[PIL.Image]`
- prompt remains the existing resolved provider prompt text

The provider must use InternVL-U's text-generation path only.

Equivalent behavior target:

- InternVL-U official text output path
- no image generation tokens
- plain text decode result returned as `CaptionResult.raw`

## Configuration

Add a new section in `config/model.toml`:

```toml
[intern_vl_local]
model_id = "InternVL-U/InternVL-U"
model_revision = "5141a7aed9208605a69d57034280a30cd12eeb0d"
model_subdir = "vlm"
processor_subdir = "processor"
max_new_tokens = 1024
do_sample = false
temperature = 0.2
top_p = 0.95
top_k = 0
repetition_penalty = 1.0
```

Notes:

- `dtype` should remain overrideable via existing runtime args / config patterns
- model and processor loads must pass the pinned `model_revision`
- Windows should default to non-flash-attn-friendly execution behavior

## Dependency Profile

Add a new optional dependency extra:

- extra name: `intern-vl-local`

This extra should include only what the VLM-only path needs, such as:

- `torch`
- `torchvision`
- `accelerate`
- `transformers`
- `timm`
- `einops`

This extra should not include:

- `diffusers`
- generation-only weights or helpers

## Files Expected To Change

- `module/providers/local_vlm/intern_vl_local.py`
- `module/providers/catalog.py`
- `module/providers/registry.py`
- `config/model.toml`
- `pyproject.toml`
- `gui/wizard/step4_caption.py`
- `4、run.ps1`
- `README.md`
- `third_party/internvl_u/`
- tests covering route, registry, extras, and provider behavior

## Error Handling

The provider must fail clearly in these cases:

- vendored InternVL-U source snapshot is missing or incomplete
- configured HF repo is missing `processor/`
- configured HF repo is missing `vlm/`
- CUDA OOM in direct mode

Expected behavior:

- no retry on OOM
- no video fallback
- no silent downgrade into generation/editing codepaths

## Testing Plan

Follow test-first implementation.

Required coverage:

1. Catalog / route tests
   - `intern_vl_local` appears in `vlm_image_model`

2. Registry tests
   - image routes to `intern_vl_local`
   - video does not route to `intern_vl_local`

3. GUI / script dependency tests
   - GUI extra map returns `intern-vl-local`
   - `4、run.ps1` mentions the provider and adds the extra

4. Provider unit tests
   - imports vendored source from `third_party/internvl_u`
   - passes `model_revision` to both processor and model loaders
   - loads processor from `processor/`
   - loads model from `vlm/`
   - handles single-image prompt assembly
   - handles paired-image prompt assembly
   - decodes text output into `CaptionResult`

## Out of Scope

- image generation
- image editing
- video support
- broader prompt redesign for InternVL-U specific reasoning behavior

## Risks

1. Official upstream code is repo-only, not a packaged wheel.
   - mitigation: vendor the minimal source snapshot at a pinned commit

2. Transformers compatibility may be narrower than other local VLM providers.
   - mitigation: keep `intern-vl-local` isolated in its own extra and pin the HF model revision

3. Upstream code may assume flash-attn in some environments.
   - mitigation: keep Windows on a conservative execution path and test with mocked direct loading
