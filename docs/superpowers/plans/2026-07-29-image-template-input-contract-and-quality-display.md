# Image Template Input Contract and Quality Display Plan

**Goal:** Make sidecar-tag input an explicit per-template contract, keep the
Quality template image-only, and remove prompt-requested inline `**` masking
from Quality display and processed captions without damaging Markdown headings
or emphasis.

**Scope:** Apply the contract to the shared Kimi request path used by
`kimi_code` and `kimi_vl`. Preserve the current default behavior when no named
template is active so existing model-specific workflows do not silently change.

## Implementation

1. Add `use_existing_tags` to each image template in `config/prompts.toml`.
   Danbooru enables it; Quality and BBOX disable it.
2. Add shared template helpers in `module/providers/image_template.py`:
   resolve the input contract and format template-specific display text.
3. Pass the active template ID and its `use_existing_tags` value from both Kimi
   providers into `attempt_kimi_vl`.
4. In `attempt_kimi_vl`, skip both sidecar and `datasets/tags.json` reads when
   the contract is disabled.
5. For the Quality template only, remove plain or escaped asterisk masking when
   it occurs between two non-whitespace characters. Keep markers at Markdown
   boundaries, such as `**Heading:**`, unchanged. Apply this to console/layout
   display and the project `postprocess_caption_content` result while retaining
   the provider raw response for audit.
6. Keep Kimi's automatic `###Short`/`###Long` normalization only for the
   provider's default caption prompt. Explicit image templates and pair prompts
   own their output contracts and must pass through unchanged.

## Verification

1. Add unit tests for config values, defaults, inline-mask cleanup, and Markdown
   preservation.
2. Add Kimi request tests proving disabled contracts do not inject
   `Existing tags`, plus provider propagation tests.
3. Run the focused template and Kimi test suites.
4. Re-run only the Quality template on the original five-image sample and save
   raw and display-formatted outputs in a new evaluation directory.
5. Assert effective Quality prompts contain no Kimi-injected Short/Long
   instruction while Danbooru retains the Short/Long format declared in its own
   template prompt.
