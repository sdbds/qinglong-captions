# OCR Version Upgrades Implementation Plan

> Execution: native, in the current workspace, without subagents, as requested.
> Use test-driven development and verification-before-completion. Do not commit
> or disturb unrelated working-tree changes.

**Goal:** Support Unlimited-OCR-2, Logics-Parsing-V3, and all three LightOnOCR-3
sizes, retaining legacy models and Markdown outputs with separate layout JSON.

**Architecture:** Version-aware adapters keep the existing provider routes.
Reuse pinned upstream format parsers rather than approximate their document
grammars. Native Transformers remains available for all models; the existing
OpenAI-compatible routes for Logics and LightOn share their document processing
with the native routes. Unlimited's existing native-only boundary is unchanged.

**Tech Stack:** Python, Transformers, Pillow, PyMuPDF, TOML, pytest.

**Spec:** User-approved scope in this chat on 2026-10-27: complete new-version
capabilities, legacy compatibility, Markdown retained, bounding boxes in JSON,
no subagents.

## Constraints

- Preserve provider names and old model IDs. Explicit version overrides support
  local directories and server aliases that do not identify their generation.
- Do not strip DSL tokens before parsing or publish token-limited output as
  successful OCR. Preserve blank-page positions and raise on failed page renders.
- Keep old Unlimited's Transformers 4.57.1 environment separate from version 2's
  officially tested 4.52.1 environment; select extras from the chosen model.
- Use model-specific LightOn loaders, resolution, prompt, and sampling defaults.
- Preserve the existing Markdown writer and use JSON sidecars for layout/state.

## Review Focus

- A second document must not inherit the first document's Logics state.
- The last single-page window of a long PDF must still use the longdoc prompt.
- Special tokens, multiline tables, and continuation markers must survive decode.
- Version 1/2 legacy selections must not receive newer-only parameters/prompts.
- Empty, malformed, truncated, or missing-page outputs must not look successful.

## Tasks

- [x] Add failing tests for version selection, structured parsing, document state,
  grounding, truncation, and old-version routing.
- [x] Integrate pinned Unlimited 2 format conversion and Logics V3 prompts/state/
  parsing, with attribution and licenses. Test cross-page merges and coordinates.
- [x] Update Unlimited native inference to use the V2 structured return and page
  guard; keep V1 inference and chunking. Export Markdown, raw output, layout JSON.
- [x] Add Logics V3 stateful windows, Qwen3.5 loading, EOS handling, structured
  sidecars and cropped figures; retain the V2 HTML path.
- [x] Add LightOn 3 architecture selection, grounding/plain modes, 0-1000 boxes,
  model-specific preprocessing and sampling; retain LightOn 2 plain OCR.
- [x] Update model catalogs, defaults, prompts, dependency extras and launcher
  selection. Add focused configuration/launcher tests and usage documentation.
- [x] Run focused tests, then attempt the repository pytest suite, lint and diff checks.
  Report any unrelated baseline failures and the limits of GPU/model validation.

## Verification Record

- Final affected-area run: 657 passed, 12 existing `locale.getdefaultlocale`
  deprecation warnings. Covers provider routing, runtime transports/configuration,
  OCR generation/format contracts, output writers, model configuration, GUI model
  selection, dependency profiles and the PowerShell launcher.
- Core adapter/helper/new test Ruff checks passed. Scoped `git diff --check`
  passed (Git reported existing LF/CRLF normalization notices).
- Real native class imports passed for Qwen3.5 and LightOnOcr. Real processor
  image/chat-template smoke tests passed for Logics-Parsing-V3,
  LightOnOCR-3-0.8B, and LightOnOCR-3-1B using Transformers 5.17.0. Logics DSL
  round-tripped through its real tokenizer. Only processor metadata/tokenizers,
  not model weights, were downloaded.
- The full `pytest -q --tb=short` command was attempted and stopped at about 22%
  after prolonged unrelated auto-rig tests. No failure was shown in the completed
  portion, but this is NOT a completed/green full-suite run.
- Self-review only, respecting the user's prohibition on subagents. Review fixes
  were verified RED/GREEN: forced Unlimited per-page truncation, Logics state
  location/level validation, malformed LightOn grounding, runtime default/override
  precedence, per-page JSON image paths, and sibling-caption image-link rebasing.
- No full-weight GPU inference or OCR quality/VRAM benchmarking was performed.
  No commit, branch move, dependency replacement in the shared `.venv`, or change
  to unrelated pre-existing working-tree edits was made.

## Decisions

- Keep Unlimited's native-only runtime boundary. Its upstream SGLang deployment
  remains a separately managed option, not a new service installed by this task.
- Keep the launcher's existing shared-environment design. New/old Unlimited use
  separate mutually exclusive dependency profiles, not concurrently isolated
  virtual environments; the usage document states this explicitly.
- Do not attach the previous LightOn generation's benchmark score or unmeasured
  VRAM estimates to the new architectures.
