# API Model Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Refresh cloud defaults and options and implement the MiniMax M3.1 contract.

**Architecture:** Retain existing providers and UI. Share only MiniMax request
policy needed by its two providers; reuse existing Kimi reasoning classification.

**Tech Stack:** Python, pytest, OpenAI-compatible clients, NiceGUI, TOML, PowerShell.

**Spec:** docs/superpowers/specs/2026-10-27-api-model-refresh-design.md

## Global Constraints

- Preserve dirty files, credentials, and explicit saved model IDs.
- No local model, scoring, dependency-stack, or unrelated provider refactor.
- No paid inference, automatic configuration migration, commit, or push.

## Review Focus

- Missing settings must use balanced defaults at every entry point.
- Legacy model selections must not be rewritten to new paid models.
- Invalid M3.1 effort must fail before network submission.
- Mixed reasoning/content deltas must retain answer text only.
- Truncated reasoning-only responses must not become successful empty captions.

## Task 1: MiniMax M3.1 Contract

Files: module/providers/cloud_vlm/minimax_api.py, minimax_code.py,
new minimax_config.py, config/model.toml, config/config.toml,
tests/test_minimax_provider.py.

- [x] Add failing payload tests for both providers: forced reasoning_split,
  medium/default and explicit effort, max_completion_tokens, no sampling knobs.
- [x] Add stream tests for mixed deltas and empty/truncated output; preserve M3
  and unknown legacy model behavior; test invalid effort before the network.
- [x] Run the tests red, implement a small shared request-policy helper and
  configuration forwarding, then run green.

## Task 2: Model Choices and Defaults

Files: module/captioner.py, gui/wizard/step4_caption.py,
module/providers/grok_build_headless.py, cloud_vlm/kimi_reasoning.py,
cloud_vlm/glm.py (if needed), 4.captioner.ps1 and affected tests.

- [x] Update catalog/default tests, including Ark and Kimi endpoints, and prove
  the current implementation fails the requested defaults.
- [x] Update defaults from the spec at all maintained entry points. Add new
  verified IDs; preserve explicit old/third-party model arguments.
- [x] Extend Kimi effort tests to new IDs and GUI toggle visibility; validate
  the GLM default request contract. Run focused tests green.

## Task 3: Verification and Delivery

- [x] Run related provider, GUI, parser, and configuration tests.
- [x] Attempt the full pytest suite; report unrelated failures without reverting
  user changes. Run diff/whitespace checks.
- [x] Review the scoped diff, document tested defaults and remaining live-API
  verification limits, and leave changes uncommitted.

## Verification Record

- User approved this plan and balanced defaults in chat.
- MiniMax contract: 34 RED failures, then 37 passing tests.
- Catalog/Kimi/Grok: 21 RED failures, then 409 passing related tests (1 skip).
- Final review found retryable token-exhaustion errors; 8 retry-wrapper tests
  failed RED before adding a typed terminal error. MiniMax + catalog now pass
  87 tests, including Ark's editable model choices and custom endpoint IDs.
- Standard PowerShell launcher passed AST syntax/default validation.
- Scoped Ruff checks and git diff whitespace checks passed.
- Final provider/GUI/parser/registry/handler/MiMo regression suite:
  **501 passed, 1 skipped**, one existing locale deprecation warning.
- Full offline suite was attempted serially and with two xdist workers. The
  serial run was interrupted after prolonged lack of visible progress. The
  isolated run reached 65% before being stopped due to costly unrelated
  auto-rig fixtures; it is **not a full-suite pass**. Timeout stack dumps point
  to `tests/test_auto_rig_spine_validator.py:51` (`spine_bundle`) and
  `tests/test_auto_rig_live2d_release_validator.py:263` (`release_fixture`).
  Logs remain in the ignored `.superpowers/sdd/2026-10-27-api-model-refresh/`.
- No credentials, personal run scripts, local models, or scoring settings were
  changed. No paid inference or model downloads, commits, or pushes were made.

## Pre-Push Review

The user subsequently authorized review, scoped commit, and push. A fresh review
found that the tracked launcher's deprecated Mistral argument could be shadowed
by the CLI default. A PowerShell argument-block integration test reproduced the
override; the launcher now sends `--mistral_model_path` while retaining its
existing configuration variable. The general legacy alias mechanism is unchanged.

Fresh pre-push verification: **527 passed, 1 skipped**, with only the existing
locale deprecation warning. This includes the launcher override regression,
nonempty reasoning/content deltas, transport failure after partial output, and
provider alias/catalog coverage. Scoped Ruff, PowerShell syntax, and staged
whitespace checks passed. The review found no remaining blocking introduced issue;
live account availability and the previously incomplete full-suite run remain
verification limits. Only the API changes are staged; unrelated working changes
are excluded, including the scoring section in `config/model.toml`.
