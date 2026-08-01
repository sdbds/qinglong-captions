# Auto-Rig Target And Override Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans and test-driven-development task by task.

**Goal:** Produce the independently recomputable Stage A target/override identities and apply `rig_overrides.json` aliases and joint observations without creating a fingerprint cycle.

**Architecture:** `input_identity.py` snapshots the already-validated raw see-through files plus canvas/tag-registry schema and deliberately excludes override-derived canonical names. `overrides.py` first identifies and strictly parses the fixed override file, exposing only syntax-valid aliases; the input adapter then canonicalizes raw optimized tags through those aliases; finally the override target fingerprint and joint coordinates are validated against the completed target identity/canvas. Override raw bytes remain a separate total identity.

**Tech Stack:** Python 3.10 standard library, immutable dataclasses, Pillow/psd-tools through the existing input adapter, JCS, pytest.

## Global Constraints

- Follow design spec Revision 22 input/override sections.
- Missing `rig_overrides.json` has a fixed non-null identity; malformed files still have a byte-derived identity available to failure reporting.
- `target_input_fingerprint` never contains override bytes, aliases, native variants, auto-rig config, or algorithm versions.
- Tag aliases change canonical interpretation but not the raw target identity; the separate override identity invalidates A.
- Alias parsing happens before canonical tag validation, but target binding and joint bounds validation happen after the input contract is complete.
- Unknown/unused aliases, canonical collisions, path-unsafe raw tags, duplicate JSON keys, stale target fingerprints, and unapproved outside-canvas points fail explicitly.
- Observe RED before production edits.

---

### Task 1: Recomputable TargetInputIdentity v1

**Files:**
- Create: `module/auto_rig/input_identity.py`
- Create: `tests/test_auto_rig_input_identity.py`
- Modify: `module/auto_rig/tag_registry.py`

- [x] Add failing tests for deterministic file inventory, canvas/tag-registry binding, snapshot mutation, PSD shared payload deduplication, and no absolute-path leakage.
- [x] Implement immutable identity records and re-hash every declared file from disk.
- [x] Prove identity stability across equivalent contract iteration order and sensitivity to every target byte/schema field.

### Task 2: Total Override Identity And Strict Parser

**Files:**
- Create: `module/auto_rig/overrides.py`
- Create: `tests/test_auto_rig_overrides.py`

- [x] Add failing tests for missing/present identity, raw-whitespace invalidation, duplicate/unknown fields, malformed JSON with recoverable identity, joint-ID grammar, finite coordinates, `allow_outside`, and alias syntax.
- [x] Implement separate identify, parse-source, and post-target validation APIs; parser errors carry the already-computed identity.

### Task 3: Alias-Aware Input Adaptation Without Identity Cycles

**Files:**
- Modify: `module/auto_rig/contracts.py`
- Modify: `tests/test_auto_rig_contracts.py`

- [x] Add failing PNG and PSD alias fixtures proving raw payload lookup remains raw while canonical base/Part IDs use the alias target.
- [x] Add failing unused-alias, mapped-tag collision, split-policy, and path-safety cases.
- [x] Implement keyword-only alias input with exact raw-tag provenance and no fallback paths.
- [x] Add the end-to-end order fixture: parse source -> alias-aware input -> target identity -> override validation.

### Task 4: Public Surface And Regression Gate

**Files:**
- Modify: `module/auto_rig/__init__.py`
- Modify: `tests/test_auto_rig_public_api.py`
- Modify: this plan.

- [x] Export stable identity/override records and functions without eager heavy imports.
- [x] Run focused tests and Ruff.
- [x] Run official SDK-backed auto-rig, see-through, dependency/uv, compileall, and diff checks.
- [x] Record verification and commit before joint geometry implementation.

## Verification

- Focused target/override/contracts/public API: `49 passed` before final parser race hardening; override suite after hardening: `15 passed`.
- Official SDK-backed auto-rig suite with Cubism Native 5-r.5 Core and E0 renderer: `434 passed, 4 skipped` after final parser race hardening.
- See-through regression suite: `54 passed`.
- Dependency and uv regression suite: `193 passed`.
- Ruff, `compileall`, and `git diff --check`: passed.
