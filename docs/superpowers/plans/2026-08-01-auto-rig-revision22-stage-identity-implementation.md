# Auto-Rig Revision 22 Stage Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans and test-driven-development task by task.

**Goal:** Bring the already-implemented manifest graph and terminal finalizer up to the frozen Revision 22 identity, status, and release-payload contract before any A-E stage writer starts using it.

**Architecture:** Stage manifests carry the three independent semantic identities (`target_input_fingerprint`, native source set, native eligibility) in addition to config/override inputs. A-F and G use disjoint status enums. Graph validation understands degradation without confusing it with failure. G revalidates the complete A-E identity chain and publishes the full dual-runtime release or failure projection; callers never invent a G fingerprint.

**Tech Stack:** Python 3.10 standard library, frozen dataclasses, RFC 8785/JCS helpers, pytest.

## Global Constraints

- Follow design spec Revision 22, especially Resume/status and terminal sections.
- Keep StageManifest paths single-owner and retain exact public inventory scans.
- `target_input_fingerprint`, `native_variant_set_sha256`, `native_variant_eligibility_sha256`, and `rig_overrides_sha256` remain separate fields and separate invalidation identities.
- A-F success status is only `stage_validated` or `stage_validated_with_degradation`; G uses `completed`, `completed_with_degradation`, or `failed`.
- G success requires identical non-null identities across A-E. Early failure may have a null target/native eligibility but must carry `observed_input_set_sha256` in the public error record.
- `is_item_completed()` continues to accept only independently computable A-E fingerprints; G is validated from its payload and current graph.
- Observe RED before each production change.

---

### Task 1: StageManifest v3 Identity And Status Contract

**Files:**
- Modify: `module/auto_rig/manifests.py`
- Modify: `tests/test_auto_rig_manifests.py`

- [x] Add failing serialization/fingerprint tests for the three semantic identity fields and their independent mutation.
- [x] Add failing status-matrix tests for A-F versus G, including nullable identities only for G failure.
- [x] Bump the manifest schema and implement strict parse/build/round-trip validation without compatibility guessing.
- [x] Preserve output inventory, owner namespace, and commit-marker invariants.

### Task 2: Degradation-Aware Stage Graph

**Files:**
- Modify: `module/auto_rig/stage_graph.py`
- Modify: `tests/test_auto_rig_stage_graph.py`

- [x] Add failing graph tests proving both A-F validated statuses are reusable, failure is not a success manifest, and both G completed statuses require reusable C/D/E.
- [x] Add failing cross-stage identity tests for target/native/override drift.
- [x] Implement deterministic graph issues for identity mismatch without weakening recursive byte validation.

### Task 3: Full Revision 22 Terminal Projection

**Files:**
- Modify: `module/auto_rig/terminal.py`
- Modify: `tests/test_auto_rig_terminal.py`

- [x] Add failing success tests for native identities, config/override identity, motion runtime digest, frozen texture runtime contract, and completed-with-degradation propagation.
- [x] Add failing failure tests for nullable target identity, required observed-input digest, native identity fields, profile/tier fields, and G-only terminal ownership.
- [x] Implement exact success/failure payload validation and make resume recompute G semantics from current C/D/E artifacts.
- [x] Retain the Y1 contract: callers provide exactly A-E fingerprints and never a self-derived G fingerprint.

### Task 4: Regression Gate

**Files:**
- Modify: `module/auto_rig/__init__.py`
- Modify: `tests/test_auto_rig_public_api.py`
- Modify: this plan.

- [x] Export only stable new terminal/manifest records and constants.
- [x] Run focused foundation tests.
- [x] Run official Cubism SDK-backed auto-rig tests, see-through tests, dependency/uv tests, Ruff, compileall, and `git diff --check`.
- [x] Record verification and commit the slice before Stage A orchestration.

## Verification

- Official Cubism SDK `E:\CubismSdkForNative-5-r.5`: `403 passed, 4 skipped` for `pytest -k auto_rig`.
- Focused manifest/graph/terminal/public API suite: `71 passed`.
- See-through regression: `63 passed`.
- Dependency and uv regression: `182 passed`.
- Ruff, `compileall`, and `git diff --check`: passed.
