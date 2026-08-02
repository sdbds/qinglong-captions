# Auto-Rig Stage G Terminal Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing terminal primitives into a real Stage G success transaction that derives every release fact from the current validated C/D/E graph and publishes a trustworthy dual-format completion marker.

**Architecture:** Add a narrow `stage_g.py` integration boundary above `terminal.py`. It reloads the canonical `RigDocument`, D/E export reports, and current A-E manifests; validates their identity, report digests, exact artifact inventories, profile, format-plan, symbol, texture, and validator evidence; derives the terminal payload; then delegates the atomic G commit and resume semantics to the existing terminal finalizer. Caller-supplied format status or validator fingerprints are forbidden.

**Tech Stack:** Python 3.10, dataclasses, RFC 8785/JCS helpers, existing `StageGraphValidator`, pytest, official Cubism Core/SDK optional-runtime fixture.

## Global Constraints

- Follow `2026-07-31-auto-rig-from-see-through-layers-design.md` Revision 27 and its Revision 28 Stage E update.
- Formal success requires exactly `spine_4_2` and `live2d_moc3_v4_00`; structural-only Live2D output cannot enter G success.
- G is the sole writer of `rig/export_manifest.json`, `rig/error.json`, and `rig/cache/G/manifest.json`.
- A-E fingerprints supplied by the scheduler must contain exactly A, B, C, D, and E; G freshness is derived from its payload and current graph.
- D/E report artifact rows must match their owner manifests exactly after excluding the report itself; directory scans are never capability evidence.
- Canonical pages remain straight-alpha sRGB bytes, Spine uses `pma:false`, and both exporters must carry byte-identical C-owned PNG copies.
- The official Cubism SDK/Core and renderer binaries are release-test dependencies only and are never copied into the Python package or public item output.
- Production changes follow strict RED/GREEN TDD, use `apply_patch`, and preserve unrelated worktree changes.

---

### Task 1: Freeze report-derived release evidence

**Files:**
- Create: `module/auto_rig/stage_g.py`
- Create: `tests/test_auto_rig_stage_g.py`

**Interfaces:**
- Consumes: canonical `rig/rig.json`, D/E `StageManifest`, `rig/spine/export_report.json`, and `rig/live2d/export_report.json`.
- Produces: private `_ValidatedFormatReport` records and `StageGError(code, message)` failures used by the transaction task.

- [x] **Step 1: Write failing report-contract tests**

  Build real Stage C and D outputs plus structural/release Stage E outputs. Assert that the future Stage G entry point rejects a non-release E report, an invalid outer `report_sha256`, a mismatched `rig_json_sha256`, a mismatched profile/symbol/format-plan digest, an artifact row not equal to the owner manifest, a false Spine `validation.validated`, or a Live2D release record without Core consistency and SDK evidence.

- [x] **Step 2: Run tests and observe the missing integration API**

  Run:
  `\.venv\Scripts\python.exe -m pytest tests/test_auto_rig_stage_g.py -m "not optional_runtime" -q`

  Expected: collection/import failure for `module.auto_rig.stage_g` or missing `execute_stage_g_success`.

- [x] **Step 3: Implement canonical report loading and validation**

  Decode report bytes with the repository's duplicate-key/I-JSON/JCS contract, require byte-canonical round trips, recompute outer and nested report digests, compare stable report identity to the validated Rig and C manifest, and compare report artifacts to the D/E manifest records excluding `export_report.json`. Derive the Spine validator fingerprint from `validation.validator_fingerprint`; derive the Live2D validator fingerprint from the structure/release validator versions, attested Core hash/version, renderer hash, and renderer protocol digest.

- [x] **Step 4: Run focused tests and commit**

  Run the non-runtime Stage G tests and `ruff check module/auto_rig/stage_g.py tests/test_auto_rig_stage_g.py`. Commit only after GREEN.

### Task 2: Derive and commit the formal terminal payload

**Files:**
- Modify: `module/auto_rig/stage_g.py`
- Modify: `module/auto_rig/terminal.py`
- Modify: `tests/test_auto_rig_stage_g.py`
- Modify: `tests/test_auto_rig_terminal.py`

**Interfaces:**
- Consumes: `execute_stage_g_success(item_root, *, config_fingerprint, expected_stage_fingerprints)`.
- Produces: existing `TerminalFinalizationResult`; no caller-provided `FormatValidation`, profile, validator, symbol, motion, or texture facts.

- [x] **Step 1: Write failing success-derivation tests**

  Assert that a valid real C/D/E graph produces `export_manifest.json` whose input/native/override identities come from the graph, profile and profile digest come from `RigDocument.format_plans.profile`, runtime digest is JCS of `RigDocument.runtime_application`, symbol digest comes from `RigDocument.export_symbols`, format files equal D/E output inventories, and the Live2D loader contract equals the attested renderer protocol digest. Assert that changing any redundant caller fact is impossible because the API does not accept it.

- [x] **Step 2: Correct stale terminal fixture paths under RED**

  Change the fake E inventory from `avatar.*` to the frozen `model.moc3`, `model.model3.json`, and `model.cdi3.json` paths, then rerun `tests/test_auto_rig_terminal.py` to prove the terminal foundation still passes independently.

- [x] **Step 3: Implement `execute_stage_g_success`**

  Validate the full preterminal graph, require a terminal-delivery profile with exactly both formal formats, construct `TextureRuntimeContract` from fixed C/D semantics plus Live2D release evidence, build two `FormatValidation` records from current manifests, and call `finalize_success`. Keep `terminal.py` generic; only narrow a helper if Stage G otherwise has to duplicate a private invariant.

- [x] **Step 4: Prove resume and tamper behavior**

  After G commit, require `is_item_completed(...A-E...)` to return true. Tamper a report, runtime artifact, C motion manifest, or add an undeclared D/E public file and require it to return false without trusting the existing export manifest.

- [x] **Step 5: Run focused tests and commit**

  Run `tests/test_auto_rig_stage_g.py -m "not optional_runtime"`, `tests/test_auto_rig_terminal.py`, and targeted Ruff before committing.

### Task 3: Official SDK-backed end-to-end release proof

**Files:**
- Modify: `tests/test_auto_rig_stage_g.py`

**Interfaces:**
- Consumes: `LIVE2D_CUBISM_CORE_PATH` and `LIVE2D_E0_RENDERER_PATH`.
- Produces: a real C/D/E/G terminal fixture proving Stage G refuses structural E and accepts the same item only after official Core/SDK release validation.

- [x] **Step 1: Write the optional-runtime integration test**

  Create actual A/B commit markers with the same semantic identities as the geometry fixture, execute C, fan out D and release-tier E, execute G, and assert exact terminal XOR, G marker-last state, both format artifact sets, official Live2D validator evidence, byte-identical texture pages, and successful `is_item_completed` revalidation.

- [x] **Step 2: Run against SDK for Native 5-r.5**

  Set `LIVE2D_CUBISM_CORE_PATH` to the official x86_64 Core DLL and `LIVE2D_E0_RENDERER_PATH` to the rebuilt SDK harness. Run the optional-runtime Stage G test and require a pass rather than a skip.

- [x] **Step 3: Commit the runtime integration test**

  Commit only after the real Core/SDK path has produced a complete G terminal result.

### Task 4: Public API, Revision 28/29 documentation, and full verification

**Files:**
- Modify: `module/auto_rig/__init__.py`
- Modify: `tests/test_auto_rig_public_api.py`
- Modify: `docs/superpowers/specs/2026-07-31-auto-rig-from-see-through-layers-design.md`
- Modify: `docs/superpowers/plans/2026-08-01-auto-rig-stage-e-live2d-implementation.md`
- Modify: this plan

**Interfaces:**
- Produces: reviewed public `StageGError` and `execute_stage_g_success`, an updated spec status, and reproducible verification evidence.

- [x] **Step 1: Add public API assertions under RED, then export the Stage G boundary**

  Keep report parser helpers private. Expose only the stable exception and success transaction entry point.

- [x] **Step 2: Update Revision 28 Stage E facts**

  Record the MOC binding correction, exact SDK/Core versions and hashes without machine paths, focused release evidence, and the remaining external Spine runtime gate honestly.

- [x] **Step 3: Update Revision 29 Stage G facts**

  Record that terminal facts are report-derived, structural E is rejected, G resume uses A-E expected fingerprints only, and formal dual artifact inventories are atomic. Do not claim the external Spine Editor/runtime gate ran when it did not.

- [x] **Step 4: Run complete verification**

  Run all auto-rig tests with official SDK variables, relevant see-through tests, full Ruff for changed modules/tests, `compileall`, `git diff --check`, and a final clean-status/diff review. Record exact counts and any intentional optional skips.

  Final split evidence covers the same 660 auto-rig cases without a single
  opaque one-hour process: non-runtime groups total `626 passed, 1 skipped`;
  official-runtime is `29 passed, 4 skipped`; combined `655 passed, 5 skipped`.
  Upstream guards add `54 passed` for see-through and `193 passed` for
  dependency/UV. The four optional-runtime skips are environment/capability
  cases already declared by their tests; the one non-runtime skip is likewise
  explicit rather than a hidden failure.

- [ ] **Step 5: Commit, integrate to main, and push**

  Follow `superpowers:finishing-a-development-branch`, preserve unrelated main-worktree changes, integrate the reviewed commits non-interactively, rerun a post-integration smoke check, and push `main` only after fresh evidence.

## Self-Review

- Report parsing, graph validation, and terminal committing are separate responsibilities; no exporter logic moves into G.
- The public Stage G API cannot lie by injecting format status or validator fingerprints.
- Structural Live2D remains useful for development but cannot become a completed terminal item.
- The official Cubism runtime dependency remains external and hash-attested; no SDK binary enters source control or item output.
- Spine structure is shipped as a required format, while the still-unavailable official Spine runtime gate remains explicitly visible rather than silently promoted.
