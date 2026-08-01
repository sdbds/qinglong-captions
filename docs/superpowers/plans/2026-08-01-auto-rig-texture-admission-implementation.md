# Auto-Rig Texture Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the shared four-page texture plan, atomically admit only resource-feasible NativeVariant bundles, materialize canonical page bytes once, and produce the final variant-aware draw sequence.

**Architecture:** `texture_sources.py` decodes digest-pinned straight RGBA crops into a format-neutral region source. `texture_plan.py` wraps pinned rectpack MaxRects BSSF behind a versioned deterministic contract and owns placement geometry plus canonical page encoding. `native_variant_admission.py` replays the packer from an empty plan for mandatory base and each quality-eligible group, applies the 1001-drawable guard, freezes `render_variant_ids`, and expands ordinary draw anchors only after admission.

**Tech Stack:** Python 3.10, Pillow 12.3.0, rectpack 0.2.2 MaxRectsBssf, NumPy 1.26.4, RFC 8785 JCS, pytest.

## Global Constraints

- Follow design spec Revision 22 `TexturePagePlan v1` and NativeVariant admission ordering.
- Work only in the isolated `codex/auto-rig-see-through` worktree and observe RED before production changes.
- Pages are exactly `2048x2048 RGBA8`, at most four, indexed `0..N-1` with basename `page_<index>.png` and no rotation.
- Packing unit is one canonical payload crop per validated see-through source Part plus one crop per admitted NativeVariant, never a mesh component.
- Each footprint is content plus a 2 px replicated extrusion ring plus a 2 px transparent safety gap: `(w+8)x(h+8)`.
- Input sort is `(max_side desc, area desc, stable_part_id asc)`; every placement tie is deterministic by page/y/x.
- Mandatory base pack failure is `texture_budget_exceeded`; optional group pack/drawable failure rejects the entire group and continues.
- Native admission priority is exactly `blink.native`, `mouth_open.native`, `mouth_form.native`.
- Each admission attempt repacks `base + already admitted + current group` from empty bins; it never appends greedily to the previous layout.
- Native projected drawable count must remain `<=1001`; mandatory base over 1001 is reported but remains C's later hard format gate.
- A plan/dry-run never encodes PNG. Canonical C materialization uses Pillow 12.3.0, zlib runtime fingerprint, `optimize=False`, `compress_level=9`, and no metadata/profile chunks.
- D/E byte-copy behavior and final Spine/Live2D references remain later exporter work; this slice produces the one C-owned shared page artifact set.

---

### Task 1: Pinned RGBA And MaxRects Runtime

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/test_auto_rig_dependencies.py`
- Create: `module/auto_rig/texture_sources.py`
- Create: `tests/test_auto_rig_texture_sources.py`

**Interfaces:**
- Produces: `LoadedTextureRegion`, `load_base_texture_regions(contract)`, and `load_native_texture_regions(variant_set)`.

- [x] **Step 1: Write failing dependency expectations** for `pillow==12.3.0` and `rectpack==0.2.2`, update the exact auto-rig extra, install the pinned local test runtime, and verify the imported distribution versions.
- [x] **Step 2: Write failing PNG/PSD/native RGBA tests** for exact crop dimensions, raw RGBA SHA, straight/sRGB declarations, file digest mutation, wrong mode/size, and deterministic source ordering.
- [x] **Step 3: Implement shared snapshot verification and lazy PNG/PSD decoding.** PSD float channels use clip plus round-half-up to uint8; native PNG bytes are reverified against the parsed candidate SHA before decode.
- [x] **Step 4: Run texture-source, mask-source, contract, dependency tests, and Ruff.**

### Task 2: Deterministic TexturePagePlan

**Files:**
- Create: `module/auto_rig/texture_plan.py`
- Create: `tests/test_auto_rig_texture_plan.py`

**Interfaces:**
- Consumes: immutable `LoadedTextureRegion` records.
- Produces: `TextureRect`, `TextureRegionPlacement`, `TexturePageRecord`, `TexturePagePlan`, `build_texture_page_plan(regions)`.

- [x] **Step 1: Write failing geometry tests** for content/extrusion/footprint offsets, `(w+8,h+8)`, fixed page profile, no rotation, canonical 0-based paths, used-page count, budget occupancy, and used-page fill.
- [x] **Step 2: Run focused tests and observe the missing planner.**
- [x] **Step 3: Implement a rectpack 0.2.2 MaxRectsBssf adapter** with `SORT_NONE`, pre-sorted stable inputs, four explicit bins, and a deterministic tie adapter; record algorithm/dependency/rectangle-builder versions in the semantic payload.
- [x] **Step 4: Write failing determinism/safety tests** for reversed input, equal-size IDs, no footprint overlap, content-only UV rect, single-region oversize, shape-fragmentation failure despite area capacity, and page indices/paths without gaps or leading zeros.
- [x] **Step 5: Implement strict validator invariants** and distinguish a returned `fit=false` dry-run from malformed planner state.
- [x] **Step 6: Run planner tests and Ruff.**

### Task 3: Atomic NativeVariant Resource Admission

**Files:**
- Create: `module/auto_rig/native_variant_admission.py`
- Create: `tests/test_auto_rig_native_variant_admission.py`

**Interfaces:**
- Consumes: base/native region sources, combined `MaskComponentPlan`, and `NativeVariantQualityPlan`.
- Produces: `NativeVariantAdmissionAttempt`, `NativeVariantEligibilityPlan`, `admit_native_variant_resources(...)`.

- [x] **Step 1: Write failing mandatory-base tests** for successful dry-run, oversize/four-page failure as `texture_budget_exceeded`, base component count reporting, and base count above 1001 not being misreported as an optional warning.
- [x] **Step 2: Implement mandatory base as a hard gate and bind its exact input-region digest/plan.**
- [x] **Step 3: Write the golden replay test** where base fits, blink is admitted, mouth-open fails packing, and smaller mouth-form is subsequently admitted from a fresh replay; verify rejected regions leave no final-plan placeholder.
- [x] **Step 4: Write drawable-budget and atomicity tests** for base 999 plus a three-component group, no group splitting, stable rejection reason, and profile-independent results.
- [x] **Step 5: Implement frozen group priority, fresh repack per attempt, drawable-first rejection priority, final replay equality, sorted `render_variant_ids`, and a JCS self-validating eligibility digest.**
- [x] **Step 6: Run admission/quality/planner tests and Ruff.**

### Task 4: Final Draw Expansion And Canonical Page Bytes

**Files:**
- Modify: `module/auto_rig/native_variant_admission.py`
- Modify: `tests/test_auto_rig_native_variant_admission.py`
- Modify: `module/auto_rig/texture_plan.py`
- Modify: `tests/test_auto_rig_texture_plan.py`

**Interfaces:**
- Adds: `FinalPartDrawRecord`, `FinalDrawOrderPlan`, `expand_draw_order_with_admitted_variants(...)`, and `materialize_canonical_texture_pages(...)`.

- [x] **Step 1: Write failing draw tests** proving ordinary relative order is unchanged, each anchor expands to `anchor + variants sorted by (semantic_role,variant_id)`, ranks are contiguous, rejected variants are absent, and external Parts cannot enter an anchor bundle.
- [x] **Step 2: Implement final draw expansion bound to ordinary-plan, eligibility, and anchor-expander version digests.**
- [x] **Step 3: Write failing pixel tests** for row-major straight RGBA content, nearest-edge 2 px extrusion, zero 2 px safety gap, transparent-source RGB preservation, raw page pixel SHA, deterministic PNG SHA/chunks, and byte-identical repeated materialization.
- [x] **Step 4: Implement one-pass page composition and canonical PNG encoding** into `rig/shared/textures/page_<index>.png`; validate Pillow/zlib fingerprints and never attach PNGInfo/ICC/EXIF/DPI.
- [x] **Step 5: Run draw/pixel tests and Ruff.**

### Task 5: Public Surface And Regression Gate

**Files:**
- Modify: `module/auto_rig/__init__.py`
- Modify: `tests/test_auto_rig_public_api.py`
- Modify: this plan.

- [x] **Step 1: Write and observe failing public API expectations for the stable source/plan/admission/draw interfaces.**
- [x] **Step 2: Export the supported surface with Pillow/NumPy/rectpack imports remaining lazy.**
- [x] **Step 3: Run official-SDK-backed auto-rig regressions, see-through regressions, dependency/uv tests, Ruff, compileall, and `git diff --check`.**
- [x] **Step 4: Record verification, commit the slice, and carry exact plan/page digests into the subsequent Stage A/C integration.**

## Verification

- Official Cubism SDK `E:\CubismSdkForNative-5-r.5`: `387 passed, 4 skipped` for `pytest -k auto_rig`.
- See-through regression: `63 passed`.
- Dependency and uv regression: `182 passed`.
- Focused texture/component/variant suite: `83 passed`.
- Ruff, `compileall`, and `git diff --check`: passed.
