# Auto-Rig NativeVariant Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Partition every manifest-valid NativeVariant exactly once and deterministically decide its role-aware coverage, scale, intrusion, and atomic quality eligibility before any texture-resource admission.

**Architecture:** Extend the A-owned `MaskComponentPlan` with explicit NativeVariant partition records while leaving ordinary see-through Parts unchanged. A new pure `native_variant_quality.py` layer reconstructs raw alpha only through digest-pinned sources and A-owned QCL supports, evaluates exact integer gates plus deterministic role envelopes, and emits immutable per-branch, per-candidate, and atomic-bundle quality records. Texture packing, projected drawable admission, final `render_variant_ids`, draw-anchor expansion, mesh generation, and runtime binding remain the next A/B/C slices.

**Tech Stack:** Python 3.10, NumPy 1.26.4, SciPy 1.15.3, scikit-image 0.25.2, QCL1, RFC 8785 JCS, pytest.

## Global Constraints

- Follow design spec Revision 22 and the frozen Revision 20 NativeVariant envelope.
- Work only in the isolated `codex/auto-rig-see-through` worktree.
- Observe RED before every production behavior change.
- `MaskComponentPlan` owns the only threshold/cleanup/component result; quality code may decode QCL but may not relabel, threshold, merge, or split.
- Use `alpha_threshold_u8=1`, 8-connectivity, the existing resolution-scaled cleanup, and full SHA-derived component IDs.
- Shared feature scale is always `sqrt(base_alpha_mass_u8_sum / 255)`; no canvas-, bbox-, span-, or variant-derived radius input is allowed.
- Role envelopes are exact rationals: eye `k=3/2,c=1/5,r=2..12`; mouth form `k=4/1,c=1/1,r=4..24`; mouth open `k=12/1,c=2/1,r=8..48`.
- Gate thresholds are `coverage_leak <= 0.01`, `alpha_mass_ratio <= k_role`, and `occlusion_intrusion <= 0.01`.
- Quality rejection is capability data, not `input_contract_mismatch`; malformed manifest/path/reference remains the parser's hard input error.
- `NativeVariantQualityPlan` must not expose candidates as final render Parts or imply atlas admission.

---

### Task 1: Candidate Component Partition

**Files:**
- Modify: `module/auto_rig/component_plan.py`
- Modify: `tests/test_auto_rig_component_plan.py`

**Interfaces:**
- Consumes: `MaskComponentPlan` from `build_base_mask_component_plan()` and `NativeVariantSet` from `load_native_variant_set()`.
- Produces: frozen `NativeVariantPartitionRecord` entries and `extend_mask_component_plan_with_variants(base_plan, variant_set, item_root)`.

- [x] **Step 1: Write failing partition tests** for an empty set, one cropped candidate, stable candidate/component ordering, QCL materialization, cleanup-empty candidate, and plan/set digest binding.
- [x] **Step 2: Run focused tests and observe missing partition API failures.**
- [x] **Step 3: Implement variant partitioning without reprocessing ordinary masks.** A successful record stores candidate identity, source `xyxy`, tight cleaned `xyxy`, QCL digest, cleaned mask digest, and component records; cleanup-empty stores a stable `empty_after_cleanup` status with no fake QCL/component.
- [x] **Step 4: Write failing side tests.** Single-side blink assigns every surviving component to its registry side without changing `part/native.<variant_id>`; coupled blink assigns `xmin/xmax` only when exactly two components pass the existing 5% reliability rule; mouth components remain unsided.
- [x] **Step 5: Implement side classification and require the final component-plan semantic payload to bind `native_variant_set_sha256`, base projected count, candidate projected count, and every partition record.**
- [x] **Step 6: Run component/native parser regressions and Ruff.**

### Task 2: Frozen Role Envelope And Exact Composite Metrics

**Files:**
- Create: `module/auto_rig/native_variant_quality.py`
- Create: `tests/test_auto_rig_native_variant_quality.py`

**Interfaces:**
- Consumes: validated base `LoadedPartAlpha` values, the combined `MaskComponentPlan`, ordinary draw order, and `NativeVariantSet`.
- Produces: `NativeVariantRoleEnvelope`, `IntrusionPartMetric`, `NativeVariantBranchMetrics`, `NativeVariantQualityResult`, and `build_native_variant_quality_plan(...)`.

- [x] **Step 1: Write failing registry and exact-radius tests** for exact role rows/digest, duplicate/missing roles, invalid rational/clamp values, `mass_B=200` radii `3/14/28`, equal mass with different bbox span, resolution independence, and min/max clamps.
- [x] **Step 2: Run focused tests and observe the missing quality module.**
- [x] **Step 3: Implement startup registry validation and integer-inequality round-half-up.** Compare `4*n^2*mass_u8` with `d^2*255*(2*q-1)^2`; never call binary-float `sqrt()` to choose the radius.
- [x] **Step 4: Write failing composite tests** for 99% coverage, role-specific alpha ratios, base-anchored Euclidean-disk face authorization, face-only/nose-only/mixed intrusion, prefix cutoff at the anchor, and exact zero entries sorted by `part_id`.
- [x] **Step 5: Implement deterministic 8-bit source-over and visible-contribution math.** Reconstruct ordinary and variant alpha through QCL support; compute integer gate numerators, use `mass_V*255` as the intrusion denominator, retain face/other/per-Part closure, and keep legacy spill only as a non-gating diagnostic.
- [x] **Step 6: Add mutation tests** proving variant support/bbox cannot expand authorization, a legal mouth-open expansion can pass, a cheek patch outside the base envelope fails, `mass_V/mass_B>12` fails, and nose/eyebrow coverage is never authorized.
- [x] **Step 7: Run quality tests and Ruff.**

### Task 3: Coupled Branches And Atomic Quality Bundles

**Files:**
- Modify: `module/auto_rig/native_variant_quality.py`
- Modify: `tests/test_auto_rig_native_variant_quality.py`

**Interfaces:**
- Adds: `NativeVariantBundleQualityResult` and `NativeVariantQualityPlan`.
- Bundle IDs: `blink.native`, `mouth_open.native`, `mouth_form.native`.

- [x] **Step 1: Write failing coupled-eye tests** proving two component branches get independent base/variant mass, radius, coverage, and intrusion equal to equivalent single-side fixtures; zero/one/three/unreliable components reject with `component_partition`.
- [x] **Step 2: Implement branch selection exclusively from the frozen component records.** Do not merge coupled alpha mass and do not regenerate connected components.
- [x] **Step 3: Write failing atomic-bundle tests** for two singles versus one coupled blink, incomplete blink, independent mouth-open, complete smile+frown, one bad mouth-form endpoint, and a completely absent source producing empty candidate/bundle records with a version-distinct fixed digest.
- [x] **Step 4: Implement stable reason priority** `component_partition`, `coverage_leak`, `alpha_mass_ratio_exceeded`, `occlusion_intrusion`; then apply `bundle_incomplete` only at the atomic group layer. Only complete groups whose every candidate/branch passes enter `quality_eligible_variant_ids`.
- [x] **Step 5: Validate every stored digest and closure invariant by rebuilding the semantic payload; changing iteration order must not change output bytes.**
- [x] **Step 6: Run focused component/parser/quality tests and Ruff.**

### Task 4: Public Surface And Regression Gate

**Files:**
- Modify: `module/auto_rig/__init__.py`
- Modify: `tests/test_auto_rig_public_api.py`
- Modify: this plan.

**Interfaces:**
- Publicly export only stable plan/record types, version constants, registry validator, component extension, and quality-plan builder.

- [x] **Step 1: Write and observe failing public API expectations.**
- [x] **Step 2: Export the supported surface without importing Pillow/NumPy/SciPy at module import time.**
- [x] **Step 3: Run official-SDK-backed `-k auto_rig`, `-k see_through`, dependency/uv tests, Ruff, and `git diff --check`.**
- [x] **Step 4: Record verification and the resource-admission boundary, then commit this slice.**

Verification completed with Cubism SDK for Native 5-r.5/Core 06.00.0001 and the D3D11 WARP E0 harness enabled: `362 passed, 4 skipped` for auto-rig, `63 passed` for see-through, and `28 passed` for dependency/uv locks; Ruff and `git diff --check` were clean. `quality_eligible_variant_ids` is intentionally not `render_variant_ids`: shared MaxRects replay, the 1001-drawable guard, atomic resource rejection, final anchor expansion, and admitted texture-plan identity remain the next A-planner slice.
