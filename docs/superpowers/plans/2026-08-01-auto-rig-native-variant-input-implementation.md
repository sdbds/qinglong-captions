# Auto-Rig NativeVariant Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse the optional `rig_inputs/variants` directory into a strict, profile-independent semantic candidate set whose paths, pixels, role references, anchor, and digest are safe for the existing A-owned component planner.

**Architecture:** `native_variants.py` owns the manifest/role registry and performs only input-contract decisions. It consumes the already normalized ordinary `MaskComponentPlan` and `OrdinaryDrawOrderPlan` to validate role-derived base sets and frontmost anchors, then emits immutable candidate records with alpha bytes and a JCS semantic set digest. Quality, coverage, intrusion, atomic-bundle eligibility, resource admission, and anchor bundle expansion remain a later A-planner layer.

**Tech Stack:** Python 3.10, Pillow, RFC 8785 JCS, pytest.

## Global Constraints

- Follow design spec Revision 22 NativeVariant contract (Revision 20 semantics).
- Work only in the isolated `codex/auto-rig-see-through` worktree.
- Observe RED before every production behavior change.
- Directory absence is a valid empty semantic set; directory presence with any malformed entry is `input_contract_mismatch`, never a procedural fallback.
- Raw manifest whitespace/key order never enters the semantic set digest; verified PNG byte SHA does.
- The parser never evaluates coverage, occlusion intrusion, component quality, atlas capacity, or runtime capability.

---

### Task 1: Manifest Schema And Filesystem Boundary

**Files:**
- Create: `module/auto_rig/native_variants.py`
- Create: `tests/test_auto_rig_native_variants.py`

**Interfaces:**
- Consumes: `AutoRigInputContract`, `MaskComponentPlan`, `OrdinaryDrawOrderPlan`.
- Produces: `NativeVariantCandidate`, `NativeVariantSet`, `load_native_variant_set(contract, component_plan, draw_order_plan)`.

The v1 JSON envelope is exact:

```json
{
  "schema_version": "native-variant-manifest-v1",
  "variants": [{
    "variant_id": "blink_xmin",
    "semantic_role": "eye_closed.xmin",
    "composite_mode": "occluding_overlay_v1",
    "base_part_ids": ["part/eyelash.xmin", "part/eyewhite.xmin", "part/irides.xmin"],
    "draw_anchor_part_id": "part/eyelash.xmin",
    "xyxy": [10, 20, 50, 40],
    "path": "blink_xmin.png",
    "rgba_mode": "RGBA",
    "alpha_mode": "straight",
    "color_space": "sRGB",
    "file_sha256": "sha256:<64 lowercase hex>"
  }]
}
```

- [x] **Step 1: Write failing empty-directory and happy-path tests** for directory absence, an empty manifest, stable candidate ordering, RGBA alpha bytes, and whitespace/key-order independent semantic digest.
- [x] **Step 2: Write failing path/identity tests** for absolute/`..`/non-exact path, invalid or reserved ID, duplicate ID/path/role, symlink/reparse, missing/extra/non-file entries, wrong SHA, wrong dimensions/mode, all-transparent PNG, unknown role, unknown field, duplicate JSON key, and non-finite JSON.
- [x] **Step 3: Run the focused tests and observe missing module failure.**
- [x] **Step 4: Implement strict schema/path/inventory/image validation** with exact `variant_id.png`, 64-character slug rules, Windows reserved basename rejection, straight-sRGB-RGBA constants, canvas-bounded exclusive `xyxy`, positive alpha mass, and digest-pinned bytes.
- [x] **Step 5: Build `native_variant_set_sha256`** from schema version and entries sorted by `variant_id`; normalize `base_part_ids` after duplicate detection and include verified PNG SHA exactly once.

### Task 2: Role-Derived References And Anchor Contract

**Files:** same as Task 1.

**Interfaces:**
- Uses normalized ordinary Part IDs/base tags/sides and `ordinary_part_order` ranks.
- Adds inherited `anchor_base_tag`, `anchor_depth_median`, and derived `part/native.<variant_id>` to each candidate.

- [x] **Step 1: Write failing role-reference tests** for eye xmin/xmax/coupled and mouth roles, including split component-promoted Parts.
- [x] **Step 2: Add rejection tests** for missing/extra/cross-family base, duplicate base before normalization, nonexistent base/anchor, anchor outside base, non-frontmost anchor, `part/native.` collision, coupled+single eye overlap, and inconsistent smile/frown base or anchor.
- [x] **Step 3: Run focused tests and observe reference checks fail.**
- [x] **Step 4: Implement registry-derived exact base sets**: side eye roles select existing `{eyewhite,irides,eyelash}` Parts on that image side; coupled selects all existing eye-family Parts; mouth roles select all existing `mouth` Parts.
- [x] **Step 5: Validate the anchor is exactly the max ordinary rank among base Parts**, reject ambiguous role overlap, and preserve incomplete blink/mouth-form bundles for later quality/atomic eligibility rather than misclassifying them as schema errors.
- [x] **Step 6: Run focused tests and Ruff.**

### Task 3: Public Surface And Regression Gate

**Files:**
- Modify: `module/auto_rig/__init__.py`
- Modify: `tests/test_auto_rig_public_api.py`
- Modify: this plan.

- [x] **Step 1: Write failing public API expectations** for the immutable records, role/schema constants, error, and loader without importing heavy geometry dependencies.
- [x] **Step 2: Export the supported surface and run focused/public tests.**
- [x] **Step 3: Run SDK-backed auto-rig regressions, see-through regressions, Ruff, and `git diff --check`.**
- [x] **Step 4: Record the quality/admission boundary explicitly and commit the parser slice.**

Verification completed with the official Cubism Native SDK/Core gate enabled: `321 passed, 4 skipped` for auto-rig, `63 passed` for see-through, `28 passed` for dependency/uv lock checks, plus clean Ruff and `git diff --check` runs. This slice stops deliberately at manifest-valid candidates. Component partitioning, role-aware coverage/intrusion quality, atomic bundle eligibility, atlas/resource admission, and draw-anchor bundle expansion belong to the next A-planner slice and must not be inferred from `NativeVariantSet.present` or candidate presence.
