# Auto-Rig Anatomy Mask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans and test-driven-development task by task.

**Goal:** Freeze the trustworthy Stage A geometry substrate that joint observation and pose cropping consume, without re-thresholding masks or conflating upstream tag suffixes with geometry-derived image sides.

**Architecture:** `component_geometry.py` reloads A-owned QCL files as an authenticated snapshot and proves every label against its `MaskComponentRecord`. `anatomy.py` builds versioned head, torso, neck, pose-body, and limb mask metrics plus explicit limb observability states. It consumes only ordinary normalized parts; NativeVariant data cannot enter joint geometry.

**Tech Stack:** Python 3.10, immutable dataclasses, NumPy 1.26, QCL1, pytest.

## Global Constraints

- Follow design spec Revision 23 Stage A and joint-observation sections.
- Never threshold source PNG/PSD again after `MaskComponentPlan`; QCL labels are the sole geometry input.
- Upstream `-r/-l` eligibility and A geometry side classification are separate registries. `legwear` and `footwear` have no v3 suffixes but may become `merged-separable` from two reliable components.
- All masks use LayerDiff canvas coordinates, top-left origin, y down.
- NativeVariant partitions are excluded from every anatomy union and pose-body bbox.
- Missing or ambiguous evidence remains explicit; no canvas-center fallback.
- Observe RED before each production behavior.

---

### Task 1: Side-Classifiable Limb Families

- [x] Add failing fixtures proving reliable two-component `legwear` and `footwear` masks become `xmin/xmax` normalized parts.
- [x] Split A side-classifier families from the upstream suffix registry and version the changed classifier.

### Task 2: Authenticated Component Geometry Loader

**Files:**
- Create: `module/auto_rig/component_geometry.py`
- Create: `tests/test_auto_rig_component_geometry.py`

- [x] Add failing tests for QCL snapshot mutation, plan-digest mismatch, label/record mismatch, and canonical load order.
- [x] Verify QCL path/digest/dimensions, labels, per-component bbox/pixel count/mask digest, and whole-part mask digest.

### Task 3: Anatomy Mask Plan And Limb States

**Files:**
- Create: `module/auto_rig/anatomy.py`
- Create: `tests/test_auto_rig_anatomy.py`

- [x] Add failing tests for `split`, `partial`, `merged-separable`, `merged-ambiguous`, and `missing` states.
- [x] Freeze versioned base-tag registries for `head_core`, `torso_core`, `neck`, `pose_body`, and limb families.
- [x] Build deterministic tight union metrics and prove head/hair exclusion plus NativeVariant exclusion.

### Task 4: Public Surface And Regression Gate

- [x] Export stable records/builders without eager NumPy imports.
- [x] Run focused tests, Ruff, compileall, SDK-backed auto-rig regression, see-through regression, and diff checks.
- [x] Commit the anatomy mask slice before implementing joint heuristics.

## Verification

- Focused component/anatomy/public API suite: `35 passed`; final anatomy validation suite: `9 passed`.
- Official SDK-backed auto-rig suite with Cubism Native 5-r.5 Core and E0 renderer: `449 passed, 4 skipped`.
- See-through regression suite: `54 passed`.
- Dependency and uv regression suite: `193 passed`.
- Ruff, `compileall`, and `git diff --check`: passed.
