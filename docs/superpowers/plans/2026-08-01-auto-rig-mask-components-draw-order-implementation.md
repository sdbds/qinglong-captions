# Auto-Rig Mask Components And Draw Order Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert validated see-through base PartSources into one deterministic A-owned component partition, materialized QCL1 label maps, normalized image-side Parts, and canonical ordinary back-to-front draw order.

**Architecture:** `mask_sources.py` is the only decoder for validated PNG/PSD alpha. `component_plan.py` performs the only alpha threshold/cleanup/connected-component pass and promotes exactly two reliable components of a splittable unsplit family to `.xmin/.xmax`; all downstream geometry consumes its records and QCL1 blobs. `draw_order.py` expands the versioned base-tag DAG over normalized Parts and applies deterministic Kahn ordering with depth buckets only inside the ready set.

**Tech Stack:** Python 3.10, Pillow, NumPy 1.26.4, SciPy 1.15.3, scikit-image 0.25.2, psd-tools 1.17.4, pytest.

## Global Constraints

- Follow design spec Revision 22 and the committed base-input contract at `68c8ad2`.
- Work only in the isolated `codex/auto-rig-see-through` worktree.
- Observe RED before every production behavior change.
- Alpha threshold is `alpha_u8 >= 1`; connectivity is 8-neighbor; v1 morphology is identity; tiny-object and tiny-hole thresholds are resolution-scaled and part of the plan descriptor.
- Connected-component library labels and traversal order never become identities.
- QCL1 is little-endian and contains no padding or trailing bytes.
- NativeVariant parsing/admission, joints, meshes, weights, and atlas packing remain outside this slice.

---

### Task 1: Geometry Dependency Profile

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_auto_rig_dependencies.py`

**Interfaces:**
- Consumes: repository optional-dependency table.
- Produces: `auto-rig` extra pinned to `numpy==1.26.4`, `scipy==1.15.3`, `scikit-image==0.25.2`, and `psd-tools[composite]==1.17.4`.

- [x] **Step 1: Write a failing pyproject contract test** that asserts the exact four dependency rows and proves `opencv-contrib-python` is absent.
- [x] **Step 2: Run `pytest tests/test_auto_rig_dependencies.py -q`** and observe failure because the extra is absent.
- [x] **Step 3: Add the minimal `auto-rig` optional dependency table.**
- [x] **Step 4: Run the dependency test and `tests/test_pyproject_uv_conflicts.py`.**

### Task 2: QCL1 Canonical Label Map Codec

**Files:**
- Create: `module/auto_rig/qcl.py`
- Create: `tests/test_auto_rig_qcl.py`

**Interfaces:**
- Produces: `CanonicalLabelMap`, `QclContractError`, `encode_qcl(labels, width, height)`, `decode_qcl(payload)`, and `materialize_qcl(item_root, payload)`.
- QCL bytes are `b"QCL1" + struct.pack("<II", width, height) + width*height little-endian uint32 labels`.

- [x] **Step 1: Write failing golden and corruption tests** for exact bytes, round-trip, bad magic, wrong dimensions, big-endian mutation, trailing bytes, non-contiguous positive labels, and hash-derived cache path.
- [x] **Step 2: Run `pytest tests/test_auto_rig_qcl.py -q`** and observe import failure.
- [x] **Step 3: Implement strict encode/decode and atomic materialization** at `rig/cache/A/components/<64hex>.qcl`; an existing same-name/different-byte file raises an invariant error.
- [x] **Step 4: Run the QCL tests and Ruff.**

### Task 3: Validated Alpha Source Decoder

**Files:**
- Create: `module/auto_rig/mask_sources.py`
- Create: `tests/test_auto_rig_mask_sources.py`

**Interfaces:**
- Consumes: `AutoRigInputContract` from `load_auto_rig_input_contract()`.
- Produces: immutable `LoadedPartAlpha(part, width, height, alpha_u8)` records from `load_validated_part_alphas(contract)`.

- [x] **Step 1: Write failing PNG/PSD parity tests** using transparent-border fixtures and assert output records are ordered by stable `part_id`.
- [x] **Step 2: Add mutation tests** proving a payload changed after contract validation is revalidated and rejected rather than silently decoded.
- [x] **Step 3: Run the focused tests and observe missing API failure.**
- [x] **Step 4: Implement one-pass PNG decoding and one-open-per-PSD-file decoding**; only the alpha channel is returned and psd-tools/NumPy imports remain lazy.
- [x] **Step 5: Run focused tests and the public lightweight-import test.**

### Task 4: MaskComponentPlan v1

**Files:**
- Create: `module/auto_rig/component_plan.py`
- Create: `tests/test_auto_rig_component_plan.py`

**Interfaces:**
- Consumes: `LoadedPartAlpha`, canonical tag metadata, canvas edge, and QCL materializer.
- Produces: `MaskComponentRecord`, `NormalizedMaskPart`, `MaskComponentPlan`, and `build_base_mask_component_plan(contract)`.

- [x] **Step 1: Write failing fixtures** for diagonal 8-connectivity, isolated speck removal, tiny-hole fill, transparent border tightening, two components, and empty cleaned mask.
- [x] **Step 2: Add identity/determinism tests** that reverse source/component traversal yet require identical component IDs, QCL bytes, normalized records, and plan digest.
- [x] **Step 3: Add side-promotion tests**: an unsplit splittable family with exactly two cleaned components becomes `.xmin/.xmax`; source `-r/-l` keeps `xmin/xmax`; ambiguous or non-splittable multi-component masks remain unsided.
- [x] **Step 4: Run focused tests and observe missing API failure.**
- [x] **Step 5: Implement the frozen cleanup descriptor** using alpha threshold 1, 8-connectivity, identity morphology, and `max(4, round_half_up(4*edge^2/1024^2))` for both small-object and small-hole thresholds.
- [x] **Step 6: Implement canonical component sorting and identity** with `(bbox.y1,bbox.x1,mask_sha256)` and `component/c_<full SHA256(JCS(identity_record))>`; build one tight QCL map per normalized Part.
- [x] **Step 7: Validate every plan**: labels are exactly `1..N`, IDs are unique, component bboxes fit the Part crop, projected drawable count equals component count, and all dependency/options versions enter the JCS plan digest.
- [x] **Step 8: Run focused tests and Ruff.**

### Task 5: DrawOrderPolicy v1

**Files:**
- Create: `module/auto_rig/draw_order.py`
- Create: `tests/test_auto_rig_draw_order.py`

**Interfaces:**
- Consumes: normalized ordinary Parts from `MaskComponentPlan`.
- Produces: `PartDrawOrderRecord`, `OrdinaryDrawOrderPlan`, `validate_draw_order_registry()`, and `build_ordinary_draw_order(parts)`.

- [x] **Step 1: Write failing tests** for depth ordering, all-equal depth, stable-ID ready ties, and depth deliberately conflicting with semantic edges.
- [x] **Step 2: Add registry tests** for the exact built-in DAG, split base-tag expansion, absent endpoints producing no phantom node, all eyewear edges, intentional headwear omission, and injected-cycle startup failure.
- [x] **Step 3: Run focused tests and observe missing API failure.**
- [x] **Step 4: Implement 256 depth buckets and deterministic Kahn ordering** with ready key `(-depth_bucket, part_id)`; semantic edges always constrain before depth comparison.
- [x] **Step 5: Serialize the policy registry, quantization rule, expanded edges, input depths, order, and ranks into a JCS plan digest; validate ranks are unique and gapless.**
- [x] **Step 6: Run focused tests and Ruff.**

### Task 6: Public Surface And Regression Gate

**Files:**
- Modify: `module/auto_rig/__init__.py`
- Modify: `tests/test_auto_rig_public_api.py`
- Modify: this plan.

**Interfaces:**
- Produces: lazy public entrypoints and immutable record types without importing NumPy, SciPy, scikit-image, or psd-tools during `import module.auto_rig`.

- [x] **Step 1: Write the failing public API expectations** and expand the heavy-import assertion to include `scipy` and `skimage`.
- [x] **Step 2: Add lazy wrappers/exports and run the public API tests.**
- [x] **Step 3: Run SDK-backed `pytest tests -q -k auto_rig`, `pytest tests -q -k see_through`, dependency tests, Ruff, and `git diff --check`.**
- [x] **Step 4: Re-read Revision 22 component/draw-order requirements, record any intentionally deferred boundary, and commit the slice.**

Verification: `294 passed, 4 skipped` for auto-rig with Cubism SDK 5-r.5 Core and the D3D11 WARP harness; `63 passed` for see-through; 28 dependency/uv-conflict tests passed; Ruff and `git diff --check` clean. NativeVariant anchor expansion, explicit limb-state diagnostics, B component-rank expansion, joints, mesh/weights, and atlas packing are intentionally deferred to their dependent slices.
