# Auto-Rig Stage A Base Input Contract Implementation Plan

> **For agentic workers:** Use test-driven development and execute every task to completion before moving into mask geometry.

**Goal:** Turn the base payload of a completed see-through item directory into one strict Stage A input snapshot without ever reading the root Marigold `info.json` or pre-postprocess tag PNGs. NativeVariant parsing, semantic set hashing, and eligibility remain a separate follow-on slice.

**Architecture:** A versioned v3 tag registry owns tag/side/Part ID semantics. `contracts.py` validates fixed relative paths, JSON schemas, square canvas identity, final tag universe, and either PNG or PSD payload geometry. The result contains references and metadata only; later mask loading must consume these validated references rather than reopening arbitrary paths.

**Tech Stack:** Python 3.10, Pillow, optional `psd-tools==1.17.4`, pytest.

## Global Constraints

- Follow design spec Revision 22.
- Work only in the isolated `codex/auto-rig-see-through` worktree.
- Observe RED before production edits.
- Accept only `tag_version=v3` and canvas edges `768/1024/1280`.
- Treat `optimized/info.json` as the sole geometry manifest.
- Treat item-root tag PNG/depth files and item-root `info.json` as permanently invalid PartSource locations.
- Reject symlink/junction/reparse traversal before parsing any required file.

### Task 1: Canonical v3 Tag Registry

**Files:** `module/auto_rig/tag_registry.py`, `tests/test_auto_rig_tag_registry.py`

- [x] Freeze 24 raw LayerDiff candidates, 23 final base tags, six LR-splittable families, source suffix mapping, semantic slugs, and typed Part IDs.
- [x] Reject unknown tags, illegal split families, mixed unsplit/split family output, incomplete LR pairs, and identity collisions.

### Task 2: Shared Manifest And Canvas Contract

**Files:** `module/auto_rig/contracts.py`, `tests/test_auto_rig_contracts.py`

- [x] Validate fixed regular-file paths and duplicate-key/non-finite JSON rejection.
- [x] Validate `layerdiff/manifest.json`, `optimized/manifest.json`, `optimized/info.json`, `src_img.png`, resolution/frame size, bbox/depth, and exact final tag set.
- [x] Cover 768/1024/1280, reject 2048/non-square/unknown tag version, and prove root fallback files are ignored.

### Task 3: PNG PartSource

**Files:** same as Task 2.

- [x] Require exact optimized inventory and per-tag RGBA/depth PNG pairs.
- [x] Require cropped dimensions to equal exclusive `xyxy`; reject missing, extra, wrong-mode, and wrong-size payloads.

### Task 4: PSD PartSource

**Files:** same as Task 2.

- [x] Lazily import psd-tools and require final/color and depth canvases to equal the contract canvas.
- [x] Require flat unique layer names exactly equal final tags and stored layer rectangles exactly equal `xyxy`, including transparent-border fixtures.

### Task 5: Public API And Verification

**Files:** `module/auto_rig/__init__.py`, public API tests.

- [x] Export only immutable contract/registry types and loader entrypoints.
- [x] Run all auto-rig and see-through regressions, ruff, `git diff --check`, then commit the Stage A base-input slice.

Verification: `253 passed, 4 skipped` for auto-rig with Cubism SDK 5-r.5 Core and the D3D11 WARP harness; `63 passed` for see-through; Ruff and `git diff --check` clean.
