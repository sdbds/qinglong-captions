# Auto-Rig Stage D Spine 4.2 Implementation Plan

> Execute with strict RED/GREEN tests. Stage D is a read-only consumer of the complete Stage C `RigDocument v1`; it must never rename symbols, repack textures, mutate format decisions, or write terminal completion.

**Goal:** Produce a deterministic Spine 4.2 runtime package containing a setup skeleton, weighted mesh attachments, canonical atlas pages, selected motion/expression animations, an independently recomputable validation report, and a D-owned stage manifest.

**Architecture:** Parse and validate the canonical public RigDocument, project it through isolated coordinate/bind/UV plans into a disposable `SpineDocument`, serialize target files canonically, reload them through a strict pure-Python validator, and publish the complete D inventory with the manifest last. Official Spine Editor/runtime loading remains an opt-in external release gate; ordinary CI must prove all format invariants it can inspect without pretending to own commercial software.

**Frozen scope:** Revision 26 C facts are immutable. Stage D supports exact Spine `4.2`, straight-alpha `pma:false`, one component per slot/setup attachment, weighted meshes, selected linear motion clips, two-key expression holds, and canonical C page byte-copy. It does not write Live2D artifacts or `export_manifest.json`.

---

## Task 1: Spine coordinate and setup-bind plans

**Files:**
- Create: `module/auto_rig/export/spine/__init__.py`
- Create: `module/auto_rig/export/spine/coordinates.py`
- Create: `module/auto_rig/export/spine/bind_plan.py`
- Create: `tests/test_auto_rig_spine_coordinates.py`
- Create: `tests/test_auto_rig_spine_bind_plan.py`

- [x] **Step 1: Write failing coordinate contract tests**

  Freeze `canvas_to_spine(x,y)=(x-W/2,H/2-y)`, the exact inverse, synthetic root identity, finite-value rejection, canonical test-point digest, and `<=1e-6 px` round trip on center/corners/joints/vertices.

- [x] **Step 2: Implement `SpineCoordinatePlan v1`**

  Keep the pure transform isolated and versioned. Materialize a digest-bearing plan from the Rig canvas and reject non-top-left/down canvas contracts rather than guessing.

- [x] **Step 3: Write failing hierarchy/bind tests**

  Include a three-level non-zero translated/rotated hierarchy. Assert parent-world inverse local heads/angles, reconstructed heads/tails, per-influence bind-local vertices, nearest-common-ancestor slot ownership, and mutations that merely subtract coordinates or reuse the slot-bone point.

- [x] **Step 4: Implement `SpineBindPlan v1`**

  Build bones in parent topology, emit root identity, compute `W_parent * T(local_head) * R(local_angle)`, normalize angles to `[-180,180)`, transform each weighted influence separately, and record maximum reconstruction residual plus input/output digests.

## Task 2: Spine symbol view, UV adapter, and atlas

**Files:**
- Create: `module/auto_rig/export/spine/symbols.py`
- Create: `module/auto_rig/export/spine/uv.py`
- Create: `module/auto_rig/export/spine/atlas.py`
- Create: `tests/test_auto_rig_spine_symbols.py`
- Create: `tests/test_auto_rig_spine_atlas.py`

- [x] **Step 1: Write failing symbol-subset tests**

  Resolve every Spine bone/slot/attachment/region/animation/page only by the Stage C typed key and symbol table. Missing, duplicate, renamed, or exporter-sanitized keys must fail.

- [x] **Step 2: Implement immutable Spine symbol lookup**

  Expose exact typed-key and internal-identity lookups without a naming fallback. Preserve attachment-key scope separately from actual attachment object and atlas region names.

- [x] **Step 3: Write failing UV/atlas tests**

  Use a non-square region and asymmetric vertices. Assert canonical page-top-left coordinates become Spine attachment-local UVs once, region offsets/sizes match the C placement, pages are ordered `page_0..N-1`, atlas uses ASCII/LF, and every page says `pma: false`.

- [x] **Step 4: Implement `Spine42UvAdapter v1` and atlas writer/parser**

  Derive local region UV from each canvas vertex and the part content rect, serialize deterministic multi-page atlas grammar, validate unique effective attachment paths, and never decode/re-encode or repack page pixels.

## Task 3: Setup skeleton and mesh encoder

**Files:**
- Create: `module/auto_rig/export/spine/model.py`
- Create: `module/auto_rig/export/spine/mesh_encoder.py`
- Create: `tests/test_auto_rig_spine_mesh_encoder.py`
- Create: `tests/test_auto_rig_spine_model.py`

- [x] **Step 1: Write failing weighted/unweighted mesh tests**

  Assert flat triangles/UV arrays and official variable-length weighted vertices `[boneCount,boneIndex,x,y,weight,...]`. Bone indices follow skeleton order, weights preserve normalized Rig values, and rigid meshes use plain local `x,y` pairs.

- [x] **Step 2: Implement `SpineMeshEncoder`**

  Consume BindPlan influence records and convert canonical absolute sampled vertices to setup-relative offsets only where required. Reject non-finite values, topology mismatch, unknown bones, invalid triangles, non-positive/unnormalized weights, and reconstructed setup error above `0.1 px`.

- [x] **Step 3: Write failing setup-document tests**

  Assert Spine metadata version `4.2`, parent-topological bones, canonical draw-order slots, one visible setup attachment per component, shared part region for sibling components, setup alpha 0 only for native variants, and complete default skin maps.

- [x] **Step 4: Implement disposable `SpineDocument` builder**

  Build setup bones/slots/skins from Rig, BindPlan, atlas and global symbols without mutating the Rig. Slot owner is the influence NCA; weighted vertices retain their real bone indices.

## Task 4: Motions and expression animations

**Files:**
- Create: `module/auto_rig/export/spine/animations.py`
- Create: `tests/test_auto_rig_spine_animations.py`

- [x] **Step 1: Write failing motion evaluation tests**

  For every Spine-supported C decision, evaluate the selected control curves through selected bindings/transfers. Assert `frame/30` times, canonical `(dx,-dy,-theta)` conversion, setup-relative deform offsets, slot opacity targets, no resampling of affine transfers, and omission of target-format `curve` for linear segments.

- [x] **Step 2: Implement bone/slot/deform timelines**

  Merge property channels deterministically by animation and target, preserve the C-selected binding union, and reject any supported decision whose artifact or visible binding is absent.

- [x] **Step 3: Write failing expression-hold tests**

  Assert each supported expression produces exactly two equal setup-relative keys at `0` and `1/30 s`, while runtime track/loop/replace/full-alpha/zero-mix semantics remain in the public manifest rather than invented Spine JSON fields.

- [x] **Step 4: Implement expression animations**

  Apply full-weight overwrite values through the same transfers as motions and encode the frozen `SpineExpressionHold v1` contract. Reject unsupported blend modes and duplicate control ownership.

## Task 5: Strict Spine 4.2 validator and deterministic serialization

**Files:**
- Create: `module/auto_rig/export/spine/validator.py`
- Create: `module/auto_rig/export/spine/serializer.py`
- Create: `tests/test_auto_rig_spine_validator.py`

- [x] **Step 1: Write failing structural mutation tests**

  Cover wrong Spine version, nested arrays, illegal timeline curves, missing setup attachment/region/page, reversed slot rank, bad bone index/weight, mismatched region path, `pma:true`, stale symbol, non-canonical JSON, and changed canonical page bytes.

- [x] **Step 2: Implement canonical JSON and ASCII atlas serialization**

  Encode `skeleton.json` and report as RFC 8785 JCS bytes; encode atlas as printable ASCII with LF, stable page/region order, no trailing metadata, and a versioned encoding descriptor in the D fingerprint.

- [x] **Step 3: Implement independent parser/validator**

  Reload public bytes, reconstruct skeleton world transforms and weighted setup vertices, verify every reference and selected animation against Rig/FormatPlan, recompute coordinate/bind/UV/artifact digests, and return a validator fingerprint suitable for G.

## Task 6: Stage D transaction and public report

**Files:**
- Create: `module/auto_rig/stage_d.py`
- Create: `tests/test_auto_rig_stage_d.py`
- Modify: `module/auto_rig/__init__.py`
- Modify: `tests/test_auto_rig_public_api.py`

- [x] **Step 1: Write failing publish/failure/resume-boundary tests**

  Require exact D inventory: `rig/spine/skeleton.json`, `skeleton.atlas`, copied `textures/page_<index>.png`, and `export_report.json`. A failure removes the old D commit marker, writes only private failure evidence, and never leaves stale D-owned public files accepted as reusable.

- [x] **Step 2: Implement `execute_stage_d`**

  Load and validate `rig/rig.json`, recompute the Spine model/preset planners and require digest identity, compile in a private staging directory, byte-copy C pages, validate staged files, publish payloads, remove obsolete D artifacts, then write the D manifest last.

- [x] **Step 3: Expose the reviewed public API**

  Export only stable plan/document/validator/execute entry points; keep codecs and transaction helpers module-private.

## Task 7: Revision 27 verification

**Files:**
- Modify: `docs/superpowers/specs/2026-07-31-auto-rig-from-see-through-layers-design.md`
- Modify: this plan

- [ ] **Step 1: Run focused Stage D and Stage C regression suites**
- [ ] **Step 2: Run official Cubism SDK-backed auto-rig regression to protect the existing E0 contract**
- [ ] **Step 3: Run see-through, dependency/uv, relevant Ruff, `compileall`, and `git diff --check`**
- [ ] **Step 4: Record exact Revision 27 implementation status and commit**

  Do not claim official Spine runtime loading unless an actual 4.2 Editor/runtime gate ran. Stage D can be structurally complete while release validation remains an explicit external gate.

## Self-Review

- Stage ownership: C remains the sole Rig/shared-page writer; D owns only `rig/spine/**` plus its private manifest/failure record.
- No false parity: Spine optional wave support is retained without fabricating a Live2D counterpart; required presets remain C-controlled.
- No hidden naming: all exported identifiers and artifact fragments come from the global typed symbol table.
- No fake runtime proof: pure-Python parser/reconstruction is mandatory CI; official Spine 4.2 loading is reported separately.
