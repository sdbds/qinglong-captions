# Auto-Rig Stage E Live2D Runtime Implementation Plan

> Execute with strict RED/GREEN tests. Stage E is a read-only consumer of the complete Stage C `RigDocument v1`; it must not rename symbols, change preset selection, repack textures, or publish terminal completion.

**Goal:** Produce a deterministic Cubism runtime bundle (`model.moc3`, `model.model3.json`, `model.cdi3.json`, selected `.motion3.json`/`.exp3.json`, canonical texture copies, and an export report), prove its structure in ordinary CI, and prove each release artifact with the configured official Cubism Core/SDK before committing the E marker.

**Architecture:** Correct the Stage C primitive identity so per-mesh sampled deformation targets ArtMesh keyforms rather than inventing an unverified WarpDeformer. Project the canonical Rig through isolated Live2D symbol, parameter, binding/liveness, coordinate, ArtMesh, animation, and artifact plans into a disposable `Moc3V400Document`. Encode through the attested V4.00 section codec, reload through a strict pure-Python validator, then run Core consistency/model-state and SDK D3D11 WARP render/playback gates. Publish the exact E inventory marker-last. The packaged E0 attestation remains the startup proof for coordinate/section semantics; each item still receives its own runtime proof.

**Frozen scope:** Revision 27 A-D outputs remain immutable. Stage E targets MOC3 V4.00 header version 3 and model3 Version 3, fixed basename `model`, at most four 2048x2048 straight-alpha pages copied byte-for-byte from C, one Part per Rig Part, one ArtMesh per Rig mesh component, single-parameter ArtMesh keyforms for sampled deform/opacity, and nested RotationDeformers keyed by `(bone_id, parameter_id)`. Multi-parameter ArtMesh grids, Glue, physics, pose files, CMO3, and formal Live2D limb bends remain unsupported.

---

## Task 0: Correct the exporter-neutral non-rigid primitive identity

**Files:**
- Modify: `module/auto_rig/primitive_candidates.py`
- Modify: `tests/test_auto_rig_primitive_candidates.py`
- Modify: `tests/test_auto_rig_format_plans.py`

- [x] **Step 1: Write the failing primitive-identity regression**

  Assert that a sampled per-mesh `deform` binding selects the existing `live2d_artmesh` typed target, while only an explicit future `structural_warp` property may allocate `live2d_warp_deformer`. Assert blink deform and opacity rows merge on the same ArtMesh target, mouth-open uses its mouth ArtMesh, and no selected production binding asks Stage E for rotation-under-warp or warp-under-rotation semantics absent from the signed frame contract.

- [x] **Step 2: Implement and version the correction**

  Bump the primitive enumerator version, reuse the static ArtMesh typed key for sampled deform/opacity bindings, preserve candidate/symbol superset invariants, and rerun Stage C/D regressions because candidate IDs and the global symbol-table digest intentionally change.

## Task 1: Live2D symbols, rigid-driver registry, parameter and binding plans

**Files:**
- Create: `module/auto_rig/export/live2d/symbols.py`
- Create: `module/auto_rig/export/live2d/rigid_drivers.py`
- Create: `module/auto_rig/export/live2d/binding_plan.py`
- Create: `tests/test_auto_rig_live2d_symbols.py`
- Create: `tests/test_auto_rig_live2d_binding_plan.py`

- [x] **Step 1: Write failing symbol and registry tests**

  Resolve Part/ArtMesh/RotationDeformer/Parameter/motion/expression IDs only from the Stage C typed symbol table. Freeze globally unique ranks `body_sway=100`, `idle=200`, `head_shake=300`, `head_nod=400`; duplicate rank, unknown control, exporter-side sanitize, missing symbol, or string-split identity recovery must fail before item staging.

- [x] **Step 2: Implement `Live2DSymbolView v1` and `RigidDriverRegistry v1`**

  Materialize immutable typed maps plus independent digests. Keep registry rows out of the attested frame-contract digest but include them in the Stage E fingerprint/report.

- [x] **Step 3: Write failing binding/liveness tests**

  Select exactly the C-supported Live2D decisions, merge multiple properties for the same `(parameter, primitive)` pair, reject two parameters on one non-rigid ArtMesh, reject duplicate parameter curves, and prove all emitted deformer/ArtMesh bindings are bidirectionally reachable. Spine-only wave candidates must remain dead and absent.

- [x] **Step 4: Implement `Live2DBindingPlan v1` and `Live2DDriverLiveness v1`**

  Emit only used parameters and `(bone_id, parameter_id)` rigid instances. Order same-bone instances by rank, find the nearest live ancestor through the Rig bone tree, fold pruned rest transforms into children/ArtMeshes, and record emitted/pruned/reparent evidence without deriving liveness from writer sections.

## Task 2: Coordinate, ArtMesh, UV and keyform plans

**Files:**
- Create: `module/auto_rig/export/live2d/coordinates.py`
- Create: `module/auto_rig/export/live2d/artmesh.py`
- Create: `module/auto_rig/export/live2d/keyforms.py`
- Create: `tests/test_auto_rig_live2d_coordinates.py`
- Create: `tests/test_auto_rig_live2d_artmesh.py`
- Create: `tests/test_auto_rig_live2d_keyforms.py`

- [x] **Step 1: Write failing coordinate/liveness reconstruction tests**

  Freeze the attested canvas-to-root transform, root RotationDeformer scale `1/PPU`, nested same/descendant-bone authoring-local pivots, lower-rank-outer ordering, and direct-parent ArtMesh local points. Reconstruct every setup vertex through the emitted stack within `0.1 px`; reversing two non-commuting instances or retaining a dead limb node must fail.

- [x] **Step 2: Implement `Live2DCoordinatePlan v1`**

  Consume the packaged frame attestation at structural startup, record every node/frame/parent and round-trip sample, and expose only attested root/rotation/ArtMesh conversions. Do not introduce a production WarpDeformer path without a new E0 signature.

- [x] **Step 3: Write failing ArtMesh/UV tests**

  Map one Rig component to one ArtMesh, preserve canonical draw rank and triangle winding, enforce signed-int16 vertex indices, convert canonical top-left UV through `CubismV400UvAdapter`, assign texture indices from C pages, and reject missing/duplicate placements, out-of-range indices, changed topology, or UV/page mismatches.

- [x] **Step 4: Implement static ArtMesh plans**

  Encode setup positions in the direct-parent local frame, setup opacity from Part visibility, one texture region per Part, and exact Part/ArtMesh IDs from the global symbols.

- [x] **Step 5: Write failing non-rigid keyform tests**

  Compile C’s exact sampled stop values into single-parameter ArtMesh position/opacity keyforms, require a rest-equivalent default key, enforce at most 17 stops, no topology change/NaN/triangle flip, scalar residual `<=1/255`, vertex residual `<=0.1 px` at stored stops and nine interval samples, and the one-million-position/64-MiB capacity guards.

- [x] **Step 6: Implement `Live2DKeyformPlan v1`**

  Merge deform and opacity properties targeting the same ArtMesh/parameter, transform every absolute canvas sample into the ArtMesh parent-local frame, preserve all piecewise-linear knots, and leave static objects with a one-key zero-band binding.

## Task 3: MOC3 V4.00 document compiler and structural validator

**Files:**
- Create: `module/auto_rig/export/live2d/document.py`
- Create: `module/auto_rig/export/live2d/validator.py`
- Create: `tests/test_auto_rig_live2d_document.py`
- Test (consolidated with the shared expensive fixture): `tests/test_auto_rig_live2d_document.py`

- [x] **Step 1: Write failing section-layout compiler tests**

  Assert all 23 count entries, Parts/RotationDeformers/ArtMeshes/Parameters, keyform bands/bindings/keys, UV/index/draw-order groups, parent/specific indices, default/rest values, zero runtime sections, `reflect=false`, and header version 3. Require decode/encode byte identity and deterministic bytes across input ordering.

- [x] **Step 2: Implement `CubismDocument v1` to `Moc3V400Document` compilation**

  Build section offsets from typed plans only. Rotation keyforms use min/default/max (deduplicated only when values coincide), `T(origin)*R(angle)*S(scale)`, opacity 1 and no reflection. ArtMesh bindings serialize exact per-parameter keys. Reject more than 32767 vertices/indices, more than four pages, non-finite data, dead parameters/deformers, or file size above 64 MiB.

- [x] **Step 3: Write failing parser/reconstruction/mutation tests**

  Reload the binary and reconstruct default/non-default model state from section data. Mutate parent indices, binding bands, parameter defaults, draw order, texture index, UV orientation, keyform positions, reflection flags, counts, or padding and require a specific structural error. Default parameter state must reproduce setup vertices/opacities/draw order within `0.1 px`.

- [x] **Step 4: Implement `Live2DStructureValidator v1`**

  Validate the binary against the immutable plans and Rig, not against self-described report claims. Report exact artifact/plan digests, capacity guards, round-trip residuals, emitted/pruned identities, and parameter-to-visible-target closure.

## Task 4: model3/cdi3, motion3 and exp3 assets

**Files:**
- Create: `module/auto_rig/export/live2d/runtime_assets.py`
- Create: `module/auto_rig/export/live2d/animations.py`
- Create: `tests/test_auto_rig_live2d_runtime_assets.py`
- Create: `tests/test_auto_rig_live2d_animations.py`

- [x] **Step 1: Write failing model3/cdi3 tests**

  Freeze basename `model`, Version 3, contiguous `textures/page_0.png...`, exact supported motion/expression references, no empty sections, no physics/pose/PMA extension, parameter/Part IDs from symbols, and EyeBlink/LipSync groups only when their used parameters exist. Missing/stale references or JCS/minified runtime JSON must fail.

- [x] **Step 2: Implement deterministic runtime JSON encoders**

  Reuse `cubism_runtime_json_bytes()` (ASCII, sorted keys, indent 2, terminal newline) so Framework 5-r.5 sees numeric terminators. Keep reports/JCS separate from runtime JSON.

- [x] **Step 3: Write failing motion/expression tests**

  Convert each C-supported MotionClip curve exactly at `frame/30`, encode only linear segment type 0, one curve per parameter, zero fade, exact duration/loop/count metadata, and no resampling. Encode each supported ExpressionPreset as full-weight `Overwrite`, zero fade, with no pseudo-time axis. Reject duplicate parameters, unsupported blend, stale artifacts, or missing parameter keyforms.

- [x] **Step 4: Implement motion3/exp3 compilation**

  Preserve C artifact names and runtime application ordering. Blink/talk remain motions; happy/sad/surprised remain expressions. Optional format asymmetry stays explicit in the C motion manifest rather than being guessed from the output directory.

## Task 5: Official Core/SDK per-item release validation

**Files:**
- Modify: `module/auto_rig/export/live2d/cubism_renderer.py`
- Modify: `tools/auto_rig_live2d_e0/main.cpp`
- Create: `module/auto_rig/export/live2d/release_validator.py`
- Create: `tests/test_auto_rig_live2d_release_validator.py`

- [x] **Step 1: Write failing Core gate tests**

  Require the configured Core binary to match the packaged attestation allowlist, `csmHasMocConsistency` to pass, the model to revive/update with finite vertices, all IDs/counts/defaults to match the structural plan, and at least one nonzero drawable. Missing/unattested Core must be a release-gate error, not a structural-test skip.

- [x] **Step 2: Implement Core consistency/model-state validation**

  Use the existing crash-isolated worker and capture model state at default plus every parameter test point. Prove each declared visible parameter changes at least one expected deformer result, vertex, or opacity and that resetting defaults restores the setup state.

- [x] **Step 3: Extend the SDK renderer to multiple texture pages**

  Accept ordered repeated `--texture` arguments, create/bind one D3D11 view per MOC texture index, and reject count/order mismatch. Keep the existing one-page CLI path compatible and rebuild with `E:\CubismSdkForNative-5-r.5`.

- [x] **Step 4: Write and implement motion/expression render gates**

  Render setup, midpoint/end samples for every required motion, every supported optional motion, and expression apply/clear states. Require nonempty alpha, observed parameter values equal canonical evaluation, visible pixel/vertex changes for declared effects, and restoration after clear. Record renderer/Core binary hashes and evidence digests without redistributing either binary.

## Task 6: Stage E transaction and public API

**Files:**
- Create: `module/auto_rig/stage_e.py`
- Modify: `module/auto_rig/__init__.py`
- Create: `tests/test_auto_rig_stage_e.py`
- Modify: `tests/test_auto_rig_public_api.py`

- [x] **Step 1: Write failing E transaction tests**

  Require exactly the current C manifest, validate `rig.json` and shared page hashes, stage the whole bundle privately, run structural plus release validators, copy pages byte-for-byte, publish exact E-owned inventory, remove obsolete files, write the E marker last, and write only private failure evidence on error. D outputs and terminal markers must remain untouched.

- [x] **Step 2: Implement `execute_stage_e`**

  Expose explicit structural/release tiers. Formal batch uses release tier and cannot commit E without Core/SDK evidence; ordinary CI may compile/validate a structural result but its report/status cannot be mistaken for formal release completion.

- [x] **Step 3: Expose only reviewed public entry points**

  Export stable plans, validators and `execute_stage_e`; keep section assembly and transaction helpers private.

## Task 7: Revision 28 integration verification

**Files:**
- Modify: `docs/superpowers/specs/2026-07-31-auto-rig-from-see-through-layers-design.md`
- Modify: this plan

- [x] **Step 1: Run focused Stage E, Stage D and Stage C suites**
- [x] **Step 2: Run the complete official SDK/Core-backed auto-rig suite**
- [x] **Step 3: Run see-through, dependency/uv, relevant Ruff, `compileall`, and `git diff --check`**
- [x] **Step 4: Record exact Revision 28 facts and commit**

  Record the Stage C primitive correction explicitly, the exact Core/SDK paths and binary versions/hashes as evidence (never as machine-specific public paths), structural and release validation counts, and the remaining Stage G boundary. Do not claim dual-format item completion before G is implemented and verified.

  Final evidence: official SDK for Native 5-r.5 with Core 06.00.0001
  (`d883c00d114fdf6cef61f439feb23e02d000fdf683e092803010470b80dfaf09`),
  D3D11 harness
  (`823b03ea43e77da5f9238ad55e2c9c23fa54a34e7d4d0dade04973bee589778a`),
  and protocol
  `51e77ee76d08072db76e1ccef0638e8c706ccba7bae5ea2ae8e19b71269283a3`.
  The later Stage G integration run extended the final split auto-rig total to
  `655 passed, 5 skipped`; see-through is `54 passed`, dependency/UV is
  `193 passed`. The regenerated E0 attestation changed only renderer source
  provenance after the reviewed multipage harness edit; the signed frame
  contract digest remained unchanged.

## Self-Review

- Per-mesh sampled output is no longer mislabeled as a structural WarpDeformer, avoiding an unsigned coordinate topology and an unnecessary grid-fitting approximation.
- RotationDeformer identity and nesting are derived before serialization from typed `(bone, parameter)` facts and globally unique ranks.
- Runtime JSON deliberately does not reuse JCS bytes; public reports/manifests remain canonical JCS.
- Structural CI never pretends to be a release gate, while formal E cannot pass without per-item official Core/SDK evidence.
- Stage E owns only `rig/live2d/**` plus its private marker/failure record and cannot publish terminal success.
