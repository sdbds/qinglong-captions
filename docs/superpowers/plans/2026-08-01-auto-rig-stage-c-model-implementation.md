# Auto-Rig Stage C Model Implementation Plan

> Execute with strict RED/GREEN tests. Stage C consumes the immutable Stage B geometry cache and is the sole writer of the complete public RigDocument and its read-only projections.

**Goal:** Implement the exporter-neutral Stage C contract: versioned controls and presets, item capabilities and model bindings, per-format feasibility decisions, the complete primitive/symbol universe, canonical shared textures, a reference-closed `RigDocument v1`, deterministic public projections, and C-owned resume/transaction boundaries.

**Architecture:** Keep semantic registries and feasibility planners pure and side-effect free. Construct every C record from typed immutable inputs, hash semantic payloads with JCS, and validate references before any public artifact is replaced. Materialization is a final transaction that writes canonical texture bytes and three public JSON projections, removes obsolete C-owned files, reloads all outputs, and only then allows the C manifest commit marker.

**Tech stack:** Python dataclasses, existing JCS/artifact/stage-manifest helpers, existing `RigGeometryCache`, `TexturePagePlan`, and canonical PNG encoder, pytest, Ruff, official Cubism SDK-backed regression for the existing E0 gates.

**Frozen scope:** Revision 25 input/geometry contracts remain unchanged. This plan produces Revision 26. D and E may consume Stage C plans but do not write Spine/MOC3 artifacts in this plan.

---

## Task 1: Freeze ControlRegistry and PresetLibrary

**Files:**
- Create: `module/auto_rig/control_registry.py`
- Create: `module/auto_rig/preset_library.py`
- Create: `tests/test_auto_rig_control_registry.py`
- Create: `tests/test_auto_rig_preset_library.py`
- Modify: `module/auto_rig/__init__.py`
- Modify: `tests/test_auto_rig_public_api.py`

**Interfaces:**
- `ControlSpec`, `Live2DParameterBinding`, `ControlRegistryPlan`
- `ControlCurve`, `MotionClipTemplate`, `ExpressionValue`, `ExpressionPresetTemplate`, `PresetLibraryPlan`
- `build_control_registry_plan()`, `validate_control_registry_plan(...)`
- `build_preset_library_plan()`, `validate_preset_library_plan(...)`

- [x] **Step 1: Write RED registry/golden tests**

  Freeze every Revision 25 control row, exact internal parameter ID, exact Live2D reserved export name, domain/default/unit, and null Spine-only wave binding. Reject duplicate control/parameter IDs, non-finite or invalid domains, illegal xmin/xmax-to-anatomical-L/R aliases, and registry digest mutation.

- [x] **Step 2: Implement immutable ControlRegistry v1**

  Materialize the complete registry independent of profile and liveness. Keep internal IDs distinct from export names. Validate ASCII IDs, global parameter uniqueness, domain invariants, and stable JCS order.

- [x] **Step 3: Write RED preset descriptor tests**

  Freeze the exact 30 Hz curves, duration, loop, kind, expression values, runtime application descriptor, and optional selection order. Reject repeated controls, out-of-domain values, non-increasing frames, missing loop endpoints, non-rest loop closure, expression transfer fields, unknown controls, and optional-priority omissions/duplicates.

- [x] **Step 4: Implement PresetLibrary motion-core-v1**

  Blink/talk are MotionClips; happy/sad/surprised are ExpressionPresets; wave sides are separate Spine-capable templates. Every template and the full library carry semantic digests. Registry load errors remain job-startup errors, not item diagnostics.

- [x] **Step 5: Export the lightweight API and verify**

  Public import must still avoid NumPy/SciPy/OpenCV/Torch. Run focused tests, Ruff, compileall, and public API coverage before committing.

## Task 2: Materialize item capabilities and atomic ControlBindings

**Files:**
- Create: `module/auto_rig/capabilities.py`
- Create: `module/auto_rig/control_bindings.py`
- Create: `tests/test_auto_rig_capabilities.py`
- Create: `tests/test_auto_rig_control_bindings.py`
- Modify: `module/auto_rig/__init__.py`
- Modify: `tests/test_auto_rig_public_api.py`

**Interfaces:**
- `RigCapability`, `CapabilityPlan`
- `TargetTransfer`, `ControlBinding`, `ControlBindingPlan`
- `derive_capabilities(cache, anatomy_plan, presets, native_variant_set) -> CapabilityPlan`
- `build_control_binding_plan(cache, anatomy_plan, controls, presets, capabilities, native_variant_set) -> ControlBindingPlan`

- [x] **Step 1: RED geometry capability matrix**

  Cover full-body, half-body, head-only, missing wrist, merged limb, missing eye layers, mouth-only, admitted native eye/mouth bundles, and degenerate mesh. Synthetic root alone never satisfies motion capability. Wave requires upper-arm/forearm/hand and mesh for the exact image side.

- [x] **Step 2: Implement reference-derived capability facts**

  Derive only from frozen Part/joint/bone/weighted-mesh/native-admission facts plus the A-owned `AnatomyMaskPlan` whose digest is already bound by `StageAJointPlan`. Record capability kind, quality tier (`native`, `procedural`, `procedural_silhouette`, `unavailable`), evidence IDs, and deterministic reason codes. Never invent joints, recompute a bbox, or reopen masks/QCL.

- [x] **Step 3: RED canonical binding and transfer tests**

  Freeze idle, breath, head nod/shake, body sway, both wave sides, native overlay opacity, and supported procedural bindings. Verify default-rest identity, visual clockwise/canvas-down semantics, geometry-normalized metric inputs, unique typed binding IDs, atomic bundle completeness, rank uniqueness, and no preset ID inside a binding.

- [x] **Step 4: Implement atomic implementation selection inputs**

  Materialize only complete item-eligible bundles. Use `canonical=0`, `native=10`, `procedural=100`; preserve all eligible alternatives for format preflight, but never mix records across an implementation. Sampled property/deform records carry evaluator/topology/input/output digests and bounded values, not Python callbacks.

- [x] **Step 5: Mutation-safe reference closure**

  Rehash the outer plan after deleting one binding, changing a target, changing default output, or cross-linking a native visibility branch. Validators must reject the structure rather than only noticing a stale outer digest.

## Task 3: Implement capability profiles and pure format preflight

**Files:**
- Create: `module/auto_rig/format_plans.py`
- Create: `tests/test_auto_rig_format_plans.py`
- Modify: `module/auto_rig/__init__.py`
- Modify: `tests/test_auto_rig_public_api.py`

**Interfaces:**
- `CapabilityProfile`, `FormatModelPlan`, `FormatPresetDecision`, `FormatPresetSetPlan`, `FormatPlanSet`
- `load_capability_profile(profile_id)`
- `build_format_plan_set(...)`
- `validate_format_plan_set(...)`

- [x] **Step 1: Freeze the three profile registries**

  `dual_runtime_core_v1` requires Spine/Live2D and idle/breath/head_nod/head_shake; avatar additionally requires blink/talk/happy/sad/surprised; `spine_4_2_dev` is non-terminal. Formal profiles cannot weaken required presets with `strict_capabilities=false`. Optional parity is `per_format`.

- [x] **Step 2: RED static model feasibility tests**

  Spine validates root/topology/components/weights/texture references. Live2D validates texture indices, section/reference bounds, and `component_count <= 1001` with exact rank mapping. Model failure is an item failure for a required format and cannot be deferred to D/E.

- [x] **Step 3: RED per-preset/bundle selector tests**

  Choose one complete implementation per semantic group by stable rank. Live2D wave is always omitted with `live2d_joint_bend_requires_glue`; Spine wave remains supported when geometry exists. Unsupported controls, missing targets, incomplete bundles, and non-rigid target conflicts carry stable reasons.

- [x] **Step 4: Implement required-first preset-set selection**

  Add all required presets atomically, then optional order `body_sway, blink, talk, surprised, happy, sad, wave.xmin, wave.xmax`, with no backtracking. Required conflicts fail; optional conflicts omit with `conflicts_with` and failed primitive keys. Persist input/output/set digests for D/E recomputation.

- [x] **Step 5: Generated superset/property tests**

  Across generated Rig shapes, every profile, and every preset, prove selected bindings are a subset of the later primitive candidate universe. Unknown selector keys must fail before writer entry.

## Task 4: Build PrimitiveCandidateSet and GlobalExportSymbolTable

**Files:**
- Create: `module/auto_rig/primitive_candidates.py`
- Create: `module/auto_rig/export_symbols.py`
- Create: `tests/test_auto_rig_primitive_candidates.py`
- Create: `tests/test_auto_rig_export_symbols.py`
- Modify: `module/auto_rig/__init__.py`
- Modify: `tests/test_auto_rig_public_api.py`

**Interfaces:**
- typed `PrimitiveCandidate`, `PrimitiveCandidateSet`
- typed `ExportNamespaceKey`, `ExportSymbol`, `GlobalExportSymbolTable`
- `enumerate_primitive_candidates(...)`
- `build_global_export_symbol_table(...)`

- [x] **Step 1: Freeze capability-independent candidate enumeration**

  Enumerate from complete Rig/control/binding/preset registries without profile pruning. Include Spine bone/slot/attachment/animation and Live2D Part/ArtMesh/Parameter/rotation/warp/motion/expression candidates only when the corresponding format binding exists. Stable IDs derive from complete typed records, never concatenated strings.

- [x] **Step 2: Implement InternalId/ExportName/SymbolKind codecs**

  Enforce grammar, 63-byte limit, namespace-scoped uniqueness, reserved parameter names, all-member collision suffixing, stable component token, and exact golden names from the spec. Different namespaces may reuse a naked name; each namespace collision class is resolved as a unit.

- [x] **Step 3: Build the profile/pruning-independent universe**

  Generate once from the complete Rig and both exporter families. Same typed key always resolves identically across profiles and exporter order. Every artifact path fragment must resolve through the table; exporters cannot sanitize again.

- [x] **Step 4: Add collision and mutation fixtures**

  Cover `a-b` versus `a.b`, 63/64 byte boundary, reserved parameter collision, multi-component token, digest collision injection, missing candidate, changed namespace, survival-set renaming, and re-sanitization attempts.

- [x] **Step 5: Prove binding-plan subset invariants**

  Generated matrices assert both candidate IDs and typed keys selected by all format plans are subsets of C's immutable candidate set and have symbol mappings.

## Task 5: Assemble and round-trip the complete RigDocument v1

**Files:**
- Create: `module/auto_rig/rig_document.py`
- Create: `tests/test_auto_rig_rig_document.py`
- Modify: `module/auto_rig/__init__.py`
- Modify: `tests/test_auto_rig_public_api.py`

**Interfaces:**
- `RigDocument`, `RigDocumentError`
- `build_rig_document(...)`, `rig_document_bytes(...)`
- `load_rig_document(...)`, `validate_rig_document(...)`, `validate_rig_document_payload(...)`

- [x] **Step 1: RED complete-schema tests**

  Require all exporter-neutral geometry plus all nine C-owned groups: capabilities, control specs, control bindings, clips, expressions, format plans, primitive candidates, export symbols, and texture pages. Missing or extra fields, C-empty partial documents, or B cache masquerading as Rig must fail.

- [x] **Step 2: Normalize Stage B geometry into public records**

  Preserve target/canvas/Part/native metrics/joint observations/resolutions/bones/weighted meshes/component ranks and diagnostics without QCL paths or A/B implementation-only cache records. Stable IDs and semantic digests remain unchanged.

- [x] **Step 3: Implement full reference-closed validation**

  Revalidate every ID/digest/domain/curve/binding/format decision/candidate/symbol/texture reference. Required decisions must cover profile formats; optional decisions may differ per format only as declared. No exporter may repair a partial document.

- [x] **Step 4: Disk round-trip and deterministic bytes**

  In-memory Rig to canonical JSON to loader must be semantically equal across shuffled input construction and `PYTHONHASHSEED`. Rehashed broken parent/joint/UV/influence/binding/candidate/symbol/texture mutations must fail.

## Task 6: Materialize canonical textures and public projections transactionally

**Files:**
- Create: `module/auto_rig/projections.py`
- Create: `module/auto_rig/stage_c.py`
- Create: `tests/test_auto_rig_projections.py`
- Create: `tests/test_auto_rig_stage_c.py`
- Modify: `module/auto_rig/manifests.py` only if a missing generic transaction hook is proven necessary
- Modify: `module/auto_rig/__init__.py`
- Modify: `tests/test_auto_rig_public_api.py`

**Interfaces:**
- `MotionManifestProjection`, `RigReportProjection`
- `project_motion_manifest(rig, profile)`, `project_rig_report(rig)`
- `execute_stage_c(...) -> StageCResult`

- [ ] **Step 1: Recompute A's final TexturePagePlan and materialize once**

  Rebuild the final-admitted part-region input from authenticated sources, require the plan to match A exactly, then encode each page once under `rig/shared/textures/page_<index>.png`. Store canonical page paths, RGBA and encoded SHA, encoder fingerprint, UV/rect contract, straight alpha, and sRGB byte semantics in Rig.

- [ ] **Step 2: Implement read-only motion/report projections**

  Projection includes `rig_json_sha256`, motion semantics digest, global symbol digest, profile, required/default/runtime application, and every per-format supported/omitted reason/artifact/incompatibility. Recompute from the just-written Rig and reject any mismatch; D/E never use projection to override Rig.

- [ ] **Step 3: Implement C's staged public transaction**

  Build under `rig/cache/C/staging`, validate bytes and schemas, atomically replace canonical textures and `rig.json`, then derive/write report and motion manifest from the on-disk Rig. Remove obsolete C-owned public files before manifest commit. A crash may leave payloads but never a reusable C marker; resume rehashes every byte.

- [ ] **Step 4: Prove exact ownership/inventory/resume behavior**

  C owns exactly `rig/rig.json`, `rig/report.json`, `rig/motion_manifest.json`, and declared shared pages. It never modifies B cache. Changed Rig C-field invalidates C/D/E/G but leaves B reusable; changed PNG encoder invalidates from C; changed TexturePagePlan semantics invalidates from A. Stale pages and hand-edited projections are rejected and removed on the next successful C commit.

- [ ] **Step 5: Failure and degraded-state tests**

  C failure removes old C commit marker and never publishes a partial success manifest. Allowed rigid fallback propagates to `stage_validated_with_degradation`; missing required capability or format-plan mismatch is a hard failure and cannot be weakened by `--allow-partial`.

## Task 7: Revision 26 integration verification

**Files:**
- Modify: `docs/superpowers/specs/2026-07-31-auto-rig-from-see-through-layers-design.md`
- Modify: this plan

- [ ] **Step 1: Run focused Stage C suites**

  Run each new module suite plus Stage B, texture, manifest, stage graph, terminal, and public API regressions.

- [ ] **Step 2: Run official SDK-backed auto-rig regression**

  Use `E:\CubismSdkForNative-5-r.5`, the official Core DLL, and the built D3D11 WARP harness. Record exact pass/skip counts.

- [ ] **Step 3: Run upstream boundary suites**

  Run see-through and dependency/uv suites, Ruff, compileall, and `git diff --check`.

- [ ] **Step 4: Update Revision 26 status and commit**

  Record exact implementation facts and counts. Do not claim D/E export completion; Stage C only freezes the complete inputs and decisions those writers must consume.

## Self-Review

- Spec coverage: all nine C-owned Rig groups, profile decisions, control/transfer separation, pure format preflight, candidate/symbol superset, shared canonical texture materialization, projections, ownership, and resume boundaries have explicit tasks and negative tests.
- Single-writer rule: B remains immutable; C alone owns the public Rig/projections/shared pages; D/E remain read-only consumers of C facts.
- No false completion: this plan does not emit Spine or Live2D runtime packages and cannot write terminal success.
- Deferred only by stage boundary: target-format encoders and runtime validators are D/E work, not silently omitted C behavior.
