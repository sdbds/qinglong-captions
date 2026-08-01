# Auto-Rig Stage B Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete B-owned bone, mesh, draw-rank, and skin-weight cache from authenticated Stage A component labels and joint evidence, without creating a partial public RigDocument.

**Architecture:** `bone_graph.py` turns the frozen joint plan into a declarative, topologically ordered bone graph. `mesh_builder.py` consumes A-owned QCL labels component by component and freezes deterministic contour sampling and Delaunay topology. `skinning.py` assigns semantic bone candidates and computes resolution-independent weights along the resolved joint-chain arc. `rig_geometry.py` validates and assembles the only B-owned public-to-C cache projection, while C and both exporters remain unable to inspect masks.

**Tech Stack:** Python 3.10, immutable dataclasses, NumPy 1.26.4, SciPy 1.15.3/Qhull, scikit-image 0.25.2, JCS, pytest.

## Global Constraints

- Follow design spec Revision 24.
- B consumes authenticated `MaskComponentPlan v1`/QCL and `StageAJointPlan v1`; it never thresholds PNG/PSD, labels connected components, or runs a pose model.
- `bone/root` is the only zero-length/synthetic bone. Every other emitted bone requires two resolved joints and a positive finite length; a missing parent is promoted to the closest emitted ancestor.
- Meshes are built per frozen component ID. No triangle may cross another component or accepted transparent/hole samples.
- Persisted vertices are quantized to `1/256` canvas px. Qhull `QJ` is forbidden; symbolic perturbation exists only in the topology copy and is `<1/4096 px`.
- Mesh rest positions remain LayerDiff canvas coordinates; UVs remain part-local top-left `u-right/v-down` values in `[0,1]`.
- Each vertex has 1-4 finite non-negative influences summing to one. Limb transitions scale with local mask/joint radius and chain arc length, never a fixed pixel constant.
- B writes only `rig/cache/B/**`. `rig/rig.json` remains C-owned and must not exist merely because B succeeded.
- Observe RED before every production behavior and commit each task only after focused and SDK-backed regression gates.

---

### Task 1: Declarative Bone Graph And Length Gate

**Files:**
- Create: `module/auto_rig/bone_graph.py`
- Create: `tests/test_auto_rig_bone_graph.py`
- Modify: `module/auto_rig/__init__.py`
- Modify: `tests/test_auto_rig_public_api.py`

**Interfaces:**
- Consumes: `StageAJointPlan` from `build_stage_a_joint_plan(...)`.
- Produces: `build_bone_graph(joints: StageAJointPlan) -> BoneGraphPlan` and `validate_bone_graph(plan) -> BoneGraphPlan`.
- `BoneGraphPlan.bones` is parent-topological and begins with `bone/root`; `RigBone` stores `bone_id`, `spec_id`, `parent_id`, `head_joint_id`, `tail_joint_id`, `role`, `head`, `tail`, `length`, and `parent_promotion`.

- [x] **Step 1: Write failing registry and topology tests**

  Cover the exact root/lower-torso/torso/neck/head, arm, and leg `BoneSpec` rows from Revision 24. Assert that missing wrist omits forearm/hand, a present child whose declared parent is absent promotes to the closest emitted ancestor, and shuffled joint-record input cannot change bytes or topology.

- [x] **Step 2: Run the focused tests and observe missing API failures**

  Run: `.\.venv\Scripts\python.exe -m pytest tests\test_auto_rig_bone_graph.py -q`

- [x] **Step 3: Implement the immutable registry and builder**

  Freeze `BONE_SPEC_REGISTRY_VERSION="bone-spec-registry-v1"`, `BONE_GRAPH_PLAN_VERSION="bone-graph-plan-v1"`, minimum non-root length `1/256 px`, and maximum non-root length `2*canvas_diagonal`. Resolve coordinates only through `StageAJointPlan.joints.resolutions`; never look at observations by source or invent `(0,0)`.

- [x] **Step 4: Add RED tests for invalid override lengths and mutation-safe validation**

  A coincident override pair and an explicitly outside pair beyond the maximum length must raise `invalid_bone_length`; a geometry-derived coincident pair must omit the affected bone with a stable diagnostic. Recompute the outer JCS digest in a mutation fixture so the validator proves reference/topology invariants rather than only detecting a stale outer hash.

- [x] **Step 5: Export the stable API and run focused regression**

  Run bone graph, joint pipeline, public API, Ruff, and `compileall` tests before committing.

### Task 2: Deterministic Per-Component Mesh Build Plan

**Files:**
- Create: `module/auto_rig/mesh_builder.py`
- Create: `tests/test_auto_rig_mesh_builder.py`
- Modify: `module/auto_rig/component_geometry.py`
- Modify: `module/auto_rig/__init__.py`
- Modify: `tests/test_auto_rig_component_geometry.py`
- Modify: `tests/test_auto_rig_public_api.py`

**Interfaces:**
- Consumes: authenticated `MaskComponentPlan`, item root, admitted `render_variant_ids`, and the final Part set.
- Produces: `load_mesh_component_sources(...) -> tuple[MeshComponentSource, ...]`, `build_mesh_plan(...) -> MeshBuildPlan`, and `validate_mesh_plan(...)`.
- Each `MeshRecord` owns exactly one A component ID and stores canonical vertices, boundary flags/order, flat triangles, part-local UVs, component mask digest, source kind, and build descriptor digest.

- [ ] **Step 1: Extend authenticated component loading to admitted variants**

  Write RED fixtures for a ready admitted variant, a rejected variant, a mutated variant QCL, and an unknown render ID. Reuse QCL authentication and component-record verification; do not reopen variant PNG or run cleanup.

- [ ] **Step 2: Freeze sampling, quantization, and identity records**

  Define `MESH_BUILD_PLAN_VERSION="mesh-build-plan-v1"`, quantization denominator `256`, perturbation denominator `4096`, canonical contour winding/start, scale-relative boundary/interior spacing, and dependency/options fingerprints. Stable mesh IDs derive from the full typed record `{schema, component_id, component_mask_sha256, mesh_plan_version}`.

- [ ] **Step 3: Implement RED/green contour and topology fixtures**

  Cover rectangle, concave C, hole, two components, duplicate/near-collinear/cocircular points, and a degenerate component. Assert positive signed area, in-range flat indices, no accepted centroid or edge quarter sample in alpha zero, no cross-component edge, canonical vertex/triangle order, and `degenerate_mesh` instead of a random joggle.

- [ ] **Step 4: Prove deterministic topology under input/library ordering changes**

  Mutation fixtures shuffle source/component/contour/sample/simplex order and monkeypatch a reversed Delaunay simplex array. The final `MeshBuildPlan` must remain equal and byte-stable. A spy must prove threshold and connected-component APIs are never called in B.

- [ ] **Step 5: Export and run mesh golden/regression gates**

  Run mesh/component/public API tests, Ruff, `compileall`, and the full auto-rig suite before committing.

### Task 3: Semantic Part Binding And Arc-Length Skin Weights

**Files:**
- Create: `module/auto_rig/skinning.py`
- Create: `tests/test_auto_rig_skinning.py`
- Modify: `module/auto_rig/__init__.py`
- Modify: `tests/test_auto_rig_public_api.py`

**Interfaces:**
- Consumes: `MeshBuildPlan`, `BoneGraphPlan`, `StageAJointPlan`, normalized Part metadata, and component masks already authenticated for Task 2.
- Produces: `build_skinning_plan(...) -> SkinningPlan` and `validate_skinning_plan(...)`.
- `WeightedMeshRecord` retains Task 2 topology and adds one canonical influence tuple per vertex plus allowed-bone registry evidence and `dynamic_candidate`.

- [ ] **Step 1: Freeze semantic Part-to-bone candidate registry**

  RED fixtures prove face/hair/eyes cannot receive arm weights, `handwear.xmin/xmax` only sees its own emitted chain, `legwear/footwear` use the matching image-side leg chain, variants inherit their admitted base/anchor candidates, and `tail/wings/objects` remain rigid root/torso dynamic candidates.

- [ ] **Step 2: Implement rigid fallback without hiding degradation**

  Non-limb Parts and limb Parts with fewer than two usable chain bones receive exactly one weight `1.0`. Missing semantic targets fall back through the frozen ancestor order and emit `rigid_fallback_applied`; an unknown tag/Part is diagnosed rather than silently attached and removed from reports.

- [ ] **Step 3: Implement polyline arc projection and radius-scaled transitions**

  Project each limb vertex to the resolved shoulder/elbow/wrist/hand-tip or hip/knee/ankle/toe polyline, use cumulative chain arc as the longitudinal coordinate, and blend only adjacent emitted bones inside a half-width derived from the joint eligibility radius. Prune sub-threshold influences, sort by bone ID, cap at four, and renormalize with a deterministic remainder rule.

- [ ] **Step 4: Add invariance and false-influence fixtures**

  Scale the same mask/joints from 768 to 1280 coordinates and assert normalized weight profiles remain within the frozen tolerance. Cover bent limbs, missing wrists, crossing image-side limbs, vertices exactly on transition boundaries, influence pruning, finite/sum-to-one validation, and unknown bone references.

- [ ] **Step 5: Export and run focused/full regression gates**

  Run skinning/bone/mesh/public API, Ruff, `compileall`, and SDK-backed auto-rig tests before committing.

### Task 4: Component Draw Ranks And RigGeometryCache

**Files:**
- Create: `module/auto_rig/rig_geometry.py`
- Create: `tests/test_auto_rig_rig_geometry.py`
- Modify: `module/auto_rig/__init__.py`
- Modify: `tests/test_auto_rig_public_api.py`
- Modify: `docs/superpowers/specs/2026-07-31-auto-rig-from-see-through-layers-design.md`

**Interfaces:**
- Consumes: target/component/final-draw/joint/bone/mesh/skinning plans plus A-C fingerprint inputs.
- Produces: `build_rig_geometry_cache(...) -> RigGeometryCache`, `validate_rig_geometry_cache(...)`, and `rig_geometry_cache_bytes(...)` for the B owner to publish at `rig/cache/B/rig_geometry.json`.
- Cache contains no capability/control/clip/expression/format/symbol/texture-page fields and therefore cannot pass as `RigDocument v1`.

- [ ] **Step 1: Implement `ComponentDrawOrderExpander v1`**

  RED fixtures assign ranks by `(part_draw_rank, component_id)`, require `0..N-1` with no gaps, and keep every Part's components contiguous. Reordering mesh construction cannot alter ranks; a mesh for an A-table-external component fails.

- [ ] **Step 2: Assemble the immutable B cache**

  Include canvas/target/A-plan digests, normalized Part facts, raw joint observations/resolutions, bones, weighted meshes, component ranks, A/B diagnostics, dependency descriptors, and degradation state. JCS arrays use canonical ID/topological/rank order.

- [ ] **Step 3: Build a reference-closed validator and negative fixtures**

  Reject duplicate/unknown IDs, broken parent or joint references, invalid triangles/UVs/influences, non-contiguous ranks, component/Part mismatches, stale nested digests, and every C-owned field. Recompute outer digest in each mutation so inner validation is exercised.

- [ ] **Step 4: Prove ownership and resume boundaries**

  Serialize only `rig/cache/B/rig_geometry.json`; test that B success never creates or mutates `rig/rig.json`. The B manifest output inventory contains the cache payload and no A/C-owned path. Changing mesh descriptor invalidates B/C but leaves A reusable.

- [ ] **Step 5: Update Revision 25 implementation status and verify**

  Run the SDK-backed auto-rig suite, see-through suite, dependency/uv suite, Ruff, `compileall`, and `git diff --check`; record exact counts in this plan and the spec, then commit the complete Stage B slice.

## Self-Review

- Spec coverage: tasks cover the BoneSpec tree, parent promotion, zero/max length gate, per-A-component topology, no re-threshold rule, deterministic Qhull handling, semantic candidate registry, arc/radius weighting, component draw expansion, private cache schema, and B/C ownership boundary.
- Deferred by design: C capabilities/control/motion/symbol/texture materialization and D/E format encoding are not Stage B work; Task 4 exposes all immutable facts they need.
- Placeholder scan: no TBD/TODO or unspecified “handle errors” step remains; each negative behavior has a named test target and diagnostic.
- Type consistency: Task 1 `BoneGraphPlan`, Task 2 `MeshBuildPlan`, Task 3 `SkinningPlan`, and Task 4 `RigGeometryCache` are the only cross-task outputs.
