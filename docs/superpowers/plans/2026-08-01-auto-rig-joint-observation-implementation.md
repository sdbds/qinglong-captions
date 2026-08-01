# Auto-Rig Joint Observation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans and test-driven-development task by task.

**Goal:** Implement the complete Stage A joint evidence model, deterministic constraint resolver, and geometry observations consumed by later bones and exporters.

**Architecture:** A central joint registry owns every legal joint ID. `joint_observations.py` keeps geometry, pose, length-prior, and override evidence on separate scales, derives content-addressed observation IDs, and applies the frozen decision table. Geometry providers consume `AnatomyMaskGeometry` only: axial observations come from torso/head cross-sections and contact bands; limb observations come from skeleton graphs and inter-part contact. Exporters never access masks.

**Tech Stack:** Python 3.10, immutable dataclasses, NumPy, SciPy ndimage, scikit-image morphology/graph primitives, JCS, pytest.

## Global Constraints

- Follow design spec Revision 23.
- `xmin/xmax` are image-side labels, never anatomical left/right.
- Override > high-confidence geometry > validated pose > weak length prior. Geometry factors and pose scores remain separate fields.
- Missing parts produce `missing`; present but ambiguous/unsolved evidence produces `unresolved`; neither may become canvas-center coordinates.
- Every geometry observation records source mask metric/component IDs and explainable factors.
- NativeVariant masks cannot enter eligibility, observations, local radii, or pose crop.
- Observe RED before each production behavior.

---

### Task 1: Joint Registry And Evidence Resolver

- [x] Centralize legal joint IDs and reuse them in override parsing.
- [x] Add failing tests for content-addressed observations, source-specific validation, override precedence, high/low geometry behavior, pose disagreement, weak priors, missing/unresolved states, and deterministic input order.
- [x] Implement immutable eligibility, observation, resolution, resolver descriptor, and plan records.

### Task 2: Torso And Head Geometry Observations

- [ ] Freeze cross-section/contact/geodesic-axis constants and dependency versions.
- [ ] Generate pelvis/spine/neck/shoulder/hip plus head-base/head-top evidence without bbox-center shortcuts.
- [ ] Add rotated, gapped-contact, hair-exclusion, missing-mask, and fragmented-head fixtures.

### Task 3: Limb Skeleton And Contact Observations

- [ ] Build deterministic medial-axis graphs from authenticated anatomy masks.
- [ ] Generate elbow/knee curvature, wrist bottleneck, hand-tip, ankle contact, and toe evidence with explicit eligibility and factors.
- [ ] Add bent/straight/branched/merged/missing-foot fixtures and false-resolve guards.

### Task 4: Stage A Joint Plan Integration

- [ ] Convert validated overrides to authoritative observations and reserve pose-provider injection without enabling a default model.
- [ ] Build the complete joint plan from anatomy + optional pose + overrides; persist all unresolved reasons.
- [ ] Export public records/builders and run SDK-backed regressions before committing.
