# Auto-Rig Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic artifact, stage-manifest, resume-validation, and terminal-finalization foundation required by the accepted auto-rig design before geometry or format exporters are allowed to write public outputs.

**Architecture:** Add a dependency-light `module.auto_rig` package. Canonical JSON and hashing form the bottom layer; strict immutable stage manifests sit above it; `StageGraphValidator` validates the A-E/G DAG and output ownership; the G finalizer is the only writer of the mutually exclusive public terminal artifacts. The first slice deliberately does not infer joints, build meshes, or claim Spine/Live2D delivery.

**Tech Stack:** Python 3.10 standard library, frozen dataclasses, pathlib, hashlib, pytest.

## Global Constraints

- Follow `docs/superpowers/specs/2026-07-31-auto-rig-from-see-through-layers-design.md` Revision 13.
- Keep model revisions and model quantization out of this implementation.
- Do not add preview, GIF, WebM, interactive editing, or partial-delivery semantics.
- A-E/G output ownership is strict. A path may be owned by exactly one stage.
- C remains the only future writer of `rig.json`; G is the only writer of `export_manifest.json` and `error.json`.
- `skip_completed` must be based on recursive manifest and artifact validation, never file existence.
- All JSON fingerprints use canonical ASCII JSON, sorted keys, compact separators, finite numeric values, and `sha256:<hex>` strings.
- Production code is written only after its focused test has failed for the expected reason.
- The existing see-through baseline must remain green in the isolated `.venv`.
- This foundation plan is one independently testable slice. Geometry, Spine 4.2, and Live2D MOC3 receive separate implementation plans after these contracts are stable.

---

### Task 1: Deterministic JSON And Artifact Digests

**Files:**
- Create: `module/auto_rig/__init__.py`
- Create: `module/auto_rig/artifacts.py`
- Test: `tests/test_auto_rig_artifacts.py`

- [x] **Step 1: Write failing canonicalization tests**

Cover mapping-order independence, `ensure_ascii=True`, newline-independent payload digesting, `NaN` rejection, `sha256:<hex>` formatting, file size capture, and atomic replacement without `.part` residue.

- [x] **Step 2: Run the focused test and confirm import failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auto_rig_artifacts.py -q`

Expected: FAIL because `module.auto_rig.artifacts` does not exist.

- [x] **Step 3: Implement the minimal artifact layer**

Provide:

```python
canonical_json_bytes(payload) -> bytes
canonical_json_sha256(payload) -> str
sha256_file(path) -> str
describe_file(root, relative_path) -> FileDigest
atomic_write_bytes(path, payload) -> None
atomic_write_json(path, payload) -> None
```

`FileDigest.path` is a normalized relative POSIX path and includes byte size plus SHA-256. Reject absolute paths, `.`/`..`, empty paths, files outside the item root, non-regular files, and malformed digest strings.

- [x] **Step 4: Run the focused tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auto_rig_artifacts.py -q`

Expected: PASS.

- [x] **Step 5: Commit**

```powershell
git add module/auto_rig/__init__.py module/auto_rig/artifacts.py tests/test_auto_rig_artifacts.py docs/superpowers/specs/2026-07-31-auto-rig-from-see-through-layers-design.md docs/superpowers/plans/2026-08-01-auto-rig-foundation-implementation.md
git commit -m "feat: add deterministic auto-rig artifacts"
```

### Task 2: Strict StageManifest Contract

**Files:**
- Create: `module/auto_rig/manifests.py`
- Test: `tests/test_auto_rig_manifests.py`

- [x] **Step 1: Write failing manifest-schema tests**

Cover a valid A manifest and rejection of unknown stages, invalid schema versions, unsorted or duplicate file records, path traversal, a manifest listing its own commit-marker path, malformed fingerprints, unknown payload fields, and output files missing at commit time.

- [x] **Step 2: Run the focused test and confirm import failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auto_rig_manifests.py -q`

Expected: FAIL because `module.auto_rig.manifests` does not exist.

- [x] **Step 3: Implement immutable manifest records**

Implement `StageManifest`, `StageManifestError`, `manifest_relative_path(stage)`, `build_stage_fingerprint(...)`, `build_stage_manifest(...)`, `read_stage_manifest(...)`, and `write_stage_manifest(...)`.

The serialized contract contains:

```text
schema_version
stage_name
stage_schema_version
algorithm_version
stage_fingerprint
upstream_manifests{}
input_file_sha256[]
relevant_config_fingerprint
rig_overrides_sha256
output_file_sha256[]
status
```

Lists and maps are canonical and sorted. The commit marker `rig/cache/<stage>/manifest.json` is never included in its own output list. `write_stage_manifest` re-describes every declared output from disk before atomically writing the marker.

- [x] **Step 4: Run artifact and manifest tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auto_rig_artifacts.py tests/test_auto_rig_manifests.py -q`

Expected: PASS.

- [x] **Step 5: Commit**

```powershell
git add module/auto_rig/manifests.py tests/test_auto_rig_manifests.py
git commit -m "feat: define auto-rig stage manifests"
```

### Task 3: Recursive StageGraphValidator And Ownership

**Files:**
- Create: `module/auto_rig/stage_graph.py`
- Test: `tests/test_auto_rig_stage_graph.py`

- [x] **Step 1: Write failing DAG tests**

Build small A, B, C, D, E, G fixtures. Assert that a valid graph is reusable and that validation reports stable issue codes for a missing marker, changed output bytes, expected fingerprint mismatch, upstream marker digest mismatch, undeclared upstream, dependency cycle, output ownership overlap, output path equal to another stage marker, and G completed without both D and E.

- [x] **Step 2: Run the focused test and confirm import failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auto_rig_stage_graph.py -q`

Expected: FAIL because `module.auto_rig.stage_graph` does not exist.

- [x] **Step 3: Implement graph validation**

Implement frozen `StageNode`, `StageValidationIssue`, and `StageGraphResult` records plus `StageGraphValidator`.

Rules:

- The release DAG is `A -> B -> C -> D`, `C -> E`, and `C/D/E -> G`; F is never a release dependency.
- Validation recursively verifies marker parsing, expected stage fingerprints, marker SHA references, every output size/hash, and global ownership disjointness.
- A stage cannot own any other stage's marker path.
- G with `status="completed"` requires reusable C, D, and E.
- Results collect deterministic sorted issues instead of throwing on the first corrupt item; programmer/configuration errors still raise at graph construction.

- [x] **Step 4: Run focused tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auto_rig_artifacts.py tests/test_auto_rig_manifests.py tests/test_auto_rig_stage_graph.py -q`

Expected: PASS.

- [x] **Step 5: Commit**

```powershell
git add module/auto_rig/stage_graph.py tests/test_auto_rig_stage_graph.py
git commit -m "feat: validate auto-rig stage DAG"
```

### Task 4: G-Owned Success And Failure Finalization

**Files:**
- Create: `module/auto_rig/terminal.py`
- Test: `tests/test_auto_rig_terminal.py`

- [x] **Step 1: Write failing terminal-state tests**

Cover:

- successful dual-runtime finalization writes `export_manifest.json`, removes stale `error.json`, and writes G marker last;
- failure finalization writes stable ordered `error.json`, removes stale export state, and records one or multiple failure stages;
- the two public terminal files are strict XOR;
- success refuses missing/unvalidated Spine or Live2D outputs;
- `invalidate_terminal()` removes only G-owned files;
- `is_item_completed()` returns true only when the entire A-E/G graph and terminal cross-digests remain valid;
- modifying a D/E artifact after success makes completion false even while `export_manifest.json` still exists.

- [x] **Step 2: Run the focused test and confirm import failure**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auto_rig_terminal.py -q`

Expected: FAIL because `module.auto_rig.terminal` does not exist.

- [x] **Step 3: Implement the terminal finalizer**

Implement `StageFailureRecord`, `TerminalFinalizationError`, `finalize_success`, `finalize_failure`, `invalidate_terminal`, and `is_item_completed`.

Success must:

- validate current C/D/E markers and output artifacts;
- require exactly the formal formats `spine_4_2` and `live2d_moc3_v4_00`, both status `validated`;
- emit sorted `{path,size,sha256}` format files and artifact-set digests;
- bind profile, validator fingerprints, motion manifest SHA, and global symbol table SHA;
- atomically publish `export_manifest.json`, then write the G marker as commit marker.

Failure must:

- normalize and sort failure records by stage;
- preserve stable diagnostics and retryability;
- create a failure-set digest;
- atomically publish `error.json`, then write the G marker as commit marker.

- [x] **Step 4: Run all foundation tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auto_rig_artifacts.py tests/test_auto_rig_manifests.py tests/test_auto_rig_stage_graph.py tests/test_auto_rig_terminal.py -q`

Expected: PASS.

- [x] **Step 5: Commit**

```powershell
git add module/auto_rig/terminal.py tests/test_auto_rig_terminal.py
git commit -m "feat: finalize auto-rig item state"
```

### Task 5: Foundation Contract Integration And Regression Gate

**Files:**
- Modify: `module/auto_rig/__init__.py`
- Modify: `docs/superpowers/plans/2026-08-01-auto-rig-foundation-implementation.md`
- Test: `tests/test_auto_rig_public_api.py`

- [ ] **Step 1: Write the public API smoke test**

Import only the intentionally supported foundation records and functions from `module.auto_rig`. Confirm importing the package does not import Torch, OpenCV, psd-tools, or Cubism dependencies.

- [ ] **Step 2: Run the smoke test and confirm it fails before exports are added**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auto_rig_public_api.py -q`

Expected: FAIL because the public exports are not complete.

- [ ] **Step 3: Export the stable foundation API**

Keep implementation helpers private and give the package an explicit `__all__`.

- [ ] **Step 4: Run foundation plus existing see-through tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_auto_rig_*.py -q
$tests = Get-ChildItem -LiteralPath tests -Filter 'test_see_through_*.py' | Sort-Object Name | ForEach-Object FullName
.venv\Scripts\python.exe -m pytest @tests -q
```

Expected: all pass.

- [ ] **Step 5: Run static checks on changed Python files**

Run:

```powershell
.venv\Scripts\python.exe -m compileall -q module/auto_rig tests/test_auto_rig_*.py
uvx ruff check module/auto_rig tests/test_auto_rig_*.py
```

Expected: PASS.

- [ ] **Step 6: Review the complete branch diff**

Run: `git diff --check` and `git status --short`.

Confirm no main-worktree user files, model revisions, or unrelated metadata were copied into the branch.

- [ ] **Step 7: Commit**

```powershell
git add module/auto_rig/__init__.py tests/test_auto_rig_public_api.py docs/superpowers/plans/2026-08-01-auto-rig-foundation-implementation.md
git commit -m "test: gate auto-rig foundation contracts"
```

---

## Follow-On Plans

After this plan is green, write separate executable plans in this order:

1. `auto-rig-e0-moc3-feasibility`: Python parser/golden first, then the opt-in Cubism Core attestation gate.
2. `auto-rig-input-and-geometry`: strict v3 PartSource, A observations, B mesh/weights, overrides.
3. `auto-rig-bindings-and-textures`: C capability/preset registries, candidates, symbol table, deterministic page pixels.
4. `auto-rig-spine-4-2-export`: D exporter and validator.
5. `auto-rig-live2d-runtime-export`: E compiler/writer, coordinate kernel, motions, expressions, release attestation.
6. `auto-rig-batch-integration`: CLI/config, continue-on-error, resume, formal dual-runtime profile, end-to-end fixtures.

This order preserves the accepted dual-format goal while preventing expensive geometry work from outrunning the still-unverified Live2D binary/runtime contract.
