# Auto-Rig E0 Runtime Writer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that this repository can deterministically emit a Cubism MOC3 V4.00 model with nested warp/rotation deformers, load it through an allowlisted official Cubism Core, evaluate parameters, and render the fixed E0 fixture before a production frame attestation is signed.

**Architecture:** Keep the binary layout and coordinate semantics in narrow source-digested kernels. A strict typed codec owns all 99 V4.00 base sections plus `quad_transforms`; an E0 fixture compiler builds only the minimum valid model; the native Core worker runs out of process and returns immutable parameter/drawable evidence. Rendering is a separate opt-in D3D11 harness so Core consistency and geometry evidence do not masquerade as pixel evidence.

**Tech Stack:** Python 3.10 standard library, `ctypes`, `struct`, pytest, Live2D Cubism SDK for Native 5-r.5, Cubism Core `06.00.0001`, Visual Studio 2022/D3D11.

## Global Constraints

- Follow `docs/superpowers/specs/2026-07-31-auto-rig-from-see-through-layers-design.md` Revision 22.
- Work only on `codex/auto-rig-see-through` in the existing isolated worktree.
- Use tests first and observe the expected RED failure before production changes.
- Emit MOC3 header version byte `3` (V4.00), even though the test Core supports newer MOC versions.
- Never bundle the proprietary Cubism Core or SDK in the Python wheel; all native gates are opt-in local probes.
- Core version/floor alone never grants release approval. The attestation pins the exact Core SHA-256, SDK identity, writer kernel digest, and E0 evidence.
- Keep Core calls in a subprocess; malformed writer output must not be able to terminate the batch process.
- No production attestation is written until consistency, default-rest, nested transform, UV/alpha render, motion, expression, and license-policy gates all pass.
- Treat StretchyStudio `24a83a27ba43e43e9d2e3de5e33994594e6199c2` and `py-moc3` `2fb112e` as MIT-licensed implementation evidence, not as official format specifications.

---

### Task 1: Official SDK Candidate And EOF Section Correction

**Files:**
- Modify: `module/auto_rig/export/live2d/moc3_layout_kernel.py`
- Modify: `module/auto_rig/export/live2d/moc3.py`
- Modify: `tests/test_auto_rig_live2d_moc3.py`
- Modify: `docs/superpowers/specs/2026-07-31-auto-rig-from-see-through-layers-design.md`

**Interfaces:**
- Consumes: `parse_moc3_v400_envelope(payload: bytes)`
- Produces: envelope acceptance for zero-length required/unused sections whose SOT offset equals EOF, while offsets beyond EOF remain invalid.

- [x] **Step 1: Add a failing zero-length-at-EOF fixture**

Set every `SOT[2..159]` entry to `file_size` while all 23 counts are zero and assert that the envelope parses.

- [x] **Step 2: Run the focused test and observe RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auto_rig_live2d_moc3.py::test_parse_moc3_v400_envelope_allows_zero_length_sections_at_eof -q`

Expected: `Moc3EnvelopeError` from the old strict `< file_size` rule.

- [x] **Step 3: Permit offsets at EOF in the envelope kernel**

Use `offset <= file_size` for the envelope only and bump the descriptor to `moc3-v400-envelope-layout-v3`. The typed codec remains responsible for rejecting non-empty arrays whose byte extent exceeds EOF.

- [x] **Step 4: Run the complete envelope suite**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auto_rig_live2d_moc3.py -q`

Expected: all non-optional tests pass.

---

### Task 2: Parameterized Core Model Snapshots

**Files:**
- Modify: `module/auto_rig/export/live2d/cubism_core.py`
- Modify: `module/auto_rig/export/live2d/_cubism_core_worker.py`
- Modify: `module/auto_rig/export/live2d/__init__.py`
- Modify: `tests/test_auto_rig_live2d_core.py`

**Interfaces:**
- Consumes: `exercise_moc_with_core(core_path, moc_path, *, parameter_values=None, capture_model_state=False, timeout_seconds=30.0)`
- Produces: immutable `CubismParameterState`, `CubismPartState`, `CubismDrawableState`, and `CubismModelState` attached to `CubismMocRuntimeResult.model_state`.

- [x] **Step 1: Add failing request-validation and official-Hiyori mutation tests**

Reject empty IDs, booleans, non-finite values, and duplicate JSON keys. With configured SDK paths, capture Hiyori at defaults and at `ParamAngleX=30`; assert exact parameter readback, stable drawable IDs/UVs, finite vertices, and a changed vertex-position digest.

- [x] **Step 2: Run the focused tests and observe RED**

Run with `LIVE2D_CUBISM_CORE_PATH` and `LIVE2D_TEST_MOC_PATH` set to SDK 5-r.5: `.venv\Scripts\python.exe -m pytest tests/test_auto_rig_live2d_core.py -q`

Expected: the new keyword arguments/state types do not exist.

- [x] **Step 3: Implement request/result schemas and native bindings**

Write a strict temporary request JSON, bind parameter/part/drawable ID and value APIs, apply requested values before the first `csmUpdateModel`, and return complete snapshots only when requested. Keep the existing summary-only default cheap.

- [x] **Step 4: Run the Core tests against SDK 5-r.5**

Expected: consistency is `True`, default and mutated snapshots load, and an unknown parameter fails in the subprocess with a controlled `CubismCoreError`.

---

### Task 3: Strict Typed MOC3 V4.00 Codec

**Files:**
- Create: `module/auto_rig/export/live2d/moc3_sections_kernel.py`
- Create: `module/auto_rig/export/live2d/moc3_codec.py`
- Modify: `module/auto_rig/export/live2d/__init__.py`
- Create: `tests/test_auto_rig_live2d_moc3_codec.py`

**Interfaces:**
- Produces: `Moc3SectionSpec`, `Moc3V400Document`, `decode_moc3_v400(payload)`, `encode_moc3_v400(document)`, and `Moc3CodecError`.

- [x] **Step 1: Add failing official-golden and mutation tests**

Assert 99 ordered base specs, `SOT[101]` `quad_transforms`, fixed element widths, count-derived array lengths, runtime zero-fill, fixed-width ASCII IDs, exact EOF extents, and deterministic encoding. Decode Hiyori/Mark/Rice and compare Core-visible counts/IDs/UVs.

- [x] **Step 2: Observe RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auto_rig_live2d_moc3_codec.py -q`

- [x] **Step 3: Implement the codec without third-party runtime imports**

Use explicit section specs and `struct`; preserve valid non-64 offsets on decode, while the repository writer emits its own deterministic per-section packing. Validate every section extent from the next SOT offset/file size and its expected count.

- [x] **Step 4: Run codec, envelope, and official-Core tests**

Expected: official V4 goldens decode; encoded fixtures round-trip structurally and pass Core consistency.

---

### Task 4: Minimal Static V4.00 Writer Fixture

**Files:**
- Create: `module/auto_rig/export/live2d/e0_fixture.py`
- Create: `tests/test_auto_rig_live2d_e0_fixture.py`

**Interfaces:**
- Produces: `build_static_e0_moc() -> bytes` with one Part, one asymmetric four-corner ArtMesh, one texture, one parameter, valid binding bands, masks, and draw-order groups.

- [x] **Step 1: Add failing deterministic/static fixture tests**

Assert byte-identical repeated output, V4 header, expected typed sections, parser acceptance, and official Core consistency/init/update with four finite vertices and expected UVs.

- [x] **Step 2: Observe RED**

Run the static E0 test with SDK 5-r.5 configured.

- [x] **Step 3: Implement the smallest valid model**

Use a non-square, four-corner-color-compatible quad so UV origin/V direction cannot pass accidentally by symmetry. Do not add deformers in this task.

- [x] **Step 4: Run static fixture and Core suites**

---

### Task 5: Nested Warp/Rotation E0 Geometry Matrix

**Files:**
- Modify: `module/auto_rig/export/live2d/e0_fixture.py`
- Modify: `tests/test_auto_rig_live2d_e0_fixture.py`
- Modify: `module/auto_rig/export/live2d/frame_kernel.py` only if official runtime evidence requires a versioned semantic correction.

**Interfaces:**
- Produces: `build_deformer_e0_moc()` with `root -> breath warp -> outer rotation -> inner rotation -> ArtMesh`, independent parameters, two-key stops plus explicit defaults when required.

- [x] **Step 1: Add failing Core snapshot matrix tests**

Evaluate defaults, each parameter endpoint, nine intermediate points, angle+origin interpolation, and both non-commuting rotations simultaneously. Compare Core vertices inverse-mapped to canvas against the canonical evaluator within `0.1 px`.

- [x] **Step 2: Observe RED**

- [x] **Step 3: Emit warp/rotation sections and binding graph**

Use one parameter per rigid deformer, lower-rank outer ordering, explicit parent indices, `reflect=false`, and writer-initialized arrays for every V4 section.

- [x] **Step 4: Run consistency/default-rest/geometry tests**

Any mismatch changes the versioned frame descriptor; do not add tolerance until the actual coordinate/sign rule is understood.

---

### Task 6: Official D3D11 Offscreen Render Harness

**Files:**
- Create: `tools/auto_rig_live2d_e0/CMakeLists.txt`
- Create: `tools/auto_rig_live2d_e0/main.cpp`
- Create: `module/auto_rig/export/live2d/cubism_renderer.py`
- Create: `tests/test_auto_rig_live2d_renderer.py`

**Interfaces:**
- Produces: an opt-in executable that loads model3/MOC/PNG/motion/expression, renders to an offscreen RGBA target through the official SDK D3D11 renderer, and writes deterministic JSON plus raw RGBA evidence.

- [x] **Step 1: Add skipped-by-default renderer contract tests**

Require exact SDK path and harness executable. Assert nonblank pixels, four-corner UV orientation, straight-alpha edge behavior, motion visibility, expression overwrite, and Add/Multiply/Overwrite full-weight conformance.

- [x] **Step 2: Build once and observe the missing harness RED**

- [x] **Step 3: Implement a WARP-device offscreen harness**

Use Visual Studio 2022, SDK 5-r.5 Framework, `D3D_DRIVER_TYPE_WARP`, no visible window, no SDK/Core redistribution, and explicit `IsPremultipliedAlpha(false)`.

- [x] **Step 4: Run renderer tests and save only hashes/vectors, not proprietary binaries**

---

### Task 7: Sign The Narrow Frame Attestation

**Files:**
- Modify: `module/auto_rig/export/live2d/attestation.py`
- Create: `module/auto_rig/export/live2d/attestations/live2d-frames-v1.json`
- Modify: `tests/test_auto_rig_live2d_attestation.py`
- Modify: `docs/superpowers/specs/2026-07-31-auto-rig-from-see-through-layers-design.md`

**Interfaces:**
- Consumes: exact kernel source digests, codec layout version, Core/SDK identities, geometry vectors, renderer hashes, and license-policy acknowledgement.
- Produces: the first production `live2d-frames-v1.json` only after every E0 gate passes.

- [x] **Step 1: Add failing real-attestation tests**

Assert the packaged file is JCS-valid, source-digest current, exact-Core allowlisted, all vectors executable, and provenance fields excluded from the semantic digest.

- [x] **Step 2: Observe RED because no production attestation exists**

- [x] **Step 3: Generate the attestation from fresh E0 evidence**

Do not hand-edit digests. Record SDK `5-r.5`, Core `06.00.0001`, SHA-256 `d883c00d114fdf6cef61f439feb23e02d000fdf683e092803010470b80dfaf09`, and the exact harness/compiler versions.

- [x] **Step 4: Run all auto-rig, see-through regression, lint, and diff checks**

Run all `test_auto_rig_*.py`, all `test_see_through_*.py`, `ruff check module/auto_rig tests/test_auto_rig_*.py`, and `git diff --check`.
