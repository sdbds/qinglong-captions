# Auto-Rig E0 MOC3 Feasibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Core-independent E0 structural gate for Live2D: deterministic frame math, a strict MOC3 V4.00 envelope parser, RFC 8785 attestation hashing, and executable pure vectors.

**Architecture:** Keep coordinate and binary field semantics in two small side-effect-free kernel modules whose normalized source digests can be attested. Put validation, diagnostics, JSON parsing, and vector orchestration outside those kernels. This plan deliberately does not package `live2d-frames-v1.json` or claim E0-core success; that requires an official Cubism Core binary, the SDK harness, and runtime vertex/UV/expression evidence.

**Tech Stack:** Python 3.10 standard library, frozen dataclasses, `struct`, `math`, `hashlib`, pytest.

## Global Constraints

- Follow `docs/superpowers/specs/2026-07-31-auto-rig-from-see-through-layers-design.md` Revision 21.
- Work only on branch `codex/auto-rig-see-through` in the existing isolated worktree.
- Use tests first and observe the expected RED failure before each production implementation.
- Keep `module.auto_rig` import-light; do not add Cubism, Pillow, NumPy, or `py-moc3` as an import-time dependency.
- Do not change the existing manifest JSON encoder. Attestation hashing uses a separate RFC 8785 implementation because float and Unicode-key semantics differ.
- Do not emit a production frame attestation until the Core-dependent E0 matrix has passed.
- Treat StretchyStudio commit `24a83a27ba43e43e9d2e3de5e33994594e6199c2` as evidence, not as an official MOC3 specification.

## File Structure

- `module/auto_rig/jcs.py`: strict RFC 8785 serialization and SHA-256 for attestation payloads only.
- `module/auto_rig/export/__init__.py`: exporter namespace marker.
- `module/auto_rig/export/live2d/__init__.py`: explicit public API for this structural slice.
- `module/auto_rig/export/live2d/frame_kernel.py`: side-effect-free canvas/root and rotation-stack math.
- `module/auto_rig/export/live2d/moc3_layout_kernel.py`: side-effect-free V4.00 binary constants and field decoders.
- `module/auto_rig/export/live2d/moc3.py`: strict envelope validation and immutable parsed records.
- `module/auto_rig/export/live2d/attestation.py`: schema validation, source digest verification, and pure-vector execution.
- `tests/test_auto_rig_jcs.py`: RFC 8785 and I-JSON fixtures.
- `tests/test_auto_rig_live2d_frame_kernel.py`: round-trip, transform order, and conflict fixtures.
- `tests/test_auto_rig_live2d_moc3.py`: synthetic V4.00 envelope and mutation fixtures.
- `tests/test_auto_rig_live2d_attestation.py`: descriptor ordering, source digest, JCS digest, and vector fixtures.

---

### Task 1: RFC 8785 Attestation Encoder

**Files:**
- Create: `module/auto_rig/jcs.py`
- Test: `tests/test_auto_rig_jcs.py`

**Interfaces:**
- Produces: `jcs_bytes(payload: Any) -> bytes`
- Produces: `jcs_sha256(payload: Any) -> str`
- Produces: `JcsContractError(ValueError)`

- [x] **Step 1: Write the failing RFC 8785 tests**

Cover the RFC number vector, UTF-16 property ordering, control escaping, negative-zero normalization, non-finite numbers, unsafe integers, lone surrogates, non-string keys, and non-JSON Python objects.

- [x] **Step 2: Run the focused test and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auto_rig_jcs.py -q`

Expected: collection fails because `module.auto_rig.jcs` does not exist.

- [x] **Step 3: Implement the strict encoder**

Use Python float shortest-round-trip digits, then normalize ECMAScript fixed/exponent thresholds (`1e-6` and `1e21`), exponent signs, and leading zeroes. Sort object keys by UTF-16 code units and serialize non-ASCII text directly as UTF-8.

- [x] **Step 4: Run the focused test and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auto_rig_jcs.py -q`

Expected: all JCS tests pass with no warnings.

---

### Task 2: Live2D Frame Semantic Kernel

**Files:**
- Create: `module/auto_rig/export/__init__.py`
- Create: `module/auto_rig/export/live2d/__init__.py`
- Create: `module/auto_rig/export/live2d/frame_kernel.py`
- Test: `tests/test_auto_rig_live2d_frame_kernel.py`

**Interfaces:**
- Produces: `canvas_to_root(point, width, height) -> tuple[float, float]`
- Produces: `root_to_canvas(point, width, height) -> tuple[float, float]`
- Produces: `apply_similarity(point, origin, angle_degrees, scale) -> tuple[float, float]`
- Produces: `invert_similarity(point, origin, angle_degrees, scale) -> tuple[float, float]`
- Produces: `rotation_stack_rank_conflict(entries) -> bool`
- Produces: `apply_rotation_stack(point, entries) -> tuple[float, float]`
- Produces: `invert_rotation_stack(point, entries) -> tuple[float, float]`

- [x] **Step 1: Write the failing frame tests**

Assert the exact PPU/Y-flip formula, rectangular-canvas round trips, `T(origin) * R(theta) * S(scale)`, lower-rank-outer application, non-commuting order visibility, duplicate-rank detection, and stack inversion.

- [x] **Step 2: Run the focused test and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auto_rig_live2d_frame_kernel.py -q`

Expected: collection fails because the Live2D package and kernel do not exist.

- [x] **Step 3: Implement only pure math**

Represent each stack entry as `(rank, origin_x, origin_y, angle_degrees, scale)`. Sort ascending to obtain outer-to-inner order, apply forward transforms in reverse order, and apply inverses in ascending order. Keep validation prose and orchestration out of this source-digested module.

- [x] **Step 4: Run the focused test and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auto_rig_live2d_frame_kernel.py -q`

Expected: all frame tests pass.

---

### Task 3: Strict MOC3 V4.00 Envelope Parser

**Files:**
- Create: `module/auto_rig/export/live2d/moc3_layout_kernel.py`
- Create: `module/auto_rig/export/live2d/moc3.py`
- Modify: `module/auto_rig/export/live2d/__init__.py`
- Test: `tests/test_auto_rig_live2d_moc3.py`

**Interfaces:**
- Produces: `Moc3EnvelopeError(ValueError)`
- Produces: immutable `Moc3Header`, `Moc3CanvasInfo`, and `Moc3Envelope`
- Produces: `parse_moc3_v400_envelope(payload: bytes) -> Moc3Envelope`
- Produces: `moc3_v400_layout_descriptor() -> dict[str, Any]`

- [x] **Step 1: Write the failing synthetic-envelope tests**

Build a 64-byte-padded V4.00 fixture with header, 160-entry SOT, count block, canvas block, and a shared empty body offset. Mutate magic, version, endian, padding, required SOT entries, offset order/bounds, counts, canvas floats, and final file alignment. Include a legal non-64-byte-aligned body section offset so `csmAlignofMoc=64` cannot be confused with a blanket SOT section rule.

- [x] **Step 2: Run the focused test and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auto_rig_live2d_moc3.py -q`

Expected: collection fails because `moc3.py` does not exist.

- [x] **Step 3: Implement field decoding and outer validation**

Freeze V4.00 as header version byte `3`, little endian, SOT `[0..101]` required, `SOT[0]=1984`, `SOT[1]=2112`, 23 non-negative counts, a 64-byte canvas block, finite positive PPU/width/height, zero padding, and non-zero/in-bounds/nondecreasing required offsets. Keep the Core's 64-byte MOC buffer-base alignment separate from typed section packing; do not claim per-section alignment or body cross-reference validation in this envelope API.

- [x] **Step 4: Run the focused test and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auto_rig_live2d_moc3.py -q`

Expected: all MOC3 envelope tests pass.

---

### Task 4: Live2D Frame Attestation Structural Gate

**Files:**
- Create: `module/auto_rig/export/live2d/attestation.py`
- Modify: `module/auto_rig/export/live2d/__init__.py`
- Test: `tests/test_auto_rig_live2d_attestation.py`

**Interfaces:**
- Produces: `Live2DAttestationError(ValueError)`
- Produces: immutable `Live2DStructuralAttestation`
- Produces: `kernel_source_sha256(source: bytes | str) -> str`
- Produces: `load_live2d_frame_attestation(source: bytes | str) -> Mapping[str, Any]`
- Produces: `validate_live2d_frame_attestation(payload, kernel_sources) -> Live2DStructuralAttestation`

- [x] **Step 1: Write the failing attestation tests**

Construct a complete synthetic attestation with exact top-level and descriptor keys. Store each layout/vector/fixture/invariant body in an actual `payload` object rather than encoded JSON text. Test provenance exclusion from the digest, array ordering and duplicate rejection, invalid nested JSON-string payload rejection, source BOM/newline normalization, source mismatch, descriptor digest mismatch, duplicate JSON object keys, and execution/failure of canvas, similarity, rotation-stack, and conflict pure vectors.

- [x] **Step 2: Run the focused test and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auto_rig_live2d_attestation.py -q`

Expected: collection fails because `attestation.py` does not exist.

- [x] **Step 3: Implement the structural gate**

Require the exact Revision 21 top-level/descriptor field sets, canonical array order, unique IDs, `sha256:<lowercase hex>` digests, exact kernel-source set, RFC 8785 descriptor digest, and deterministic execution of the four supported vector operations. Return only structural evidence; do not expose a release-success flag.

- [x] **Step 4: Run the focused test and verify GREEN**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auto_rig_live2d_attestation.py -q`

Expected: all attestation tests pass.

---

### Task 5: Public Slice Verification And Boundary Check

**Files:**
- Modify: `docs/superpowers/plans/2026-08-01-auto-rig-e0-moc3-feasibility.md`

- [x] **Step 1: Run the new structural suite**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auto_rig_jcs.py tests/test_auto_rig_live2d_frame_kernel.py tests/test_auto_rig_live2d_moc3.py tests/test_auto_rig_live2d_attestation.py -q`

- [x] **Step 2: Run all auto-rig tests**

Run in PowerShell:

```powershell
$tests = Get-ChildItem tests -Filter 'test_auto_rig_*.py' | ForEach-Object { $_.FullName }
.venv\Scripts\python.exe -m pytest @tests -q
```

- [x] **Step 3: Run the see-through regression suite**

Run in PowerShell:

```powershell
$tests = Get-ChildItem tests -Filter 'test_see_through_*.py' | ForEach-Object { $_.FullName }
.venv\Scripts\python.exe -m pytest @tests -q
```

- [x] **Step 4: Run static checks**

Run in PowerShell:

```powershell
$tests = Get-ChildItem tests -Filter 'test_auto_rig_*.py' | ForEach-Object { $_.FullName }
ruff check module/auto_rig @tests
```

Run: `git diff --check`

- [x] **Step 5: Record the hard boundary**

Confirm that no production attestation file exists under `module/auto_rig/export/live2d/attestations/` and no code reports E0-core/release success. Revision 21 subsequently added real-golden/parser evidence and an opt-in Core runtime probe below, but did not close the consistency, writer, renderer, or license gates.

---

## Follow-On Boundary

This plan proves deterministic structural behavior plus a development-only native runtime probe. The next E0 slice must parse all 99 base sections plus `quad_transforms`, implement the minimal nested warp/rotation writer, and run the fixed E0 protocol against an allowlisted Cubism Core binary and SDK/Viewer renderer. Only that slice may create `live2d-frames-v1.json`.

### Post-plan Revision 21 empirical correction

- [x] Add `tests/test_auto_rig_live2d_core.py` first and observe the missing-module RED failure.
- [x] Add `cubism_core.py` with version/API classification and a subprocess-isolated MOC exercise API; add `_cubism_core_worker.py` for aligned revive/initialize/update/vertex evidence.
- [x] Probe the user-supplied x64 DLL: valid Live2D Authenticode signature, SHA-256 `e20e8364850e4c0b726566237855c3b359ad946b10e7f628f4ddad15bbd3730e`, Core `04.02.0002`, latest MOC enum `4`, no `csmHasMocConsistency` export.
- [x] Load the official `CubismWebSamples` `4-r.4` Hiyori V4.00 MOC in the isolated worker: revive/initialize/update succeeds with 70 parameters, 24 parts, 134 drawables and 2,822 finite vertices.
- [x] Run Hiyori/Mark/Rice through the envelope parser and correct the false blanket 64-byte section-alignment rule exposed by Mark's legal SOT offset `2312`.
- [ ] Obtain an official Core `>=04.02.0004`, run consistency plus the writer-generated E0 matrix, and complete the SDK/Viewer render and license gates. The current DLL cannot satisfy this item and must not sign an attestation.
