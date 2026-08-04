# Auto-Rig Generic Expression Implementation Plan

**Goal:** Replace the false-positive procedural facial path with deterministic, exporter-neutral eye/mouth variants, expose a useful expression set in Spine 4.2 and Live2D, and validate declared effects in their anatomical ROI.

**Design source:** `2026-07-31-auto-rig-from-see-through-layers-design.md`, Revision 31.

## Constraints

- Authored `rig_inputs/variants` entries win over generated variants for the same semantic role.
- Generated pixels come only from the validated item snapshot; no Anime2.5DRig sample artwork is copied.
- Stage A owns generated PNGs and `GenericVariantSynthesisPlan`; Stage C remains the only public Rig writer.
- Spine and Live2D consume the same admitted parts, controls, transfers, presets, and symbols.
- A whole-frame SHA change is evidence only. It is never sufficient proof of a facial capability.
- Every behavior change starts with a failing focused test and ends with the real Lucy SDK render.

## Task 1: Freeze Generic Variant Facts

1. Add failing unit tests for deterministic generated eye-closed, mouth-closed, smile, and frown records.
2. Add `generic_variants.py` with a versioned synthesis plan and canonical PNG encoder.
3. Derive eye geometry from sided eyewhite/irides/eyelash union and mouth geometry from the ordinary mouth layer; derive colors from alpha-weighted item pixels.
4. Persist generated PNGs below `rig/cache/A/generated_variants/`, merge them with authored candidates, and include the plan/output inventory in Stage A.
5. Verify byte determinism, authored override precedence, missing-feature degradation, and no path escape.

## Task 2: Add Crossfade Composite Semantics

1. Add failing quality/admission tests for transparent `crossfade_overlay_v1` candidates.
2. Extend candidate/composite schemas without weakening `occluding_overlay_v1` replacement gates.
3. Require complete open/closed target pairs and side-correct anchors; admit the atomic blink and mouth-crossfade bundles before atlas planning.
4. Add open-target opacity bindings and closed-target inverse bindings to the same control.
5. Verify default rest opacity and endpoints in the common Rig validator.

## Task 3: Fix Preset Selection and Expression Coverage

1. Add failing planner tests showing `talk`, `happy`, and `sad` can coexist when their typed targets do not overlap.
2. Delete the hard-coded incompatibility map; keep only primitive-union conflict analysis.
3. Add `unimpressed`, `wink_screen_left`, and `wink_screen_right` to the versioned preset/capability registries.
4. Ensure happy/sad smile/frown targets are independent of talk's open/closed targets.
5. Verify both format plans expose every supported preset with matching IDs and manifest records.

## Task 4: Project to Spine and Live2D

1. Add exporter tests for crossfade opacity timelines/keyforms and all expression files.
2. Update Spine animations and Live2D bindings/model3 groups without exporter-local semantic inference.
3. Validate default setup, parameter endpoints, expression application, exact artifact inventory, and deterministic bytes.

## Task 5: Replace SHA-Only Runtime Validation

1. Add failing tests where a one-pixel mutation changes SHA but must fail semantic validation.
2. Record rest/effect RGBA and canonical face-part ROIs from the public Rig.
3. Implement eye, iris, mouth, and brow metrics with the fixed Revision 31 thresholds.
4. Make required facial effects fail with `runtime_effect_semantics_failed` when the target ROI contract is not met.

## Task 6: Add Minimum Depth-Aware Head Motion

1. Add tests showing equal motion for every head layer is rejected.
2. Bind bounded translation/shear gains from normalized `depth_median`, with face as the reference plane.
3. Verify near/far relative displacement in common Rig, Spine, Live2D parser, and SDK render evidence.

## Task 7: Real Lucy Release Verification

1. Remove stale generated Stage A-E/G owner artifacts through the normal inventory transaction.
2. Run Lucy with `dual_runtime_avatar_v1`, official SDK/Core 5-r.5, and the D3D11 renderer.
3. Render rest, every facial motion/expression, and head controls at 768x768.
4. Assert the Revision 31 ROI metrics, inspect the contact sheet, run focused/full tests, Ruff, compileall, and `git diff --check`.
5. Report exact supported/omitted presets and any residual visual limitation; do not call a hash-only effect fixed.
