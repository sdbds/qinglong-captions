# Auto-Rig SDPose And DETRPose Implementation Plan

**Goal:** Resolve ambiguous see-through limb joints with an optional pose observation provider, using the official 17-point SDPose-Body weights consolidated once into a local single-file bundle and a DETRPose-X CrowdPose comparison backend without letting either model create bones or bypass anatomy validation.

**Architecture:** Stage A first builds its existing geometry-only joint plan. `pose_mode=auto` triggers a provider only when merged limb families still have unresolved hinge joints. Providers share one crop/coordinate contract and return raw typed keypoints; a mapper creates `JointObservation` records, the existing resolver and anatomy gate remain authoritative, and provider/runtime provenance enters Stage A fingerprints. Spine may weight one merged mesh across two resolved pose chains without cutting texture. Live2D only publishes actions that pass its existing runtime and seam gates.

**Tech Stack:** Python 3.11, PyTorch 2.13, Diffusers 0.39, flash-attn 2.8.4, safetensors, Hugging Face Hub, NumPy/SciPy, pytest.

## Frozen Model Contracts

- SDPose source repo: `teemosliang/SDPose-Body@5a34e0c7df4c8ea5fc8774c5f2ae4229e962238c`.
- Required learned sources are the pinned Body17 UNet (`3470311272`, SHA-256 `a75d358808e58cd5eb305dd3362d0d1457d243d787ac4bf1905b64da71d8934a`), VAE (`334643276`, SHA-256 `a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815`) and 17-point decoder (`6986756`, SHA-256 `32994dfc90beb84786c8e9296eeef60dad66980d86d7349be7c7ff80f3aaa8a4`).
- The local `sdpose_body17_fp16.safetensors` is a generated cache artifact, not a fourth-party model source. Its semantic contract pins source digests, configs, tensor prefixes/dtypes, empty-prompt conditioning and converter version; its actual file SHA enters the item fingerprint.
- `Comfy-Org/SDPose` currently provides only the 133-point WholeBody single file. It remains implementation provenance for the fixed-conditioning/single-file layout but is not downloaded or used by the production provider.
- DETRPose repo: `SebasJanampa/DETRPose_X_CROWDPOSE@cebc9cb1ad6289f262262412f604fd03a1d4d6a4`.
- DETRPose file: `model.safetensors`, size `298505628`, SHA-256 `563431b5f20434a1954ba2998f2010d1e960a672996b4a07f2f32ab694e125ee`.
- RT-DETR files in the Comfy repo are bbox detectors, never pose backends.
- ONNX is selected only for a registered pose coherence group with frozen graph I/O and digest. Neither pinned model currently supplies such an ONNX artifact.
- Quantization is deferred.

## Task 1: Artifact And Runtime Resolution

**Files:**
- Create `module/auto_rig/pose/artifacts.py`
- Create `module/auto_rig/pose/runtime.py`
- Modify `utils/transformer_loader.py`
- Test `tests/test_auto_rig_pose_artifacts.py`

- [x] Write RED tests for exact repo/revision/path/size/SHA, RT-DETR rejection, incomplete ONNX rejection, ONNX-over-PyTorch selection, and GBK-safe Hugging Face progress descriptions.
- [x] Reuse `snapshot_download_with_reporting`; do not add a second downloader.
- [x] Resolve `onnx -> torch-fa2 -> torch-sdpa` and record the actually exercised backend, dtype, device and fallback reason.
- [x] Run focused tests and verify the official Body17 source snapshot plus generated bundle.

## Task 2: Provider-Neutral Keypoint Mapping And Trigger

**Files:**
- Create `module/auto_rig/pose/contracts.py`
- Create `module/auto_rig/pose/mapping.py`
- Create `module/auto_rig/pose/selection.py`
- Test `tests/test_auto_rig_pose_mapping.py`

- [x] Write RED fixtures for COCO Body17, CrowdPose 14, image-space side mapping, neck/pelvis derivation and confidence thresholds.
- [x] Write RED fixtures proving `auto` does not load on resolved geometry, triggers on merged unresolved elbows/knees, and explicit provider modes are deterministic.
- [x] Filter raw points through the existing observation-anatomy validator; never clamp invalid points or create absent anatomy.
- [x] Persist a comparison report whose scores use only common joints and calibrated geometry metrics, not raw provider confidence.

## Task 3: Official Body17 Consolidator And Provider

**Files:**
- Create `module/auto_rig/pose/sdpose.py`
- Create `module/auto_rig/pose/heatmap.py`
- Add a small pinned null-conditioning asset or generated constant module with provenance and SHA
- Test `tests/test_auto_rig_sdpose.py`

- [x] Write RED source-contract, bundle-schema, tensor-shape, checkpoint-prefix, heatmap decode, crop round-trip and fixed-conditioning digest tests.
- [x] Download only the pinned Body17 source components, verify every byte contract, convert learned floating tensors to FP16, prefix them as `unet.`, `vae.` and `decoder.`, and atomically emit one self-describing local safetensors bundle.
- [x] Load VAE, UNet and the 320-channel/17-point head strictly from that bundle and reject missing/unexpected learned tensors.
- [x] Implement the deterministic single `t=999` x0 forward and capture the final 320-channel up-block feature.
- [x] Install a Diffusers-compatible FA2 processor and exercise both self/cross attention; fall back to SDPA on a typed failure.
- [x] Audit the pinned upstream release for an official Body17 golden fixture. It publishes weights/configs but no canonical image-plus-keypoint fixture, so parity is recorded as unavailable rather than fabricated from Lucy2; independent decoder tests and real inference remain separate evidence.

## Task 4: DETRPose-X CrowdPose Adapter

**Files:**
- Create `module/auto_rig/pose/detrpose.py`
- Update `pyproject.toml` with a separate comparison extra if required
- Test `tests/test_auto_rig_detrpose.py`

- [x] Write RED tests for the CrowdPose 14-point order and model/person selection.
- [x] Load the pinned HF config/weights through the inference-only DETRPose implementation without importing its visualizer or saving images.
- [x] Return raw keypoints in the same canvas contract as SDPose and record the complete model/runtime fingerprint.
- [x] Keep this backend optional; failure cannot silently relabel RT-DETR as DETRPose.

## Task 5: Stage A And Merged-Limb Skinning Integration

**Files:**
- Modify `module/auto_rig/pipeline.py`
- Modify `module/auto_rig/joint_pipeline.py`
- Modify `module/auto_rig/skinning.py`
- Modify public API/tests

- [x] Write RED pipeline tests with an injected fake provider, proving provider output changes Stage A/B fingerprints and resume identities.
- [x] Add `pose_mode`, provider injection/local artifact options and comparison report paths without importing torch on pose-disabled jobs.
- [x] For a merged unsided limb mesh with two resolved chains, choose the nearest chain per vertex and compute existing arc-length weights; do not split texture or component topology. Side-specific distal parts instead choose one chain per component so crossed limbs cannot tear one shoe across both chains.
- [x] Keep Live2D joint-bend omissions honest unless its sampled-deform/runtime gates pass.

## Task 6: Real Lucy2 Comparison And Verification

- [x] Run geometry-only, SDPose-Body17 and DETRPose-X on `E:/Code/qinglong-captions/datasets/321/outputs/lucy2.jpg`.
- [x] Save raw provider results, accepted/rejected observations, overlay images and common-joint comparison metrics under an experimental F-owned report path.
- [x] Run the full Stage A-E/G release pipeline with the selected provider and verify emitted limb bones, crossed-distal component binding, Spine weighted animation and Live2D capability claims.
- [x] Run focused pytest and every auto-rig test file in bounded responsibility groups (the monolithic invocation exceeds the 60-minute command limit), plus Ruff, compileall and `git diff --check` before declaring completion.
