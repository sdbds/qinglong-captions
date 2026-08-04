# PSD Auto-Rig Export

Auto-rig consumes one completed see-through item directory, or that item's
`final.psd`, and emits Spine 4.2 and Live2D runtime artifacts under `rig/`.
It is intentionally a separate follow-up command rather than part of the
see-through batch CLI.

Install the geometry dependencies and, when SDPose or DETRPose may be needed,
the pose dependencies:

```powershell
uv sync --extra auto-rig --extra auto-rig-pose
```

The command has one positional argument:

```powershell
qinglong-auto-rig E:\path\to\item\final.psd
```

Formal Live2D delivery requires deployment-provided Core and renderer paths:

```powershell
$env:LIVE2D_CORE_PATH = "E:\CubismSdkForNative-5-r.5\Core\dll\windows\x86_64\Live2DCubismCore.dll"
$env:LIVE2D_RENDERER_PATH = "E:\path\to\auto_rig_live2d_e0.exe"
qinglong-auto-rig E:\path\to\item\final.psd
```

`SPINE_RUNTIME_VALIDATOR_PATH` is optional. Without it, the Spine 4.2 JSON and
atlas still pass their native structural validator and are delivered; the
export report records the external official-runtime gate as `not_run` rather
than pretending it passed. Other deployment settings use
`AUTO_RIG_POSE_MODE`, `AUTO_RIG_POSE_DEVICE`, `AUTO_RIG_SDPOSE_BUNDLE`,
`AUTO_RIG_DETRPOSE_WEIGHTS`, and `QINGLONG_CAPTIONS_MODEL_CACHE`.

For a Spine-only development artifact, no Cubism Core is needed:

```powershell
$env:AUTO_RIG_PROFILE = "spine_4_2_dev"
qinglong-auto-rig E:\path\to\item\final.psd
```

That profile stops after Stage D and reports `stage_validated`; it is not a
formal dual-runtime completion marker.

Each successful stage is content-addressed and reusable. A repeated command
rehashes declared outputs and owner inventories before skipping work. A failed
item publishes `rig/error.json`; a formal success publishes
`rig/export_manifest.json`.
