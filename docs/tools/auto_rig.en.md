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

Formal Live2D delivery requires an installed Cubism SDK for Native. Point the
tool at the SDK root, not at a Core library or validator executable:

```powershell
$env:CUBISM_SDK_ROOT = "E:\CubismSdkForNative-5-r.5"
qinglong-auto-rig E:\path\to\item\final.psd
```

On the first release-tier Live2D export, auto-rig builds the matching native
validator and reuses it by content digest on later jobs and worktrees. The
default cache is `~/.cache/qinglong-captions/runtimes`; set
`QINGLONG_CAPTIONS_RUNTIME_CACHE` to move it. There is no public setting for a
Live2D validator executable.

Windows x86_64 needs CMake 3.20+ and Visual Studio 2022 with the v143 MSVC C++
toolchain. Linux x86_64 needs CMake, Ninja, GCC or Clang, EGL/OpenGL development
libraries, and network access for the pinned GLEW source archive on the first
build. Linux validation requires Mesa's surfaceless EGL platform, uses an EGL
pbuffer, and does not require X11, Wayland, `DISPLAY`, `WAYLAND_DISPLAY`, or
`xvfb`.

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
formal dual-runtime completion marker and never resolves or builds the Live2D
validator. Structural-tier validation likewise does not build it.

Each successful stage is content-addressed and reusable. A repeated command
rehashes declared outputs and owner inventories before skipping work. A failed
item publishes `rig/error.json`; a formal success publishes
`rig/export_manifest.json`.

Toolchain failures are reported under `rig/cache/E/` with a stable code and a
path to the retained configure, build, or probe log. Common codes distinguish a
missing or incomplete SDK, unsupported platform, missing compiler or graphics
dependencies, dependency download failure, build/probe failure, and a missing
runtime attestation for the exact platform/backend/Core/protocol/active-native-source tuple.
