# Live2D Validator Auto-Build Design

**Status:** Approved design direction

**Date:** 2026-08-05

**Parent specification:** `2026-07-31-auto-rig-from-see-through-layers-design.md` Revision 41

## Goal

Make the Live2D native release validator an internal, lazily built production dependency instead of a user-selected executable. The same public workflow must support Windows x86_64 and headless Linux x86_64, reuse a content-addressed build across jobs and worktrees, and preserve the existing official-runtime release gate.

## Scope

This change owns:

- removing the validator executable path from the GUI, see-through CLI, persisted configuration, and public auto-rig CLI environment contract;
- resolving an installed Cubism SDK for Native root without redistributing the proprietary SDK;
- lazily building and caching a platform-specific validator when Stage E first needs release-tier validation;
- Windows D3D11 WARP and Linux headless EGL/OpenGL rendering backends;
- platform-specific runtime attestation, cache invalidation, concurrency, failure diagnostics, and tests.

This change does not:

- download or redistribute Cubism Core or the Cubism SDK;
- install system packages with `sudo`, `apt`, `dnf`, `pacman`, or another package manager;
- require a validator for Spine-only or structural-only output;
- add macOS or Linux arm64 release support. Linux arm64 remains unsupported because the SDK marks that Core build experimental;
- make the validator part of exported Live2D or Spine assets.

## Product Contract

The normal GUI must not show a Live2D validator executable path. It may show one Cubism SDK directory selector because the SDK cannot be downloaded or redistributed by this project. The label is `Cubism SDK for Native directory`, and the selected directory must contain `Core`, `Framework`, and `cubism-info.yml`.

The public see-through CLI accepts `--auto_rig_sdk_root`. The standalone auto-rig CLI accepts `CUBISM_SDK_ROOT`. `--auto_rig_renderer_path`, `auto_rig_renderer_path`, and `LIVE2D_RENDERER_PATH` are removed from the public contract. Low-level validator functions may still accept an executable path as an internal dependency-injection boundary for tests and attestation generation.

SDK resolution uses this deterministic precedence:

1. explicit `auto_rig_sdk_root` / `--auto_rig_sdk_root`;
2. `CUBISM_SDK_ROOT`;
3. a legacy `LIVE2D_CORE_PATH`, converted to an SDK root only when its normalized suffix exactly matches the supported platform Core layout;
4. supported platform discovery roots, sorted by normalized absolute path and then selecting the highest parsed SDK release.

Windows discovery checks drive roots for `CubismSdkForNative-*`. Linux discovery checks `~/CubismSdkForNative-*`, `~/.local/share/CubismSdkForNative-*`, and `/opt/CubismSdkForNative-*`. Discovery never scans the whole filesystem. Candidates are ordered by parsed SDK release descending, then normalized absolute path ascending, so equal-version selection is deterministic.

## Runtime Manager

Create `module/auto_rig/export/live2d/runtime_toolchain.py` as the only owner of SDK discovery and validator construction.

Its public boundary is:

```python
@dataclass(frozen=True, slots=True)
class Live2DRuntimeToolchain:
    sdk_root: Path
    core_path: Path
    validator_path: Path
    platform_id: Literal["windows-x86_64", "linux-x86_64"]
    backend_id: Literal["d3d11-warp", "opengl-egl-headless"]
    cache_key: str
    core_sha256: str
    validator_sha256: str


def ensure_live2d_runtime_toolchain(
    *,
    sdk_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Live2DRuntimeToolchain:
    ...
```

Stage E calls this function only when all of the following are true:

- the selected profile requests Live2D;
- `validation_tier == "release"`;
- Stage E has reached runtime validation after Stage D has already committed its Spine artifacts.

This ordering ensures a missing compiler, SDK, or EGL stack cannot prevent valid Spine output from being produced. It may prevent the dual-runtime item from receiving a completed terminal marker.

## Stable Cache

The default cache root is:

```text
QINGLONG_CAPTIONS_RUNTIME_CACHE, when set
otherwise ~/.cache/qinglong-captions/runtimes
```

The validator lives at:

```text
<cache-root>/live2d-validator/<platform-id>/<cache-key>/bin/<platform executable>
```

The cache key is SHA-256 over canonical JSON containing exactly:

- runtime builder schema version;
- validator protocol version;
- platform ID, architecture, and backend ID;
- validator CMake and all validator source file SHA-256 values;
- Cubism Core binary/static-library SHA-256 values used by that platform;
- the SHA-256 inventory of Framework `CMakeLists.txt`, `.cpp`, `.c`, `.hpp`, `.h`, and shader files;
- `cubism-info.yml` SHA-256;
- CMake version, generator ID, C/C++ compiler identity, and compiler version;
- pinned GLFW and GLEW source release identifiers and archive SHA-256 values on Linux.

Absolute repository, SDK, worktree, and cache paths do not enter the key. Moving an identical SDK or checking out another worktree must reuse the same build.

Each completed cache entry contains `runtime-manifest.json`. Reuse requires:

- the manifest schema and cache key match;
- the declared executable and required shader files exist;
- every declared output byte hash matches disk;
- the validator reports the expected protocol and backend during a smoke probe.

Any mismatch invalidates the entry. The manager builds into a sibling temporary directory and atomically renames it into the final cache path only after all validation succeeds.

## Concurrency And Recovery

A cross-platform standard-library lock guards each cache key. Windows uses `msvcrt.locking`; Linux uses `fcntl.flock`. The lock file is outside the final immutable cache entry.

Waiters re-check the cache after taking the lock. Only the lock owner configures and builds. A killed process can leave a temporary build directory, but it cannot leave a valid manifest in the final entry. Temporary directories older than 24 hours may be removed while holding the same lock.

Build logs are retained as `configure.log`, `build.log`, and `probe.log` in a failed-attempt diagnostics directory. Formal item diagnostics report a stable error code and the log path, not the entire compiler output.

## Platform Backends

### Windows x86_64

Windows retains the existing official Framework D3D11 renderer and WARP software device. The build uses the SDK's `Core/lib/windows/x86_64/143/Live2DCubismCore_MD.lib`, Framework sources, D3D11, D3DCompiler, WIC, and the Framework D3D11 shaders.

The builder requires CMake 3.20 or newer and MSVC toolset 141, 142, or 143. It selects the SDK static library matching the active compiler toolset and rejects an unsupported or mismatched toolset before configure. It invokes configure and build non-interactively and produces `qinglong_live2d_validator.exe`.

### Linux x86_64

Linux uses the SDK's stable `Core/lib/linux/x86_64/libLive2DCubismCore.a` and official Framework OpenGL renderer. It creates a headless OpenGL context through EGL with GLFW 3.4 Null Platform, so neither X11, Wayland, `DISPLAY`, nor `xvfb` is required.

The build pins GLFW 3.4 and GLEW 2.2.0 by release archive and SHA-256. Dependencies are fetched into the build cache, never into the SDK tree. CMake disables GLFW X11 and Wayland backends. The runtime initializes `GLFW_PLATFORM_NULL`, requests `GLFW_EGL_CONTEXT_API`, creates an invisible pbuffer-backed context, initializes GLEW, and renders into an explicit framebuffer object.

Textures are decoded with the SDK sample's `stb_image.h`. Readback rows are flipped into the canonical top-left RGBA convention before evidence is written. The report backend is `opengl-egl-headless` and includes the OpenGL vendor, renderer, and version for diagnostics.

The Linux builder requires:

- x86_64 Linux;
- CMake 3.20 or newer;
- GCC 11 or newer or Clang 14 or newer;
- EGL and OpenGL development libraries;
- `dl`, pthreads, and a working EGL software or hardware implementation.

Missing prerequisites cause an actionable error. The project does not invoke a privileged package manager. On a headless production host, Mesa llvmpipe is an acceptable renderer.

## Native Source Layout

Refactor `tools/auto_rig_live2d_e0` without duplicating the protocol logic:

```text
tools/auto_rig_live2d_e0/
  CMakeLists.txt
  common/
    validator_common.cpp
    validator_common.hpp
  windows/
    main_d3d11.cpp
  linux/
    main_egl.cpp
```

The common unit owns argument parsing, model and animation loading, parameter application, report schema, alpha summaries, and protocol constants. Each backend owns context/device creation, PNG upload, draw submission, RGBA readback, and backend diagnostics.

The executable name is platform-neutral at the source level: `qinglong_live2d_validator` plus the platform's normal executable suffix.

## Attestation

The frame contract remains shared, but runtime rendering evidence becomes platform-specific. The packaged attestation contains one record per `(platform_id, architecture, backend_id, Core SHA-256, validator protocol)` tuple.

Windows D3D11 WARP and Linux EGL/OpenGL may produce different RGBA hashes. Each backend receives its own E0 fixture hashes and alpha bounds. Cross-platform acceptance compares semantic invariants, not byte-identical rendered pixels:

- Core consistency passes;
- default state matches rest state within 0.1 canvas pixel;
- requested parameter values are observed within `1e-6`;
- required motions and expressions cause the expected regional changes;
- expression and blink clearing restores the baseline;
- UV orientation and straight-alpha edge fixtures pass;
- every render is finite and has nonzero alpha.

Changing common protocol code or a platform backend invalidates only the affected runtime attestation record plus the shared protocol digest when applicable. A Linux build is not formally releasable until its exact Core and backend tuple has passed E0 on Linux.

## Pipeline And Fingerprints

Public pipeline entry points accept `sdk_root`, not `renderer_path`. Stage E resolves the toolchain internally. The Stage E fingerprint records:

- platform and backend IDs;
- Core SHA-256;
- validator SHA-256;
- runtime cache key;
- applicable attestation record digest.

It does not record the absolute SDK, cache, or executable path. Identical toolchains in different worktrees therefore preserve resume behavior.

Structural validation and Spine Stage D fingerprints do not depend on the Live2D runtime toolchain.

## GUI Behavior

The auto-rig sub-options show:

- capability profile;
- pose model, device, and FlashAttention selection;
- Cubism SDK for Native directory when the selected profile includes Live2D;
- optional Spine runtime validator configuration until it receives a separate auto-build design.

The GUI does not show, persist, or pass a Live2D validator executable. Starting a job does not synchronously build in the browser event handler. Stage E emits progress messages for SDK resolution, cache hit, configure, build, and probe through the existing process log.

## Failure Contract

Stable failures are:

- `live2d_sdk_not_found`: no supported SDK root was resolved;
- `live2d_sdk_layout_invalid`: required Core, Framework, metadata, or shader files are missing;
- `live2d_validator_platform_unsupported`: OS or architecture is outside Windows/Linux x86_64;
- `live2d_validator_build_tool_missing`: CMake or a supported compiler is absent;
- `live2d_validator_graphics_dependency_missing`: Linux EGL/OpenGL prerequisites are absent;
- `live2d_validator_dependency_fetch_failed`: a pinned GLFW or GLEW archive could not be downloaded or failed its SHA-256 check;
- `live2d_validator_build_failed`: configure or compilation failed;
- `live2d_validator_probe_failed`: the completed executable failed its protocol/backend smoke probe;
- `live2d_runtime_attestation_missing`: the exact platform/Core/backend tuple has no signed E0 record.

These failures occur at Stage E. Existing Spine output remains independently available, but a required dual-runtime profile cannot receive `completed` status.

## Verification

Unit tests must cover:

- the GUI and emitted see-through command contain no validator path;
- SDK resolution precedence and exact legacy Core-to-root conversion on Windows and Linux;
- platform/architecture rejection;
- cache keys are path-independent and change for source, Framework, Core, compiler, protocol, or dependency changes;
- valid cache reuse performs no configure/build command;
- corrupt outputs, missing shaders, and protocol probe mismatches rebuild;
- two concurrent callers produce one committed cache entry;
- failed builds never publish a reusable entry and preserve diagnostic logs;
- Stage D can commit before a Stage E toolchain failure;
- Stage E fingerprints are stable when an identical cache is moved;
- Linux command planning disables X11/Wayland and selects the EGL Null Platform backend.

Integration gates must cover:

- Windows: clean-cache build against `CubismSdkForNative-5-r.5`, E0 attestation, and one real PSD release validation;
- Linux x86_64: clean-cache build without `DISPLAY` or `WAYLAND_DISPLAY`, EGL smoke probe, E0 attestation, and the same fixture bundle;
- warm-cache rerun on both platforms performs no native rebuild;
- deleting the source worktree does not invalidate or remove the user cache entry.

Windows verification can run locally. Linux verification runs in WSL2 or a native Linux runner with the same official SDK package mounted into Linux. A Windows-only unit test that merely synthesizes `sys.platform == "linux"` is not sufficient evidence for Linux support.

## Rejected Alternatives

**Prebuilt validator downloads:** rejected because binaries bind to Core/Framework versions, weaken content attestation, and may redistribute proprietary SDK code.

**Python-only renderer:** rejected because it would reproduce Live2D runtime semantics in the same implementation family as the writer instead of exercising the official Framework.

**Hidden GLFW X11/Wayland window:** rejected because production batch workers may have no display server and `xvfb` would become an undeclared runtime dependency.

**Build at GUI startup:** rejected because users who never request formal Live2D output should not pay compiler, network, or cache costs.
