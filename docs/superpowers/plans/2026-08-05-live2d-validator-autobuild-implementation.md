# Live2D Validator Auto-Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the user-selected Live2D validator executable and lazily build a reusable official-SDK validator on Windows x86_64 and headless Linux x86_64.

**Architecture:** A new Python runtime-toolchain manager resolves the user-installed Cubism SDK, derives the platform Core and renderer backend, builds into a content-addressed per-user cache under a cross-process lock, and returns immutable Core/validator identities to Stage E. The native validator shares protocol/model logic while selecting D3D11 WARP on Windows and GLFW Null Platform plus EGL/OpenGL on Linux. Stage D remains independent and Stage E alone owns toolchain failures.

**Tech Stack:** Python 3.10+, pytest, CMake 3.20+, C++17, Cubism SDK for Native 5-r.5, D3D11/WIC on Windows, EGL/OpenGL/GLEW/GLFW on Linux.

## Global Constraints

- The GUI, persisted config, public see-through CLI, and standalone auto-rig CLI must not accept or display a validator executable path.
- The project must not download or redistribute Cubism Core or the Cubism SDK.
- Windows x86_64 uses D3D11 WARP; Linux x86_64 uses EGL plus GLFW Null Platform and must run with `DISPLAY` and `WAYLAND_DISPLAY` unset.
- Linux arm64 and macOS remain unsupported by this change.
- The validator is built lazily only for release-tier Live2D Stage E and is never copied into exported model artifacts.
- A valid Spine Stage D transaction must remain available when toolchain construction or Live2D Stage E fails.
- The cache key must be content-based and must not include repository, worktree, SDK, cache, or executable absolute paths.
- Linux dependency pins are GLFW 3.4 archive SHA-256 `c038d34200234d071fae9345bc455e4a8f2f544ab60150765d7704e08f3dac01` and GLEW 2.2.0 archive SHA-256 `f781d57097cdd076c6e34656d3aae239abaa03da7fd60e2249ee29df546e3d1e`.
- No production code is written before its focused regression test has been observed failing for the intended reason.
- Existing uncommitted GUI auto-rig work must be edited in place and must not be reverted.

---

### Task 1: Replace The Public Validator Path With The SDK Root

**Files:**
- Modify: `config/model.toml`
- Modify: `gui/utils/i18n.py`
- Modify: `gui/wizard/step6_tools.py`
- Modify: `module/see_through/cli.py`
- Modify: `module/see_through/runner.py`
- Modify: `module/auto_rig/cli.py`
- Modify: `tests/test_see_through_cli.py`
- Modify: `tests/test_see_through_config.py`
- Modify: `tests/test_see_through_fingerprint.py`
- Modify: `tests/test_see_through_runner.py`
- Modify: `tests/test_see_through_tools_step.py`
- Modify: `tests/test_auto_rig_cli.py`

**Interfaces:**
- Consumes: existing GUI auto-rig sub-options and `SeeThroughRunConfig`.
- Produces: `auto_rig_sdk_root: Path | None`, CLI flag `--auto_rig_sdk_root`, environment variable `CUBISM_SDK_ROOT`; removes the public renderer path.

- [ ] **Step 1: Write failing public-contract tests**

Update the CLI and GUI assertions to require one SDK directory and forbid every validator path:

```python
def test_build_run_config_maps_auto_rig_sdk_root(tmp_path):
    sdk_root = tmp_path / "CubismSdkForNative-5-r.5"
    args = build_parser().parse_args([
        "--input_dir=foo",
        "--auto_rig",
        f"--auto_rig_sdk_root={sdk_root}",
    ])
    config = build_run_config(args)
    assert config.auto_rig_sdk_root == sdk_root
    assert not hasattr(config, "auto_rig_renderer_path")


def test_tools_step_never_emits_live2d_validator_path(captured, step, sdk_root):
    assert f"--auto_rig_sdk_root={sdk_root}" in captured["args"]
    assert not any("renderer_path" in argument for argument in captured["args"])
    assert "see_through_auto_rig_renderer_path" not in step.config
```

Add a config-schema assertion that `[see_through]` has `auto_rig_sdk_root` and lacks `auto_rig_core_path` and `auto_rig_renderer_path`. Update `tests/test_auto_rig_cli.py` so `CUBISM_SDK_ROOT` is forwarded and `LIVE2D_RENDERER_PATH` is ignored.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m pytest tests/test_see_through_cli.py tests/test_see_through_config.py tests/test_see_through_tools_step.py tests/test_auto_rig_cli.py -q
```

Expected: failures mention missing `auto_rig_sdk_root` and existing renderer-path fields/arguments.

- [ ] **Step 3: Implement the minimal public-contract change**

Change `SeeThroughRunConfig` to:

```python
auto_rig_sdk_root: Path | None = None
```

Remove `auto_rig_core_path` and `auto_rig_renderer_path` from the public config. Replace the two Live2D file selectors with a single directory selector labeled `Cubism SDK for Native directory`. Keep the optional Spine validator field unchanged. Pass `sdk_root` through `_run_auto_rig_phase` and the standalone CLI. Preserve `LIVE2D_CORE_PATH` only inside the future SDK resolver as a migration input, not as a public GUI or CLI output.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Commit the public contract**

```powershell
git add config/model.toml gui/utils/i18n.py gui/wizard/step6_tools.py module/see_through/cli.py module/see_through/runner.py module/auto_rig/cli.py tests/test_see_through_cli.py tests/test_see_through_config.py tests/test_see_through_fingerprint.py tests/test_see_through_runner.py tests/test_see_through_tools_step.py tests/test_auto_rig_cli.py
git commit -m "feat: hide Live2D validator behind SDK configuration"
```

---

### Task 2: Implement SDK Discovery And Content-Addressed Build Planning

**Files:**
- Create: `module/auto_rig/export/live2d/runtime_toolchain.py`
- Create: `tests/test_auto_rig_live2d_runtime_toolchain.py`
- Modify: `module/auto_rig/export/live2d/__init__.py`
- Modify: `module/auto_rig/__init__.py`

**Interfaces:**
- Consumes: an optional SDK root, environment mapping, platform/architecture facts, repository validator sources, and compiler probe results.
- Produces: `Live2DRuntimeBuildPlan`, `Live2DRuntimeToolchain`, `resolve_cubism_sdk_root()`, and stable `Live2DRuntimeToolchainError.code` values.

- [ ] **Step 1: Write failing discovery and cache-key tests**

Create fixtures with minimal SDK layouts for both platforms and assert:

```python
def test_resolve_sdk_prefers_explicit_over_environment(tmp_path):
    explicit = make_sdk(tmp_path / "explicit", platform_id="windows-x86_64")
    other = make_sdk(tmp_path / "environment", platform_id="windows-x86_64")
    assert resolve_cubism_sdk_root(
        explicit,
        environ={"CUBISM_SDK_ROOT": str(other)},
        platform_id="windows-x86_64",
        discovery_roots=(),
    ) == explicit.resolve()


def test_build_plan_cache_key_is_path_independent(tmp_path):
    left = make_identical_sdk(tmp_path / "left")
    right = make_identical_sdk(tmp_path / "right")
    assert build_live2d_runtime_plan(left, facts=FACTS).cache_key == build_live2d_runtime_plan(right, facts=FACTS).cache_key


def test_build_plan_cache_key_changes_for_semantic_input(tmp_path):
    sdk = make_sdk(tmp_path / "sdk", platform_id="windows-x86_64")
    baseline = build_live2d_runtime_plan(sdk, facts=FACTS)
    (sdk / "Framework" / "src" / "Model" / "CubismModel.cpp").write_text("changed", encoding="utf-8")
    changed = build_live2d_runtime_plan(sdk, facts=FACTS)
    assert changed.cache_key != baseline.cache_key
```

Also assert exact legacy Core suffix conversion, deterministic discovery ties, Linux arm64 rejection, and no absolute paths in the canonical key payload.

- [ ] **Step 2: Run the new tests and verify RED**

```powershell
python -m pytest tests/test_auto_rig_live2d_runtime_toolchain.py -q
```

Expected: import failure for the new module.

- [ ] **Step 3: Implement immutable plans and deterministic hashing**

Define:

```python
@dataclass(frozen=True, slots=True)
class Live2DRuntimeBuildPlan:
    sdk_root: Path
    core_path: Path
    core_link_path: Path
    platform_id: Literal["windows-x86_64", "linux-x86_64"]
    backend_id: Literal["d3d11-warp", "opengl-egl-headless"]
    cache_root: Path
    cache_key: str
    entry_root: Path
    executable_path: Path
    configure_command: tuple[str, ...]
    build_command: tuple[str, ...]
    identity_payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class Live2DRuntimeToolchain:
    sdk_root: Path
    core_path: Path
    validator_path: Path
    platform_id: str
    backend_id: str
    cache_key: str
    core_sha256: str
    validator_sha256: str

    def fingerprint_payload(self, *, attestation_record_sha256: str) -> Mapping[str, str]:
        return {
            "platform_id": self.platform_id,
            "backend_id": self.backend_id,
            "core_sha256": self.core_sha256,
            "validator_sha256": self.validator_sha256,
            "runtime_cache_key": self.cache_key,
            "attestation_record_sha256": attestation_record_sha256,
        }


class Live2DRuntimeToolchainError(RuntimeError):
    def __init__(self, code: str, message: str, *, log_path: Path | None = None):
        self.code = code
        self.log_path = log_path
        super().__init__(f"{code}: {message}")
```

Use canonical sorted JSON and streamed SHA-256. Hash the exact validator source inventory, supported Framework source/shader inventory, `cubism-info.yml`, Core runtime/static libraries, backend dependency pins, and compiler/CMake identities. Default the cache to `QINGLONG_CAPTIONS_RUNTIME_CACHE` or `~/.cache/qinglong-captions/runtimes`.

- [ ] **Step 4: Run the new tests and verify GREEN**

```powershell
python -m pytest tests/test_auto_rig_live2d_runtime_toolchain.py -q
```

- [ ] **Step 5: Commit discovery and planning**

```powershell
git add module/auto_rig/export/live2d/runtime_toolchain.py module/auto_rig/export/live2d/__init__.py module/auto_rig/__init__.py tests/test_auto_rig_live2d_runtime_toolchain.py
git commit -m "feat: plan content-addressed Live2D validator builds"
```

---

### Task 3: Refactor The Windows Native Harness Into A Shared Protocol

**Files:**
- Delete: `tools/auto_rig_live2d_e0/main.cpp`
- Create: `tools/auto_rig_live2d_e0/common/validator_common.hpp`
- Create: `tools/auto_rig_live2d_e0/common/validator_common.cpp`
- Create: `tools/auto_rig_live2d_e0/windows/main_d3d11.cpp`
- Modify: `tools/auto_rig_live2d_e0/CMakeLists.txt`
- Modify: `module/auto_rig/export/live2d/cubism_renderer.py`
- Modify: `tests/test_auto_rig_live2d_renderer.py`

**Interfaces:**
- Consumes: existing validator CLI arguments and Cubism Framework.
- Produces: `qinglong_live2d_validator`, report schema `auto-rig-live2d-render-v2`, and a side-effect-free `--probe-report <path>` command.

- [ ] **Step 1: Write failing source/protocol tests**

Require the platform split and protocol probe:

```python
def test_native_validator_has_shared_and_platform_sources():
    root = ROOT / "tools" / "auto_rig_live2d_e0"
    assert (root / "common" / "validator_common.cpp").is_file()
    assert (root / "windows" / "main_d3d11.cpp").is_file()
    assert not (root / "main.cpp").exists()


def test_renderer_accepts_attested_backend_and_v2_diagnostics(tmp_path, fake_harness):
    evidence = render_moc_with_offscreen_harness(
        fake_harness.executable,
        fake_harness.moc_path,
        fake_harness.texture_path,
        expected_backend="d3d11-warp",
    )
    assert evidence.driver_type == "d3d11-warp"
    assert evidence.runtime_info["api"] == "d3d11"
```

Add a real probe invocation test guarded by `LIVE2D_E0_RENDERER_PATH` until the new binary is built.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
python -m pytest tests/test_auto_rig_live2d_renderer.py -q
```

- [ ] **Step 3: Split common model logic and retain D3D11 behavior**

Move parsing, file I/O, Framework lifetime, MOC/model creation, parameter/motion/expression application, alpha summary, JSON reporting, and protocol constants into `validator_common`. Keep WIC texture upload, D3D11 WARP creation, shader setup, draw, and staging readback in `main_d3d11.cpp`.

The probe report contains exactly:

```json
{
  "backend_id": "d3d11-warp",
  "protocol_digest": LIVE2D_E0_VALIDATOR_PROTOCOL_DIGEST,
  "schema_version": "auto-rig-live2d-probe-v1"
}
```

The render report v2 adds `runtime_info` while retaining all v1 semantic fields. `render_moc_with_offscreen_harness()` requires an explicit expected backend and rejects a mismatch.

- [ ] **Step 4: Build and run the Windows validator**

```powershell
cmake -S tools/auto_rig_live2d_e0 -B .cache/live2d-validator-windows-check -DCUBISM_SDK_ROOT=E:/CubismSdkForNative-5-r.5
cmake --build .cache/live2d-validator-windows-check --config Release --parallel
python -m pytest tests/test_auto_rig_live2d_renderer.py -q
```

Expected: CMake/build exit 0 and renderer tests pass with the new executable supplied through the integration-test environment.

- [ ] **Step 5: Commit the Windows protocol refactor**

```powershell
git add tools/auto_rig_live2d_e0 module/auto_rig/export/live2d/cubism_renderer.py tests/test_auto_rig_live2d_renderer.py
git commit -m "refactor: share Live2D validator runtime protocol"
```

---

### Task 4: Add The Linux Headless EGL Backend

**Files:**
- Create: `tools/auto_rig_live2d_e0/linux/main_egl.cpp`
- Modify: `tools/auto_rig_live2d_e0/CMakeLists.txt`
- Modify: `tests/test_auto_rig_live2d_renderer.py`
- Modify: `tests/test_auto_rig_live2d_runtime_toolchain.py`

**Interfaces:**
- Consumes: shared validator protocol, Cubism Framework OpenGL renderer, SDK `stb_image.h`, EGL/OpenGL, pinned GLFW 3.4 and GLEW 2.2.0.
- Produces: backend `opengl-egl-headless` with canonical top-left straight-alpha RGBA evidence.

- [ ] **Step 1: Write failing Linux backend contract tests**

Assert the CMake and source prohibit display-system coupling:

```python
def test_linux_backend_is_null_platform_egl():
    source = (TOOLS / "linux" / "main_egl.cpp").read_text("utf-8")
    cmake = (TOOLS / "CMakeLists.txt").read_text("utf-8")
    assert "GLFW_PLATFORM_NULL" in source
    assert "GLFW_EGL_CONTEXT_API" in source
    assert "GLFW_BUILD_X11 OFF" in cmake
    assert "GLFW_BUILD_WAYLAND OFF" in cmake
    assert "opengl-egl-headless" in source
    assert "DISPLAY" not in source
```

Add a Linux-only integration test that unsets `DISPLAY` and `WAYLAND_DISPLAY`, invokes `--probe-report`, and renders the static orientation fixture.

- [ ] **Step 2: Run source tests and verify RED**

```powershell
python -m pytest tests/test_auto_rig_live2d_renderer.py tests/test_auto_rig_live2d_runtime_toolchain.py -q
```

- [ ] **Step 3: Implement the EGL/OpenGL renderer**

On Linux, CMake imports `Core/lib/linux/x86_64/libLive2DCubismCore.a`, selects `FRAMEWORK_SOURCE OpenGL`, defines `CSM_TARGET_LINUX_GL`, and builds pinned GLFW/GLEW inside the cache. `main_egl.cpp` must:

```cpp
glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_NULL);
glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
```

Create an explicit RGBA8 framebuffer, upload straight-alpha PNG pages through `stb_image`, bind them to `CubismRenderer_OpenGLES2`, draw, `glReadPixels`, flip rows, and emit OpenGL vendor/renderer/version diagnostics.

- [ ] **Step 4: Build and test inside WSL2 without a display**

Run from WSL2 after installing the documented non-project system prerequisites:

```bash
unset DISPLAY WAYLAND_DISPLAY
cmake -S tools/auto_rig_live2d_e0 -B ~/.cache/qinglong-captions/check/live2d-egl \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUBISM_SDK_ROOT=/mnt/e/CubismSdkForNative-5-r.5
cmake --build ~/.cache/qinglong-captions/check/live2d-egl --parallel
LIVE2D_E0_RENDERER_PATH=~/.cache/qinglong-captions/check/live2d-egl/qinglong_live2d_validator \
LIVE2D_CORE_PATH=/mnt/e/CubismSdkForNative-5-r.5/Core/dll/linux/x86_64/libLive2DCubismCore.so \
python -m pytest tests/test_auto_rig_live2d_renderer.py -q
```

Expected: build and test exit 0 while both display variables are unset.

- [ ] **Step 5: Commit the Linux backend**

```powershell
git add tools/auto_rig_live2d_e0 tests/test_auto_rig_live2d_renderer.py tests/test_auto_rig_live2d_runtime_toolchain.py
git commit -m "feat: add headless EGL Live2D validation"
```

---

### Task 5: Build, Lock, Probe, And Reuse Runtime Cache Entries

**Files:**
- Modify: `module/auto_rig/export/live2d/runtime_toolchain.py`
- Modify: `tests/test_auto_rig_live2d_runtime_toolchain.py`

**Interfaces:**
- Consumes: `Live2DRuntimeBuildPlan` and the native `--probe-report` protocol.
- Produces: `ensure_live2d_runtime_toolchain()` with atomic content-addressed reuse.

- [ ] **Step 1: Write failing cache lifecycle tests**

Cover cache hits, corruption, locking, and failed builds:

```python
def test_valid_cache_hit_runs_no_build_command(tmp_path, prepared_entry, refusing_runner):
    toolchain = ensure_live2d_runtime_toolchain(
        sdk_root=prepared_entry.sdk_root,
        cache_root=tmp_path / "cache",
        _command_runner=refusing_runner,
    )
    assert toolchain.validator_sha256 == prepared_entry.validator_sha256


def test_two_callers_publish_one_entry(tmp_path, sdk_root, counting_runner):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda _: ensure_live2d_runtime_toolchain(
                sdk_root=sdk_root,
                cache_root=tmp_path / "cache",
                _command_runner=counting_runner,
            ),
            range(2),
        ))
    assert results[0] == results[1]
    assert counting_runner.build_count == 1


def test_failed_build_never_publishes_manifest(tmp_path, sdk_root, failing_runner):
    with pytest.raises(Live2DRuntimeToolchainError) as error:
        ensure_live2d_runtime_toolchain(
            sdk_root=sdk_root,
            cache_root=tmp_path / "cache",
            _command_runner=failing_runner,
        )
    assert error.value.code == "live2d_validator_build_failed"
    assert not plan.entry_root.exists()
    assert error.value.log_path.is_file()
```

Use an injected command runner only at the subprocess boundary; exercise real hashing, locking, staging, manifest validation, and atomic rename logic.

- [ ] **Step 2: Run cache tests and verify RED**

```powershell
python -m pytest tests/test_auto_rig_live2d_runtime_toolchain.py -q
```

- [ ] **Step 3: Implement the cache transaction**

Implement `msvcrt.locking`/`fcntl.flock`, waiter recheck, 24-hour staging cleanup under lock, configure/build log capture, probe validation, canonical `runtime-manifest.json`, output rehash on every reuse, and final directory rename. Map configure, dependency fetch, graphics dependency, compiler, build, and probe failures to the exact design codes.

- [ ] **Step 4: Run cache tests and warm-cache integration**

```powershell
python -m pytest tests/test_auto_rig_live2d_runtime_toolchain.py -q
```

Then call `ensure_live2d_runtime_toolchain()` twice against `E:\CubismSdkForNative-5-r.5` and assert the second call does not change the executable or manifest timestamps.

- [ ] **Step 5: Commit runtime cache management**

```powershell
git add module/auto_rig/export/live2d/runtime_toolchain.py tests/test_auto_rig_live2d_runtime_toolchain.py
git commit -m "feat: cache native Live2D validator builds"
```

---

### Task 6: Make Runtime Attestation Platform-Specific

**Files:**
- Modify: `module/auto_rig/export/live2d/attestation.py`
- Modify: `module/auto_rig/export/live2d/e0_attestation.py`
- Modify: `module/auto_rig/export/live2d/release_validator.py`
- Modify: `module/auto_rig/export/live2d/attestations/live2d-frames-v1.json`
- Modify: `tests/test_auto_rig_live2d_attestation.py`
- Modify: `tests/test_auto_rig_live2d_release_validator.py`

**Interfaces:**
- Consumes: exact platform, backend, Core, validator protocol, and E0 fixture evidence.
- Produces: `Live2DRuntimeAttestation`, `select_runtime_attestation()`, and one signed runtime record per Windows D3D11 or Linux EGL tuple.

The immutable selection result is:

```python
@dataclass(frozen=True, slots=True)
class Live2DRuntimeAttestation:
    platform_id: str
    backend_id: str
    core_sha256: str
    validator_protocol_digest: str
    record_sha256: str
    payload: Mapping[str, object]
```

- [ ] **Step 1: Write failing schema and selection tests**

```python
def test_attestation_selects_exact_runtime_tuple():
    record = select_runtime_attestation(
        payload,
        platform_id="linux-x86_64",
        backend_id="opengl-egl-headless",
        core_sha256=LINUX_CORE_SHA256,
        validator_protocol_digest=PROTOCOL,
    )
    assert record["backend_id"] == "opengl-egl-headless"


def test_windows_record_cannot_attest_linux_backend():
    with pytest.raises(Live2DAttestationError, match="exact runtime tuple"):
        select_runtime_attestation(
            payload,
            platform_id="linux-x86_64",
            backend_id="opengl-egl-headless",
            core_sha256=WINDOWS_CORE_SHA256,
            validator_protocol_digest=PROTOCOL,
        )
```

Require runtime records to have unique tuple keys and backend-specific E0 evidence.

- [ ] **Step 2: Run attestation tests and verify RED**

```powershell
python -m pytest tests/test_auto_rig_live2d_attestation.py tests/test_auto_rig_live2d_release_validator.py -q
```

- [ ] **Step 3: Implement tuple selection and backend-aware generation**

Move `approved_core_binaries` and renderer fixture hashes into canonical `runtime_attestations[]`. Keep shared coordinate/layout kernels outside the records. Update `generate_live2d_e0_attestation()` to merge or replace exactly one runtime tuple without changing another backend's record.

- [ ] **Step 4: Regenerate and verify Windows and Linux E0 records**

Run the generator against the newly cached Windows validator and official Windows DLL. In WSL2, run it against the cached EGL validator and Linux `.so`. Merge both exact records, then run:

```powershell
python -m pytest tests/test_auto_rig_live2d_attestation.py tests/test_auto_rig_live2d_release_validator.py -q
```

Expected: both runtime records validate and a cross-platform mismatch is rejected.

- [ ] **Step 5: Commit platform attestations**

```powershell
git add module/auto_rig/export/live2d/attestation.py module/auto_rig/export/live2d/e0_attestation.py module/auto_rig/export/live2d/release_validator.py module/auto_rig/export/live2d/attestations/live2d-frames-v1.json tests/test_auto_rig_live2d_attestation.py tests/test_auto_rig_live2d_release_validator.py
git commit -m "feat: attest Live2D runtimes per platform backend"
```

---

### Task 7: Integrate Lazy Toolchain Resolution Into Stage E

**Files:**
- Modify: `module/auto_rig/stage_e.py`
- Modify: `module/auto_rig/pipeline.py`
- Modify: `module/auto_rig/export/live2d/release_validator.py`
- Modify: `tests/test_auto_rig_stage_e.py`
- Modify: `tests/test_auto_rig_pipeline.py`
- Modify: `tests/test_auto_rig_stage_g.py`
- Modify: `tests/test_see_through_runner.py`

**Interfaces:**
- Consumes: `sdk_root`, `ensure_live2d_runtime_toolchain()`, and `select_runtime_attestation()`.
- Produces: Stage E fingerprints based on toolchain and attestation identities rather than paths, while Stage D commits before any toolchain resolution.

- [ ] **Step 1: Write failing pipeline-order and fingerprint tests**

```python
def test_stage_d_commits_before_live2d_toolchain_failure(monkeypatch, item_root):
    def fail_toolchain(**_kwargs):
        raise Live2DRuntimeToolchainError("live2d_sdk_not_found", "SDK missing")

    monkeypatch.setattr(pipeline, "ensure_live2d_runtime_toolchain", fail_toolchain)
    with pytest.raises(StageEError, match="live2d_sdk_not_found"):
        run_auto_rig_item(
            item_root,
            profile_id="dual_runtime_core_v1",
            sdk_root=item_root / "missing-sdk",
            pose_mode="disabled",
        )
    assert (item_root / "rig/spine/skeleton.json").is_file()
    assert (item_root / "rig/cache/D/manifest.json").is_file()
    failure = json.loads((item_root / "rig/cache/E/failure.json").read_text(encoding="utf-8"))
    assert failure["diagnostics"][0]["code"] == "live2d_sdk_not_found"


def test_stage_e_fingerprint_ignores_toolchain_absolute_paths(toolchain, tmp_path):
    left = replace(
        toolchain,
        sdk_root=tmp_path / "left-sdk",
        core_path=tmp_path / "left-sdk/Core/core",
        validator_path=tmp_path / "left-cache/validator",
    )
    right = replace(
        toolchain,
        sdk_root=tmp_path / "right-sdk",
        core_path=tmp_path / "right-sdk/Core/core",
        validator_path=tmp_path / "right-cache/validator",
    )
    assert left.fingerprint_payload(attestation_record_sha256=ATTESTATION_SHA) == right.fingerprint_payload(
        attestation_record_sha256=ATTESTATION_SHA
    )
```

Update runner tests to assert it passes `sdk_root` only.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
python -m pytest tests/test_auto_rig_stage_e.py tests/test_auto_rig_pipeline.py tests/test_auto_rig_stage_g.py tests/test_see_through_runner.py -q
```

- [ ] **Step 3: Resolve after Stage D and map failures in Stage E**

Replace the `core_path` and `renderer_path` keyword parameters on `run_auto_rig_item()` with `sdk_root: str | Path | None = None`. After Stage D commits, resolve the toolchain and exact attestation record before constructing the Stage E fingerprint. Pass the immutable toolchain and attestation into `execute_stage_e()`; do not rebuild or reselect inside the writer transaction. When resolution fails, call a new `publish_stage_e_toolchain_failure(root, error)` helper so `rig/cache/E/failure.json` retains the stable toolchain code before the exception is re-raised.

Convert `Live2DRuntimeToolchainError` to `StageEError` without losing its stable code or diagnostic log path. Build the E config fingerprint with:

```python
toolchain.fingerprint_payload(
    attestation_record_sha256=runtime_attestation.record_sha256,
)
```

Do not include `sdk_root`, `core_path`, or `validator_path`. Structural tier uses `{"runtime_toolchain": "not-required"}` and never resolves the SDK.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2.

- [ ] **Step 5: Commit pipeline integration**

```powershell
git add module/auto_rig/stage_e.py module/auto_rig/pipeline.py module/auto_rig/export/live2d/release_validator.py tests/test_auto_rig_stage_e.py tests/test_auto_rig_pipeline.py tests/test_auto_rig_stage_g.py tests/test_see_through_runner.py
git commit -m "feat: resolve Live2D validator lazily after Spine"
```

---

### Task 8: Run Release Regression And Document Operations

**Files:**
- Modify: `docs/tools/auto_rig.en.md`
- Modify: `docs/superpowers/specs/2026-07-31-auto-rig-from-see-through-layers-design.md`
- Modify: `docs/superpowers/specs/2026-08-05-live2d-validator-autobuild-design.md`

**Interfaces:**
- Consumes: completed Windows/Linux toolchains and the real upstream PSD fixtures.
- Produces: verified operating instructions and release evidence.

- [ ] **Step 1: Run the complete targeted Python suite**

```powershell
python -m pytest tests/test_auto_rig_live2d_runtime_toolchain.py tests/test_auto_rig_live2d_renderer.py tests/test_auto_rig_live2d_attestation.py tests/test_auto_rig_live2d_release_validator.py tests/test_auto_rig_stage_e.py tests/test_auto_rig_pipeline.py tests/test_auto_rig_stage_g.py tests/test_see_through_cli.py tests/test_see_through_runner.py tests/test_see_through_tools_step.py -q
```

- [ ] **Step 2: Run a clean-cache Windows production item**

Use `E:\CubismSdkForNative-5-r.5` and a previously accepted real PSD fixture. Delete only the dedicated new runtime cache entry, run formal auto-rig, record build/cache paths, and verify the exported Live2D model in the same runtime gate used by Stage E.

- [ ] **Step 3: Run warm-cache and worktree-independence checks**

Run the same item again and verify no configure/build command is emitted. Invoke from the main checkout with identical source content or copy the source identity fixture to a second worktree and verify the cache entry is reused by digest, not path.

- [ ] **Step 4: Run WSL2 headless release validation**

Unset both display variables, run the E0 and release fixture through the Linux cache, and capture the backend diagnostics proving `opengl-egl-headless`. Do not claim Linux support if this step is skipped or only the Python planning tests pass.

- [ ] **Step 5: Update documentation and spec revision notes**

Document `CUBISM_SDK_ROOT`, cache location, first-build prerequisites, failure codes, cache invalidation, and the fact that no validator executable is user-configurable. Update the parent spec revision table with the supplementary design decisions.

- [ ] **Step 6: Run formatting, static checks, and full relevant regression**

```powershell
python -m compileall module/auto_rig module/see_through gui/wizard/step6_tools.py
python -m ruff check module/auto_rig module/see_through gui/wizard/step6_tools.py tests/test_auto_rig_live2d_runtime_toolchain.py tests/test_auto_rig_live2d_renderer.py
git diff --check
```

Then run the repository's established auto-rig and see-through test selection. Record the exact pass/fail count.

- [ ] **Step 7: Commit documentation and final verification fixes**

```powershell
git add docs/tools/auto_rig.en.md docs/superpowers/specs/2026-07-31-auto-rig-from-see-through-layers-design.md docs/superpowers/specs/2026-08-05-live2d-validator-autobuild-design.md
git commit -m "docs: document automatic Live2D validator builds"
```
