from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from module.auto_rig.export.live2d.runtime_toolchain import (
    DEFAULT_LINUX_DEPENDENCY_PINS,
    Live2DRuntimeBuildFacts,
    Live2DRuntimeBuildPlan,
    Live2DRuntimeToolchain,
    Live2DRuntimeToolchainError,
    _default_build_facts,
    _windows_discovery_roots,
    build_live2d_runtime_plan,
    default_live2d_runtime_cache_root,
    ensure_live2d_runtime_toolchain,
    live2d_validator_source_inventory,
    live2d_validator_source_sha256,
    resolve_cubism_sdk_root,
)


def _write(path: Path, payload: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload, encoding="utf-8")
    return path


def _make_source(root: Path) -> Path:
    source = root / "tools" / "auto_rig_live2d_e0"
    _write(source / "CMakeLists.txt", "project(qinglong_live2d_validator)\n")
    _write(source / "common" / "validator_common.cpp", "int common_validator() { return 0; }\n")
    _write(source / "common" / "validator_common.hpp", "int common_validator();\n")
    _write(source / "windows" / "main_d3d11.cpp", "int main() { return 0; }\n")
    _write(source / "linux" / "main_egl.cpp", "int main() { return 0; }\n")
    return source


def test_validator_source_identity_normalizes_checkout_line_endings(tmp_path: Path) -> None:
    relative_payloads = {
        "CMakeLists.txt": b"line one\nline two\n",
        "common/validator_common.cpp": b"common one\ncommon two\n",
        "common/validator_common.hpp": b"header one\nheader two\n",
        "windows/main_d3d11.cpp": b"windows one\nwindows two\n",
    }
    roots = []
    for directory, newline in (("lf", b"\n"), ("crlf", b"\r\n")):
        root = tmp_path / directory
        for relative_path, payload in relative_payloads.items():
            _write(root / relative_path, payload.replace(b"\n", newline))
        roots.append(root)

    assert live2d_validator_source_inventory(
        roots[0],
        platform_id="windows-x86_64",
    ) == live2d_validator_source_inventory(
        roots[1],
        platform_id="windows-x86_64",
    )
    assert live2d_validator_source_sha256(
        roots[0],
        platform_id="windows-x86_64",
    ) == live2d_validator_source_sha256(
        roots[1],
        platform_id="windows-x86_64",
    )


def _make_sdk(root: Path, *, platform_id: str, marker: str = "same") -> Path:
    _write(root / "cubism-info.yml", "id: 12\nversion: 5-r.5\n")
    _write(root / "Core" / "include" / "Live2DCubismCore.h", f"header:{marker}\n")
    _write(root / "Framework" / "CMakeLists.txt", f"framework:{marker}\n")
    _write(root / "Framework" / "src" / "Model" / "CubismModel.cpp", f"model:{marker}\n")
    _write(root / "Framework" / "src" / "Rendering" / "shader.txt", f"shader:{marker}\n")
    if platform_id == "windows-x86_64":
        _write(root / "Core" / "dll" / "windows" / "x86_64" / "Live2DCubismCore.dll", b"core-runtime")
        _write(
            root / "Core" / "lib" / "windows" / "x86_64" / "143" / "Live2DCubismCore_MD.lib",
            b"core-link",
        )
    elif platform_id == "linux-x86_64":
        _write(root / "Core" / "dll" / "linux" / "x86_64" / "libLive2DCubismCore.so", b"core-runtime")
        _write(root / "Core" / "lib" / "linux" / "x86_64" / "libLive2DCubismCore.a", b"core-link")
        _write(root / "Samples" / "OpenGL" / "thirdParty" / "stb" / "stb_image.h", b"stb-image")
    else:
        raise AssertionError(platform_id)
    return root


def _facts(tmp_path: Path, *, platform_id: str = "windows-x86_64") -> Live2DRuntimeBuildFacts:
    return Live2DRuntimeBuildFacts(
        platform_id=platform_id,
        validator_source_root=_make_source(tmp_path / "repository"),
        cmake_identity="cmake 3.30.0",
        compiler_identity="compiler 19.40",
        generator_identity=("Visual Studio 17 2022:x64" if platform_id == "windows-x86_64" else "Ninja:1.12"),
        protocol_digest="sha256:" + "1" * 64,
        dependency_pins=DEFAULT_LINUX_DEPENDENCY_PINS if platform_id == "linux-x86_64" else {},
    )


class _FakeRuntimeRunner:
    def __init__(
        self,
        *,
        platform_id: str = "windows-x86_64",
        fail_phase: str | None = None,
        probe_backend: str | None = None,
        build_delay: float = 0.0,
    ) -> None:
        self.platform_id = platform_id
        self.fail_phase = fail_phase
        self.probe_backend = probe_backend
        self.build_delay = build_delay
        self.configure_count = 0
        self.build_count = 0
        self.probe_count = 0
        self._guard = threading.Lock()

    def __call__(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, timeout_seconds
        phase: str
        if "--build" in command:
            phase = "build"
            with self._guard:
                self.build_count += 1
            if self.build_delay:
                time.sleep(self.build_delay)
            build_root = Path(command[command.index("--build") + 1])
            bin_root = build_root.parent / "bin"
            executable = (
                "qinglong_live2d_validator.exe"
                if self.platform_id == "windows-x86_64"
                else "qinglong_live2d_validator"
            )
            _write(bin_root / executable, b"validator-binary")
            _write(bin_root / "FrameworkShaders" / "fixture.shader", b"shader")
        elif len(command) >= 2 and command[1] == "--probe-report":
            phase = "probe"
            with self._guard:
                self.probe_count += 1
            backend = self.probe_backend or (
                "d3d11-warp" if self.platform_id == "windows-x86_64" else "opengl-egl-headless"
            )
            _write(
                Path(command[2]),
                json.dumps(
                    {
                        "backend_id": backend,
                        "protocol_digest": "sha256:" + "1" * 64,
                        "schema_version": "auto-rig-live2d-probe-v1",
                    }
                ),
            )
        else:
            phase = "configure"
            with self._guard:
                self.configure_count += 1
        if self.fail_phase == phase:
            return subprocess.CompletedProcess(command, 1, "", f"synthetic {phase} failure")
        return subprocess.CompletedProcess(command, 0, f"synthetic {phase} success", "")


def _ensure_fake_runtime(
    tmp_path: Path,
    runner: _FakeRuntimeRunner,
    *,
    platform_id: str = "windows-x86_64",
):
    sdk = _make_sdk(tmp_path / "sdk", platform_id=platform_id)
    facts = _facts(tmp_path, platform_id=platform_id)
    return ensure_live2d_runtime_toolchain(
        sdk_root=sdk,
        cache_root=tmp_path / "cache",
        _facts=facts,
        _command_runner=runner,
    )


def test_resolve_sdk_prefers_explicit_over_environment(tmp_path: Path) -> None:
    explicit = _make_sdk(tmp_path / "explicit", platform_id="windows-x86_64")
    other = _make_sdk(tmp_path / "environment", platform_id="windows-x86_64")

    actual = resolve_cubism_sdk_root(
        explicit,
        environ={"CUBISM_SDK_ROOT": str(other)},
        platform_id="windows-x86_64",
        discovery_roots=(),
    )

    assert actual == explicit.resolve()


@pytest.mark.parametrize(
    ("platform_id", "relative_core"),
    [
        ("windows-x86_64", Path("Core/dll/windows/x86_64/Live2DCubismCore.dll")),
        ("linux-x86_64", Path("Core/dll/linux/x86_64/libLive2DCubismCore.so")),
    ],
)
def test_resolve_sdk_converts_only_exact_legacy_core_suffix(
    tmp_path: Path,
    platform_id: str,
    relative_core: Path,
) -> None:
    sdk = _make_sdk(tmp_path / "sdk", platform_id=platform_id)

    assert resolve_cubism_sdk_root(
        None,
        environ={"LIVE2D_CORE_PATH": str(sdk / relative_core)},
        platform_id=platform_id,
        discovery_roots=(),
    ) == sdk.resolve()

    with pytest.raises(Live2DRuntimeToolchainError) as error:
        resolve_cubism_sdk_root(
            None,
            environ={"LIVE2D_CORE_PATH": str(sdk / "Core" / relative_core.name)},
            platform_id=platform_id,
            discovery_roots=(),
        )
    assert error.value.code == "live2d_sdk_not_found"


def test_resolve_sdk_discovery_is_deterministic_for_equal_versions(tmp_path: Path) -> None:
    discovery = tmp_path / "sdks"
    expected = _make_sdk(discovery / "CubismSdkForNative-5-r.5-a", platform_id="windows-x86_64")
    _make_sdk(discovery / "CubismSdkForNative-5-r.5-b", platform_id="windows-x86_64")

    actual = resolve_cubism_sdk_root(
        None,
        environ={},
        platform_id="windows-x86_64",
        discovery_roots=(discovery,),
    )

    assert actual == expected.resolve()


def test_windows_sdk_discovery_enumerates_every_logical_drive() -> None:
    roots = _windows_discovery_roots((1 << 2) | (1 << 4), current_drive="Z:")

    assert roots == (Path("C:\\"), Path("E:\\"))
    assert _windows_discovery_roots(0, current_drive="Z:") == (Path("Z:\\"),)


def test_resolve_sdk_rejects_explicit_invalid_layout_without_fallback(tmp_path: Path) -> None:
    fallback = _make_sdk(tmp_path / "fallback", platform_id="windows-x86_64")
    invalid = tmp_path / "invalid"
    invalid.mkdir()

    with pytest.raises(Live2DRuntimeToolchainError) as error:
        resolve_cubism_sdk_root(
            invalid,
            environ={"CUBISM_SDK_ROOT": str(fallback)},
            platform_id="windows-x86_64",
            discovery_roots=(),
        )

    assert error.value.code == "live2d_sdk_layout_invalid"


def test_default_cache_root_uses_environment_then_home(tmp_path: Path) -> None:
    configured = tmp_path / "configured"
    assert default_live2d_runtime_cache_root(
        environ={"QINGLONG_CAPTIONS_RUNTIME_CACHE": str(configured)},
        home=tmp_path / "home",
    ) == configured.resolve()
    assert default_live2d_runtime_cache_root(environ={}, home=tmp_path / "home") == (
        tmp_path / "home" / ".cache" / "qinglong-captions" / "runtimes"
    ).resolve()


def test_linux_dependency_registry_uses_direct_content_addressed_archives() -> None:
    assert DEFAULT_LINUX_DEPENDENCY_PINS["glew_url"] == (
        "https://downloads.sourceforge.net/project/glew/glew/2.2.0/glew-2.2.0.tgz"
    )
    assert DEFAULT_LINUX_DEPENDENCY_PINS["glew_sha256"] == (
        "d4fc82893cfb00109578d0a1a2337fb8ca335b3ceccf97b97e5cc7f08e4353e1"
    )
    assert not any(key.startswith("glfw_") for key in DEFAULT_LINUX_DEPENDENCY_PINS)


@pytest.mark.parametrize(
    ("platform_id", "backend_id", "executable_name"),
    [
        ("windows-x86_64", "d3d11-warp", "qinglong_live2d_validator.exe"),
        ("linux-x86_64", "opengl-egl-headless", "qinglong_live2d_validator"),
    ],
)
def test_build_plan_selects_platform_layout(
    tmp_path: Path,
    platform_id: str,
    backend_id: str,
    executable_name: str,
) -> None:
    sdk = _make_sdk(tmp_path / "sdk", platform_id=platform_id)
    plan = build_live2d_runtime_plan(
        sdk,
        facts=_facts(tmp_path, platform_id=platform_id),
        cache_root=tmp_path / "cache",
    )

    assert plan.platform_id == platform_id
    assert plan.backend_id == backend_id
    assert plan.executable_path.name == executable_name
    assert plan.core_path.is_file()
    assert plan.core_link_path.is_file()
    generator_index = plan.configure_command.index("-G") + 1
    assert plan.configure_command[generator_index] == (
        "Visual Studio 17 2022" if platform_id == "windows-x86_64" else "Ninja"
    )


def test_windows_build_tree_uses_short_process_temp_namespace(tmp_path: Path) -> None:
    sdk = _make_sdk(tmp_path / "sdk", platform_id="windows-x86_64")
    plan = build_live2d_runtime_plan(
        sdk,
        facts=_facts(tmp_path),
        cache_root=tmp_path / "deliberately-long-runtime-cache-directory",
    )
    build_root = Path(plan.configure_command[plan.configure_command.index("-B") + 1])

    assert ".staging" not in build_root.parts
    assert plan.transient_root.parent.name == "ql2d"
    assert len(plan.transient_root.name) == 43
    assert len(str(build_root)) < len(str(plan.entry_root))


def test_build_plan_rejects_linux_arm64_before_reading_sdk(tmp_path: Path) -> None:
    facts = replace(_facts(tmp_path), platform_id="linux-arm64")
    with pytest.raises(Live2DRuntimeToolchainError) as error:
        build_live2d_runtime_plan(tmp_path / "missing", facts=facts, cache_root=tmp_path / "cache")
    assert error.value.code == "live2d_platform_unsupported"


def test_linux_build_plan_requires_and_hashes_sdk_stb_image(tmp_path: Path) -> None:
    sdk = _make_sdk(tmp_path / "sdk", platform_id="linux-x86_64")
    facts = _facts(tmp_path, platform_id="linux-x86_64")
    baseline = build_live2d_runtime_plan(sdk, facts=facts, cache_root=tmp_path / "cache")

    stb_path = sdk / "Samples" / "OpenGL" / "thirdParty" / "stb" / "stb_image.h"
    stb_path.write_bytes(b"changed-stb-image")
    changed = build_live2d_runtime_plan(sdk, facts=facts, cache_root=tmp_path / "cache")
    assert changed.cache_key != baseline.cache_key

    stb_path.unlink()
    with pytest.raises(Live2DRuntimeToolchainError) as error:
        build_live2d_runtime_plan(sdk, facts=facts, cache_root=tmp_path / "cache")
    assert error.value.code == "live2d_sdk_layout_invalid"


def test_build_plan_cache_key_is_path_independent(tmp_path: Path) -> None:
    facts = _facts(tmp_path)
    left = _make_sdk(tmp_path / "left", platform_id="windows-x86_64")
    right = _make_sdk(tmp_path / "right", platform_id="windows-x86_64")

    left_plan = build_live2d_runtime_plan(left, facts=facts, cache_root=tmp_path / "cache-left")
    right_plan = build_live2d_runtime_plan(right, facts=facts, cache_root=tmp_path / "cache-right")

    assert left_plan.cache_key == right_plan.cache_key
    assert left_plan.identity_payload == right_plan.identity_payload


def test_build_plan_cache_key_only_hashes_native_inputs_for_active_backend(tmp_path: Path) -> None:
    sdk = _make_sdk(tmp_path / "sdk", platform_id="windows-x86_64")
    facts = _facts(tmp_path, platform_id="windows-x86_64")
    baseline = build_live2d_runtime_plan(sdk, facts=facts, cache_root=tmp_path / "cache")

    _write(facts.validator_source_root / "generate_attestation.py", "print('changed helper')\n")
    _write(facts.validator_source_root / "__pycache__" / "helper.pyc", b"generated-bytecode")
    _write(facts.validator_source_root / "linux" / "main_egl.cpp", "changed inactive backend")
    irrelevant = build_live2d_runtime_plan(sdk, facts=facts, cache_root=tmp_path / "cache")

    assert irrelevant.cache_key == baseline.cache_key
    assert irrelevant.identity_payload == baseline.identity_payload
    serialized = json.dumps(irrelevant.identity_payload, sort_keys=True)
    assert "generate_attestation.py" not in serialized
    assert "__pycache__" not in serialized
    assert "linux/main_egl.cpp" not in serialized

    _write(facts.validator_source_root / "windows" / "main_d3d11.cpp", "changed active backend")
    active_backend_changed = build_live2d_runtime_plan(
        sdk,
        facts=facts,
        cache_root=tmp_path / "cache",
    )
    assert active_backend_changed.cache_key != baseline.cache_key


def test_linux_default_facts_pin_cmake_to_the_compilers_whose_identity_is_hashed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    compiler = str(Path(sys.executable).resolve())
    monkeypatch.setenv("CC", compiler)
    monkeypatch.setenv("CXX", compiler)
    facts = replace(
        _default_build_facts("linux-x86_64"),
        validator_source_root=_make_source(tmp_path / "repository"),
        cmake_identity="cmake-test",
        generator_identity="Ninja:test",
    )
    sdk = _make_sdk(tmp_path / "sdk", platform_id="linux-x86_64")
    plan = build_live2d_runtime_plan(sdk, facts=facts, cache_root=tmp_path / "cache")

    assert facts.c_compiler_executable == compiler
    assert facts.cxx_compiler_executable == compiler
    assert f"-DCMAKE_C_COMPILER={compiler}" in plan.configure_command
    assert f"-DCMAKE_CXX_COMPILER={compiler}" in plan.configure_command
    assert compiler not in json.dumps(plan.identity_payload, sort_keys=True)


@pytest.mark.parametrize(
    "semantic_change",
    ["core", "framework", "validator", "compiler", "cmake", "generator", "protocol", "dependency"],
)
def test_build_plan_cache_key_changes_for_every_semantic_input(
    tmp_path: Path,
    semantic_change: str,
) -> None:
    platform_id = "linux-x86_64" if semantic_change == "dependency" else "windows-x86_64"
    sdk = _make_sdk(tmp_path / "sdk", platform_id=platform_id)
    facts = _facts(tmp_path, platform_id=platform_id)
    baseline = build_live2d_runtime_plan(sdk, facts=facts, cache_root=tmp_path / "cache")

    if semantic_change == "core":
        baseline.core_path.write_bytes(b"changed-core")
    elif semantic_change == "framework":
        _write(sdk / "Framework" / "src" / "Model" / "CubismModel.cpp", "changed")
    elif semantic_change == "validator":
        _write(facts.validator_source_root / "common" / "validator_common.cpp", "changed")
    elif semantic_change == "compiler":
        facts = replace(facts, compiler_identity="compiler changed")
    elif semantic_change == "cmake":
        facts = replace(facts, cmake_identity="cmake changed")
    elif semantic_change == "generator":
        facts = replace(facts, generator_identity="generator changed")
    elif semantic_change == "protocol":
        facts = replace(facts, protocol_digest="sha256:" + "2" * 64)
    elif semantic_change == "dependency":
        facts = replace(facts, dependency_pins={**facts.dependency_pins, "glfw_sha256": "changed"})

    changed = build_live2d_runtime_plan(sdk, facts=facts, cache_root=tmp_path / "cache")
    assert changed.cache_key != baseline.cache_key


def test_build_plan_identity_payload_contains_no_absolute_paths(tmp_path: Path) -> None:
    sdk = _make_sdk(tmp_path / "sdk", platform_id="windows-x86_64")
    plan = build_live2d_runtime_plan(
        sdk,
        facts=_facts(tmp_path),
        cache_root=tmp_path / "runtime-cache",
    )
    serialized = json.dumps(plan.identity_payload, sort_keys=True)

    assert str(tmp_path.resolve()) not in serialized
    assert str(sdk.resolve()) not in serialized
    assert "validator/CMakeLists.txt" in serialized
    assert "sdk/Core/runtime" in serialized


def test_toolchain_fingerprint_excludes_all_absolute_paths(tmp_path: Path) -> None:
    left = Live2DRuntimeToolchain(
        sdk_root=tmp_path / "one-sdk",
        core_path=tmp_path / "one-sdk" / "core.dll",
        validator_path=tmp_path / "one-cache" / "validator.exe",
        platform_id="windows-x86_64",
        backend_id="d3d11-warp",
        cache_key="a" * 64,
        core_sha256="sha256:" + "b" * 64,
        validator_sha256="sha256:" + "c" * 64,
        validator_source_sha256="sha256:" + "e" * 64,
    )
    right = replace(
        left,
        sdk_root=tmp_path / "two-sdk",
        core_path=tmp_path / "two-sdk" / "core.dll",
        validator_path=tmp_path / "two-cache" / "validator.exe",
    )

    expected = left.fingerprint_payload(attestation_record_sha256="sha256:" + "d" * 64)
    assert right.fingerprint_payload(attestation_record_sha256="sha256:" + "d" * 64) == expected
    assert str(tmp_path.resolve()) not in json.dumps(expected, sort_keys=True)
    assert expected["validator_source_sha256"] == "sha256:" + "e" * 64


def test_valid_cache_hit_rehashes_and_probes_without_rebuilding(tmp_path: Path) -> None:
    runner = _FakeRuntimeRunner()
    first = _ensure_fake_runtime(tmp_path, runner)
    executable_mtime = first.validator_path.stat().st_mtime_ns
    manifest_path = first.validator_path.parents[1] / "runtime-manifest.json"
    manifest_mtime = manifest_path.stat().st_mtime_ns

    second = _ensure_fake_runtime(tmp_path, runner)

    assert second == first
    assert runner.configure_count == 1
    assert runner.build_count == 1
    assert runner.probe_count >= 2
    assert second.validator_path.stat().st_mtime_ns == executable_mtime
    assert manifest_path.stat().st_mtime_ns == manifest_mtime


@pytest.mark.parametrize("corrupt_target", ["executable", "shader", "manifest", "unexpected"])
def test_corrupt_cache_entry_is_rebuilt(tmp_path: Path, corrupt_target: str) -> None:
    runner = _FakeRuntimeRunner()
    first = _ensure_fake_runtime(tmp_path, runner)
    entry_root = first.validator_path.parents[1]
    if corrupt_target == "executable":
        first.validator_path.write_bytes(b"corrupt")
    elif corrupt_target == "shader":
        (first.validator_path.parent / "FrameworkShaders" / "fixture.shader").unlink()
    else:
        if corrupt_target == "manifest":
            (entry_root / "runtime-manifest.json").write_text("{}", encoding="utf-8")
        else:
            _write(first.validator_path.parent / "stale-page.bin", b"undeclared")

    second = _ensure_fake_runtime(tmp_path, runner)

    assert runner.build_count == 2
    assert second.validator_sha256 == first.validator_sha256


def test_build_cleans_only_staging_attempts_older_than_24_hours(tmp_path: Path) -> None:
    sdk = _make_sdk(tmp_path / "sdk", platform_id="windows-x86_64")
    facts = _facts(tmp_path)
    plan = build_live2d_runtime_plan(sdk, facts=facts, cache_root=tmp_path / "cache")
    staging_parent = plan.cache_root / ".staging" / plan.platform_id
    stale = staging_parent / f"{plan.cache_key}.stale"
    recent = staging_parent / f"{plan.cache_key}.recent"
    _write(stale / "partial", b"stale")
    _write(recent / "partial", b"recent")
    stale_time = time.time() - 25 * 60 * 60
    os.utime(stale, (stale_time, stale_time))

    ensure_live2d_runtime_toolchain(
        sdk_root=sdk,
        cache_root=tmp_path / "cache",
        _facts=facts,
        _command_runner=_FakeRuntimeRunner(),
    )

    assert not stale.exists()
    assert recent.is_dir()


def test_two_callers_publish_one_cache_entry(tmp_path: Path) -> None:
    runner = _FakeRuntimeRunner(build_delay=0.15)
    sdk = _make_sdk(tmp_path / "sdk", platform_id="windows-x86_64")
    facts = _facts(tmp_path)

    def ensure() -> Live2DRuntimeToolchain:
        return ensure_live2d_runtime_toolchain(
            sdk_root=sdk,
            cache_root=tmp_path / "cache",
            _facts=facts,
            _command_runner=runner,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(lambda _index: ensure(), range(2)))

    assert results[0] == results[1]
    assert runner.configure_count == 1
    assert runner.build_count == 1


@pytest.mark.parametrize(
    ("fail_phase", "expected_code", "expected_log"),
    [
        ("configure", "live2d_validator_build_failed", "configure.log"),
        ("build", "live2d_validator_build_failed", "build.log"),
        ("probe", "live2d_validator_probe_failed", "probe.log"),
    ],
)
def test_failed_attempt_never_publishes_entry_and_retains_log(
    tmp_path: Path,
    fail_phase: str,
    expected_code: str,
    expected_log: str,
) -> None:
    runner = _FakeRuntimeRunner(fail_phase=fail_phase)
    sdk = _make_sdk(tmp_path / "sdk", platform_id="windows-x86_64")
    facts = _facts(tmp_path)
    plan = build_live2d_runtime_plan(sdk, facts=facts, cache_root=tmp_path / "cache")

    with pytest.raises(Live2DRuntimeToolchainError) as error:
        ensure_live2d_runtime_toolchain(
            sdk_root=sdk,
            cache_root=tmp_path / "cache",
            _facts=facts,
            _command_runner=runner,
        )

    assert error.value.code == expected_code
    assert not plan.entry_root.exists()
    assert error.value.log_path is not None
    assert error.value.log_path.name == expected_log
    assert error.value.log_path.is_file()
    assert f"synthetic {fail_phase} failure" in error.value.log_path.read_text(encoding="utf-8")


def test_probe_backend_mismatch_never_publishes_entry(tmp_path: Path) -> None:
    runner = _FakeRuntimeRunner(probe_backend="opengl-egl-headless")
    sdk = _make_sdk(tmp_path / "sdk", platform_id="windows-x86_64")
    facts = _facts(tmp_path)
    plan = build_live2d_runtime_plan(sdk, facts=facts, cache_root=tmp_path / "cache")

    with pytest.raises(Live2DRuntimeToolchainError) as error:
        ensure_live2d_runtime_toolchain(
            sdk_root=sdk,
            cache_root=tmp_path / "cache",
            _facts=facts,
            _command_runner=runner,
        )

    assert error.value.code == "live2d_validator_probe_failed"
    assert not plan.entry_root.exists()


@pytest.mark.parametrize(
    ("stderr", "expected_code"),
    [
        ("Could NOT find OpenGL (missing: OPENGL_egl_LIBRARY)", "live2d_validator_graphics_dependency_missing"),
        ("Each download failed; URL_HASH mismatch", "live2d_validator_dependency_fetch_failed"),
        ("No CMAKE_CXX_COMPILER could be found", "live2d_validator_build_tool_missing"),
    ],
)
def test_configure_failure_has_stable_actionable_code(
    tmp_path: Path,
    stderr: str,
    expected_code: str,
) -> None:
    sdk = _make_sdk(tmp_path / "sdk", platform_id="windows-x86_64")
    facts = _facts(tmp_path)

    def runner(command, *, cwd, timeout_seconds):
        del cwd, timeout_seconds
        return subprocess.CompletedProcess(command, 1, "", stderr)

    with pytest.raises(Live2DRuntimeToolchainError) as error:
        ensure_live2d_runtime_toolchain(
            sdk_root=sdk,
            cache_root=tmp_path / "cache",
            _facts=facts,
            _command_runner=runner,
        )
    assert error.value.code == expected_code
    assert error.value.log_path is not None and error.value.log_path.is_file()


def test_missing_cmake_has_build_tool_code_and_retained_log(tmp_path: Path) -> None:
    sdk = _make_sdk(tmp_path / "sdk", platform_id="windows-x86_64")
    facts = _facts(tmp_path)

    def runner(command, *, cwd, timeout_seconds):
        del command, cwd, timeout_seconds
        raise FileNotFoundError("cmake")

    with pytest.raises(Live2DRuntimeToolchainError) as error:
        ensure_live2d_runtime_toolchain(
            sdk_root=sdk,
            cache_root=tmp_path / "cache",
            _facts=facts,
            _command_runner=runner,
        )
    assert error.value.code == "live2d_validator_build_tool_missing"
    assert error.value.log_path is not None and error.value.log_path.is_file()


def test_runtime_manifest_is_canonical_and_path_independent(tmp_path: Path) -> None:
    toolchain = _ensure_fake_runtime(tmp_path, _FakeRuntimeRunner())
    manifest_path = toolchain.validator_path.parents[1] / "runtime-manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="ascii"))
    serialized = manifest_path.read_text(encoding="ascii")

    assert serialized == json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    assert str(tmp_path.resolve()) not in serialized
    assert payload["cache_key"] == toolchain.cache_key
    assert payload["executable_path"] == f"bin/{toolchain.validator_path.name}"
    assert payload["validator_source_sha256"] == toolchain.validator_source_sha256


def test_runtime_toolchain_contract_is_exported_from_public_auto_rig_package() -> None:
    import module.auto_rig as auto_rig
    import module.auto_rig.export.live2d as live2d

    assert auto_rig.Live2DRuntimeBuildPlan is Live2DRuntimeBuildPlan
    assert auto_rig.Live2DRuntimeToolchain is Live2DRuntimeToolchain
    assert auto_rig.ensure_live2d_runtime_toolchain is ensure_live2d_runtime_toolchain
    assert live2d.resolve_cubism_sdk_root is resolve_cubism_sdk_root
