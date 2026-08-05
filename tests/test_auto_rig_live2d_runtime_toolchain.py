from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from module.auto_rig.export.live2d.runtime_toolchain import (
    DEFAULT_LINUX_DEPENDENCY_PINS,
    Live2DRuntimeBuildFacts,
    Live2DRuntimeBuildPlan,
    Live2DRuntimeToolchain,
    Live2DRuntimeToolchainError,
    build_live2d_runtime_plan,
    default_live2d_runtime_cache_root,
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
    _write(source / "main.cpp", "int main() { return 0; }\n")
    return source


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
        protocol_digest="sha256:" + "1" * 64,
        dependency_pins=DEFAULT_LINUX_DEPENDENCY_PINS if platform_id == "linux-x86_64" else {},
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


@pytest.mark.parametrize(
    "semantic_change",
    ["core", "framework", "validator", "compiler", "cmake", "protocol", "dependency"],
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
        _write(facts.validator_source_root / "main.cpp", "changed")
    elif semantic_change == "compiler":
        facts = replace(facts, compiler_identity="compiler changed")
    elif semantic_change == "cmake":
        facts = replace(facts, cmake_identity="cmake changed")
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


def test_runtime_toolchain_contract_is_exported_from_public_auto_rig_package() -> None:
    import module.auto_rig as auto_rig
    import module.auto_rig.export.live2d as live2d

    assert auto_rig.Live2DRuntimeBuildPlan is Live2DRuntimeBuildPlan
    assert auto_rig.Live2DRuntimeToolchain is Live2DRuntimeToolchain
    assert live2d.resolve_cubism_sdk_root is resolve_cubism_sdk_root
