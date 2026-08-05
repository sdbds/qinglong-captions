from __future__ import annotations

import hashlib
import os
import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Sequence

from ...artifacts import canonical_json_bytes, sha256_file

LIVE2D_RUNTIME_PLAN_VERSION = "live2d-runtime-build-plan-v1"
LIVE2D_VALIDATOR_PROTOCOL_DIGEST = "sha256:" + hashlib.sha256(
    b"qinglong-live2d-validator-protocol-v2"
).hexdigest()

DEFAULT_LINUX_DEPENDENCY_PINS: Mapping[str, str] = {
    "glew_sha256": "f781d57097cdd076c6e34656d3aae239abaa03da7fd60e2249ee29df546e3d1e",
    "glew_url": "https://github.com/nigels-com/glew/archive/refs/tags/glew-2.2.0.tar.gz",
    "glfw_sha256": "c038d34200234d071fae9345bc455e4a8f2f544ab60150765d7704e08f3dac01",
    "glfw_url": "https://github.com/glfw/glfw/archive/refs/tags/3.4.tar.gz",
}

_SUPPORTED_PLATFORMS = {
    "windows-x86_64": {
        "backend_id": "d3d11-warp",
        "core_runtime": "Core/dll/windows/x86_64/Live2DCubismCore.dll",
        "core_link": "Core/lib/windows/x86_64/143/Live2DCubismCore_MD.lib",
        "executable": "qinglong_live2d_validator.exe",
    },
    "linux-x86_64": {
        "backend_id": "opengl-egl-headless",
        "core_runtime": "Core/dll/linux/x86_64/libLive2DCubismCore.so",
        "core_link": "Core/lib/linux/x86_64/libLive2DCubismCore.a",
        "executable": "qinglong_live2d_validator",
    },
}

_LEGACY_CORE_SUFFIXES = {
    "windows-x86_64": ("core", "dll", "windows", "x86_64", "live2dcubismcore.dll"),
    "linux-x86_64": ("core", "dll", "linux", "x86_64", "liblive2dcubismcore.so"),
}


class Live2DRuntimeToolchainError(RuntimeError):
    def __init__(self, code: str, message: str, *, log_path: Path | None = None):
        self.code = code
        self.log_path = log_path
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class Live2DRuntimeBuildFacts:
    platform_id: str
    validator_source_root: Path
    cmake_identity: str
    compiler_identity: str
    protocol_digest: str = LIVE2D_VALIDATOR_PROTOCOL_DIGEST
    dependency_pins: Mapping[str, str] = field(default_factory=dict)
    cmake_executable: str = "cmake"


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


def detect_live2d_runtime_platform(
    *,
    system: str | None = None,
    machine: str | None = None,
) -> str:
    normalized_system = (system or platform.system()).strip().lower()
    normalized_machine = (machine or platform.machine()).strip().lower()
    x86_64_aliases = {"amd64", "x86_64", "x64"}
    if normalized_system == "windows" and normalized_machine in x86_64_aliases:
        return "windows-x86_64"
    if normalized_system == "linux" and normalized_machine in x86_64_aliases:
        return "linux-x86_64"
    raise Live2DRuntimeToolchainError(
        "live2d_platform_unsupported",
        f"unsupported Live2D validator platform: {normalized_system}/{normalized_machine}",
    )


def default_live2d_runtime_cache_root(
    *,
    environ: Mapping[str, str] | None = None,
    home: str | Path | None = None,
) -> Path:
    environment = os.environ if environ is None else environ
    configured = str(environment.get("QINGLONG_CAPTIONS_RUNTIME_CACHE", "")).strip()
    if configured:
        return Path(configured).expanduser().resolve()
    home_root = Path.home() if home is None else Path(home)
    return (home_root / ".cache" / "qinglong-captions" / "runtimes").expanduser().resolve()


def _platform_layout(platform_id: str) -> Mapping[str, str]:
    try:
        return _SUPPORTED_PLATFORMS[platform_id]
    except KeyError as exc:
        raise Live2DRuntimeToolchainError(
            "live2d_platform_unsupported",
            f"unsupported Live2D validator platform: {platform_id}",
        ) from exc


def _sdk_missing_paths(sdk_root: Path, platform_id: str) -> tuple[str, ...]:
    layout = _platform_layout(platform_id)
    required = (
        "cubism-info.yml",
        "Core/include",
        "Framework/CMakeLists.txt",
        "Framework/src",
        layout["core_runtime"],
        layout["core_link"],
    )
    return tuple(relative for relative in required if not (sdk_root / relative).exists())


def _require_sdk_layout(sdk_root: str | Path, platform_id: str) -> Path:
    resolved = Path(sdk_root).expanduser().resolve()
    missing = _sdk_missing_paths(resolved, platform_id)
    if missing:
        raise Live2DRuntimeToolchainError(
            "live2d_sdk_layout_invalid",
            f"Cubism SDK layout is incomplete at {resolved}; missing: {', '.join(missing)}",
        )
    return resolved


def _legacy_sdk_root(value: str, platform_id: str) -> Path | None:
    candidate = Path(value).expanduser().resolve()
    suffix = _LEGACY_CORE_SUFFIXES[platform_id]
    parts = tuple(part.casefold() for part in candidate.parts)
    if len(parts) < len(suffix) or parts[-len(suffix) :] != suffix:
        return None
    return candidate.parents[len(suffix) - 1]


def _sdk_version_key(root: Path) -> tuple[int, ...]:
    try:
        text = (root / "cubism-info.yml").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ()
    match = re.search(r"(?m)^version:\s*([^\r\n#]+)", text)
    if match is None:
        return ()
    return tuple(int(value) for value in re.findall(r"\d+", match.group(1)))


def _default_discovery_roots() -> tuple[Path, ...]:
    roots = [Path.home(), Path.home() / ".local" / "share", Path("/opt")]
    if os.name == "nt":
        drive = Path.cwd().drive
        if drive:
            roots.append(Path(f"{drive}\\"))
    return tuple(roots)


def _discover_sdk_roots(roots: Sequence[str | Path], platform_id: str) -> tuple[Path, ...]:
    candidates: dict[str, Path] = {}
    for raw_root in roots:
        root = Path(raw_root).expanduser()
        possible = [root]
        if root.is_dir():
            try:
                possible.extend(
                    child for child in root.iterdir() if child.is_dir() and child.name.startswith("CubismSdkForNative-")
                )
            except OSError:
                pass
        for candidate in possible:
            resolved = candidate.resolve()
            try:
                if not _sdk_missing_paths(resolved, platform_id):
                    candidates[str(resolved).casefold()] = resolved
            except OSError:
                continue
    return tuple(candidates[key] for key in sorted(candidates))


def resolve_cubism_sdk_root(
    explicit: str | Path | None,
    *,
    environ: Mapping[str, str] | None = None,
    platform_id: str | None = None,
    discovery_roots: Sequence[str | Path] | None = None,
) -> Path:
    selected_platform = platform_id or detect_live2d_runtime_platform()
    _platform_layout(selected_platform)
    environment = os.environ if environ is None else environ

    if explicit is not None and str(explicit).strip():
        return _require_sdk_layout(explicit, selected_platform)

    configured = str(environment.get("CUBISM_SDK_ROOT", "")).strip()
    if configured:
        return _require_sdk_layout(configured, selected_platform)

    legacy_core = str(environment.get("LIVE2D_CORE_PATH", "")).strip()
    if legacy_core:
        legacy_root = _legacy_sdk_root(legacy_core, selected_platform)
        if legacy_root is not None:
            return _require_sdk_layout(legacy_root, selected_platform)

    candidates = _discover_sdk_roots(
        _default_discovery_roots() if discovery_roots is None else discovery_roots,
        selected_platform,
    )
    if candidates:
        return sorted(candidates, key=lambda item: (tuple(-value for value in _sdk_version_key(item)), str(item).casefold()))[0]

    raise Live2DRuntimeToolchainError(
        "live2d_sdk_not_found",
        "Cubism SDK for Native was not configured or discovered",
    )


def _command_identity(command: Sequence[str], *, fallback: str) -> str:
    try:
        result = subprocess.run(
            tuple(command),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return fallback
    output = (result.stdout or result.stderr).strip().splitlines()
    return output[0].strip() if output else fallback


def _default_build_facts(platform_id: str) -> Live2DRuntimeBuildFacts:
    repository_root = Path(__file__).resolve().parents[4]
    compiler_command = ("cl",) if platform_id == "windows-x86_64" else ("c++", "--version")
    compiler_fallback = "msvc-v143-via-cmake" if platform_id == "windows-x86_64" else "c++-unavailable"
    return Live2DRuntimeBuildFacts(
        platform_id=platform_id,
        validator_source_root=repository_root / "tools" / "auto_rig_live2d_e0",
        cmake_identity=_command_identity(("cmake", "--version"), fallback="cmake-unavailable"),
        compiler_identity=_command_identity(compiler_command, fallback=compiler_fallback),
        dependency_pins=DEFAULT_LINUX_DEPENDENCY_PINS if platform_id == "linux-x86_64" else {},
    )


def _file_record(label: str, path: Path) -> Mapping[str, object]:
    return {
        "path": label,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _inventory_tree(root: Path, *, label_root: str) -> list[Mapping[str, object]]:
    if not root.is_dir():
        raise Live2DRuntimeToolchainError(
            "live2d_validator_source_invalid",
            f"required source directory is missing: {root}",
        )
    records: list[Mapping[str, object]] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        records.append(_file_record(f"{label_root}/{relative}", path))
    return records


def _build_identity_payload(
    *,
    sdk_root: Path,
    facts: Live2DRuntimeBuildFacts,
    core_path: Path,
    core_link_path: Path,
    backend_id: str,
) -> Mapping[str, object]:
    validator_inventory = _inventory_tree(facts.validator_source_root.resolve(), label_root="validator")
    framework_inventory = _inventory_tree(sdk_root / "Framework" / "src", label_root="sdk/Framework/src")
    framework_inventory.append(
        _file_record("sdk/Framework/CMakeLists.txt", sdk_root / "Framework" / "CMakeLists.txt")
    )
    core_headers = _inventory_tree(sdk_root / "Core" / "include", label_root="sdk/Core/include")
    sdk_inventory = [
        _file_record("sdk/cubism-info.yml", sdk_root / "cubism-info.yml"),
        _file_record("sdk/Core/runtime", core_path),
        _file_record("sdk/Core/link", core_link_path),
        *core_headers,
        *framework_inventory,
    ]
    return {
        "schema_version": LIVE2D_RUNTIME_PLAN_VERSION,
        "platform_id": facts.platform_id,
        "backend_id": backend_id,
        "protocol_digest": facts.protocol_digest,
        "cmake_identity": facts.cmake_identity,
        "compiler_identity": facts.compiler_identity,
        "dependency_pins": dict(sorted(facts.dependency_pins.items())),
        "validator_inventory": validator_inventory,
        "sdk_inventory": sdk_inventory,
    }


def build_live2d_runtime_plan(
    sdk_root: str | Path,
    *,
    facts: Live2DRuntimeBuildFacts | None = None,
    cache_root: str | Path | None = None,
) -> Live2DRuntimeBuildPlan:
    selected_platform = facts.platform_id if facts is not None else detect_live2d_runtime_platform()
    layout = _platform_layout(selected_platform)
    active_facts = facts or _default_build_facts(selected_platform)
    resolved_sdk = _require_sdk_layout(sdk_root, selected_platform)
    source_root = active_facts.validator_source_root.expanduser().resolve()
    if not (source_root / "CMakeLists.txt").is_file():
        raise Live2DRuntimeToolchainError(
            "live2d_validator_source_invalid",
            f"validator CMake project is missing: {source_root}",
        )
    resolved_cache = (
        default_live2d_runtime_cache_root() if cache_root is None else Path(cache_root).expanduser().resolve()
    )
    core_path = resolved_sdk / layout["core_runtime"]
    core_link_path = resolved_sdk / layout["core_link"]
    identity_payload = _build_identity_payload(
        sdk_root=resolved_sdk,
        facts=active_facts,
        core_path=core_path,
        core_link_path=core_link_path,
        backend_id=layout["backend_id"],
    )
    cache_key = hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest()
    entry_root = resolved_cache / "live2d-validator" / selected_platform / cache_key
    staging_root = resolved_cache / ".staging" / selected_platform / cache_key
    build_root = staging_root / "build"
    staged_bin = staging_root / "bin"
    configure_command = (
        active_facts.cmake_executable,
        "-S",
        str(source_root),
        "-B",
        str(build_root),
        f"-DCUBISM_SDK_ROOT={resolved_sdk}",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={staged_bin}",
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={staged_bin}",
    )
    build_command: tuple[str, ...] = (
        active_facts.cmake_executable,
        "--build",
        str(build_root),
        "--config",
        "Release",
        "--parallel",
    )
    return Live2DRuntimeBuildPlan(
        sdk_root=resolved_sdk,
        core_path=core_path,
        core_link_path=core_link_path,
        platform_id=selected_platform,  # type: ignore[arg-type]
        backend_id=layout["backend_id"],  # type: ignore[arg-type]
        cache_root=resolved_cache,
        cache_key=cache_key,
        entry_root=entry_root,
        executable_path=entry_root / "bin" / layout["executable"],
        configure_command=configure_command,
        build_command=build_command,
        identity_payload=identity_payload,
    )


__all__ = [
    "DEFAULT_LINUX_DEPENDENCY_PINS",
    "LIVE2D_RUNTIME_PLAN_VERSION",
    "LIVE2D_VALIDATOR_PROTOCOL_DIGEST",
    "Live2DRuntimeBuildFacts",
    "Live2DRuntimeBuildPlan",
    "Live2DRuntimeToolchain",
    "Live2DRuntimeToolchainError",
    "build_live2d_runtime_plan",
    "default_live2d_runtime_cache_root",
    "detect_live2d_runtime_platform",
    "resolve_cubism_sdk_root",
]
