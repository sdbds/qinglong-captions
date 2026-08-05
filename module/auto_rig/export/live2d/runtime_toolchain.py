from __future__ import annotations

import base64
import ctypes
import errno
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import string
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Literal, Mapping, Sequence

from ...artifacts import (
    ArtifactContractError,
    atomic_write_bytes,
    canonical_json_bytes,
    normalize_relative_path,
    sha256_file,
)

LIVE2D_RUNTIME_PLAN_VERSION = "live2d-runtime-build-plan-v2"
LIVE2D_RUNTIME_CACHE_SCHEMA_VERSION = "live2d-runtime-cache-v2"
LIVE2D_VALIDATOR_SOURCE_IDENTITY_VERSION = "live2d-validator-source-inventory-v1"
LIVE2D_VALIDATOR_PROTOCOL_DIGEST = "sha256:" + hashlib.sha256(
    b"qinglong-live2d-validator-protocol-v2"
).hexdigest()

DEFAULT_LINUX_DEPENDENCY_PINS: Mapping[str, str] = {
    "glew_sha256": "d4fc82893cfb00109578d0a1a2337fb8ca335b3ceccf97b97e5cc7f08e4353e1",
    "glew_url": "https://downloads.sourceforge.net/project/glew/glew/2.2.0/glew-2.2.0.tgz",
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

_CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
_STAGING_MAX_AGE_SECONDS = 24 * 60 * 60
_CONFIGURE_TIMEOUT_SECONDS = 15 * 60.0
_BUILD_TIMEOUT_SECONDS = 30 * 60.0
_PROBE_TIMEOUT_SECONDS = 30.0
_RUNTIME_MANIFEST_FIELDS = {
    "backend_id",
    "cache_key",
    "core_sha256",
    "executable_path",
    "identity_sha256",
    "outputs",
    "platform_id",
    "protocol_digest",
    "schema_version",
    "validator_source_sha256",
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
    generator_identity: str
    protocol_digest: str = LIVE2D_VALIDATOR_PROTOCOL_DIGEST
    dependency_pins: Mapping[str, str] = field(default_factory=dict)
    cmake_executable: str = "cmake"
    c_compiler_executable: str | None = None
    cxx_compiler_executable: str | None = None


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
    transient_root: Path
    built_executable_path: Path
    configure_command: tuple[str, ...]
    build_command: tuple[str, ...]
    identity_payload: Mapping[str, object]
    validator_source_sha256: str


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
    validator_source_sha256: str

    def fingerprint_payload(self, *, attestation_record_sha256: str) -> Mapping[str, str]:
        return {
            "platform_id": self.platform_id,
            "backend_id": self.backend_id,
            "core_sha256": self.core_sha256,
            "validator_sha256": self.validator_sha256,
            "runtime_cache_key": self.cache_key,
            "attestation_record_sha256": attestation_record_sha256,
            "validator_source_sha256": self.validator_source_sha256,
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
    required = [
        "cubism-info.yml",
        "Core/include",
        "Framework/CMakeLists.txt",
        "Framework/src",
        layout["core_runtime"],
        layout["core_link"],
    ]
    if platform_id == "linux-x86_64":
        required.append("Samples/OpenGL/thirdParty/stb/stb_image.h")
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


def _windows_discovery_roots(drive_mask: int, *, current_drive: str) -> tuple[Path, ...]:
    if drive_mask:
        return tuple(
            Path(f"{letter}:\\")
            for index, letter in enumerate(string.ascii_uppercase)
            if drive_mask & (1 << index)
        )
    return (Path(f"{current_drive}\\"),) if current_drive else ()


def _default_discovery_roots() -> tuple[Path, ...]:
    roots = [Path.home(), Path.home() / ".local" / "share", Path("/opt")]
    if os.name == "nt":
        try:
            drive_mask = int(ctypes.windll.kernel32.GetLogicalDrives())
        except (AttributeError, OSError, ValueError):
            drive_mask = 0
        roots.extend(
            _windows_discovery_roots(
                drive_mask,
                current_drive=Path.cwd().drive,
            )
        )
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


def _resolve_compiler_executable(configured: str | None, fallback: str) -> str:
    requested = str(configured or fallback).strip()
    candidate = Path(requested).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    located = shutil.which(requested)
    return str(Path(located).resolve()) if located else requested


def _compiler_executable_identity(role: str, executable: str) -> str:
    path = Path(executable)
    digest = "sha256:unavailable"
    if path.is_file():
        try:
            digest = sha256_file(path)
        except (ArtifactContractError, OSError):
            pass
    version = _command_identity((executable, "--version"), fallback="version-unavailable")
    return f"{role}:{path.name or executable}:{version}:{digest}"


def _windows_compiler_identity() -> str:
    installer_root = Path(
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    ) / "Microsoft Visual Studio" / "Installer"
    vswhere = installer_root / "vswhere.exe"
    if not vswhere.is_file():
        return "msvc-v143-undetected"
    try:
        result = subprocess.run(
            (
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ),
            capture_output=True,
            check=False,
            text=True,
            errors="replace",
            timeout=10,
        )
        installation = Path(result.stdout.strip()).resolve()
        version_file = installation / "VC" / "Auxiliary" / "Build" / "Microsoft.VCToolsVersion.default.txt"
        toolset_version = version_file.read_text(encoding="utf-8").strip()
        compiler = installation / "VC" / "Tools" / "MSVC" / toolset_version / "bin" / "Hostx64" / "x64" / "cl.exe"
        if result.returncode != 0 or not compiler.is_file():
            return "msvc-v143-undetected"
        return f"msvc:{toolset_version}:{sha256_file(compiler)}"
    except (ArtifactContractError, OSError, subprocess.SubprocessError, UnicodeError):
        return "msvc-v143-undetected"


def _default_build_facts(platform_id: str) -> Live2DRuntimeBuildFacts:
    repository_root = Path(__file__).resolve().parents[4]
    if platform_id == "windows-x86_64":
        compiler_identity = _windows_compiler_identity()
        generator_identity = "Visual Studio 17 2022:x64"
    else:
        c_compiler = _resolve_compiler_executable(os.environ.get("CC"), "cc")
        cxx_compiler = _resolve_compiler_executable(os.environ.get("CXX"), "c++")
        compiler_identity = ";".join(
            (
                _compiler_executable_identity("cc", c_compiler),
                _compiler_executable_identity("cxx", cxx_compiler),
            )
        )
        ninja_identity = _command_identity(("ninja", "--version"), fallback="ninja-unavailable")
        generator_identity = f"Ninja:{ninja_identity}"
    return Live2DRuntimeBuildFacts(
        platform_id=platform_id,
        validator_source_root=repository_root / "tools" / "auto_rig_live2d_e0",
        cmake_identity=_command_identity(("cmake", "--version"), fallback="cmake-unavailable"),
        compiler_identity=compiler_identity,
        generator_identity=generator_identity,
        dependency_pins=DEFAULT_LINUX_DEPENDENCY_PINS if platform_id == "linux-x86_64" else {},
        c_compiler_executable=(c_compiler if platform_id == "linux-x86_64" else None),
        cxx_compiler_executable=(cxx_compiler if platform_id == "linux-x86_64" else None),
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


def _validator_source_inventory(
    root: Path,
    *,
    platform_id: str,
) -> list[Mapping[str, object]]:
    if not root.is_dir():
        raise Live2DRuntimeToolchainError(
            "live2d_validator_source_invalid",
            f"required source directory is missing: {root}",
        )
    platform_source = {
        "windows-x86_64": "windows",
        "linux-x86_64": "linux",
    }.get(platform_id)
    if platform_source is None:
        _platform_layout(platform_id)
        raise AssertionError("unreachable")

    candidates = [root / "CMakeLists.txt"]
    for source_dir in (root / "common", root / platform_source):
        if source_dir.is_dir():
            candidates.extend(path for path in source_dir.rglob("*") if path.is_file())
    native_suffixes = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inl", ".cmake"}
    candidates.extend(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.casefold() in native_suffixes
    )

    unique = {path.resolve(): path for path in candidates if path.is_file()}
    return [
        _file_record(f"validator/{path.relative_to(root).as_posix()}", path)
        for path in sorted(unique.values(), key=lambda item: item.relative_to(root).as_posix())
    ]


def live2d_validator_source_inventory(
    source_root: str | Path,
    *,
    platform_id: str,
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        _validator_source_inventory(
            Path(source_root).expanduser().resolve(),
            platform_id=platform_id,
        )
    )


def _validator_source_sha256(
    inventory: Sequence[Mapping[str, object]],
    *,
    platform_id: str,
) -> str:
    payload = {
        "schema_version": LIVE2D_VALIDATOR_SOURCE_IDENTITY_VERSION,
        "platform_id": platform_id,
        "files": list(inventory),
    }
    return f"sha256:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def live2d_validator_source_sha256(
    source_root: str | Path,
    *,
    platform_id: str,
) -> str:
    inventory = live2d_validator_source_inventory(source_root, platform_id=platform_id)
    return _validator_source_sha256(inventory, platform_id=platform_id)


def _build_identity_payload(
    *,
    sdk_root: Path,
    facts: Live2DRuntimeBuildFacts,
    core_path: Path,
    core_link_path: Path,
    backend_id: str,
) -> Mapping[str, object]:
    validator_inventory = _validator_source_inventory(
        facts.validator_source_root.resolve(),
        platform_id=facts.platform_id,
    )
    validator_source_sha256 = _validator_source_sha256(
        validator_inventory,
        platform_id=facts.platform_id,
    )
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
    if facts.platform_id == "linux-x86_64":
        sdk_inventory.append(
            _file_record(
                "sdk/Samples/OpenGL/thirdParty/stb/stb_image.h",
                sdk_root / "Samples" / "OpenGL" / "thirdParty" / "stb" / "stb_image.h",
            )
        )
    return {
        "schema_version": LIVE2D_RUNTIME_PLAN_VERSION,
        "platform_id": facts.platform_id,
        "backend_id": backend_id,
        "protocol_digest": facts.protocol_digest,
        "cmake_identity": facts.cmake_identity,
        "compiler_identity": facts.compiler_identity,
        "generator_identity": facts.generator_identity,
        "dependency_pins": dict(sorted(facts.dependency_pins.items())),
        "validator_source_sha256": validator_source_sha256,
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
    transient_identity = hashlib.sha256(
        canonical_json_bytes(
            {
                "cache_root": str(resolved_cache),
                "cache_key": cache_key,
                "platform_id": selected_platform,
            }
        )
    ).digest()
    transient_name = base64.urlsafe_b64encode(transient_identity).decode("ascii").rstrip("=")
    transient_root = Path(tempfile.gettempdir()).resolve() / "ql2d" / transient_name
    build_root = transient_root / "build"
    built_bin = transient_root / "bin"
    generator_arguments = (
        ("-G", "Visual Studio 17 2022", "-A", "x64")
        if selected_platform == "windows-x86_64"
        else ("-G", "Ninja")
    )
    compiler_arguments = (
        (
            f"-DCMAKE_C_COMPILER={active_facts.c_compiler_executable}",
            f"-DCMAKE_CXX_COMPILER={active_facts.cxx_compiler_executable}",
        )
        if selected_platform == "linux-x86_64"
        and active_facts.c_compiler_executable
        and active_facts.cxx_compiler_executable
        else ()
    )
    configure_command = (
        active_facts.cmake_executable,
        "-S",
        str(source_root),
        "-B",
        str(build_root),
        *generator_arguments,
        *compiler_arguments,
        f"-DCUBISM_SDK_ROOT={resolved_sdk}",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={built_bin}",
        f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={built_bin}",
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
        transient_root=transient_root,
        built_executable_path=built_bin / layout["executable"],
        configure_command=configure_command,
        build_command=build_command,
        identity_payload=identity_payload,
        validator_source_sha256=str(identity_payload["validator_source_sha256"]),
    )


def _default_command_runner(
    command: tuple[str, ...],
    *,
    cwd: Path,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        check=False,
        text=True,
        errors="replace",
        timeout=timeout_seconds,
    )


def _write_command_log(
    log_path: Path,
    command: tuple[str, ...],
    *,
    returncode: int | None,
    stdout: str = "",
    stderr: str = "",
    exception: BaseException | None = None,
) -> None:
    lines = [f"command: {shlex.join(command)}"]
    if returncode is not None:
        lines.append(f"returncode: {returncode}")
    if exception is not None:
        lines.append(f"exception: {type(exception).__name__}: {exception}")
    lines.extend(("stdout:", stdout.rstrip(), "stderr:", stderr.rstrip()))
    atomic_write_bytes(log_path, ("\n".join(lines).rstrip() + "\n").encode("utf-8"))


def _command_failure_code(*, phase: str, log_text: str) -> str:
    if phase == "probe":
        return "live2d_validator_probe_failed"
    lowered = log_text.casefold()
    if phase == "configure":
        graphics_markers = (
            "could not find opengl",
            "could not find egl",
            "opengl_egl_found",
            "egl development",
        )
        if any(marker in lowered for marker in graphics_markers):
            return "live2d_validator_graphics_dependency_missing"
        dependency_markers = (
            "download failed",
            "each download failed",
            "hash mismatch",
            "url_hash",
            "fetchcontent",
        )
        if any(marker in lowered for marker in dependency_markers):
            return "live2d_validator_dependency_fetch_failed"
        compiler_markers = (
            "cmake_c_compiler not set",
            "cmake_cxx_compiler not set",
            "no cmake_c_compiler could be found",
            "no cmake_cxx_compiler could be found",
        )
        if any(marker in lowered for marker in compiler_markers):
            return "live2d_validator_build_tool_missing"
    return "live2d_validator_build_failed"


def _invoke_command(
    runner: _CommandRunner,
    command: tuple[str, ...],
    *,
    cwd: Path,
    log_path: Path,
    timeout_seconds: float,
    phase: str,
) -> subprocess.CompletedProcess[str]:
    try:
        result = runner(command, cwd=cwd, timeout_seconds=timeout_seconds)
    except (FileNotFoundError, PermissionError) as exc:
        _write_command_log(log_path, command, returncode=None, exception=exc)
        code = (
            "live2d_validator_probe_failed"
            if phase == "probe"
            else "live2d_validator_build_tool_missing"
        )
        raise Live2DRuntimeToolchainError(code, f"{phase} command could not start", log_path=log_path) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        _write_command_log(log_path, command, returncode=None, exception=exc)
        code = "live2d_validator_probe_failed" if phase == "probe" else "live2d_validator_build_failed"
        raise Live2DRuntimeToolchainError(code, f"{phase} command did not complete", log_path=log_path) from exc
    if not isinstance(result, subprocess.CompletedProcess):
        error = TypeError("command runner must return subprocess.CompletedProcess")
        _write_command_log(log_path, command, returncode=None, exception=error)
        raise Live2DRuntimeToolchainError(
            "live2d_validator_build_failed",
            "command runner violated the runtime build contract",
            log_path=log_path,
        ) from error
    stdout = "" if result.stdout is None else str(result.stdout)
    stderr = "" if result.stderr is None else str(result.stderr)
    _write_command_log(
        log_path,
        command,
        returncode=result.returncode,
        stdout=stdout,
        stderr=stderr,
    )
    if result.returncode != 0:
        code = _command_failure_code(phase=phase, log_text=f"{stdout}\n{stderr}")
        raise Live2DRuntimeToolchainError(
            code,
            f"{phase} command failed with exit code {result.returncode}",
            log_path=log_path,
        )
    return result


@contextmanager
def _exclusive_cache_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        raise
                    time.sleep(0.1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _managed_remove_tree(path: Path, *, cache_root: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink():
        path.unlink()
        return
    resolved = path.resolve()
    root = cache_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise Live2DRuntimeToolchainError(
            "live2d_validator_build_failed",
            f"refusing to remove a path outside the runtime cache: {resolved}",
        ) from exc
    if resolved == root:
        raise Live2DRuntimeToolchainError(
            "live2d_validator_build_failed",
            "refusing to remove the runtime cache root",
        )
    shutil.rmtree(path)


def _staging_root(plan: Live2DRuntimeBuildPlan) -> Path:
    return plan.cache_root / ".staging" / plan.platform_id / plan.cache_key


def _lock_path(plan: Live2DRuntimeBuildPlan) -> Path:
    return (
        plan.cache_root
        / "live2d-validator"
        / ".locks"
        / plan.platform_id
        / f"{plan.cache_key}.lock"
    )


def _cleanup_stale_staging_attempts(plan: Live2DRuntimeBuildPlan) -> None:
    staging_parent = _staging_root(plan).parent
    if not staging_parent.is_dir():
        return
    cutoff = time.time() - _STAGING_MAX_AGE_SECONDS
    for candidate in staging_parent.glob(f"{plan.cache_key}.*"):
        try:
            modified = candidate.lstat().st_mtime
        except OSError:
            continue
        if modified < cutoff:
            _managed_remove_tree(candidate, cache_root=plan.cache_root)


def _probe_validator(
    plan: Live2DRuntimeBuildPlan,
    executable: Path,
    *,
    runner: _CommandRunner,
    report_path: Path,
    log_path: Path,
) -> None:
    _invoke_command(
        runner,
        (str(executable), "--probe-report", str(report_path)),
        cwd=executable.parent,
        log_path=log_path,
        timeout_seconds=_PROBE_TIMEOUT_SECONDS,
        phase="probe",
    )
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Live2DRuntimeToolchainError(
            "live2d_validator_probe_failed",
            "validator did not write a valid probe report",
            log_path=log_path,
        ) from exc
    expected = {
        "backend_id": plan.backend_id,
        "protocol_digest": plan.identity_payload["protocol_digest"],
        "schema_version": "auto-rig-live2d-probe-v1",
    }
    if type(payload) is not dict or payload != expected:
        raise Live2DRuntimeToolchainError(
            "live2d_validator_probe_failed",
            "validator probe protocol or backend does not match the build plan",
            log_path=log_path,
        )


def _manifest_output_records(root: Path) -> list[Mapping[str, object]]:
    records: list[Mapping[str, object]] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative == "runtime-manifest.json":
            continue
        records.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _runtime_manifest(plan: Live2DRuntimeBuildPlan, root: Path) -> Mapping[str, object]:
    executable_path = f"bin/{plan.executable_path.name}"
    outputs = _manifest_output_records(root)
    output_paths = {str(record["path"]) for record in outputs}
    if executable_path not in output_paths:
        raise Live2DRuntimeToolchainError(
            "live2d_validator_build_failed",
            "native build did not produce the expected validator executable",
        )
    if not any(path.startswith("bin/FrameworkShaders/") for path in output_paths):
        raise Live2DRuntimeToolchainError(
            "live2d_validator_build_failed",
            "native build did not publish the required Framework shader files",
        )
    return {
        "schema_version": LIVE2D_RUNTIME_CACHE_SCHEMA_VERSION,
        "cache_key": plan.cache_key,
        "platform_id": plan.platform_id,
        "backend_id": plan.backend_id,
        "protocol_digest": plan.identity_payload["protocol_digest"],
        "identity_sha256": f"sha256:{plan.cache_key}",
        "core_sha256": sha256_file(plan.core_path),
        "validator_source_sha256": plan.validator_source_sha256,
        "executable_path": executable_path,
        "outputs": outputs,
    }


def _load_manifest_toolchain(
    plan: Live2DRuntimeBuildPlan,
    *,
    root: Path,
) -> Live2DRuntimeToolchain | None:
    manifest_path = root / "runtime-manifest.json"
    try:
        raw = manifest_path.read_bytes()
        payload = json.loads(raw.decode("ascii"))
        if type(payload) is not dict or set(payload) != _RUNTIME_MANIFEST_FIELDS:
            return None
        if raw != canonical_json_bytes(payload) + b"\n":
            return None
        expected_scalars = {
            "schema_version": LIVE2D_RUNTIME_CACHE_SCHEMA_VERSION,
            "cache_key": plan.cache_key,
            "platform_id": plan.platform_id,
            "backend_id": plan.backend_id,
            "protocol_digest": plan.identity_payload["protocol_digest"],
            "identity_sha256": f"sha256:{plan.cache_key}",
            "core_sha256": sha256_file(plan.core_path),
            "validator_source_sha256": plan.validator_source_sha256,
            "executable_path": f"bin/{plan.executable_path.name}",
        }
        if any(payload[field] != value for field, value in expected_scalars.items()):
            return None
        if type(payload["outputs"]) is not list or not payload["outputs"]:
            return None

        declared_paths: set[str] = set()
        validator_sha256: str | None = None
        for record in payload["outputs"]:
            if type(record) is not dict or set(record) != {"path", "size", "sha256"}:
                return None
            relative = normalize_relative_path(record["path"])
            if relative in declared_paths or relative == "runtime-manifest.json":
                return None
            declared_paths.add(relative)
            size = record["size"]
            digest = record["sha256"]
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                return None
            if not isinstance(digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
                return None
            candidate = root / Path(*relative.split("/"))
            if candidate.is_symlink() or not candidate.is_file():
                return None
            resolved = candidate.resolve()
            resolved.relative_to(root.resolve())
            if candidate.stat().st_size != size or sha256_file(candidate) != digest:
                return None
            if relative == expected_scalars["executable_path"]:
                validator_sha256 = digest

        actual_paths: set[str] = set()
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                return None
            if candidate.is_file():
                relative = candidate.relative_to(root).as_posix()
                if relative != "runtime-manifest.json":
                    actual_paths.add(relative)
        if actual_paths != declared_paths:
            return None
        if validator_sha256 is None:
            return None
        if not any(path.startswith("bin/FrameworkShaders/") for path in declared_paths):
            return None
    except (
        ArtifactContractError,
        json.JSONDecodeError,
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
    ):
        return None
    return Live2DRuntimeToolchain(
        sdk_root=plan.sdk_root,
        core_path=plan.core_path,
        validator_path=root / Path(*str(payload["executable_path"]).split("/")),
        platform_id=plan.platform_id,
        backend_id=plan.backend_id,
        cache_key=plan.cache_key,
        core_sha256=str(payload["core_sha256"]),
        validator_sha256=validator_sha256,
        validator_source_sha256=str(payload["validator_source_sha256"]),
    )


def _load_cached_toolchain(
    plan: Live2DRuntimeBuildPlan,
    *,
    runner: _CommandRunner,
    probe: bool = True,
) -> Live2DRuntimeToolchain | None:
    toolchain = _load_manifest_toolchain(plan, root=plan.entry_root)
    if toolchain is None or not probe:
        return toolchain
    probe_root = plan.cache_root / ".probe"
    probe_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{plan.cache_key[:12]}-", dir=probe_root) as temporary:
        temporary_root = Path(temporary)
        try:
            _probe_validator(
                plan,
                toolchain.validator_path,
                runner=runner,
                report_path=temporary_root / "probe.json",
                log_path=temporary_root / "probe.log",
            )
        except Live2DRuntimeToolchainError:
            return None
    return toolchain


def _retain_failed_logs(
    plan: Live2DRuntimeBuildPlan,
    staging_root: Path,
    error: Live2DRuntimeToolchainError,
) -> Path:
    attempt_id = f"{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    destination = (
        plan.cache_root
        / "live2d-validator"
        / ".failures"
        / plan.platform_id
        / plan.cache_key
        / attempt_id
    )
    destination.mkdir(parents=True, exist_ok=False)
    for source in sorted(staging_root.glob("*.log")):
        shutil.copy2(source, destination / source.name)
    log_name = error.log_path.name if error.log_path is not None else "runtime.log"
    retained = destination / log_name
    if not retained.is_file():
        atomic_write_bytes(retained, (str(error) + "\n").encode("utf-8"))
    return retained


def _build_and_publish_toolchain(
    plan: Live2DRuntimeBuildPlan,
    *,
    runner: _CommandRunner,
) -> Live2DRuntimeToolchain:
    staging_root = _staging_root(plan)
    transient_cache_root = Path(tempfile.gettempdir()).resolve() / "ql2d"
    if staging_root.exists() or staging_root.is_symlink():
        _managed_remove_tree(staging_root, cache_root=plan.cache_root)
    if plan.transient_root.exists() or plan.transient_root.is_symlink():
        _managed_remove_tree(plan.transient_root, cache_root=transient_cache_root)
    if plan.entry_root.exists() or plan.entry_root.is_symlink():
        _managed_remove_tree(plan.entry_root, cache_root=plan.cache_root)
    staging_root.mkdir(parents=True, exist_ok=False)
    plan.transient_root.mkdir(parents=True, exist_ok=False)
    try:
        _invoke_command(
            runner,
            plan.configure_command,
            cwd=plan.transient_root,
            log_path=staging_root / "configure.log",
            timeout_seconds=_CONFIGURE_TIMEOUT_SECONDS,
            phase="configure",
        )
        _invoke_command(
            runner,
            plan.build_command,
            cwd=plan.transient_root,
            log_path=staging_root / "build.log",
            timeout_seconds=_BUILD_TIMEOUT_SECONDS,
            phase="build",
        )
        if not plan.built_executable_path.is_file():
            missing = Live2DRuntimeToolchainError(
                "live2d_validator_build_failed",
                "native build completed without the expected validator executable",
                log_path=staging_root / "build.log",
            )
            raise missing
        _probe_validator(
            plan,
            plan.built_executable_path,
            runner=runner,
            report_path=staging_root / "probe.json",
            log_path=staging_root / "probe.log",
        )

        shutil.copytree(plan.built_executable_path.parent, staging_root / "bin")
        _managed_remove_tree(plan.transient_root, cache_root=transient_cache_root)
        for transient in (
            staging_root / "configure.log",
            staging_root / "build.log",
            staging_root / "probe.log",
            staging_root / "probe.json",
        ):
            transient.unlink(missing_ok=True)
        manifest = _runtime_manifest(plan, staging_root)
        atomic_write_bytes(
            staging_root / "runtime-manifest.json",
            canonical_json_bytes(manifest) + b"\n",
        )
        staged_toolchain = _load_manifest_toolchain(plan, root=staging_root)
        if staged_toolchain is None:
            raise Live2DRuntimeToolchainError(
                "live2d_validator_build_failed",
                "staged runtime cache entry failed its manifest validation",
            )
        plan.entry_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging_root, plan.entry_root)
        return Live2DRuntimeToolchain(
            sdk_root=staged_toolchain.sdk_root,
            core_path=staged_toolchain.core_path,
            validator_path=plan.executable_path,
            platform_id=staged_toolchain.platform_id,
            backend_id=staged_toolchain.backend_id,
            cache_key=staged_toolchain.cache_key,
            core_sha256=staged_toolchain.core_sha256,
            validator_sha256=staged_toolchain.validator_sha256,
            validator_source_sha256=staged_toolchain.validator_source_sha256,
        )
    except Live2DRuntimeToolchainError as error:
        retained_log = _retain_failed_logs(plan, staging_root, error)
        if plan.transient_root.exists() or plan.transient_root.is_symlink():
            _managed_remove_tree(plan.transient_root, cache_root=transient_cache_root)
        if staging_root.exists() or staging_root.is_symlink():
            _managed_remove_tree(staging_root, cache_root=plan.cache_root)
        raise Live2DRuntimeToolchainError(
            error.code,
            str(error).split(": ", 1)[-1],
            log_path=retained_log,
        ) from error
    except OSError as error:
        wrapped = Live2DRuntimeToolchainError(
            "live2d_validator_build_failed",
            f"runtime cache transaction failed: {error}",
        )
        retained_log = _retain_failed_logs(plan, staging_root, wrapped)
        if plan.transient_root.exists() or plan.transient_root.is_symlink():
            _managed_remove_tree(plan.transient_root, cache_root=transient_cache_root)
        if staging_root.exists() or staging_root.is_symlink():
            _managed_remove_tree(staging_root, cache_root=plan.cache_root)
        raise Live2DRuntimeToolchainError(
            wrapped.code,
            str(wrapped).split(": ", 1)[-1],
            log_path=retained_log,
        ) from error


def ensure_live2d_runtime_toolchain(
    *,
    sdk_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    _facts: Live2DRuntimeBuildFacts | None = None,
    _command_runner: _CommandRunner | None = None,
) -> Live2DRuntimeToolchain:
    selected_platform = _facts.platform_id if _facts is not None else detect_live2d_runtime_platform()
    resolved_sdk = resolve_cubism_sdk_root(sdk_root, platform_id=selected_platform)
    plan = build_live2d_runtime_plan(
        resolved_sdk,
        facts=_facts,
        cache_root=cache_root,
    )
    runner = _default_command_runner if _command_runner is None else _command_runner
    plan.cache_root.mkdir(parents=True, exist_ok=True)
    cached = _load_cached_toolchain(plan, runner=runner)
    if cached is not None:
        return cached

    with _exclusive_cache_lock(_lock_path(plan)):
        _cleanup_stale_staging_attempts(plan)
        cached = _load_cached_toolchain(plan, runner=runner)
        if cached is not None:
            return cached
        return _build_and_publish_toolchain(plan, runner=runner)


__all__ = [
    "DEFAULT_LINUX_DEPENDENCY_PINS",
    "LIVE2D_RUNTIME_CACHE_SCHEMA_VERSION",
    "LIVE2D_RUNTIME_PLAN_VERSION",
    "LIVE2D_VALIDATOR_PROTOCOL_DIGEST",
    "LIVE2D_VALIDATOR_SOURCE_IDENTITY_VERSION",
    "Live2DRuntimeBuildFacts",
    "Live2DRuntimeBuildPlan",
    "Live2DRuntimeToolchain",
    "Live2DRuntimeToolchainError",
    "build_live2d_runtime_plan",
    "default_live2d_runtime_cache_root",
    "detect_live2d_runtime_platform",
    "ensure_live2d_runtime_toolchain",
    "live2d_validator_source_inventory",
    "live2d_validator_source_sha256",
    "resolve_cubism_sdk_root",
]
