from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..artifacts import canonical_json_bytes
from .artifacts import (
    PoseArtifactError,
    PoseSourceContract,
    verify_pose_source_snapshot,
)
from .conditioning import sdpose_empty_prompt, sdpose_empty_prompt_provenance

if TYPE_CHECKING:
    import torch

SDPOSE_BODY_BUNDLE_SCHEMA = "sdpose-body17-bundle-v1"
SDPOSE_BODY_BUNDLE_FILENAME = "sdpose_body17_fp16.safetensors"
SDPOSE_BODY_BUNDLE_CONVERTER = "sdpose-body17-converter-v1"

_COMPONENT_FILES = {
    "unet": "unet/diffusion_pytorch_model.safetensors",
    "vae": "vae/diffusion_pytorch_model.safetensors",
    "decoder": "decoder/decoder.safetensors",
}
_CONFIG_FILES = {
    "unet_config_jcs": "unet/config.json",
    "vae_config_jcs": "vae/config.json",
    "scheduler_config_jcs": "scheduler/scheduler_config.json",
}


@dataclass(frozen=True, slots=True)
class SDPoseBundleInfo:
    path: Path
    file_sha256: str
    tensor_payload_sha256: str
    source_contract_sha256: str
    tensor_count: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_contract_payload(contract: PoseSourceContract) -> dict[str, Any]:
    return {
        "files": [
            {
                "path": item.relative_path,
                "sha256": item.sha256,
                "size": item.size,
            }
            for item in sorted(contract.files, key=lambda value: value.relative_path)
        ],
        "provider_id": contract.provider_id,
        "repo_id": contract.repo_id,
        "revision": contract.revision,
    }


def _canonical_json_text(payload: Any) -> str:
    return canonical_json_bytes(payload).decode("ascii")


def _read_config(root: Path, relative_path: str) -> str:
    path = root / Path(*relative_path.split("/"))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PoseArtifactError(f"invalid SDPose config: {relative_path}") from exc
    return _canonical_json_text(payload)


def _update_tensor_digest(digest: Any, name: str, tensor: torch.Tensor) -> None:
    import torch

    contiguous = tensor.detach().cpu().contiguous()
    name_bytes = name.encode("utf-8")
    dtype_bytes = str(contiguous.dtype).removeprefix("torch.").encode("ascii")
    digest.update(struct.pack("<Q", len(name_bytes)))
    digest.update(name_bytes)
    digest.update(struct.pack("<Q", len(dtype_bytes)))
    digest.update(dtype_bytes)
    digest.update(struct.pack("<Q", contiguous.ndim))
    for dimension in contiguous.shape:
        digest.update(struct.pack("<q", int(dimension)))
    byte_view = contiguous.view(torch.uint8).numpy()
    digest.update(memoryview(byte_view))


def _tensor_payload_sha256(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        _update_tensor_digest(digest, name, tensors[name])
    return digest.hexdigest()


def _safetensors_dtype(tensor: torch.Tensor) -> str:
    import torch

    mapping = {
        torch.bool: "BOOL",
        torch.uint8: "U8",
        torch.int8: "I8",
        torch.int16: "I16",
        torch.int32: "I32",
        torch.int64: "I64",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.float32: "F32",
        torch.float64: "F64",
    }
    try:
        return mapping[tensor.dtype]
    except KeyError as exc:
        raise PoseArtifactError(f"unsupported SDPose bundle tensor dtype: {tensor.dtype}") from exc


def _write_deterministic_safetensors(
    path: Path,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
) -> None:
    import torch

    if sys.byteorder != "little":
        raise PoseArtifactError("SDPose bundle writer requires little-endian tensors")
    header: dict[str, Any] = {"__metadata__": dict(metadata)}
    offset = 0
    ordered: list[tuple[str, torch.Tensor]] = []
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        size = tensor.numel() * tensor.element_size()
        header[name] = {
            "data_offsets": [offset, offset + size],
            "dtype": _safetensors_dtype(tensor),
            "shape": list(tensor.shape),
        }
        ordered.append((name, tensor))
        offset += size
    header_bytes = canonical_json_bytes(header)
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(header_bytes)))
        stream.write(header_bytes)
        for _, tensor in ordered:
            raw = tensor.view(torch.uint8).reshape(-1).numpy()
            stream.write(memoryview(raw))
        stream.flush()
        os.fsync(stream.fileno())


def _validate_empty_prompt(empty_prompt: torch.Tensor) -> torch.Tensor:
    import torch

    if (
        not isinstance(empty_prompt, torch.Tensor)
        or tuple(empty_prompt.shape) != (1, 2, 1024)
        or empty_prompt.dtype != torch.float16
        or not bool(torch.isfinite(empty_prompt).all())
    ):
        raise PoseArtifactError("SDPose empty prompt must be a finite FP16 tensor with shape (1,2,1024)")
    return empty_prompt.detach().cpu().contiguous()


def _load_component_tensors(root: Path) -> dict[str, torch.Tensor]:
    import torch
    from safetensors import safe_open

    tensors: dict[str, torch.Tensor] = {}
    for component, relative_path in _COMPONENT_FILES.items():
        path = root / Path(*relative_path.split("/"))
        try:
            with safe_open(path, framework="pt", device="cpu") as source:
                for source_name in sorted(source.keys()):
                    export_name = f"{component}.{source_name}"
                    if export_name in tensors:
                        raise PoseArtifactError(f"duplicate SDPose bundle tensor: {export_name}")
                    tensor = source.get_tensor(source_name)
                    if tensor.is_floating_point():
                        tensor = tensor.to(dtype=torch.float16)
                    tensors[export_name] = tensor.detach().cpu().contiguous()
        except PoseArtifactError:
            raise
        except Exception as exc:
            raise PoseArtifactError(f"failed to read SDPose component: {relative_path}") from exc
    return tensors


def _metadata(
    root: Path,
    contract: PoseSourceContract,
    *,
    tensor_payload_sha256: str,
    tensor_count: int,
) -> dict[str, str]:
    contract_text = _canonical_json_text(_source_contract_payload(contract))
    metadata = {
        "bundle_schema": SDPOSE_BODY_BUNDLE_SCHEMA,
        "converter_version": SDPOSE_BODY_BUNDLE_CONVERTER,
        "provider_id": contract.provider_id,
        "source_contract_jcs": contract_text,
        "source_contract_sha256": hashlib.sha256(contract_text.encode("ascii")).hexdigest(),
        "tensor_count": str(tensor_count),
        "tensor_payload_sha256": tensor_payload_sha256,
        "conditioning_provenance_jcs": _canonical_json_text(sdpose_empty_prompt_provenance()),
    }
    metadata.update({metadata_name: _read_config(root, relative_path) for metadata_name, relative_path in _CONFIG_FILES.items()})
    return metadata


def _inspect_metadata(
    metadata: dict[str, str] | None,
    source_contract: PoseSourceContract,
) -> tuple[str, str, int]:
    if metadata is None:
        raise PoseArtifactError("SDPose bundle has no metadata")
    contract_text = _canonical_json_text(_source_contract_payload(source_contract))
    expected_contract_sha = hashlib.sha256(contract_text.encode("ascii")).hexdigest()
    if metadata.get("bundle_schema") != SDPOSE_BODY_BUNDLE_SCHEMA:
        raise PoseArtifactError("SDPose bundle schema mismatch")
    if metadata.get("converter_version") != SDPOSE_BODY_BUNDLE_CONVERTER:
        raise PoseArtifactError("SDPose bundle converter mismatch")
    if metadata.get("source_contract_jcs") != contract_text:
        raise PoseArtifactError("SDPose bundle source contract mismatch")
    if metadata.get("source_contract_sha256") != expected_contract_sha:
        raise PoseArtifactError("SDPose bundle source contract digest mismatch")
    try:
        tensor_count = int(metadata["tensor_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PoseArtifactError("SDPose bundle tensor count is invalid") from exc
    payload_sha = metadata.get("tensor_payload_sha256", "")
    if len(payload_sha) != 64:
        raise PoseArtifactError("SDPose bundle tensor payload digest is invalid")
    for config_name in _CONFIG_FILES:
        try:
            json.loads(metadata[config_name])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise PoseArtifactError(f"SDPose bundle metadata is missing {config_name}") from exc
    if metadata.get("conditioning_provenance_jcs") != _canonical_json_text(sdpose_empty_prompt_provenance()):
        raise PoseArtifactError("SDPose bundle conditioning provenance mismatch")
    return expected_contract_sha, payload_sha, tensor_count


def inspect_sdpose_body_bundle(
    path: str | Path,
    *,
    source_contract: PoseSourceContract,
) -> SDPoseBundleInfo:
    from safetensors import safe_open

    candidate = Path(path).expanduser().resolve(strict=True)
    try:
        with safe_open(candidate, framework="pt", device="cpu") as bundle:
            contract_sha, expected_payload_sha, expected_count = _inspect_metadata(bundle.metadata(), source_contract)
            tensors = {name: bundle.get_tensor(name) for name in bundle.keys()}
    except PoseArtifactError:
        raise
    except Exception as exc:
        raise PoseArtifactError("failed to read SDPose body bundle") from exc
    if len(tensors) != expected_count:
        raise PoseArtifactError("SDPose bundle tensor count mismatch")
    actual_payload_sha = _tensor_payload_sha256(tensors)
    if actual_payload_sha != expected_payload_sha:
        raise PoseArtifactError("SDPose bundle tensor payload SHA-256 mismatch")
    return SDPoseBundleInfo(
        path=candidate,
        file_sha256=_sha256_file(candidate),
        tensor_payload_sha256=actual_payload_sha,
        source_contract_sha256=contract_sha,
        tensor_count=len(tensors),
    )


def build_sdpose_body_bundle(
    source_snapshot: str | Path,
    output_path: str | Path,
    *,
    source_contract: PoseSourceContract,
) -> SDPoseBundleInfo:
    root = verify_pose_source_snapshot(source_snapshot, source_contract)
    conditioning = _validate_empty_prompt(sdpose_empty_prompt())
    tensors = _load_component_tensors(root)
    tensors["conditioning.empty_prompt"] = conditioning
    payload_sha = _tensor_payload_sha256(tensors)
    metadata = _metadata(
        root,
        source_contract,
        tensor_payload_sha256=payload_sha,
        tensor_count=len(tensors),
    )

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f"{target.name}.",
        suffix=".part",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_deterministic_safetensors(temporary, tensors, metadata)
        info = inspect_sdpose_body_bundle(
            temporary,
            source_contract=source_contract,
        )
        os.replace(temporary, target)
        return SDPoseBundleInfo(
            path=target,
            file_sha256=info.file_sha256,
            tensor_payload_sha256=info.tensor_payload_sha256,
            source_contract_sha256=info.source_contract_sha256,
            tensor_count=info.tensor_count,
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "SDPOSE_BODY_BUNDLE_CONVERTER",
    "SDPOSE_BODY_BUNDLE_FILENAME",
    "SDPOSE_BODY_BUNDLE_SCHEMA",
    "SDPoseBundleInfo",
    "build_sdpose_body_bundle",
    "inspect_sdpose_body_bundle",
]
