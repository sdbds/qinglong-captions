from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .artifacts import SDPOSE_BODY_SOURCE, PoseArtifactError, PoseSourceContract
from .attention import FlashAttentionBackendError, SDPoseFlashAttnProcessor
from .bundle import inspect_sdpose_body_bundle
from .heatmap import SDPoseBodyHeatmapHead
from .sdpose import SDPoseLoadedModels


def load_prefixed_state_dict(
    bundle_path: str | Path,
    component: str,
) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    if not component or "." in component:
        raise ValueError("SDPose component must be one non-empty identifier")
    path = Path(bundle_path).expanduser().resolve(strict=True)
    prefix = component + "."
    try:
        with safe_open(path, framework="pt", device="cpu") as bundle:
            state = {name[len(prefix) :]: bundle.get_tensor(name) for name in bundle.keys() if name.startswith(prefix)}
    except Exception as exc:
        raise PoseArtifactError(f"failed to read SDPose {component} component") from exc
    if not state or any(not name for name in state):
        raise PoseArtifactError(f"SDPose bundle has no valid {component} component")
    return state


def normalize_sdpose_vae_state_dict(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    replacements = (
        (".query.", ".to_q."),
        (".key.", ".to_k."),
        (".value.", ".to_v."),
        (".proj_attn.", ".to_out.0."),
    )
    normalized: dict[str, torch.Tensor] = {}
    for source_name, tensor in state.items():
        target_name = source_name
        for old, new in replacements:
            target_name = target_name.replace(old, new)
        if target_name in normalized:
            raise PoseArtifactError(f"SDPose VAE key conversion collision: {target_name}")
        normalized[target_name] = tensor
    return normalized


def _bundle_metadata(bundle_path: Path) -> dict[str, str]:
    from safetensors import safe_open

    try:
        with safe_open(bundle_path, framework="pt", device="cpu") as bundle:
            metadata = bundle.metadata()
    except Exception as exc:
        raise PoseArtifactError("failed to read SDPose bundle metadata") from exc
    if metadata is None:
        raise PoseArtifactError("SDPose bundle metadata is missing")
    return metadata


def _strict_load(module: Any, state: dict[str, torch.Tensor], component: str) -> None:
    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise PoseArtifactError(f"SDPose {component} tensor inventory does not match its architecture") from exc


def load_sdpose_body_models(
    bundle_path: str | Path,
    *,
    source_contract: PoseSourceContract = SDPOSE_BODY_SOURCE,
    device: str | torch.device | None = None,
    prefer_fa2: bool = True,
    verify_bundle: bool = True,
) -> SDPoseLoadedModels:
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from diffusers.models.attention_processor import AttnProcessor2_0
    from safetensors import safe_open

    path = Path(bundle_path).expanduser().resolve(strict=True)
    info = inspect_sdpose_body_bundle(path, source_contract=source_contract) if verify_bundle else None
    metadata = _bundle_metadata(path)
    try:
        unet_config = json.loads(metadata["unet_config_jcs"])
        vae_config = json.loads(metadata["vae_config_jcs"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise PoseArtifactError("SDPose bundle model configs are invalid") from exc

    resolved_device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if resolved_device.type == "cuda" else torch.float32
    unet = UNet2DConditionModel.from_config(unet_config).eval()
    vae = AutoencoderKL.from_config(vae_config).eval()
    decoder = SDPoseBodyHeatmapHead().eval()
    unet.to(device=resolved_device, dtype=dtype)
    vae.to(device=resolved_device, dtype=dtype)
    decoder.to(device=resolved_device, dtype=dtype)

    for component, module in (
        ("unet", unet),
        ("vae", vae),
        ("decoder", decoder),
    ):
        state = load_prefixed_state_dict(path, component)
        if component == "vae":
            state = normalize_sdpose_vae_state_dict(state)
        _strict_load(module, state, component)
        del state

    try:
        with safe_open(path, framework="pt", device="cpu") as bundle:
            empty_prompt = bundle.get_tensor("conditioning.empty_prompt")
    except Exception as exc:
        raise PoseArtifactError("SDPose bundle conditioning tensor is missing") from exc

    flash_processor: SDPoseFlashAttnProcessor | None = None
    fallback_reason: str | None = None
    if resolved_device.type != "cuda":
        unet.set_attn_processor(AttnProcessor2_0())
        backend = "torch-cpu"
    elif prefer_fa2:
        try:
            flash_processor = SDPoseFlashAttnProcessor()
            unet.set_attn_processor(flash_processor)
            backend = "torch-fa2"
        except FlashAttentionBackendError as exc:
            unet.set_attn_processor(AttnProcessor2_0())
            backend = "torch-sdpa"
            fallback_reason = str(exc)
    else:
        unet.set_attn_processor(AttnProcessor2_0())
        backend = "torch-sdpa"
        fallback_reason = "flash_attention_disabled"

    bundle_sha256 = info.file_sha256 if info is not None else _sha256_file(path)
    return SDPoseLoadedModels(
        vae=vae,
        unet=unet,
        decoder=decoder,
        empty_prompt=empty_prompt,
        device=resolved_device,
        dtype=dtype,
        backend=backend,
        bundle_sha256=bundle_sha256,
        flash_processor=flash_processor,
        fallback_reason=fallback_reason,
    )


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "load_prefixed_state_dict",
    "load_sdpose_body_models",
    "normalize_sdpose_vae_state_dict",
]
