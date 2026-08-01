from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .artifacts import ArtifactContractError, sha256_file
from .contracts import AutoRigContractError, AutoRigInputContract, AutoRigPartContract
from .jcs import jcs_sha256
from .native_variants import NativeVariantCandidate, NativeVariantSet

TEXTURE_PIXEL_CONTRACT_VERSION = "texture-pixel-rgba8-straight-srgb-v1"


def _error(message: str) -> AutoRigContractError:
    return AutoRigContractError("input_contract_mismatch", message)


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True, slots=True)
class LoadedTextureRegion:
    part_id: str
    source_kind: Literal["see_through", "native_variant"]
    variant_id: str | None
    xyxy: tuple[int, int, int, int]
    width: int
    height: int
    rgba_u8: bytes
    rgba_sha256: str
    source_file_sha256: str
    alpha_mode: Literal["straight"]
    color_space: Literal["srgb_bytes"]
    pixel_contract_version: str

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise _error("texture region dimensions must be positive")
        if self.xyxy[2] - self.xyxy[0] != self.width or self.xyxy[3] - self.xyxy[1] != self.height:
            raise _error("texture region dimensions differ from xyxy")
        if not isinstance(self.rgba_u8, bytes) or len(self.rgba_u8) != self.width * self.height * 4:
            raise _error("texture region RGBA byte count differs from its dimensions")
        if _sha256_bytes(self.rgba_u8) != self.rgba_sha256:
            raise _error("texture region raw RGBA digest is invalid")
        if self.source_kind == "native_variant" and self.variant_id is None:
            raise _error("native texture region requires variant_id")
        if self.source_kind == "see_through" and self.variant_id is not None:
            raise _error("see-through texture region cannot carry variant_id")
        if self.alpha_mode != "straight" or self.color_space != "srgb_bytes":
            raise _error("texture region alpha/color contract is unsupported")
        if self.pixel_contract_version != TEXTURE_PIXEL_CONTRACT_VERSION:
            raise _error("texture region pixel contract version is unsupported")

    def semantic_entry(self) -> dict[str, object]:
        return {
            "part_id": self.part_id,
            "source_kind": self.source_kind,
            "variant_id": self.variant_id,
            "xyxy": list(self.xyxy),
            "width": self.width,
            "height": self.height,
            "rgba_sha256": self.rgba_sha256,
            "source_file_sha256": self.source_file_sha256,
            "alpha_mode": self.alpha_mode,
            "color_space": self.color_space,
            "pixel_contract_version": self.pixel_contract_version,
        }


def _verify_base_snapshot(parts: tuple[AutoRigPartContract, ...]) -> None:
    expected_by_path: dict[Path, str] = {}
    for part in parts:
        for path, expected in (
            (part.source.color_path, part.source.color_sha256),
            (part.source.depth_path, part.source.depth_sha256),
        ):
            previous = expected_by_path.setdefault(path, expected)
            if previous != expected:
                raise _error(f"validated source has conflicting digests: {path.name}")
    for path, expected in expected_by_path.items():
        try:
            actual = sha256_file(path)
        except ArtifactContractError as exc:
            raise _error(f"validated PartSource is unavailable: {path.name}") from exc
        if actual != expected:
            raise _error(f"validated PartSource changed after validation: {path.name}")


def _loaded_region(
    part: AutoRigPartContract,
    rgba: bytes,
) -> LoadedTextureRegion:
    width = part.xyxy[2] - part.xyxy[0]
    height = part.xyxy[3] - part.xyxy[1]
    return LoadedTextureRegion(
        part_id=part.part_id,
        source_kind="see_through",
        variant_id=None,
        xyxy=part.xyxy,
        width=width,
        height=height,
        rgba_u8=rgba,
        rgba_sha256=_sha256_bytes(rgba),
        source_file_sha256=part.source.color_sha256,
        alpha_mode="straight",
        color_space="srgb_bytes",
        pixel_contract_version=TEXTURE_PIXEL_CONTRACT_VERSION,
    )


def _load_png_regions(parts: tuple[AutoRigPartContract, ...]) -> tuple[LoadedTextureRegion, ...]:
    from PIL import Image, UnidentifiedImageError

    loaded = []
    for part in parts:
        width = part.xyxy[2] - part.xyxy[0]
        height = part.xyxy[3] - part.xyxy[1]
        try:
            with Image.open(part.source.color_path) as image:
                image.load()
                if image.format != "PNG" or image.mode != "RGBA" or image.size != (width, height):
                    raise _error(f"validated color PNG metadata changed: {part.source_tag}")
                rgba = image.tobytes()
        except AutoRigContractError:
            raise
        except (OSError, UnidentifiedImageError) as exc:
            raise _error(f"validated color PNG cannot be decoded: {part.source_tag}") from exc
        loaded.append(_loaded_region(part, rgba))
    return tuple(loaded)


def _load_psd_regions(parts: tuple[AutoRigPartContract, ...]) -> tuple[LoadedTextureRegion, ...]:
    try:
        import numpy as np
        from psd_tools import PSDImage
    except ImportError as exc:  # pragma: no cover - dependency profile owns this branch
        raise _error("PSD RGBA decoding requires the auto-rig dependency profile") from exc

    try:
        psd = PSDImage.open(parts[0].source.color_path)
        layers = {layer.name: layer for layer in psd}
    except Exception as exc:
        raise _error("validated final.psd cannot be decoded") from exc
    loaded = []
    for part in parts:
        layer_name = part.source.layer_name
        if layer_name is None or layer_name not in layers:
            raise _error(f"validated PSD layer is unavailable: {part.source_tag}")
        width = part.xyxy[2] - part.xyxy[0]
        height = part.xyxy[3] - part.xyxy[1]
        layer = layers[layer_name]
        if tuple(layer.bbox) != part.xyxy:
            raise _error(f"validated PSD layer rectangle changed: {part.source_tag}")
        pixels = layer.numpy()
        if pixels.ndim != 3 or pixels.shape != (height, width, 4):
            raise _error(f"validated PSD color layer is not RGBA: {part.source_tag}")
        rgba = np.floor(np.clip(pixels, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        loaded.append(_loaded_region(part, rgba.tobytes(order="C")))
    return tuple(loaded)


def load_base_texture_regions(
    contract: AutoRigInputContract,
) -> tuple[LoadedTextureRegion, ...]:
    """Decode canonical base RGBA crops from a validated input snapshot."""

    if not isinstance(contract, AutoRigInputContract):
        raise _error("texture decoder requires an AutoRigInputContract")
    parts = tuple(sorted(contract.parts, key=lambda item: item.part_id))
    if not parts:
        raise _error("texture decoder requires at least one validated PartSource")
    _verify_base_snapshot(parts)
    if contract.payload_mode == "png":
        return _load_png_regions(parts)
    if contract.payload_mode == "psd":
        return _load_psd_regions(parts)
    raise _error(f"unsupported validated PartSource mode: {contract.payload_mode}")


def _load_native_region(candidate: NativeVariantCandidate) -> LoadedTextureRegion:
    from PIL import Image, UnidentifiedImageError

    try:
        actual_file_sha = sha256_file(candidate.png_path)
    except ArtifactContractError as exc:
        raise _error(f"validated NativeVariant PNG is unavailable: {candidate.variant_id}") from exc
    if actual_file_sha != candidate.file_sha256:
        raise _error(f"validated NativeVariant PNG changed after validation: {candidate.variant_id}")
    width = candidate.xyxy[2] - candidate.xyxy[0]
    height = candidate.xyxy[3] - candidate.xyxy[1]
    try:
        with Image.open(candidate.png_path) as image:
            image.load()
            if image.format != "PNG" or image.mode != "RGBA" or image.size != (width, height):
                raise _error(f"validated NativeVariant PNG metadata changed: {candidate.variant_id}")
            rgba = image.tobytes()
            alpha = image.getchannel("A").tobytes()
    except AutoRigContractError:
        raise
    except (OSError, UnidentifiedImageError) as exc:
        raise _error(f"validated NativeVariant PNG cannot be decoded: {candidate.variant_id}") from exc
    if alpha != candidate.alpha_u8:
        raise _error(f"validated NativeVariant alpha changed after validation: {candidate.variant_id}")
    return LoadedTextureRegion(
        part_id=candidate.part_id,
        source_kind="native_variant",
        variant_id=candidate.variant_id,
        xyxy=candidate.xyxy,
        width=width,
        height=height,
        rgba_u8=rgba,
        rgba_sha256=_sha256_bytes(rgba),
        source_file_sha256=candidate.file_sha256,
        alpha_mode="straight",
        color_space="srgb_bytes",
        pixel_contract_version=TEXTURE_PIXEL_CONTRACT_VERSION,
    )


def load_native_texture_regions(
    variant_set: NativeVariantSet,
) -> tuple[LoadedTextureRegion, ...]:
    """Decode every manifest-valid NativeVariant without making admission decisions."""

    if not isinstance(variant_set, NativeVariantSet):
        raise _error("native texture decoder requires a NativeVariantSet")
    if jcs_sha256(variant_set.semantic_payload()) != variant_set.native_variant_set_sha256:
        raise _error("NativeVariant set digest is invalid")
    return tuple(
        _load_native_region(candidate)
        for candidate in sorted(variant_set.entries, key=lambda item: item.variant_id)
    )


__all__ = [
    "TEXTURE_PIXEL_CONTRACT_VERSION",
    "LoadedTextureRegion",
    "load_base_texture_regions",
    "load_native_texture_regions",
]
