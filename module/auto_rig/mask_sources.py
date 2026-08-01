from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .artifacts import ArtifactContractError, sha256_file
from .contracts import AutoRigContractError, AutoRigInputContract, AutoRigPartContract


def _error(message: str) -> AutoRigContractError:
    return AutoRigContractError("input_contract_mismatch", message)


@dataclass(frozen=True, slots=True)
class LoadedPartAlpha:
    part: AutoRigPartContract
    width: int
    height: int
    alpha_u8: bytes

    def __post_init__(self) -> None:
        if (
            isinstance(self.width, bool)
            or not isinstance(self.width, int)
            or self.width <= 0
            or isinstance(self.height, bool)
            or not isinstance(self.height, int)
            or self.height <= 0
        ):
            raise _error("loaded alpha dimensions must be positive integers")
        if not isinstance(self.alpha_u8, bytes) or len(self.alpha_u8) != self.width * self.height:
            raise _error("loaded alpha byte count differs from its dimensions")


def _verify_snapshot_files(parts: tuple[AutoRigPartContract, ...]) -> None:
    expected_by_path: dict[Path, str] = {}
    for part in parts:
        for path, expected in (
            (part.source.color_path, part.source.color_sha256),
            (part.source.depth_path, part.source.depth_sha256),
        ):
            prior = expected_by_path.setdefault(path, expected)
            if prior != expected:
                raise _error(f"validated source has conflicting digests: {path.name}")
    for path, expected in expected_by_path.items():
        try:
            actual = sha256_file(path)
        except ArtifactContractError as exc:
            raise _error(f"validated PartSource is unavailable: {path.name}") from exc
        if actual != expected:
            raise _error(f"validated PartSource changed after validation: {path.name}")


def _load_png_alphas(parts: tuple[AutoRigPartContract, ...]) -> tuple[LoadedPartAlpha, ...]:
    from PIL import Image, UnidentifiedImageError

    loaded: list[LoadedPartAlpha] = []
    for part in parts:
        width = part.xyxy[2] - part.xyxy[0]
        height = part.xyxy[3] - part.xyxy[1]
        try:
            with Image.open(part.source.color_path) as image:
                image.load()
                if image.format != "PNG" or image.mode != "RGBA" or image.size != (width, height):
                    raise _error(f"validated color PNG metadata changed: {part.source_tag}")
                alpha_u8 = image.getchannel("A").tobytes()
        except AutoRigContractError:
            raise
        except (OSError, UnidentifiedImageError) as exc:
            raise _error(f"validated color PNG cannot be decoded: {part.source_tag}") from exc
        loaded.append(
            LoadedPartAlpha(
                part=part,
                width=width,
                height=height,
                alpha_u8=alpha_u8,
            )
        )
    return tuple(loaded)


def _load_psd_alphas(parts: tuple[AutoRigPartContract, ...]) -> tuple[LoadedPartAlpha, ...]:
    try:
        import numpy as np
        from psd_tools import PSDImage
    except ImportError as exc:  # pragma: no cover - dependency profile test owns this branch
        raise _error("PSD alpha decoding requires the auto-rig dependency profile") from exc

    color_path = parts[0].source.color_path
    try:
        psd = PSDImage.open(color_path)
        layers = {layer.name: layer for layer in psd}
    except Exception as exc:
        raise _error("validated final.psd cannot be decoded") from exc
    loaded: list[LoadedPartAlpha] = []
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
        alpha = np.floor(np.clip(pixels[..., 3], 0.0, 1.0) * 255.0 + 0.5).astype(
            np.uint8
        )
        loaded.append(
            LoadedPartAlpha(
                part=part,
                width=width,
                height=height,
                alpha_u8=alpha.tobytes(order="C"),
            )
        )
    return tuple(loaded)


def load_validated_part_alphas(
    contract: AutoRigInputContract,
) -> tuple[LoadedPartAlpha, ...]:
    """Decode alpha only from the immutable PartSource snapshot."""

    if not isinstance(contract, AutoRigInputContract):
        raise _error("alpha decoder requires an AutoRigInputContract")
    parts = tuple(sorted(contract.parts, key=lambda item: item.part_id))
    if not parts:
        raise _error("alpha decoder requires at least one validated PartSource")
    _verify_snapshot_files(parts)
    if contract.payload_mode == "png":
        return _load_png_alphas(parts)
    if contract.payload_mode == "psd":
        return _load_psd_alphas(parts)
    raise _error(f"unsupported validated PartSource mode: {contract.payload_mode}")


__all__ = ["LoadedPartAlpha", "load_validated_part_alphas"]
