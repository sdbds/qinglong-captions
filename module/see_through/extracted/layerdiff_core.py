# Extracted and adapted from:
# - inference/scripts/inference_psd.py
# - common/utils/inference_utils.py
# Upstream repository: shitagaki-lab/see-through
# Notes: globals removed, project-local IO/state conventions applied.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..see_through_profile import normalize_quant_mode
from ..vendor_bootstrap import ensure_vendor_imports
from utils.transformer_loader import (
    is_quantized_pretrained_component,
    load_pretrained_component,
    move_pretrained_component,
)


DEFAULT_SEED = 42


def _is_cuda_device(device: str | None) -> bool:
    return str(device or "").startswith("cuda")


def _maybe_enable_group_offload(*, pipeline: Any, enabled: bool, device: str, console: Any | None = None) -> None:
    if not enabled:
        return
    if not _is_cuda_device(device):
        if console is not None:
            console.print("[yellow]Skipping group offload:[/yellow] CUDA is unavailable.")
        return
    enable_group_offload = getattr(pipeline, "enable_group_offload", None)
    if not callable(enable_group_offload):
        if console is not None:
            console.print("[yellow]Skipping group offload:[/yellow] pipeline does not expose enable_group_offload().")
        return
    enable_group_offload(device, num_blocks_per_group=1)


def _load_json_weights(vae_ckpt: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from safetensors.torch import load_file

    td_sd: dict[str, Any] = {}
    vae_sd: dict[str, Any] = {}
    sd = load_file(vae_ckpt)
    for key, value in sd.items():
        if key.startswith("trans_decoder."):
            td_sd[key.removeprefix("trans_decoder.")] = value
        elif key.startswith("vae."):
            vae_sd[key.replace("vae.", "", 1)] = value
    return td_sd, vae_sd


def load_layerdiff_pipeline(
    *,
    repo_id: str,
    runtime_context: Any,
    vae_ckpt: str | None = None,
    unet_ckpt: str | None = None,
    group_offload: bool = False,
    quant_mode: str = "none",
    console: Any | None = None,
) -> Any:
    ensure_vendor_imports()
    quant_mode = normalize_quant_mode(quant_mode)

    from modules.layerdiffuse.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
    from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
    from modules.layerdiffuse.vae import TransparentVAE

    if console is not None:
        console.print(f"[cyan]Loading LayerDiff pipeline:[/cyan] {repo_id} [dim](quant_mode={quant_mode})[/dim]")

    if quant_mode != "none" and not _is_cuda_device(getattr(runtime_context, "device", "cpu")):
        raise RuntimeError("See-through quant_mode=nf4 requires CUDA for layerdiff inference.")

    trans_vae = load_pretrained_component(
        TransparentVAE,
        repo_id,
        console=console,
        component_name="LayerDiff transparent VAE",
        subfolder="trans_vae",
    )
    if unet_ckpt is None:
        unet = load_pretrained_component(
            UNetFrameConditionModel,
            repo_id,
            console=console,
            component_name="LayerDiff UNet",
            subfolder="unet",
        )
    else:
        if console is not None:
            console.print(f"[cyan]Loading LayerDiff UNet checkpoint:[/cyan] {unet_ckpt}")
        unet = load_pretrained_component(
            UNetFrameConditionModel,
            unet_ckpt,
            console=console,
            component_name="LayerDiff UNet checkpoint",
        )

    pipeline = load_pretrained_component(
        KDiffusionStableDiffusionXLPipeline,
        repo_id,
        console=console,
        component_name="LayerDiff pipeline",
        trans_vae=trans_vae,
        unet=unet,
        scheduler=None,
    )

    if vae_ckpt is not None:
        if console is not None:
            console.print(f"[cyan]Loading LayerDiff VAE checkpoint:[/cyan] {vae_ckpt}")
        td_sd, vae_sd = _load_json_weights(vae_ckpt)
        if vae_sd:
            pipeline.vae.load_state_dict(vae_sd)
        if td_sd:
            pipeline.trans_vae.decoder.load_state_dict(td_sd)

    if quant_mode == "none":
        for module_name in ("vae", "trans_vae", "unet", "text_encoder", "text_encoder_2"):
            move_pretrained_component(
                getattr(pipeline, module_name, None),
                device=runtime_context.device,
                dtype=runtime_context.dtype,
            )
    else:
        for module_name in ("vae", "trans_vae", "unet", "text_encoder", "text_encoder_2"):
            module = getattr(pipeline, module_name, None)
            if module_name in {"unet", "text_encoder", "text_encoder_2"} and is_quantized_pretrained_component(module):
                continue
            move_pretrained_component(
                module,
                device=runtime_context.device,
                dtype=runtime_context.dtype,
            )

    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=True)
    _maybe_enable_group_offload(
        pipeline=pipeline,
        enabled=group_offload,
        device=runtime_context.device,
        console=console,
    )
    if hasattr(pipeline, "cache_tag_embeds"):
        pipeline.cache_tag_embeds()
    if console is not None:
        console.print(
            f"[green]LayerDiff pipeline ready:[/green] {repo_id} "
            f"(device={runtime_context.device}, dtype={runtime_context.dtype}, group_offload={group_offload})"
        )
    return pipeline


def run_layerdiff_phase(
    *,
    source_path: Path,
    output_dir: Path,
    pipeline: Any,
    resolution: int,
    generator_device: str = "cpu",
    seed: int = DEFAULT_SEED,
) -> dict[str, Path]:
    ensure_vendor_imports()

    import cv2
    import numpy as np
    import torch
    from PIL import Image

    from utils.cv import center_square_pad_resize, smart_resize
    from utils.inference_utils import VALID_BODY_PARTS_V2

    output_dir.mkdir(parents=True, exist_ok=True)
    input_img = np.array(Image.open(source_path).convert("RGBA"))
    fullpage, fullpage_pad_size, fullpage_pad_pos = center_square_pad_resize(input_img, resolution, return_pad_info=True)
    scale = fullpage_pad_size[0] / resolution
    src_image_path = output_dir / "src_img.png"
    Image.fromarray(fullpage).save(src_image_path)

    rng = torch.Generator(device=generator_device).manual_seed(int(seed))
    saved_paths: list[Path] = []

    tag_version = pipeline.unet.get_tag_version()
    if tag_version == "v2":
        pipeline_output = pipeline(
            strength=1.0,
            num_inference_steps=30,
            batch_size=1,
            generator=rng,
            guidance_scale=1.0,
            prompt=VALID_BODY_PARTS_V2,
            negative_prompt="",
            fullpage=fullpage,
        )
        for rst, tag in zip(pipeline_output.images, VALID_BODY_PARTS_V2):
            save_path = output_dir / f"{tag}.png"
            Image.fromarray(rst).save(save_path)
            saved_paths.append(save_path)
    elif tag_version == "v3":
        body_tag_list = [
            "front hair",
            "back hair",
            "head",
            "neck",
            "neckwear",
            "topwear",
            "handwear",
            "bottomwear",
            "legwear",
            "footwear",
            "tail",
            "wings",
            "objects",
        ]
        body_output = pipeline(
            strength=1.0,
            num_inference_steps=30,
            batch_size=1,
            generator=rng,
            guidance_scale=1.0,
            prompt=body_tag_list,
            negative_prompt="",
            fullpage=fullpage,
            group_index=0,
        )
        for rst, tag in zip(body_output.images, body_tag_list):
            save_path = output_dir / f"{tag}.png"
            Image.fromarray(rst).save(save_path)
            saved_paths.append(save_path)

        head_img = body_output.images[2]
        head_tag_list = [
            "headwear",
            "face",
            "irides",
            "eyebrow",
            "eyewhite",
            "eyelash",
            "eyewear",
            "ears",
            "earwear",
            "nose",
            "mouth",
        ]
        hx0, hy0, hw, hh = cv2.boundingRect(cv2.findNonZero((head_img[..., -1] > 15).astype(np.uint8)))

        hx = int(hx0 * scale) - fullpage_pad_pos[0]
        hy = int(hy0 * scale) - fullpage_pad_pos[1]
        hw = int(hw * scale)
        hh = int(hh * scale)

        def _crop_head(img: np.ndarray, xywh: list[int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
            x, y, w, h = xywh
            ih, iw = img.shape[:2]
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            if w < iw // 2:
                px = min(iw - x - w, x, w // 5)
                x1 = min(max(x - px, 0), iw)
                x2 = min(max(x + w + px, 0), iw)
            if h < ih // 2:
                py = min(ih - y - h, y, h // 5)
                y1 = min(max(y - py, 0), ih)
                y2 = min(max(y + h + py, 0), ih)
            return img[y1:y2, x1:x2], (x1, y1, x2, y2)

        input_head, (hx1, hy1, _, _) = _crop_head(input_img, [hx, hy, hw, hh])
        hx1 = int(hx1 / scale + fullpage_pad_pos[0] / scale)
        hy1 = int(hy1 / scale + fullpage_pad_pos[1] / scale)
        ih, iw = input_head.shape[:2]
        input_head, head_pad_size, head_pad_pos = center_square_pad_resize(input_head, resolution, return_pad_info=True)
        Image.fromarray(input_head).save(output_dir / "src_head.png")
        head_output = pipeline(
            strength=1.0,
            num_inference_steps=30,
            batch_size=1,
            generator=rng,
            guidance_scale=1.0,
            prompt=head_tag_list,
            negative_prompt="",
            fullpage=input_head,
            group_index=1,
        )
        canvas = np.zeros((resolution, resolution, 4), dtype=np.uint8)
        py1, py2, px1, px2 = (np.array([head_pad_pos[1], head_pad_pos[1] + ih, head_pad_pos[0], head_pad_pos[0] + iw]) / scale).astype(np.int64)
        scale_size = (int(head_pad_size[0] / scale), int(head_pad_size[1] / scale))

        for rst, tag in zip(head_output.images, head_tag_list):
            rst = smart_resize(rst, scale_size)[py1:py2, px1:px2]
            full = canvas.copy()
            full[hy1 : hy1 + rst.shape[0], hx1 : hx1 + rst.shape[1]] = rst
            save_path = output_dir / f"{tag}.png"
            Image.fromarray(full).save(save_path)
            saved_paths.append(save_path)
    else:
        raise ValueError(f"Unsupported LayerDiff tag version: {tag_version}")

    manifest_dir = output_dir / "layerdiff"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.json"
    manifest = {
        "source_path": str(source_path),
        "resolution": int(resolution),
        "tag_version": tag_version,
        "parts": [path.name for path in sorted(saved_paths)],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"src_image": src_image_path, "manifest": manifest_path}
