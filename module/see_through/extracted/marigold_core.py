# Extracted and adapted from:
# - inference/scripts/inference_psd.py
# - common/utils/inference_utils.py
# Upstream repository: shitagaki-lab/see-through
# Notes: globals removed, project-local IO/state conventions applied.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..see_through_profile import DEFAULT_DEPTH_INFERENCE_STEPS, DEFAULT_SEED, normalize_quant_mode
from ..vendor_bootstrap import ensure_vendor_imports
from utils.transformer_loader import (
    is_quantized_pretrained_component,
    load_pretrained_component,
    move_pretrained_component,
)


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


def load_marigold_pipeline(
    *,
    repo_id: str,
    runtime_context: Any,
    group_offload: bool = False,
    quant_mode: str = "none",
    console: Any | None = None,
) -> Any:
    ensure_vendor_imports()
    quant_mode = normalize_quant_mode(quant_mode)

    from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
    from modules.marigold import MarigoldDepthPipeline

    if console is not None:
        console.print(f"[cyan]Loading Marigold pipeline:[/cyan] {repo_id} [dim](quant_mode={quant_mode})[/dim]")

    if quant_mode != "none" and not _is_cuda_device(getattr(runtime_context, "device", "cpu")):
        raise RuntimeError("See-through quant_mode=nf4 requires CUDA for marigold inference.")

    load_kwargs: dict[str, Any] = {}
    if quant_mode != "none":
        load_kwargs["torch_dtype"] = runtime_context.dtype

    unet = load_pretrained_component(
        UNetFrameConditionModel,
        repo_id,
        console=console,
        component_name="Marigold UNet",
        subfolder="unet",
        **load_kwargs,
    )
    pipeline = load_pretrained_component(
        MarigoldDepthPipeline,
        repo_id,
        console=console,
        component_name="Marigold pipeline",
        unet=unet,
        **load_kwargs,
    )

    if quant_mode == "none":
        pipeline.to(device=runtime_context.device, dtype=runtime_context.dtype)
    else:
        move_pretrained_component(getattr(pipeline, "vae", None), device=runtime_context.device, dtype=runtime_context.dtype)
        move_pretrained_component(getattr(pipeline, "unet", None), device=runtime_context.device, dtype=runtime_context.dtype)
        text_encoder = getattr(pipeline, "text_encoder", None)
        if text_encoder is not None and not is_quantized_pretrained_component(text_encoder):
            move_pretrained_component(text_encoder, device=runtime_context.device, dtype=runtime_context.dtype)

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
            f"[green]Marigold pipeline ready:[/green] {repo_id} "
            f"(device={runtime_context.device}, dtype={runtime_context.dtype}, group_offload={group_offload})"
        )
    return pipeline


def _manifest_resolution_value(height: int, width: int) -> int | list[int]:
    if height == width:
        return int(height)
    return [int(height), int(width)]


def _resolve_depth_target_size(*, resolution_depth: int, src_height: int, src_width: int) -> list[int]:
    value = int(resolution_depth)
    if value == -1:
        return [int(src_height), int(src_width)]
    if value <= 0:
        raise ValueError("resolution_depth must be -1 or a positive integer.")
    return [value, value]


def run_marigold_phase(
    *,
    source_path: Path,
    output_dir: Path,
    pipeline: Any,
    resolution_depth: int,
    inference_steps_depth: int = DEFAULT_DEPTH_INFERENCE_STEPS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Path]:
    ensure_vendor_imports()

    import numpy as np
    import torch
    from PIL import Image

    from utils.cv import img_alpha_blending, smart_resize
    from utils.inference_utils import VALID_BODY_PARTS_V2
    from utils.io_utils import dict2json, json2dict
    from utils.torch_utils import seed_everything

    output_dir.mkdir(parents=True, exist_ok=True)
    src_img_path = output_dir / "src_img.png"
    fullpage = np.array(Image.open(src_img_path).convert("RGBA"))
    src_height, src_width = fullpage.shape[:2]
    effective_depth_resolution = _resolve_depth_target_size(
        resolution_depth=resolution_depth,
        src_height=src_height,
        src_width=src_width,
    )
    target_depth_size = tuple(int(value) for value in effective_depth_resolution)

    img_list: list[np.ndarray] = []
    blended_alpha = np.zeros((src_height, src_width), dtype=np.float32)
    empty_array = np.zeros((src_height, src_width, 4), dtype=np.uint8)

    compose_list = {"eyes": ["eyewhite", "irides", "eyelash", "eyebrow"], "hair": ["back hair", "front hair"]}
    for tag in VALID_BODY_PARTS_V2:
        tag_path = output_dir / f"{tag}.png"
        if tag_path.exists():
            tag_arr = np.array(Image.open(tag_path))
            tag_arr[..., -1][tag_arr[..., -1] < 15] = 0
            img_list.append(tag_arr)
        else:
            img_list.append(empty_array.copy())

    compose_dict: dict[str, dict[str, Any]] = {}
    for composed_tag, tags in compose_list.items():
        composed_images: list[np.ndarray] = []
        composed_labels: list[str] = []
        for tag in tags:
            tag_path = output_dir / f"{tag}.png"
            if tag_path.exists():
                tag_arr = np.array(Image.open(tag_path))
                tag_arr[..., -1][tag_arr[..., -1] < 15] = 0
                composed_images.append(tag_arr)
                composed_labels.append(tag)
        if composed_images:
            img_list[VALID_BODY_PARTS_V2.index(composed_tag)] = img_alpha_blending(composed_images, premultiplied=False)
            compose_dict[composed_tag] = {"taglist": composed_labels, "imlist": composed_images}

    for img in img_list:
        blended_alpha += img[..., -1].astype(np.float32) / 255
    blended_alpha = (np.clip(blended_alpha, 0, 1) * 255).astype(np.uint8)
    fullpage[..., -1] = blended_alpha
    img_list.append(fullpage)

    seed_everything(int(seed))
    img_list_input = img_list
    src_rescaled = target_depth_size != (src_height, src_width)
    if src_rescaled:
        img_list_input = [smart_resize(img, target_depth_size) for img in img_list]

    pipeline_kwargs: dict[str, Any] = {
        "color_map": None,
        "show_progress_bar": False,
        "img_list": img_list_input,
    }
    if int(inference_steps_depth) >= 1:
        pipeline_kwargs["denoising_steps"] = int(inference_steps_depth)

    pipe_out = pipeline(**pipeline_kwargs)
    depth_pred = pipe_out.depth_tensor.to(device="cpu", dtype=torch.float32).numpy()
    if src_rescaled:
        depth_pred = [smart_resize(depth, (src_height, src_width)) for depth in depth_pred]

    drawables = [{"img": img, "depth": depth} for img, depth in zip(img_list[:-1], depth_pred[:-1])]
    blended = img_alpha_blending(drawables, premultiplied=False)

    info_path = output_dir / "info.json"
    info = json2dict(str(info_path)) if info_path.exists() else {"parts": {}}
    parts = info["parts"]

    for index, depth in enumerate(depth_pred[:-1]):
        depth_u8 = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
        tag = VALID_BODY_PARTS_V2[index]
        if tag in compose_dict:
            mask = blended_alpha > 256
            for composed_name, composed_img in zip(
                compose_dict[tag]["taglist"][::-1],
                compose_dict[tag]["imlist"][::-1],
            ):
                mask_local = composed_img[..., -1] > 15
                mask_invis = np.bitwise_and(mask, mask_local)
                depth_local = np.full((src_height, src_width), fill_value=255, dtype=np.uint8)
                depth_local[mask_local] = depth_u8[mask_local]
                if np.any(mask_invis):
                    visible_mask = np.bitwise_and(mask_local, np.bitwise_not(mask_invis))
                    if np.any(visible_mask):
                        depth_local[mask_invis] = np.median(depth_u8[visible_mask])
                    else:
                        depth_local[mask_invis] = np.median(depth_u8[mask_local])
                mask = np.bitwise_or(mask, mask_local)
                Image.fromarray(depth_local).save(output_dir / f"{composed_name}_depth.png")
                parts[composed_name] = parts.get(composed_name, {})
            continue

        Image.fromarray(depth_u8).save(output_dir / f"{tag}_depth.png")
        parts[tag] = parts.get(tag, {})

    dict2json(info, str(info_path))
    Image.fromarray(blended).save(output_dir / "reconstruction.png")

    full_depth_u8 = (np.clip(depth_pred[-1], 0, 1) * 255).astype(np.uint8)
    depth_dir = output_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    depth_image_path = depth_dir / "depth.png"
    Image.fromarray(full_depth_u8).save(depth_image_path)
    manifest_path = depth_dir / "manifest.json"
    manifest = {
        "source_path": str(source_path),
        "resolution": _manifest_resolution_value(src_height, src_width),
        "resolution_depth": _manifest_resolution_value(*target_depth_size),
        "parts": sorted(path.name for path in output_dir.glob("*_depth.png")),
        "info_path": str(info_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"depth": depth_image_path, "manifest": manifest_path}
