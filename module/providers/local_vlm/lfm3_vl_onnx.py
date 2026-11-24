"""Image preprocessing and graph inputs for LiquidAI's two-graph LFM3-VL export."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from module.onnx_runtime.artifacts import build_component_filename, download_repo_file_set
from module.onnx_runtime.config import OnnxRuntimeConfig
from module.onnx_runtime.session import build_execution_providers

_VISION_PRECISION = {"q4": "q8", "q8": "fp16", "q4f16": "fp16", "q8f16": "fp16", "fp16": "fp16", "fp32": "fp32"}
_PROCESSOR_FILES = ("processor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def _has_cuda_device() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def resolve_bundle(runtime: OnnxRuntimeConfig, variant: str) -> tuple[OnnxRuntimeConfig, dict[str, str]]:
    providers = build_execution_providers(runtime)
    names = [ep[0] if isinstance(ep, tuple) else ep for ep in providers]
    if runtime.execution_provider.strip().lower() == "auto":
        # Use the exporter's native CUDA/CPU path, not TensorRT engine compilation.
        accelerated = "CUDAExecutionProvider" in names and _has_cuda_device()
        runtime = replace(runtime, execution_provider="cuda" if accelerated else "cpu")
    else:
        accelerated = names[0] in {
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
            "NvTensorRtRtxExecutionProvider",
            "ROCMExecutionProvider",
        }

    precision = variant.strip().lower()
    if precision in {"", "auto"}:
        precision = "q4f16" if accelerated else "q4"
    if precision not in _VISION_PRECISION:
        raise ValueError(f"Unknown LFM3 decoder_variant {variant!r}; use auto or {', '.join(_VISION_PRECISION)}")
    return runtime, {
        "vision_encoder": build_component_filename("vision_encoder", _VISION_PRECISION[precision]),
        "decoder": build_component_filename("decoder", precision),
    }


def load_preprocessing(model_id: str, local_dir: Path, *, force_download: bool, logger) -> dict[str, Any]:
    from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor

    paths = download_repo_file_set(
        model_id,
        {name: name for name in _PROCESSOR_FILES},
        local_dir=local_dir,
        force_download=force_download,
        logger=logger,
    )
    # The ONNX repo ships processor settings but not the custom AutoProcessor Python files.
    settings = json.loads(paths["processor_config.json"].read_text(encoding="utf-8"))["image_processor"]
    settings.pop("auto_map", None)
    settings.pop("image_processor_type", None)
    # Transformers 4.56 resets the size dict unless the same bounds are supplied as pixel limits.
    settings.update(min_pixels=settings["size"]["shortest_edge"], max_pixels=settings["size"]["longest_edge"])
    return {
        "processor": Qwen2VLImageProcessor(**settings),
        # The export's empty extra_special_tokens list must be a mapping on Transformers 4.x.
        "tokenizer": PreTrainedTokenizerFast.from_pretrained(local_dir, local_files_only=True, extra_special_tokens={}),
    }


def prepare_prefill(
    processor: Any,
    tokenizer: Any,
    vision: Any,
    messages: list[dict],
    images: list[Image.Image],
    config: dict,
) -> dict[str, np.ndarray]:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = np.asarray(tokenizer.encode(prompt, add_special_tokens=False), dtype=np.int64)
    positions = np.flatnonzero(ids == tokenizer.convert_tokens_to_ids("<|image|>"))
    if len(positions) != len(images):
        raise ValueError(f"LFM3 prompt has {len(positions)} image placeholders for {len(images)} images")

    merge = processor.merge_size
    pixels_per_token = (processor.patch_size * merge) ** 2
    size = {edge: processor.size[edge] for edge in ("shortest_edge", "longest_edge")}
    for setting, edge in (("min_image_tokens", "shortest_edge"), ("max_image_tokens", "longest_edge")):
        if config.get(setting) not in (None, ""):
            size[edge] = int(config[setting]) * pixels_per_token
    if not 0 < size["shortest_edge"] <= size["longest_edge"]:
        raise ValueError("LFM3 requires 0 < min_image_tokens <= max_image_tokens")

    repeats = np.ones(len(ids), dtype=np.intp)
    features = []
    for position, image in zip(positions, images):
        inputs = processor(images=image, size=size, return_tensors="np")
        ((t, h, w),) = inputs["image_grid_thw"]
        if t != 1 or h % merge or w % merge:
            raise ValueError(f"Invalid LFM3 image patch grid {(t, h, w)} for merge_size {merge}")
        pixels = np.asarray(inputs["pixel_values"], dtype=np.float32)
        # Qwen orders patches by merge block; the vision graph takes the unflattened 5-D grid.
        blocks = pixels.reshape(h // merge, w // merge, merge, merge, pixels.shape[-1])
        rows = vision.run(None, {"pixel_values": np.ascontiguousarray(blocks)})[0]
        repeats[position] = h * w // merge**2
        if len(rows) != repeats[position]:
            raise ValueError(f"LFM3 vision encoder returned {len(rows)} rows for {repeats[position]} image tokens")
        features.append(rows)

    image_features = np.concatenate(features)
    return {
        "input_ids": np.repeat(ids, repeats)[None, :],
        "image_features": image_features,
        "video_features": image_features[:0],
        "logits_to_keep": np.array([1], dtype=np.int64),
    }
