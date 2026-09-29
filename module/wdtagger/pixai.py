"""PixAI ONNX downloads and the official RGB preprocessing contract."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from module.wdtagger.taxonomy import LabelData

PIXAI_REPO_ID = "bdsqlsz/pixai-tagger-v1.0-ONNX"
PIXAI_SOURCE_REPO_ID = "pixai-labs/pixai-tagger-v1.0"
PIXAI_THRESHOLDS = {"general": 0.17, "character": 0.27, "style": 0.15, "copyright": 0.24, "meta": 0.17, "rating": 0.41}


def is_pixai_repo(repo_id: str) -> bool:
    return str(repo_id).strip() in {PIXAI_REPO_ID, PIXAI_SOURCE_REPO_ID}


@dataclass(frozen=True)
class PixaiInferenceContext:
    image_size: int = 1008
    input_name: str = "pixel_values"
    output_name: str = "logits"
    category_thresholds: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PixaiOnnxBundle:
    session: Any
    providers: tuple
    inference_context: PixaiInferenceContext
    label_data: LabelData


def load_pixai_config(path: Path) -> tuple[LabelData, PixaiInferenceContext]:
    data = json.loads(path.read_text(encoding="utf-8"))
    names = data["tags"]
    categories = {}
    offset = 0
    for category, count in data["tags_split"]:
        if category in categories or not isinstance(count, int) or count < 0:
            raise ValueError("Invalid PixAI tags_split")
        categories[category] = np.arange(offset, offset + count, dtype=np.int64)
        offset += count
    if offset != len(names) or not all(isinstance(name, str) and name for name in names):
        raise ValueError("PixAI tags_split does not match the tag vocabulary")
    size = int(data["img_size"])
    if size <= 0:
        raise ValueError("PixAI image_size must be positive")
    labels = LabelData(names, categories, {int(i): cat for cat, indices in categories.items() for i in indices})
    thresholds = {category: float(value) for category, value in (data.get("category_best_threshold") or {}).items()}
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in thresholds.values()):
        raise ValueError("PixAI category thresholds must be finite values between 0 and 1")
    return labels, PixaiInferenceContext(image_size=size, category_thresholds=thresholds)


def resolve_pixai_thresholds(args, context: PixaiInferenceContext) -> dict[str, float]:
    return {**PIXAI_THRESHOLDS, **context.category_thresholds, **getattr(args, "pixai_threshold_overrides", {})}


def preprocess_pixai_images(images: list[Image.Image], image_size: int = 1008) -> np.ndarray:
    import torch
    from torchvision.transforms import functional as vf

    tensors = []
    for image in images:
        if image.mode != "RGB":
            rgba = image.convert("RGBA")
            canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            canvas.alpha_composite(rgba)
            image = canvas.convert("RGB")
        tensor = vf.to_tensor(image)
        h, w = tensor.shape[-2:]
        if (h, w) != (image_size, image_size):
            ratio = min(image_size / h, image_size / w)
            new_h, new_w = int(h * ratio), int(w * ratio)
            if min(new_h, new_w) < 1:
                raise ValueError(f"Image aspect ratio is too extreme for PixAI: {w}x{h}")
            tensor = vf.resize(tensor, [new_h, new_w], antialias=True)
            ph, pw = image_size - new_h, image_size - new_w
            tensor = vf.pad(tensor, [pw // 2, ph // 2, pw - pw // 2, ph - ph // 2], 0)
        tensors.append(vf.normalize(tensor, [0.5] * 3, [0.5] * 3))
    return torch.stack(tensors).numpy()


def load_pixai_bundle(*, model_dir, runtime_config, repo_id=PIXAI_REPO_ID, logger=None) -> PixaiOnnxBundle:
    from module.onnx_runtime import OnnxModelSpec, OnnxRuntimeConfig, load_single_model_bundle

    directory = Path(model_dir) / str(repo_id).strip().replace("/", "_")
    runtime_data = (runtime_config or OnnxRuntimeConfig()).as_dict()
    # The exported FP32 graph is validated on CUDA/CPU, not implicit FP16 TensorRT.
    if runtime_data["execution_provider"].lower() == "auto":
        runtime_data["execution_provider"] = "cuda"
    runtime_data["provider_options"]["cuda"].setdefault("use_tf32", False)
    bundle = load_single_model_bundle(
        spec=OnnxModelSpec(
            repo_id=PIXAI_REPO_ID,
            onnx_filename="model.onnx",
            local_dir=directory,
            bundle_key=f"wdtagger:{repo_id}",
            support_files={"config": "config.json"},
        ),
        runtime_config=OnnxRuntimeConfig.from_mapping(runtime_data),
        logger=logger,
    )
    labels, context = load_pixai_config(bundle.support_paths["config"])
    session = bundle.session
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("PixAI ONNX must have one image input and one logits output")
    input_meta, output_meta = inputs[0], outputs[0]
    if input_meta.shape[1:] != [3, context.image_size, context.image_size] or output_meta.shape[-1] != len(labels.names):
        raise ValueError("PixAI ONNX shapes do not match config.json")
    context = replace(context, input_name=input_meta.name, output_name=output_meta.name)
    if logger:
        logger(f"[green]Using PixAI ONNX[/green] {bundle.model_path}")
    return PixaiOnnxBundle(session, tuple(bundle.providers), context, labels)


def process_pixai_batch(images, session, context: PixaiInferenceContext) -> np.ndarray:
    pixels = preprocess_pixai_images(images, context.image_size)
    logits = session.run([context.output_name], {context.input_name: pixels})[0]
    if not np.isfinite(logits).all():
        raise ValueError("PixAI returned non-finite logits")
    return 1 / (1 + np.exp(-np.clip(logits.astype(np.float32), -80, 80)))
