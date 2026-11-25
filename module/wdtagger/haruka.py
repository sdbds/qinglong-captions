"""Haruka's indexed vocabulary and RGB/ImageNet ONNX input contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from module.wdtagger.model_defaults import HARUKA_THRESHOLDS
from module.wdtagger.taxonomy import LabelData

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


@dataclass(frozen=True)
class HarukaInferenceContext:
    image_size: int
    num_tags: int
    input_name: str
    output_name: str


def load_haruka_metadata(path: Path, session) -> tuple[LabelData, HarukaInferenceContext]:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    size, count = data["img_size"], data["total_tags"]
    indexed, categories = data["idx_to_tag"], data["tag_to_category"]
    if not isinstance(size, int) or size < 1 or not isinstance(count, int) or count < 1:
        raise ValueError("Invalid Haruka image size or vocabulary size")
    if not isinstance(indexed, dict) or len(indexed) != count or not isinstance(categories, dict):
        raise ValueError("Invalid Haruka vocabulary maps")

    names = []
    by_category = {category: [] for category in HARUKA_THRESHOLDS}
    index_to_category = {}
    for index in range(count):
        tag = indexed.get(str(index))
        if not isinstance(tag, str) or not tag or categories.get(tag) not in by_category:
            raise ValueError(f"Missing or invalid Haruka tag/category at index {index}")
        category = categories[tag]
        names.append(tag)
        by_category[category].append(index)
        index_to_category[index] = category
    if len(set(names)) != count:
        raise ValueError("Duplicate tags in Haruka vocabulary")

    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Haruka ONNX must have one image input and one logits output")
    pixels, logits = inputs[0], outputs[0]
    if (
        pixels.type != "tensor(float)"
        or pixels.shape[1:] != [3, size, size]
        or logits.type != "tensor(float)"
        or logits.shape[1:] != [count]
    ):
        raise ValueError("Haruka ONNX shapes or types do not match haruka-tagger_metadata.json")
    labels = LabelData(names, {key: np.asarray(ids, dtype=np.int64) for key, ids in by_category.items()}, index_to_category)
    return labels, HarukaInferenceContext(size, count, pixels.name, logits.name)


def _letterbox(image: Image.Image, size: int) -> Image.Image:
    if image.mode != "RGB":
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        image = background.convert("RGB")
    if image.size == (size, size):
        return image
    width, height = image.size
    scale = size / max(width, height)
    width, height = max(1, int(width * scale)), max(1, int(height * scale))
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image.resize((width, height), Image.Resampling.BICUBIC), ((size - width) // 2, (size - height) // 2))
    return canvas


def process_haruka_batch(images, session, context: HarukaInferenceContext) -> np.ndarray:
    size = context.image_size
    pixels = np.empty((len(images), 3, size, size), dtype=np.float32)
    for row, image in enumerate(images):
        pixels[row] = np.asarray(_letterbox(image, size)).transpose(2, 0, 1)
    pixels /= 255.0
    pixels -= _MEAN
    pixels /= _STD
    logits = session.run([context.output_name], {context.input_name: pixels})[0]
    if logits.shape != (len(images), context.num_tags) or not np.isfinite(logits).all():
        raise ValueError("Haruka returned invalid logits for the input batch and vocabulary")
    return 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
