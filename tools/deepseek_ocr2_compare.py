"""Prepare a synthetic two-page fixture or compare captured DeepSeek-OCR-2 outputs.

Each independent run has a run.json with backend ("legacy" or "native") and
cases {"png": ["png/1", "png/2"], "pdf": ["pdf/1", "pdf/2"]} in actual page
order. Each page directory contains raw.txt (original grounded model output),
result.md or result.mmd, and the actual crop linked in that Markdown. Optionally
include the SHA-256 of truth.json as fixture_sha256 in each run manifest.
Native captures must retain the provider's per-page grounding.json. When a run
records a prompt or weights SHA-256, every run must record the same digest.
For a native upgrade, supply reference-native runs and both exact native
Transformers versions. This mode requires matching input and CUDA runtime
provenance; cross-version comparisons are not treated as same-version reruns.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import random
import re
import sys
import unicodedata
from itertools import combinations, product
from pathlib import Path, PureWindowsPath
from urllib.parse import unquote

from PIL import Image, ImageDraw, ImageFont

SIZE = (960, 1184)
FIGURES = [(96, 436, 864, 934), (96, 402, 864, 900)]
TEXT = [
    ("青石海湾", ["第一站  青石海湾", "清晨潮水退去，白色灯塔映在海面。", "记录员在岸边测量水位。", "图一  海岸灯塔"]),
    ("松林高地", ["第二站  松林高地", "午后云层散开，山间湖水映出群峰。", "考察队沿湖记录积雪厚度。", "图二  山地湖泊"]),
]


def cjk_font(requested=None):
    options = (
        [requested]
        if requested
        else [
            Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/msyh.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/System/Library/Fonts/PingFang.ttc",
        ]
    )
    for path in options:
        if path and Path(path).is_file():
            font = ImageFont.truetype(str(path), 36)
            missing = bytes(font.getmask("\U0010ffff"))
            if all(bytes(font.getmask(c)) != missing for c in "青石海湾松林高地图"):
                return Path(path)
    raise ValueError("A font with Chinese glyphs is required; pass --font PATH.")


def _landscape(w, h, station):
    import numpy as np

    rng = random.Random(9400 + station)
    noise = np.random.default_rng(121 + station)
    y = np.linspace(0, 1, h)
    colors = (
        [(42, 105, 155), (194, 216, 225), (113, 161, 178), (36, 92, 115)]
        if station == 0
        else [(40, 94, 143), (212, 225, 232), (83, 140, 147), (26, 77, 86)]
    )
    sky = np.stack([np.interp(y, [0, 0.48, 0.55, 1], [c[k] for c in colors]) for k in range(3)], axis=-1)
    image = Image.fromarray(np.broadcast_to(sky[:, None, :], (h, w, 3)).astype("uint8").copy())
    draw = ImageDraw.Draw(image)
    if station == 0:
        # A distant headland, fine ripples, and a sunlit lighthouse on a rocky promontory.
        draw.polygon(
            [(0, 236), (88, 220), (162, 226), (254, 211), (342, 207), (405, 225), (490, 237), (w, 243), (w, 252), (0, 249)],
            fill=(112, 142, 148),
        )
        for _ in range(2400):
            x, yy = rng.randrange(w), rng.randrange(254, h)
            v = rng.randrange(35, 100) + (h - yy) // 9
            draw.line((x, yy, x + rng.randrange(2, 17), yy), fill=(v, v + 40, v + 54))
        shore = [(i, int(325 + i * 0.17 + 16 * np.sin(i / 43) + rng.randrange(-7, 8))) for i in range(-12, w + 12, 8)]
        draw.polygon(shore + [(w, h), (0, h)], fill=(72, 71, 62))
        for _ in range(7600):
            x, yy = rng.randrange(w), rng.randrange(315, h)
            if yy > 331 + x * 0.17 + 16 * np.sin(x / 43):
                v = rng.randrange(45, 110)
                draw.line((x, yy, x + rng.randrange(1, 9), yy - 1), fill=(v + 8, v + 6, v - 1))
        draw.polygon([(225, 312), (239, 147), (265, 147), (280, 329)], fill=(236, 230, 210))
        draw.polygon([(254, 148), (265, 147), (280, 329), (258, 324)], fill=(187, 183, 165))
        for yy in (178, 220, 263):
            draw.rectangle((247, yy, 252, yy + 12), fill=(49, 65, 68))
        draw.rectangle((232, 137, 272, 150), fill=(62, 65, 59))
        draw.rectangle((239, 109, 264, 136), fill=(151, 185, 188), outline=(53, 60, 58), width=3)
        draw.polygon([(232, 109), (250, 94), (271, 109)], fill=(138, 65, 48))
    else:
        ridge = [
            (0, 248),
            (63, 216),
            (113, 176),
            (148, 193),
            (209, 151),
            (241, 161),
            (316, 85),
            (335, 94),
            (373, 60),
            (417, 118),
            (458, 146),
            (521, 106),
            (565, 141),
            (592, 130),
            (650, 180),
            (702, 152),
            (w, 209),
        ]
        draw.polygon(ridge + [(w, 286), (0, 282)], fill=(97, 112, 122))
        for j in (2, 4, 6, 8, 11, 13, 15):
            x, yy = ridge[j]
            draw.polygon(
                [(x, yy), (x + 24, yy + 31), (x + 12, yy + 23), (x - 3, yy + 35), (x + 1, yy + 16), (x - 23, yy + 29)],
                fill=(229, 233, 228),
            )
        for _ in range(1800):
            x, yy = rng.randrange(w), rng.randrange(237, 285)
            k = rng.randrange(3, 14)
            v = rng.randrange(32, 77)
            draw.polygon([(x, yy - k), (x + k // 3, yy), (x - k // 3, yy)], fill=(v - 5, v + 17, v + 7))
        for _ in range(2700):
            x, yy = rng.randrange(w), rng.randrange(288, h)
            v = rng.randrange(46, 105) + (h - yy) // 14
            draw.line((x, yy, x + rng.randrange(3, 23), yy), fill=(v - 8, v + 35, v + 38))
    pixels = np.asarray(image).astype("int16") + noise.normal(0, 3, (h, w, 1)).astype("int16")
    return Image.fromarray(np.clip(pixels, 0, 255).astype("uint8"))


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(out, font=None):
    import fitz

    out = Path(out)
    font = cjk_font(font)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"Fixture output must be empty: {out}")
    fonts = [ImageFont.truetype(str(font), size) for size in (62, 40, 40, 34)]
    pages = []
    with fitz.open() as document:
        for n, ((marker, lines), box) in enumerate(zip(TEXT, FIGURES), 1):
            img = Image.new("RGB", SIZE, "white")
            draw = ImageDraw.Draw(img)
            for text, y, face in zip(lines, (99, 247, 313, box[3] + 30), fonts):
                draw.text((96, y), text, font=face, fill=(27, 31, 33))
            img.paste(_landscape(box[2] - box[0], box[3] - box[1], n - 1), box[:2])
            file = out / f"page-{n}.png"
            img.save(file)
            # The production PDF reader defaults to 144 dpi; half-point dimensions give a 1:1 raster.
            page = document.new_page(width=SIZE[0] / 2, height=SIZE[1] / 2)
            page.insert_image(page.rect, filename=str(file))
            pages.append(dict(file=file.name, sha256=_digest(file), marker=marker, lines=lines, figure_box_px=box))
        pdf = out / "pages.pdf"
        document.save(str(pdf), deflate=True, no_new_id=True)
    truth = {"schema": 1, "size": list(SIZE), "pdf": pdf.name, "pdf_sha256": _digest(pdf), "pages": pages}
    _json(out / "truth.json", truth)
    return truth


_GROUND = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>(.*?)<\|/det\|>", re.DOTALL)
_IMAGES = re.compile(
    r"""!\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))(?:\s+["'][^"']*["'])?\s*\)"""
    r"""|<img\b[^>]*?\bsrc\s*=\s*["']([^"']+)["'][^>]*>""",
    re.IGNORECASE,
)
_FIGURE_LABELS = {"image", "figure", "picture", "photograph", "photo", "illustration", "图片", "图像", "插图", "照片"}
_ABSOLUTE = {
    "text_accuracy": 0.90,
    "raw_text_accuracy": 0.90,
    "page_order": 1,
    "box_iou": 0.75,
    "crop_pixel_similarity": 0.90,
    "crop_correlation": 0.85,
    "crop_aspect_similarity": 0.88,
    "crop_resolution_ratio": 0.5,
    "min_region_text_accuracy": 0.80,
    "min_region_box_iou": 0.50,
    "min_region_ink_coverage": 0.80,
}
_AGREEMENT = {
    "text_similarity": 0.95,
    "box_iou": 0.85,
    "crop_pixel_similarity": 0.90,
    "min_region_text_similarity": 0.95,
    "min_region_box_iou": 0.85,
}
_REPEAT = {
    "text_similarity": 1.0,
    "box_iou": 0.98,
    "crop_pixel_similarity": 0.98,
    "min_region_text_similarity": 1.0,
    "min_region_box_iou": 0.98,
}
_EXPECTED_ERRORS = (ValueError, OSError, TypeError, KeyError, IndexError)


def _inside(root, relative):
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("expected a nonempty relative path")
    relative = html.unescape(unquote(relative)).replace("\\", "/")
    if ":" in relative or PureWindowsPath(relative).anchor or ".." in Path(relative).parts:
        raise ValueError(f"expected an in-directory relative path: {relative}")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"relative path escapes its directory: {relative}")
    return target


def _normal(text):
    text = _IMAGES.sub("", _GROUND.sub("", text))
    text = re.sub(r"<[^>]*>", "", html.unescape(text))
    normalized = "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())
    if len(normalized) > 2048:
        raise ValueError("OCR result is implausibly long for this synthetic fixture")
    return normalized


def _edits(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def _iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    total = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / total if total else 0


def _normalized_box(box):
    return tuple(min(v / 999, 1) for v in box)


def _regions(raw):
    regions = []
    matches = list(_GROUND.finditer(raw))
    for i, marker in enumerate(matches):
        label = marker[1].strip()
        candidates = json.loads(marker[2])
        if not isinstance(candidates, list) or not candidates or not label:
            raise ValueError("invalid grounded region")
        if candidates and isinstance(candidates[0], (int, float)):
            candidates = [candidates]
        for box in candidates:
            if (
                not isinstance(box, list)
                or len(box) != 4
                or not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1000 for v in box)
                or box[0] >= box[2]
                or box[1] >= box[3]
            ):
                raise ValueError(f"malformed normalized {label} box")
        tail = raw[marker.end() : matches[i + 1].start() if i + 1 < len(matches) else len(raw)]
        regions.append(dict(label=label, boxes=candidates, text=_normal(tail)))
    return regions


def _figure(regions):
    boxes = [b for r in regions if r["label"].casefold() in _FIGURE_LABELS for b in r["boxes"]]
    if len(boxes) != 1:
        raise ValueError(f"expected one figure box, found {len(boxes)}")
    return _normalized_box(boxes[0])


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _ink_lines(source, figure_box):
    """Recover ink geometry independently of the font, OCR output, or model coordinates."""
    import numpy as np

    ink = np.asarray(source.convert("L")) < 160
    x1, y1, x2, y2 = figure_box
    ink[y1:y2, x1:x2] = False
    rows = np.flatnonzero(ink.any(axis=1))
    bands = np.split(rows, np.flatnonzero(np.diff(rows) > max(2, round(source.height * 0.007))) + 1)
    if len(bands) != 4 or not all(len(band) for band in bands):
        raise ValueError("fixture must contain four visually separate lines of text ink outside its figure")
    result = []
    for band in bands:
        top, bottom = int(band[0]), int(band[-1]) + 1
        columns = np.flatnonzero(ink[top:bottom].any(axis=0))
        left, right = int(columns[0]), int(columns[-1]) + 1
        result.append(((left, top, right, bottom), ink[top:bottom, left:right].copy()))
    return result


def _text_regions(regions, truth_lines, ink_lines, size):
    import numpy as np

    text = [r for r in regions if r["label"].casefold() not in _FIGURE_LABELS]
    if len(truth_lines) != 4 or len(text) not in (3, 4):
        raise ValueError("expected grounded title, one or two body regions, and a figure caption")
    spans = [(0,), (1, 2), (3,)] if len(text) == 3 else [(0,), (1,), (2,), (3,)]
    axes = (*size, *size)
    roles = {}
    for segment, lines in zip(text, spans):
        role = "title" if lines[0] == 0 else "caption" if lines[0] == 3 else "body"
        boxes = [_normalized_box(b) for b in segment["boxes"]]
        bbox = _union(boxes)
        true_box = _union([ink_lines[line][0] for line in lines])
        truth = _normal("".join(truth_lines[line] for line in lines))
        accuracy = max(0, 1 - _edits(truth, segment["text"]) / max(1, len(truth)))
        iou = _iou(bbox, [v / a for v, a in zip(true_box, axes)])
        pixel_boxes = [tuple(round(v * a) for v, a in zip(b, axes)) for b in boxes]
        hits = [False] * len(boxes)
        coverage = []
        for line in lines:
            (x1, y1, x2, y2), ink = ink_lines[line]
            covered = np.zeros_like(ink)
            for j, (left, top, right, bottom) in enumerate(pixel_boxes):
                x, y = max(left, x1) - x1, max(top, y1) - y1
                xx, yy = min(right, x2) - x1, min(bottom, y2) - y1
                if x < xx and y < yy:
                    hits[j] |= bool(ink[y:yy, x:xx].any())
                    covered[y:yy, x:xx] = True
            coverage.append(int(np.count_nonzero(ink & covered)) / int(np.count_nonzero(ink)))
        # A stray extra detection must not be hidden by the correct primary box.
        fraction = min(coverage) if all(hits) else 0.0
        entry = roles.setdefault(role, dict(text="", boxes=[], groups=[], ink_coverage=1.0, box_iou=1.0, text_accuracy=1.0))
        entry["text"] += segment["text"]
        entry["boxes"].extend(boxes)
        entry["groups"].append(
            dict(
                label=segment["label"],
                lines=[n + 1 for n in lines],
                truth_box_pixels=true_box,
                text=segment["text"],
                text_accuracy=accuracy,
                box_iou=iou,
                ink_coverage=coverage,
                predicted_boxes=boxes,
            )
        )
        entry["ink_coverage"] = min(entry["ink_coverage"], fraction)
        entry["box_iou"] = min(entry["box_iou"], iou)
        entry["text_accuracy"] = min(entry["text_accuracy"], accuracy)
    for entry in roles.values():
        entry["box"] = _union(entry.pop("boxes"))
    return roles


def _native_manifest(directory, page_number, size, regions, measurement):
    data = json.loads(_inside(directory, "grounding.json").read_text(encoding="utf-8-sig"))
    w, h = size
    header = dict(schema_version=1, coordinate_system="normalized_0_999", page_number=page_number, width=w, height=h)
    if not isinstance(data, dict) or any(
        data.get(k) != value or type(data.get(k)) is not type(value) for k, value in header.items()
    ):
        raise ValueError("grounding.json: incorrect schema, page number, dimensions or coordinate system")
    detected = [(r, box) for r in regions for box in r["boxes"]]
    recorded = data.get("regions")
    if not isinstance(recorded, list) or len(detected) != len(recorded):
        raise ValueError("grounding.json: region count differs from raw.txt")
    for (r, box), region in zip(detected, recorded):
        pixels = [int(v * a // 999) for v, a in zip(box, (w, h, w, h))]
        if (
            not isinstance(region, dict)
            or region.get("label") != r["label"]
            or region.get("bbox") != box
            or region.get("bbox_pixels") != pixels
            or not isinstance(region.get("text"), str)
            or _normal(region["text"]) != r["text"]
        ):
            raise ValueError("grounding.json: label, text, bbox or bbox_pixels differs from raw.txt and source")
        if r["label"].casefold() in _FIGURE_LABELS:
            if region.get("crop") != measurement["crop"] or measurement["crop_size"] != [
                pixels[2] - pixels[0],
                pixels[3] - pixels[1],
            ]:
                raise ValueError("grounding.json: linked crop path or dimensions disagree")
        elif "crop" in region:
            raise ValueError("grounding.json: unexpected crop for a text region")


def _sample(image):
    import numpy as np

    return np.asarray(image.convert("RGB").resize((128, 80), Image.Resampling.BILINEAR), dtype="float32")


def _pixel_scores(a, b):
    import numpy as np

    similarity = 1 - np.abs(a - b).mean().item() / 255
    x = a - a.mean(axis=(0, 1))
    y = b - b.mean(axis=(0, 1))
    denominator = math.sqrt((x * x).sum().item() * (y * y).sum().item())
    correlation = (x * y).sum().item() / denominator if denominator else 0
    return similarity, min(1.0, max(-1.0, correlation))


def _fixture(path):
    data = json.loads(_inside(path, "truth.json").read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != 1 or len(data["pages"]) != 2:
        raise ValueError("expected a two-page version 1 fixture")
    artifacts = [(data["pdf"], data["pdf_sha256"])] + [(p["file"], p["sha256"]) for p in data["pages"]]
    for name, expected in artifacts:
        if _digest(_inside(path, name)) != expected:
            raise ValueError(f"fixture file hash mismatch: {name}")
    if len({p["marker"] for p in data["pages"]}) != 2 or not all(p["marker"] for p in data["pages"]):
        raise ValueError("two distinct nonempty page markers are required")
    for p in data["pages"]:
        with Image.open(_inside(path, p["file"])) as original:
            p["ink_lines"] = _ink_lines(original, p["figure_box_px"])
    return data


def _score_page(directory, fixture, truth, i):
    raw = _inside(directory, "raw.txt").read_text(encoding="utf-8-sig")
    md = _inside(directory, "result.md")
    if not md.exists():
        md = _inside(directory, "result.mmd")
    markdown = md.read_text(encoding="utf-8-sig")
    regions = _regions(raw)
    bbox = _figure(regions)
    links = [next(value for value in match.groups() if value is not None) for match in _IMAGES.finditer(markdown)]
    if len(links) != 1:
        raise ValueError(f"expected one linked crop in Markdown, found {len(links)}")
    crop_path = _inside(directory, links[0])
    expected = truth["pages"][i]
    reference = _normal("".join(expected["lines"]))
    text, raw_text = _normal(markdown), _normal(raw)
    if not reference:
        raise ValueError("fixture page has empty reference text")
    with Image.open(_inside(fixture, expected["file"])) as source, Image.open(crop_path) as crop:
        w, h = source.size
        if [w, h] != truth["size"]:
            raise ValueError("fixture size differs from source PNG")
        axes = (w, h, w, h)
        actual = tuple(round(p * a) for p, a in zip(bbox, axes))
        if actual[2] <= actual[0] or actual[3] <= actual[1]:
            raise ValueError("figure box is empty at source resolution")
        target_box = tuple(v / a for v, a in zip(expected["figure_box_px"], axes))
        pixels = _sample(crop)
        similarity, corr = _pixel_scores(pixels, _sample(source.crop(actual)))
        truth_similarity, _ = _pixel_scores(pixels, _sample(source.crop(expected["figure_box_px"])))
        aspect = (crop.width / crop.height) / ((actual[2] - actual[0]) / (actual[3] - actual[1]))
        resolution = min(crop.width / (actual[2] - actual[0]), crop.height / (actual[3] - actual[1]))
        size = list(crop.size)
    good_order = _normal(expected["marker"]) in text and _normal(truth["pages"][1 - i]["marker"]) not in text
    distance = _edits(reference, text)
    result = {
        "normalized_text": text,
        "reference_text": reference,
        "character_edits": distance,
        "character_error_rate": distance / len(reference),
        "text_accuracy": max(0, 1 - distance / len(reference)),
        "raw_text_accuracy": max(0, 1 - _edits(reference, raw_text) / len(reference)),
        "page_order": good_order,
        "figure_box_normalized": bbox,
        "box_iou": _iou(bbox, target_box),
        "crop": links[0],
        "crop_size": size,
        "crop_pixel_similarity": similarity,
        "crop_correlation": corr,
        "crop_aspect_similarity": min(aspect, 1 / aspect),
        "crop_resolution_ratio": resolution,
        "crop_vs_entire_reference_figure": truth_similarity,
    }
    try:
        roles = _text_regions(regions, expected["lines"], expected["ink_lines"], truth["size"])
        result.update(
            text_regions=roles,
            min_region_box_iou=min(r["box_iou"] for r in roles.values()),
            min_region_ink_coverage=min(r["ink_coverage"] for r in roles.values()),
            min_region_text_accuracy=min(r["text_accuracy"] for r in roles.values()),
        )
    except ValueError as exc:
        result.update(
            text_regions={}, min_region_box_iou=0, min_region_ink_coverage=0, min_region_text_accuracy=0, grounding_error=str(exc)
        )
    return result, pixels, regions


def compare(fixture, legacy, native, *, reference_native=None, native_version=None, reference_native_version=None):
    """Measure all available runs; missing inputs produce a failed, never a passing, report."""
    fixture = Path(fixture).resolve()
    upgrade = reference_native is not None or native_version is not None or reference_native_version is not None
    groups = {"legacy": legacy, "native": native}
    cross_groups = [("backends", "legacy", "native")]
    report = {
        "schema": "deepseek-ocr2-comparison/v3" if upgrade else "deepseek-ocr2-comparison/v2",
        "passed": False,
        "runs": [],
        "comparisons": [],
        "errors": [],
        "comparable_inputs": {},
        "summary": {"measured_pages": 0, "max_character_error_rate": None},
        "minimums": {
            "absolute": _ABSOLUTE,
            "backends": dict(_AGREEMENT, native_accuracy_delta=-0.03),
            "formats": _AGREEMENT,
            "repeat": _REPEAT,
        },
    }
    errors = report["errors"]
    if upgrade:
        groups["reference-native"] = reference_native or []
        cross_groups.extend([("backends", "legacy", "reference-native"), ("native_versions", "reference-native", "native")])
        report["minimums"]["native_versions"] = dict(report["minimums"]["backends"])
        versions = {"legacy": "4.46.3", "native": native_version, "reference-native": reference_native_version}
        report["expected_transformers"] = versions
        for group in ("native", "reference-native"):
            if not isinstance(versions[group], str) or not re.fullmatch(r"5\.\d+\.\d+", versions[group]):
                errors.append(f"{group}: specify an exact native Transformers 5.x.y version")
        if native_version == reference_native_version:
            errors.append("native and reference-native must identify distinct Transformers versions")
    try:
        truth = _fixture(fixture)
    except _EXPECTED_ERRORS as exc:
        errors.append(f"fixture: {exc}")
        return report
    report["fixture_sha256"] = _digest(fixture / "truth.json")

    def grade(label, scores, minimums):
        for metric, minimum in minimums.items():
            if scores[metric] < minimum:
                errors.append(f"{label}: {metric} {scores[metric]:.4f} < {minimum}")

    observed = {}
    ids = {group: [] for group in groups}
    seen = set()
    page_dirs = set()
    linked_raw = set()
    for group, selected in groups.items():
        backend = "legacy" if group == "legacy" else "native"
        if len(set(Path(path).resolve() for path in selected)) < 2:
            errors.append(f"need at least two distinct {group} runs")
        for n, path in enumerate(selected, 1):
            run, name = Path(path).resolve(), f"{group}-{n}"
            if run in seen:
                errors.append(f"{name}: capture directory must be distinct")
                continue
            seen.add(run)
            ids[group].append(name)
            summary = {"run": name, "backend": backend, "group": group, "path": str(run), "pages": []}
            report["runs"].append(summary)
            try:
                manifest = _inside(run, "run.json")
                meta = json.loads(manifest.read_text(encoding="utf-8-sig"))
                if not isinstance(meta, dict) or meta.get("backend") != backend:
                    raise ValueError(f"backend metadata does not identify {backend}")
                summary["manifest_sha256"] = _digest(manifest)
                for field in ("fixture_sha256", "prompt_sha256", "weights_sha256", "revision", "transformers", "torch", "gpu"):
                    if field in meta:
                        summary[field] = meta[field]
                if upgrade and meta.get("transformers") != versions[group]:
                    errors.append(f"{name}: expected transformers {versions[group]!r}, recorded {meta.get('transformers')!r}")
                if upgrade or "fixture_sha256" in meta:
                    digest = meta.get("fixture_sha256")
                    if not isinstance(digest, str) or digest.lower() != report["fixture_sha256"]:
                        raise ValueError("capture fixture hash (fixture_sha256) missing or different from the evaluated fixture")
                cases = meta["cases"]
                if not isinstance(cases, dict):
                    raise ValueError("cases must map png and pdf to ordered page directories")
                for case in ("png", "pdf"):
                    if not isinstance(cases.get(case), list) or len(cases[case]) != 2:
                        errors.append(f"{name}/{case}: expected two pages in actual processing order")
                        continue
                    for i, relative in enumerate(cases[case]):
                        label = f"{name}/{case}/{i + 1}"
                        try:
                            folder = _inside(run, relative)
                            if folder in page_dirs:
                                raise ValueError("every input needs a distinct capture directory")
                            page_dirs.add(folder)
                            if upgrade:
                                raw_stat = _inside(folder, "raw.txt").stat()
                                if raw_stat.st_nlink > 1:
                                    identity = (raw_stat.st_dev, raw_stat.st_ino)
                                    if not raw_stat.st_ino or identity in linked_raw:
                                        raise ValueError("every input needs a distinct raw prediction file")
                                    linked_raw.add(identity)
                            measurement, pixels, regions = _score_page(folder, fixture, truth, i)
                            page_report = dict(case=case, page=i + 1, **measurement)
                            summary["pages"].append(page_report)
                            observed[name, case, i] = measurement, pixels
                            grade(label, measurement, _ABSOLUTE)
                            if "grounding_error" in measurement:
                                errors.append(f"{label}: {measurement['grounding_error']}")
                            if backend == "native":
                                page_report["native_grounding_verified"] = False
                                _native_manifest(folder, 1 if case == "png" else i + 1, truth["size"], regions, measurement)
                                page_report["native_grounding_verified"] = True
                        except _EXPECTED_ERRORS as exc:
                            errors.append(f"{label}: {exc}")
            except _EXPECTED_ERRORS as exc:
                errors.append(f"{name}: {exc}")

    for field in ("prompt_sha256", "weights_sha256"):
        runs = report["runs"]
        if upgrade or any(field in run for run in runs):
            values = set()
            for run in runs:
                value = run.get(field)
                if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
                    errors.append(f"{run['run']}: missing or invalid {field}")
                else:
                    values.add(value.lower())
            if len(values) != 1:
                errors.append(f"{field} does not match across selected runs")
            elif all(isinstance(r.get(field), str) and r[field].lower() in values for r in runs):
                report["comparable_inputs"][field] = values.pop()
    for backend in ("legacy", "native"):
        selected = [r for r in report["runs"] if r["backend"] == backend]
        if upgrade or any("revision" in r for r in selected):
            values = [r.get("revision") for r in selected]
            if (
                not all(isinstance(v, str) and v for v in values)
                or (upgrade and not all(re.fullmatch(r"[a-fA-F0-9]{40}", v) for v in values))
                or any(v != values[0] for v in values)
            ):
                errors.append(f"{backend}: revision differs or is missing/invalid across runs")
    if upgrade:
        report["common_runtime"] = {}
        for field in ("torch", "gpu"):
            values = [r.get(field) for r in report["runs"]]
            if not values or not all(isinstance(v, str) and v.strip() and v == values[0] for v in values):
                errors.append(f"{field} differs or is missing across selected runs")
            else:
                report["common_runtime"][field] = values[0]

    def agreement(kind, left, right):
        if left not in observed or right not in observed:
            return
        (a, x), (b, y) = observed[left], observed[right]
        t1, t2 = a["normalized_text"], b["normalized_text"]
        metrics = {
            "text_similarity": 1 - _edits(t1, t2) / max(1, len(t1), len(t2)),
            "box_iou": _iou(a["figure_box_normalized"], b["figure_box_normalized"]),
            "crop_pixel_similarity": _pixel_scores(x, y)[0],
        }
        pair_regions = {}
        for role in ("title", "body", "caption"):
            if role not in a["text_regions"] or role not in b["text_regions"]:
                pair_regions[role] = {"box_iou": 0, "text_similarity": 0}
                continue
            aa, bb = a["text_regions"][role], b["text_regions"][role]
            atext, btext = aa["text"], bb["text"]
            box_scores = [_iou(aa["box"], bb["box"])]
            text_scores = [1 - _edits(atext, btext) / max(1, len(atext), len(btext))]
            if len(aa["groups"]) == len(bb["groups"]):
                # Retain line-level drift when both models segmented these lines separately.
                for ga, gb in zip(aa["groups"], bb["groups"]):
                    aboxes, bboxes = ga["predicted_boxes"], gb["predicted_boxes"]
                    box_scores.append(_iou(_union(aboxes), _union(bboxes)))
                    if len(aboxes) == len(bboxes):
                        box_scores.extend(_iou(p, q) for p, q in zip(aboxes, bboxes))
                    text_scores.append(1 - _edits(ga["text"], gb["text"]) / max(1, len(ga["text"]), len(gb["text"])))
            pair_regions[role] = {"box_iou": min(box_scores), "text_similarity": min(text_scores)}
        metrics["min_region_box_iou"] = min(s["box_iou"] for s in pair_regions.values())
        metrics["min_region_text_similarity"] = min(s["text_similarity"] for s in pair_regions.values())
        if kind in ("backends", "native_versions"):
            metrics["native_accuracy_delta"] = b["text_accuracy"] - a["text_accuracy"]
        lref, rref = (f"{name}/{case}/{i + 1}" for name, case, i in (left, right))
        report["comparisons"].append(dict(kind=kind, left=lref, right=rref, text_regions=pair_regions, **metrics))
        grade(f"{kind} {lref} vs {rref}", metrics, report["minimums"][kind])

    for i in (0, 1):
        for group in ids.values():
            for run in group:
                agreement("formats", (run, "png", i), (run, "pdf", i))
            for run_a, run_b in combinations(group, 2):
                for case in ("png", "pdf"):
                    agreement("repeat", (run_a, case, i), (run_b, case, i))
        for kind, group_a, group_b in cross_groups:
            for run_a, run_b in product(ids[group_a], ids[group_b]):
                for case in ("png", "pdf"):
                    agreement(kind, (run_a, case, i), (run_b, case, i))
    pages = [page for run in report["runs"] for page in run["pages"]]
    if pages:
        report["summary"] = {
            "measured_pages": len(pages),
            "max_character_error_rate": max(p["character_error_rate"] for p in pages),
            "min_figure_box_iou": min(p["box_iou"] for p in pages),
            "min_crop_pixel_similarity": min(p["crop_pixel_similarity"] for p in pages),
            "min_text_region_iou": min(p["min_region_box_iou"] for p in pages),
            "min_text_region_ink_coverage": min(p["min_region_ink_coverage"] for p in pages),
        }
    report["passed"] = not errors
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cmd = commands.add_parser("prepare", help="write two PNGs, their lossless PDF, and ground truth")
    cmd.add_argument("--out", type=Path, required=True)
    cmd.add_argument("--font", type=Path)
    cmd = commands.add_parser("compare", help="score captured raw predictions, Markdown and linked crops")
    cmd.add_argument("--fixture", type=Path, required=True)
    cmd.add_argument("--legacy", type=Path, action="append", default=[], help="legacy run directory; repeat for reruns")
    cmd.add_argument("--native", type=Path, action="append", default=[], help="native run directory; repeat for reruns")
    cmd.add_argument("--reference-native", type=Path, action="append", help="previous native version run; repeat for reruns")
    cmd.add_argument("--native-version", help="exact candidate Transformers version for the three-way upgrade gate")
    cmd.add_argument("--reference-native-version", help="exact reference Transformers version for the three-way upgrade gate")
    cmd.add_argument("--out", type=Path, required=True, help="new JSON report path; existing files are not replaced")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args.out, args.font)
            return 0
        result = compare(
            args.fixture,
            args.legacy,
            args.native,
            reference_native=args.reference_native,
            native_version=args.native_version,
            reference_native_version=args.reference_native_version,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print(f"{'PASS' if result['passed'] else 'FAIL'}: {len(result['errors'])} issue(s); {args.out}")
        return 0 if result["passed"] else 1
    except (ValueError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
