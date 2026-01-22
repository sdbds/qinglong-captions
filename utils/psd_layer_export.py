# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from utils.stream_util import sanitize_filename

try:
    from psd_tools import PSDImage
except Exception as e:  # pragma: no cover
    PSDImage = None  # type: ignore
    _PSD_IMPORT_ERROR = e
else:
    _PSD_IMPORT_ERROR = None


@dataclass(frozen=True)
class PsdLayerInfo:
    name: str
    is_group: bool
    visible: bool
    opacity: float
    bbox: Tuple[int, int, int, int]


@dataclass(frozen=True)
class LayerFeatures:
    avg_saturation: float
    alpha_coverage: float
    color_entropy: float
    bbox_coverage: float
    center_offset: float


def _ensure_psd_tools_available() -> None:
    if PSDImage is None:
        raise RuntimeError(
            "psd-tools is not available. Please install dependencies first (pip install -r requirements.txt). "
            f"Import error: {_PSD_IMPORT_ERROR}"
        )


def _layer_bbox(layer) -> Tuple[int, int, int, int]:
    bbox = getattr(layer, "bbox", None)
    if bbox is None:
        return (0, 0, 0, 0)
    try:
        return (
            int(round(float(bbox.x1))),
            int(round(float(bbox.y1))),
            int(round(float(bbox.x2))),
            int(round(float(bbox.y2))),
        )
    except Exception:
        try:
            return tuple(map(int, bbox))  # type: ignore[arg-type]
        except Exception:
            return (0, 0, 0, 0)


def read_psd_layers(psd_path: Path, include_invisible: bool = False, verbose: bool = False) -> List[PsdLayerInfo]:
    """Read PSD and return a flat list of layer info (including groups).

    Notes:
    - This does not export anything.
    - If include_invisible is False, invisible layers are filtered out.
    """

    _ensure_psd_tools_available()

    psd_path = Path(psd_path)
    psd = PSDImage.open(psd_path)

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    infos: List[PsdLayerInfo] = []

    def walk(layer_iter: Iterable) -> None:
        for layer in layer_iter:
            visible = bool(getattr(layer, "is_visible", lambda: True)())
            if include_invisible or visible:
                infos.append(
                    PsdLayerInfo(
                        name=str(getattr(layer, "name", "")),
                        is_group=bool(getattr(layer, "is_group", lambda: False)()),
                        visible=visible,
                        opacity=float(getattr(layer, "opacity", 1.0)),
                        bbox=_layer_bbox(layer),
                    )
                )
            if bool(getattr(layer, "is_group", lambda: False)()):
                walk(layer)

    _log(f"PSD opened: {psd_path}")
    walk(psd)
    _log(f"Items collected: {len(infos)}")
    return infos


def _iter_leaf_layers(root, include_invisible: bool) -> List:
    leaves: List = []

    def walk(layer_iter: Iterable) -> None:
        for layer in layer_iter:
            visible = bool(getattr(layer, "is_visible", lambda: True)())
            if not include_invisible and not visible:
                continue

            is_group = bool(getattr(layer, "is_group", lambda: False)())
            if is_group:
                walk(layer)
            else:
                leaves.append(layer)

    walk(root)
    return leaves


def _iter_leaf_layers_with_top_group_name(root, include_invisible: bool) -> List[Tuple[str, object]]:
    leaves: List[Tuple[str, object]] = []

    def walk(layer_iter: Iterable, top_group_name: str) -> None:
        for layer in layer_iter:
            visible = bool(getattr(layer, "is_visible", lambda: True)())
            if not include_invisible and not visible:
                continue

            is_group = bool(getattr(layer, "is_group", lambda: False)())
            if is_group:
                name = str(getattr(layer, "name", ""))
                next_top = top_group_name or name
                walk(layer, next_top)
            else:
                leaves.append((top_group_name, layer))

    walk(root, "")
    return leaves


def _iter_leaf_layers_with_lineart_context(
    root,
    *,
    include_invisible: bool,
    lineart_keywords: Optional[Sequence[str]],
    inherited_lineart: bool = False,
) -> List[Tuple[object, bool]]:
    leaves: List[Tuple[object, bool]] = []

    def walk(layer_iter: Iterable, inherited: bool) -> None:
        for layer in layer_iter:
            visible = bool(getattr(layer, "is_visible", lambda: True)())
            if not include_invisible and not visible:
                continue

            is_group = bool(getattr(layer, "is_group", lambda: False)())
            if is_group:
                name = str(getattr(layer, "name", ""))
                next_inherited = inherited or _is_lineart_name(name, extra_keywords=lineart_keywords)
                walk(layer, next_inherited)
            else:
                leaves.append((layer, inherited))

    walk(root, inherited_lineart)
    return leaves


def _iter_top_level_groups(root, include_invisible: bool) -> List:
    groups: List = []
    for layer in root:
        visible = bool(getattr(layer, "is_visible", lambda: True)())
        if not include_invisible and not visible:
            continue
        if bool(getattr(layer, "is_group", lambda: False)()):
            groups.append(layer)
    return groups


def _analyze_layer_pixels(layer, canvas_size: Tuple[int, int], viewport=None) -> LayerFeatures:
    try:
        img = getattr(layer, "composite")(viewport=viewport) if viewport else getattr(layer, "composite")()
        if img is None:
            return LayerFeatures(0.0, 0.0, 0.0, 0.0, 1.0)

        # Downsample for performance (analyze max 100x100 pixels)
        target_size = (min(img.width, 100), min(img.height, 100))
        if img.size != target_size:
            img = img.resize(target_size, Image.Resampling.NEAREST)

        arr = np.array(img)
        if arr.size == 0:
            return LayerFeatures(0.0, 0.0, 0.0, 0.0, 1.0)

        # Extract RGB and alpha channels
        if arr.ndim == 2:
            # Grayscale
            rgb = np.stack([arr, arr, arr], axis=2)
            alpha = np.ones_like(arr) * 255
        elif arr.shape[2] == 3:
            rgb = arr
            alpha = np.ones(arr.shape[:2], dtype=np.uint8) * 255
        elif arr.shape[2] == 4:
            rgb = arr[..., :3]
            alpha = arr[..., 3]
        else:
            return LayerFeatures(0.0, 0.0, 0.0, 0.0, 1.0)

        # Filter non-transparent pixels
        non_transparent = alpha > 10
        if non_transparent.sum() == 0:
            return LayerFeatures(0.0, 0.0, 0.0, 0.0, 1.0)

        # Calculate saturation: (max(R,G,B) - min(R,G,B)) / max(R,G,B)
        rgb_visible = rgb[non_transparent].astype(np.float32)
        rgb_max = rgb_visible.max(axis=1)
        rgb_min = rgb_visible.min(axis=1)
        saturation_values = np.where(rgb_max > 0, (rgb_max - rgb_min) / rgb_max, 0.0)
        avg_saturation = float(np.mean(saturation_values))

        # Alpha coverage
        alpha_coverage = float(non_transparent.sum() / non_transparent.size)

        # Color entropy (binned histogram)
        try:
            # Quantize RGB to 16 bins per channel
            quantized = (rgb_visible / 16).astype(np.int32)
            color_codes = quantized[:, 0] * 256 + quantized[:, 1] * 16 + quantized[:, 2]
            hist, _ = np.histogram(color_codes, bins=min(256, len(color_codes)))
            hist = hist[hist > 0]
            if len(hist) > 0:
                prob = hist / hist.sum()
                color_entropy = float(-np.sum(prob * np.log2(prob)))
            else:
                color_entropy = 0.0
        except Exception:
            color_entropy = 0.0

        # Bbox coverage
        bbox = getattr(layer, "bbox", None)
        if bbox is not None:
            try:
                bbox_area = max(0, int(getattr(bbox, "x2", 0)) - int(getattr(bbox, "x1", 0))) * max(
                    0, int(getattr(bbox, "y2", 0)) - int(getattr(bbox, "y1", 0))
                )
                canvas_area = canvas_size[0] * canvas_size[1]
                bbox_coverage = float(bbox_area / canvas_area) if canvas_area > 0 else 0.0
            except Exception:
                bbox_coverage = 0.0
        else:
            bbox_coverage = 0.0

        # Center offset
        if bbox is not None:
            try:
                cx = (int(getattr(bbox, "x1", 0)) + int(getattr(bbox, "x2", 0))) / 2.0
                cy = (int(getattr(bbox, "y1", 0)) + int(getattr(bbox, "y2", 0))) / 2.0
                center_dist = math.hypot(cx - canvas_size[0] / 2.0, cy - canvas_size[1] / 2.0)
                max_dist = math.hypot(canvas_size[0] / 2.0, canvas_size[1] / 2.0)
                center_offset = float(center_dist / max_dist) if max_dist > 0 else 0.0
            except Exception:
                center_offset = 1.0
        else:
            center_offset = 1.0

        return LayerFeatures(avg_saturation, alpha_coverage, color_entropy, bbox_coverage, center_offset)
    except Exception:
        return LayerFeatures(0.0, 0.0, 0.0, 0.0, 1.0)


def _is_lineart_name(name: str, extra_keywords: Optional[Sequence[str]] = None) -> bool:
    n = (name or "").lower()
    keywords = [
        "lineart",
        "line art",
        "lines",
        "sketch",
        "draft",
        "ink",
        "outline",
        # Japanese
        "線画",
        "ライン",
        "下書き",
        "ラフ",
        "スケッチ",
        # Korean
        "선화",
        "라인아트",
        "스케치",
        "밑그림",
        "초안",
        "线稿",
        "线搞",
        "草稿",
        "描边",
        "线条",
        "原画",
        "原稿",
    ]
    if extra_keywords:
        keywords.extend([k.lower() for k in extra_keywords])
    return any(k in n for k in keywords)


def _safe_stem_name(psd_path: Path) -> str:
    # Keep consistency with existing utils naming policy.
    # sanitize_filename lowercases and limits length.
    return sanitize_filename(psd_path.stem)


def _safe_filename_component(name: str, default: str) -> str:
    s = (name or "").strip()
    if not s:
        return default
    forbidden = '<>:"/\\|?*'
    s = "".join("_" if (ch in forbidden or ord(ch) < 32) else ch for ch in s)
    s = s.strip(" .")
    if not s:
        return default
    if len(s) > 80:
        s = s[:80]
    return s


def _layer_output_name(layer_name: str, idx: int) -> str:
    base = _safe_filename_component(layer_name, default=f"layer-{idx:03d}")
    return f"{idx:03d}_{base}.png"


def _group_output_name(group_name: str, idx: int) -> str:
    base = _safe_filename_component(group_name, default=f"group-{idx:03d}")
    return f"{idx:03d}_group_{base}.png"


def _merged_output_name(kind: str, idx: int) -> str:
    base = sanitize_filename(kind) if kind else f"merged-{idx:03d}"
    return f"{idx:03d}_merged_{base}.png"


def _classify_content_name(name: str) -> str:
    n = (name or "").lower()

    bg_keywords = [
        "bg",
        "background",
        "back",
        "背景",
        "バック",
        "배경",
        "后",
    ]
    subject_keywords = [
        "subject",
        "main",
        "character",
        "body",
        "人物",
        "主体",
        "角色",
        "キャラ",
        "キャラク",
        "本体",
        "주체",
        "인물",
        "캐릭",
    ]
    fg_keywords = [
        "fg",
        "foreground",
        "front",
        "前景",
        "前",
        "手前",
        "전경",
    ]
    fx_keywords = [
        "fx",
        "effect",
        "overlay",
        "glow",
        "light",
        "エフェクト",
        "効果",
        "효과",
        "效果",
        "加工",
        "调整",
        "滤镜",
        "模糊",
    ]

    coloring_keywords = [
        "color",
        "colour",
        "paint",
        "fill",
        "shade",
        "shading",
        "render",
        "base",
        "flat",
        "skin",
        "上色",
        "上彩",
        "颜色",
        "皮肤",
        "塗り",
        "着色",
        "色",
        "컬러",
        "채색",
        "색",
    ]

    other_keywords = [
        "other",
        "misc",
        "extra",
        "others",
        "其它",
        "其他",
        "雑項",
        "その他",
        "기타",
        "蒙版",
        "底稿",
    ]

    if any(k in n for k in bg_keywords):
        return "background"
    if any(k in n for k in subject_keywords):
        return "subject"
    if any(k in n for k in fg_keywords):
        return "foreground"
    if any(k in n for k in coloring_keywords):
        return "coloring"
    if any(k in n for k in fx_keywords):
        return "effects"
    if any(k in n for k in other_keywords):
        return "other"
    return "coloring"


def _classify_layer(
    name: str,
    features: LayerFeatures,
    z_index: int,
    total_layers: int,
    lineart_keywords: Optional[Sequence[str]] = None,
) -> str:
    # Rule 1: Low saturation + sparse coverage → lineart
    # Lineart typically has low color saturation and sparse pixel distribution
    if features.avg_saturation < 0.15 and features.alpha_coverage < 0.3:
        return "lineart"

    # Rule 2: Very low saturation with any coverage → likely lineart
    if features.avg_saturation < 0.1:
        return "lineart"

    # Rule 3: Name-based lineart detection (keep as fallback)
    if _is_lineart_name(name, extra_keywords=lineart_keywords):
        return "lineart"

    # Rule 4: Full canvas coverage + low z-index → background
    # Background layers typically cover the entire canvas and are at the bottom
    if features.bbox_coverage > 0.95 and z_index < total_layers * 0.3:
        return "background"

    # Rule 5: High color complexity + centered → subject
    # Main characters have rich colors and are usually centered
    if features.color_entropy > 3.0 and features.center_offset < 0.3:
        return "subject"

    # Rule 6: Medium-high saturation + centered → likely subject coloring
    if features.avg_saturation > 0.3 and features.center_offset < 0.4:
        cat = _classify_content_name(name)
        if cat in ("subject", "coloring"):
            return cat

    # Rule 7: High z-index → foreground
    # Top layers are usually foreground elements
    if z_index > total_layers * 0.7:
        return "foreground"

    # Rule 8: Low entropy + high coverage → likely background or effects
    if features.color_entropy < 2.0 and features.bbox_coverage > 0.7:
        cat = _classify_content_name(name)
        if cat in ("background", "effects"):
            return cat
        return "background"

    # Fallback to name-based classification
    return _classify_content_name(name)


def _paste_to_canvas(img: Image.Image, bbox, canvas_size: Tuple[int, int]) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    if img.size == canvas_size:
        return img
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))

    if bbox is None:
        x1 = 0
        y1 = 0
    else:
        x1 = int(round(float(getattr(bbox, "x1", 0))))
        y1 = int(round(float(getattr(bbox, "y1", 0))))
    try:
        canvas.paste(img, (x1, y1), img)
    except Exception:
        canvas.paste(img, (x1, y1))
    return canvas


def _alpha_merge(images: Sequence[Image.Image], canvas_size: Tuple[int, int]) -> Optional[Image.Image]:
    if not images:
        return None
    base = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    # psd-tools iterates layers from background to foreground (bottom to top).
    # Composite in the given order so upper layers correctly cover lower ones.
    for img in images:
        if img is None:
            continue
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        if img.size != canvas_size:
            tmp = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
            try:
                tmp.paste(img, (0, 0), img)
            except Exception:
                tmp.paste(img, (0, 0))
            img = tmp
        base = Image.alpha_composite(base, img)
    return base


def export_psd_layers(
    psd_path: Path,
    results_root: Optional[Path] = None,
    max_direct_layers: int = 7,
    include_invisible: bool = False,
    lineart_keywords: Optional[Sequence[str]] = None,
    force_seven_layers: bool = False,
    merge_lineart: bool = True,
    verbose: bool = False,
) -> Optional[Path]:
    """Export PSD layers to PNG based on simple rules.

    Rules:
    - If there is only 1 leaf layer: skip exporting.
    - If leaf layers <= max_direct_layers: export each leaf as PNG.
    - If leaf layers > max_direct_layers:
      - Export each top-level group as a merged PNG, except groups with lineart names.
      - For lineart groups, export its leaf layers separately.
    """

    _ensure_psd_tools_available()

    psd_path = Path(psd_path)
    if not psd_path.exists():
        raise FileNotFoundError(f"PSD not found: {psd_path}")

    psd = PSDImage.open(psd_path)

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    leaf_layers = _iter_leaf_layers(psd, include_invisible=include_invisible)
    leaf_count = len(leaf_layers)

    _log(f"PSD opened: {psd_path}")
    _log(f"Leaf layer count: {leaf_count}")

    if leaf_count <= 1:
        return None

    if results_root is None:
        results_root = psd_path.parent / "results"

    out_dir = Path(results_root) / _safe_stem_name(psd_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_image(img: Image.Image, out_path: Path) -> None:
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA")
        img.save(out_path, format="PNG")

    if not force_seven_layers and leaf_count <= max_direct_layers:
        canvas_w = int(getattr(psd, "width", 0))
        canvas_h = int(getattr(psd, "height", 0))
        if canvas_w <= 0 or canvas_h <= 0:
            try:
                canvas_w, canvas_h = map(int, getattr(psd, "size"))
            except Exception:
                canvas_w, canvas_h = 0, 0
        if canvas_w <= 0 or canvas_h <= 0:
            raise RuntimeError(f"Invalid PSD canvas size: {psd_path}")
        canvas_size = (canvas_w, canvas_h)
        canvas_viewport = (0, 0, canvas_w, canvas_h)

        direct_leaves = _iter_leaf_layers_with_top_group_name(psd, include_invisible=include_invisible)
        lineart_layers: List[Tuple[str, object]] = []
        non_lineart_leaves: List[Tuple[str, object]] = []
        for top_group_name, layer in direct_leaves:
            lname = str(getattr(layer, "name", ""))
            if _is_lineart_name(lname, extra_keywords=lineart_keywords) or _is_lineart_name(
                top_group_name, extra_keywords=lineart_keywords
            ):
                lineart_layers.append((top_group_name, layer))
            else:
                non_lineart_leaves.append((top_group_name, layer))

        # Export order: foreground -> background.
        idx = 1
        for top_group_name, layer in reversed(non_lineart_leaves):
            lname = str(getattr(layer, "name", ""))
            export_name = top_group_name or lname
            _log(f"Compositing leaf[{idx}/{leaf_count}]: {export_name}")
            t0 = time.monotonic()
            img = getattr(layer, "composite")(viewport=canvas_viewport)
            _log(f"Composite done in {time.monotonic() - t0:.3f}s: {export_name}")
            if img is None:
                continue
            out_path = out_dir / _layer_output_name(export_name, idx)
            save_image(img, out_path)
            idx += 1

        if lineart_layers:
            if merge_lineart:
                lineart_imgs: List[Image.Image] = []
                for _, layer in lineart_layers:
                    img = getattr(layer, "composite")(viewport=canvas_viewport)
                    if img is None:
                        continue
                    lineart_imgs.append(img)
                merged_lineart = _alpha_merge(lineart_imgs, canvas_size)
                if merged_lineart is not None:
                    out_path = out_dir / _layer_output_name("lineart", idx)
                    save_image(merged_lineart, out_path)
            else:
                for top_group_name, layer in reversed(lineart_layers):
                    lname = str(getattr(layer, "name", ""))
                    export_name = top_group_name or lname
                    img = getattr(layer, "composite")(viewport=canvas_viewport)
                    if img is None:
                        continue
                    out_path = out_dir / _layer_output_name(export_name, idx)
                    save_image(img, out_path)
                    idx += 1

        return out_dir

    if force_seven_layers:

        def bbox_area(bbox) -> int:
            if bbox is None:
                return 0
            try:
                w = max(0, int(getattr(bbox, "x2", 0)) - int(getattr(bbox, "x1", 0)))
                h = max(0, int(getattr(bbox, "y2", 0)) - int(getattr(bbox, "y1", 0)))
                return w * h
            except Exception:
                return 0

        canvas_w = int(getattr(psd, "width", 0))
        canvas_h = int(getattr(psd, "height", 0))
        if canvas_w <= 0 or canvas_h <= 0:
            try:
                canvas_w, canvas_h = map(int, getattr(psd, "size"))
            except Exception:
                canvas_w, canvas_h = 0, 0
        if canvas_w <= 0 or canvas_h <= 0:
            raise RuntimeError(f"Invalid PSD canvas size: {psd_path}")
        canvas_size = (canvas_w, canvas_h)
        canvas_viewport = (0, 0, canvas_w, canvas_h)

        _log(f"Canvas size: {canvas_size}")

        _log("Top-level items:")
        for top in psd:
            visible = bool(getattr(top, "is_visible", lambda: True)())
            if not include_invisible and not visible:
                continue
            is_group = bool(getattr(top, "is_group", lambda: False)())
            tname = str(getattr(top, "name", ""))
            kind = "group" if is_group else "layer"
            _log(f"  - {kind}: {tname}")

        if merge_lineart:
            stage_items: List[Tuple[str, Image.Image]] = []
            stage_lineart_images: List[Image.Image] = []

            def add_lineart_leaf(leaf, source_name: str) -> None:
                lname = str(getattr(leaf, "name", ""))
                _log(f"Compositing leaf: {source_name} / {lname}")
                t0 = time.monotonic()
                img = leaf.composite(viewport=canvas_viewport)
                _log(f"Composite done in {time.monotonic() - t0:.3f}s: {lname}")
                if img is None:
                    return
                stage_lineart_images.append(img)

            def add_non_lineart_image(name: str, img: Optional[Image.Image]) -> None:
                if img is None:
                    return
                stage_items.append((name, img))

            for top in psd:
                visible = bool(getattr(top, "is_visible", lambda: True)())
                if not include_invisible and not visible:
                    continue

                is_group = bool(getattr(top, "is_group", lambda: False)())
                tname = str(getattr(top, "name", ""))

                if is_group:
                    group_is_lineart = _is_lineart_name(tname, extra_keywords=lineart_keywords)
                    leaves = _iter_leaf_layers_with_lineart_context(
                        top,
                        include_invisible=include_invisible,
                        lineart_keywords=lineart_keywords,
                        inherited_lineart=group_is_lineart,
                    )
                    has_lineart_leaf = any(is_line for _, is_line in leaves)

                    if not has_lineart_leaf and not group_is_lineart:
                        img = top.composite(viewport=canvas_viewport)
                        if img is None:
                            continue
                        stage_items.append((tname or "group", img))
                        continue

                    non_lineart_imgs: List[Image.Image] = []
                    for leaf, inherited_is_line in leaves:
                        lname = str(getattr(leaf, "name", ""))
                        is_line = inherited_is_line or _is_lineart_name(lname, extra_keywords=lineart_keywords) or group_is_lineart
                        if is_line:
                            add_lineart_leaf(leaf, source_name=tname or "group")
                            continue

                        _log(f"Compositing leaf: {tname} / {lname}")
                        t0 = time.monotonic()
                        img = leaf.composite(viewport=canvas_viewport)
                        _log(f"Composite done in {time.monotonic() - t0:.3f}s: {lname}")
                        if img is None:
                            continue
                        non_lineart_imgs.append(img)

                    merged = _alpha_merge(non_lineart_imgs, canvas_size)
                    if merged is not None:
                        stage_items.append((tname or "group", merged))
                else:
                    if _is_lineart_name(tname, extra_keywords=lineart_keywords):
                        add_lineart_leaf(top, source_name="<top>")
                        continue
                    _log(f"Compositing leaf: <top> / {tname}")
                    t0 = time.monotonic()
                    img = top.composite(viewport=canvas_viewport)
                    _log(f"Composite done in {time.monotonic() - t0:.3f}s: {tname}")
                    if img is None:
                        continue
                    stage_items.append((tname or "layer", img))

            if stage_lineart_images:
                merged_lineart = _alpha_merge(stage_lineart_images, canvas_size)
                if merged_lineart is not None:
                    stage_items.append(("lineart", merged_lineart))

            if 0 < len(stage_items) <= max_direct_layers:
                # Export order: foreground -> background.
                if stage_items:
                    if stage_lineart_images and stage_items[-1][0] == "lineart":
                        lineart_item = stage_items.pop()
                        stage_items.reverse()
                        stage_items.append(lineart_item)
                    else:
                        stage_items.reverse()
                for i, (name, img) in enumerate(stage_items, start=1):
                    base = _safe_filename_component(name, default=f"layer-{i:03d}")
                    out_path = out_dir / f"{i:03d}_{base}.png"
                    save_image(img, out_path)
                return out_dir

        # Determine subject group: among top-level visible groups, choose by leaf count desc then bbox area desc.
        subject_group = None
        subject_area = -1
        subject_leaf_count = -1
        for top in psd:
            visible = bool(getattr(top, "is_visible", lambda: True)())
            if not include_invisible and not visible:
                continue

            if not bool(getattr(top, "is_group", lambda: False)()):
                continue

            tname = str(getattr(top, "name", ""))
            if _is_lineart_name(tname, extra_keywords=lineart_keywords):
                continue

            if _classify_content_name(tname) == "background":
                continue

            area = bbox_area(getattr(top, "bbox", None))
            leaf_cnt = len(_iter_leaf_layers(top, include_invisible=include_invisible))
            _log(f"Subject candidate: {tname} | leaf_cnt={leaf_cnt} area={area}")
            if leaf_cnt > subject_leaf_count or (leaf_cnt == subject_leaf_count and area > subject_area):
                subject_group = top
                subject_area = area
                subject_leaf_count = leaf_cnt

        if subject_group is not None:
            subject_name = str(getattr(subject_group, "name", ""))
            _log(f"Selected subject group: {subject_name} | leaf_cnt={subject_leaf_count} area={subject_area}")
        else:
            _log("Selected subject group: <none>")

        categories: Dict[str, List[Image.Image]] = {
            "background": [],
            "subject": [],
            "foreground": [],
            "lineart": [],
            "coloring": [],
            "effects": [],
            "other": [],
        }

        leaf_plans: List[Tuple[object, str, str, Optional[str], bool, Tuple[int, int, int, int], int, bool]] = []
        z = 0

        for top in psd:
            visible = bool(getattr(top, "is_visible", lambda: True)())
            if not include_invisible and not visible:
                continue

            is_group = bool(getattr(top, "is_group", lambda: False)())
            tname = str(getattr(top, "name", ""))
            if is_group:
                default_cat = "subject" if (subject_group is not None and top is subject_group) else None
                group_is_lineart = _is_lineart_name(tname, extra_keywords=lineart_keywords)
                for leaf, inherited_is_line in _iter_leaf_layers_with_lineart_context(
                    top,
                    include_invisible=include_invisible,
                    lineart_keywords=lineart_keywords,
                    inherited_lineart=group_is_lineart,
                ):
                    lname = str(getattr(leaf, "name", ""))
                    is_line = group_is_lineart or inherited_is_line or _is_lineart_name(lname, extra_keywords=lineart_keywords)
                    x1, y1, x2, y2 = _layer_bbox(leaf)
                    covers_canvas = x1 <= 0 and y1 <= 0 and x2 >= canvas_w and y2 >= canvas_h
                    leaf_plans.append((leaf, lname, tname, default_cat, is_line, (x1, y1, x2, y2), z, covers_canvas))
                    z += 1
            else:
                lname = tname
                is_line = _is_lineart_name(lname, extra_keywords=lineart_keywords)
                x1, y1, x2, y2 = _layer_bbox(top)
                covers_canvas = x1 <= 0 and y1 <= 0 and x2 >= canvas_w and y2 >= canvas_h
                leaf_plans.append((top, lname, "<top>", None, is_line, (x1, y1, x2, y2), z, covers_canvas))
                z += 1

        total_leaf_count = len(leaf_plans)

        for leaf, lname, source_name, default_cat, is_line, _, pz, covers_canvas in leaf_plans:
            if source_name:
                _log(f"Compositing leaf: {source_name} / {lname}")
            else:
                _log(f"Compositing leaf: {lname}")
            t0 = time.monotonic()
            img = getattr(leaf, "composite")(viewport=canvas_viewport)
            _log(f"Composite done in {time.monotonic() - t0:.3f}s: {lname}")
            if img is None:
                continue

            # Use pixel-based classification for better accuracy
            if not is_line:
                # Analyze pixel features for non-lineart layers
                features = _analyze_layer_pixels(leaf, canvas_size, viewport=canvas_viewport)
                _log(
                    f"  Features: sat={features.avg_saturation:.3f}, cov={features.alpha_coverage:.3f}, "
                    f"ent={features.color_entropy:.2f}, bbox={features.bbox_coverage:.2f}, "
                    f"center={features.center_offset:.2f}"
                )

                # Use improved classification with pixel features
                cat = _classify_layer(lname, features, pz, total_leaf_count, lineart_keywords)
                _log(f"  Classified as: {cat}")

                # Override with default_cat if explicitly set (from subject group detection)
                if default_cat is not None:
                    cat = default_cat
                    _log(f"  Override to: {cat} (from subject group)")
            else:
                # Lineart layers go directly to lineart category
                cat = "lineart"

            if cat not in categories:
                cat = "other"

            categories[cat].append(img)

        def merge_or_blank(imgs: List[Image.Image]) -> Image.Image:
            merged = _alpha_merge(imgs, canvas_size)
            if merged is None:
                return Image.new("RGBA", canvas_size, (0, 0, 0, 0))
            return merged

        outputs = [
            (1, "background"),
            (2, "subject"),
            (3, "foreground"),
            (4, "lineart"),
            (5, "coloring"),
            (6, "effects"),
            (7, "other"),
        ]

        for idx, key in outputs:
            if key == "lineart" and merge_lineart:
                img = merge_or_blank(categories[key])
                out_path = out_dir / f"{idx:03d}_{key}.png"
                save_image(img, out_path)
                continue

            if key == "lineart" and not merge_lineart:
                # Export each lineart layer separately (may exceed 7 files).
                if not categories[key]:
                    out_path = out_dir / f"{idx:03d}_{key}.png"
                    save_image(Image.new("RGBA", canvas_size, (0, 0, 0, 0)), out_path)
                    continue
                for j, li in enumerate(categories[key], start=1):
                    out_path = out_dir / f"{idx:03d}_{key}_{j:03d}.png"
                    save_image(li, out_path)
                continue

            img = merge_or_blank(categories[key])
            out_path = out_dir / f"{idx:03d}_{key}.png"
            save_image(img, out_path)

        return out_dir

    # leaf_count > max_direct_layers: merge by groups
    # Export order: foreground -> background.
    groups = list(reversed(_iter_top_level_groups(psd, include_invisible=include_invisible)))

    canvas_w = int(getattr(psd, "width", 0))
    canvas_h = int(getattr(psd, "height", 0))
    if canvas_w <= 0 or canvas_h <= 0:
        try:
            canvas_w, canvas_h = map(int, getattr(psd, "size"))
        except Exception:
            canvas_w, canvas_h = 0, 0
    if canvas_w <= 0 or canvas_h <= 0:
        raise RuntimeError(f"Invalid PSD canvas size: {psd_path}")
    canvas_size = (canvas_w, canvas_h)
    canvas_viewport = (0, 0, canvas_w, canvas_h)

    non_lineart_items = []
    lineart_layer_items: List[Tuple[str, object]] = []
    ordered_items: List[Tuple[str, object]] = []

    for group in groups:
        gname = str(getattr(group, "name", ""))
        group_is_lineart = _is_lineart_name(gname, extra_keywords=lineart_keywords)
        leaves_with_ctx = _iter_leaf_layers_with_lineart_context(
            group,
            include_invisible=include_invisible,
            lineart_keywords=lineart_keywords,
            inherited_lineart=group_is_lineart,
        )

        has_lineart_leaf = any(
            inherited_is_line or _is_lineart_name(str(getattr(leaf, "name", "")), extra_keywords=lineart_keywords)
            for leaf, inherited_is_line in leaves_with_ctx
        )

        if group_is_lineart:
            # Lineart group: export its leaves separately.
            # Export order: foreground -> background.
            for layer, _ in reversed(leaves_with_ctx):
                lineart_layer_items.append((gname, layer))
                ordered_items.append((gname, layer))
            continue

        if not has_lineart_leaf:
            # Preserve group-level effects by using group.composite when there is no lineart to split.
            img = group.composite(viewport=canvas_viewport)
            if img is None:
                continue
            non_lineart_items.append((gname, img))
            ordered_items.append((gname, img))
            continue

        # Mixed group: split lineart leaves out, merge remaining leaves for the group.
        non_lineart_imgs: List[Image.Image] = []
        lineart_leaves: List[object] = []
        for leaf, inherited_is_line in leaves_with_ctx:
            lname = str(getattr(leaf, "name", ""))
            is_line = inherited_is_line or _is_lineart_name(lname, extra_keywords=lineart_keywords)
            if is_line:
                lineart_leaves.append(leaf)
                continue

            img = getattr(leaf, "composite")(viewport=canvas_viewport)
            if img is None:
                continue
            non_lineart_imgs.append(img)

        # Export order: foreground -> background.
        for leaf in reversed(lineart_leaves):
            lineart_layer_items.append((gname, leaf))
            ordered_items.append((gname, leaf))

        merged = _alpha_merge(non_lineart_imgs, canvas_size)
        if merged is None:
            continue
        non_lineart_items.append((gname, merged))
        ordered_items.append((gname, merged))

    # If there are no groups (flat PSD), fallback to exporting first max_direct_layers leaf layers.
    if not groups and not non_lineart_items and not lineart_layer_items:
        # Export order: foreground -> background.
        for i, layer in enumerate(list(reversed(leaf_layers))[:max_direct_layers], start=1):
            img = layer.composite(viewport=canvas_viewport)
            if img is None:
                continue
            out_path = out_dir / _layer_output_name(getattr(layer, "name", ""), i)
            save_image(img, out_path)
        return out_dir

    direct_count = len(non_lineart_items) + len(lineart_layer_items)

    # If already small enough, export directly.
    if direct_count <= max_direct_layers:
        idx = 1
        for gname, item in ordered_items:
            if isinstance(item, Image.Image):
                out_path = out_dir / _group_output_name(gname, idx)
                save_image(item, out_path)
                idx += 1
                continue

            img = getattr(item, "composite")(viewport=canvas_viewport)
            if img is None:
                continue
            lname = str(getattr(item, "name", ""))
            export_name = gname or lname
            out_path = out_dir / _layer_output_name(export_name, idx)
            save_image(img, out_path)
            idx += 1
        return out_dir

    # Still too many: merge non-lineart items by content categories (lineart remains separated).
    categorized: Dict[str, List[Image.Image]] = {}
    for gname, img in non_lineart_items:
        kind = _classify_content_name(gname)
        categorized.setdefault(kind, []).append(img)

    kind_order = ["background", "subject", "foreground", "coloring", "effects", "other"]
    extra_kinds = [k for k in categorized.keys() if k not in kind_order]
    merged_images: List[Tuple[str, Image.Image]] = []
    for kind in kind_order + sorted(extra_kinds):
        merged = _alpha_merge(categorized.get(kind, []), canvas_size)
        if merged is None:
            continue
        merged_images.append((kind, merged))

    merged_idx = 1
    for kind, merged in merged_images:
        out_path = out_dir / _merged_output_name(kind, merged_idx)
        save_image(merged, out_path)
        merged_idx += 1

    if merge_lineart:
        lineart_imgs: List[Image.Image] = []
        for _, layer in lineart_layer_items:
            img = getattr(layer, "composite")(viewport=canvas_viewport)
            if img is None:
                continue
            lineart_imgs.append(img)
        merged_lineart = _alpha_merge(lineart_imgs, canvas_size)
        if merged_lineart is not None:
            out_path = out_dir / _merged_output_name("lineart", merged_idx)
            save_image(merged_lineart, out_path)
    else:
        leaf_idx = 1
        for gname, layer in lineart_layer_items:
            img = getattr(layer, "composite")(viewport=canvas_viewport)
            if img is None:
                continue
            lname = str(getattr(layer, "name", ""))
            export_name = gname or lname
            out_path = out_dir / _layer_output_name(export_name, leaf_idx)
            save_image(img, out_path)
            leaf_idx += 1

    return out_dir


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Export PSD layers to PNG.")
    parser.add_argument("psd", type=str, help="Path to .psd")
    parser.add_argument("--results-root", type=str, default="", help="Root output directory (default: <psd_dir>/results)")
    parser.add_argument("--max-direct-layers", type=int, default=7, help="Direct export threshold")
    parser.add_argument("--include-invisible", action="store_true", help="Include invisible layers")
    args = parser.parse_args()

    psd_path = Path(args.psd)
    results_root = Path(args.results_root) if args.results_root else None

    out_dir = export_psd_layers(
        psd_path=psd_path,
        results_root=results_root,
        max_direct_layers=int(args.max_direct_layers),
        include_invisible=bool(args.include_invisible),
    )

    if out_dir is None:
        print("Skipped: only one leaf layer.")
        return 0

    print(f"Exported to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
