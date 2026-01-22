# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from module import lanceImport, lanceexport
from utils.psd_layer_export import export_psd_layers


def process_psd_folder(
    psd_dir: Path,
    results_root: Optional[Path] = None,
    max_direct_layers: int = 7,
    include_invisible: bool = False,
    force_seven_layers: bool = True,
    merge_lineart: bool = True,
    verbose: bool = False,
    resize_max_size: int = 0,
    resize_results_root: Optional[Path] = None,
    export_lance_to_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Process a folder of PSD files into a PNG dataset and optionally Lance.

    Steps:
    1) Export layers for each PSD into results/<psd_name>/
    2) Convert the resulting image dataset folder into Lance using module.lanceImport
    3) Optionally extract from Lance back to a folder using module.lanceexport

    Returns:
        Path to generated .lance dataset, or None if no PSDs were processed.
    """

    psd_dir = Path(psd_dir)
    if not psd_dir.exists():
        raise FileNotFoundError(f"psd_dir not found: {psd_dir}")

    if results_root is None:
        results_root = psd_dir / "results"

    base_results_root = Path(results_root)
    if resize_max_size > 0:
        if resize_results_root is None:
            results_root = base_results_root.parent / f"{base_results_root.name}_{int(resize_max_size)}"
        else:
            results_root = Path(resize_results_root)
    else:
        results_root = base_results_root

    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    def calculate_dimensions(width: int, height: int, max_size: int) -> Tuple[int, int]:
        aspect_ratio = width / height
        if width > height:
            new_width = min(max_size, (width // 16) * 16)
            new_height = ((int(new_width / aspect_ratio)) // 16) * 16
        else:
            new_height = min(max_size, (height // 16) * 16)
            new_width = ((int(new_height * aspect_ratio)) // 16) * 16

        if new_width > max_size:
            new_width = max_size
            new_height = ((int(new_width / aspect_ratio)) // 16) * 16
        if new_height > max_size:
            new_height = max_size
            new_width = ((int(new_height * aspect_ratio)) // 16) * 16

        new_width = max(16, (new_width // 16) * 16)
        new_height = max(16, (new_height // 16) * 16)
        return new_width, new_height

    def resize_pngs_inplace(dir_path: Path, max_size: int) -> None:
        for p in sorted(dir_path.glob("*.png")):
            try:
                with Image.open(p) as im:
                    im.load()
                    w, h = im.size
                    new_w, new_h = calculate_dimensions(int(w), int(h), int(max_size))
                    if (new_w, new_h) != (w, h):
                        im = im.resize((new_w, new_h), Image.LANCZOS)
                    im.save(p, format="PNG")
                if verbose:
                    print(f"Resized: {p.name}", flush=True)
            except Exception as e:
                print(f"Resize failed: {p} | {e}", flush=True)

    psd_files = sorted([p for p in psd_dir.rglob("*.psd") if p.is_file()])
    if not psd_files:
        return None

    print(f"Found {len(psd_files)} PSD file(s) under: {psd_dir}", flush=True)

    for psd_path in psd_files:
        print(f"Processing PSD: {psd_path}", flush=True)
        out_dir = export_psd_layers(
            psd_path=psd_path,
            results_root=results_root,
            max_direct_layers=max_direct_layers,
            include_invisible=include_invisible,
            force_seven_layers=force_seven_layers,
            merge_lineart=merge_lineart,
            verbose=verbose,
        )

        if resize_max_size > 0 and out_dir is not None:
            resize_pngs_inplace(out_dir, int(resize_max_size))

    # Convert exported PNG dataset to Lance
    dataset_root = results_root
    print(f"Converting exported PNGs to Lance dataset under: {dataset_root}", flush=True)
    lance_ds = lanceImport.transform2lance(
        dataset_dir=str(dataset_root),
        output_name="dataset",
        save_binary=False,
        not_save_disk=False,
        tag="PSDExport",
    )

    if lance_ds is None:
        return None

    lance_path = Path(dataset_root) / "dataset.lance"

    if export_lance_to_dir is not None:
        lanceexport.extract_from_lance(
            str(lance_path),
            str(export_lance_to_dir),
            version="PSDExport",
            clip_with_caption=False,
        )

    return lance_path


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Process PSD folder: export layers -> lance import -> optional lance export"
    )
    parser.add_argument("psd_dir", type=str, help="Folder containing PSD files")
    parser.add_argument(
        "--results-root",
        type=str,
        default="",
        help="Root output dir for exported PNGs (default: <psd_dir>/results)",
    )
    parser.add_argument("--max-direct-layers", type=int, default=7, help="Direct export threshold")
    parser.add_argument("--include-invisible", action="store_true", help="Include invisible layers")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--no-force-seven-layers",
        action="store_true",
        help="Disable fixed 7-layer export mode and fallback to dynamic export rules",
    )
    parser.add_argument(
        "--no-merge-lineart",
        action="store_true",
        help="Do not merge lineart layers (may produce more than 7 files)",
    )
    parser.add_argument(
        "--export-lance-to",
        type=str,
        default="",
        help="If set, extract from lance to this folder",
    )
    parser.add_argument(
        "--resize-max-size",
        type=int,
        default=0,
        help="If >0, also export resized PNGs (max edge, multiple of 16) and build lance from resized outputs",
    )
    parser.add_argument(
        "--resize-results-root",
        type=str,
        default="",
        help="Optional resized output root (default: sibling folder results_<max_size>)",
    )
    args = parser.parse_args()

    psd_dir = Path(args.psd_dir)
    results_root = Path(args.results_root) if args.results_root else None
    export_lance_to_dir = Path(args.export_lance_to) if args.export_lance_to else None
    resize_results_root = Path(args.resize_results_root) if args.resize_results_root else None

    lance_path = process_psd_folder(
        psd_dir=psd_dir,
        results_root=results_root,
        max_direct_layers=int(args.max_direct_layers),
        include_invisible=bool(args.include_invisible),
        force_seven_layers=not bool(args.no_force_seven_layers),
        merge_lineart=not bool(args.no_merge_lineart),
        verbose=bool(args.verbose),
        resize_max_size=int(args.resize_max_size),
        resize_results_root=resize_results_root,
        export_lance_to_dir=export_lance_to_dir,
    )

    if lance_path is None:
        print("No PSD processed.")
        return 0

    print(f"Lance dataset: {lance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
