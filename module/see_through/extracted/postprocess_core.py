# Extracted and adapted from:
# - inference/scripts/inference_psd.py
# - common/utils/inference_utils.py
# Upstream repository: shitagaki-lab/see-through
# Notes: globals removed, project-local IO/state conventions applied.

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..vendor_bootstrap import ensure_vendor_imports


def run_postprocess_core(*, source_path: Path, output_dir: Path, save_to_psd: bool, tblr_split: bool) -> dict[str, Path]:
    ensure_vendor_imports()

    import utils.inference_utils as vendor_inference_utils
    import utils.torchcv as vendor_torchcv

    original_torchcv_cluster_inpaint_part = vendor_torchcv.cluster_inpaint_part
    original_inference_cluster_inpaint_part = vendor_inference_utils.cluster_inpaint_part

    def cluster_inpaint_part_cv2(*args, **kwargs):
        kwargs.setdefault("inpaint", "cv2")
        return original_torchcv_cluster_inpaint_part(*args, **kwargs)

    vendor_torchcv.cluster_inpaint_part = cluster_inpaint_part_cv2
    vendor_inference_utils.cluster_inpaint_part = cluster_inpaint_part_cv2
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        vendor_inference_utils.further_extr(
            str(output_dir),
            rotate=False,
            save_to_psd=save_to_psd,
            tblr_split=tblr_split,
        )
    finally:
        vendor_torchcv.cluster_inpaint_part = original_torchcv_cluster_inpaint_part
        vendor_inference_utils.cluster_inpaint_part = original_inference_cluster_inpaint_part

    optimized_dir = output_dir / "optimized"
    optimized_dir.mkdir(parents=True, exist_ok=True)

    legacy_psd_path = output_dir.parent / f"{output_dir.name}.psd"
    legacy_depth_psd_path = output_dir.parent / f"{output_dir.name}_depth.psd"
    legacy_json_path = output_dir.parent / f"{output_dir.name}.psd.json"

    results: dict[str, Path] = {}
    if save_to_psd and legacy_psd_path.exists():
        final_psd_path = output_dir / "final.psd"
        if final_psd_path.exists():
            final_psd_path.unlink()
        shutil.move(str(legacy_psd_path), str(final_psd_path))
        results["psd"] = final_psd_path

    if legacy_depth_psd_path.exists():
        final_depth_psd_path = output_dir / "final_depth.psd"
        if final_depth_psd_path.exists():
            final_depth_psd_path.unlink()
        shutil.move(str(legacy_depth_psd_path), str(final_depth_psd_path))
        results["depth_psd"] = final_depth_psd_path

    if legacy_json_path.exists():
        optimized_info_path = optimized_dir / "info.json"
        if optimized_info_path.exists():
            optimized_info_path.unlink()
        shutil.move(str(legacy_json_path), str(optimized_info_path))
        results["info"] = optimized_info_path
    elif (optimized_dir / "info.json").exists():
        results["info"] = optimized_dir / "info.json"

    manifest_path = optimized_dir / "manifest.json"
    manifest = {
        "source_path": str(source_path),
        "save_to_psd": bool(save_to_psd),
        "tblr_split": bool(tblr_split),
        "generated_files": sorted(path.name for path in optimized_dir.glob("*")),
        "final_psd": str(results["psd"]) if "psd" in results else None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    results["manifest"] = manifest_path
    return results
