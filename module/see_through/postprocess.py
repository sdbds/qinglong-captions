from __future__ import annotations

from pathlib import Path

from .extracted.postprocess_core import run_postprocess_core


def run_postprocess(*, source_path: Path, output_dir: Path, save_to_psd: bool, tblr_split: bool) -> dict[str, Path]:
    return run_postprocess_core(
        source_path=source_path,
        output_dir=output_dir,
        save_to_psd=save_to_psd,
        tblr_split=tblr_split,
    )

