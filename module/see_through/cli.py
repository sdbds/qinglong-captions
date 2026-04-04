from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.loader import load_config
from module.see_through.see_through_profile import (
    DEFAULT_DEPTH_RESOLUTION,
    DEFAULT_DEPTH_INFERENCE_STEPS,
    DEFAULT_SEED,
    DEFAULT_QUANT_MODE,
    normalize_quant_mode,
    resolve_see_through_repo_ids,
)

CONFIG_DIR = ROOT / "config"


@dataclass(frozen=True)
class SeeThroughRunConfig:
    input_dir: Path
    output_dir: Path
    repo_id_layerdiff: str
    repo_id_depth: str
    resolution: int
    resolution_depth: int
    inference_steps_depth: int
    seed: int
    dtype: str
    quant_mode: str
    group_offload: bool
    offload_policy: str
    skip_completed: bool
    continue_on_error: bool
    save_to_psd: bool
    tblr_split: bool
    limit_images: int
    force_eager_attention: bool
    vae_ckpt: str | None = None
    unet_ckpt: str | None = None


def build_parser(config_dir: str | Path = CONFIG_DIR) -> argparse.ArgumentParser:
    config = load_config(str(config_dir))
    defaults = dict(config.get("see_through", {}))

    parser = argparse.ArgumentParser(description="Run see-through batch processing over an input image directory.")
    parser.add_argument("--config_dir", default=str(config_dir), help="Path to config directory containing model.toml")
    parser.add_argument("--input_dir", required=True, help="Input image directory")
    parser.add_argument(
        "--output_dir",
        default=str(defaults.get("output_dir", "workspace/see_through_output")),
        help="Final artifact directory for this run",
    )
    parser.add_argument("--repo_id_layerdiff", default=defaults.get("repo_id_layerdiff", "layerdifforg/seethroughv0.0.2_layerdiff3d"))
    parser.add_argument("--repo_id_depth", default=defaults.get("repo_id_depth", "24yearsold/seethroughv0.0.1_marigold"))
    parser.add_argument("--resolution", type=int, default=int(defaults.get("resolution", 1280)))
    parser.add_argument(
        "--resolution_depth",
        type=int,
        default=int(defaults.get("resolution_depth", DEFAULT_DEPTH_RESOLUTION)),
        help="Marigold depth inference resolution; set -1 to match the layerdiff canvas resolution.",
    )
    parser.add_argument(
        "--inference_steps_depth",
        type=int,
        default=int(defaults.get("inference_steps_depth", DEFAULT_DEPTH_INFERENCE_STEPS)),
        help="Marigold denoising steps; set -1 to use the pipeline default.",
    )
    parser.add_argument("--seed", type=int, default=int(defaults.get("seed", DEFAULT_SEED)))
    parser.add_argument("--dtype", default=str(defaults.get("dtype", "bfloat16")))
    parser.add_argument(
        "--quant_mode",
        choices=["none", "nf4"],
        default=str(defaults.get("quant_mode", DEFAULT_QUANT_MODE)),
        help="See-through model profile: 'none' uses the standard repos, 'nf4' switches to the pre-quantized repos.",
    )
    parser.add_argument(
        "--group_offload",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("group_offload", False)),
        help="Enable diffusers group offload for lower peak VRAM during inference.",
    )
    parser.add_argument("--offload_policy", choices=["delete", "cpu"], default=str(defaults.get("offload_policy", "delete")))
    parser.add_argument("--skip_completed", action=argparse.BooleanOptionalAction, default=bool(defaults.get("skip_completed", True)))
    parser.add_argument("--continue_on_error", action=argparse.BooleanOptionalAction, default=bool(defaults.get("continue_on_error", True)))
    parser.add_argument("--save_to_psd", action=argparse.BooleanOptionalAction, default=bool(defaults.get("save_to_psd", True)))
    parser.add_argument("--tblr_split", action=argparse.BooleanOptionalAction, default=bool(defaults.get("tblr_split", False)))
    parser.add_argument("--limit_images", type=int, default=int(defaults.get("limit_images", 0)))
    parser.add_argument("--force_eager_attention", action=argparse.BooleanOptionalAction, default=bool(defaults.get("force_eager_attention", False)))
    parser.add_argument("--vae_ckpt", default=defaults.get("vae_ckpt"))
    parser.add_argument("--unet_ckpt", default=defaults.get("unet_ckpt"))
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config_dir", default=str(CONFIG_DIR))
    known, _ = probe.parse_known_args(list(argv) if argv is not None else None)
    parser = build_parser(known.config_dir)
    return parser.parse_args(list(argv) if argv is not None else None)


def build_run_config(args: argparse.Namespace) -> SeeThroughRunConfig:
    quant_mode = normalize_quant_mode(args.quant_mode)
    resolved_repos = resolve_see_through_repo_ids(
        quant_mode=quant_mode,
        repo_id_layerdiff=args.repo_id_layerdiff,
        repo_id_depth=args.repo_id_depth,
    )
    return SeeThroughRunConfig(
        input_dir=Path(args.input_dir).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        repo_id_layerdiff=resolved_repos.repo_id_layerdiff,
        repo_id_depth=resolved_repos.repo_id_depth,
        resolution=int(args.resolution),
        resolution_depth=int(args.resolution_depth),
        inference_steps_depth=int(args.inference_steps_depth),
        seed=int(args.seed),
        dtype=str(args.dtype),
        quant_mode=quant_mode,
        group_offload=bool(args.group_offload),
        offload_policy=str(args.offload_policy),
        skip_completed=bool(args.skip_completed),
        continue_on_error=bool(args.continue_on_error),
        save_to_psd=bool(args.save_to_psd),
        tblr_split=bool(args.tblr_split),
        limit_images=int(args.limit_images),
        force_eager_attention=bool(args.force_eager_attention),
        vae_ckpt=args.vae_ckpt,
        unet_ckpt=args.unet_ckpt,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_run_config(args)
    if not config.input_dir.exists() or not config.input_dir.is_dir():
        raise SystemExit("input_dir must be an existing directory")
    from module.see_through.runner import run_see_through_batch

    return run_see_through_batch(config)


if __name__ == "__main__":
    sys.exit(main())
