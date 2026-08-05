from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from .format_plans import load_capability_profile
from .pipeline import run_auto_rig_item


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m module.auto_rig",
        description="Convert one completed see-through item into Spine 4.2 and Live2D artifacts.",
        allow_abbrev=False,
    )
    parser.add_argument("item", type=Path, help="Item directory or its final.psd")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _optional_env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    profile_id = os.environ.get("AUTO_RIG_PROFILE", "dual_runtime_core_v1")
    validation_tier = os.environ.get("AUTO_RIG_VALIDATION_TIER", "release")
    pose_mode = os.environ.get("AUTO_RIG_POSE_MODE", "auto")
    try:
        profile = load_capability_profile(profile_id)
        result = run_auto_rig_item(
            args.item,
            profile_id=profile_id,
            validation_tier=validation_tier,
            sdk_root=_optional_env("CUBISM_SDK_ROOT"),
            spine_runtime_path=_optional_env("SPINE_RUNTIME_VALIDATOR_PATH"),
            pose_mode=pose_mode,
            pose_model_cache_dir=_optional_env("QINGLONG_CAPTIONS_MODEL_CACHE"),
            sdpose_bundle_path=_optional_env("AUTO_RIG_SDPOSE_BUNDLE"),
            detrpose_weights_path=_optional_env("AUTO_RIG_DETRPOSE_WEIGHTS"),
            pose_device=_optional_env("AUTO_RIG_POSE_DEVICE"),
            prefer_pose_fa2=_env_bool("AUTO_RIG_POSE_FA2", default=True),
            finalize=profile.terminal_delivery,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "status": "failed",
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    terminal_status = result.terminal.payload.get("status") if result.terminal is not None else "stage_validated"
    print(
        json.dumps(
            {
                "item": str(result.item_root),
                "reused_stages": list(result.reused_stages),
                "status": terminal_status,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


__all__ = ["build_parser", "main", "parse_args"]
