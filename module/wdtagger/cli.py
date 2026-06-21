from __future__ import annotations

import argparse

from module.wdtagger import constants
from utils.wdtagger_siglip2 import (
    CL_TAGGER_V2_DEFAULT_VERSION,
    default_cl_tagger_v2_threshold,
    is_cl_tagger_v2_repo,
    normalize_cl_tagger_v2_version,
)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="Directory containing images to process")
    parser.add_argument(
        "--repo_id",
        type=str,
        default=constants.DEFAULT_WD14_TAGGER_REPO,
        help="Repository ID for WD14 tagger model on Hugging Face",
    )
    parser.add_argument("--model_dir", type=str, default="wd14_tagger_model", help="Directory to store WD14 tagger model")
    parser.add_argument("--force_download", action="store_true", help="Force downloading WD14 tagger model")
    parser.add_argument(
        "--cl_tagger_v2_version",
        type=str,
        default=CL_TAGGER_V2_DEFAULT_VERSION,
        help="cl_tagger v2 model version to download, e.g. v2_01a or 2.01a",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--caption_extension", type=str, default=".txt", help="Extension for caption files")
    parser.add_argument(
        "--thresh",
        type=float,
        default=None,
        help="Default threshold for tag confidence; cl_tagger v2 defaults to the selected version recommendation",
    )
    parser.add_argument("--general_threshold", type=float, default=None, help="Threshold for general category tags (defaults to --thresh)")
    parser.add_argument("--character_threshold", type=float, default=None, help="Threshold for character category tags (defaults to --thresh)")
    parser.add_argument("--overwrite", action="store_true", help="Skip processing images in subfolders")
    parser.add_argument("--remove_underscore", action="store_true", help="Replace underscores with spaces in output tags")
    parser.add_argument("--undesired_tags", type=str, default="", help="Comma-separated list of tags to exclude from output")
    parser.add_argument("--frequency_tags", action="store_true", help="Sort final tags by confidence score instead of default order.")
    parser.add_argument("--add_tags_threshold", action="store_true", help="Add confidence threshold after each tag in output")
    parser.add_argument("--append_tags", action="store_true", help="Append new tags to existing caption files instead of overwriting")
    parser.add_argument("--use_rating_tags", action="store_true", help="Add rating tags as the first tag")
    parser.add_argument("--use_quality_tags", action="store_true", help="Add quality tags to the output.")
    parser.add_argument("--use_model_tags", action="store_true", help="Add model tags to the output.")
    parser.add_argument("--use_rating_tags_as_last_tag", action="store_true", help="Add rating tags as the last tag")
    parser.add_argument("--character_tags_first", action="store_true", help="Insert character tags before general tags")
    parser.add_argument(
        "--always_first_tags",
        type=str,
        default=None,
        help="Comma-separated list of tags to always put at the beginning (e.g. '1girl,1boy')",
    )
    parser.add_argument("--caption_separator", type=str, default=", ", help="Separator for caption tags (include spaces if needed)")
    parser.add_argument(
        "--tag_replacement",
        type=str,
        default=None,
        help="Tag replacements in format 'source1,target1;source2,target2'. Escape ',' and ';' with '\\'",
    )
    parser.add_argument(
        "--character_tag_expand",
        action="store_true",
        help="Expand character tags with parentheses (e.g. 'name_(series)' becomes 'name, series')",
    )
    parser.add_argument(
        "--remove_parents_tag",
        action="store_true",
        help="Remove parent tags if a child tag is present (e.g., remove 'uniform' if 'school_uniform' is present).",
    )
    parser.add_argument(
        "--merge_batch_size",
        type=int,
        default=100,
        help="Batch size for merge_insert when --lance_update_mode=merge (default: 100)",
    )
    parser.add_argument(
        "--lance_update_mode",
        choices=("rebuild", "merge", "none"),
        default="rebuild",
        help="How to write captions back to Lance: rebuild from sidecar files after tagging (default), use Lance merge_insert incrementally, or skip Lance writes.",
    )
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.cl_tagger_v2_version = normalize_cl_tagger_v2_version(args.cl_tagger_v2_version)

    if args.thresh is None:
        args.thresh = default_cl_tagger_v2_threshold(args.cl_tagger_v2_version) if is_cl_tagger_v2_repo(args.repo_id) else 0.35

    if args.general_threshold is None:
        args.general_threshold = args.thresh
    if args.character_threshold is None:
        args.character_threshold = args.thresh

    return args
