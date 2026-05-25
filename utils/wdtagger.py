from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huggingface_hub import hf_hub_download

from module.onnx_runtime import load_single_model_bundle
from module.wdtagger import constants
from module.wdtagger.cli import finalize_args, setup_parser
from module.wdtagger.lance_io import (
    count_wdtagger_candidate_rows as _count_wdtagger_candidate_rows,
    filter_uncaptioned_batch as _filter_uncaptioned_batch,
    is_empty_caption_list as _is_empty_caption_list,
    rebuild_lance_from_sidecars as _rebuild_lance_from_sidecars,
    resolve_lance_rebuild_source as _resolve_lance_rebuild_source,
    scan_wdtagger_candidate_batches as _scan_wdtagger_candidate_batches,
)
from module.wdtagger.model_loader import load_model_and_tags as _load_model_and_tags
from module.wdtagger.outputs import (
    has_sidecar_caption as _has_sidecar_caption,
    read_sidecar_caption as _read_sidecar_caption,
)
from module.wdtagger.preprocess import (
    load_and_preprocess_batch,
    load_siglip2_rgb_batch,
    preprocess_image,
    process_batch,
)
from module.wdtagger.runner import main as _runner_main
from module.wdtagger.tag_assembly import (
    assemble_final_tags,
    assemble_tags_json,
    format_description,
    get_tags_official,
    process_tags,
    split_name_series,
)
from module.wdtagger.taxonomy import LabelData
from utils.wdtagger_siglip2 import (
    CL_TAGGER_V2_DEFAULT_VERSION,
    Siglip2InferenceContext,
    default_cl_tagger_v2_threshold,
    is_cl_tagger_v2_repo,
    load_cl_tagger_v2_bundle,
    normalize_cl_tagger_v2_version,
    process_siglip2_batch,
)


console = constants.console
IMAGE_MIME_FILTER = constants.IMAGE_MIME_FILTER
IMAGE_SIZE = constants.IMAGE_SIZE
DEFAULT_WD14_TAGGER_REPO = constants.DEFAULT_WD14_TAGGER_REPO
FILES = constants.FILES
CL_FILES = constants.CL_FILES
CSV_FILE = constants.CSV_FILE
JSON_FILE = constants.JSON_FILE
PARENTS_CSV = constants.PARENTS_CSV
CONFIG_DIR = constants.CONFIG_DIR
SERIES_EXCLUDE_LIST = constants.SERIES_EXCLUDE_LIST
_cfg = constants.CONFIG


def load_model_and_tags(args):
    return _load_model_and_tags(
        args,
        load_single_model_bundle_fn=load_single_model_bundle,
        load_cl_tagger_v2_bundle_fn=load_cl_tagger_v2_bundle,
        hf_hub_download_fn=hf_hub_download,
    )


def main(args):
    return _runner_main(args, load_model_and_tags_fn=load_model_and_tags)


__all__ = [
    "CL_TAGGER_V2_DEFAULT_VERSION",
    "CONFIG_DIR",
    "CSV_FILE",
    "CL_FILES",
    "DEFAULT_WD14_TAGGER_REPO",
    "FILES",
    "IMAGE_MIME_FILTER",
    "IMAGE_SIZE",
    "JSON_FILE",
    "LabelData",
    "PARENTS_CSV",
    "SERIES_EXCLUDE_LIST",
    "Siglip2InferenceContext",
    "_cfg",
    "_count_wdtagger_candidate_rows",
    "_filter_uncaptioned_batch",
    "_has_sidecar_caption",
    "_is_empty_caption_list",
    "_read_sidecar_caption",
    "_rebuild_lance_from_sidecars",
    "_resolve_lance_rebuild_source",
    "_scan_wdtagger_candidate_batches",
    "assemble_final_tags",
    "assemble_tags_json",
    "console",
    "default_cl_tagger_v2_threshold",
    "finalize_args",
    "format_description",
    "get_tags_official",
    "hf_hub_download",
    "is_cl_tagger_v2_repo",
    "load_and_preprocess_batch",
    "load_cl_tagger_v2_bundle",
    "load_model_and_tags",
    "load_siglip2_rgb_batch",
    "load_single_model_bundle",
    "main",
    "normalize_cl_tagger_v2_version",
    "preprocess_image",
    "process_batch",
    "process_siglip2_batch",
    "process_tags",
    "setup_parser",
    "split_name_series",
]


if __name__ == "__main__":
    parser = setup_parser()
    main(finalize_args(parser.parse_args()))
