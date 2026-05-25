from __future__ import annotations

import time
from pathlib import Path

from huggingface_hub import hf_hub_download

from module.onnx_runtime import OnnxModelSpec, load_single_model_bundle, resolve_tool_runtime_config
from module.wdtagger import constants
from module.wdtagger.taxonomy import (
    LabelData,
    load_cl_tagger_label_data,
    load_csv_label_data,
    load_parent_to_child_map,
    normalize_parent_to_child_map,
)
from utils.wdtagger_siglip2 import (
    CL_TAGGER_V2_DEFAULT_VERSION,
    is_cl_tagger_v2_repo,
    load_cl_tagger_v2_bundle,
)


def _download_model_files(args, *, hf_hub_download_fn=hf_hub_download) -> None:
    model_path = (
        Path(args.model_dir) / args.repo_id.replace("/", "_") / constants.CL_FILES[0]
        if args.repo_id.startswith("cella110n/cl_tagger")
        else Path(args.model_dir) / args.repo_id.replace("/", "_") / constants.FILES[0]
    )

    if not model_path.exists() or args.force_download:
        files_to_download = constants.CL_FILES if args.repo_id.startswith("cella110n/cl_tagger") else constants.FILES
        for file in files_to_download:
            file_path = Path(args.model_dir) / args.repo_id.replace("/", "_") / file
            if not file_path.exists() or args.force_download:
                file_path = Path(
                    hf_hub_download_fn(
                        repo_id=args.repo_id,
                        filename=file,
                        local_dir=Path(args.model_dir) / args.repo_id.replace("/", "_"),
                        force_download=args.force_download,
                    )
                )
                constants.console.print(f"[blue]Downloaded {file} to {file_path}[/blue]")
            else:
                constants.console.print(f"[green]Using existing {file}[/green]")


def _load_legacy_label_data(args) -> tuple[LabelData, int]:
    if args.repo_id.startswith("cella110n/cl_tagger"):
        json_path = Path(args.model_dir) / args.repo_id.replace("/", "_") / constants.JSON_FILE
        return load_cl_tagger_label_data(json_path)

    csv_path = Path(args.model_dir) / args.repo_id.replace("/", "_") / constants.CSV_FILE
    return load_csv_label_data(csv_path)


def _load_parent_map(args, *, hf_hub_download_fn=hf_hub_download):
    parent_to_child_map = {}
    if not args.remove_parents_tag:
        return parent_to_child_map

    csv_file_path = Path(args.model_dir) / constants.PARENTS_CSV
    if not csv_file_path.exists() or args.force_download:
        csv_file_path = Path(
            hf_hub_download_fn(
                repo_id="deepghs/danbooru_wikis_full",
                filename=constants.PARENTS_CSV,
                local_dir=args.model_dir,
                force_download=True,
                force_filename=constants.PARENTS_CSV,
                repo_type="dataset",
            )
        )
        constants.console.print(f"[blue]Downloaded {constants.PARENTS_CSV} to {csv_file_path}[/blue]")
    else:
        constants.console.print(f"[green]Using existing {constants.PARENTS_CSV}[/green]")

    parent_to_child_map = load_parent_to_child_map(csv_file_path)
    constants.console.print(f"[green]Loaded {len(parent_to_child_map)} parent tags.[/green]")

    if args.remove_underscore:
        constants.console.print("[blue]Normalizing underscores in parent/child tag map...[/blue]")
        parent_to_child_map = normalize_parent_to_child_map(parent_to_child_map)

    return parent_to_child_map


def load_model_and_tags(
    args,
    *,
    load_single_model_bundle_fn=load_single_model_bundle,
    load_cl_tagger_v2_bundle_fn=load_cl_tagger_v2_bundle,
    hf_hub_download_fn=hf_hub_download,
):
    start_time = time.time()
    runtime_config = resolve_tool_runtime_config(
        constants.CONFIG,
        tool_name="wdtagger",
        cli_override={"force_download": args.force_download},
    )

    if is_cl_tagger_v2_repo(args.repo_id):
        bundle = load_cl_tagger_v2_bundle_fn(
            repo_id=args.repo_id,
            model_dir=args.model_dir,
            runtime_config=runtime_config,
            version=getattr(args, "cl_tagger_v2_version", CL_TAGGER_V2_DEFAULT_VERSION),
            force_download=args.force_download,
            logger=constants.console.print,
        )
        label_data = LabelData(
            names=bundle.vocabulary.names,
            category_indices=bundle.vocabulary.category_indices,
            tag_index_to_category=bundle.vocabulary.tag_index_to_category,
        )
        total_tags = sum(1 for tag in label_data.names if tag)
        ort_sess = bundle.session
        input_name = bundle.inference_context
        constants.console.print(f"[blue]Providers: {bundle.providers}[/blue]")
    else:
        _download_model_files(args, hf_hub_download_fn=hf_hub_download_fn)
        label_data, total_tags = _load_legacy_label_data(args)
        spec = OnnxModelSpec(
            repo_id=args.repo_id,
            onnx_filename=constants.CL_FILES[0] if args.repo_id.startswith("cella110n/cl_tagger") else constants.FILES[0],
            local_dir=Path(args.model_dir) / args.repo_id.replace("/", "_"),
            bundle_key=f"wdtagger:{args.repo_id}",
        )
        bundle = load_single_model_bundle_fn(spec=spec, runtime_config=runtime_config, logger=constants.console.print)
        ort_sess = bundle.session
        input_name = bundle.input_metas[0].name if bundle.input_metas else ort_sess.get_inputs()[0].name
        constants.console.print(f"[blue]Providers: {bundle.providers}[/blue]")

    constants.console.print(f"[blue]Tags loaded: {total_tags} total[/blue]")
    for category, indices in sorted(label_data.category_indices.items()):
        if len(indices) > 0:
            constants.console.print(f"[blue]  - {category.capitalize()}: {len(indices)} tags[/blue]")

    parent_to_child_map = _load_parent_map(args, hf_hub_download_fn=hf_hub_download_fn)
    constants.console.print(f"[green]Model loaded in {time.time() - start_time:.2f} seconds[/green]")
    return ort_sess, input_name, label_data, parent_to_child_map
