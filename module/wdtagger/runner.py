from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from module.wdtagger import constants
from module.wdtagger.lance_io import (
    count_wdtagger_candidate_rows,
    merge_caption_updates,
    rebuild_lance_from_sidecars,
    resolve_dataset,
    resolve_lance_rebuild_source,
    scan_wdtagger_candidate_batches,
    update_wdtagger_tag,
)
from module.wdtagger.model_loader import load_model_and_tags
from module.wdtagger.outputs import write_sidecar_caption, write_tags_json
from module.wdtagger.preprocess import load_and_preprocess_batch, load_siglip2_rgb_batch, process_batch
from module.wdtagger.tag_assembly import assemble_final_tags, assemble_tags_json, get_tags_official, process_tags
from utils.console_util import print_exception
from utils.tag_highlighting import get_tag_classifier
from utils.wdtagger_siglip2 import is_cl_tagger_v2_repo


def _print_tag_frequencies(tag_freq: Dict[str, int]) -> None:
    constants.console.print("\n[yellow]Tag frequencies:[/yellow]")
    tag_classifier = get_tag_classifier()
    sorted_tags = sorted(tag_freq.items(), key=lambda item: item[1], reverse=True)

    for tag, freq in sorted_tags:
        classified_result = tag_classifier.classify([tag])
        colored_tag = tag
        for tag_list in classified_result.values():
            if tag_list:
                colored_tag = tag_list[0]
                break
        constants.console.print(f"{colored_tag}: {freq}")


def main(args, *, load_model_and_tags_fn=load_model_and_tags) -> None:
    dataset_ref = resolve_dataset(args.train_data_dir)
    dataset = dataset_ref.dataset

    lance_update_mode = getattr(args, "lance_update_mode", "rebuild")
    rebuild_source_dir = resolve_lance_rebuild_source(args.train_data_dir, dataset_ref.dataset_path)
    rebuild_output_name = dataset_ref.dataset_path.stem if dataset_ref.dataset_path is not None else "dataset"
    if lance_update_mode == "rebuild" and rebuild_source_dir is None:
        constants.console.print("[yellow]Lance rebuild source unavailable; using merge_insert updates.[/yellow]")
        lance_update_mode = "merge"

    ort_sess, input_name, label_data, parent_to_child_map = load_model_and_tags_fn(args)
    processed_names = process_tags(label_data, args)

    tag_freq: Dict[str, int] = {}
    total_images = count_wdtagger_candidate_rows(dataset, args)
    merge_batch_size = getattr(args, "merge_batch_size", 100)
    results: List[tuple[str, List[str]]] = []
    all_json_tags: Dict[str, Dict[str, List[str]]] = {}

    def flush_merge_insert() -> None:
        nonlocal lance_update_mode

        if not results:
            return
        if lance_update_mode != "merge":
            results.clear()
            return

        try:
            merge_caption_updates(dataset, results)
        except Exception as e:
            constants.console.print(
                "[yellow]Lance merge_insert failed; continuing with sidecar captions "
                "and rebuilding the dataset at the end.[/yellow]"
            )
            print_exception(constants.console, e, prefix="Lance merge_insert failed")
            lance_update_mode = "rebuild" if rebuild_source_dir is not None else "none"
        results.clear()

    with Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(spinner_name="dots"),
        MofNCompleteColumn(separator="/"),
        BarColumn(bar_width=40, complete_style="green", finished_style="bold green"),
        TextColumn("•"),
        TaskProgressColumn(),
        TextColumn("•"),
        TransferSpeedColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        expand=True,
    ) as progress:
        task = progress.add_task("[bold cyan]Processing images...", total=total_images)
        constants.console = progress.console

        for batch in scan_wdtagger_candidate_batches(dataset, args):
            uris = batch["uris"].to_pylist()

            if is_cl_tagger_v2_repo(args.repo_id):
                valid_uris, batch_images = load_siglip2_rgb_batch(uris)
            else:
                is_cl_tagger = args.repo_id.startswith("cella110n/cl_tagger")
                batch_images = load_and_preprocess_batch(uris, is_cl_tagger)
                valid_uris = uris

            if not batch_images:
                progress.update(task, advance=len(uris))
                continue

            probs = process_batch(batch_images, ort_sess, input_name)
            general_confidence = args.general_threshold or args.thresh
            character_confidence = args.character_threshold or args.thresh
            if probs is not None:
                for path, prob in zip(valid_uris, probs):
                    tags_result = get_tags_official(
                        prob,
                        label_data,
                        general_confidence,
                        character_confidence,
                        args.use_rating_tags,
                        args.use_quality_tags,
                        args.use_model_tags,
                        processed_names,
                    )
                    found_tags = assemble_final_tags(tags_result, args, parent_to_child_map, tag_freq)

                    output_path = Path(path).with_suffix(args.caption_extension)
                    if args.append_tags and output_path.exists():
                        with output_path.open("r", encoding="utf-8") as f:
                            existing_tags = f.read().strip()
                            found_tags = existing_tags.split(args.caption_separator) + found_tags

                    results.append((path, found_tags))
                    if len(results) >= merge_batch_size:
                        flush_merge_insert()

                    write_sidecar_caption(
                        path,
                        found_tags,
                        caption_extension=args.caption_extension,
                        caption_separator=args.caption_separator,
                    )

                    categorized = assemble_tags_json(
                        tags_result,
                        add_tags_threshold=args.add_tags_threshold,
                        remove_parents_tag=args.remove_parents_tag,
                        parent_to_child_map=parent_to_child_map,
                    )
                    all_json_tags[str(Path(path))] = categorized

            progress.update(task, advance=len(batch["uris"].to_pylist()))

    _print_tag_frequencies(tag_freq)
    write_tags_json(args.train_data_dir, all_json_tags)
    flush_merge_insert()

    lance_dataset_updated = False
    if lance_update_mode == "rebuild":
        rebuilt_dataset = rebuild_lance_from_sidecars(
            rebuild_source_dir,
            rebuild_output_name,
            dataset,
            args.caption_extension,
        )
        if rebuilt_dataset is not None:
            dataset = rebuilt_dataset
            lance_dataset_updated = True
    elif lance_update_mode == "merge":
        update_wdtagger_tag(dataset)
        lance_dataset_updated = True

    if lance_dataset_updated:
        constants.console.print("[green]Successfully updated dataset with new captions[/green]")
    else:
        constants.console.print("[green]Successfully wrote sidecar captions; Lance update was skipped[/green]")
