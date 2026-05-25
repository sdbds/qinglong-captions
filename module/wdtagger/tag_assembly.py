from __future__ import annotations

import argparse
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from module.wdtagger import constants
from module.wdtagger.taxonomy import LabelData
from utils.tag_highlighting import get_tag_classifier


def get_tags_official(
    probs,
    labels: LabelData,
    gen_threshold,
    char_threshold,
    use_rating_tags,
    use_quality_tags,
    use_model_tags,
    processed_names=None,
):
    tag_names = processed_names if processed_names is not None else labels.names
    result = {
        "rating": [],
        "general": [],
        "character": [],
        "copyright": [],
        "artist": [],
        "meta": [],
        "quality": [],
        "model": [],
    }

    pick_highest_categories = []
    if use_rating_tags:
        pick_highest_categories.append("rating")
    if use_quality_tags:
        pick_highest_categories.append("quality")
    if use_model_tags:
        pick_highest_categories.append("model")

    for category_name in pick_highest_categories:
        category_indices = labels.category_indices.get(category_name)
        if category_indices is not None and len(category_indices) > 0:
            valid_indices = category_indices[category_indices < len(probs)]
            if len(valid_indices) > 0:
                category_probs = probs[valid_indices]
                best_local_idx = np.argmax(category_probs)
                confidence = category_probs[best_local_idx]
                global_idx = valid_indices[best_local_idx]
                tag_name = tag_names[global_idx]
                result.setdefault(category_name, []).append((tag_name, float(confidence)))

    category_map = {}
    for category_name, category_indices in labels.category_indices.items():
        if category_name in pick_highest_categories:
            continue
        threshold = char_threshold if category_name in ["character", "copyright", "artist"] else gen_threshold
        category_map[category_name] = (category_indices, threshold)

    for category_name, (category_indices, threshold) in category_map.items():
        if len(category_indices) > 0:
            valid_indices = category_indices[category_indices < len(probs)]
            if len(valid_indices) == 0:
                continue

            mask = probs[valid_indices] >= threshold
            passed_indices = valid_indices[mask]
            for idx in passed_indices:
                if idx < len(tag_names) and tag_names[idx] is not None:
                    tag_name = tag_names[idx]
                    confidence = probs[idx]
                    result.setdefault(category_name, []).append((tag_name, float(confidence)))

    for key in result:
        result[key] = sorted(result[key], key=lambda x: x[1], reverse=True)

    return result


def process_tags(label_data: LabelData, args: argparse.Namespace) -> List[str]:
    processed_names = label_data.names.copy()

    if args.undesired_tags:
        undesired_set = {tag.strip() for tag in args.undesired_tags.split(",")}
        constants.console.print(f"[blue]Undesired tags: {undesired_set}[/blue]")
        constants.console.print(f"[blue]Excluding {len(undesired_set)} undesired tags...[/blue]")
        for i, name in enumerate(processed_names):
            if name in undesired_set:
                processed_names[i] = ""

    if args.tag_replacement:
        replacement_map = {}
        escaped_replacements = args.tag_replacement.replace("\\,", "<COMMA>").replace("\\;", "<SEMICOLON>")
        for pair in escaped_replacements.split(";"):
            parts = pair.split(",")
            if len(parts) == 2:
                old_tag = parts[0].strip().replace("<COMMA>", ",").replace("<SEMICOLON>", ";")
                new_tag = parts[1].strip().replace("<COMMA>", ",").replace("<SEMICOLON>", ";")
                if old_tag:
                    replacement_map[old_tag] = new_tag

        if replacement_map:
            constants.console.print(f"[blue]Replacement map: {replacement_map}[/blue]")
            constants.console.print(f"[blue]Applying {len(replacement_map)} tag replacements...[/blue]")
            processed_names = [replacement_map.get(name, name) for name in processed_names]

    if args.remove_underscore:
        constants.console.print("[blue]Removing underscores from tags...[/blue]")
        processed_names = [name.replace("_", " ") if len(name) > 3 else name for name in processed_names]

    if args.character_tag_expand:
        character_indices = label_data.category_indices.get("character", np.array([]))
        for i in character_indices:
            if i < len(processed_names):
                tag = processed_names[i]
                if tag and tag.endswith(")"):
                    parts = tag.split("(")
                    if len(parts) > 1:
                        processed_names[i] = "(".join(parts[:-1]).strip()

    return processed_names


def assemble_final_tags(
    tags_result: Dict[str, List[Tuple[str, float]]],
    args: argparse.Namespace,
    parent_to_child_map: Dict[str, List[str]],
    tag_freq: Optional[Dict[str, int]] = None,
) -> List[str]:
    found_tags = []
    all_tags_by_category = {}
    for category, tags_with_conf in tags_result.items():
        if not tags_with_conf:
            continue
        if category == "unknown":
            continue
        if category == "quality" and not args.use_quality_tags:
            continue
        if category == "rating" and not args.use_rating_tags:
            continue
        if category == "model" and not args.use_model_tags:
            continue

        best_list = [max(tags_with_conf, key=lambda x: x[1])] if category in ("quality", "rating", "model") else tags_with_conf
        if args.add_tags_threshold:
            all_tags_by_category[category] = [f"{tag}:{conf:.2f}" for tag, conf in best_list if tag]
        else:
            all_tags_by_category[category] = [tag for tag, conf in best_list if tag]

    rating_related_tags = ["rating", "quality", "meta", "model"]
    character_related_tags = ["character", "copyright", "artist"]

    if args.use_rating_tags and not args.use_rating_tags_as_last_tag:
        for category in rating_related_tags:
            found_tags.extend(all_tags_by_category.get(category, []))

    if args.character_tags_first:
        for category in character_related_tags:
            found_tags.extend(all_tags_by_category.get(category, []))
        found_tags.extend(all_tags_by_category.get("general", []))
    else:
        found_tags.extend(all_tags_by_category.get("general", []))
        for category in character_related_tags:
            found_tags.extend(all_tags_by_category.get(category, []))

    if args.use_rating_tags and args.use_rating_tags_as_last_tag:
        for category in rating_related_tags:
            found_tags.extend(all_tags_by_category.get(category, []))

    processed_categories = set(rating_related_tags) | set(character_related_tags) | {"general"}
    remaining_categories = set(all_tags_by_category.keys()) - processed_categories
    for category in sorted(list(remaining_categories)):
        found_tags.extend(all_tags_by_category[category])

    if args.frequency_tags:
        confidence_map = {tag: conf for category in tags_result.values() for tag, conf in category}
        found_tags.sort(key=lambda tag: confidence_map.get(tag, 0.0), reverse=True)

    if args.always_first_tags:
        always_first = [tag.strip() for tag in args.always_first_tags.split(",")]
        existing_first_tags = [tag for tag in always_first if tag in found_tags]
        other_tags = [tag for tag in found_tags if tag not in existing_first_tags]
        found_tags = existing_first_tags + other_tags

    if args.remove_parents_tag and parent_to_child_map:
        found_tags_set = set(found_tags)
        tags_to_remove = set()
        for parent_tag, child_tags in parent_to_child_map.items():
            if parent_tag in found_tags_set and any(child_tag in found_tags_set for child_tag in child_tags):
                tags_to_remove.add(parent_tag)
        if tags_to_remove:
            found_tags = [tag for tag in found_tags if tag not in tags_to_remove]

    if args.remove_parents_tag:
        noun_root_counts = {}
        for tag in found_tags:
            parts = tag.split()
            if not parts:
                continue
            noun_root = parts[-1]
            noun_root_counts[noun_root] = noun_root_counts.get(noun_root, 0) + 1
        found_tags = [tag for tag in found_tags if not (len(tag.split()) == 1 and noun_root_counts.get(tag.split()[-1], 0) > 1)]

    if tag_freq is not None:
        for tag in found_tags:
            clean_tag = tag.split(":")[0] if ":" in tag else tag
            tag_freq[clean_tag] = tag_freq.get(clean_tag, 0) + 1

    return found_tags


def assemble_tags_json(
    tags_result: Dict[str, List[Tuple[str, float]]],
    *,
    add_tags_threshold: bool,
    remove_parents_tag: bool,
    parent_to_child_map: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[str]]:
    category_to_tags: Dict[str, List[str]] = {}
    flat_tags_set: set = set()
    base_categories = [
        "rating",
        "general",
        "character",
        "copyright",
        "artist",
        "meta",
        "quality",
        "model",
    ]
    all_categories = base_categories + sorted(category for category in tags_result if category not in base_categories)

    for category in all_categories:
        tags_with_conf = tags_result.get(category, [])
        if not tags_with_conf:
            category_to_tags[category] = []
            continue

        best_list = [max(tags_with_conf, key=lambda x: x[1])] if category in ("quality", "rating", "model") else tags_with_conf
        if add_tags_threshold:
            tags = [f"{tag}:{confidence:.2f}" for tag, confidence in best_list if tag]
        else:
            tags = [tag for tag, _ in best_list if tag]

        category_to_tags[category] = tags
        for tag in tags:
            clean = tag.split(":")[0] if ":" in tag else tag
            flat_tags_set.add(clean)

    if remove_parents_tag and parent_to_child_map:
        tags_to_remove = set()
        for parent_tag, child_tags in parent_to_child_map.items():
            if parent_tag in flat_tags_set and any(child in flat_tags_set for child in child_tags):
                tags_to_remove.add(parent_tag)

        if tags_to_remove:
            for category, tags in list(category_to_tags.items()):
                kept = []
                for tag in tags:
                    clean = tag.split(":")[0] if ":" in tag else tag
                    if clean not in tags_to_remove:
                        kept.append(tag)
                category_to_tags[category] = kept

    return category_to_tags


def split_name_series(names: str) -> str:
    name_list = []
    items = [item.strip().replace("_", ":") for item in names.split(",")]
    for item in items:
        if item.endswith(" (cosplay)"):
            item = item.replace(" (cosplay)", "")
        if ("c.c_") or ("c.c") in item:
            item = item.replace("c.c_", "c.c.")
        if ("k:da") in item:
            item = item.replace("k:da", "k/da")
        if ("ranma 1:2") in item:
            item = item.replace("ranma 1:2", "ranma 1/2")
        match = re.match(r"(.*)\((.*?)\)$", item)
        if match and match.group(2).strip() not in constants.SERIES_EXCLUDE_LIST:
            full_name = match.group(1).strip()
            series = match.group(2).strip()
            name_list.append(f"<{full_name}> from ({series})")
        else:
            name_list.append(f"<{item}>")
    return ", ".join(name_list)


def format_description(text: str, tag_description: str = "") -> str:
    has_green = bool(re.search(r"<[^>]+>", text))
    has_purple = bool(re.search(r"\([^)]+\)", text))
    text = re.sub(r"<([^>]+)>", r"[magenta]\1[/magenta]", text)
    text = re.sub(r"\(([^)]+)\)", r"[dark_magenta]\1[/dark_magenta]", text)

    tag_classifier = get_tag_classifier()
    matched_tags = set()
    tags = []
    if tag_description:
        tags = [tag.strip() for tag in tag_description.split(",") if tag.strip()]
        if tags:
            tags_sorted = sorted(tags, key=len, reverse=True)
            pattern = r"(?<!\w)(" + "|".join(re.escape(tag) for tag in tags_sorted) + r")(?!\w)"

            def _replace(match: re.Match) -> str:
                matched_tags.add(match.group(0).lower())
                return tag_classifier.get_colored_tag(match.group(0))

            text = re.sub(pattern, _replace, text, flags=re.IGNORECASE)

    highlight_count = len(matched_tags) + int(has_green) + int(has_purple)
    colors = ["red", "orange", "yellow", "green", "cyan", "blue", "magenta"]
    tag_count = len(tags)
    rate = (highlight_count / tag_count * 100) if tag_count else 0
    color_index = min(int(rate / (100 / len(colors))), len(colors) - 1)
    color = colors[color_index]
    style = f"{color} bold" if rate > 50 else color
    highlight_rate = f"[{style}]{rate:.2f}%[/{style}]"

    return text, highlight_rate
