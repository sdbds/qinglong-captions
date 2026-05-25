from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class LabelData:
    """Tag metadata with category indexes."""

    names: List[str]
    category_indices: Dict[str, np.ndarray]
    tag_index_to_category: Dict[int, str]

    def __post_init__(self):
        for category, indices in self.category_indices.items():
            setattr(self, category, indices)

    def get_tags_by_category(self, category: str) -> List[str]:
        indices = self.category_indices.get(category.lower(), np.array([], dtype=np.int64))
        return [self.names[i] for i in indices if i < len(self.names) and self.names[i]]


def load_cl_tagger_label_data(json_path: Path) -> tuple[LabelData, int]:
    if not json_path.exists():
        raise Exception(f"Tags file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        tag_data = json.load(f)

    tag_data_int_keys = {int(k): v for k, v in tag_data.items()}
    idx_to_tag = {idx: data["tag"] for idx, data in tag_data_int_keys.items()}
    tag_to_category = {data["tag"]: data["category"] for data in tag_data_int_keys.values()}

    max_idx = max(idx_to_tag.keys())
    names = [None] * (max_idx + 1)
    for idx, tag in idx_to_tag.items():
        names[idx] = tag

    category_to_tags = {}
    for tag, category in tag_to_category.items():
        if category not in category_to_tags:
            category_to_tags[category] = []
        category_to_tags[category].append(tag)

    tag_to_idx = {tag: i for i, tag in idx_to_tag.items()}
    category_indices = {}
    for category, tags_in_category in category_to_tags.items():
        category_key = category.lower()
        indices = [tag_to_idx[tag] for tag in tags_in_category if tag in tag_to_idx]
        category_indices[category_key] = np.array(indices, dtype=np.int64)

    tag_index_to_category = {
        idx: category.lower()
        for category, tags in category_to_tags.items()
        for tag in tags
        if (idx := tag_to_idx.get(tag)) is not None
    }

    return (
        LabelData(
            names=names,
            category_indices=category_indices,
            tag_index_to_category=tag_index_to_category,
        ),
        len(tag_data),
    )


def load_csv_label_data(csv_path: Path) -> tuple[LabelData, int]:
    if not csv_path.exists():
        raise Exception(f"Tags file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        line = [row for row in reader]
        header = line[0]
        rows = line[1:]

    assert header[0] == "tag_id" and header[1] == "name" and header[2] == "category", f"unexpected csv format: {header}"

    names = []
    rating_indices, general_indices, character_indices = [], [], []
    for i, row in enumerate(rows):
        tag_name = row[1]
        category = row[2]
        names.append(tag_name)

        if category == "9":
            rating_indices.append(i)
        elif category == "0":
            general_indices.append(i)
        elif category == "4":
            character_indices.append(i)

    category_indices = {
        "rating": np.array(rating_indices, dtype=np.int64),
        "general": np.array(general_indices, dtype=np.int64),
        "character": np.array(character_indices, dtype=np.int64),
        "copyright": np.array([], dtype=np.int64),
        "artist": np.array([], dtype=np.int64),
        "meta": np.array([], dtype=np.int64),
        "quality": np.array([], dtype=np.int64),
        "model": np.array([], dtype=np.int64),
    }
    tag_index_to_category = {}
    for cat, indices in category_indices.items():
        for idx in indices:
            tag_index_to_category[idx] = cat

    return (
        LabelData(
            names=names,
            category_indices=category_indices,
            tag_index_to_category=tag_index_to_category,
        ),
        len(rows),
    )


def load_parent_to_child_map(csv_file_path: Path) -> Dict[str, List[str]]:
    parent_to_child_map: Dict[str, List[str]] = {}
    with csv_file_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        line = [row for row in reader]
        header = line[0]
        rows = line[1:]

    assert header[3] == "antecedent_name" and header[4] == "consequent_name", f"unexpected csv format: {header}"
    for row in rows:
        child_tag, parent_tag = row[3], row[4]
        if parent_tag not in parent_to_child_map:
            parent_to_child_map[parent_tag] = []
        parent_to_child_map[parent_tag].append(child_tag)
    return parent_to_child_map


def normalize_parent_to_child_map(parent_to_child_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
    normalized_map: Dict[str, List[str]] = {}
    for parent, children in parent_to_child_map.items():
        norm_parent = parent.replace("_", " ") if len(parent) > 3 else parent
        target_list = normalized_map.setdefault(norm_parent, [])
        for child in children:
            norm_child = child.replace("_", " ") if len(child) > 3 else child
            target_list.append(norm_child)
    for key in list(normalized_map.keys()):
        normalized_map[key] = sorted(set(normalized_map[key]))
    return normalized_map
