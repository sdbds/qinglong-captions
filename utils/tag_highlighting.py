from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import toml


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "config"
WDTAGGER_MODEL_DIR = ROOT_DIR / "wd14_tagger_model"
TAGS_JSON_PATH = ROOT_DIR / "datasets" / "tags.json"
CLASSIFIED_TAGS_CSV = WDTAGGER_MODEL_DIR / "selected_tags_classified.csv"


def _load_config() -> dict[str, Any]:
    try:
        from config.loader import load_config

        return load_config(str(CONFIG_DIR))
    except Exception:
        legacy_path = CONFIG_DIR / "config.toml"
        if legacy_path.exists():
            return toml.load(legacy_path)
        return {}


def _ensure_classified_tags_csv() -> Path:
    if CLASSIFIED_TAGS_CSV.exists():
        return CLASSIFIED_TAGS_CSV

    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return CLASSIFIED_TAGS_CSV

    try:
        hf_hub_download(
            repo_id="deepghs/sankaku_tags_categorize_for_WD14Tagger",
            filename="selected_tags_classified.csv",
            local_dir=str(WDTAGGER_MODEL_DIR),
            force_download=True,
            force_filename="selected_tags_classified.csv",
            repo_type="dataset",
        )
    except Exception:
        return CLASSIFIED_TAGS_CSV

    return CLASSIFIED_TAGS_CSV


def _merge_tags_json(tag_categories: dict[str, str]) -> None:
    if not TAGS_JSON_PATH.exists():
        return

    category_map = {
        "general": "0",
        "artist": "1",
        "studio": "2",
        "copyright": "3",
        "franchise": "3",
        "character": "4",
        "genre": "5",
        "medium": "8",
        "meta": "9",
        "fashion": "10",
        "anatomy": "11",
        "pose": "12",
        "activity": "13",
        "role": "14",
        "flora": "15",
        "fauna": "16",
        "entity": "17",
        "object": "18",
        "substance": "19",
        "setting": "20",
        "language": "21",
        "automatic": "22",
    }

    try:
        data = json.loads(TAGS_JSON_PATH.read_text(encoding="utf-8"))
    except Exception:
        return

    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        for category_name, tags in entry.items():
            category_id = category_map.get(category_name)
            if not category_id or not isinstance(tags, list):
                continue
            for tag in tags:
                tag_key = str(tag).lower()
                if tag_key and tag_key not in tag_categories:
                    tag_categories[tag_key] = category_id


@lru_cache(maxsize=1)
def _load_tag_type() -> dict[str, dict[str, Any]]:
    config = _load_config()
    fields = config.get("tag_type", {}).get("fields", [])
    return {str(field["id"]): field for field in fields if isinstance(field, dict) and "id" in field}


@lru_cache(maxsize=1)
def _load_tag_categories() -> dict[str, str]:
    csv_path = _ensure_classified_tags_csv()
    tag_categories: dict[str, str] = {}

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            tag_categories.update(
                {
                    row["name"].replace("_", " ").lower(): row["category"]
                    for row in reader
                    if row.get("name") and row.get("category")
                }
            )

    _merge_tags_json(tag_categories)
    return tag_categories


class TagClassifier:
    def __init__(
        self,
        tag_type: dict[str, dict[str, Any]] | None = None,
        tag_categories: dict[str, str] | None = None,
    ) -> None:
        self.tag_type = tag_type if tag_type is not None else _load_tag_type()
        self.tag_categories = tag_categories if tag_categories is not None else _load_tag_categories()

    def classify(self, tags: list[str]) -> dict[str, list[str]]:
        colored_tags: dict[str, list[str]] = {}
        for tag in tags:
            category_id = self.tag_categories.get(tag.lower(), "0")
            category_info = self.tag_type.get(category_id)
            color = category_info["color"] if category_info else "orange3"
            colored_tag = f"[{color}]{tag}[/{color}]"
            colored_tags.setdefault(category_id, []).append(colored_tag)
        return colored_tags

    def get_colored_tag(self, tag: str) -> str:
        suffix = ""
        if tag.endswith(","):
            tag = tag[:-1]
            suffix = ","
        elif tag.endswith("."):
            tag = tag[:-1]
            suffix = "."

        category_id = self.tag_categories.get(tag.lower(), "0")
        category_info = self.tag_type.get(category_id)
        color = category_info["color"] if category_info else "orange3"
        return f"[{color}]{tag}[/{color}]{suffix}"


@lru_cache(maxsize=1)
def get_tag_classifier() -> TagClassifier:
    return TagClassifier()
