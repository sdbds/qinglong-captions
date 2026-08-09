"""Application-owned scorer defaults and threshold policy."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import tomlkit

DEFAULT_SCORER = "aesthetic_predictor_v2_5"
DEFAULT_THRESHOLD_COLOR = "white"
_THRESHOLD_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class Threshold:
    name: str
    max_score: float
    color: str = DEFAULT_THRESHOLD_COLOR

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _THRESHOLD_NAME.fullmatch(self.name):
            raise ValueError(f"invalid threshold name: {self.name!r}")
        if isinstance(self.max_score, bool) or not isinstance(self.max_score, (int, float)):
            raise ValueError("threshold max_score must be numeric")
        normalized_score = float(self.max_score)
        if not math.isfinite(normalized_score):
            raise ValueError("threshold max_score must be finite")
        if not isinstance(self.color, str) or not self.color.strip():
            raise ValueError("threshold color must be a nonempty string")
        object.__setattr__(self, "max_score", normalized_score)
        object.__setattr__(self, "color", self.color.strip())

    @property
    def folder_name(self) -> str:
        return self.name.replace("_", " ")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "max_score": self.max_score,
            "color": self.color,
        }


@dataclass(frozen=True, slots=True)
class RewardPolicy:
    default_scorer: str
    profiles: Mapping[str, tuple[Threshold, ...]]

    @property
    def scorer_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.profiles))

    def thresholds_for(self, scorer: str) -> tuple[Threshold, ...]:
        return self.profiles.get(scorer, ())


def normalize_thresholds(rows: object, *, scorer: str) -> tuple[Threshold, ...]:
    if not isinstance(rows, list):
        raise ValueError(f"thresholds for scorer {scorer!r} must be a list")

    normalized: list[Threshold] = []
    names: set[str] = set()
    scores: set[float] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"threshold row {index} for scorer {scorer!r} must be a table"
            )
        if "name" not in raw:
            raise ValueError(f"threshold row {index} for scorer {scorer!r} needs name")
        if "max_score" not in raw:
            raise ValueError(
                f"threshold row {index} for scorer {scorer!r} needs max_score"
            )
        try:
            row = Threshold(
                name=raw["name"],
                max_score=raw["max_score"],
                color=raw.get("color", DEFAULT_THRESHOLD_COLOR),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid threshold row {index} for scorer {scorer!r}: {error}"
            ) from error
        if row.name in names:
            raise ValueError(
                f"duplicate threshold name {row.name!r} at row {index} "
                f"for scorer {scorer!r}"
            )
        if row.max_score in scores:
            raise ValueError(
                f"duplicate max_score {row.max_score!r} at row {index} "
                f"for scorer {scorer!r}"
            )
        names.add(row.name)
        scores.add(row.max_score)
        normalized.append(row)

    return tuple(sorted(normalized, key=lambda row: row.max_score))


def parse_reward_policy(config: Mapping[str, Any]) -> RewardPolicy:
    if not isinstance(config, Mapping):
        raise ValueError("configuration root must be a table")
    raw_reward = config.get("reward_model", {})
    if not isinstance(raw_reward, Mapping):
        raise ValueError("reward_model must be a table")

    default_scorer = raw_reward.get("default_scorer", DEFAULT_SCORER)
    if not isinstance(default_scorer, str) or not default_scorer.strip():
        raise ValueError("reward_model.default_scorer must be a nonempty string")
    default_scorer = default_scorer.strip()

    raw_scorers = raw_reward.get("scorers", {})
    if not isinstance(raw_scorers, Mapping):
        raise ValueError("reward_model.scorers must be a table")

    profiles: dict[str, tuple[Threshold, ...]] = {}
    for scorer, raw_profile in raw_scorers.items():
        if not isinstance(scorer, str) or not scorer.strip():
            raise ValueError("reward_model scorer names must be nonempty strings")
        if not isinstance(raw_profile, Mapping):
            raise ValueError(f"profile for scorer {scorer!r} must be a table")
        raw_rows = raw_profile.get("thresholds", [])
        profiles[scorer] = normalize_thresholds(raw_rows, scorer=scorer)

    return RewardPolicy(
        default_scorer=default_scorer,
        profiles=MappingProxyType(profiles),
    )


def load_reward_policy(config_dir: str | Path) -> RewardPolicy:
    from config.loader import load_config

    try:
        config = load_config(str(config_dir))
    except OSError:
        return parse_reward_policy({})
    return parse_reward_policy(config)


def assign_threshold(
    score: float,
    rows: Sequence[Threshold],
) -> Threshold | None:
    if not rows:
        return None
    for row in rows:
        if score <= row.max_score:
            return row
    return rows[-1]


def _table(parent: Any, key: str) -> Any:
    value = parent.get(key)
    if value is None:
        value = tomlkit.table()
        parent[key] = value
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a TOML table")
    return value


def save_threshold_profile(
    model_toml: str | Path,
    scorer: str,
    rows: object,
) -> tuple[Threshold, ...]:
    if not isinstance(scorer, str) or not scorer.strip():
        raise ValueError("scorer must be a nonempty string")
    scorer = scorer.strip()
    normalized = normalize_thresholds(rows, scorer=scorer)

    path = Path(model_toml)
    document = (
        tomlkit.parse(path.read_text(encoding="utf-8"))
        if path.exists()
        else tomlkit.document()
    )
    reward_model = _table(document, "reward_model")
    scorers = _table(reward_model, "scorers")
    profile = _table(scorers, scorer)

    if normalized:
        threshold_tables = tomlkit.aot()
        for row in normalized:
            table = tomlkit.table()
            table.add("name", row.name)
            table.add("max_score", row.max_score)
            table.add("color", row.color)
            threshold_tables.append(table)
        profile["thresholds"] = threshold_tables
    elif "thresholds" in profile:
        del profile["thresholds"]

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(tomlkit.dumps(document), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return normalized


__all__ = [
    "DEFAULT_SCORER",
    "DEFAULT_THRESHOLD_COLOR",
    "RewardPolicy",
    "Threshold",
    "assign_threshold",
    "load_reward_policy",
    "normalize_thresholds",
    "parse_reward_policy",
    "save_threshold_profile",
]
