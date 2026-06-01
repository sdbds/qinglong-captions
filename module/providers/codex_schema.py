"""Shared Codex caption schema, prompt, and output parsing helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


CODEX_CAPTION_SCHEMA_VERSION = "codex-caption-v2"
CODEX_SCORE_DIMENSIONS = (
    "Costume & Makeup & Prop Presentation/Accuracy",
    "Character Portrayal & Posing",
    "Setting & Environment Integration",
    "Lighting & Mood",
    "Composition & Framing",
    "Storytelling & Concept",
    "Level of S*e*x*y",
    "Figure",
    "Overall Impact & Uniqueness",
)
CODEX_CAPTION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "short_description",
        "long_description",
        "tags",
        "rating",
        "confidence",
        "scores",
        "total_score",
        "average_score",
    ],
    "properties": {
        "short_description": {"type": "string"},
        "long_description": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "rating": {"type": "string", "enum": ["general", "sensitive", "questionable", "explicit"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "scores": {
            "type": "object",
            "additionalProperties": False,
            "required": list(CODEX_SCORE_DIMENSIONS),
            "properties": {dimension: {"type": "number", "minimum": 0, "maximum": 10} for dimension in CODEX_SCORE_DIMENSIONS},
        },
        "total_score": {"type": "number", "minimum": 0},
        "average_score": {"type": "number", "minimum": 0, "maximum": 10},
    },
}


class CodexCaptionOutputError(ValueError):
    """Raised when Codex returns data that does not match the caption contract."""

    def __init__(self, message: str, *, raw: str = ""):
        super().__init__(message)
        self.raw = raw


def write_default_caption_schema(path: str | Path) -> Path:
    schema_path = Path(path)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(CODEX_CAPTION_SCHEMA, ensure_ascii=False, indent=2), encoding="utf-8")
    return schema_path


def load_caption_schema(path: str | Path | None = None) -> dict[str, Any]:
    if not path:
        return dict(CODEX_CAPTION_SCHEMA)
    schema_path = Path(path).expanduser()
    if not schema_path.exists():
        raise FileNotFoundError(f"Codex output schema does not exist: {schema_path}")
    loaded = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Codex output schema must be a JSON object: {schema_path}")
    return loaded


def build_codex_caption_prompt(*, system_prompt: str, user_prompt: str) -> str:
    sections = [
        "You are a captioning and rating engine. Return only JSON matching the provided schema.",
        "Do not include Markdown, commentary, file paths, tool output, or analysis steps.",
        "Describe visible content only. Use project-provided character naming hints if present, but do not infer identities, private attributes, or unverifiable facts beyond those hints.",
        "If project prompts request ratings or scores, put numeric dimension ratings in scores, total_score, and average_score while still filling the caption fields.",
    ]
    if system_prompt.strip():
        sections.extend(["", "Project system prompt:", system_prompt.strip()])
    if user_prompt.strip():
        sections.extend(["", "Project user prompt:", user_prompt.strip()])
    sections.extend(
        [
            "",
            "Task:",
            "Generate caption and rating metadata for the attached image. Use concise tags and a natural long description.",
            "Fill short_description, long_description, tags, rating, confidence, scores, total_score, and average_score.",
            "If a scoring dimension is not applicable, use 0 for that dimension and explain the visible content in long_description.",
            "If uncertain, lower confidence instead of inventing details.",
        ]
    )
    return "\n".join(sections)


def strip_markdown_json_fence(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else cleaned


def parse_codex_caption_output(text: str) -> dict[str, Any]:
    cleaned = strip_markdown_json_fence(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise CodexCaptionOutputError(f"Codex output was not valid JSON: {exc}", raw=text) from exc
    if not isinstance(parsed, dict):
        raise CodexCaptionOutputError("Codex output JSON must be an object.", raw=text)
    return normalize_codex_caption_payload(parsed)


def normalize_codex_caption_payload(parsed: dict[str, Any]) -> dict[str, Any]:
    result = dict(parsed)
    short = str(result.get("short_description") or result.get("short") or "").strip()
    long = str(
        result.get("long_description")
        or result.get("long")
        or result.get("description")
        or result.get("caption")
        or ""
    ).strip()
    if not short and long:
        short = long[:240].strip()
    if not long and short:
        long = short

    tags = result.get("tags")
    if isinstance(tags, str):
        tags = [part.strip() for part in re.split(r"[,，]", tags) if part.strip()]
    elif not isinstance(tags, list):
        tags = []
    else:
        tags = [str(tag).strip() for tag in tags if str(tag).strip()]

    rating = str(result.get("rating") or "general").strip().lower()
    if rating not in {"general", "sensitive", "questionable", "explicit"}:
        rating = "general"

    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    scores = _normalize_scores(result.get("scores"))
    total_score = _coerce_score_number(result.get("total_score"))
    if total_score == 0.0 and scores:
        total_score = sum(scores.values())

    average_score = _coerce_score_number(result.get("average_score", result.get("avg_score")))
    if average_score == 0.0 and scores:
        average_score = total_score / len(scores)

    normalized = {
        "short_description": short,
        "long_description": long,
        "tags": tags,
        "rating": rating,
        "confidence": confidence,
        "scores": scores,
        "total_score": _compact_number(total_score),
        "average_score": _compact_number(average_score),
    }
    for key in ("character_name", "series"):
        value = str(result.get(key) or "").strip()
        if value:
            normalized[key] = value
    if not normalized["short_description"] and not normalized["long_description"]:
        raise CodexCaptionOutputError("Codex output did not include a caption.")
    return normalized


def _coerce_score_number(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _compact_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else value


def _normalize_scores(value: Any) -> dict[str, int | float]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, int | float] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        score = _coerce_score_number(raw_value)
        normalized[key] = _compact_number(score)
    return normalized


def filter_caption_payload_by_mode(parsed: dict[str, Any], mode: str) -> dict[str, Any]:
    result = dict(parsed)
    if mode == "short":
        result.pop("long_description", None)
    elif mode == "long":
        result.pop("short_description", None)
    return result
