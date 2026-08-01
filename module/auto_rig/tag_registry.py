from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

CANONICAL_TAG_REGISTRY_VERSION = "canonical-tag-registry-v3"
V3_RAW_TAGS = (
    "front hair",
    "back hair",
    "head",
    "neck",
    "neckwear",
    "topwear",
    "handwear",
    "bottomwear",
    "legwear",
    "footwear",
    "tail",
    "wings",
    "objects",
    "headwear",
    "face",
    "irides",
    "eyebrow",
    "eyewhite",
    "eyelash",
    "eyewear",
    "ears",
    "earwear",
    "nose",
    "mouth",
)
V3_BASE_TAGS = tuple(tag for tag in V3_RAW_TAGS if tag != "head")
V3_SPLIT_FAMILIES = frozenset(
    {"handwear", "eyewhite", "irides", "eyelash", "eyebrow", "ears"}
)


class AutoRigTagContractError(ValueError):
    """Raised when see-through tags cannot map to the frozen v3 registry."""


@dataclass(frozen=True, slots=True)
class CanonicalPartTag:
    source_tag: str
    base_tag: str
    semantic_slug: str
    side: str | None
    part_id: str


def decode_v3_source_tag(source_tag: str) -> CanonicalPartTag:
    if not isinstance(source_tag, str) or not source_tag:
        raise AutoRigTagContractError("v3 source tag must be a non-empty string")
    base_tag = source_tag
    side = None
    if source_tag.endswith("-r") or source_tag.endswith("-l"):
        base_tag = source_tag[:-2]
        if base_tag not in V3_SPLIT_FAMILIES:
            raise AutoRigTagContractError(f"v3 tag is not an LR-splittable family: {source_tag}")
        side = "xmin" if source_tag.endswith("-r") else "xmax"
    if base_tag not in V3_BASE_TAGS:
        raise AutoRigTagContractError(f"v3 final tag is not registered: {source_tag}")
    semantic_slug = base_tag.replace(" ", "-")
    part_id = f"part/{semantic_slug}" + (f".{side}" if side is not None else "")
    return CanonicalPartTag(
        source_tag=source_tag,
        base_tag=base_tag,
        semantic_slug=semantic_slug,
        side=side,
        part_id=part_id,
    )


def validate_v3_final_tag_set(
    source_tags: Iterable[str],
    *,
    tblr_split: bool,
) -> tuple[CanonicalPartTag, ...]:
    if type(tblr_split) is not bool:
        raise AutoRigTagContractError("tblr_split must be a boolean")
    tags = tuple(source_tags)
    if not tags:
        raise AutoRigTagContractError("v3 final tag set must not be empty")
    if len(tags) != len(set(tags)):
        raise AutoRigTagContractError("v3 final tag set contains duplicate source tags")
    decoded = tuple(decode_v3_source_tag(tag) for tag in tags)
    if not tblr_split and any(part.side is not None for part in decoded):
        raise AutoRigTagContractError("split source tags require tblr_split=true")
    if tblr_split:
        tag_set = set(tags)
        for family in V3_SPLIT_FAMILIES:
            present = tag_set & {family, f"{family}-r", f"{family}-l"}
            if present not in (set(), {family}, {f"{family}-r", f"{family}-l"}):
                raise AutoRigTagContractError(f"v3 split family has an ambiguous final state: {family}")
    part_ids = tuple(part.part_id for part in decoded)
    if len(part_ids) != len(set(part_ids)):
        raise AutoRigTagContractError("v3 source tags collide after Part ID encoding")
    return tuple(sorted(decoded, key=lambda part: part.part_id))


def validate_v3_layerdiff_part_files(files: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(files)
    if any(
        not isinstance(name, str)
        or not name
        or "/" in name
        or "\\" in name
        for name in normalized
    ):
        raise AutoRigTagContractError("LayerDiff part entries must be plain file names")
    if len(normalized) != len(set(normalized)):
        raise AutoRigTagContractError("LayerDiff part manifest contains duplicate file names")
    expected = {f"{tag}.png" for tag in V3_RAW_TAGS}
    if set(normalized) != expected:
        missing = sorted(expected - set(normalized))
        extra = sorted(set(normalized) - expected)
        raise AutoRigTagContractError(
            f"LayerDiff v3 raw part inventory mismatch; missing={missing}, extra={extra}"
        )
    return tuple(sorted(normalized))


__all__ = [
    "CANONICAL_TAG_REGISTRY_VERSION",
    "AutoRigTagContractError",
    "CanonicalPartTag",
    "V3_BASE_TAGS",
    "V3_RAW_TAGS",
    "V3_SPLIT_FAMILIES",
    "decode_v3_source_tag",
    "validate_v3_final_tag_set",
    "validate_v3_layerdiff_part_files",
]
