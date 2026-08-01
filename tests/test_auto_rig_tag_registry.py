from __future__ import annotations

import pytest

from module.auto_rig.tag_registry import (
    V3_BASE_TAGS,
    V3_RAW_TAGS,
    V3_SPLIT_FAMILIES,
    AutoRigTagContractError,
    decode_v3_source_tag,
    validate_v3_final_tag_set,
    validate_v3_layerdiff_part_files,
)


def test_v3_registry_freezes_raw_final_and_split_universes() -> None:
    assert len(V3_RAW_TAGS) == 24
    assert len(V3_BASE_TAGS) == 23
    assert set(V3_RAW_TAGS) - set(V3_BASE_TAGS) == {"head"}
    assert V3_SPLIT_FAMILIES == frozenset(
        {"handwear", "eyewhite", "irides", "eyelash", "eyebrow", "ears"}
    )


def test_v3_registry_maps_source_tags_to_stable_part_ids_and_image_sides() -> None:
    front_hair = decode_v3_source_tag("front hair")
    image_right_handwear = decode_v3_source_tag("handwear-r")
    image_left_handwear = decode_v3_source_tag("handwear-l")

    assert front_hair.semantic_slug == "front-hair"
    assert front_hair.part_id == "part/front-hair"
    assert front_hair.side is None
    assert image_right_handwear.base_tag == "handwear"
    assert image_right_handwear.side == "xmin"
    assert image_right_handwear.part_id == "part/handwear.xmin"
    assert image_left_handwear.side == "xmax"
    assert image_left_handwear.part_id == "part/handwear.xmax"


@pytest.mark.parametrize("tag", ("head", "hair", "hairf", "handwear-left", "mouth-r"))
def test_v3_registry_rejects_non_final_or_illegal_split_tags(tag: str) -> None:
    with pytest.raises(AutoRigTagContractError):
        decode_v3_source_tag(tag)


def test_v3_final_set_accepts_unsplit_or_complete_split_family() -> None:
    unsplit = validate_v3_final_tag_set(
        ("face", "handwear", "front hair"),
        tblr_split=True,
    )
    split = validate_v3_final_tag_set(
        ("face", "handwear-r", "handwear-l", "front hair"),
        tblr_split=True,
    )

    assert {part.part_id for part in unsplit} == {
        "part/face",
        "part/front-hair",
        "part/handwear",
    }
    assert {part.part_id for part in split} == {
        "part/face",
        "part/front-hair",
        "part/handwear.xmin",
        "part/handwear.xmax",
    }


@pytest.mark.parametrize(
    "tags,tblr_split",
    (
        (("face", "face"), False),
        (("handwear-r", "handwear-l"), False),
        (("handwear-r",), True),
        (("handwear", "handwear-r", "handwear-l"), True),
        (("head",), False),
    ),
)
def test_v3_final_set_rejects_duplicates_and_ambiguous_split_state(
    tags: tuple[str, ...],
    tblr_split: bool,
) -> None:
    with pytest.raises(AutoRigTagContractError):
        validate_v3_final_tag_set(tags, tblr_split=tblr_split)


def test_layerdiff_manifest_requires_the_exact_v3_raw_png_inventory() -> None:
    files = tuple(f"{tag}.png" for tag in reversed(V3_RAW_TAGS))

    assert validate_v3_layerdiff_part_files(files) == tuple(sorted(files))

    with pytest.raises(AutoRigTagContractError):
        validate_v3_layerdiff_part_files(files[:-1])
    with pytest.raises(AutoRigTagContractError):
        validate_v3_layerdiff_part_files((*files, "unknown.png"))
